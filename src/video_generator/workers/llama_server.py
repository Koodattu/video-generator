from __future__ import annotations

import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GpuSnapshot:
    observable: bool
    used_mb: int | None
    process_ids: tuple[int, ...]


def gpu_snapshot() -> GpuSnapshot:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return GpuSnapshot(observable=False, used_mb=None, process_ids=())
    try:
        memory = subprocess.run(
            [executable, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
        )
        processes = subprocess.run(
            [executable, "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return GpuSnapshot(observable=False, used_mb=None, process_ids=())
    if memory.returncode != 0 or processes.returncode != 0:
        return GpuSnapshot(observable=False, used_mb=None, process_ids=())
    try:
        used_values = [int(line.strip().split(",", 1)[0]) for line in memory.stdout.splitlines() if line.strip()]
        process_ids = tuple(
            sorted(
                {
                    int(line.strip().split(",", 1)[0])
                    for line in processes.stdout.splitlines()
                    if line.strip() and line.strip().split(",", 1)[0].isdigit()
                }
            )
        )
    except ValueError:
        return GpuSnapshot(observable=False, used_mb=None, process_ids=())
    return GpuSnapshot(
        observable=bool(used_values),
        used_mb=sum(used_values) if used_values else None,
        process_ids=process_ids,
    )


class LlamaServerSession:
    """Own one loopback-only stock llama-server process for a contiguous text batch."""

    def __init__(
        self,
        *,
        executable: Path,
        model: Path,
        draft_model: Path | None,
        arguments: list[str],
        startup_timeout_seconds: float = 600,
        request_timeout_seconds: float = 600,
        cleanup_timeout_seconds: float = 30,
        vram_tolerance_mb: int = 512,
    ) -> None:
        self.executable = executable.resolve()
        self.model = model.resolve()
        self.draft_model = draft_model.resolve() if draft_model else None
        self.arguments = [str(value) for value in arguments]
        self.startup_timeout_seconds = startup_timeout_seconds
        self.request_timeout_seconds = request_timeout_seconds
        self.cleanup_timeout_seconds = cleanup_timeout_seconds
        self.vram_tolerance_mb = vram_tolerance_mb
        self.process: subprocess.Popen[bytes] | None = None
        self.port = 0
        self.api_key = ""
        self.baseline = GpuSnapshot(False, None, ())
        self.loaded = GpuSnapshot(False, None, ())
        self.peak_used_mb: int | None = None
        self.startup_elapsed_seconds: float | None = None
        self._cleanup: dict[str, Any] | None = None
        self._validate_arguments()

    def _validate_arguments(self) -> None:
        forbidden = {
            "-m",
            "--model",
            "-md",
            "--model-draft",
            "--spec-draft-model",
            "-hf",
            "--hf-repo",
            "--host",
            "--port",
            "--api-key",
            "--api-key-file",
            "--models-dir",
            "--models-preset",
            "--tools",
            "--agent",
        }
        for value in self.arguments:
            name = value.split("=", 1)[0].casefold()
            if name in forbidden:
                raise ValueError(f"llama-server launch setting is managed by the orchestrator: {name}")

    @staticmethod
    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.bind(("127.0.0.1", 0))
            return int(server.getsockname()[1])

    @property
    def base_url(self) -> str:
        if not self.port:
            raise RuntimeError("llama-server has not been started")
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        self.baseline = gpu_snapshot()
        started_at = time.monotonic()
        self.port = self._free_port()
        self.api_key = secrets.token_urlsafe(32)
        command = [
            str(self.executable),
            "--model",
            str(self.model),
            *self.arguments,
        ]
        if self.draft_model is not None:
            command.extend(["--model-draft", str(self.draft_model)])
        command.extend(
            ["--host", "127.0.0.1", "--port", str(self.port), "--no-ui"]
        )
        environment = dict(os.environ)
        environment["LLAMA_API_KEY"] = self.api_key
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
                subprocess, "CREATE_NO_WINDOW", 0
            )
        self.process = subprocess.Popen(
            command,
            cwd=self.executable.parent,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=sys.stderr,
            stderr=sys.stderr,
            creationflags=creationflags,
        )
        deadline = time.monotonic() + self.startup_timeout_seconds
        try:
            while True:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"llama-server exited during startup with code {self.process.returncode}"
                    )
                try:
                    value = self._request_json("GET", "/health", timeout_seconds=5)
                    if value.get("status") == "ok":
                        break
                except urllib.error.HTTPError as exc:
                    if exc.code != 503:
                        raise RuntimeError(f"llama-server health probe returned HTTP {exc.code}") from exc
                except (urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"llama-server did not become healthy within {self.startup_timeout_seconds:.0f}s"
                    )
                time.sleep(0.25)
        except BaseException:
            self.close()
            raise
        self.loaded = gpu_snapshot()
        self.peak_used_mb = self.loaded.used_mb
        self.startup_elapsed_seconds = time.monotonic() - started_at

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Accept": "application/json"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            headers=headers,
            method=method,
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=timeout_seconds or self.request_timeout_seconds) as response:
            raw = response.read(50_000_001)
        if len(raw) > 50_000_000:
            raise RuntimeError("llama-server response exceeded 50 MB")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("llama-server returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise RuntimeError("llama-server returned a non-object response")
        return value

    def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.start()
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("llama-server is not running")
        try:
            value = self._request_json("POST", "/v1/chat/completions", payload=payload)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                response = json.loads(exc.read(1_000_000).decode("utf-8"))
                error = response.get("error", {}) if isinstance(response, dict) else {}
                if isinstance(error, dict):
                    detail = str(error.get("message") or "")
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(f"llama-server returned HTTP {exc.code}{suffix}") from exc
        snapshot = gpu_snapshot()
        if snapshot.used_mb is not None:
            self.peak_used_mb = max(self.peak_used_mb or 0, snapshot.used_mb)
        return value

    def close(self) -> dict[str, Any]:
        if self._cleanup is not None:
            return self._cleanup
        process, self.process = self.process, None
        pid = process.pid if process is not None else None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._kill_process_tree(process)

        deadline = time.monotonic() + self.cleanup_timeout_seconds
        post = gpu_snapshot()
        while pid is not None and post.observable and pid in post.process_ids and time.monotonic() < deadline:
            time.sleep(0.25)
            post = gpu_snapshot()
        pid_released = pid is None or not post.observable or pid not in post.process_ids
        within_tolerance = (
            self.baseline.used_mb is None
            or post.used_mb is None
            or post.used_mb <= self.baseline.used_mb + self.vram_tolerance_mb
        )
        self._cleanup = {
            "server_pid": pid,
            "process_exited": process is None or process.poll() is not None,
            "gpu_process_released": pid_released,
            "vram_within_tolerance": within_tolerance,
            "vram_tolerance_mb": self.vram_tolerance_mb,
            "baseline": asdict(self.baseline),
            "loaded": asdict(self.loaded),
            "peak_used_mb": self.peak_used_mb,
            "post_exit": asdict(post),
        }
        self.api_key = ""
        if not self._cleanup["process_exited"] or not pid_released:
            raise RuntimeError("llama-server process or GPU allocation remained after shutdown")
        return self._cleanup

    @staticmethod
    def _kill_process_tree(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(
                ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                timeout=20,
                check=False,
            )
        else:
            process.kill()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)

    def __enter__(self) -> "LlamaServerSession":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
