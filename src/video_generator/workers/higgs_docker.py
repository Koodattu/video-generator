from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
import wave
from pathlib import Path
from typing import Any

from ..errors import ErrorKind, VideoGeneratorError
from ..util import atomic_write_bytes, replace_path


HIGGS_MODEL_ID = "bosonai/higgs-tts-3-4b"
HIGGS_SAMPLE_RATE = 24000
_REFERENCE_AUDIO_SUFFIXES = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"}
_SPEED_TOKENS = {
    "slow": "<|prosody:speed_slow|>",
    "standard": "",
    "fast": "<|prosody:speed_fast|>",
}
_CLIENT_SCRIPT = r"""
import json
import pathlib
import sys
import urllib.request

payload = json.load(sys.stdin)
output_path = pathlib.Path(payload.pop("_output_path"))
port = int(payload.pop("_port"))
request = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/audio/speech",
    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=1200) as response:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.read())
    print(json.dumps({
        "status": response.status,
        "content_type": response.headers.get("Content-Type", ""),
    }))
""".strip()


def compile_higgs_text(text: str, delivery: dict[str, Any] | None) -> tuple[str, list[str]]:
    if not text or len(text) > 4096:
        raise ValueError("Higgs narration text must contain 1-4096 characters")
    pace = str((delivery or {}).get("pace") or "standard")
    if pace not in _SPEED_TOKENS:
        raise ValueError(f"unsupported Higgs narration pace: {pace}")
    tokens = [token for token in (_SPEED_TOKENS[pace],) if token]
    compiled = "".join(tokens) + text
    stripped = compiled
    for token in tokens:
        if not stripped.startswith(token):
            raise ValueError("Higgs control-token compiler changed canonical narration")
        stripped = stripped[len(token) :]
    if stripped != text:
        raise ValueError("Higgs control-token compiler changed canonical narration")
    return compiled, tokens


class HiggsDockerWorker:
    def __init__(self, paths: Any) -> None:
        docker = shutil.which("docker")
        if not docker:
            raise VideoGeneratorError(
                "Docker CLI is unavailable for Higgs TTS",
                kind=ErrorKind.NOT_READY,
            )
        self.docker = docker
        self.paths = paths
        self.container_name = os.environ["VIDEO_GENERATOR_DOCKER_CONTAINER_NAME"]
        self.image_reference = os.environ["VIDEO_GENERATOR_DOCKER_IMAGE_REFERENCE"]
        self.image_id = os.environ["VIDEO_GENERATOR_DOCKER_IMAGE_ID"]
        self.server_revision = os.environ["VIDEO_GENERATOR_DOCKER_SERVER_REVISION"]
        self.port = int(os.environ.get("VIDEO_GENERATOR_DOCKER_INTERNAL_PORT", "8000"))
        self.model_path = paths.read_model(os.environ["VIDEO_GENERATOR_MODEL_PATH"])
        self.voice_scratch = paths.read_runtime(
            os.environ["VIDEO_GENERATOR_HIGGS_VOICE_SCRATCH"]
        )
        expected_scratch_parent = (
            paths.runtime_root
            / "local--higgs-tts-3-4b"
            / "voice-scratch"
        ).resolve()
        if (
            not self.voice_scratch.is_dir()
            or self.voice_scratch.parent != expected_scratch_parent
            or self.voice_scratch.name != self.container_name
        ):
            raise VideoGeneratorError(
                "Higgs TTS voice scratch directory is not manager-owned",
                kind=ErrorKind.NOT_READY,
            )
        self.started_at = time.monotonic()
        self._closed = False
        try:
            self._start_container()
            self._wait_until_healthy()
        except BaseException:
            self._remove_container(force=True)
            self._remove_voice_scratch()
            raise

    def _run(
        self,
        arguments: list[str],
        *,
        timeout: float = 120,
        input_text: str | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            [self.docker, *arguments],
            input=input_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        if check and completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "Docker command failed"
            raise VideoGeneratorError(detail, kind=ErrorKind.NOT_READY)
        return completed

    def _start_container(self) -> None:
        model_mount = f"type=bind,source={self.model_path},target=/models/higgs-tts-3-4b,readonly"
        run_mount = f"type=bind,source={self.paths.run_root},target=/run"
        voice_mount = f"type=bind,source={self.voice_scratch},target=/voices,readonly"
        command = [
            "run",
            "--detach",
            "--name",
            self.container_name,
            "--label",
            "video-generator.managed=true",
            "--gpus",
            "all",
            "--network",
            "none",
            "--shm-size",
            "32g",
            "--stop-timeout",
            "120",
            "--read-only",
            "--security-opt",
            "no-new-privileges",
            "--cap-drop",
            "ALL",
            "--tmpfs",
            "/tmp:rw,exec,nosuid,size=8g",
            "--mount",
            model_mount,
            "--mount",
            run_mount,
            "--mount",
            voice_mount,
            "--env",
            "HOME=/tmp/home",
            "--env",
            "HF_HOME=/tmp/huggingface",
            "--env",
            "XDG_CACHE_HOME=/tmp/cache",
            "--env",
            "TORCH_HOME=/tmp/torch",
            "--env",
            "HF_HUB_OFFLINE=1",
            "--env",
            "TRANSFORMERS_OFFLINE=1",
            self.image_reference,
            "/opt/omni/bin/sgl-omni",
            "serve",
            "--model-path",
            "/models/higgs-tts-3-4b",
            "--model-name",
            HIGGS_MODEL_ID,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--allowed-local-media-path",
            "/voices",
        ]
        self._run(command, timeout=120)

    def _container_running(self) -> bool:
        completed = self._run(
            ["inspect", "--format", "{{.State.Running}}", self.container_name],
            timeout=30,
            check=False,
        )
        return completed.returncode == 0 and completed.stdout.strip().casefold() == "true"

    def _wait_until_healthy(self) -> None:
        deadline = time.monotonic() + 1200
        health_script = (
            "import urllib.request; "
            f"print(urllib.request.urlopen('http://127.0.0.1:{self.port}/health', timeout=3).read().decode())"
        )
        last_error = ""
        while time.monotonic() < deadline:
            if not self._container_running():
                break
            completed = self._run(
                ["exec", self.container_name, "/opt/omni/bin/python", "-c", health_script],
                timeout=10,
                check=False,
            )
            if completed.returncode == 0:
                return
            last_error = completed.stderr.strip() or completed.stdout.strip()
            time.sleep(3)
        logs_result = self._run(["logs", "--tail", "120", self.container_name], check=False)
        logs = "\n".join(value for value in (logs_result.stdout, logs_result.stderr) if value.strip())
        detail = (last_error + "\n" + logs).strip()[-8000:]
        raise VideoGeneratorError(
            f"Higgs TTS container did not become healthy: {detail}",
            kind=ErrorKind.NOT_READY,
        )

    def health(self) -> dict[str, Any]:
        return {
            "runtime": "sglang-omni-docker",
            "container_name": self.container_name,
            "image_reference": self.image_reference,
            "image_id": self.image_id,
            "server_revision": self.server_revision,
            "model_path": str(self.model_path),
            "model_id": HIGGS_MODEL_ID,
            "sample_rate": HIGGS_SAMPLE_RATE,
            "voice_scratch": str(self.voice_scratch),
            "startup_elapsed_seconds": time.monotonic() - self.started_at,
        }

    def _stage_reference(self, reference: Path) -> str:
        if not reference.is_file():
            raise ValueError("Higgs TTS reference audio must be a regular file")
        suffix = reference.suffix.casefold()
        if suffix not in _REFERENCE_AUDIO_SUFFIXES:
            raise ValueError(f"unsupported Higgs TTS reference audio type: {suffix or '(none)'}")
        reference_bytes = reference.read_bytes()
        if not reference_bytes:
            raise ValueError("Higgs TTS reference audio is empty")
        if len(reference_bytes) > 10 * 1024 * 1024:
            raise ValueError("Higgs TTS reference audio exceeds the 10 MiB API limit")
        for child in self.voice_scratch.iterdir():
            if child.is_symlink() or not child.is_file():
                raise VideoGeneratorError(
                    "Higgs TTS voice scratch contains an unexpected entry",
                    kind=ErrorKind.NOT_READY,
                )
            child.unlink()
        reference_digest = hashlib.sha256(reference_bytes).hexdigest()[:24]
        filename = f"reference-{reference_digest}{suffix}"
        staged = self.voice_scratch / filename
        staging = self.voice_scratch.parent / f".{self.container_name}-{filename}.staging"
        try:
            atomic_write_bytes(staging, reference_bytes)
            replace_path(staging, staged)
        finally:
            staging.unlink(missing_ok=True)
        entries = list(self.voice_scratch.iterdir())
        if entries != [staged] or staged.is_symlink() or not staged.is_file():
            raise VideoGeneratorError(
                "Higgs TTS could not isolate the selected reference audio",
                kind=ErrorKind.NOT_READY,
            )
        return f"/voices/{filename}"

    def dispatch(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        if operation != "speech.synthesize":
            raise ValueError(f"unsupported Higgs TTS operation: {operation}")
        voice = payload.get("voice") or {}
        if not voice.get("reference_audio") or not voice.get("reference_transcript"):
            raise ValueError("Higgs TTS requires an authorized reference audio file and exact transcript")
        reference = self.paths.read_private(str(voice["reference_audio"]))
        transcript = self.paths.read_private(str(voice["reference_transcript"])).read_text(
            encoding="utf-8"
        ).strip()
        if not transcript:
            raise ValueError("Higgs TTS reference transcript is empty")
        output = self.paths.output_run(str(payload["output_path"]))
        output_in_container = "/run/" + output.relative_to(self.paths.run_root).as_posix()
        reference_in_container = self._stage_reference(reference)
        compiled_text, control_tokens = compile_higgs_text(
            str(payload["text"]),
            payload.get("delivery") if isinstance(payload.get("delivery"), dict) else None,
        )
        seed = int.from_bytes(
            hashlib.sha256(
                (str(payload["scene_id"]) + "\0" + str(payload["text"])).encode("utf-8")
            ).digest()[:4],
            "big",
        )
        request = {
            "model": HIGGS_MODEL_ID,
            "input": compiled_text,
            "response_format": "wav",
            "speed": 1.0,
            "references": [{"audio_path": reference_in_container, "text": transcript}],
            "temperature": 0.8,
            "top_p": 0.8,
            "top_k": 30,
            "repetition_penalty": 1.1,
            "max_new_tokens": 2048,
            "seed": seed,
            "_output_path": output_in_container,
            "_port": self.port,
        }
        started = time.monotonic()
        completed = self._run(
            ["exec", "-i", self.container_name, "/opt/omni/bin/python", "-c", _CLIENT_SCRIPT],
            timeout=1500,
            input_text=json.dumps(request, ensure_ascii=False),
        )
        try:
            response = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise VideoGeneratorError(
                "Higgs TTS client returned invalid response metadata",
                kind=ErrorKind.INVALID_OUTPUT,
            ) from exc
        if response.get("status") != 200 or "audio/wav" not in str(response.get("content_type", "")):
            raise VideoGeneratorError(
                "Higgs TTS returned an unexpected media response",
                kind=ErrorKind.INVALID_OUTPUT,
            )
        try:
            with wave.open(str(output), "rb") as handle:
                sample_rate = int(handle.getframerate())
                channels = int(handle.getnchannels())
                frames = int(handle.getnframes())
        except (OSError, EOFError, wave.Error) as exc:
            raise VideoGeneratorError(
                "Higgs TTS returned an invalid WAV file",
                kind=ErrorKind.INVALID_OUTPUT,
            ) from exc
        if sample_rate != HIGGS_SAMPLE_RATE or channels != 1 or frames <= 0:
            raise VideoGeneratorError(
                "Higgs TTS returned unexpected WAV properties",
                kind=ErrorKind.INVALID_OUTPUT,
            )
        return {
            "audio_path": payload["output_path"],
            "mime_type": "audio/wav",
            "duration_seconds": frames / sample_rate,
            "sample_rate": sample_rate,
            "channels": channels,
            "timing_precision": "none",
            "word_timings": [],
            "seed": seed,
            "control_tokens": control_tokens,
            "usage": {"elapsed_seconds": time.monotonic() - started},
        }

    def _remove_container(self, *, force: bool) -> bool:
        if self._closed:
            return True
        if force:
            self._run(["rm", "-f", self.container_name], timeout=120, check=False)
        else:
            self._run(["stop", "--time", "120", self.container_name], timeout=180, check=False)
            self._run(["rm", "-f", self.container_name], timeout=120, check=False)
        absent = self._run(["inspect", self.container_name], timeout=30, check=False).returncode != 0
        self._closed = absent
        return absent

    def _remove_voice_scratch(self) -> bool:
        if not self.voice_scratch.exists():
            return True
        expected_parent = (
            self.paths.runtime_root
            / "local--higgs-tts-3-4b"
            / "voice-scratch"
        ).resolve()
        try:
            resolved = self.voice_scratch.resolve()
            resolved.relative_to(expected_parent)
        except (OSError, ValueError):
            return False
        if resolved.parent != expected_parent or resolved.name != self.container_name:
            return False
        shutil.rmtree(resolved)
        return not resolved.exists()

    def close(self) -> dict[str, Any]:
        container_absent = self._remove_container(force=False)
        voice_scratch_absent = (
            self._remove_voice_scratch()
            if container_absent
            else False
        )
        return {
            "container_absent": container_absent,
            "voice_scratch_absent": voice_scratch_absent,
        }
