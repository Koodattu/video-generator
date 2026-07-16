from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, IO, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .contracts import ProbeItem, ProbeReport
from .errors import BackendError, ErrorKind
from .profiles import EXPECTED_LOCAL_MODEL_REVISIONS
from .util import atomic_write_json, read_json, sha256_file


_CUDA_RUNNER_KINDS = {
    "voxcpm",
    "flux",
    "omnivoice",
    "moss-tts",
    "xvoice",
    "z-image",
    "ideogram4",
    "qwen-image",
}


def runner_torch_backend(model_family: str) -> str:
    if model_family in {"moss-tts", "xvoice"}:
        return "cu128"
    if model_family in {"ideogram4", "qwen-image"}:
        return "cu130"
    return "auto" if model_family in _CUDA_RUNNER_KINDS else ""


def runner_setup_source_revision(model_family: str) -> str:
    package_root = Path(__file__).resolve().parent
    requirements = package_root / "assets" / "runners" / f"{model_family}.in"
    sources = {
        "requirements": requirements,
        "requirements_lock": requirements.with_suffix(".lock"),
        "runner_manager": Path(__file__).resolve(),
        "setup_implementation": package_root / "setup.py",
        "profile_pins": package_root / "profiles.py",
        "worker_implementation": package_root / "workers" / "main.py",
        "snapshot_implementation": package_root / "workers" / "prepare.py",
    }
    if model_family == "xvoice":
        sources["conda_lock"] = requirements.with_name("xvoice-conda-win-64.lock")
    if model_family == "higgs-docker":
        sources["docker_worker"] = package_root / "workers" / "higgs_docker.py"
    source_revisions = [
        f"{name}={sha256_file(path) if path.is_file() else '<missing>'}"
        for name, path in sorted(sources.items())
    ]
    torch_backend = runner_torch_backend(model_family)
    return hashlib.sha256(
        "\0".join([*source_revisions, f"torch-backend={torch_backend}"]).encode("utf-8")
    ).hexdigest()


def runner_slug(backend_id: str) -> str:
    return backend_id.replace(":", "--").replace("/", "-")


def decode_wsl_output(value: bytes) -> str:
    """Decode wsl.exe management output, which is UTF-16LE when redirected on Windows."""

    if value.startswith((b"\xff\xfe", b"\xfe\xff")):
        return value.decode("utf-16", errors="replace").lstrip("\ufeff")
    if value and value.count(b"\x00") >= max(1, len(value) // 4):
        return value.decode("utf-16-le", errors="replace").lstrip("\ufeff")
    return value.decode("utf-8-sig", errors="replace")


class DockerRuntimeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_version: Literal[1] = 1
    image_reference: str
    image_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    server_revision: str = Field(min_length=7)
    internal_port: int = Field(default=8000, ge=1, le=65535)
    docker_server_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_immutable_image(self) -> "DockerRuntimeSpec":
        if "@sha256:" not in self.image_reference:
            raise ValueError("Docker image_reference must be pinned by sha256 digest")
        digest = self.image_reference.rsplit("@", 1)[-1]
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise ValueError("Docker image_reference contains an invalid digest")
        return self


class RunnerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_version: Literal[1] = 1
    backend_id: str
    platform: str = Field(pattern="^(native|wsl|docker)$")
    command: list[str] = Field(min_length=1)
    model_family: str
    requires_cuda: bool = True
    timeout_seconds: float = Field(default=600, gt=0, le=7200)
    startup_timeout_seconds: float = Field(default=180, gt=0, le=1800)
    wsl_distribution: str = ""
    docker: DockerRuntimeSpec | None = None
    environment: dict[str, str] = Field(default_factory=dict)
    model_paths: list[str] = Field(min_length=1)
    asset_manifests: dict[str, str] = Field(min_length=1)
    asset_revisions: dict[str, str] = Field(default_factory=dict)
    runtime_files: dict[str, str] = Field(min_length=1)
    runtime_revision: str
    model_revision: str
    setup_source_revision: str
    license_name: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_platform_settings(self) -> "RunnerSpec":
        if self.platform == "docker" and self.docker is None:
            raise ValueError("docker Runner requires docker runtime settings")
        if self.platform != "docker" and self.docker is not None:
            raise ValueError("docker runtime settings are only valid for docker Runners")
        if self.platform == "wsl" and not self.wsl_distribution:
            raise ValueError("WSL Runner requires wsl_distribution")
        return self


def windows_to_wsl(path: Path) -> str:
    path = path.resolve()
    drive = path.drive.rstrip(":").lower()
    if not drive or not path.is_absolute():
        raise ValueError(f"cannot map non-drive Windows path into WSL: {path}")
    tail = path.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{tail}"


class GpuLease:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: IO[bytes] | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        handle.seek(0)
        if handle.read(1) == b"":
            handle.seek(0)
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise BackendError(
                "another local model process holds the exclusive GPU lease",
                kind=ErrorKind.NOT_READY,
                action="Wait for the other Run to finish or terminate its local runner cleanly.",
            ) from exc
        self._handle = handle

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            self._handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None


@dataclass
class _RunnerProcess:
    spec: RunnerSpec
    process: subprocess.Popen[str]
    responses: queue.Queue[dict[str, Any]]
    reader: threading.Thread
    stderr_handle: IO[str]
    started_at: float = field(default_factory=time.monotonic)
    health: dict[str, Any] = field(default_factory=dict)
    gpu_baseline: dict[str, Any] = field(default_factory=dict)
    container_name: str = ""


class RunnerManager:
    def __init__(
        self,
        *,
        project_root: Path,
        run_root: Path,
        cache_root: Path | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.run_root = run_root.resolve()
        self.cache_root = (cache_root or self.project_root / ".cache").resolve()
        self.manifest_root = self.cache_root / "runners"
        self.model_root = self.cache_root / "models"
        self.lease = GpuLease(self.cache_root / "locks" / "gpu.lock")
        self.current: _RunnerProcess | None = None
        self._lease_held = False
        self._request_lock = threading.Lock()
        self.last_cleanup: dict[str, dict[str, Any]] = {}
        self._cleanup_sequence = 0

    def manifest_path(self, backend_id: str) -> Path:
        return self.manifest_root / runner_slug(backend_id) / "runner.json"

    def load_spec(self, backend_id: str) -> RunnerSpec:
        path = self.manifest_path(backend_id)
        try:
            spec = RunnerSpec.model_validate(read_json(path))
            if spec.backend_id != backend_id:
                raise ValueError(f"runner manifest declares {spec.backend_id}, expected {backend_id}")
            expected_revision = EXPECTED_LOCAL_MODEL_REVISIONS.get(backend_id)
            if expected_revision and spec.model_revision != expected_revision:
                raise ValueError(
                    f"runner model revision {spec.model_revision} does not match expected {expected_revision}"
                )
            if spec.model_family == "acestep":
                expected_setup_source = "dce621408bee8c31b4fcf4811682eb9359e1bc94"
            else:
                expected_setup_source = runner_setup_source_revision(spec.model_family)
            if spec.setup_source_revision != expected_setup_source:
                raise ValueError("runner was built from an obsolete Setup requirements source")
            return spec
        except FileNotFoundError as exc:
            raise BackendError(
                f"local Backend {backend_id} is not prepared",
                kind=ErrorKind.NOT_READY,
                action=f"Run: video-generator setup --backend {backend_id}",
            ) from exc
        except (ValidationError, ValueError) as exc:
            raise BackendError(
                f"invalid runner manifest for {backend_id}: {exc}", kind=ErrorKind.NOT_READY
            ) from exc

    def probe(self, backend_id: str, *, live: bool = False) -> ProbeReport:
        items: list[ProbeItem] = []
        try:
            spec = self.load_spec(backend_id)
        except BackendError as exc:
            return ProbeReport(
                backend_id=backend_id,
                ready=False,
                items=[ProbeItem(name="runner_manifest", ready=False, detail=exc.message, action=exc.action)],
            )
        items.append(ProbeItem(name="runner_manifest", ready=True, detail=str(self.manifest_path(backend_id))))
        if spec.platform == "native":
            executable = spec.command[0] if spec.command else ""
            found = bool(executable and (Path(executable).is_file() or shutil.which(executable)))
            items.append(
                ProbeItem(
                    name="runner_executable",
                    ready=found,
                    detail=executable if found else f"executable not found: {executable}",
                    action=None if found else f"Repair {self.manifest_path(backend_id)} or rerun Setup.",
                )
            )
            if spec.requires_cuda:
                nvidia_smi = shutil.which("nvidia-smi")
                cuda_ready = False
                used_gpu_memory_mb: int | None = None
                if nvidia_smi:
                    try:
                        cuda_ready = subprocess.run(
                            [nvidia_smi, "-L"],
                            capture_output=True,
                            timeout=30,
                            check=False,
                        ).returncode == 0
                        memory_probe = subprocess.run(
                            [
                                nvidia_smi,
                                "--query-gpu=memory.used",
                                "--format=csv,noheader,nounits",
                            ],
                            capture_output=True,
                            text=True,
                            encoding="utf-8",
                            errors="replace",
                            timeout=30,
                            check=False,
                        )
                        if memory_probe.returncode == 0:
                            values = [
                                int(line.strip().split(",", 1)[0])
                                for line in memory_probe.stdout.splitlines()
                                if line.strip()
                            ]
                            used_gpu_memory_mb = sum(values) if values else None
                    except (OSError, ValueError, subprocess.TimeoutExpired):
                        pass
                items.append(
                    ProbeItem(
                        name="cuda_gpu",
                        ready=cuda_ready,
                        detail="CUDA-capable NVIDIA GPU visible" if cuda_ready else "NVIDIA GPU probe failed",
                        action=None if cuda_ready else "Install/repair the NVIDIA driver before local generation.",
                    )
                )
                if used_gpu_memory_mb is not None:
                    detail = f"{used_gpu_memory_mb} MB aggregate GPU memory in use before model launch"
                    if used_gpu_memory_mb > 2048:
                        detail += "; close unrelated GPU applications before fit or speed benchmarks"
                    items.append(
                        ProbeItem(name="gpu_memory_baseline", ready=True, detail=detail)
                    )
        elif spec.platform == "wsl":
            wsl = shutil.which("wsl.exe") or shutil.which("wsl")
            distro_ready = False
            python_ready = False
            ffmpeg_ready = False
            cuda_ready = False
            if wsl and spec.wsl_distribution:
                try:
                    listed = subprocess.run(
                        [wsl, "-l", "-q"],
                        capture_output=True,
                        timeout=30,
                        check=False,
                    )
                    distro_ready = spec.wsl_distribution.casefold() in {
                        line.strip().casefold()
                        for line in decode_wsl_output(listed.stdout).splitlines()
                        if line.strip()
                    }
                    if distro_ready:
                        python_ready = subprocess.run(
                            [wsl, "-d", spec.wsl_distribution, "--", "test", "-x", spec.command[0]],
                            capture_output=True,
                            timeout=30,
                            check=False,
                        ).returncode == 0
                        ffmpeg_ready = subprocess.run(
                            [wsl, "-d", spec.wsl_distribution, "--", "which", "ffmpeg"],
                            capture_output=True,
                            timeout=30,
                            check=False,
                        ).returncode == 0
                        cuda_ready = subprocess.run(
                            [wsl, "-d", spec.wsl_distribution, "--", "nvidia-smi", "-L"],
                            capture_output=True,
                            timeout=30,
                            check=False,
                        ).returncode == 0
                except (OSError, subprocess.TimeoutExpired):
                    pass
            items.append(
                ProbeItem(
                    name="wsl",
                    ready=distro_ready,
                    detail=(
                        f"WSL distribution: {spec.wsl_distribution}"
                        if distro_ready
                        else "WSL executable or configured distribution is unavailable"
                    ),
                    action=None if distro_ready else "Install the configured WSL2 distribution, then rerun Setup.",
                )
            )
            items.extend(
                [
                    ProbeItem(
                        name="wsl_runner_python",
                        ready=python_ready,
                        detail=spec.command[0] if python_ready else "runner Python is missing inside WSL",
                        action=None if python_ready else f"Rerun Setup for {backend_id}.",
                    ),
                    ProbeItem(
                        name="wsl_ffmpeg",
                        ready=ffmpeg_ready,
                        detail="Linux FFmpeg available" if ffmpeg_ready else "Linux FFmpeg is missing",
                        action=None if ffmpeg_ready else f"Install ffmpeg inside {spec.wsl_distribution}.",
                    ),
                    ProbeItem(
                        name="wsl_cuda",
                        ready=cuda_ready,
                        detail="CUDA GPU visible in WSL" if cuda_ready else "CUDA GPU is not visible in WSL",
                        action=None if cuda_ready else "Repair the NVIDIA WSL2 driver/CUDA integration.",
                    ),
                ]
            )
        else:
            items.extend(self._probe_docker_runtime(spec))
        for model_path in spec.model_paths:
            path = (self.project_root / model_path).resolve()
            in_cache = False
            try:
                relative_cache_path = path.relative_to(self.cache_root)
                in_cache = bool(
                    relative_cache_path.parts
                    and relative_cache_path.parts[0] in {"models", "runtimes"}
                )
            except ValueError:
                pass
            exists = path.exists() and in_cache
            items.append(
                ProbeItem(
                    name=f"model:{model_path}",
                    ready=exists,
                    detail=str(path) if exists else f"missing or outside managed cache: {path}",
                    action=None if exists else f"Run: video-generator setup --backend {backend_id}",
                )
            )
        for runtime_value, expected_hash in spec.runtime_files.items():
            runtime_path = (self.project_root / runtime_value).resolve()
            try:
                runtime_path.relative_to(self.cache_root / "runtimes")
                ready = runtime_path.is_file() and sha256_file(runtime_path) == expected_hash
            except ValueError:
                ready = False
            items.append(
                ProbeItem(
                    name=f"runtime_file:{runtime_value}",
                    ready=ready,
                    detail=str(runtime_path) if ready else "runtime provenance file is missing or changed",
                    action=None if ready else f"Rerun Setup for {backend_id}.",
                )
            )
        for manifest_value, expected_hash in spec.asset_manifests.items():
            manifest_path = (self.project_root / manifest_value).resolve()
            try:
                manifest_path.relative_to(self.cache_root)
                hash_matches = manifest_path.is_file() and sha256_file(manifest_path) == expected_hash
                detail = (
                    self._verify_asset_manifest(
                        manifest_path,
                        expected_revision=(
                            None
                            if manifest_path.name == "runtime-source.asset-manifest.json"
                            else spec.asset_revisions.get(manifest_value, spec.model_revision)
                        ),
                    )
                    if hash_matches
                    else "manifest hash mismatch"
                )
                ready = hash_matches and detail == "verified"
            except (OSError, ValueError) as exc:
                ready = False
                detail = str(exc)
            items.append(
                ProbeItem(
                    name=f"asset_manifest:{manifest_value}",
                    ready=ready,
                    detail=detail,
                    action=None if ready else f"Rerun Setup for {backend_id}; cached assets failed verification.",
                )
            )
        if live and all(item.ready for item in items):
            self.last_cleanup.pop(backend_id, None)
            try:
                result = self.invoke(backend_id, "health", {}, timeout_seconds=spec.startup_timeout_seconds)
                items.append(
                    ProbeItem(
                        name="live_worker_health",
                        ready=True,
                        detail=json.dumps(result, ensure_ascii=False, sort_keys=True)[:1000],
                    )
                )
            except BackendError as exc:
                items.append(
                    ProbeItem(
                        name="live_worker_health",
                        ready=False,
                        detail=exc.message,
                        action=exc.action or f"Repair the runtime/model for {backend_id} and rerun live Preflight.",
                    )
                )
            finally:
                self.stop_current()
                if self._lease_held:
                    self.lease.release()
                    self._lease_held = False
            cleanup = self.last_cleanup.get(backend_id)
            if cleanup is not None:
                process_exited = bool(cleanup.get("process_exited"))
                gpu_released = bool(cleanup.get("gpu_process_released"))
                within_tolerance = bool(cleanup.get("vram_within_tolerance"))
                post_exit = cleanup.get("post_exit")
                gpu_observable = bool(
                    isinstance(post_exit, dict) and post_exit.get("observable")
                )
                requires_gpu_evidence = bool(
                    spec.requires_cuda or spec.model_family == "llama-server"
                )
                container_absent = bool(cleanup.get("container_absent", True))
                ready = process_exited and gpu_released and container_absent and (
                    not requires_gpu_evidence
                    or (gpu_observable and within_tolerance)
                )
                items.append(
                    ProbeItem(
                        name="live_worker_cleanup",
                        ready=ready,
                        detail=json.dumps(cleanup, ensure_ascii=False, sort_keys=True)[:2000],
                        action=(
                            None
                            if ready
                            else (
                                "Close unrelated GPU applications and repeat live Preflight; "
                                "the managed process exited, but aggregate Windows VRAM did not return near baseline."
                                if process_exited
                                and gpu_released
                                and gpu_observable
                                and not within_tolerance
                                else "Terminate the residual runner process and repeat live Preflight."
                            )
                        ),
                    )
                )
            elif spec.requires_cuda or spec.model_family == "llama-server":
                items.append(
                    ProbeItem(
                        name="live_worker_cleanup",
                        ready=False,
                        detail="the live probe produced no fresh managed GPU cleanup evidence",
                        action=(
                            "Inspect the runner log, terminate any residual managed GPU process, "
                            "and repeat live Preflight."
                        ),
                    )
                )
        return ProbeReport(
            backend_id=backend_id,
            ready=all(item.ready for item in items),
            items=items,
        )

    @staticmethod
    def _docker_json(docker: str, arguments: list[str], *, timeout: float = 30) -> Any:
        completed = subprocess.run(
            [docker, *arguments],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "Docker command failed"
            raise OSError(detail)
        return json.loads(completed.stdout)

    def _probe_docker_runtime(self, spec: RunnerSpec) -> list[ProbeItem]:
        settings = spec.docker
        if settings is None:
            return [ProbeItem(name="docker_runtime", ready=False, detail="Docker settings are missing")]
        docker = shutil.which("docker")
        if not docker:
            return [
                ProbeItem(
                    name="docker_runtime",
                    ready=False,
                    detail="Docker CLI is unavailable",
                    action="Install or start Docker Desktop with its WSL2 Linux engine.",
                )
            ]
        try:
            info = self._docker_json(docker, ["info", "--format", "{{json .}}"])
            image_values = self._docker_json(
                docker,
                ["image", "inspect", settings.image_reference],
            )
            image = image_values[0] if isinstance(image_values, list) and image_values else {}
        except (OSError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
            return [
                ProbeItem(
                    name="docker_runtime",
                    ready=False,
                    detail=f"Docker runtime probe failed: {exc}",
                    action="Start Docker Desktop, then rerun Setup for this Backend.",
                )
            ]
        linux_ready = isinstance(info, dict) and str(info.get("OSType", "")).casefold() == "linux"
        runtimes = info.get("Runtimes", {}) if isinstance(info, dict) else {}
        nvidia_ready = isinstance(runtimes, dict) and "nvidia" in runtimes
        server_version = str(info.get("ServerVersion") or "") if isinstance(info, dict) else ""
        version_ready = server_version == settings.docker_server_version
        image_id = str(image.get("Id") or "") if isinstance(image, dict) else ""
        repo_digests = set(image.get("RepoDigests") or []) if isinstance(image, dict) else set()
        image_ready = image_id == settings.image_id and settings.image_reference in repo_digests
        return [
            ProbeItem(
                name="docker_linux_engine",
                ready=linux_ready,
                detail="Docker Linux engine available" if linux_ready else "Docker is not using a Linux engine",
                action=None if linux_ready else "Enable Docker Desktop's WSL2 Linux engine.",
            ),
            ProbeItem(
                name="docker_nvidia_runtime",
                ready=nvidia_ready,
                detail="NVIDIA container runtime available" if nvidia_ready else "NVIDIA container runtime missing",
                action=None if nvidia_ready else "Repair Docker Desktop GPU support for WSL2.",
            ),
            ProbeItem(
                name="docker_server_version",
                ready=version_ready,
                detail=server_version or "Docker server version unavailable",
                action=None if version_ready else f"Rerun Setup for {spec.backend_id} after the Docker update.",
            ),
            ProbeItem(
                name="docker_image",
                ready=image_ready,
                detail=image_id or "pinned Docker image is unavailable",
                action=None if image_ready else f"Rerun Setup for {spec.backend_id} without --no-download.",
            ),
        ]

    @staticmethod
    def _verify_asset_manifest(manifest_path: Path, *, expected_revision: str | None) -> str:
        manifest = read_json(manifest_path)
        recorded_revisions = {
            str(value)
            for key in ("revision", "source_revision", "xl_revision")
            if (value := manifest.get(key))
        }
        if expected_revision is not None and expected_revision not in recorded_revisions:
            raise ValueError(
                f"asset manifest does not declare expected revision {expected_revision}"
            )
        root = (manifest_path.parent / str(manifest.get("root") or ".")).resolve()
        root.relative_to(manifest_path.parent.resolve())
        files = manifest.get("files")
        if isinstance(files, list):
            if not files:
                raise ValueError("asset manifest contains no files")
            recorded_paths: set[str] = set()
            for item in files:
                if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
                    raise ValueError("asset manifest contains a malformed file entry")
                recorded_paths.add(Path(str(item["path"])).as_posix())
                path = (root / str(item["path"])).resolve()
                path.relative_to(root)
                if not path.is_file():
                    raise ValueError(f"asset file is missing: {item['path']}")
                if item.get("size") is not None and path.stat().st_size != int(item["size"]):
                    raise ValueError(f"asset file size changed: {item['path']}")
                if sha256_file(path) != str(item["sha256"]):
                    raise ValueError(f"asset file hash changed: {item['path']}")
            exact_suffixes = manifest.get("exact_file_suffixes")
            if isinstance(exact_suffixes, list) and exact_suffixes:
                suffixes = {str(value).casefold() for value in exact_suffixes}
                excluded_roots = {
                    str(value).casefold()
                    for value in manifest.get("exact_exclude_roots", [])
                }

                actual_executable_sources: set[str] = set()
                for directory, child_directories, filenames in os.walk(root):
                    child_directories[:] = [
                        name for name in child_directories if name.casefold() not in excluded_roots
                    ]
                    directory_path = Path(directory)
                    for filename in filenames:
                        candidate = directory_path / filename
                        if candidate.suffix.casefold() in suffixes:
                            actual_executable_sources.add(candidate.relative_to(root).as_posix())
                recorded_executable_sources = {
                    value
                    for value in recorded_paths
                    if Path(value).suffix.casefold() in suffixes
                    and not any(part.casefold() in excluded_roots for part in Path(value).parts)
                }
                if actual_executable_sources != recorded_executable_sources:
                    added = sorted(actual_executable_sources - recorded_executable_sources)
                    removed = sorted(recorded_executable_sources - actual_executable_sources)
                    changed = (added or removed)[0]
                    state = "unexpected" if added else "missing"
                    raise ValueError(f"{state} executable runtime source: {changed}")
            return "verified"
        filename = manifest.get("file")
        expected = manifest.get("sha256")
        if filename and expected:
            path = (root / str(filename)).resolve()
            path.relative_to(root)
            if not path.is_file() or sha256_file(path) != str(expected):
                raise ValueError(f"asset file hash changed: {filename}")
            return "verified"
        raise ValueError("asset manifest has no verifiable files")

    def _command(
        self,
        spec: RunnerSpec,
        *,
        container_name: str = "",
    ) -> tuple[list[str], dict[str, str]]:
        inherited_names = {
            "PATH",
            "PATHEXT",
            "SYSTEMROOT",
            "WINDIR",
            "COMSPEC",
            "TEMP",
            "TMP",
            "HOME",
            "USERPROFILE",
            "CUDA_PATH",
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
            "WSLENV",
        }
        environment = {name: value for name, value in os.environ.items() if name.upper() in inherited_names}
        hf_modules_directory = self.run_root / "scratch" / f"hf-modules-{uuid.uuid4().hex}"
        environment.update(spec.environment)
        environment.update(
            {
                "HF_HOME": str(self.model_root / "huggingface"),
                "HUGGINGFACE_HUB_CACHE": str(self.model_root / "huggingface" / "hub"),
                "HF_MODULES_CACHE": str(hf_modules_directory),
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_OFFLINE": "1",
                "DIFFUSERS_OFFLINE": "1",
                "PYTHONNOUSERSITE": "1",
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
                "VIDEO_GENERATOR_PROJECT_ROOT": str(self.project_root),
                "VIDEO_GENERATOR_RUN_ROOT": str(self.run_root),
            }
        )
        if spec.platform in {"native", "docker"}:
            if spec.platform == "docker":
                if spec.docker is None or not container_name:
                    raise BackendError(
                        f"Docker runner for {spec.backend_id} is missing managed runtime settings",
                        kind=ErrorKind.NOT_READY,
                        action=f"Rerun Setup for {spec.backend_id}.",
                    )
                environment.update(
                    {
                        "VIDEO_GENERATOR_DOCKER_CONTAINER_NAME": container_name,
                        "VIDEO_GENERATOR_DOCKER_IMAGE_REFERENCE": spec.docker.image_reference,
                        "VIDEO_GENERATOR_DOCKER_IMAGE_ID": spec.docker.image_id,
                        "VIDEO_GENERATOR_DOCKER_INTERNAL_PORT": str(spec.docker.internal_port),
                        "VIDEO_GENERATOR_DOCKER_SERVER_REVISION": spec.docker.server_revision,
                    }
                )
            return list(spec.command), environment
        wsl = shutil.which("wsl.exe") or shutil.which("wsl")
        if not wsl or not spec.wsl_distribution:
            raise BackendError(
                f"WSL runner for {spec.backend_id} is not ready",
                kind=ErrorKind.NOT_READY,
                action=f"Run: video-generator setup --backend {spec.backend_id}",
            )
        wsl_root = windows_to_wsl(self.project_root)
        wsl_model_root = windows_to_wsl(self.model_root)
        wsl_run_root = windows_to_wsl(self.run_root)
        wsl_hf_modules_directory = windows_to_wsl(hf_modules_directory)
        assignments = [
            f"HF_HOME={wsl_model_root}/huggingface",
            f"HUGGINGFACE_HUB_CACHE={wsl_model_root}/huggingface/hub",
            f"HF_MODULES_CACHE={wsl_hf_modules_directory}",
            "TRANSFORMERS_OFFLINE=1",
            "HF_HUB_OFFLINE=1",
            "DIFFUSERS_OFFLINE=1",
            "PYTHONNOUSERSITE=1",
            "PYTHONUTF8=1",
            "PYTHONIOENCODING=utf-8",
            f"VIDEO_GENERATOR_PROJECT_ROOT={wsl_root}",
            f"VIDEO_GENERATOR_RUN_ROOT={wsl_run_root}",
        ]
        assignments.extend(f"{key}={value}" for key, value in spec.environment.items())
        command = [
            wsl,
            "-d",
            spec.wsl_distribution,
            "--cd",
            wsl_root,
            "--",
            "env",
            *assignments,
            *spec.command,
        ]
        return command, environment

    def _start(self, spec: RunnerSpec) -> _RunnerProcess:
        if not self._lease_held:
            self.lease.acquire()
            self._lease_held = True
        container_name = (
            f"video-generator-{runner_slug(spec.backend_id)}-{uuid.uuid4().hex[:12]}"
            if spec.platform == "docker"
            else ""
        )
        command, environment = self._command(spec, container_name=container_name)
        logs = self.run_root / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        stderr_handle = (logs / f"runner-{runner_slug(spec.backend_id)}.log").open(
            "a", encoding="utf-8", errors="replace"
        )
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        gpu_baseline: dict[str, Any] = {}
        if spec.requires_cuda or spec.model_family == "llama-server":
            from .workers.llama_server import gpu_snapshot

            baseline = gpu_snapshot()
            gpu_baseline = {
                "observable": baseline.observable,
                "used_mb": baseline.used_mb,
                "process_ids": list(baseline.process_ids),
            }
        try:
            process = subprocess.Popen(
                command,
                cwd=self.project_root,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr_handle,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
        except OSError as exc:
            stderr_handle.close()
            raise BackendError(
                f"could not start runner for {spec.backend_id}: {exc}", kind=ErrorKind.NOT_READY
            ) from exc
        responses: queue.Queue[dict[str, Any]] = queue.Queue()

        def read_stdout() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                try:
                    value = json.loads(line)
                    if isinstance(value, dict):
                        responses.put(value)
                except json.JSONDecodeError:
                    responses.put(
                        {
                            "protocol_version": 1,
                            "request_id": "",
                            "ok": False,
                            "error": {
                                "kind": "invalid_output",
                                "message": "runner wrote non-JSON data to stdout",
                            },
                        }
                    )

        reader = threading.Thread(target=read_stdout, name=f"runner-{runner_slug(spec.backend_id)}", daemon=True)
        reader.start()
        running = _RunnerProcess(
            spec,
            process,
            responses,
            reader,
            stderr_handle,
            gpu_baseline=gpu_baseline,
            container_name=container_name,
        )
        self.current = running
        try:
            running.health = self._invoke_current(
                "health", {}, timeout=spec.startup_timeout_seconds
            )
        except BaseException:
            self.stop_current()
            if self._lease_held:
                self.lease.release()
                self._lease_held = False
            raise
        return running

    def _ensure(self, backend_id: str) -> _RunnerProcess:
        spec = self.load_spec(backend_id)
        if self.current and self.current.spec.backend_id == spec.backend_id:
            if self.current.process.poll() is None:
                return self.current
        self.stop_current()
        return self._start(spec)

    def invoke(
        self,
        backend_id: str,
        operation: str,
        payload: Mapping[str, Any],
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        with self._request_lock:
            runner = self._ensure(backend_id)
            return self._invoke_current(operation, dict(payload), timeout=timeout_seconds or runner.spec.timeout_seconds)

    def _invoke_current(
        self,
        operation: str,
        payload: dict[str, Any],
        *,
        timeout: float,
        stop_on_failure: bool = True,
    ) -> dict[str, Any]:
        runner = self.current
        if runner is None or runner.process.poll() is not None or runner.process.stdin is None:
            raise BackendError("local runner exited unexpectedly", kind=ErrorKind.INTERNAL)
        request_id = uuid.uuid4().hex
        envelope = {
            "protocol_version": 1,
            "request_id": request_id,
            "operation": operation,
            "payload": payload,
        }
        try:
            runner.process.stdin.write(json.dumps(envelope, ensure_ascii=False) + "\n")
            runner.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            if stop_on_failure:
                self.stop_current()
            raise BackendError("local runner pipe closed", kind=ErrorKind.INTERNAL) from exc
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if stop_on_failure:
                    self.stop_current()
                raise BackendError(f"local runner timed out after {timeout:.0f}s", kind=ErrorKind.TRANSIENT)
            try:
                response = runner.responses.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                if runner.process.poll() is not None:
                    if stop_on_failure:
                        self.stop_current()
                    raise BackendError("local runner crashed", kind=ErrorKind.INTERNAL)
                continue
            if response.get("request_id") not in {request_id, ""}:
                continue
            if response.get("protocol_version") != 1:
                raise BackendError("local runner protocol version mismatch", kind=ErrorKind.UNSUPPORTED)
            if not response.get("ok"):
                error = response.get("error") if isinstance(response.get("error"), dict) else {}
                kind_value = str(error.get("kind") or "internal")
                try:
                    kind = ErrorKind(kind_value)
                except ValueError:
                    kind = ErrorKind.INTERNAL
                raise BackendError(
                    str(error.get("message") or "local runner failed"),
                    kind=kind,
                    action=error.get("action"),
                    details=error.get("details") if isinstance(error.get("details"), dict) else {},
                )
            result = response.get("result")
            if not isinstance(result, dict):
                raise BackendError("local runner result was not an object", kind=ErrorKind.INVALID_OUTPUT)
            return result

    @staticmethod
    def _valid_managed_container_name(value: str) -> bool:
        return bool(
            re.fullmatch(r"video-generator-[a-z0-9][a-z0-9_.-]{1,110}", value)
        )

    @classmethod
    def _docker_container_exists(cls, docker: str, container_name: str) -> bool:
        if not cls._valid_managed_container_name(container_name):
            return False
        completed = subprocess.run(
            [docker, "container", "inspect", container_name],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
        return completed.returncode == 0

    @classmethod
    def _remove_managed_container(cls, container_name: str) -> bool:
        if not cls._valid_managed_container_name(container_name):
            return False
        docker = shutil.which("docker")
        if not docker:
            return False
        try:
            if cls._docker_container_exists(docker, container_name):
                subprocess.run(
                    [docker, "rm", "-f", container_name],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=90,
                    check=False,
                )
            return not cls._docker_container_exists(docker, container_name)
        except (OSError, subprocess.TimeoutExpired):
            return False

    def stop_current(self) -> None:
        runner = self.current
        if runner is None:
            return
        process = runner.process
        platform = str(getattr(runner.spec, "platform", "native"))
        lifecycle: dict[str, Any] | None = None
        forced = False
        if process.poll() is None and process.stdin is not None:
            try:
                result = self._invoke_current(
                    "shutdown",
                    {},
                    timeout=90 if platform == "docker" else 45,
                    stop_on_failure=False,
                )
                reported_lifecycle = result.get("lifecycle")
                if isinstance(reported_lifecycle, dict):
                    lifecycle = dict(reported_lifecycle)
                process.wait(timeout=10)
            except (BackendError, OSError, subprocess.TimeoutExpired):
                forced = True
                self._kill_process_tree(process)
        container_absent = True
        container_name = str(getattr(runner, "container_name", "") or "")
        if platform == "docker":
            container_absent = self._remove_managed_container(container_name)
            if not container_absent:
                forced = True
        requires_gpu_evidence = bool(
            getattr(runner.spec, "requires_cuda", False)
            or runner.spec.model_family == "llama-server"
        )
        if lifecycle is None and requires_gpu_evidence:
            lifecycle = {}
        if lifecycle is not None:
            post_exit = lifecycle.get("post_exit")
            needs_observable_host_probe = bool(
                requires_gpu_evidence
                and not (
                    isinstance(post_exit, dict)
                    and post_exit.get("observable") is True
                )
            )
            if (
                forced
                or not {"process_exited", "gpu_process_released"}.issubset(lifecycle)
                or needs_observable_host_probe
            ):
                from .workers.llama_server import gpu_snapshot

                post_exit = gpu_snapshot()
                process_exited = process.poll() is not None
                tracked_pids = {process.pid}
                server_pid = getattr(runner, "health", {}).get("server_pid")
                if isinstance(server_pid, int) and server_pid > 0:
                    tracked_pids.add(server_pid)
                gpu_baseline = getattr(runner, "gpu_baseline", {})
                baseline_used_mb = (
                    gpu_baseline.get("used_mb")
                    if isinstance(gpu_baseline, dict)
                    else None
                )
                vram_tolerance_mb = int(lifecycle.get("vram_tolerance_mb") or 512)
                vram_within_tolerance = bool(
                    process_exited
                    and post_exit.observable
                    and baseline_used_mb is not None
                    and post_exit.used_mb is not None
                    and post_exit.used_mb <= baseline_used_mb + vram_tolerance_mb
                )
                lifecycle = {
                    **lifecycle,
                    "worker_pid": process.pid,
                    "forced": forced,
                    "process_exited": process_exited,
                    "gpu_process_released": (
                        process_exited
                        and post_exit.observable
                        and tracked_pids.isdisjoint(post_exit.process_ids)
                    ),
                    "vram_within_tolerance": vram_within_tolerance,
                    "vram_tolerance_mb": vram_tolerance_mb,
                    "manager_baseline": gpu_baseline,
                    "post_exit": {
                        "observable": post_exit.observable,
                        "used_mb": post_exit.used_mb,
                        "process_ids": list(post_exit.process_ids),
                    },
                }
            if platform == "docker":
                lifecycle = {
                    **lifecycle,
                    "container_name": container_name,
                    "container_absent": container_absent,
                }
            self.last_cleanup[runner.spec.backend_id] = lifecycle
            self._cleanup_sequence += 1
            atomic_write_json(
                self.run_root
                / "logs"
                / (
                    f"runner-cleanup-{self._cleanup_sequence:03d}-"
                    f"{runner_slug(runner.spec.backend_id)}.json"
                ),
                {
                    "backend_id": runner.spec.backend_id,
                    "model_family": runner.spec.model_family,
                    "lifecycle": lifecycle,
                },
            )
        self.current = None
        if process.stdin:
            process.stdin.close()
        if process.stdout:
            process.stdout.close()
        runner.reader.join(timeout=5)
        runner.stderr_handle.close()

    @staticmethod
    def _kill_process_tree(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(
                ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        else:
            process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)

    def close(self) -> None:
        self.stop_current()
        if self._lease_held:
            self.lease.release()
            self._lease_held = False

    def __enter__(self) -> "RunnerManager":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
