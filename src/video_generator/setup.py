from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .contracts import ProbeItem
from .errors import ErrorKind, VideoGeneratorError
from .local_llm import LocalLlmProfile, load_local_llm_profile
from .profiles import BACKEND_DESCRIPTORS, PROFILES
from .runners import RunnerManager, RunnerSpec, decode_wsl_output, runner_slug, windows_to_wsl
from .util import atomic_write_json, atomic_write_text, relative_path, replace_path, sha256_file


VOXCPM_REVISION = "bffb3df5a29440629464e5e839f4d214c8714c3d"
FASTER_WHISPER_REVISION = "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"
PARAKEET_REVISION = "7c35754d166cca382ad1e53e68b01e7c575f3a1d"
FLUX_REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
ACE_REPOSITORY_TAG = "v0.1.8"
ACE_REPOSITORY_REVISION = "dce621408bee8c31b4fcf4811682eb9359e1bc94"
ACE_XL_REVISION = "d4a0b288b83ebb7e25a8c0b32c573c22e134e8ee"
MINIMUM_UV_TORCH_BACKEND_VERSION = (0, 6, 9)


@dataclass(frozen=True)
class LocalDefinition:
    backend_id: str
    kind: str
    platform: str
    python_version: str
    requirements_name: str
    model_repo: str
    model_revision: str
    model_subdir: str
    allow_patterns: tuple[str, ...] = ()


@dataclass(frozen=True)
class CuratedLlmArtifact:
    role: str
    filename: str
    sha256: str


@dataclass(frozen=True)
class CuratedLlmCandidate:
    candidate_id: str
    model_id: str
    repository: str
    revision: str
    license_name: str
    quantization: str
    mtp: str
    speculative_tokens: int
    estimated_download_gb: float
    artifacts: tuple[CuratedLlmArtifact, ...]


HUGGINGFACE_HUB_TOOL_VERSION = "0.36.2"
_DOWNLOAD_ENVIRONMENT_NAMES = {
    "ALL_PROXY",
    "APPDATA",
    "COMSPEC",
    "HOME",
    "HOMEDRIVE",
    "HOMEPATH",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "LOCALAPPDATA",
    "NO_PROXY",
    "PATH",
    "PATHEXT",
    "PROGRAMDATA",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "WINDIR",
    "http_proxy",
    "https_proxy",
    "no_proxy",
}
_DOWNLOAD_ENVIRONMENT_NAMES_CASEFOLD = {name.casefold() for name in _DOWNLOAD_ENVIRONMENT_NAMES}

# These are benchmark candidates, not silently selected defaults. Each entry pins the exact
# third-party quantization named in docs/model-matrix.md and independently verified file hashes.
CURATED_LLM_CANDIDATES: dict[str, CuratedLlmCandidate] = {
    "qwen3.6-27b-q4-mtp": CuratedLlmCandidate(
        candidate_id="qwen3.6-27b-q4-mtp",
        model_id="Qwen/Qwen3.6-27B",
        repository="unsloth/Qwen3.6-27B-MTP-GGUF",
        revision="5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace",
        license_name="Apache-2.0",
        quantization="UD-Q4_K_XL",
        mtp="embedded",
        speculative_tokens=2,
        estimated_download_gb=17.9,
        artifacts=(
            CuratedLlmArtifact(
                role="model",
                filename="Qwen3.6-27B-UD-Q4_K_XL.gguf",
                sha256="4085665ee36d82a672a238a43f0e5643f2f0e39f2d7bd5d373f0ef10ecf53095",
            ),
        ),
    ),
    "gemma-4-26b-a4b-q4-mtp": CuratedLlmCandidate(
        candidate_id="gemma-4-26b-a4b-q4-mtp",
        model_id="google/gemma-4-26b-a4b-it-qat",
        repository="unsloth/gemma-4-26B-A4B-it-qat-GGUF",
        revision="9e8946010e8234901f15b8c10e74b51723c26832",
        license_name="Apache-2.0",
        quantization="UD-Q4_K_XL",
        mtp="separate-drafter",
        speculative_tokens=4,
        estimated_download_gb=14.5,
        artifacts=(
            CuratedLlmArtifact(
                role="model",
                filename="gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
                sha256="dcf179a91153e3a7ece792e48ef872180d9d6ef9b7677f0a0bd3e83cfe624d5e",
            ),
            CuratedLlmArtifact(
                role="draft-model",
                filename="mtp-gemma-4-26B-A4B-it.gguf",
                sha256="62bd3af7f66c9308de9a5454233852f8c7324c93767e8dfb824ed45b9179864a",
            ),
        ),
    ),
}


LOCAL_DEFINITIONS: dict[str, LocalDefinition] = {
    "local:voxcpm2": LocalDefinition(
        backend_id="local:voxcpm2",
        kind="voxcpm",
        platform="native",
        python_version="3.11",
        requirements_name="voxcpm.in",
        model_repo="openbmb/VoxCPM2",
        model_revision=VOXCPM_REVISION,
        model_subdir="voxcpm2",
    ),
    "local:parakeet-tdt-0.6b-v3": LocalDefinition(
        backend_id="local:parakeet-tdt-0.6b-v3",
        kind="parakeet",
        platform="wsl",
        python_version="3.12",
        requirements_name="parakeet.in",
        model_repo="nvidia/parakeet-tdt-0.6b-v3",
        model_revision=PARAKEET_REVISION,
        model_subdir="parakeet-tdt-0.6b-v3",
        allow_patterns=("*.nemo", "README.md", "LICENSE*"),
    ),
    "local:faster-whisper-large-v3-turbo": LocalDefinition(
        backend_id="local:faster-whisper-large-v3-turbo",
        kind="faster-whisper",
        platform="native",
        python_version="3.11",
        requirements_name="faster-whisper.in",
        model_repo="dropbox-dash/faster-whisper-large-v3-turbo",
        model_revision=FASTER_WHISPER_REVISION,
        model_subdir="faster-whisper-large-v3-turbo",
        allow_patterns=(
            "config.json",
            "model.bin",
            "preprocessor_config.json",
            "tokenizer.json",
            "vocabulary.json",
            "README.md",
        ),
    ),
    "local:flux.2-klein-4b": LocalDefinition(
        backend_id="local:flux.2-klein-4b",
        kind="flux",
        platform="native",
        python_version="3.11",
        requirements_name="flux.in",
        model_repo="black-forest-labs/FLUX.2-klein-4B",
        model_revision=FLUX_REVISION,
        model_subdir="flux.2-klein-4b",
    ),
}


def selected_backends(*, profile: str | None, backend_id: str | None) -> list[str]:
    if profile and backend_id:
        raise ValueError("--profile and --backend are mutually exclusive")
    if backend_id:
        if backend_id not in BACKEND_DESCRIPTORS:
            raise ValueError(f"unknown Backend: {backend_id}")
        return [backend_id]
    profile = profile or "local"
    try:
        return sorted(
            backend_id
            for backend_id in set(PROFILES[profile].values())
            if BACKEND_DESCRIPTORS[backend_id].revision != "evaluation-gated"
        )
    except KeyError as exc:
        raise ValueError(f"unknown Run Profile: {profile}") from exc


def _run(command: Sequence[str], *, cwd: Path, environment: Mapping[str, str] | None = None, timeout: float = 7200) -> str:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(environment) if environment is not None else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        detail = "\n".join((completed.stderr or completed.stdout).splitlines()[-40:])
        raise VideoGeneratorError(
            f"Setup command failed ({Path(command[0]).name} exit {completed.returncode}):\n{detail}",
            kind=ErrorKind.NOT_READY,
        )
    return completed.stdout


def _native_python(environment_root: Path) -> Path:
    return environment_root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _requirements_path(name: str) -> Path:
    return Path(__file__).resolve().parent / "assets" / "runners" / name


def _find_uv() -> str | None:
    sibling = Path(sys.executable).with_name("uv.exe" if os.name == "nt" else "uv")
    if sibling.is_file():
        return str(sibling)
    return shutil.which("uv")


def _parse_uv_version(value: str) -> tuple[int, int, int] | None:
    match = re.search(r"\buv\s+(\d+)\.(\d+)\.(\d+)\b", value)
    return tuple(int(part) for part in match.groups()) if match else None


def _validate_curated_llm_candidate(candidate: CuratedLlmCandidate, requested_id: str) -> None:
    if candidate.candidate_id != requested_id or not re.fullmatch(
        r"[a-z0-9][a-z0-9._-]{0,79}", candidate.candidate_id
    ):
        raise VideoGeneratorError("invalid curated local LLM candidate ID", kind=ErrorKind.NOT_READY)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", candidate.repository):
        raise VideoGeneratorError("invalid curated Hugging Face repository", kind=ErrorKind.NOT_READY)
    if not re.fullmatch(r"[0-9a-fA-F]{40}", candidate.revision) or set(
        candidate.revision.casefold()
    ) == {"0"}:
        raise VideoGeneratorError("invalid curated model revision", kind=ErrorKind.NOT_READY)
    if not candidate.artifacts or sum(artifact.role == "model" for artifact in candidate.artifacts) != 1:
        raise VideoGeneratorError(
            "a curated candidate must contain exactly one model artifact",
            kind=ErrorKind.NOT_READY,
        )
    names: set[str] = set()
    for artifact in candidate.artifacts:
        path = Path(artifact.filename)
        folded = artifact.filename.casefold()
        if (
            path.name != artifact.filename
            or path.is_absolute()
            or path.suffix.casefold() != ".gguf"
            or folded in names
        ):
            raise VideoGeneratorError(
                f"invalid curated model filename: {artifact.filename}",
                kind=ErrorKind.NOT_READY,
            )
        names.add(folded)
        if not re.fullmatch(r"[0-9a-fA-F]{64}", artifact.sha256) or set(
            artifact.sha256.casefold()
        ) == {"0"}:
            raise VideoGeneratorError(
                f"invalid curated SHA-256 for {artifact.filename}",
                kind=ErrorKind.NOT_READY,
            )


def _curated_llm_paths(project_root: Path, candidate: CuratedLlmCandidate) -> tuple[Path, Path]:
    project_root = project_root.resolve()
    managed_root = (project_root / ".cache" / "models" / "llm").resolve()
    try:
        managed_root.relative_to(project_root)
    except ValueError as exc:
        raise VideoGeneratorError(
            "the managed local LLM cache resolves outside the project",
            kind=ErrorKind.NOT_READY,
        ) from exc
    destination = (managed_root / candidate.candidate_id).resolve()
    try:
        destination.relative_to(managed_root)
    except ValueError as exc:
        raise VideoGeneratorError(
            "the curated model destination resolves outside the managed cache",
            kind=ErrorKind.NOT_READY,
        ) from exc
    return managed_root, destination


def _model_download_environment(environment: Mapping[str, str], project_root: Path) -> dict[str, str]:
    download_environment = {
        name: value
        for name, value in environment.items()
        if name.casefold() in _DOWNLOAD_ENVIRONMENT_NAMES_CASEFOLD and value
    }
    download_environment.update(
        {
            "HF_HOME": str(project_root / ".cache" / "models" / "huggingface"),
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_DOWNLOAD_TIMEOUT": "60",
            "NO_COLOR": "1",
            "UV_CACHE_DIR": str(project_root / ".cache" / "tools" / "uv"),
        }
    )
    return download_environment


def download_curated_llm_candidate(
    *,
    project_root: Path,
    candidate_id: str,
    environment: Mapping[str, str],
) -> Path:
    try:
        candidate = CURATED_LLM_CANDIDATES[candidate_id]
    except KeyError as exc:
        raise VideoGeneratorError(
            f"unknown curated local LLM candidate: {candidate_id}",
            kind=ErrorKind.NOT_READY,
        ) from exc

    project_root = project_root.resolve()
    _validate_curated_llm_candidate(candidate, candidate_id)
    managed_root, destination = _curated_llm_paths(project_root, candidate)
    missing: list[CuratedLlmArtifact] = []
    verified_hashes: dict[str, str] = {}
    for artifact in candidate.artifacts:
        path = destination / artifact.filename
        if path.is_file():
            actual = sha256_file(path)
            if actual.casefold() == artifact.sha256.casefold():
                verified_hashes[artifact.filename] = actual
                continue
        missing.append(artifact)

    if missing:
        uv = _find_uv()
        if not uv:
            raise VideoGeneratorError(
                "uv is required to download curated local LLM candidates",
                kind=ErrorKind.NOT_READY,
            )
        managed_root.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{candidate.candidate_id}.",
                suffix=".download",
                dir=managed_root,
            )
        )
        download_environment = _model_download_environment(environment, project_root)
        download_environment.update(
            {"TEMP": str(staging), "TMP": str(staging), "TMPDIR": str(staging)}
        )
        _run(
            [
                uv,
                "tool",
                "run",
                "--from",
                f"huggingface-hub=={HUGGINGFACE_HUB_TOOL_VERSION}",
                "hf",
                "download",
                candidate.repository,
                *[artifact.filename for artifact in missing],
                "--revision",
                candidate.revision,
                "--local-dir",
                str(staging),
            ],
            cwd=project_root,
            environment=download_environment,
            timeout=21600,
        )
        staged_hashes: dict[str, str] = {}
        for artifact in missing:
            staged = staging / artifact.filename
            if not staged.is_file():
                raise VideoGeneratorError(
                    f"Hugging Face download did not produce {artifact.filename}",
                    kind=ErrorKind.NOT_READY,
                )
            actual = sha256_file(staged)
            if actual.casefold() != artifact.sha256.casefold():
                staged.unlink(missing_ok=True)
                raise VideoGeneratorError(
                    f"SHA-256 mismatch for downloaded {artifact.filename}",
                    kind=ErrorKind.NOT_READY,
                    action=f"Expected {artifact.sha256.lower()}, got {actual.lower()}.",
                )
            staged_hashes[artifact.filename] = actual
        destination.mkdir(parents=True, exist_ok=True)
        for artifact in missing:
            staged = staging / artifact.filename
            replace_path(staged, destination / artifact.filename)
            verified_hashes[artifact.filename] = staged_hashes[artifact.filename]
        shutil.rmtree(staging, ignore_errors=True)

    files = []
    for artifact in candidate.artifacts:
        path = destination / artifact.filename
        if not path.is_file():
            raise VideoGeneratorError(
                f"curated model artifact is missing: {path}",
                kind=ErrorKind.NOT_READY,
            )
        actual = verified_hashes[artifact.filename]
        files.append(
            {
                "role": artifact.role,
                "path": artifact.filename,
                "size": path.stat().st_size,
                "sha256": actual,
            }
        )
    atomic_write_json(
        destination / "asset-manifest.json",
        {
            "schema_version": 1,
            "candidate_id": candidate.candidate_id,
            "source": f"https://huggingface.co/{candidate.repository}",
            "repository": candidate.repository,
            "revision": candidate.revision,
            "model_id": candidate.model_id,
            "license_name": candidate.license_name,
            "model_card": (
                f"https://huggingface.co/{candidate.repository}/blob/{candidate.revision}/README.md"
            ),
            "quantization": candidate.quantization,
            "mtp": candidate.mtp,
            "speculative_tokens": candidate.speculative_tokens,
            "files": files,
        },
    )
    return destination


def _install_native_environment(project_root: Path, definition: LocalDefinition) -> tuple[Path, Path]:
    uv = _find_uv()
    if not uv:
        raise VideoGeneratorError("uv is required for local Setup", kind=ErrorKind.NOT_READY)
    runtime = project_root / ".cache" / "runtimes" / runner_slug(definition.backend_id)
    runtime.mkdir(parents=True, exist_ok=True)
    lock = runtime / "requirements.lock"
    source = _requirements_path(definition.requirements_name)
    source_marker = runtime / "requirements.source.sha256"
    source_revision = sha256_file(source)
    recorded_source = source_marker.read_text(encoding="utf-8").strip() if source_marker.is_file() else ""
    torch_backend_args = ["--torch-backend", "auto"] if definition.kind in {"voxcpm", "flux"} else []
    if torch_backend_args:
        uv_version = _parse_uv_version(_run([uv, "--version"], cwd=project_root, timeout=30))
        if uv_version is None or uv_version < MINIMUM_UV_TORCH_BACKEND_VERSION:
            raise VideoGeneratorError(
                "local CUDA Setup requires uv 0.6.9 or newer",
                kind=ErrorKind.NOT_READY,
                action=(
                    "Activate the project .venv, run: python -m pip install --upgrade uv, "
                    "then rerun Setup."
                ),
            )
    if not lock.is_file() or recorded_source != source_revision:
        _run(
            [
                uv,
                "pip",
                "compile",
                str(source),
                "--python-version",
                definition.python_version,
                *torch_backend_args,
                "-o",
                str(lock),
            ],
            cwd=project_root,
        )
        atomic_write_text(source_marker, source_revision + "\n")
    python = _native_python(runtime)
    if not python.is_file():
        _run([uv, "venv", "--python", definition.python_version, str(runtime / ".venv")], cwd=project_root)
    _run(
        [uv, "pip", "sync", "--python", str(python), *torch_backend_args, str(lock)],
        cwd=project_root,
        environment=os.environ,
    )
    return python, lock


def _install_wsl_environment(
    project_root: Path, definition: LocalDefinition, distribution: str
) -> tuple[str, Path]:
    wsl = shutil.which("wsl.exe") or shutil.which("wsl")
    if not wsl:
        raise VideoGeneratorError("WSL2 is not installed", kind=ErrorKind.NOT_READY)
    try:
        listed_process = subprocess.run(
            [wsl, "-l", "-q"],
            cwd=project_root,
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise VideoGeneratorError(f"could not list WSL distributions: {exc}", kind=ErrorKind.NOT_READY) from exc
    if listed_process.returncode != 0:
        detail = decode_wsl_output(listed_process.stderr or listed_process.stdout).strip()
        raise VideoGeneratorError(
            f"could not list WSL distributions: {detail or 'wsl.exe failed'}",
            kind=ErrorKind.NOT_READY,
        )
    listed = decode_wsl_output(listed_process.stdout)
    if distribution.casefold() not in {line.strip().casefold() for line in listed.splitlines() if line.strip()}:
        raise VideoGeneratorError(
            f"WSL distribution {distribution!r} is not installed",
            kind=ErrorKind.NOT_READY,
            action="Install a WSL2 distribution explicitly; Setup never installs one automatically.",
        )
    runtime = project_root / ".cache" / "runtimes" / runner_slug(definition.backend_id)
    runtime.mkdir(parents=True, exist_ok=True)
    environment_root = runtime / ".venv"
    requirements = _requirements_path(definition.requirements_name)
    lock = runtime / "requirements.lock"
    source_marker = runtime / "requirements.source.sha256"
    source_revision = sha256_file(requirements)
    recorded_source = source_marker.read_text(encoding="utf-8").strip() if source_marker.is_file() else ""
    if lock.is_file() and recorded_source != source_revision:
        lock.unlink()
    root_wsl = windows_to_wsl(project_root)
    env_wsl = windows_to_wsl(environment_root)
    requirements_wsl = windows_to_wsl(requirements)
    python_wsl = f"{env_wsl}/bin/python"
    if not (environment_root / "bin" / "python").exists():
        _run(
            [
                wsl,
                "-d",
                distribution,
                "--cd",
                root_wsl,
                "--",
                f"python{definition.python_version}",
                "-m",
                "venv",
                env_wsl,
            ],
            cwd=project_root,
        )
    install_source = windows_to_wsl(lock) if lock.is_file() else requirements_wsl
    _run(
        [wsl, "-d", distribution, "--cd", root_wsl, "--", python_wsl, "-m", "pip", "install", "-r", install_source],
        cwd=project_root,
    )
    if not lock.is_file():
        frozen = _run(
            [wsl, "-d", distribution, "--cd", root_wsl, "--", python_wsl, "-m", "pip", "freeze", "--all"],
            cwd=project_root,
        )
        atomic_write_text(lock, frozen)
        atomic_write_text(source_marker, source_revision + "\n")
    return python_wsl, lock


def _download_snapshot(
    *,
    project_root: Path,
    python_command: Sequence[str],
    definition: LocalDefinition,
    destination: Path,
    environment: Mapping[str, str],
    wsl_distribution: str = "",
) -> None:
    command = [
        *python_command,
        "-m",
        "video_generator.workers.prepare",
        "--repo",
        definition.model_repo,
        "--revision",
        definition.model_revision,
        "--destination",
        str(destination),
    ]
    for pattern in definition.allow_patterns:
        command.extend(["--allow", pattern])
    setup_environment = dict(environment)
    setup_environment["PYTHONPATH"] = str(project_root / "src")
    if definition.platform == "wsl":
        wsl = shutil.which("wsl.exe") or shutil.which("wsl")
        root_wsl = windows_to_wsl(project_root)
        destination_wsl = windows_to_wsl(destination)
        translated = list(command)
        destination_index = translated.index("--destination") + 1
        translated[destination_index] = destination_wsl
        source_path = windows_to_wsl(project_root / "src")
        existing_wslenv = setup_environment.get("WSLENV", "")
        setup_environment["WSLENV"] = "HF_TOKEN" + (f":{existing_wslenv}" if existing_wslenv else "")
        command = [
            str(wsl),
            "-d",
            wsl_distribution,
            "--cd",
            root_wsl,
            "--",
            "env",
            f"PYTHONPATH={source_path}",
            *translated,
        ]
    _run(command, cwd=project_root, environment=setup_environment)


def _write_runner_manifest(
    *,
    project_root: Path,
    definition: LocalDefinition,
    command_python: str,
    lock_path: Path,
    model_path: Path,
    wsl_distribution: str = "",
    extra_environment: dict[str, str] | None = None,
) -> Path:
    environment = {
        "PYTHONPATH": str(project_root / "src"),
        "VIDEO_GENERATOR_MODEL_PATH": relative_path(model_path, project_root),
        "VIDEO_GENERATOR_RUNTIME_REVISION": sha256_file(lock_path),
        "VIDEO_GENERATOR_MODEL_REVISION": definition.model_revision,
    }
    environment.update(extra_environment or {})
    if definition.platform == "wsl":
        environment["PYTHONPATH"] = windows_to_wsl(project_root / "src")
    asset_manifest = model_path / "asset-manifest.json" if model_path.is_dir() else model_path.parent / "asset-manifest.json"
    if not asset_manifest.is_file():
        raise VideoGeneratorError(
            f"model asset manifest is missing: {asset_manifest}",
            kind=ErrorKind.NOT_READY,
        )
    spec = RunnerSpec(
        backend_id=definition.backend_id,
        platform=definition.platform,
        command=[command_python, "-m", "video_generator.workers.main", "--kind", definition.kind],
        model_family=definition.kind,
        startup_timeout_seconds=600,
        wsl_distribution=wsl_distribution,
        environment=environment,
        model_paths=[relative_path(model_path, project_root)],
        asset_manifests={
            relative_path(asset_manifest, project_root): sha256_file(asset_manifest)
        },
        runtime_files={
            relative_path(lock_path, project_root): sha256_file(lock_path),
            relative_path(lock_path.parent / "requirements.source.sha256", project_root): sha256_file(
                lock_path.parent / "requirements.source.sha256"
            ),
        },
        runtime_revision=sha256_file(lock_path),
        model_revision=definition.model_revision,
        setup_source_revision=sha256_file(_requirements_path(definition.requirements_name)),
        license_name=BACKEND_DESCRIPTORS[definition.backend_id].license_name,
    )
    path = project_root / ".cache" / "runners" / runner_slug(definition.backend_id) / "runner.json"
    atomic_write_json(path, spec.model_dump(mode="json"))
    return path


def _copy_verified_file(source: Path, destination: Path, expected_sha256: str) -> Path:
    source = source.resolve()
    if not source.is_file():
        raise VideoGeneratorError(f"required local artifact is missing: {source}", kind=ErrorKind.NOT_READY)
    actual = sha256_file(source)
    if actual.casefold() != expected_sha256.casefold():
        raise VideoGeneratorError(
            f"SHA-256 mismatch for {source.name}",
            kind=ErrorKind.NOT_READY,
            action=f"Expected {expected_sha256.lower()}, got {actual.lower()}.",
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256_file(destination).casefold() == actual.casefold():
        return destination
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copy2(source, temporary)
    replace_path(temporary, destination)
    return destination


def _manifest_file(root: Path, path: Path) -> dict[str, str | int]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _remove_untracked_ace_build_tree(runtime: Path, tracked_files: Sequence[str]) -> None:
    runtime = runtime.resolve()
    relative = Path("acestep/third_parts/nano-vllm/build")
    build_path = runtime / relative
    if build_path.is_symlink():
        raise VideoGeneratorError(
            "ACE-Step generated build path is a symbolic link",
            kind=ErrorKind.NOT_READY,
            action="Inspect the pinned ACE-Step checkout before rerunning Setup.",
        )
    build_tree = build_path.resolve()
    build_tree.relative_to(runtime)
    prefix = relative.as_posix().casefold().rstrip("/") + "/"
    tracked_under_build = [
        value
        for value in tracked_files
        if Path(value).as_posix().casefold().startswith(prefix)
    ]
    if tracked_under_build:
        raise VideoGeneratorError(
            "ACE-Step generated build directory contains Git-tracked files",
            kind=ErrorKind.NOT_READY,
            action="Inspect the pinned ACE-Step checkout before removing generated build output.",
        )
    if not build_tree.exists():
        return
    if not build_tree.is_dir():
        raise VideoGeneratorError(
            "ACE-Step generated build path is not a regular directory",
            kind=ErrorKind.NOT_READY,
            action="Inspect the pinned ACE-Step checkout before rerunning Setup.",
        )
    shutil.rmtree(build_tree)


def _sync_tracked_ace_checkpoint_code(
    runtime: Path,
    checkpoint: Path,
    tracked_files: Sequence[str],
) -> list[Path]:
    runtime = runtime.resolve()
    checkpoint = checkpoint.resolve()
    checkpoint.relative_to(runtime / "checkpoints")
    source_dir = runtime / "acestep" / "models" / "xl_turbo"
    tracked = {Path(value).as_posix().casefold() for value in tracked_files}
    synced: list[Path] = []
    for source in sorted(source_dir.glob("*.py")):
        if source.name == "__init__.py":
            continue
        relative = source.relative_to(runtime).as_posix()
        if relative.casefold() not in tracked:
            raise VideoGeneratorError(
                f"ACE-Step checkpoint sync source is not Git-tracked: {relative}",
                kind=ErrorKind.NOT_READY,
                action="Inspect the pinned ACE-Step checkout before rerunning Setup.",
            )
        destination = (checkpoint / source.name).resolve()
        destination.relative_to(checkpoint)
        shutil.copy2(source, destination)
        synced.append(destination)
    return synced


def _refresh_ace_model_asset_manifest(
    checkpoint: Path,
    synced_files: Sequence[Path],
    *,
    expected_revision: str,
) -> None:
    manifest_path = checkpoint / "asset-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("revision") != expected_revision:
        raise VideoGeneratorError(
            "ACE-Step model asset manifest has an unexpected revision",
            kind=ErrorKind.NOT_READY,
            action="Rerun the pinned ACE-Step Setup.",
        )
    entries = {
        str(item.get("path")): item
        for item in manifest.get("files", [])
        if isinstance(item, dict) and item.get("path")
    }
    for path in synced_files:
        relative = path.resolve().relative_to(checkpoint.resolve()).as_posix()
        entry = entries.get(relative)
        if entry is None:
            raise VideoGeneratorError(
                f"ACE-Step model asset manifest does not cover synced file: {relative}",
                kind=ErrorKind.NOT_READY,
                action="Rerun the pinned ACE-Step Setup.",
            )
        entry["size"] = path.stat().st_size
        entry["sha256"] = sha256_file(path)
    atomic_write_json(manifest_path, manifest)


def prepare_llama_server_backend(project_root: Path, profile_path: Path) -> ProbeItem:
    profile_path = profile_path.resolve()
    profile: LocalLlmProfile = load_local_llm_profile(profile_path)
    source_model = profile.resolve_path(profile.model_path, profile_path)
    source_server = profile.resolve_path(profile.llama_server_path, profile_path)
    if source_model.suffix.casefold() != ".gguf":
        raise VideoGeneratorError("local LLM model_path must name a GGUF file", kind=ErrorKind.NOT_READY)
    if source_server.name.casefold() != "llama-server.exe":
        raise VideoGeneratorError(
            "llama_server_path must name llama-server.exe from a stock Windows build",
            kind=ErrorKind.NOT_READY,
        )

    managed_llm_root = (project_root / ".cache" / "models" / "llm").resolve()
    try:
        source_model.relative_to(managed_llm_root)
        model_root = source_model.parent
    except ValueError:
        model_root = managed_llm_root / "assets" / profile.model_sha256[:16].lower()
    model = _copy_verified_file(
        source_model,
        model_root / source_model.name,
        profile.model_sha256,
    )
    model_files = [model]
    draft_model: Path | None = None
    if profile.draft_model_path:
        source_draft = profile.resolve_path(profile.draft_model_path, profile_path)
        if source_draft.suffix.casefold() != ".gguf":
            raise VideoGeneratorError("draft_model_path must name a GGUF file", kind=ErrorKind.NOT_READY)
        draft_model = _copy_verified_file(
            source_draft,
            model_root / source_draft.name,
            profile.draft_model_sha256,
        )
        model_files.append(draft_model)
    model_manifest = model_root / "asset-manifest.json"
    atomic_write_json(
        model_manifest,
        {
            "schema_version": 1,
            "root": ".",
            "repo": profile.model_repo,
            "revision": profile.model_revision,
            "model_id": profile.model_id,
            "license_name": profile.license_name,
            "profile": profile.model_dump(mode="json"),
            "files": [_manifest_file(model_root, path) for path in model_files],
        },
    )

    runtime_root = (
        project_root
        / ".cache"
        / "runtimes"
        / "llama.cpp"
        / f"{profile.llama_cpp_revision[:12].lower()}-{profile.llama_server_sha256[:12].lower()}"
    )
    source_runtime_files = [
        source_server,
        *sorted(source_server.parent.glob("*.dll"), key=lambda path: path.name.casefold()),
    ]
    actual_names = {path.name.casefold() for path in source_runtime_files}
    declared_names = {name.casefold() for name in profile.llama_runtime_files}
    if actual_names != declared_names:
        missing = sorted(declared_names - actual_names)
        extra = sorted(actual_names - declared_names)
        detail = []
        if missing:
            detail.append("missing: " + ", ".join(missing))
        if extra:
            detail.append("undeclared: " + ", ".join(extra))
        raise VideoGeneratorError(
            "llama.cpp runtime files do not match local-llm.toml (" + "; ".join(detail) + ")",
            kind=ErrorKind.NOT_READY,
        )
    server = _copy_verified_file(
        source_server,
        runtime_root / "llama-server.exe",
        profile.llama_server_sha256,
    )
    runtime_files = [server]
    for source_dll in source_runtime_files[1:]:
        expected_hash = next(
            value
            for name, value in profile.llama_runtime_files.items()
            if name.casefold() == source_dll.name.casefold()
        )
        runtime_files.append(
            _copy_verified_file(
                source_dll,
                runtime_root / source_dll.name,
                expected_hash,
            )
        )
    runtime_manifest = runtime_root / "runtime-source.asset-manifest.json"
    atomic_write_json(
        runtime_manifest,
        {
            "schema_version": 1,
            "root": ".",
            "revision": profile.llama_cpp_revision,
            "exact_file_suffixes": [".exe", ".dll"],
            "files": [_manifest_file(runtime_root, path) for path in runtime_files],
        },
    )

    arguments = [
        "--ctx-size",
        str(profile.context_size),
        "--batch-size",
        str(profile.batch_size),
        "--ubatch-size",
        str(profile.micro_batch_size),
        "--n-gpu-layers",
        str(profile.gpu_layers),
        "--parallel",
        "1",
        "--flash-attn",
        "on" if profile.flash_attention else "off",
    ]
    if profile.speculation == "draft-mtp":
        arguments.extend(
            ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(profile.speculative_tokens)]
        )
    requirements = _requirements_path("llama-server.in")
    environment = {
        "PYTHONPATH": str(project_root / "src"),
        "VIDEO_GENERATOR_MODEL_PATH": relative_path(model, project_root),
        "VIDEO_GENERATOR_DRAFT_MODEL_PATH": (
            relative_path(draft_model, project_root) if draft_model is not None else ""
        ),
        "VIDEO_GENERATOR_LLAMA_SERVER": relative_path(server, project_root),
        "VIDEO_GENERATOR_LLAMA_ARGUMENTS": json.dumps(arguments, separators=(",", ":")),
        "VIDEO_GENERATOR_RUNTIME_REVISION": profile.llama_cpp_revision,
        "VIDEO_GENERATOR_MODEL_REVISION": profile.model_revision,
        "VIDEO_GENERATOR_MODEL_ID": profile.model_id,
        "VIDEO_GENERATOR_LLM_PROFILE_ID": profile.profile_id,
        "VIDEO_GENERATOR_LLAMA_CONTEXT": str(profile.context_size),
        "VIDEO_GENERATOR_LLAMA_SPECULATION": profile.speculation,
    }
    spec = RunnerSpec(
        backend_id="local:llama-server",
        platform="native",
        command=[sys.executable, "-m", "video_generator.workers.main", "--kind", "llama-server"],
        model_family="llama-server",
        startup_timeout_seconds=900,
        environment=environment,
        model_paths=[relative_path(path, project_root) for path in model_files],
        asset_manifests={
            relative_path(model_manifest, project_root): sha256_file(model_manifest),
            relative_path(runtime_manifest, project_root): sha256_file(runtime_manifest),
        },
        runtime_files={
            relative_path(path, project_root): sha256_file(path) for path in runtime_files
        },
        runtime_revision=profile.llama_cpp_revision,
        model_revision=profile.model_revision,
        setup_source_revision=sha256_file(requirements),
        license_name=profile.license_name,
        metadata={
            "local_llm_profile": profile.model_dump(mode="json"),
            "model_asset_manifest_sha256": sha256_file(model_manifest),
            "runtime_asset_manifest_sha256": sha256_file(runtime_manifest),
        },
    )
    path = project_root / ".cache" / "runners" / runner_slug(spec.backend_id) / "runner.json"
    atomic_write_json(path, spec.model_dump(mode="json"))
    return ProbeItem(
        name=spec.backend_id,
        ready=True,
        detail=f"prepared profile {profile.profile_id}: {path}",
    )


def prepare_standard_backend(
    *,
    project_root: Path,
    definition: LocalDefinition,
    environment: Mapping[str, str],
    download: bool,
    wsl_distribution: str,
) -> ProbeItem:
    if definition.platform == "native":
        python, lock = _install_native_environment(project_root, definition)
        python_command = [str(python)]
        command_python = str(python)
    else:
        command_python, lock = _install_wsl_environment(project_root, definition, wsl_distribution)
        python_command = [command_python]
    model_root = project_root / ".cache" / "models" / definition.model_subdir
    model_path = model_root
    if download:
        _download_snapshot(
            project_root=project_root,
            python_command=python_command,
            definition=definition,
            destination=model_root,
            environment=environment,
            wsl_distribution=wsl_distribution,
        )
    if not (model_root / "asset-manifest.json").is_file():
        raise VideoGeneratorError(
            f"model snapshot is not prepared: {model_root}",
            kind=ErrorKind.NOT_READY,
            action=f"Run Setup again without --no-download for {definition.backend_id}.",
        )
    if definition.kind == "parakeet":
        nemo_files = list(model_root.glob("*.nemo"))
        if len(nemo_files) != 1:
            raise VideoGeneratorError("Parakeet Setup expected exactly one .nemo file", kind=ErrorKind.NOT_READY)
        model_path = nemo_files[0]
    path = _write_runner_manifest(
        project_root=project_root,
        definition=definition,
        command_python=command_python,
        lock_path=lock,
        model_path=model_path,
        wsl_distribution=wsl_distribution,
        extra_environment={"VIDEO_GENERATOR_VOXCPM_OPTIMIZE": "0"} if definition.kind == "voxcpm" else None,
    )
    return ProbeItem(name=definition.backend_id, ready=True, detail=f"prepared: {path}")


def prepare_ace_step(project_root: Path, *, download: bool) -> ProbeItem:
    git = shutil.which("git")
    uv = _find_uv()
    if not git or not uv:
        raise VideoGeneratorError("git and uv are required for ACE-Step Setup", kind=ErrorKind.NOT_READY)
    runtime = project_root / ".cache" / "runtimes" / "ace-step-1.5"
    if not runtime.exists():
        if not download:
            raise VideoGeneratorError("ACE-Step runtime is missing", kind=ErrorKind.NOT_READY)
        _run(
            [git, "clone", "--branch", ACE_REPOSITORY_TAG, "--depth", "1", "https://github.com/ace-step/ACE-Step-1.5.git", str(runtime)],
            cwd=project_root,
        )
    commit = _run([git, "rev-parse", "HEAD"], cwd=runtime, timeout=60).strip()
    if commit.casefold() != ACE_REPOSITORY_REVISION.casefold():
        raise VideoGeneratorError(
            f"ACE-Step checkout is {commit}, expected {ACE_REPOSITORY_REVISION}",
            kind=ErrorKind.NOT_READY,
            action="Remove the incomplete ACE-Step runtime directory and rerun Setup to clone the pinned release.",
        )
    _run([uv, "sync", "--locked"], cwd=runtime)
    tracked_files = [line.strip() for line in _run([git, "ls-files"], cwd=runtime, timeout=60).splitlines() if line.strip()]
    _remove_untracked_ace_build_tree(runtime, tracked_files)
    status_lines = [
        line
        for line in _run(
            [git, "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=runtime,
            timeout=60,
        ).splitlines()
        if line.strip()
    ]
    generated_paths = {
        "checkpoints.asset-manifest.json",
        "runtime-source.asset-manifest.json",
    }
    unexpected_changes = []
    for line in status_lines:
        if line.startswith("?? "):
            path = line[3:].strip().replace("\\", "/")
            if (
                path in generated_paths
                or path.startswith(".venv/")
                or path.startswith("checkpoints/")
            ):
                continue
        unexpected_changes.append(line)
    if unexpected_changes:
        raise VideoGeneratorError(
            "ACE-Step runtime contains modified or unexpected source files",
            kind=ErrorKind.NOT_READY,
            action=(
                "Restore or remove the ACE-Step runtime and rerun Setup. "
                f"First unexpected entry: {unexpected_changes[0]}"
            ),
        )
    environment = dict(os.environ)
    environment.update(
        {
            "CHECK_UPDATE": "false",
            "ACESTEP_INIT_LLM": "false",
            "ACESTEP_CHECKPOINTS_DIR": str(runtime / "checkpoints"),
        }
    )
    if download:
        _run([uv, "run", "acestep-download"], cwd=runtime, environment=environment)
        xl = runtime / "checkpoints" / "acestep-v15-xl-turbo"
        _run(
            [
                uv,
                "run",
                "python",
                "-m",
                "video_generator.workers.prepare",
                "--repo",
                "ACE-Step/acestep-v15-xl-turbo",
                "--revision",
                ACE_XL_REVISION,
                "--destination",
                str(xl),
            ],
            cwd=runtime,
            environment={**environment, "PYTHONPATH": str(project_root / "src")},
        )
    core = runtime / "checkpoints"
    xl = core / "acestep-v15-xl-turbo"
    if not core.is_dir() or not xl.is_dir():
        raise VideoGeneratorError("ACE-Step checkpoints are incomplete", kind=ErrorKind.NOT_READY)
    if download:
        synced_files = _sync_tracked_ace_checkpoint_code(runtime, xl, tracked_files)
        _refresh_ace_model_asset_manifest(
            xl,
            synced_files,
            expected_revision=ACE_XL_REVISION,
        )
    checkpoint_manifest = runtime / "checkpoints.asset-manifest.json"
    if download:
        files = []
        for path in sorted(item for item in core.rglob("*") if item.is_file()):
            relative = path.relative_to(core)
            if ".cache" in relative.parts:
                continue
            files.append(
                {
                    "path": relative.as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        if not files:
            raise VideoGeneratorError("ACE-Step checkpoint download produced no files", kind=ErrorKind.NOT_READY)
        atomic_write_json(
            checkpoint_manifest,
            {
                "schema_version": 1,
                "root": "checkpoints",
                "runtime_revision": commit,
                "xl_revision": ACE_XL_REVISION,
                "files": files,
            },
        )
    if not checkpoint_manifest.is_file():
        raise VideoGeneratorError(
            "ACE-Step checkpoint manifest is missing",
            kind=ErrorKind.NOT_READY,
            action="Rerun Setup without --no-download to prepare and hash the checkpoints.",
        )
    runtime_manifest = runtime / "runtime-source.asset-manifest.json"
    exact_file_suffixes = [".py", ".pyi", ".pyd", ".so", ".dll"]
    exact_exclude_roots = [".git", ".venv", "checkpoints", "__pycache__"]
    atomic_write_json(
        runtime_manifest,
        {
            "schema_version": 1,
            "root": ".",
            "revision": ACE_REPOSITORY_REVISION,
            "exact_file_suffixes": exact_file_suffixes,
            "exact_exclude_roots": exact_exclude_roots,
            "files": [
                {
                    "path": value.replace("\\", "/"),
                    "size": (runtime / value).stat().st_size,
                    "sha256": sha256_file(runtime / value),
                }
                for value in tracked_files
                if (runtime / value).is_file()
            ],
        },
    )
    python = _native_python(runtime)
    spec = RunnerSpec(
        backend_id="local:ace-step-1.5-xl-turbo",
        platform="native",
        command=[str(python), "-m", "video_generator.workers.main", "--kind", "acestep"],
        model_family="acestep",
        startup_timeout_seconds=600,
        environment={
            "PYTHONPATH": os.pathsep.join([str(project_root / "src"), str(runtime)]),
            "VIDEO_GENERATOR_ACESTEP_PROJECT_ROOT": str(runtime),
            "ACESTEP_CHECKPOINTS_DIR": str(core),
            "ACESTEP_INIT_LLM": "false",
            "ACESTEP_DISABLE_TQDM": "1",
            "CHECK_UPDATE": "false",
            "VIDEO_GENERATOR_RUNTIME_REVISION": commit,
            "VIDEO_GENERATOR_MODEL_REVISION": ACE_XL_REVISION,
        },
        model_paths=[relative_path(core, project_root), relative_path(xl, project_root)],
        asset_manifests={
            relative_path(checkpoint_manifest, project_root): sha256_file(checkpoint_manifest),
            relative_path(runtime_manifest, project_root): sha256_file(runtime_manifest),
        },
        runtime_files={
            relative_path(runtime / "uv.lock", project_root): sha256_file(runtime / "uv.lock")
        },
        runtime_revision=commit,
        model_revision=ACE_XL_REVISION,
        setup_source_revision=ACE_REPOSITORY_REVISION,
        license_name="MIT",
    )
    path = project_root / ".cache" / "runners" / runner_slug(spec.backend_id) / "runner.json"
    atomic_write_json(path, spec.model_dump(mode="json"))
    return ProbeItem(name=spec.backend_id, ready=True, detail=f"prepared: {path}")


def setup_backends(
    *,
    project_root: Path,
    backend_ids: Sequence[str],
    environment: Mapping[str, str],
    download: bool,
    wsl_distribution: str,
    llm_profile: Path | None = None,
) -> list[ProbeItem]:
    project_root = project_root.resolve()
    if download and "local:llama-server" in backend_ids and llm_profile is None:
        raise VideoGeneratorError(
            "local llama-server Setup requires --llm-profile",
            kind=ErrorKind.NOT_READY,
            action=(
                "Copy local-llm.example.toml, fill exact model/runtime commits, paths, hashes, and "
                "launch settings, then rerun Setup with --llm-profile."
            ),
        )
    if not download:
        results: list[ProbeItem] = []
        with RunnerManager(
            project_root=project_root,
            run_root=project_root / "runs" / ".setup-verification",
        ) as manager:
            for backend_id in backend_ids:
                descriptor = BACKEND_DESCRIPTORS[backend_id]
                if descriptor.cloud:
                    missing = [name for name in descriptor.required_env if not environment.get(name)]
                    results.append(
                        ProbeItem(
                            name=backend_id,
                            ready=not missing,
                            detail="credentials configured" if not missing else f"missing: {', '.join(missing)}",
                            action=None if not missing else f"Add {', '.join(missing)} to .env.",
                        )
                    )
                elif descriptor.provider == "deterministic":
                    results.append(ProbeItem(name=backend_id, ready=True, detail="built into the package"))
                else:
                    report = manager.probe(backend_id)
                    failed = [item.detail for item in report.items if not item.ready]
                    results.append(
                        ProbeItem(
                            name=backend_id,
                            ready=report.ready,
                            detail="runner and assets verified" if report.ready else "; ".join(failed),
                            action=None if report.ready else f"Rerun Setup for {backend_id} without --no-download.",
                        )
                    )
        return results
    llm_result = (
        prepare_llama_server_backend(project_root, llm_profile)
        if "local:llama-server" in backend_ids and llm_profile is not None
        else None
    )
    (project_root / ".cache" / "models").mkdir(parents=True, exist_ok=True)
    results = [llm_result] if llm_result is not None else []
    for backend_id in backend_ids:
        descriptor = BACKEND_DESCRIPTORS[backend_id]
        if descriptor.cloud:
            missing = [name for name in descriptor.required_env if not environment.get(name)]
            results.append(
                ProbeItem(
                    name=backend_id,
                    ready=not missing,
                    detail="credentials configured" if not missing else f"missing: {', '.join(missing)}",
                    action=None if not missing else f"Add {', '.join(missing)} to .env.",
                )
            )
            continue
        if descriptor.provider == "deterministic":
            results.append(ProbeItem(name=backend_id, ready=True, detail="built into the package"))
            continue
        if backend_id == "local:qwen3.6-27b-q4-vision":
            results.append(
                ProbeItem(
                    name=backend_id,
                    ready=False,
                    detail="local Qwen vision remains evaluation-gated on 24 GB VRAM",
                    action="Use draft quality or configure a separately evaluated vision runner.",
                )
            )
            continue
        if backend_id == "local:llama-server":
            continue
        if backend_id == "local:ace-step-1.5-xl-turbo":
            results.append(prepare_ace_step(project_root, download=download))
            continue
        definition = LOCAL_DEFINITIONS.get(backend_id)
        if not definition:
            results.append(ProbeItem(name=backend_id, ready=False, detail="no built-in Setup definition"))
            continue
        results.append(
            prepare_standard_backend(
                project_root=project_root,
                definition=definition,
                environment=environment,
                download=download,
                wsl_distribution=wsl_distribution,
            )
        )
    return results
