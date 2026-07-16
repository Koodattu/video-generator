from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .contracts import ProbeItem
from .errors import ErrorKind, VideoGeneratorError
from .local_llm import LocalLlmProfile, load_local_llm_profile
from .profiles import BACKEND_DESCRIPTORS, PROFILES
from .runners import (
    DockerRuntimeSpec,
    RunnerManager,
    RunnerSpec,
    decode_wsl_output,
    runner_setup_source_revision,
    runner_slug,
    runner_torch_backend,
    windows_to_wsl,
)
from .util import (
    atomic_write_json,
    atomic_write_text,
    read_json,
    relative_path,
    replace_path,
    sha256_file,
)


VOXCPM_REVISION = "bffb3df5a29440629464e5e839f4d214c8714c3d"
FASTER_WHISPER_REVISION = "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"
PARAKEET_REVISION = "7c35754d166cca382ad1e53e68b01e7c575f3a1d"
FLUX_REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
OMNIVOICE_REVISION = "c5fdb5ccb189668d56333f77ba2629f4cd7535f4"
MOSS_TTS_REVISION = "be7766a6735b98bd793f7c79fb720b4d0f5d13b8"
MOSS_AUDIO_TOKENIZER_REVISION = "f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169"
XVOICE_SOURCE_REPOSITORY = "https://github.com/sunnyxrxrx/X-Voice.git"
XVOICE_SOURCE_REVISION = "b1a5d25459aecdea5dfce6e892da384400ac32e9"
XVOICE_MODEL_REVISION = "7f24fe778ddf7a47e25d87e5d5153599c1d4d5c2"
XVOICE_VOCOS_REVISION = "0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21"
XVOICE_MICROMAMBA_VERSION = "2.8.1"
XVOICE_MICROMAMBA_URL = "https://micro.mamba.pm/api/micromamba/win-64/2.8.1"
XVOICE_MICROMAMBA_ARCHIVE_SHA256 = (
    "8648c6d302bb6d7432e5d620bc862c2be14c00f39ddd7e10c5eb6d73250dbba1"
)
XVOICE_MICROMAMBA_EXE_SHA256 = (
    "8a51f88ec02600488ea20c3acd93fbd4da6c0f03fc499aa53fd234c6749b94b0"
)
XVOICE_ESPEAK_VERSION = "1.52.0"
XVOICE_ESPEAK_EXE_SHA256 = (
    "3080ec3822c1b266ef557c710bc79a97d20a7ab133a34bac308b81ab0afc733e"
)
XVOICE_ESPEAK_DLL_SHA256 = (
    "e737572df0a35a32b7bd444537c661c1c916b13b0b91351030c7f1d531307beb"
)
XVOICE_ESPEAK_BUNDLE_SHA256 = (
    "cab76dca69fd1a5c5b6aecff9cd9a62b8668346983bd735bd1fc52ba2e705bc9"
)
XVOICE_FASTTEXT_LID_URL = (
    "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
)
XVOICE_FASTTEXT_LID_SHA256 = (
    "8f3472cfe8738a7b6099e8e999c3cbfae0dcd15696aac7d7738a8039db603e83"
)
XVOICE_NLTK_DATA_REVISION = "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a"
XVOICE_CMUDICT_URL = (
    "https://raw.githubusercontent.com/nltk/nltk_data/"
    f"{XVOICE_NLTK_DATA_REVISION}/packages/corpora/cmudict.zip"
)
XVOICE_CMUDICT_SHA256 = (
    "d07cca47fd72ad32ea9d8ad1219f85301eeaf4568f8b6b73747506a71fb5afd6"
)
HIGGS_MODEL_REVISION = "7556c17e05201fccd9c8cc120bc216dcc7b5d561"
HIGGS_IMAGE_REFERENCE = (
    "lmsysorg/sglang-omni@"
    "sha256:46235435997d1fa93fc81fb1c2d5b7fd8470d77395a5c348c0176094ffddf95e"
)
HIGGS_SERVER_REVISION = "bdb6748882465e0b4c0b3298f95cb6e47adf5e10"
Z_IMAGE_TURBO_REVISION = "f332072aa78be7aecdf3ee76d5c247082da564a6"
IDEOGRAM_4_NF4_REVISION = "1874bc70267ba2c823a7239e1d70dd308c8d64dc"
QWEN_IMAGE_2512_REVISION = "25468b98e3276ca6700de15c6628e51b7de54a26"
ACE_REPOSITORY_TAG = "v0.1.8"
ACE_REPOSITORY_REVISION = "dce621408bee8c31b4fcf4811682eb9359e1bc94"
ACE_XL_REVISION = "d4a0b288b83ebb7e25a8c0b32c573c22e134e8ee"
MINIMUM_UV_TORCH_BACKEND_VERSION = (0, 6, 9)


@dataclass(frozen=True)
class SupportingModelDefinition:
    environment_name: str
    model_repo: str
    model_revision: str
    model_subdir: str
    allow_patterns: tuple[str, ...] = ()


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
    supporting_models: tuple[SupportingModelDefinition, ...] = ()
    timeout_seconds: float = 600
    startup_timeout_seconds: float = 600


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
_SETUP_TOOL_ENVIRONMENT_PREFIXES = ("cuda_", "nvidia_", "pip_", "torch_", "uv_")
_SNAPSHOT_EXECUTABLE_SUFFIXES = {".py", ".pyi", ".pyc", ".pyd", ".dll", ".so"}
_SNAPSHOT_EXCLUDED_ROOTS = {".cache", "__pycache__"}

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
    "eurollm-22b-instruct-2512-q4": CuratedLlmCandidate(
        candidate_id="eurollm-22b-instruct-2512-q4",
        model_id="utter-project/EuroLLM-22B-Instruct-2512",
        repository="bartowski/utter-project_EuroLLM-22B-Instruct-2512-GGUF",
        revision="1dbd312ffa10e83e5faa855e3727cbf4abc45f08",
        license_name="Apache-2.0",
        quantization="Q4_K_M",
        mtp="none",
        speculative_tokens=2,
        estimated_download_gb=13.7,
        artifacts=(
            CuratedLlmArtifact(
                role="model",
                filename="utter-project_EuroLLM-22B-Instruct-2512-Q4_K_M.gguf",
                sha256="2a222374c4adacd55b55795e2f9dca42a2f100d5a2d5858442f928c4c8bdf5e7",
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
    "local:omnivoice": LocalDefinition(
        backend_id="local:omnivoice",
        kind="omnivoice",
        platform="native",
        python_version="3.11",
        requirements_name="omnivoice.in",
        model_repo="k2-fsa/OmniVoice",
        model_revision=OMNIVOICE_REVISION,
        model_subdir="omnivoice",
    ),
    "local:moss-tts-v1.5": LocalDefinition(
        backend_id="local:moss-tts-v1.5",
        kind="moss-tts",
        platform="native",
        python_version="3.12",
        requirements_name="moss-tts.in",
        model_repo="OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
        model_revision=MOSS_TTS_REVISION,
        model_subdir="moss-tts-v1.5",
        supporting_models=(
            SupportingModelDefinition(
                environment_name="VIDEO_GENERATOR_CODEC_PATH",
                model_repo="OpenMOSS-Team/MOSS-Audio-Tokenizer-v2",
                model_revision=MOSS_AUDIO_TOKENIZER_REVISION,
                model_subdir="moss-audio-tokenizer-v2",
            ),
        ),
        timeout_seconds=1200,
        startup_timeout_seconds=1200,
    ),
    "local:x-voice": LocalDefinition(
        backend_id="local:x-voice",
        kind="xvoice",
        platform="native",
        python_version="3.11",
        requirements_name="xvoice.in",
        model_repo="XRXRX/X-Voice",
        model_revision=XVOICE_MODEL_REVISION,
        model_subdir="x-voice-stage1",
        allow_patterns=(
            "XVoice_Base_Stage1/model_600000.safetensors",
            "XVoice_Base_Stage1/vocab.txt",
            "README.md",
            "LICENSE*",
        ),
        supporting_models=(
            SupportingModelDefinition(
                environment_name="VIDEO_GENERATOR_XVOICE_VOCODER_PATH",
                model_repo="charactr/vocos-mel-24khz",
                model_revision=XVOICE_VOCOS_REVISION,
                model_subdir="vocos-mel-24khz",
                allow_patterns=("config.yaml", "pytorch_model.bin", "README.md", "LICENSE*"),
            ),
        ),
        timeout_seconds=1200,
        startup_timeout_seconds=900,
    ),
    "local:z-image-turbo": LocalDefinition(
        backend_id="local:z-image-turbo",
        kind="z-image",
        platform="native",
        python_version="3.11",
        requirements_name="z-image.in",
        model_repo="Tongyi-MAI/Z-Image-Turbo",
        model_revision=Z_IMAGE_TURBO_REVISION,
        model_subdir="z-image-turbo",
        timeout_seconds=900,
        startup_timeout_seconds=900,
    ),
    "local:ideogram-4-nf4": LocalDefinition(
        backend_id="local:ideogram-4-nf4",
        kind="ideogram4",
        platform="native",
        python_version="3.11",
        requirements_name="ideogram4.in",
        model_repo="ideogram-ai/ideogram-4-nf4-diffusers",
        model_revision=IDEOGRAM_4_NF4_REVISION,
        model_subdir="ideogram-4-nf4",
        timeout_seconds=1200,
        startup_timeout_seconds=1200,
    ),
    "local:qwen-image-2512-nf4": LocalDefinition(
        backend_id="local:qwen-image-2512-nf4",
        kind="qwen-image",
        platform="native",
        python_version="3.11",
        requirements_name="qwen-image.in",
        model_repo="Qwen/Qwen-Image-2512",
        model_revision=QWEN_IMAGE_2512_REVISION,
        model_subdir="qwen-image-2512",
        timeout_seconds=1800,
        startup_timeout_seconds=1200,
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


def _setup_tool_environment(environment: Mapping[str, str], project_root: Path) -> dict[str, str]:
    setup_environment = _model_download_environment(environment, project_root)
    setup_environment.update(
        {
            name: value
            for name, value in environment.items()
            if value and name.casefold().startswith(_SETUP_TOOL_ENVIRONMENT_PREFIXES)
        }
    )
    return setup_environment


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
        try:
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
        finally:
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


def _install_native_environment(
    project_root: Path,
    definition: LocalDefinition,
    *,
    environment: Mapping[str, str] | None = None,
) -> tuple[Path, Path]:
    uv = _find_uv()
    if not uv:
        raise VideoGeneratorError("uv is required for local Setup", kind=ErrorKind.NOT_READY)
    runtime = project_root / ".cache" / "runtimes" / runner_slug(definition.backend_id)
    runtime.mkdir(parents=True, exist_ok=True)
    lock = runtime / "requirements.lock"
    source = _requirements_path(definition.requirements_name)
    reviewed_lock = source.with_suffix(".lock")
    source_marker = runtime / "requirements.source.sha256"
    source_revision = runner_setup_source_revision(definition.kind)
    recorded_source = source_marker.read_text(encoding="utf-8").strip() if source_marker.is_file() else ""
    setup_environment = _setup_tool_environment(environment or os.environ, project_root)
    torch_backend = runner_torch_backend(definition.kind)
    torch_backend_args = ["--torch-backend", torch_backend] if torch_backend else []
    if torch_backend_args:
        uv_version = _parse_uv_version(
            _run(
                [uv, "--version"],
                cwd=project_root,
                environment=setup_environment,
                timeout=30,
            )
        )
        if uv_version is None or uv_version < MINIMUM_UV_TORCH_BACKEND_VERSION:
            raise VideoGeneratorError(
                "local CUDA Setup requires uv 0.6.9 or newer",
                kind=ErrorKind.NOT_READY,
                action=(
                    "Activate the project .venv, run: python -m pip install --upgrade uv, "
                    "then rerun Setup."
                ),
            )
    if not reviewed_lock.is_file():
        raise VideoGeneratorError(
            f"reviewed Windows dependency lock is missing: {reviewed_lock}",
            kind=ErrorKind.NOT_READY,
        )
    if (
        not lock.is_file()
        or recorded_source != source_revision
        or sha256_file(lock) != sha256_file(reviewed_lock)
    ):
        _copy_verified_file(reviewed_lock, lock, sha256_file(reviewed_lock))
        atomic_write_text(source_marker, source_revision + "\n")
    python = _native_python(runtime)
    if not python.is_file():
        _run(
            [uv, "venv", "--python", definition.python_version, str(runtime / ".venv")],
            cwd=project_root,
            environment=setup_environment,
        )
    _run(
        [
            uv,
            "pip",
            "sync",
            "--python",
            str(python),
            *torch_backend_args,
            "--require-hashes",
            str(lock),
        ],
        cwd=project_root,
        environment=setup_environment,
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
    project_root = project_root.resolve()
    destination = destination.resolve()
    model_root = (project_root / ".cache" / "models").resolve()
    destination.relative_to(model_root)
    existing_manifest = destination / "asset-manifest.json"
    if existing_manifest.is_file():
        try:
            _verify_snapshot_manifest(existing_manifest, definition)
            return
        except (OSError, ValueError):
            pass
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            suffix=".download",
            dir=destination.parent,
        )
    )
    command = [
        *python_command,
        "-m",
        "video_generator.workers.prepare",
        "--repo",
        definition.model_repo,
        "--revision",
        definition.model_revision,
        "--destination",
        str(staging),
    ]
    for pattern in definition.allow_patterns:
        command.extend(["--allow", pattern])
    setup_environment = _model_download_environment(environment, project_root)
    if environment.get("HF_TOKEN"):
        setup_environment["HF_TOKEN"] = environment["HF_TOKEN"]
    setup_environment["PYTHONPATH"] = str(project_root / "src")
    if definition.platform == "wsl":
        wsl = shutil.which("wsl.exe") or shutil.which("wsl")
        root_wsl = windows_to_wsl(project_root)
        destination_wsl = windows_to_wsl(staging)
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
    backup: Path | None = None
    try:
        _run(command, cwd=project_root, environment=setup_environment)
        staged_manifest = staging / "asset-manifest.json"
        try:
            _verify_snapshot_manifest(staged_manifest, definition)
        except (OSError, ValueError) as exc:
            raise VideoGeneratorError(
                f"downloaded model snapshot failed verification: {exc}",
                kind=ErrorKind.NOT_READY,
            ) from exc
        if destination.exists():
            backup = Path(
                tempfile.mkdtemp(
                    prefix=f".{destination.name}.",
                    suffix=".backup",
                    dir=destination.parent,
                )
            )
            backup.rmdir()
            replace_path(destination, backup)
        try:
            replace_path(staging, destination)
        except BaseException:
            if backup is not None and backup.exists() and not destination.exists():
                replace_path(backup, destination)
            raise
        if backup is not None:
            shutil.rmtree(backup)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _verify_snapshot_manifest(
    manifest_path: Path,
    definition: LocalDefinition,
) -> None:
    manifest = read_json(manifest_path)
    expected_source = f"https://huggingface.co/{definition.model_repo}"
    if manifest.get("source") != expected_source:
        raise ValueError(f"asset manifest source does not match {expected_source}")
    if tuple(manifest.get("allow_patterns") or ()) != definition.allow_patterns:
        raise ValueError("asset manifest allow-pattern policy changed")
    suffixes = {str(value).casefold() for value in manifest.get("exact_file_suffixes") or ()}
    if suffixes != _SNAPSHOT_EXECUTABLE_SUFFIXES:
        raise ValueError("asset manifest executable-source policy is obsolete")
    excluded_roots = {
        str(value).casefold() for value in manifest.get("exact_exclude_roots") or ()
    }
    if excluded_roots != _SNAPSHOT_EXCLUDED_ROOTS:
        raise ValueError("asset manifest executable-source exclusions are obsolete")
    RunnerManager._verify_asset_manifest(
        manifest_path,
        expected_revision=definition.model_revision,
    )


def _write_runner_manifest(
    *,
    project_root: Path,
    definition: LocalDefinition,
    command_python: str,
    lock_path: Path,
    model_path: Path,
    wsl_distribution: str = "",
    extra_environment: dict[str, str] | None = None,
    supporting_model_paths: Mapping[SupportingModelDefinition, Path] | None = None,
) -> Path:
    environment = {
        "PYTHONPATH": str(project_root / "src"),
        "VIDEO_GENERATOR_MODEL_PATH": relative_path(model_path, project_root),
        "VIDEO_GENERATOR_RUNTIME_REVISION": sha256_file(lock_path),
        "VIDEO_GENERATOR_MODEL_REVISION": definition.model_revision,
    }
    environment.update(extra_environment or {})
    supporting_model_paths = dict(supporting_model_paths or {})
    for supporting, path in supporting_model_paths.items():
        environment[supporting.environment_name] = relative_path(path, project_root)
    if definition.platform == "wsl":
        environment["PYTHONPATH"] = windows_to_wsl(project_root / "src")
    asset_manifest = model_path / "asset-manifest.json" if model_path.is_dir() else model_path.parent / "asset-manifest.json"
    if not asset_manifest.is_file():
        raise VideoGeneratorError(
            f"model asset manifest is missing: {asset_manifest}",
            kind=ErrorKind.NOT_READY,
        )
    supporting_manifests = {
        path / "asset-manifest.json": supporting
        for supporting, path in supporting_model_paths.items()
    }
    missing_supporting_manifest = next(
        (path for path in supporting_manifests if not path.is_file()),
        None,
    )
    if missing_supporting_manifest is not None:
        raise VideoGeneratorError(
            f"supporting model asset manifest is missing: {missing_supporting_manifest}",
            kind=ErrorKind.NOT_READY,
        )
    spec = RunnerSpec(
        backend_id=definition.backend_id,
        platform=definition.platform,
        command=[command_python, "-m", "video_generator.workers.main", "--kind", definition.kind],
        model_family=definition.kind,
        timeout_seconds=definition.timeout_seconds,
        startup_timeout_seconds=definition.startup_timeout_seconds,
        wsl_distribution=wsl_distribution,
        environment=environment,
        model_paths=[
            relative_path(path, project_root)
            for path in (model_path, *supporting_model_paths.values())
        ],
        asset_manifests={
            relative_path(path, project_root): sha256_file(path)
            for path in (asset_manifest, *supporting_manifests)
        },
        asset_revisions={
            relative_path(asset_manifest, project_root): definition.model_revision,
            **{
                relative_path(path, project_root): supporting.model_revision
                for path, supporting in supporting_manifests.items()
            },
        },
        runtime_files={
            relative_path(lock_path, project_root): sha256_file(lock_path),
            relative_path(lock_path.parent / "requirements.source.sha256", project_root): sha256_file(
                lock_path.parent / "requirements.source.sha256"
            ),
        },
        runtime_revision=sha256_file(lock_path),
        model_revision=definition.model_revision,
        setup_source_revision=runner_setup_source_revision(definition.kind),
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
        "VIDEO_GENERATOR_LLAMA_STRUCTURED_MODE": profile.structured_output_mode,
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
        setup_source_revision=runner_setup_source_revision("llama-server"),
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
        python, lock = _install_native_environment(
            project_root,
            definition,
            environment=environment,
        )
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
    supporting_model_paths: dict[SupportingModelDefinition, Path] = {}
    for supporting in definition.supporting_models:
        supporting_root = project_root / ".cache" / "models" / supporting.model_subdir
        if download:
            supporting_definition = LocalDefinition(
                backend_id=definition.backend_id,
                kind=definition.kind,
                platform=definition.platform,
                python_version=definition.python_version,
                requirements_name=definition.requirements_name,
                model_repo=supporting.model_repo,
                model_revision=supporting.model_revision,
                model_subdir=supporting.model_subdir,
                allow_patterns=supporting.allow_patterns,
            )
            _download_snapshot(
                project_root=project_root,
                python_command=python_command,
                definition=supporting_definition,
                destination=supporting_root,
                environment=environment,
                wsl_distribution=wsl_distribution,
            )
        if not (supporting_root / "asset-manifest.json").is_file():
            raise VideoGeneratorError(
                f"supporting model snapshot is not prepared: {supporting_root}",
                kind=ErrorKind.NOT_READY,
                action=f"Run Setup again without --no-download for {definition.backend_id}.",
            )
        supporting_model_paths[supporting] = supporting_root
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
        supporting_model_paths=supporting_model_paths,
    )
    return ProbeItem(name=definition.backend_id, ready=True, detail=f"prepared: {path}")


def prepare_higgs_docker_backend(
    project_root: Path,
    *,
    environment: Mapping[str, str],
    download: bool,
) -> ProbeItem:
    docker = shutil.which("docker")
    if not docker:
        raise VideoGeneratorError(
            "Docker Desktop is required for local:higgs-tts-3-4b",
            kind=ErrorKind.NOT_READY,
            action="Install or start Docker Desktop with its WSL2 Linux engine.",
        )
    setup_environment = _setup_tool_environment(environment, project_root)
    info_text = _run(
        [docker, "info", "--format", "{{json .}}"],
        cwd=project_root,
        environment=setup_environment,
        timeout=60,
    )
    try:
        info = json.loads(info_text)
    except json.JSONDecodeError as exc:
        raise VideoGeneratorError(
            "Docker returned invalid runtime metadata",
            kind=ErrorKind.NOT_READY,
        ) from exc
    if str(info.get("OSType") or "").casefold() != "linux":
        raise VideoGeneratorError(
            "Higgs TTS requires Docker Desktop's WSL2 Linux engine",
            kind=ErrorKind.NOT_READY,
        )
    runtimes = info.get("Runtimes") or {}
    if not isinstance(runtimes, dict) or "nvidia" not in runtimes:
        raise VideoGeneratorError(
            "Docker's NVIDIA runtime is unavailable",
            kind=ErrorKind.NOT_READY,
            action="Repair Docker Desktop GPU support for WSL2.",
        )
    inspect = subprocess.run(
        [docker, "image", "inspect", HIGGS_IMAGE_REFERENCE],
        cwd=project_root,
        env=setup_environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )
    if inspect.returncode != 0:
        if not download:
            raise VideoGeneratorError(
                "the pinned SGLang-Omni Docker image is not prepared",
                kind=ErrorKind.NOT_READY,
                action="Rerun Setup without --no-download for local:higgs-tts-3-4b.",
            )
        _run(
            [docker, "pull", HIGGS_IMAGE_REFERENCE],
            cwd=project_root,
            environment=setup_environment,
            timeout=7200,
        )
        inspect_text = _run(
            [docker, "image", "inspect", HIGGS_IMAGE_REFERENCE],
            cwd=project_root,
            environment=setup_environment,
            timeout=60,
        )
    else:
        inspect_text = inspect.stdout
    try:
        image_values = json.loads(inspect_text)
        image = image_values[0]
        image_id = str(image["Id"])
        repo_digests = {str(value) for value in image.get("RepoDigests") or []}
    except (IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise VideoGeneratorError(
            "the pinned SGLang-Omni image could not be attested",
            kind=ErrorKind.NOT_READY,
        ) from exc
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) or HIGGS_IMAGE_REFERENCE not in repo_digests:
        raise VideoGeneratorError(
            "the local SGLang-Omni image does not match its pinned digest",
            kind=ErrorKind.NOT_READY,
        )

    model_definition = LocalDefinition(
        backend_id="local:higgs-tts-3-4b",
        kind="higgs-docker",
        platform="docker",
        python_version="3.12",
        requirements_name="higgs-docker.in",
        model_repo="bosonai/higgs-tts-3-4b",
        model_revision=HIGGS_MODEL_REVISION,
        model_subdir="higgs-tts-3-4b",
        timeout_seconds=1800,
        startup_timeout_seconds=1200,
    )
    model_root = (project_root / ".cache" / "models" / model_definition.model_subdir).resolve()
    model_cache = (project_root / ".cache" / "models").resolve()
    model_root.relative_to(model_cache)
    model_manifest = model_root / "asset-manifest.json"
    if model_manifest.is_file():
        try:
            _verify_snapshot_manifest(model_manifest, model_definition)
        except (OSError, ValueError) as exc:
            raise VideoGeneratorError(
                "the cached Higgs model snapshot failed verification",
                kind=ErrorKind.NOT_READY,
                action="Move the invalid snapshot aside, then rerun Setup.",
            ) from exc
    elif not download:
        raise VideoGeneratorError(
            "the pinned Higgs model snapshot is not prepared",
            kind=ErrorKind.NOT_READY,
            action="Rerun Setup without --no-download for local:higgs-tts-3-4b.",
        )
    else:
        if model_root.exists() and any(model_root.iterdir()):
            raise VideoGeneratorError(
                "the Higgs model directory is nonempty but has no verified asset manifest",
                kind=ErrorKind.NOT_READY,
                action="Move the incomplete directory aside, then rerun Setup.",
            )
        model_cache.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".higgs-tts-3-4b-", dir=model_cache)).resolve()
        staging.relative_to(model_cache)
        try:
            download_environment = dict(setup_environment)
            docker_environment = ["--env", "HF_HUB_DISABLE_TELEMETRY=1"]
            if environment.get("HF_TOKEN"):
                download_environment["HF_TOKEN"] = environment["HF_TOKEN"]
                docker_environment.extend(["--env", "HF_TOKEN"])
            download_script = (
                "import os,sys; from huggingface_hub import snapshot_download; "
                "snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], "
                "local_dir=sys.argv[3], token=os.environ.get('HF_TOKEN') or None)"
            )
            _run(
                [
                    docker,
                    "run",
                    "--rm",
                    "--mount",
                    f"type=bind,source={staging},target=/download",
                    *docker_environment,
                    HIGGS_IMAGE_REFERENCE,
                    "/opt/omni/bin/python",
                    "-c",
                    download_script,
                    model_definition.model_repo,
                    model_definition.model_revision,
                    "/download",
                ],
                cwd=project_root,
                environment=download_environment,
                timeout=14400,
            )
            from .workers.prepare import write_asset_manifest

            write_asset_manifest(
                staging,
                repo=model_definition.model_repo,
                revision=model_definition.model_revision,
            )
            _verify_snapshot_manifest(staging / "asset-manifest.json", model_definition)
            if model_root.exists():
                model_root.rmdir()
            replace_path(staging, model_root)
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
        model_manifest = model_root / "asset-manifest.json"

    runtime_root = project_root / ".cache" / "runtimes" / "local--higgs-tts-3-4b"
    runtime_root.mkdir(parents=True, exist_ok=True)
    runtime_attestation = runtime_root / "container-runtime.json"
    atomic_write_json(
        runtime_attestation,
        {
            "schema_version": 1,
            "image_reference": HIGGS_IMAGE_REFERENCE,
            "image_id": image_id,
            "repo_digests": sorted(repo_digests),
            "docker_server_version": str(info.get("ServerVersion") or ""),
            "server_revision": HIGGS_SERVER_REVISION,
            "model_id": model_definition.model_repo,
            "model_revision": model_definition.model_revision,
            "model_asset_manifest_sha256": sha256_file(model_manifest),
        },
    )
    spec = RunnerSpec(
        backend_id=model_definition.backend_id,
        platform="docker",
        command=[sys.executable, "-m", "video_generator.workers.main", "--kind", "higgs-docker"],
        model_family="higgs-docker",
        requires_cuda=True,
        timeout_seconds=model_definition.timeout_seconds,
        startup_timeout_seconds=model_definition.startup_timeout_seconds,
        docker=DockerRuntimeSpec(
            image_reference=HIGGS_IMAGE_REFERENCE,
            image_id=image_id,
            server_revision=HIGGS_SERVER_REVISION,
            internal_port=8000,
            docker_server_version=str(info.get("ServerVersion") or ""),
        ),
        environment={
            "PYTHONPATH": str(project_root / "src"),
            "VIDEO_GENERATOR_MODEL_PATH": relative_path(model_root, project_root),
            "VIDEO_GENERATOR_RUNTIME_REVISION": sha256_file(runtime_attestation),
            "VIDEO_GENERATOR_MODEL_REVISION": model_definition.model_revision,
        },
        model_paths=[relative_path(model_root, project_root)],
        asset_manifests={
            relative_path(model_manifest, project_root): sha256_file(model_manifest),
        },
        asset_revisions={
            relative_path(model_manifest, project_root): model_definition.model_revision,
        },
        runtime_files={
            relative_path(runtime_attestation, project_root): sha256_file(runtime_attestation),
        },
        runtime_revision=sha256_file(runtime_attestation),
        model_revision=model_definition.model_revision,
        setup_source_revision=runner_setup_source_revision(model_definition.kind),
        license_name=BACKEND_DESCRIPTORS[model_definition.backend_id].license_name,
    )
    runner_path = project_root / ".cache" / "runners" / runner_slug(spec.backend_id) / "runner.json"
    atomic_write_json(runner_path, spec.model_dump(mode="json"))
    return ProbeItem(name=spec.backend_id, ready=True, detail=f"prepared: {runner_path}")


def _download_verified_url(
    *,
    url: str,
    destination: Path,
    expected_sha256: str,
    download: bool,
) -> Path:
    if destination.is_file() and sha256_file(destination).casefold() == expected_sha256.casefold():
        return destination
    if not download:
        raise VideoGeneratorError(
            f"required pinned X-Voice artifact is missing or changed: {destination}",
            kind=ErrorKind.NOT_READY,
            action="Rerun Setup without --no-download for local:x-voice.",
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".download",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "video-generator-setup/1"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        actual = sha256_file(temporary)
        if actual.casefold() != expected_sha256.casefold():
            raise VideoGeneratorError(
                f"SHA-256 mismatch for downloaded {destination.name}",
                kind=ErrorKind.NOT_READY,
                action=f"Expected {expected_sha256.lower()}, got {actual.lower()}.",
            )
        replace_path(temporary, destination)
    except VideoGeneratorError:
        raise
    except (OSError, TimeoutError) as exc:
        raise VideoGeneratorError(
            f"could not download pinned X-Voice artifact {destination.name}: {exc}",
            kind=ErrorKind.NOT_READY,
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _prepare_xvoice_micromamba(runtime: Path, *, download: bool) -> tuple[Path, Path]:
    tools_root = runtime / "tools"
    archive = tools_root / f"micromamba-{XVOICE_MICROMAMBA_VERSION}-win-64.tar.bz2"
    executable = tools_root / "micromamba.exe"
    if executable.is_file() and sha256_file(executable) == XVOICE_MICROMAMBA_EXE_SHA256:
        if archive.is_file() and sha256_file(archive) == XVOICE_MICROMAMBA_ARCHIVE_SHA256:
            return executable, archive
    _download_verified_url(
        url=XVOICE_MICROMAMBA_URL,
        destination=archive,
        expected_sha256=XVOICE_MICROMAMBA_ARCHIVE_SHA256,
        download=download,
    )
    tools_root.mkdir(parents=True, exist_ok=True)
    temporary = executable.with_name(executable.name + ".tmp")
    try:
        with tarfile.open(archive, mode="r:bz2") as bundle:
            member = bundle.getmember("Library/bin/micromamba.exe")
            if not member.isfile():
                raise VideoGeneratorError(
                    "the pinned micromamba archive does not contain a regular Windows executable",
                    kind=ErrorKind.NOT_READY,
                )
            source = bundle.extractfile(member)
            if source is None:
                raise VideoGeneratorError(
                    "the pinned micromamba executable could not be extracted",
                    kind=ErrorKind.NOT_READY,
                )
            with source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output)
        actual = sha256_file(temporary)
        if actual != XVOICE_MICROMAMBA_EXE_SHA256:
            raise VideoGeneratorError(
                "the extracted micromamba executable failed SHA-256 verification",
                kind=ErrorKind.NOT_READY,
            )
        replace_path(temporary, executable)
    finally:
        temporary.unlink(missing_ok=True)
    return executable, archive


def _validated_xvoice_runtime_path(
    project_root: Path,
    runtime: Path,
    candidate: Path,
) -> Path:
    project = project_root.resolve()
    managed_root_path = project / ".cache" / "runtimes"
    managed_root = managed_root_path.resolve()
    expected_runtime = managed_root / runner_slug("local:x-voice")
    expected_candidate = expected_runtime / candidate.name
    if (
        managed_root != managed_root_path
        or runtime.resolve() != expected_runtime
        or candidate.parent.resolve() != expected_runtime
        or candidate.resolve() != expected_candidate
    ):
        raise VideoGeneratorError(
            "X-Voice Setup refused a redirected managed runtime path",
            kind=ErrorKind.NOT_READY,
            action="Remove cache junctions or links and rerun Setup inside the project runtime cache.",
        )
    return candidate


def _xvoice_active_environment(
    project_root: Path,
    candidates: Sequence[Path],
) -> tuple[Path | None, bool]:
    manifest = (
        project_root
        / ".cache"
        / "runners"
        / runner_slug("local:x-voice")
        / "runner.json"
    )
    command_python: Path | None = None
    manifest_valid = False
    try:
        data = read_json(manifest)
        command = data.get("command")
        if (
            data.get("backend_id") == "local:x-voice"
            and isinstance(command, list)
            and command
            and isinstance(command[0], str)
        ):
            value = Path(command[0])
            if value.is_absolute():
                command_python = value.resolve()
                manifest_valid = True
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    if command_python is not None:
        for candidate in candidates:
            python = candidate / "python.exe"
            if python.is_file() and python.resolve() == command_python:
                return candidate, True
    if manifest_valid:
        return None, False
    return None, not any(candidate.exists() for candidate in candidates)


def _xvoice_environment_target(
    project_root: Path,
    legacy_prefix: Path,
    slot_a: Path,
    slot_b: Path,
) -> tuple[Path | None, Path]:
    active_prefix, identity_known = _xvoice_active_environment(
        project_root,
        (legacy_prefix, slot_a, slot_b),
    )
    if active_prefix == slot_a:
        return active_prefix, slot_b
    if active_prefix in {legacy_prefix, slot_b}:
        return active_prefix, slot_a
    if identity_known:
        return None, slot_a
    for candidate in (slot_a, slot_b):
        if not (candidate / "python.exe").is_file():
            return None, candidate
    raise VideoGeneratorError(
        "X-Voice Setup could not identify a safe inactive environment slot",
        kind=ErrorKind.NOT_READY,
        action=(
            "Restore the last valid X-Voice runner manifest or remove one known-inactive "
            "conda slot, then rerun Setup."
        ),
    )


def _prepare_xvoice_environment(
    project_root: Path,
    runtime: Path,
    *,
    environment: Mapping[str, str],
    download: bool,
) -> tuple[Path, Path, Path, Path, Path]:
    legacy_prefix = _validated_xvoice_runtime_path(
        project_root, runtime, runtime / "conda"
    )
    slot_a = _validated_xvoice_runtime_path(
        project_root, runtime, runtime / "conda-a"
    )
    slot_b = _validated_xvoice_runtime_path(
        project_root, runtime, runtime / "conda-b"
    )
    active_prefix, conda_prefix = _xvoice_environment_target(
        project_root,
        legacy_prefix,
        slot_a,
        slot_b,
    )
    slot_name = "a" if conda_prefix == slot_a else "b"
    python = conda_prefix / "python.exe"
    uv = _find_uv()
    if not uv:
        raise VideoGeneratorError("uv is required for X-Voice Setup", kind=ErrorKind.NOT_READY)
    micromamba, micromamba_archive = _prepare_xvoice_micromamba(runtime, download=download)
    conda_source = _requirements_path("xvoice-conda-win-64.lock")
    conda_lock = _validated_xvoice_runtime_path(
        project_root,
        runtime,
        runtime / f"conda-{slot_name}.explicit.lock",
    )
    atomic_write_text(conda_lock, conda_source.read_text(encoding="utf-8"))
    conda_root = runtime / "micromamba-root"
    setup_environment = _setup_tool_environment(environment, project_root)
    setup_environment.update(
        {
            "MAMBA_ROOT_PREFIX": str(conda_root),
            "MAMBA_NO_BANNER": "1",
        }
    )
    requirements = _requirements_path("xvoice.in")
    reviewed_requirements_lock = requirements.with_suffix(".lock")
    requirements_lock = _validated_xvoice_runtime_path(
        project_root,
        runtime,
        runtime / f"conda-{slot_name}.requirements.lock",
    )
    source_marker = _validated_xvoice_runtime_path(
        project_root,
        runtime,
        runtime / f"conda-{slot_name}.source.sha256",
    )
    source_revision = runner_setup_source_revision("xvoice")
    if not reviewed_requirements_lock.is_file():
        raise VideoGeneratorError(
            f"reviewed Windows dependency lock is missing: {reviewed_requirements_lock}",
            kind=ErrorKind.NOT_READY,
        )
    if (
        not requirements_lock.is_file()
        or sha256_file(requirements_lock) != sha256_file(reviewed_requirements_lock)
    ):
        _copy_verified_file(
            reviewed_requirements_lock,
            requirements_lock,
            sha256_file(reviewed_requirements_lock),
        )
    atomic_write_text(source_marker, "preparing\n")
    if conda_prefix.exists():
        shutil.rmtree(
            _validated_xvoice_runtime_path(project_root, runtime, conda_prefix)
        )
    if active_prefix in {slot_a, slot_b} and legacy_prefix.exists():
        shutil.rmtree(
            _validated_xvoice_runtime_path(project_root, runtime, legacy_prefix)
        )
    try:
        _run(
            [
                str(micromamba),
                "create",
                "--yes",
                "--no-rc",
                "--root-prefix",
                str(conda_root),
                "--prefix",
                str(conda_prefix),
                "--file",
                str(conda_lock),
            ],
            cwd=project_root,
            environment=setup_environment,
            timeout=3600,
        )
        if not python.is_file():
            raise VideoGeneratorError(
                "the pinned X-Voice Conda environment did not produce python.exe",
                kind=ErrorKind.NOT_READY,
            )
        uv_version = _parse_uv_version(
            _run([uv, "--version"], cwd=project_root, environment=setup_environment, timeout=30)
        )
        if uv_version is None or uv_version < MINIMUM_UV_TORCH_BACKEND_VERSION:
            raise VideoGeneratorError(
                "X-Voice CUDA Setup requires uv 0.6.9 or newer",
                kind=ErrorKind.NOT_READY,
            )
        _run(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(python),
                "--torch-backend",
                "cu128",
                "--require-hashes",
                "-r",
                str(requirements_lock),
            ],
            cwd=project_root,
            environment=setup_environment,
            timeout=3600,
        )
        pynini_version = _run(
            [str(python), "-c", "import pynini; print(pynini.__version__)"],
            cwd=project_root,
            environment=setup_environment,
            timeout=60,
        ).strip()
        if pynini_version != "2.1.7":
            raise VideoGeneratorError(
                f"X-Voice Setup expected Pynini 2.1.7, got {pynini_version or 'no version'}",
                kind=ErrorKind.NOT_READY,
            )
        atomic_write_text(source_marker, source_revision + "\n")
    except BaseException:
        if conda_prefix.exists():
            shutil.rmtree(
                _validated_xvoice_runtime_path(project_root, runtime, conda_prefix)
            )
        conda_lock.unlink(missing_ok=True)
        requirements_lock.unlink(missing_ok=True)
        source_marker.unlink(missing_ok=True)
        raise
    return python, conda_lock, requirements_lock, source_marker, micromamba_archive


def _prepare_xvoice_source(
    project_root: Path,
    runtime: Path,
    *,
    download: bool,
) -> tuple[Path, Path]:
    git = shutil.which("git")
    if not git:
        raise VideoGeneratorError("git is required for X-Voice Setup", kind=ErrorKind.NOT_READY)
    source = runtime / "source"
    if not source.exists():
        if not download:
            raise VideoGeneratorError("the pinned X-Voice source checkout is missing", kind=ErrorKind.NOT_READY)
        staging = Path(tempfile.mkdtemp(prefix=".xvoice-source.", dir=runtime))
        try:
            _run([git, "init", str(staging)], cwd=project_root, timeout=60)
            _run(
                [git, "remote", "add", "origin", XVOICE_SOURCE_REPOSITORY],
                cwd=staging,
                timeout=60,
            )
            _run(
                [git, "fetch", "--depth", "1", "origin", XVOICE_SOURCE_REVISION],
                cwd=staging,
                timeout=600,
            )
            _run([git, "checkout", "--detach", "FETCH_HEAD"], cwd=staging, timeout=120)
            replace_path(staging, source)
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
    if not (source / ".git").exists():
        raise VideoGeneratorError(
            "the X-Voice runtime is not a verifiable Git checkout",
            kind=ErrorKind.NOT_READY,
        )
    commit = _run([git, "rev-parse", "HEAD"], cwd=source, timeout=60).strip()
    if commit.casefold() != XVOICE_SOURCE_REVISION.casefold():
        raise VideoGeneratorError(
            f"X-Voice checkout is {commit}, expected {XVOICE_SOURCE_REVISION}",
            kind=ErrorKind.NOT_READY,
            action="Remove the managed X-Voice runtime and rerun Setup.",
        )
    status = [
        line
        for line in _run(
            [git, "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=source,
            timeout=60,
        ).splitlines()
        if line.strip() and line.strip() != "?? runtime-source.asset-manifest.json"
    ]
    if status:
        raise VideoGeneratorError(
            "the pinned X-Voice checkout contains modified or unexpected files",
            kind=ErrorKind.NOT_READY,
            action=f"Inspect the managed runtime; first unexpected entry: {status[0]}",
        )
    tracked = [
        line.strip().replace("\\", "/")
        for line in _run([git, "ls-files"], cwd=source, timeout=60).splitlines()
        if line.strip() and (source / line.strip()).is_file()
    ]
    manifest = source / "runtime-source.asset-manifest.json"
    atomic_write_json(
        manifest,
        {
            "schema_version": 1,
            "root": ".",
            "source": XVOICE_SOURCE_REPOSITORY,
            "source_revision": commit,
            "revision": commit,
            "exact_file_suffixes": [".py", ".pyi", ".pyd", ".dll", ".so"],
            "exact_exclude_roots": [".git", "__pycache__"],
            "files": [
                {
                    "path": value,
                    "size": (source / value).stat().st_size,
                    "sha256": sha256_file(source / value),
                }
                for value in tracked
            ],
        },
    )
    RunnerManager._verify_asset_manifest(manifest, expected_revision=None)
    return source, manifest


def _manifest_file_set_sha256(metadata: Mapping[str, object]) -> str:
    files = metadata.get("files")
    if not isinstance(files, list):
        raise VideoGeneratorError(
            "asset manifest has no file set",
            kind=ErrorKind.NOT_READY,
        )
    values = []
    for item in files:
        if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
            raise VideoGeneratorError(
                "asset manifest contains a malformed file entry",
                kind=ErrorKind.NOT_READY,
            )
        values.append((str(item["path"]), str(item["sha256"]).casefold()))
    return hashlib.sha256(
        json.dumps(sorted(values), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _prepare_xvoice_espeak(runtime: Path) -> tuple[Path, Path, str]:
    destination = runtime / "espeak"
    manifest = destination / "asset-manifest.json"
    if manifest.is_file():
        metadata = read_json(manifest)
        revision = str(metadata.get("revision") or "")
        RunnerManager._verify_asset_manifest(
            manifest,
            expected_revision=XVOICE_ESPEAK_VERSION,
        )
        if revision != XVOICE_ESPEAK_VERSION:
            raise VideoGeneratorError(
                f"managed X-Voice eSpeak version is {revision or 'unknown'}, "
                f"expected {XVOICE_ESPEAK_VERSION}",
                kind=ErrorKind.NOT_READY,
            )
        if _manifest_file_set_sha256(metadata) != XVOICE_ESPEAK_BUNDLE_SHA256:
            raise VideoGeneratorError(
                "managed X-Voice eSpeak bundle is not the reviewed Windows build",
                kind=ErrorKind.NOT_READY,
            )
        return destination / "libespeak-ng.dll", manifest, revision
    if destination.exists():
        raise VideoGeneratorError(
            "the managed X-Voice eSpeak runtime is incomplete",
            kind=ErrorKind.NOT_READY,
            action="Remove the managed X-Voice runtime and rerun Setup.",
        )
    executable_value = shutil.which("espeak-ng")
    if not executable_value:
        raise VideoGeneratorError(
            "X-Voice requires a native Windows eSpeak NG installation",
            kind=ErrorKind.NOT_READY,
            action="Install eSpeak NG for Windows and ensure espeak-ng.exe is on PATH.",
        )
    executable = Path(executable_value).resolve()
    source_root = executable.parent
    source_dll = source_root / "libespeak-ng.dll"
    source_data = source_root / "espeak-ng-data"
    if not source_dll.is_file() or not source_data.is_dir():
        raise VideoGeneratorError(
            "the Windows eSpeak NG installation is missing its DLL or language data",
            kind=ErrorKind.NOT_READY,
        )
    if any(path.is_symlink() for path in source_data.rglob("*")):
        raise VideoGeneratorError(
            "the eSpeak NG data tree contains unsupported symbolic links",
            kind=ErrorKind.NOT_READY,
        )
    version_output = _run([str(executable), "--version"], cwd=runtime, timeout=30)
    match = re.search(r"\b(\d+\.\d+(?:\.\d+)?)\b", version_output)
    revision = match.group(1) if match else ""
    if revision != XVOICE_ESPEAK_VERSION:
        raise VideoGeneratorError(
            f"X-Voice requires the reviewed eSpeak NG {XVOICE_ESPEAK_VERSION} Windows build",
            kind=ErrorKind.NOT_READY,
        )
    if sha256_file(executable) != XVOICE_ESPEAK_EXE_SHA256:
        raise VideoGeneratorError(
            "the eSpeak NG executable does not match the reviewed Windows build",
            kind=ErrorKind.NOT_READY,
        )
    if sha256_file(source_dll) != XVOICE_ESPEAK_DLL_SHA256:
        raise VideoGeneratorError(
            "the eSpeak NG library does not match the reviewed Windows build",
            kind=ErrorKind.NOT_READY,
        )
    staging = Path(tempfile.mkdtemp(prefix=".espeak.", dir=runtime))
    try:
        shutil.copy2(source_dll, staging / source_dll.name)
        shutil.copytree(source_data, staging / source_data.name)
        files = [
            {
                "path": path.relative_to(staging).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(item for item in staging.rglob("*") if item.is_file())
        ]
        if _manifest_file_set_sha256({"files": files}) != XVOICE_ESPEAK_BUNDLE_SHA256:
            raise VideoGeneratorError(
                "the eSpeak NG data bundle does not match the reviewed Windows build",
                kind=ErrorKind.NOT_READY,
            )
        atomic_write_json(
            staging / "asset-manifest.json",
            {
                "schema_version": 1,
                "root": ".",
                "source": str(source_root),
                "revision": revision,
                "exact_file_suffixes": [".dll"],
                "exact_exclude_roots": [],
                "files": files,
            },
        )
        RunnerManager._verify_asset_manifest(
            staging / "asset-manifest.json", expected_revision=revision
        )
        replace_path(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return destination / "libespeak-ng.dll", destination / "asset-manifest.json", revision


def _prepare_xvoice_support(runtime: Path, *, download: bool) -> tuple[Path, Path, str]:
    destination = runtime / "support"
    manifest = destination / "asset-manifest.json"
    revision = f"fasttext-lid.176.ftz+nltk-data-{XVOICE_NLTK_DATA_REVISION}"
    if manifest.is_file():
        RunnerManager._verify_asset_manifest(manifest, expected_revision=revision)
        return destination, manifest, revision
    if destination.exists():
        raise VideoGeneratorError(
            "the managed X-Voice support-data runtime is incomplete",
            kind=ErrorKind.NOT_READY,
            action="Remove the managed X-Voice runtime and rerun Setup.",
        )
    staging = Path(tempfile.mkdtemp(prefix=".xvoice-support.", dir=runtime))
    try:
        language_id = _download_verified_url(
            url=XVOICE_FASTTEXT_LID_URL,
            destination=staging / "home" / ".cache" / "lid.176.ftz",
            expected_sha256=XVOICE_FASTTEXT_LID_SHA256,
            download=download,
        )
        cmudict = _download_verified_url(
            url=XVOICE_CMUDICT_URL,
            destination=staging / "nltk_data" / "corpora" / "cmudict.zip",
            expected_sha256=XVOICE_CMUDICT_SHA256,
            download=download,
        )
        atomic_write_json(
            staging / "asset-manifest.json",
            {
                "schema_version": 1,
                "root": ".",
                "revision": revision,
                "sources": {
                    "fasttext_language_id": XVOICE_FASTTEXT_LID_URL,
                    "nltk_data_revision": XVOICE_NLTK_DATA_REVISION,
                },
                "files": [
                    {
                        "path": language_id.relative_to(staging).as_posix(),
                        "size": language_id.stat().st_size,
                        "sha256": sha256_file(language_id),
                    },
                    {
                        "path": cmudict.relative_to(staging).as_posix(),
                        "size": cmudict.stat().st_size,
                        "sha256": sha256_file(cmudict),
                    },
                ],
            },
        )
        RunnerManager._verify_asset_manifest(
            staging / "asset-manifest.json", expected_revision=revision
        )
        replace_path(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return destination, destination / "asset-manifest.json", revision


def prepare_xvoice_backend(
    project_root: Path,
    *,
    environment: Mapping[str, str],
    download: bool,
) -> ProbeItem:
    if os.name != "nt":
        raise VideoGeneratorError(
            "the X-Voice adapter is intentionally supported only on native Windows",
            kind=ErrorKind.UNSUPPORTED,
        )
    definition = LOCAL_DEFINITIONS["local:x-voice"]
    runtime = project_root / ".cache" / "runtimes" / runner_slug(definition.backend_id)
    runtime.mkdir(parents=True, exist_ok=True)
    python, conda_lock, requirements_lock, source_marker, micromamba_archive = (
        _prepare_xvoice_environment(
            project_root,
            runtime,
            environment=environment,
            download=download,
        )
    )
    source, source_manifest = _prepare_xvoice_source(
        project_root,
        runtime,
        download=download,
    )
    espeak_dll, espeak_manifest, espeak_revision = _prepare_xvoice_espeak(runtime)
    support, support_manifest, support_revision = _prepare_xvoice_support(
        runtime,
        download=download,
    )
    model_root = project_root / ".cache" / "models" / definition.model_subdir
    if download:
        _download_snapshot(
            project_root=project_root,
            python_command=[str(python)],
            definition=definition,
            destination=model_root,
            environment=environment,
        )
    model_manifest = model_root / "asset-manifest.json"
    if not model_manifest.is_file():
        raise VideoGeneratorError("the X-Voice Stage1 snapshot is missing", kind=ErrorKind.NOT_READY)
    supporting = definition.supporting_models[0]
    vocoder_root = project_root / ".cache" / "models" / supporting.model_subdir
    supporting_definition = LocalDefinition(
        backend_id=definition.backend_id,
        kind=definition.kind,
        platform=definition.platform,
        python_version=definition.python_version,
        requirements_name=definition.requirements_name,
        model_repo=supporting.model_repo,
        model_revision=supporting.model_revision,
        model_subdir=supporting.model_subdir,
        allow_patterns=supporting.allow_patterns,
    )
    if download:
        _download_snapshot(
            project_root=project_root,
            python_command=[str(python)],
            definition=supporting_definition,
            destination=vocoder_root,
            environment=environment,
        )
    vocoder_manifest = vocoder_root / "asset-manifest.json"
    if not vocoder_manifest.is_file():
        raise VideoGeneratorError("the pinned X-Voice Vocos snapshot is missing", kind=ErrorKind.NOT_READY)
    matplotlib_cache = runtime / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    micromamba = runtime / "tools" / "micromamba.exe"
    runtime_revision = hashlib.sha256(
        (
            XVOICE_SOURCE_REVISION
            + "\0"
            + sha256_file(conda_lock)
            + "\0"
            + sha256_file(requirements_lock)
        ).encode("utf-8")
    ).hexdigest()
    environment_values = {
        "PYTHONPATH": os.pathsep.join([str(project_root / "src"), str(source / "src")]),
        "VIDEO_GENERATOR_MODEL_PATH": relative_path(model_root, project_root),
        "VIDEO_GENERATOR_XVOICE_VOCODER_PATH": relative_path(vocoder_root, project_root),
        "VIDEO_GENERATOR_XVOICE_SOURCE_PATH": relative_path(source, project_root),
        "VIDEO_GENERATOR_RUNTIME_REVISION": runtime_revision,
        "VIDEO_GENERATOR_MODEL_REVISION": definition.model_revision,
        "PHONEMIZER_ESPEAK_LIBRARY": str(espeak_dll),
        "ESPEAK_DATA_PATH": str(espeak_dll.parent / "espeak-ng-data"),
        "USERPROFILE": str(support / "home"),
        "HOME": str(support / "home"),
        "NLTK_DATA": str(support / "nltk_data"),
        "MPLCONFIGDIR": str(matplotlib_cache),
        "WANDB_MODE": "disabled",
        "WANDB_SILENT": "true",
    }
    manifests = {
        model_manifest: definition.model_revision,
        vocoder_manifest: supporting.model_revision,
        source_manifest: XVOICE_SOURCE_REVISION,
        espeak_manifest: espeak_revision,
        support_manifest: support_revision,
    }
    spec = RunnerSpec(
        backend_id=definition.backend_id,
        platform="native",
        command=[str(python), "-m", "video_generator.workers.main", "--kind", "xvoice"],
        model_family="xvoice",
        timeout_seconds=definition.timeout_seconds,
        startup_timeout_seconds=definition.startup_timeout_seconds,
        environment=environment_values,
        model_paths=[
            relative_path(model_root, project_root),
            relative_path(vocoder_root, project_root),
            relative_path(source, project_root),
        ],
        asset_manifests={
            relative_path(path, project_root): sha256_file(path) for path in manifests
        },
        asset_revisions={
            relative_path(path, project_root): revision for path, revision in manifests.items()
        },
        runtime_files={
            relative_path(path, project_root): sha256_file(path)
            for path in (
                conda_lock,
                requirements_lock,
                source_marker,
                micromamba,
                micromamba_archive,
            )
        },
        runtime_revision=runtime_revision,
        model_revision=definition.model_revision,
        setup_source_revision=runner_setup_source_revision("xvoice"),
        license_name=BACKEND_DESCRIPTORS[definition.backend_id].license_name,
        metadata={
            "source_repository": XVOICE_SOURCE_REPOSITORY,
            "source_revision": XVOICE_SOURCE_REVISION,
            "vocoder_revision": supporting.model_revision,
            "micromamba_version": XVOICE_MICROMAMBA_VERSION,
            "micromamba_archive_sha256": XVOICE_MICROMAMBA_ARCHIVE_SHA256,
            "pynini_version": "2.1.7",
            "supported_languages": ["en", "fi"],
            "weights_usage": "noncommercial",
        },
    )
    path = project_root / ".cache" / "runners" / runner_slug(spec.backend_id) / "runner.json"
    atomic_write_json(path, spec.model_dump(mode="json"))
    return ProbeItem(name=spec.backend_id, ready=True, detail=f"prepared: {path}")


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
        if backend_id == "local:x-voice":
            results.append(
                prepare_xvoice_backend(
                    project_root,
                    environment=environment,
                    download=download,
                )
            )
            continue
        if backend_id == "local:higgs-tts-3-4b":
            results.append(
                prepare_higgs_docker_backend(
                    project_root,
                    environment=environment,
                    download=download,
                )
            )
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
