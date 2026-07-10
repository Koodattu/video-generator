from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from .config import active_backend_ids
from .contracts import ResolvedRunConfig
from .errors import CheckpointError, ErrorKind
from .profiles import BACKEND_DESCRIPTORS
from .runners import runner_slug
from .util import hash_value, read_json, sha256_file


def _tool_version(executable: str | None) -> str:
    if not executable:
        return "<missing>"
    try:
        completed = subprocess.run(
            [executable, "-version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "<unavailable>"
    first_line = (completed.stdout or completed.stderr).splitlines()
    return first_line[0].strip() if first_line else f"exit-{completed.returncode}"


def build_runtime_snapshot(config: ResolvedRunConfig) -> dict[str, Any]:
    project_root = Path(config.project_root).resolve()
    package_root = Path(__file__).resolve().parent
    code_files = sorted(
        [*package_root.rglob("*.py"), *(package_root / "assets" / "runners").glob("*.in")]
    )
    for candidate in (project_root / "pyproject.toml", project_root / "uv.lock"):
        if candidate.is_file():
            code_files.append(candidate)
    code_hashes: dict[str, str] = {}
    for path in sorted(set(code_files)):
        resolved = path.resolve()
        try:
            name = resolved.relative_to(project_root).as_posix()
        except ValueError:
            name = "installed-package/" + resolved.relative_to(package_root).as_posix()
        code_hashes[name] = sha256_file(resolved)

    runner_manifests: dict[str, Any] = {}
    for backend_id in sorted(active_backend_ids(config)):
        if BACKEND_DESCRIPTORS[backend_id].provider != "local":
            continue
        path = project_root / ".cache" / "runners" / runner_slug(backend_id) / "runner.json"
        runner_manifests[backend_id] = (
            {"sha256": sha256_file(path), "spec": read_json(path)}
            if path.is_file()
            else "<missing>"
        )

    private_inputs: dict[str, str] = {}
    speech_descriptor = BACKEND_DESCRIPTORS[config.task_bindings["narration_synthesis"]]
    if speech_descriptor.provider == "local" and speech_descriptor.supports_voice_cloning:
        for value in (config.voice.reference_audio, config.voice.reference_transcript):
            if value:
                path = (project_root / value).resolve()
                private_inputs[value] = sha256_file(path) if path.is_file() else "<missing>"

    snapshot = {
        "schema_version": 1,
        "code_files": code_hashes,
        "code_hash": hash_value(code_hashes),
        "runner_manifests": runner_manifests,
        "private_inputs": private_inputs,
        "ffmpeg": {
            "path": shutil.which("ffmpeg") or "<missing>",
            "version": _tool_version(shutil.which("ffmpeg")),
        },
        "ffprobe": {
            "path": shutil.which("ffprobe") or "<missing>",
            "version": _tool_version(shutil.which("ffprobe")),
        },
    }
    snapshot["snapshot_hash"] = hash_value(snapshot)
    return snapshot


def verify_runtime_snapshot(config: ResolvedRunConfig, frozen_assets: dict[str, Any]) -> None:
    expected = frozen_assets.get("runtime_snapshot")
    if expected is None:
        return
    current = build_runtime_snapshot(config)
    if current != expected:
        raise CheckpointError(
            "the orchestrator, media tools, local runner manifests, or private voice inputs changed since Run creation",
            kind=ErrorKind.UNSUPPORTED,
            action="Restore the frozen runtime inputs or use rerun --from the earliest affected stage.",
            details={
                "expected_snapshot_hash": expected.get("snapshot_hash") if isinstance(expected, dict) else None,
                "current_snapshot_hash": current.get("snapshot_hash"),
            },
        )
