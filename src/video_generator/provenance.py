from __future__ import annotations

import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import active_backend_ids
from .contracts import ResolvedRunConfig, VideoStyle
from .errors import CheckpointError, ErrorKind
from .profiles import BACKEND_DESCRIPTORS
from .remotion_renderer import _remotion_subprocess_environment
from .runners import runner_slug
from .util import hash_value, read_json, sha256_file


def _tool_version(
    executable: str | None,
    argument: str = "-version",
    *,
    environment: Mapping[str, str] | None = None,
) -> str:
    if not executable:
        return "<missing>"
    try:
        completed = subprocess.run(
            [executable, argument],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
            env=dict(environment) if environment is not None else None,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "<unavailable>"
    first_line = (completed.stdout or completed.stderr).splitlines()
    return first_line[0].strip() if first_line else f"exit-{completed.returncode}"


def _tree_attestation(
    root: Path,
    *,
    excluded_names: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    resolved_root = root.resolve()
    records: list[dict[str, Any]] = []
    files: list[tuple[Path, str, int]] = []
    total_bytes = 0
    if not resolved_root.is_dir():
        return {"file_count": 0, "total_bytes": 0, "tree_sha256": hash_value(records)}
    for path in sorted(resolved_root.rglob("*")):
        relative = path.relative_to(resolved_root)
        if any(part in excluded_names for part in relative.parts):
            continue
        if path.is_symlink():
            target = path.resolve()
            try:
                target_relative = target.relative_to(resolved_root).as_posix()
            except ValueError as exc:
                raise CheckpointError(
                    f"Remotion runtime symlink escapes its managed root: {path}",
                    kind=ErrorKind.UNSUPPORTED,
                    action="Reinstall the Remotion runtime from the pinned lockfile.",
                ) from exc
            records.append(
                {
                    "path": relative.as_posix(),
                    "type": "symlink",
                    "target": target_relative,
                }
            )
            continue
        if not path.is_file():
            continue
        size = path.stat().st_size
        total_bytes += size
        files.append((path, relative.as_posix(), size))

    def file_record(item: tuple[Path, str, int]) -> dict[str, Any]:
        path, relative, size = item
        return {
            "path": relative,
            "type": "file",
            "size": size,
            "sha256": sha256_file(path),
        }

    with ThreadPoolExecutor(max_workers=16) as pool:
        records.extend(pool.map(file_record, files))
    records.sort(key=lambda record: record["path"])
    return {
        "file_count": sum(record["type"] == "file" for record in records),
        "total_bytes": total_bytes,
        "tree_sha256": hash_value(records),
    }


def build_runtime_snapshot(config: ResolvedRunConfig) -> dict[str, Any]:
    project_root = Path(config.project_root).resolve()
    tool_environment = _remotion_subprocess_environment()
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
            "version": _tool_version(
                shutil.which("ffmpeg"), environment=tool_environment
            ),
        },
        "ffprobe": {
            "path": shutil.which("ffprobe") or "<missing>",
            "version": _tool_version(
                shutil.which("ffprobe"), environment=tool_environment
            ),
        },
    }
    if config.video_style is VideoStyle.REMOTION_EXPLAINER:
        remotion_root = project_root / "remotion"
        remotion_files = [
            *remotion_root.glob("package*.json"),
            remotion_root / "tsconfig.json",
            *remotion_root.joinpath("src").rglob("*"),
            *remotion_root.joinpath("scripts").rglob("*"),
        ]
        remotion_hashes = {
            path.relative_to(project_root).as_posix(): sha256_file(path)
            for path in sorted(set(remotion_files))
            if path.is_file()
        }
        media_library = project_root / "media-library"
        media_library_hashes = {
            path.relative_to(project_root).as_posix(): sha256_file(path)
            for path in sorted(media_library.rglob("*"))
            if path.is_file()
        }
        package_paths = {
            "remotion": remotion_root / "node_modules" / "remotion" / "package.json",
            "@remotion/bundler": remotion_root
            / "node_modules"
            / "@remotion"
            / "bundler"
            / "package.json",
            "@remotion/media": remotion_root
            / "node_modules"
            / "@remotion"
            / "media"
            / "package.json",
            "@remotion/renderer": remotion_root
            / "node_modules"
            / "@remotion"
            / "renderer"
            / "package.json",
        }
        package_specs = {
            name: read_json(path) if path.is_file() else {}
            for name, path in package_paths.items()
        }
        browser_root = (
            remotion_root / "node_modules" / ".remotion" / "chrome-headless-shell"
        )
        browser_version_file = browser_root / "VERSION"
        try:
            browser_version = browser_version_file.read_text(encoding="utf-8").strip()
        except OSError:
            browser_version = "<missing>"
        browser_executable = next(
            (
                path
                for pattern in (
                    "chrome-headless-shell.exe",
                    "chrome-headless-shell",
                    "headless_shell",
                )
                for path in browser_root.rglob(pattern)
                if path.is_file()
            ),
            None,
        )
        node = shutil.which("node")
        npm = shutil.which("npm")
        node_path = Path(node).resolve() if node else None
        installed_node_modules = _tree_attestation(
            remotion_root / "node_modules",
            excluded_names=frozenset({".remotion"}),
        )
        browser_files = _tree_attestation(browser_root)
        node_sha256 = (
            sha256_file(node_path) if node_path is not None and node_path.is_file() else "<missing>"
        )
        bundle_runtime_hash = hash_value(
            {
                "node_sha256": node_sha256,
                "installed_node_modules_tree_sha256": installed_node_modules[
                    "tree_sha256"
                ],
            }
        )
        snapshot["remotion"] = {
            "source_files": remotion_hashes,
            "source_hash": hash_value(remotion_hashes),
            "media_library_files": media_library_hashes,
            "media_library_hash": hash_value(media_library_hashes),
            "packages": {
                name: {
                    "version": str(package_specs[name].get("version") or "<missing>"),
                    "manifest_sha256": sha256_file(path) if path.is_file() else "<missing>",
                }
                for name, path in package_paths.items()
            },
            "node": {
                "path": node or "<missing>",
                "version": _tool_version(
                    node, "--version", environment=tool_environment
                ),
                "sha256": node_sha256,
            },
            "npm": {
                "path": npm or "<missing>",
                "version": _tool_version(
                    npm, "--version", environment=tool_environment
                ),
            },
            "browser": {
                "version": browser_version,
                "path": str(browser_executable) if browser_executable else "<missing>",
                "size": browser_executable.stat().st_size if browser_executable else 0,
                "sha256": sha256_file(browser_executable) if browser_executable else "<missing>",
                "files": browser_files,
            },
            "installed_node_modules": installed_node_modules,
            "bundle_runtime_hash": bundle_runtime_hash,
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
            "the orchestrator, media tools, Remotion renderer, local runner manifests, or private voice inputs changed since Run creation",
            kind=ErrorKind.UNSUPPORTED,
            action="Restore the frozen runtime inputs or use rerun --from the earliest affected stage.",
            details={
                "expected_snapshot_hash": expected.get("snapshot_hash") if isinstance(expected, dict) else None,
                "current_snapshot_hash": current.get("snapshot_hash"),
            },
        )
