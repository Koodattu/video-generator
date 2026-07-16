from __future__ import annotations

import json
import math
import os
import shutil
import struct
import subprocess
import wave
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from .contracts import (
    MediaReference,
    OutputLanguage,
    ProbeItem,
    RemotionAssetBundle,
    RemotionEditPlan,
    RemotionMotionPreset,
    RemotionTemplate,
)
from .errors import ErrorKind, MediaError
from .net import validate_public_http_url
from .util import atomic_write_json, relative_path, sha256_file


REMOTION_VERSION = "4.0.489"
REMOTION_BROWSER_VERSION = "149.0.7790.0"

_REMOTION_ENVIRONMENT_NAMES = {
    "ALL_PROXY",
    "APPDATA",
    "COMSPEC",
    "HOME",
    "HOMEDRIVE",
    "HOMEPATH",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "LOCALAPPDATA",
    "NODE_EXTRA_CA_CERTS",
    "NO_PROXY",
    "NPM_CONFIG_CACHE",
    "PATH",
    "PATHEXT",
    "PROGRAMDATA",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "WINDIR",
}
_REMOTION_ENVIRONMENT_NAMES_CASEFOLD = {
    name.casefold() for name in _REMOTION_ENVIRONMENT_NAMES
}


def _remotion_subprocess_environment(
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    source = os.environ if environment is None else environment
    return {
        name: value
        for name, value in source.items()
        if name.casefold() in _REMOTION_ENVIRONMENT_NAMES_CASEFOLD
    }

REMOTION_LABELS: dict[OutputLanguage, dict[str, str]] = {
    OutputLanguage.ENGLISH: {
        "payAttention": "PAY ATTENTION",
        "source": "THE SOURCE",
        "citedSource": "cited source",
        "takeaway": "THE TAKEAWAY",
        "before": "Before",
        "after": "After",
    },
    OutputLanguage.FINNISH: {
        "payAttention": "HUOMIO",
        "source": "LÄHDE",
        "citedSource": "viitattu lähde",
        "takeaway": "TÄRKEIN AJATUS",
        "before": "Ennen",
        "after": "Jälkeen",
    },
}


def remotion_root(project_root: Path) -> Path:
    return project_root.resolve() / "remotion"


def _ensure_remotion_project(project_root: Path) -> Path:
    root = remotion_root(project_root)
    if (root / "package.json").is_file():
        return root
    packaged = Path(__file__).resolve().parent / "remotion"
    if not (packaged / "package.json").is_file():
        return root
    if root.exists() and any(root.iterdir()):
        raise OSError(
            f"cannot install the bundled Remotion scaffold into nonempty directory: {root}"
        )
    root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(packaged, root, dirs_exist_ok=True)
    return root


def remotion_motion_for_template(template: RemotionTemplate) -> RemotionMotionPreset:
    return {
        RemotionTemplate.KINETIC_HOOK: RemotionMotionPreset.PUNCH_IN,
        RemotionTemplate.HEADLINE_ZOOM: RemotionMotionPreset.SLIDE_UP,
        RemotionTemplate.SOURCE_SCREENSHOT: RemotionMotionPreset.PAN,
        RemotionTemplate.CODE_REVEAL: RemotionMotionPreset.TYPE_ON,
        RemotionTemplate.DIAGRAM_FLOW: RemotionMotionPreset.BUILD,
        RemotionTemplate.COMPARISON_SPLIT: RemotionMotionPreset.SLIDE_UP,
        RemotionTemplate.MEME_CUTAWAY: RemotionMotionPreset.PUNCH_IN,
        RemotionTemplate.CONCLUSION: RemotionMotionPreset.HOLD,
    }[template]


def probe_remotion_runtime(project_root: Path) -> list[ProbeItem]:
    root = remotion_root(project_root)
    node = shutil.which("node")
    checks = [
        ProbeItem(
            name="remotion_node",
            ready=node is not None,
            detail=node or "Node.js is not on PATH",
            action=None if node else "Install current Node.js LTS for Windows.",
        ),
        ProbeItem(
            name="remotion_lockfile",
            ready=(root / "package-lock.json").is_file(),
            detail=str(root / "package-lock.json"),
            action=None if (root / "package-lock.json").is_file() else "Run npm install in remotion/.",
        ),
    ]
    package_paths = {
        "remotion": root / "node_modules" / "remotion" / "package.json",
        "@remotion/bundler": root / "node_modules" / "@remotion" / "bundler" / "package.json",
        "@remotion/media": root / "node_modules" / "@remotion" / "media" / "package.json",
        "@remotion/renderer": root / "node_modules" / "@remotion" / "renderer" / "package.json",
    }
    versions: dict[str, str] = {}
    for name, package_path in package_paths.items():
        try:
            versions[name] = str(
                json.loads(package_path.read_text(encoding="utf-8")).get("version") or ""
            )
        except OSError:
            versions[name] = "<missing>"
        except (ValueError, TypeError):
            versions[name] = "<invalid>"
    dependencies_ready = all(
        version == REMOTION_VERSION for version in versions.values()
    )
    version_detail = ", ".join(f"{name} {version}" for name, version in versions.items())
    checks.append(
        ProbeItem(
            name="remotion_dependencies",
            ready=dependencies_ready,
            detail=version_detail,
            action=(
                None
                if dependencies_ready
                else "Run npm ci in remotion/; all Remotion packages must remain on 4.0.489."
            ),
        )
    )
    browser_root = root / "node_modules" / ".remotion" / "chrome-headless-shell"
    browser_version_file = browser_root / "VERSION"
    try:
        browser_version = browser_version_file.read_text(encoding="utf-8").strip()
    except OSError:
        browser_version = ""
    browser_executable = next(
        (
            path
            for pattern in ("chrome-headless-shell.exe", "chrome-headless-shell", "headless_shell")
            for path in browser_root.rglob(pattern)
            if path.is_file()
        ),
        None,
    )
    browser_ready = (
        browser_version == REMOTION_BROWSER_VERSION and browser_executable is not None
    )
    checks.append(
        ProbeItem(
            name="remotion_browser",
            ready=browser_ready,
            detail=(
                f"Chrome Headless Shell {browser_version}: {browser_executable}"
                if browser_version and browser_executable
                else "Pinned Chrome Headless Shell is not installed"
            ),
            action=None if browser_ready else "Run npm run ensure-browser in remotion/.",
        )
    )
    return checks


def setup_remotion_runtime(
    project_root: Path,
    *,
    download: bool,
    environment: Mapping[str, str] | None = None,
) -> list[ProbeItem]:
    try:
        root = _ensure_remotion_project(project_root)
    except OSError as exc:
        return [
            ProbeItem(
                name="remotion_scaffold",
                ready=False,
                detail=str(exc),
                action="Clear or repair the project remotion/ directory, then run Setup again.",
            )
        ]
    npm = shutil.which("npm")
    if not npm:
        return [
            ProbeItem(
                name="remotion_setup",
                ready=False,
                detail="npm is not on PATH",
                action="Install current Node.js LTS for Windows.",
            )
        ]
    commands = []
    if download:
        commands.append(("npm_ci", [npm, "ci"], 900))
    commands.append(("remotion_typecheck", [npm, "run", "build"], 300))
    if download:
        commands.append(("remotion_browser_setup", [npm, "run", "ensure-browser"], 900))
    commands.append(("remotion_unit_tests", [npm, "test"], 300))
    results: list[ProbeItem] = []
    for name, command, timeout in commands:
        try:
            completed = subprocess.run(
                command,
                cwd=root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
                env=_remotion_subprocess_environment(environment),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            results.append(
                ProbeItem(
                    name=name,
                    ready=False,
                    detail=str(exc),
                    action="Run npm ci, npm run build, and npm run ensure-browser in remotion/.",
                )
            )
            return results
        output = "\n".join((completed.stderr or completed.stdout).splitlines()[-8:])
        results.append(
            ProbeItem(
                name=name,
                ready=completed.returncode == 0,
                detail=output or f"exit {completed.returncode}",
                action=(
                    None
                    if completed.returncode == 0
                    else "Run npm ci, npm run build, and npm run ensure-browser in remotion/."
                ),
            )
        )
        if completed.returncode != 0:
            return results
    return [*results, *probe_remotion_runtime(project_root)]


def _run_node(
    project_root: Path,
    script: str,
    arguments: list[str],
    *,
    timeout: int,
    environment: Mapping[str, str] | None = None,
) -> None:
    node = shutil.which("node")
    if not node:
        raise MediaError(
            "Node.js is required for the Remotion renderer",
            kind=ErrorKind.NOT_READY,
            action="Install current Node.js LTS for Windows.",
        )
    script_path = remotion_root(project_root) / "scripts" / script
    if not script_path.is_file():
        raise MediaError(
            f"Remotion script is missing: {script_path}", kind=ErrorKind.NOT_READY
        )
    try:
        completed = subprocess.run(
            [node, str(script_path), *arguments],
            cwd=remotion_root(project_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            env=_remotion_subprocess_environment(environment),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MediaError(
            f"Remotion command failed to start or timed out: {exc}",
            kind=ErrorKind.INTERNAL,
        ) from exc
    if completed.returncode != 0:
        detail = "\n".join((completed.stderr or completed.stdout).splitlines()[-40:])
        raise MediaError(
            f"Remotion command failed ({script}, exit {completed.returncode}):\n{detail}",
            kind=ErrorKind.INVALID_OUTPUT,
        )


def capture_source_screenshot(
    project_root: Path,
    *,
    url: str,
    output_path: Path,
    width: int,
    height: int,
    allowed_hosts: Sequence[str],
) -> None:
    failed = [check for check in probe_remotion_runtime(project_root) if not check.ready]
    if failed:
        detail = "; ".join(check.detail for check in failed)
        raise MediaError(
            f"Remotion runtime is not ready: {detail}",
            kind=ErrorKind.NOT_READY,
            action="Run npm install and npm run ensure-browser in remotion/.",
        )
    parsed = validate_public_http_url(url)
    hostname = (parsed.hostname or "").rstrip(".").casefold()
    normalized_hosts = [host.rstrip(".").casefold() for host in allowed_hosts]
    if not any(
        hostname == allowed or hostname.endswith("." + allowed)
        for allowed in normalized_hosts
    ):
        raise MediaError(
            f"source screenshot host is outside the frozen trust allowlist: {hostname}",
            kind=ErrorKind.UNSUPPORTED,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    request_path = output_path.with_suffix(".request.json")
    atomic_write_json(
        request_path,
        {
            "schema_version": 1,
            "url": url,
            "output": str(output_path.resolve()),
            "width": width,
            "height": height,
            "allowedHosts": normalized_hosts,
        },
    )
    _run_node(
        project_root,
        "screenshot.mjs",
        [str(request_path)],
        timeout=90,
    )
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise MediaError(
            "source screenshot did not produce an image", kind=ErrorKind.INVALID_OUTPUT
        )


def _write_wav(path: Path, samples: list[float], *, sample_rate: int = 48_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = b"".join(
        struct.pack("<h", max(-32768, min(32767, round(sample * 32767))))
        for sample in samples
    )
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)


def _sfx_samples(name: str, *, sample_rate: int = 48_000) -> list[float]:
    duration = {"click": 0.07, "pop": 0.16, "whoosh": 0.34}[name]
    count = round(duration * sample_rate)
    state = 17
    samples = []
    for index in range(count):
        position = index / max(1, count - 1)
        envelope = (1 - position) ** (7 if name == "click" else 3)
        if name == "click":
            value = math.sin(2 * math.pi * 1800 * index / sample_rate)
        elif name == "pop":
            frequency = 520 - 260 * position
            value = math.sin(2 * math.pi * frequency * index / sample_rate)
        else:
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            noise = state / 0x3FFFFFFF - 1
            value = noise * math.sin(math.pi * position)
        samples.append(value * envelope * 0.55)
    return samples


def _copy_media(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise MediaError(f"Remotion media is missing: {source}", kind=ErrorKind.NOT_READY)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def build_remotion_manifest(
    *,
    project_root: Path,
    work_dir: Path,
    edit_plan: RemotionEditPlan,
    assets: RemotionAssetBundle,
    narration_path: Path,
    output_language: OutputLanguage,
    music_path: Path | None,
    captions_enabled: bool,
) -> Path:
    media_dir = work_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    narration_file = "media/narration" + narration_path.suffix.casefold()
    _copy_media(narration_path, work_dir / narration_file)
    music_file = None
    if music_path is not None:
        music_file = "media/music" + music_path.suffix.casefold()
        _copy_media(music_path, work_dir / music_file)

    assets_by_shot = {asset.shot_id: asset for asset in assets.assets}
    asset_files: dict[str, str] = {}
    for asset in assets.assets:
        suffix = Path(asset.normalized.path).suffix.casefold()
        name = f"media/{asset.asset_id}{suffix}"
        _copy_media(project_root / asset.normalized.path, work_dir / name)
        asset_files[asset.shot_id] = name

    used_sfx = {shot.sfx.value for shot in edit_plan.shots if shot.sfx.value != "none"}
    sfx_files: dict[str, str] = {}
    for name in sorted(used_sfx):
        relative = f"media/sfx-{name}.wav"
        _write_wav(work_dir / relative, _sfx_samples(name))
        sfx_files[name] = relative

    shot_payload = []
    for shot in edit_plan.shots:
        asset = assets_by_shot.get(shot.shot_id)
        source_label = ""
        if asset and asset.source_page_url:
            source_label = urlparse(asset.source_page_url).hostname or asset.provider
        payload: dict[str, Any] = {
            "shotId": shot.shot_id,
            "sceneId": shot.scene_id,
            "startWordId": shot.start_word_id,
            "endWordId": shot.end_word_id,
            "narrationExcerpt": shot.narration_excerpt,
            "durationFrames": shot.end_frame - shot.start_frame,
            "template": shot.template.value,
            "purpose": shot.purpose,
            "headline": shot.headline,
            "supportingText": shot.supporting_text,
            "bodyLines": shot.body_lines,
            "motion": shot.motion.value,
            "transitionIn": shot.transition_in.value,
            "sfx": shot.sfx.value,
        }
        if asset is not None:
            payload.update(
                assetFile=asset_files[shot.shot_id],
                assetMediaKind=asset.media_kind,
                sourceLabel=source_label,
            )
        shot_payload.append(payload)
    caption_words = []
    for word in edit_plan.words:
        start_frame = min(
            edit_plan.duration_frames - 1,
            max(0, round(word.start_seconds * edit_plan.fps)),
        )
        end_frame = min(
            edit_plan.duration_frames,
            max(start_frame + 1, round(word.end_seconds * edit_plan.fps)),
        )
        caption_words.append(
            {
                "wordId": word.word_id,
                "text": word.text,
                "startFrame": start_frame,
                "endFrame": end_frame,
            }
        )
    manifest_path = work_dir / "render-manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "schemaVersion": 1,
            "title": edit_plan.title,
            "width": edit_plan.width,
            "height": edit_plan.height,
            "fps": edit_plan.fps,
            "durationFrames": edit_plan.duration_frames,
            "assetBaseUrl": "",
            "labels": REMOTION_LABELS[output_language],
            "narrationFile": narration_file,
            "captionsEnabled": captions_enabled,
            "captionWords": caption_words,
            **({"musicFile": music_file} if music_file else {}),
            "sfxFiles": sfx_files,
            "shots": shot_payload,
        },
    )
    return manifest_path


def render_remotion_video(
    project_root: Path,
    *,
    manifest_path: Path,
    output_path: Path,
    bundle_runtime_hash: str,
    proxy: bool = False,
) -> MediaReference:
    if len(bundle_runtime_hash) != 64 or any(
        character not in "0123456789abcdef" for character in bundle_runtime_hash
    ):
        raise MediaError(
            "Remotion bundle runtime hash is missing or invalid",
            kind=ErrorKind.NOT_READY,
        )
    failed = [check for check in probe_remotion_runtime(project_root) if not check.ready]
    if failed:
        detail = "; ".join(check.detail for check in failed)
        raise MediaError(
            f"Remotion runtime is not ready: {detail}",
            kind=ErrorKind.NOT_READY,
            action="Run npm install and npm run ensure-browser in remotion/.",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run_node(
        project_root,
        "render.mjs",
        [
            str(manifest_path.resolve()),
            str(output_path.resolve()),
            bundle_runtime_hash,
            *(["proxy"] if proxy else []),
        ],
        timeout=3600,
    )
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise MediaError("Remotion produced no video", kind=ErrorKind.INVALID_OUTPUT)
    return MediaReference(
        path=relative_path(output_path, project_root),
        sha256=sha256_file(output_path),
        mime_type="video/mp4",
    )


def manifest_reference(path: Path, project_root: Path) -> MediaReference:
    return MediaReference(
        path=relative_path(path, project_root),
        sha256=sha256_file(path),
        mime_type="application/json",
    )
