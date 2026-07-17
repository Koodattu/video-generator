from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Mapping

from pydantic import ValidationError

from .contracts import (
    ContentMode,
    CreativeBrief,
    NarrationDeliverySpec,
    NarrationPace,
    OutputLanguage,
    ProtocolName,
    Quality,
    RawRunConfig,
    RemotionAssetPolicy,
    ResolvedRunConfig,
    TASK_PROTOCOL,
    VideoOrientation,
    VideoStyle,
    VisualShotMode,
)
from .errors import ConfigurationError
from .profiles import BACKEND_DESCRIPTORS, PRICING_SNAPSHOT, PROFILE_VERSION, resolve_profile


SECRET_NAMES = {
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "ELEVENLABS_API_KEY",
    "HF_TOKEN",
    "BRAVE_SEARCH_API_KEY",
    "PEXELS_API_KEY",
}


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError as exc:
        raise ConfigurationError(f"file does not exist: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigurationError(f"invalid TOML in {path}: {exc}") from exc


def load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ConfigurationError(f"invalid .env assignment at {path}:{line_number}")
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if value and value[0] in {'"', "'"}:
            if len(value) < 2 or value[-1] != value[0]:
                raise ConfigurationError(f"unterminated quoted .env value at {path}:{line_number}")
            value = value[1:-1]
        values[name] = value
    return values


def load_environment(config_path: Path, environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Load project .env, then let the real process environment take precedence."""

    project_root = find_project_root(config_path.parent)
    merged = load_dotenv(project_root / ".env")
    merged.update(dict(os.environ if environ is None else environ))
    return merged


def find_project_root(start: Path) -> Path:
    candidate = start.resolve()
    for directory in (candidate, *candidate.parents):
        if (directory / "pyproject.toml").is_file():
            return directory
    return candidate


def resolve_narration_delivery(
    output_language: OutputLanguage,
    pace: NarrationPace,
    custom_direction: str = "",
) -> NarrationDeliverySpec:
    base_words_per_second = 2.55 if output_language is OutputLanguage.ENGLISH else 1.95
    pace_factor = {
        NarrationPace.SLOW: 0.86,
        NarrationPace.STANDARD: 1.0,
        NarrationPace.FAST: 1.18,
    }[pace]
    target = round(base_words_per_second * pace_factor, 3)
    pause_target, pause_maximum = {
        NarrationPace.SLOW: (0.45, 1.1),
        NarrationPace.STANDARD: (0.25, 0.75),
        NarrationPace.FAST: (0.08, 0.3),
    }[pace]
    description = custom_direction.strip() or {
        NarrationPace.SLOW: "Measured and reflective, with deliberate emphasis and room to absorb ideas.",
        NarrationPace.STANDARD: "Natural and engaged, with concise pauses and clear emphasis.",
        NarrationPace.FAST: "Urgent and tightly edited, with energetic delivery and almost no dead air.",
    }[pace]
    return NarrationDeliverySpec(
        pace=pace,
        target_words_per_second=target,
        minimum_words_per_second=round(target * 0.88, 3),
        maximum_words_per_second=round(target * 1.12, 3),
        target_pause_seconds=pause_target,
        maximum_pause_seconds=pause_maximum,
        tempo_multiplier=pace_factor,
        description=description,
    )


def load_raw_config(path: Path) -> RawRunConfig:
    try:
        return RawRunConfig.model_validate(_read_toml(path.resolve()))
    except ValidationError as exc:
        raise ConfigurationError(f"invalid Run configuration:\n{exc}") from exc


def load_brief(path: Path) -> CreativeBrief:
    try:
        return CreativeBrief.model_validate(_read_toml(path.resolve()))
    except ValidationError as exc:
        raise ConfigurationError(f"invalid Creative Brief:\n{exc}") from exc


def _relative_private_path(value: str, config_dir: Path, project_root: Path) -> str:
    if not value:
        return ""
    candidate = (config_dir / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    private_root = (project_root / "private").resolve()
    try:
        relative = candidate.relative_to(private_root)
    except ValueError as exc:
        raise ConfigurationError(
            f"voice reference must stay under {private_root}: {candidate}",
            action="Move the authorized recording and transcript under private/ and update config.toml.",
        ) from exc
    return (Path("private") / relative).as_posix()


def resolve_config(
    path: Path,
    *,
    overrides: Mapping[str, Any] | None = None,
    environment: Mapping[str, str] | None = None,
) -> ResolvedRunConfig:
    path = path.resolve()
    raw_data = _read_toml(path)
    for key, value in (overrides or {}).items():
        if value is not None:
            raw_data[key] = value
    try:
        raw = RawRunConfig.model_validate(raw_data)
    except ValidationError as exc:
        raise ConfigurationError(f"invalid Run configuration:\n{exc}") from exc

    project_root = find_project_root(path.parent)
    resolved_environment = load_environment(path, environ=environment)
    elevenlabs_voice_id = (
        raw.voice.elevenlabs_voice_id.strip()
        or resolved_environment.get("ELEVENLABS_VOICE_ID", "").strip()
    )
    bindings = resolve_profile(raw.profile)
    bindings.update(raw.task_overrides)
    for task_id, backend_id in bindings.items():
        descriptor = BACKEND_DESCRIPTORS.get(backend_id)
        if descriptor is None:
            raise ConfigurationError(f"task {task_id!r} selects unknown Backend {backend_id!r}")
        expected = TASK_PROTOCOL[task_id]
        if expected not in descriptor.protocols:
            raise ConfigurationError(
                f"Backend {backend_id!r} does not implement {expected.value} required by {task_id!r}"
            )
        if raw.output_language not in descriptor.languages:
            raise ConfigurationError(
                f"Backend {backend_id!r} does not support Output Language {raw.output_language.value!r}"
            )
        if raw.usage_purpose not in descriptor.allowed_usage_purposes:
            raise ConfigurationError(
                f"Backend {backend_id!r} is not declared compatible with {raw.usage_purpose!r}"
            )

    active_tasks = set(TASK_PROTOCOL)
    if raw.offline or raw.research_query_limit == 0:
        active_tasks.discard("search")
    effective_visual_shot_mode = (
        VisualShotMode.CADENCED
        if raw.video_style is VideoStyle.REMOTION_EXPLAINER
        else raw.visual_shot_mode
    )
    if (
        not raw.captions_enabled and effective_visual_shot_mode is VisualShotMode.SCENE_LOCKED
    ) or BACKEND_DESCRIPTORS[bindings["narration_synthesis"]].supports_word_timing:
        active_tasks.discard("caption_alignment")
    if raw.video_style is VideoStyle.STILL_IMAGE:
        active_tasks -= {
            "remotion_rhythm",
            "remotion_direction",
            "remotion_asset_select",
        }
    else:
        active_tasks -= {"visual_plan", "image_prompt_compile"}
    if raw.quality is Quality.DRAFT:
        active_tasks.discard("visual_review")
    if not raw.music_enabled:
        active_tasks -= {"music_brief", "music_generate"}
    if raw.content_mode is ContentMode.FICTION:
        active_tasks -= {"claim_inventory", "factual_review"}
    if raw.offline:
        cloud_bindings = sorted(
            {
                bindings[task_id]
                for task_id in active_tasks
                if BACKEND_DESCRIPTORS[bindings[task_id]].cloud
            }
        )
        if cloud_bindings:
            raise ConfigurationError(
                f"offline Run selects cloud Backends: {', '.join(cloud_bindings)}",
                action="Use only local task bindings or set offline = false. No automatic rerouting is performed.",
            )

    if raw.quality is Quality.FINAL:
        review = BACKEND_DESCRIPTORS[bindings["visual_review"]]
        if not review.supports_vision:
            raise ConfigurationError("final quality requires a vision-capable visual_review Backend")

    width, height = (1280, 720) if raw.quality is Quality.DRAFT else (1920, 1080)
    if raw.orientation is VideoOrientation.PORTRAIT:
        width, height = height, width
    voice = raw.voice.model_copy(
        update={
            "reference_audio": _relative_private_path(raw.voice.reference_audio, path.parent, project_root),
            "reference_transcript": _relative_private_path(
                raw.voice.reference_transcript, path.parent, project_root
            ),
            "elevenlabs_voice_id": elevenlabs_voice_id,
        }
    )
    speech = BACKEND_DESCRIPTORS[bindings["narration_synthesis"]]
    if speech.provider == "elevenlabs" and not voice.elevenlabs_voice_id:
        raise ConfigurationError(
            "the selected narration Backend requires voice.elevenlabs_voice_id",
            action=(
                "Create or choose an authorized ElevenLabs voice, then set ELEVENLABS_VOICE_ID "
                "in .env or voice.elevenlabs_voice_id in config.toml."
            ),
        )

    return ResolvedRunConfig(
        profile=raw.profile,
        profile_version=PROFILE_VERSION,
        output_language=raw.output_language,
        duration_seconds=raw.duration_seconds,
        quality=raw.quality,
        content_mode=raw.content_mode,
        content_format=raw.content_format,
        narration_pace=raw.narration_pace,
        narration_delivery=raw.narration_delivery,
        narration_delivery_spec=resolve_narration_delivery(
            raw.output_language,
            raw.narration_pace,
            raw.narration_delivery,
        ),
        audience=raw.audience,
        orientation=raw.orientation,
        video_style=raw.video_style,
        style=raw.style,
        style_description=raw.style_description,
        motion_style=raw.motion_style,
        remotion_asset_policy=(
            RemotionAssetPolicy.LOCAL_ONLY
            if raw.offline
            else raw.remotion_asset_policy
        ),
        remotion_allow_share_alike=raw.remotion_allow_share_alike,
        remotion_require_asset_approval=raw.remotion_require_asset_approval,
        remotion_source_screenshot_hosts=(
            [] if raw.offline else raw.remotion_source_screenshot_hosts
        ),
        offline=raw.offline,
        cost_ceiling_usd=raw.cost_ceiling_usd,
        failure_policy=raw.failure_policy,
        usage_purpose=raw.usage_purpose,
        idea_candidates=raw.idea_candidates,
        research_query_limit=raw.research_query_limit,
        research_source_limit=raw.research_source_limit,
        visual_target_seconds=raw.visual_target_seconds,
        visual_min_seconds=raw.visual_min_seconds,
        visual_max_seconds=raw.visual_max_seconds,
        visual_shot_mode=effective_visual_shot_mode,
        shot_target_seconds=raw.shot_target_seconds,
        shot_min_seconds=raw.shot_min_seconds,
        shot_max_seconds=raw.shot_max_seconds,
        music_enabled=raw.music_enabled,
        captions_enabled=raw.captions_enabled,
        animated_captions=raw.animated_captions,
        voice=voice,
        task_bindings=bindings,
        delivery_width=width,
        delivery_height=height,
        project_root=str(project_root),
        pricing_snapshot=PRICING_SNAPSHOT,
    )


def active_task_ids(config: ResolvedRunConfig) -> set[str]:
    task_ids = set(TASK_PROTOCOL)
    if config.offline or config.research_query_limit == 0:
        task_ids.discard("search")
    speech = BACKEND_DESCRIPTORS[config.task_bindings["narration_synthesis"]]
    if (
        not config.captions_enabled and config.visual_shot_mode is VisualShotMode.SCENE_LOCKED
    ) or speech.supports_word_timing:
        task_ids.discard("caption_alignment")
    if config.video_style is VideoStyle.STILL_IMAGE:
        task_ids -= {
            "remotion_rhythm",
            "remotion_direction",
            "remotion_asset_select",
        }
    else:
        task_ids -= {"visual_plan", "image_prompt_compile"}
    if config.quality is Quality.DRAFT:
        task_ids.discard("visual_review")
    if not config.music_enabled:
        task_ids -= {"music_brief", "music_generate"}
    if config.content_mode is ContentMode.FICTION:
        task_ids -= {"claim_inventory", "factual_review"}
    return task_ids


def active_backend_ids(config: ResolvedRunConfig) -> set[str]:
    return {config.task_bindings[task_id] for task_id in active_task_ids(config)}


def required_secret_names(config: ResolvedRunConfig) -> set[str]:
    result: set[str] = set()
    for backend_id in active_backend_ids(config):
        result.update(BACKEND_DESCRIPTORS[backend_id].required_env)
    return result


def redact_environment(environment: Mapping[str, str]) -> dict[str, str]:
    return {name: "<configured>" for name in SECRET_NAMES if environment.get(name)}
