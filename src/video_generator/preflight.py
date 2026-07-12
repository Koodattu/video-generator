from __future__ import annotations

import math
import os
import shutil
from pathlib import Path
from typing import Mapping

from .config import active_task_ids
from .contracts import (
    CostEstimate,
    PreflightReport,
    ProbeItem,
    ProbeReport,
    ProtocolName,
    Quality,
    ResolvedRunConfig,
    PUBLIC_STAGES,
    TASK_PROTOCOL,
)
from .errors import VideoGeneratorError
from .media import MediaTools
from .profiles import BACKEND_DESCRIPTORS, image_generation_dimensions
from .registry import BackendRegistry


TASK_LAST_USE_STAGE: dict[str, str] = {
    "search": "research",
    "research": "research",
    "ideate": "ideate",
    "select": "select",
    "outline": "outline",
    "script_draft": "script-draft",
    "review_story": "review-story",
    "review_spoken": "review-spoken",
    "review_constraints": "review-constraints",
    "script_revision": "script-revision",
    "factual_review": "script-revision",
    "narration_synthesis": "narration",
    "duration_repair": "narration",
    "caption_alignment": "captions",
    "visual_plan": "visual-plan",
    "image_prompt_compile": "image-prompt-compile",
    "image_generate": "visual-review",
    "visual_review": "visual-review",
    "music_brief": "music-brief",
    "music_generate": "music",
}


def _remaining_tasks(config: ResolvedRunConfig, from_stage: str | None) -> set[str]:
    tasks = active_task_ids(config)
    if from_stage is None:
        return tasks
    if from_stage not in PUBLIC_STAGES:
        raise ValueError(f"unknown public stage: {from_stage}")
    start = PUBLIC_STAGES.index(from_stage)
    return {
        task_id
        for task_id in tasks
        if PUBLIC_STAGES.index(
            "images"
            if task_id == "image_generate" and config.quality is Quality.DRAFT
            else TASK_LAST_USE_STAGE[task_id]
        )
        >= start
    }


def _remaining_backends(config: ResolvedRunConfig, from_stage: str | None) -> set[str]:
    return {config.task_bindings[task_id] for task_id in _remaining_tasks(config, from_stage)}


def estimate_cost(
    config: ResolvedRunConfig,
    *,
    from_stage: str | None = None,
    already_reserved_usd: float = 0,
    completed_calls: Mapping[str, int] | None = None,
) -> CostEstimate:
    """Estimate the planned first attempts plus the one allowed image regeneration batch.

    This is deliberately conservative but is not a provider quote. Runtime reservations remain the
    authoritative hard Cost Ceiling and account for retries before each request.
    """

    target_scene_count = max(
        1, math.ceil(float(config.duration_seconds) / float(config.visual_target_seconds))
    )
    scene_count = target_scene_count + 1
    line_items: dict[str, float] = {}
    completed_calls = dict(completed_calls or {})
    active_tasks = _remaining_tasks(config, from_stage)
    text_tasks = [task_id for task_id in active_tasks if TASK_PROTOCOL[task_id] is ProtocolName.STRUCTURED_TEXT]
    for task_id in text_tasks:
        descriptor = BACKEND_DESCRIPTORS[config.task_bindings[task_id]]
        if not descriptor.cloud:
            continue
        calls = (
            scene_count * 2
            if task_id == "visual_review"
            else scene_count
            if task_id == "image_prompt_compile"
            else 1
        )
        calls = max(0, calls - completed_calls.get(task_id, 0))
        # Structured calls reserve above the descriptor base according to output/input size.
        line_items[f"text:{task_id}"] = float(descriptor.reservation_usd) * 2.25 * calls

    if not config.offline and "search" in active_tasks:
        descriptor = BACKEND_DESCRIPTORS[config.task_bindings["search"]]
        if descriptor.cloud and descriptor.reservation_usd > 0:
            calls = max(0, config.research_query_limit - completed_calls.get("search", 0))
            line_items["search"] = float(descriptor.reservation_usd) * calls

    speech = BACKEND_DESCRIPTORS[config.task_bindings["narration_synthesis"]]
    if "narration_synthesis" in active_tasks and speech.cloud:
        # Include the single measured Duration Repair, which may regenerate every Scene.
        speech_calls = scene_count * 2
        speech_calls = max(0, speech_calls - completed_calls.get("narration_synthesis", 0))
        line_items["speech"] = float(speech.reservation_usd) * speech_calls

    if "caption_alignment" in active_tasks and config.captions_enabled and not speech.supports_word_timing:
        alignment = BACKEND_DESCRIPTORS[config.task_bindings["caption_alignment"]]
        if alignment.cloud:
            calls = max(0, scene_count - completed_calls.get("caption_alignment", 0))
            line_items["alignment"] = float(alignment.reservation_usd) * calls

    image = BACKEND_DESCRIPTORS[config.task_bindings["image_generate"]]
    if "image_generate" in active_tasks and image.cloud:
        quality_factor = 0.6 if config.quality is Quality.DRAFT else 2.0
        generation_width, generation_height = image_generation_dimensions(
            image.backend_id,
            delivery_width=config.delivery_width,
            delivery_height=config.delivery_height,
        )
        pixel_factor = max(
            1.0,
            generation_width * generation_height / (2048 * 1152),
        )
        reference_factor = 1.75 if image.supports_reference_images else 1.0
        generation_count = scene_count * (2 if config.quality is Quality.FINAL else 1)
        generation_count = max(0, generation_count - completed_calls.get("image_generate", 0))
        line_items["images"] = (
            float(image.reservation_usd)
            * quality_factor
            * pixel_factor
            * reference_factor
            * generation_count
        )

    if "music_generate" in active_tasks and config.music_enabled:
        music = BACKEND_DESCRIPTORS[config.task_bindings["music_generate"]]
        if music.cloud:
            generated_seconds = min(
                float(config.duration_seconds),
                float(music.max_duration_seconds or config.duration_seconds),
            )
            line_items["music"] = float(music.reservation_usd) * max(1.0, generated_seconds / 60)

    total = round(sum(line_items.values()), 4)
    return CostEstimate(
        estimated_usd=total,
        already_reserved_usd=round(already_reserved_usd, 4),
        projected_total_usd=round(already_reserved_usd + total, 4),
        ceiling_usd=float(config.cost_ceiling_usd),
        scene_count=scene_count,
        line_items={key: round(value, 4) for key, value in sorted(line_items.items())},
        basis=(
            "planned first attempts from "
            + (from_stage or "research")
            + " plus the one allowed final-quality image regeneration batch; retries are additional"
        ),
    )


def _voice_checks(config: ResolvedRunConfig, project_root: Path) -> list[ProbeItem]:
    backend_id = config.task_bindings["narration_synthesis"]
    descriptor = BACKEND_DESCRIPTORS[backend_id]
    checks = [
        ProbeItem(
            name="voice_authorization",
            ready=config.voice.authorization in {"self", "explicit_permission"},
            detail=f"authorization: {config.voice.authorization}",
        )
    ]
    if descriptor.provider == "local" and descriptor.supports_voice_cloning:
        if not config.voice.reference_audio:
            checks.append(
                ProbeItem(
                    name="voice_reference_audio",
                    ready=False,
                    detail="local voice cloning requires a reference recording",
                    action="Set voice.reference_audio to a WAV file under private/.",
                )
            )
        else:
            audio = (project_root / config.voice.reference_audio).resolve()
            audio_exists = audio.is_file()
            checks.append(
                ProbeItem(
                    name="voice_reference_audio",
                    ready=audio_exists,
                    detail=str(audio) if audio_exists else f"missing: {audio}",
                    action=None if audio_exists else "Add the authorized recording under private/.",
                )
            )
            if audio_exists:
                try:
                    probe = MediaTools.discover().probe_audio(audio)
                    duration_ready = 0.5 <= probe.duration_seconds <= 120
                    checks.append(
                        ProbeItem(
                            name="voice_reference_audio_media",
                            ready=duration_ready,
                            detail=(
                                f"{probe.duration_seconds:.2f}s, {probe.sample_rate} Hz, "
                                f"{probe.channels} channel(s)"
                            ),
                            action=(
                                None
                                if duration_ready
                                else "Use a decodable 0.5-120 second voice reference clip."
                            ),
                        )
                    )
                except VideoGeneratorError as exc:
                    checks.append(
                        ProbeItem(
                            name="voice_reference_audio_media",
                            ready=False,
                            detail=exc.message,
                            action="Replace the file with a decodable voice reference recording.",
                        )
                    )
        if config.voice.reference_transcript:
            transcript = (project_root / config.voice.reference_transcript).resolve()
            transcript_ready = False
            transcript_detail = f"missing: {transcript}"
            if transcript.is_file():
                try:
                    transcript_text = transcript.read_text(encoding="utf-8").strip()
                    transcript_ready = bool(transcript_text)
                    transcript_detail = (
                        f"{transcript} ({len(transcript_text)} characters)"
                        if transcript_text
                        else f"empty: {transcript}"
                    )
                except (OSError, UnicodeError) as exc:
                    transcript_detail = f"unreadable UTF-8 transcript: {exc}"
            checks.append(
                ProbeItem(
                    name="voice_reference_transcript",
                    ready=transcript_ready,
                    detail=transcript_detail,
                    action=(
                        None
                        if transcript_ready
                        else "Add the nonempty exact UTF-8 transcript under private/ or clear the setting."
                    ),
                )
            )
    elif descriptor.provider == "elevenlabs":
        checks.append(
            ProbeItem(
                name="elevenlabs_voice_id",
                ready=bool(config.voice.elevenlabs_voice_id),
                detail="configured" if config.voice.elevenlabs_voice_id else "missing",
                action=(
                    None
                    if config.voice.elevenlabs_voice_id
                    else "Set ELEVENLABS_VOICE_ID in .env or voice.elevenlabs_voice_id in config.toml."
                ),
            )
        )
    return checks


def run_preflight(
    *,
    config: ResolvedRunConfig,
    environment: Mapping[str, str],
    live: bool = False,
    run_root: Path | None = None,
    from_stage: str | None = None,
    already_reserved_usd: float = 0,
    completed_calls: Mapping[str, int] | None = None,
) -> PreflightReport:
    project_root = Path(config.project_root).resolve()
    checks: list[ProbeItem] = []
    warnings: list[str] = []

    remaining_tasks = _remaining_tasks(config, from_stage)
    remaining_backends = _remaining_backends(config, from_stage)
    required_secrets = {
        name
        for backend_id in remaining_backends
        for name in BACKEND_DESCRIPTORS[backend_id].required_env
    }
    missing_secrets = sorted(name for name in required_secrets if not environment.get(name))
    checks.append(
        ProbeItem(
            name="credentials",
            ready=not missing_secrets,
            detail="all required credentials configured" if not missing_secrets else f"missing: {', '.join(missing_secrets)}",
            action=None if not missing_secrets else "Add the missing values to .env or the process environment.",
        )
    )
    if "narration_synthesis" in remaining_tasks:
        checks.extend(_voice_checks(config, project_root))

    try:
        tools = MediaTools.discover()
        media_checks = tools.capability_checks(animated_captions=config.animated_captions)
        checks.extend(
            ProbeItem(name=item.name, ready=item.passed, detail=item.detail, action=None if item.passed else "Install a full FFmpeg build and rerun Preflight.")
            for item in media_checks
        )
    except VideoGeneratorError as exc:
        checks.append(ProbeItem(name="ffmpeg", ready=False, detail=exc.message, action=exc.action))

    free_bytes = shutil.disk_usage(project_root).free
    local_active = any(not BACKEND_DESCRIPTORS[item].cloud for item in remaining_backends)
    minimum_free = 5 * 1024**3 if local_active else 2 * 1024**3
    checks.append(
        ProbeItem(
            name="free_disk_space",
            ready=free_bytes >= minimum_free,
            detail=f"{free_bytes / 1024**3:.1f} GiB free; minimum {minimum_free / 1024**3:.0f} GiB for a Run",
            action=None if free_bytes >= minimum_free else "Free disk space before starting a Run.",
        )
    )
    checks.append(
        ProbeItem(
            name="project_writable",
            ready=os.access(project_root, os.W_OK),
            detail=str(project_root),
            action=None if os.access(project_root, os.W_OK) else "Grant write access to the project directory.",
        )
    )

    cost = estimate_cost(
        config,
        from_stage=from_stage,
        already_reserved_usd=already_reserved_usd,
        completed_calls=completed_calls,
    )
    cost_ready = cost.projected_total_usd <= cost.ceiling_usd + 1e-9
    checks.append(
        ProbeItem(
            name="cost_ceiling",
            ready=cost_ready,
            detail=(
                f"${cost.already_reserved_usd:.2f} already reserved + "
                f"${cost.estimated_usd:.2f} remaining estimate = "
                f"${cost.projected_total_usd:.2f} against ${cost.ceiling_usd:.2f} ceiling"
            ),
            action=None if cost_ready else "Increase cost_ceiling_usd explicitly or choose lower-cost task bindings.",
        )
    )
    if cost.line_items:
        warnings.append("The estimate is conservative but not a provider quote; transient retries can reserve additional cost.")

    backend_reports: list[ProbeReport] = []
    backend_root = (run_root or project_root / "runs" / ".preflight").resolve()
    with BackendRegistry(config=config, environment=environment, run_root=backend_root) as registry:
        for backend_id in sorted(remaining_backends):
            descriptor = BACKEND_DESCRIPTORS[backend_id]
            missing = [name for name in descriptor.required_env if not environment.get(name)]
            if missing:
                backend_reports.append(
                    ProbeReport(
                        backend_id=backend_id,
                        ready=False,
                        items=[ProbeItem(name="credentials", ready=False, detail=f"missing: {', '.join(missing)}")],
                    )
                )
                continue
            try:
                backend_reports.append(registry.probe(backend_id, live=live))
            except VideoGeneratorError as exc:
                backend_reports.append(
                    ProbeReport(
                        backend_id=backend_id,
                        ready=False,
                        items=[ProbeItem(name="probe", ready=False, detail=exc.message, action=exc.action)],
                    )
                )

    ready = all(item.ready for item in checks) and all(report.ready for report in backend_reports)
    return PreflightReport(
        ready=ready,
        profile=config.profile,
        output_language=config.output_language,
        active_backends=sorted(remaining_backends),
        checks=checks,
        backend_reports=backend_reports,
        cost=cost,
        live=live,
        warnings=warnings,
    )
