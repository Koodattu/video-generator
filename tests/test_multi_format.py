from __future__ import annotations

from types import SimpleNamespace

import pytest

from video_generator.config import resolve_narration_delivery
from video_generator.contracts import (
    CaptionTrack,
    ContentFormat,
    ContentMode,
    MediaReference,
    NarrationPace,
    NarrationScript,
    NarrationTimeline,
    OutputLanguage,
    RawRunConfig,
    ScriptScene,
    TimelineScene,
    TimedImageRequest,
    TimedVisualPlan,
    VisualShotMode,
    VoiceSettings,
    WordTiming,
)
from video_generator.errors import BackendError
from video_generator.profiles import BACKEND_DESCRIPTORS, PROFILES
from video_generator.prompting import MULTI_FORMAT_PROMPT_SET_VERSION, build_frozen_assets
from video_generator.task_models import task_output_models
from video_generator.workflow import CaptionBundle, ImageRequestSet, NarrationBundle, WorkflowEngine


def test_production_profiles_never_select_the_programmatic_image_backend() -> None:
    for profile_name, bindings in PROFILES.items():
        if profile_name == "deterministic-test":
            continue
        image_backend = BACKEND_DESCRIPTORS[bindings["image_generate"]]
        assert image_backend.provider != "deterministic", profile_name


def test_cadenced_mythbuster_scene_count_follows_editorial_arc_not_image_target() -> None:
    engine = object.__new__(WorkflowEngine)
    engine.workflow_policy_version = 12
    engine.config = SimpleNamespace(
        duration_seconds=24,
        visual_target_seconds=10,
        visual_min_seconds=6,
        visual_max_seconds=14,
        visual_shot_mode=VisualShotMode.CADENCED,
        content_format=ContentFormat.MYTHBUSTER,
    )

    assert engine._outline_scene_count_bounds() == (5, 4, 5)
    engine.config.content_format = ContentFormat.EXPLAINER
    assert engine._outline_scene_count_bounds() == (4, 3, 5)


def test_multi_format_prompt_pack_selects_timed_contracts(resolved_config) -> None:
    delivery = resolve_narration_delivery(OutputLanguage.ENGLISH, NarrationPace.FAST)
    config = resolved_config.model_copy(
        update={
            "content_mode": ContentMode.FACTUAL,
            "content_format": ContentFormat.MYTHBUSTER,
            "narration_pace": NarrationPace.FAST,
            "narration_delivery_spec": delivery,
            "visual_shot_mode": VisualShotMode.CADENCED,
        }
    )

    models = task_output_models(config)
    assets = build_frozen_assets(config)

    assert models["visual_plan"] is TimedVisualPlan
    assert models["image_prompt_compile"] is TimedImageRequest
    assert assets["prompt_set_version"] == MULTI_FORMAT_PROMPT_SET_VERSION
    assert assets["workflow_policy_version"] == 26
    claim_properties = assets["schemas"]["claim_inventory"]["$defs"]["ExtractedClaim"][
        "properties"
    ]
    assert set(claim_properties) == {"exact_text", "qualification"}
    assert "shots" in assets["schemas"]["visual_plan"]["properties"]
    assert set(assets["schemas"]["research"]["properties"]) == {"evidence"}
    evidence_schema = assets["schemas"]["research"]["properties"]["evidence"]
    assert evidence_schema["maxItems"] == 12
    evidence_properties = assets["schemas"]["research"]["$defs"][
        "EvidenceRecordDraft"
    ]["properties"]
    assert "evidence_id" not in evidence_properties
    assert evidence_properties["supported_statement"]["maxLength"] == 600
    assert evidence_properties["limitations"]["maxItems"] == 4
    assert evidence_properties["limitations"]["items"]["maxLength"] == 240
    assert "ResearchFindingDraft" not in assets["schemas"]["research"]["$defs"]
    assert "urgent" in assets["prompts"]["script_draft"]["instructions"].lower()


def test_shot_schedule_is_frame_aligned_and_keeps_parent_scene_ids() -> None:
    engine = object.__new__(WorkflowEngine)
    engine.config = SimpleNamespace(
        fps=30,
        shot_target_seconds=3,
        shot_min_seconds=2,
        shot_max_seconds=5,
    )
    script = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(
                scene_id="scene-001",
                spoken_text="One two three four five six seven eight nine ten eleven twelve.",
                pause_after_seconds=0,
            )
        ],
    )
    media = MediaReference(path="fixture.wav", sha256="0" * 64, mime_type="audio/wav")
    timeline = NarrationTimeline(
        narration_audio=media,
        duration_seconds=10,
        delivery_duration_seconds=10,
        fps=30,
        scenes=[
            TimelineScene(
                scene_id="scene-001",
                audio=media,
                start_seconds=0,
                speech_end_seconds=10,
                end_seconds=10,
            )
        ],
    )
    words = [
        WordTiming(text=word, start_seconds=index * 0.8, end_seconds=index * 0.8 + 0.6)
        for index, word in enumerate(script.scenes[0].spoken_text.split())
    ]
    captions = CaptionBundle(
        enabled=False,
        track=CaptionTrack(language=OutputLanguage.ENGLISH, words=words),
        scene_words={"scene-001": words},
    )
    narration = NarrationBundle(script=script, timeline=timeline, items=[])

    schedule = engine._build_shot_schedule(narration, captions)

    assert [shot["shot_id"] for shot in schedule] == ["shot-001", "shot-002", "shot-003"]
    assert {shot["scene_id"] for shot in schedule} == {"scene-001"}
    assert schedule[0]["start_seconds"] == 0
    assert schedule[-1]["end_seconds"] == 10
    assert all(
        abs(round(shot["start_seconds"] * 30) - shot["start_seconds"] * 30) < 0.0001
        for shot in schedule
    )
    assert all(
        abs(round(shot["end_seconds"] * 30) - shot["end_seconds"] * 30) < 0.0001
        for shot in schedule
    )


def test_delivery_presets_are_quantitative() -> None:
    slow = resolve_narration_delivery(OutputLanguage.ENGLISH, NarrationPace.SLOW)
    fast = resolve_narration_delivery(OutputLanguage.ENGLISH, NarrationPace.FAST)

    assert slow.target_words_per_second < fast.target_words_per_second
    assert slow.maximum_pause_seconds > fast.maximum_pause_seconds
    assert slow.tempo_multiplier < 1 < fast.tempo_multiplier


def test_factual_narrative_prompt_does_not_fall_back_to_fiction(resolved_config) -> None:
    config = resolved_config.model_copy(
        update={
            "content_mode": ContentMode.FACTUAL,
            "content_format": ContentFormat.NARRATIVE,
        }
    )

    assets = build_frozen_assets(config)
    ideate = assets["prompts"]["ideate"]["instructions"].lower()
    outline = assets["prompts"]["outline"]["instructions"].lower()

    assert "factual narrative" in ideate
    assert "loose inspiration" not in ideate
    assert "factual research pack" in outline


def test_oversized_cadenced_plan_is_rejected_before_a_run_starts() -> None:
    with pytest.raises(ValueError, match="single-plan limit"):
        RawRunConfig(
            duration_seconds=600,
            visual_shot_mode=VisualShotMode.CADENCED,
            voice=VoiceSettings(name="fixture"),
        )


def test_infeasible_scene_shot_bounds_are_rejected() -> None:
    engine = object.__new__(WorkflowEngine)
    engine.config = SimpleNamespace(
        fps=30,
        shot_target_seconds=4.5,
        shot_min_seconds=4,
        shot_max_seconds=5,
    )
    script = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(
                scene_id="scene-001",
                spoken_text="One two three four five six.",
                pause_after_seconds=0,
            )
        ],
    )
    media = MediaReference(path="fixture.wav", sha256="0" * 64, mime_type="audio/wav")
    timeline = NarrationTimeline(
        narration_audio=media,
        duration_seconds=6,
        delivery_duration_seconds=6,
        fps=30,
        scenes=[
            TimelineScene(
                scene_id="scene-001",
                audio=media,
                start_seconds=0,
                speech_end_seconds=6,
                end_seconds=6,
            )
        ],
    )
    narration = NarrationBundle(script=script, timeline=timeline, items=[])

    with pytest.raises(BackendError, match="cannot be divided into Shots"):
        engine._build_shot_schedule(narration, CaptionBundle(enabled=False))


def test_short_scene_uses_one_boundary_limited_shot() -> None:
    engine = object.__new__(WorkflowEngine)
    engine.config = SimpleNamespace(
        fps=30,
        shot_target_seconds=3,
        shot_min_seconds=2,
        shot_max_seconds=5,
    )
    script = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(
                scene_id="scene-001",
                spoken_text="What is happening?",
                pause_after_seconds=0,
            )
        ],
    )
    media = MediaReference(path="fixture.wav", sha256="0" * 64, mime_type="audio/wav")
    timeline = NarrationTimeline(
        narration_audio=media,
        duration_seconds=1.8,
        delivery_duration_seconds=1.8,
        fps=30,
        scenes=[
            TimelineScene(
                scene_id="scene-001",
                audio=media,
                start_seconds=0,
                speech_end_seconds=1.8,
                end_seconds=1.8,
            )
        ],
    )
    narration = NarrationBundle(script=script, timeline=timeline, items=[])

    schedule = engine._build_shot_schedule(narration, CaptionBundle(enabled=False))

    assert schedule == [
        {
            "shot_id": "shot-001",
            "scene_id": "scene-001",
            "narration_excerpt": "What is happening?",
            "start_seconds": 0.0,
            "end_seconds": 1.8,
        }
    ]


def test_duplicate_visual_request_ids_are_rejected() -> None:
    request = TimedImageRequest(
        scene_id="scene-001",
        shot_id="shot-001",
        target_backend_id="local:flux.2-klein-4b",
        prompt="A blue boot compresses white snow crystals, simple diagram, no text.",
        width=1024,
        height=576,
    )

    with pytest.raises(ValueError, match="visual IDs must be nonempty and unique"):
        ImageRequestSet(requests=[request, request])
