from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from video_generator.backends import deterministic as deterministic_backend
from video_generator.config import resolve_narration_delivery
from video_generator.contracts import (
    ContentFormat,
    ContentMode,
    CreativeBrief,
    FactualRevisedScript,
    NarrationPace,
    OutputLanguage,
    RevisedScript,
    TimedVisualPlan,
    VisualPlan,
    VisualShotMode,
)
from video_generator.errors import BackendError
from video_generator.profiles import PROFILES
from video_generator.prompting import build_frozen_assets
from video_generator.provenance import build_runtime_snapshot
from video_generator.run_store import RunStore
from video_generator.workflow import RenderBundle, WorkflowEngine


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="the deterministic end-to-end workflow requires FFmpeg and ffprobe",
)
@pytest.mark.parametrize(
    ("language", "music_enabled"),
    [
        (OutputLanguage.ENGLISH, False),
        (OutputLanguage.FINNISH, False),
        (OutputLanguage.ENGLISH, True),
    ],
)
def test_deterministic_workflow_delivers_video_and_captions(
    tmp_path: Path,
    resolved_config,
    language: OutputLanguage,
    music_enabled: bool,
) -> None:
    config = resolved_config.model_copy(
        update={
            "profile": "deterministic-test",
            "project_root": str(tmp_path),
            "task_bindings": dict(PROFILES["deterministic-test"]),
            "output_language": language,
            "duration_seconds": 10,
            "visual_target_seconds": 5,
            "visual_min_seconds": 4,
            "visual_max_seconds": 8,
            "idea_candidates": 2,
            "research_query_limit": 1,
            "research_source_limit": 2,
            "music_enabled": music_enabled,
            "captions_enabled": True,
            "animated_captions": True,
        }
    )
    brief = CreativeBrief(idea_direction="A tiny mystery on a snowy path")
    frozen_assets = build_frozen_assets(config)
    frozen_assets["runtime_snapshot"] = build_runtime_snapshot(config)
    store = RunStore.create(
        project_root=tmp_path,
        config=config,
        brief=brief,
        frozen_assets=frozen_assets,
    )

    with WorkflowEngine(store=store, environment={}) as workflow:
        delivery = workflow.run()

    assert delivery is not None
    assert store.manifest.status == "complete"
    assert {output.role for output in delivery.outputs} >= {
        "primary_video",
        "burned_caption_video",
        "caption_sidecar",
    }
    assert all(check.passed for check in delivery.checks)
    for output in delivery.outputs:
        assert (tmp_path / output.media.path).is_file()
    if music_enabled:
        render_record = store.stage_record("render")
        assert render_record is not None
        render_artifact = store.load_artifact(render_record, RenderBundle)
        assert render_artifact.plan.music_path
    store.validate_completed_outputs()


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="the deterministic end-to-end workflow requires FFmpeg and ffprobe",
)
def test_factual_fast_mythbuster_uses_evidence_gate_and_timed_shots(
    tmp_path: Path,
    resolved_config,
) -> None:
    delivery = resolve_narration_delivery(OutputLanguage.ENGLISH, NarrationPace.FAST)
    config = resolved_config.model_copy(
        update={
            "profile": "deterministic-test",
            "project_root": str(tmp_path),
            "task_bindings": dict(PROFILES["deterministic-test"]),
            "output_language": OutputLanguage.ENGLISH,
            "duration_seconds": 10,
            "content_mode": ContentMode.FACTUAL,
            "content_format": ContentFormat.MYTHBUSTER,
            "narration_pace": NarrationPace.FAST,
            "narration_delivery_spec": delivery,
            "style": "editorial_doodle",
            "style_description": "Minimal black ink doodles with one blue accent and no text.",
            "visual_target_seconds": 5,
            "visual_min_seconds": 4,
            "visual_max_seconds": 8,
            "visual_shot_mode": VisualShotMode.CADENCED,
            "shot_target_seconds": 3,
            "shot_min_seconds": 2,
            "shot_max_seconds": 5,
            "idea_candidates": 2,
            "research_query_limit": 1,
            "research_source_limit": 2,
            "music_enabled": False,
            "captions_enabled": False,
            "animated_captions": False,
            "offline": False,
        }
    )
    brief = CreativeBrief(
        idea_direction="Explain why very cold snow squeaks under a boot.",
        research_focus=["why cold snow squeaks"],
        modern_anchor="the sound under your winter boot",
        central_question="Why does very cold snow sound different?",
        misconception="Only boot weight controls the sound.",
    )
    frozen_assets = build_frozen_assets(config)
    frozen_assets["runtime_snapshot"] = build_runtime_snapshot(config)
    store = RunStore.create(
        project_root=tmp_path,
        config=config,
        brief=brief,
        frozen_assets=frozen_assets,
    )

    with WorkflowEngine(store=store, environment={}) as workflow:
        delivery_manifest = workflow.run()

    assert delivery_manifest is not None
    revision_record = store.stage_record("script-revision")
    visual_record = store.stage_record("visual-plan")
    render_record = store.stage_record("render")
    assert revision_record is not None and visual_record is not None and render_record is not None
    revision = store.load_artifact(revision_record, FactualRevisedScript)
    visual_plan = store.load_artifact(visual_record, TimedVisualPlan)
    rendered = store.load_artifact(render_record, RenderBundle)
    assert revision.factual_review.passed
    assert len(visual_plan.shots) > len(revision.script.scenes)
    assert all(scene.shot_id for scene in rendered.plan.scenes)
    assert rendered.plan.scenes[-1].end_seconds == rendered.plan.duration_seconds
    store.validate_completed_outputs()


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="the deterministic end-to-end workflow requires FFmpeg and ffprobe",
)
def test_fiction_slow_explainer_keeps_scene_locked_visuals(
    tmp_path: Path,
    resolved_config,
) -> None:
    delivery = resolve_narration_delivery(OutputLanguage.ENGLISH, NarrationPace.SLOW)
    config = resolved_config.model_copy(
        update={
            "profile": "deterministic-test",
            "project_root": str(tmp_path),
            "task_bindings": dict(PROFILES["deterministic-test"]),
            "output_language": OutputLanguage.ENGLISH,
            "duration_seconds": 10,
            "content_mode": ContentMode.FICTION,
            "content_format": ContentFormat.EXPLAINER,
            "narration_pace": NarrationPace.SLOW,
            "narration_delivery_spec": delivery,
            "style": "paper_cutout",
            "style_description": "Simple layered paper shapes with soft fibers and no text.",
            "visual_target_seconds": 5,
            "visual_min_seconds": 4,
            "visual_max_seconds": 8,
            "visual_shot_mode": VisualShotMode.SCENE_LOCKED,
            "idea_candidates": 2,
            "research_query_limit": 1,
            "research_source_limit": 2,
            "music_enabled": False,
            "captions_enabled": False,
            "animated_captions": False,
            "offline": False,
        }
    )
    brief = CreativeBrief(
        idea_direction="Explain how an imaginary lantern stores a traveler's memories.",
        modern_anchor="the battery icon on your phone",
        central_question="How could a memory lantern run out of room?",
        desired_takeaway="Treat its fictional rules as a clear, cumulative explanation.",
    )
    frozen_assets = build_frozen_assets(config)
    frozen_assets["runtime_snapshot"] = build_runtime_snapshot(config)
    store = RunStore.create(
        project_root=tmp_path,
        config=config,
        brief=brief,
        frozen_assets=frozen_assets,
    )

    with WorkflowEngine(store=store, environment={}) as workflow:
        delivery_manifest = workflow.run()

    assert delivery_manifest is not None
    revision_record = store.stage_record("script-revision")
    visual_record = store.stage_record("visual-plan")
    render_record = store.stage_record("render")
    assert revision_record is not None and visual_record is not None and render_record is not None
    revision = store.load_artifact(revision_record, RevisedScript)
    visual_plan = store.load_artifact(visual_record, VisualPlan)
    rendered = store.load_artifact(render_record, RenderBundle)
    assert revision.script.scenes
    assert len(visual_plan.scenes) == len(revision.script.scenes)
    assert all(scene.shot_id is None for scene in rendered.plan.scenes)
    assert "claim_inventory" not in frozen_assets["prompts"]
    store.validate_completed_outputs()


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="WorkflowEngine requires FFmpeg discovery even when the factual gate blocks TTS",
)
def test_unsupported_factual_claim_blocks_all_tts_calls(
    tmp_path: Path,
    resolved_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = resolved_config.model_copy(
        update={
            "profile": "deterministic-test",
            "project_root": str(tmp_path),
            "task_bindings": dict(PROFILES["deterministic-test"]),
            "output_language": OutputLanguage.ENGLISH,
            "duration_seconds": 10,
            "content_mode": ContentMode.FACTUAL,
            "content_format": ContentFormat.MYTHBUSTER,
            "visual_target_seconds": 5,
            "visual_min_seconds": 4,
            "visual_max_seconds": 8,
            "idea_candidates": 1,
            "research_query_limit": 1,
            "research_source_limit": 1,
            "offline": False,
        }
    )
    brief = CreativeBrief(
        idea_direction="Explain why very cold snow squeaks.",
        research_focus=["why cold snow squeaks"],
        modern_anchor="the sound under a winter boot",
        central_question="Why does very cold snow squeak?",
        misconception="Only boot weight controls the sound.",
    )
    original_fixture = deterministic_backend._fake_structured
    selector_saw_research = False

    def unsupported_fixture(request):
        nonlocal selector_saw_research
        if request.task_id == "select":
            selector_saw_research = "research_pack" in request.input_data
        if request.task_id != "factual_review":
            return original_fixture(request)
        claims = request.input_data["claim_inventory"]["claims"]
        return {
            "schema_version": 1,
            "passed": False,
            "claims": [
                {
                    "claim_id": claim["claim_id"],
                    "verdict": "unsupported",
                    "evidence_ids": [],
                    "rationale": "The bounded evidence does not directly support this wording.",
                }
                for claim in claims
            ],
            "uncovered_claims": [],
            "summary": "Narration is blocked.",
        }

    speech_calls = 0
    original_synthesize = deterministic_backend.DeterministicSpeechBackend.synthesize

    def count_speech(self, request):
        nonlocal speech_calls
        speech_calls += 1
        return original_synthesize(self, request)

    monkeypatch.setattr(deterministic_backend, "_fake_structured", unsupported_fixture)
    monkeypatch.setattr(
        deterministic_backend.DeterministicSpeechBackend,
        "synthesize",
        count_speech,
    )
    store = RunStore.create(
        project_root=tmp_path,
        config=config,
        brief=brief,
        frozen_assets=build_frozen_assets(config),
    )

    with pytest.raises(BackendError, match="factual accuracy gate blocked narration"):
        with WorkflowEngine(store=store, environment={}) as workflow:
            workflow.run()

    assert speech_calls == 0
    assert selector_saw_research
    assert store.stage_record("narration") is None
