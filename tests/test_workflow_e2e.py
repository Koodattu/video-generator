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
    ExplainerOutline,
    FactualRevisedScript,
    NarrationScript,
    NarrationPace,
    OutputLanguage,
    ResearchSource,
    RevisedScript,
    SearchResult,
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
    monkeypatch: pytest.MonkeyPatch,
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
    original_fixture = deterministic_backend._fake_structured
    visual_requests = []
    image_prompt_requests = []

    def capture_visual_requests(request):
        if request.task_id == "visual_plan":
            visual_requests.append(request)
        if request.task_id == "image_prompt_compile":
            image_prompt_requests.append(request)
        return original_fixture(request)

    monkeypatch.setattr(
        deterministic_backend,
        "_fake_structured",
        capture_visual_requests,
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
    foundation_requests = [
        request
        for request in visual_requests
        if request.input_data.get("visual_strategy") == "foundation-v1"
    ]
    content_requests = [
        request
        for request in visual_requests
        if request.input_data.get("visual_strategy") == "single-visual-v1"
    ]
    assert len(foundation_requests) == 1
    assert set(foundation_requests[0].output_schema["properties"]) == {
        "style",
        "characters",
    }
    assert len(content_requests) == len(visual_plan.shots)
    host_owned_visual_fields = {
        "shot_id",
        "scene_id",
        "narration_excerpt",
        "start_seconds",
        "end_seconds",
        "duration_seconds",
        "style_profile",
        "characters",
    }
    assert all(
        not host_owned_visual_fields
        & set(request.output_schema["properties"])
        for request in content_requests
    )
    assert set(visual_record.item_ids) == {
        "foundation",
        *[f"content-{shot.shot_id}" for shot in visual_plan.shots],
    }
    assert len(image_prompt_requests) == len(visual_plan.shots)
    assert all(
        set(request.output_schema["properties"])
        == {"prompt", "negative_prompt"}
        for request in image_prompt_requests
    )
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
    reason="WorkflowEngine requires FFmpeg discovery for a stopped workflow",
)
def test_research_source_budget_is_shared_across_queries(
    tmp_path: Path,
    resolved_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = resolved_config.model_copy(
        update={
            "profile": "deterministic-test",
            "project_root": str(tmp_path),
            "task_bindings": dict(PROFILES["deterministic-test"]),
            "research_query_limit": 2,
            "research_source_limit": 5,
            "offline": False,
        }
    )
    brief = CreativeBrief(
        idea_direction="A tiny mystery on a snowy path.",
        research_focus=["first bounded topic", "second bounded topic"],
    )
    frozen_assets = build_frozen_assets(config)
    frozen_assets["runtime_snapshot"] = build_runtime_snapshot(config)
    store = RunStore.create(
        project_root=tmp_path,
        config=config,
        brief=brief,
        frozen_assets=frozen_assets,
    )
    requests = []

    def bounded_search(request):
        requests.append(request)
        query_number = len(requests)
        return SearchResult(
            query=request.query,
            sources=[
                ResearchSource(
                    source_id=f"raw-{query_number}-{index}",
                    url=f"https://example.test/{query_number}/{index}",
                    title=f"Fixture source {query_number}-{index}",
                    excerpt="A bounded fixture excerpt.",
                )
                for index in range(1, request.max_results + 1)
            ],
        )

    with WorkflowEngine(
        store=store,
        environment={},
        stop_after="research",
    ) as workflow:
        monkeypatch.setattr(workflow.executor, "search", bounded_search)
        assert workflow.run() is None

    assert [request.max_results for request in requests] == [3, 2]
    research_record = store.stage_record("research")
    assert research_record is not None
    assert research_record.item_ids == ["query-001", "query-002"]


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="WorkflowEngine requires FFmpeg discovery for a stopped workflow",
)
def test_scene_local_draft_returns_only_text_and_host_assembles_script(
    tmp_path: Path,
    resolved_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = resolved_config.model_copy(
        update={
            "profile": "deterministic-test",
            "project_root": str(tmp_path),
            "task_bindings": dict(PROFILES["deterministic-test"]),
            "content_format": ContentFormat.EXPLAINER,
            "duration_seconds": 12,
            "visual_target_seconds": 5,
            "visual_min_seconds": 4,
            "visual_max_seconds": 8,
            "idea_candidates": 1,
            "research_query_limit": 0,
            "research_source_limit": 0,
            "offline": True,
        }
    )
    brief = CreativeBrief(idea_direction="Explain an imaginary pocket weather machine.")
    original_fixture = deterministic_backend._fake_structured
    draft_requests = []

    def capture_scene_drafts(request):
        if (
            request.task_id == "script_draft"
            and request.input_data.get("draft_strategy") == "single-scene-v1"
        ):
            draft_requests.append(request)
        return original_fixture(request)

    monkeypatch.setattr(deterministic_backend, "_fake_structured", capture_scene_drafts)
    frozen_assets = build_frozen_assets(config)
    frozen_assets["runtime_snapshot"] = build_runtime_snapshot(config)
    store = RunStore.create(
        project_root=tmp_path,
        config=config,
        brief=brief,
        frozen_assets=frozen_assets,
    )

    with WorkflowEngine(
        store=store,
        environment={},
        stop_after="script-draft",
    ) as workflow:
        assert workflow.run() is None

    outline_record = store.stage_record("outline")
    draft_record = store.stage_record("script-draft")
    assert outline_record is not None and draft_record is not None
    outline = store.load_artifact(outline_record, ExplainerOutline)
    draft = store.load_artifact(draft_record, NarrationScript)
    expected_ids = [scene.scene_id for scene in outline.scenes]
    assert len(draft_requests) == len(expected_ids)
    assert [
        request.input_data["outline_scene"]["scene_id"]
        for request in draft_requests
    ] == expected_ids
    assert draft_record.item_ids == [f"draft-{scene_id}" for scene_id in expected_ids]
    assert all(
        set(request.output_schema["properties"]) == {"spoken_text"}
        for request in draft_requests
    )
    assert draft.title == outline.title
    assert [scene.scene_id for scene in draft.scenes] == expected_ids
    assert draft.scenes[-1].pause_after_seconds == 0
    assert all(
        scene.pause_after_seconds <= config.narration_delivery_spec.maximum_pause_seconds
        for scene in draft.scenes[:-1]
    )
    for request, scene in zip(draft_requests, draft.scenes, strict=True):
        word_count = len(scene.spoken_text.split())
        assert request.input_data["minimum_word_count"] <= word_count
        assert word_count <= request.input_data["maximum_word_count"]


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="WorkflowEngine requires FFmpeg discovery for a stopped workflow",
)
def test_scene_local_draft_allows_scene_word_counts_to_compensate(
    tmp_path: Path,
    resolved_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = resolved_config.model_copy(
        update={
            "profile": "deterministic-test",
            "project_root": str(tmp_path),
            "task_bindings": dict(PROFILES["deterministic-test"]),
            "content_format": ContentFormat.EXPLAINER,
            "duration_seconds": 12,
            "visual_target_seconds": 5,
            "visual_min_seconds": 4,
            "visual_max_seconds": 8,
            "idea_candidates": 1,
            "research_query_limit": 0,
            "research_source_limit": 0,
            "offline": True,
        }
    )
    brief = CreativeBrief(idea_direction="Explain an imaginary pocket weather machine.")
    original_fixture = deterministic_backend._fake_structured
    draft_requests = []
    fit_requests = []
    carried_deficit = 0

    def uneven_scene_fixture(request):
        nonlocal carried_deficit
        if request.task_id != "script_draft":
            return original_fixture(request)
        strategy = request.input_data.get("draft_strategy")
        if strategy == "single-scene-word-fit-v1":
            fit_requests.append(request)
            return original_fixture(request)
        if strategy != "single-scene-v1":
            return original_fixture(request)
        draft_requests.append(request)
        generated = original_fixture(request)["spoken_text"].split()
        position = int(request.input_data["scene_position"])
        target = int(request.input_data["target_word_count"])
        if position == 1:
            count = max(1, int(request.input_data["minimum_word_count"]) - 2)
            carried_deficit = target - count
        elif position == 2:
            count = target + carried_deficit
        else:
            count = target
        while len(generated) < count:
            generated.append("clearly")
        return {"spoken_text": " ".join(generated[:count]).rstrip(".,;:") + "."}

    monkeypatch.setattr(deterministic_backend, "_fake_structured", uneven_scene_fixture)
    store = RunStore.create(
        project_root=tmp_path,
        config=config,
        brief=brief,
        frozen_assets=build_frozen_assets(config),
    )

    with WorkflowEngine(
        store=store,
        environment={},
        stop_after="script-draft",
    ) as workflow:
        assert workflow.run() is None

    draft_record = store.stage_record("script-draft")
    assert draft_record is not None
    draft = store.load_artifact(draft_record, NarrationScript)
    assert len(draft_requests) == len(draft.scenes)
    assert fit_requests == []
    assert len(draft.scenes[0].spoken_text.split()) < int(
        draft_requests[0].input_data["minimum_word_count"]
    )
    assert sum(len(scene.spoken_text.split()) for scene in draft.scenes) == sum(
        int(request.input_data["target_word_count"]) for request in draft_requests
    )
    assert all(
        request.input_data["scene_word_policy"]
        == "advisory-with-host-aggregate-fit-v1"
        for request in draft_requests
    )


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="WorkflowEngine requires FFmpeg discovery for a stopped workflow",
)
def test_scene_local_revision_only_replaces_affected_spoken_text(
    tmp_path: Path,
    resolved_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = resolved_config.model_copy(
        update={
            "profile": "deterministic-test",
            "project_root": str(tmp_path),
            "task_bindings": dict(PROFILES["deterministic-test"]),
            "content_format": ContentFormat.EXPLAINER,
            "duration_seconds": 12,
            "visual_target_seconds": 5,
            "visual_min_seconds": 4,
            "visual_max_seconds": 8,
            "idea_candidates": 1,
            "research_query_limit": 0,
            "research_source_limit": 0,
            "offline": True,
        }
    )
    brief = CreativeBrief(idea_direction="Explain an imaginary pocket weather machine.")
    original_fixture = deterministic_backend._fake_structured
    replacement_requests = []

    def scene_revision_fixture(request):
        if request.task_id == "review_spoken":
            return {
                "schema_version": 1,
                "review_type": "wrong-on-purpose",
                "passed": True,
                "findings": [
                    {
                        "finding_id": "model-owned-id",
                        "severity": "major",
                        "scene_id": "scene-002",
                        "evidence": "The sentence is abrupt.",
                        "recommendation": "Add one connective word.",
                    }
                ],
            }
        if (
            request.task_id == "script_revision"
            and request.input_data.get("revision_strategy")
            == "single-scene-replacement-v1"
        ):
            replacement_requests.append(request)
            return {"spoken_text": request.input_data["spoken_text"] + " Therefore."}
        return original_fixture(request)

    monkeypatch.setattr(deterministic_backend, "_fake_structured", scene_revision_fixture)
    frozen_assets = build_frozen_assets(config)
    frozen_assets["runtime_snapshot"] = build_runtime_snapshot(config)
    store = RunStore.create(
        project_root=tmp_path,
        config=config,
        brief=brief,
        frozen_assets=frozen_assets,
    )

    with WorkflowEngine(
        store=store,
        environment={},
        stop_after="script-revision",
    ) as workflow:
        assert workflow.run() is None

    draft_record = store.stage_record("script-draft")
    revision_record = store.stage_record("script-revision")
    assert draft_record is not None and revision_record is not None
    draft = store.load_artifact(draft_record, NarrationScript)
    revision = store.load_artifact(revision_record, RevisedScript)
    assert len(replacement_requests) == 1
    assert set(replacement_requests[0].output_schema["properties"]) == {"spoken_text"}
    assert revision_record.item_ids == ["replacement-scene-002"]
    for original_scene, revised_scene in zip(
        draft.scenes,
        revision.script.scenes,
        strict=True,
    ):
        assert revised_scene.scene_id == original_scene.scene_id
        assert revised_scene.pause_after_seconds == original_scene.pause_after_seconds
        if revised_scene.scene_id == "scene-002":
            assert revised_scene.spoken_text == original_scene.spoken_text + " Therefore."
        else:
            assert revised_scene.spoken_text == original_scene.spoken_text


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
    repair_requests = []
    extracted_scene_texts = []

    def unsupported_fixture(request):
        nonlocal selector_saw_research
        if request.task_id == "select":
            selector_saw_research = "research_pack" in request.input_data
        if request.task_id == "claim_inventory":
            extracted_scene_texts.append(request.input_data["spoken_text"])
        if (
            request.task_id == "script_revision"
            and request.input_data.get("repair_strategy") == "factual-claim-repair-v1"
        ):
            repair_requests.append(request)
        if request.task_id != "factual_review":
            return original_fixture(request)
        if request.input_data.get("review_strategy") == "single-claim-v1":
            return {
                "verdict": "unsupported",
                "evidence_ids": [],
                "rationale": "The bounded evidence does not directly support this wording.",
            }
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
    assert len(extracted_scene_texts) == len(repair_requests)


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="the deterministic end-to-end workflow requires FFmpeg and ffprobe",
)
def test_factual_gate_repairs_one_scene_with_text_only_output(
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
    repair_requests = []
    fit_requests = []
    extraction_requests = []
    review_requests = []

    def repairable_fixture(request):
        if request.task_id == "claim_inventory":
            extraction_requests.append(request)
        if (
            request.task_id == "script_revision"
            and request.input_data.get("revision_strategy")
            == "single-scene-word-fit-v1"
        ):
            fit_requests.append(request)
            return original_fixture(request)
        if (
            request.task_id == "script_revision"
            and request.input_data.get("repair_strategy") == "factual-claim-repair-v1"
        ):
            repair_requests.append(request)
            return {"spoken_text": "Verified."}
        if (
            request.task_id == "factual_review"
            and request.input_data.get("review_strategy") == "single-claim-v1"
        ):
            review_requests.append(request)
            claim = request.input_data["claim"]
            if (
                claim["scene_id"] == "scene-001"
                and not request.input_data["scene_spoken_text"].startswith("Verified")
            ):
                return {
                    "verdict": "needs_qualification",
                    "evidence_ids": claim.get("evidence_ids", []),
                    "rationale": "The wording needs to be narrowed to the bounded evidence.",
                }
            return {
                "verdict": "supported",
                "evidence_ids": claim.get("evidence_ids", []),
                "rationale": "The bounded evidence directly supports this wording.",
            }
        return original_fixture(request)

    monkeypatch.setattr(deterministic_backend, "_fake_structured", repairable_fixture)
    store = RunStore.create(
        project_root=tmp_path,
        config=config,
        brief=brief,
        frozen_assets=build_frozen_assets(config),
    )

    with WorkflowEngine(store=store, environment={}) as workflow:
        delivery = workflow.run()

    assert delivery is not None
    assert len(repair_requests) == 1
    repair_request = repair_requests[0]
    assert set(repair_request.output_schema["properties"]) == {"spoken_text"}
    assert "script" not in repair_request.input_data
    assert "scene_id" not in repair_request.input_data
    assert "required_word_count" not in repair_request.input_data
    assert "count_method" not in repair_request.input_data
    assert len(fit_requests) == 1
    fit_request = fit_requests[0]
    assert fit_request.input_data["spoken_text"] == "Verified."
    assert set(fit_request.output_schema["properties"]) == {"spoken_text"}
    assert "script" not in fit_request.input_data
    assert "scene_id" not in fit_request.input_data
    revision_record = store.stage_record("script-revision")
    assert revision_record is not None
    revision = store.load_artifact(revision_record, FactualRevisedScript)
    assert revision.factual_review.passed
    assert revision.script.scenes[0].spoken_text.startswith("Verified")
    total_words = sum(len(scene.spoken_text.split()) for scene in revision.script.scenes)
    aggregate_range = fit_request.input_data["aggregate_word_counts"]
    assert aggregate_range["minimum"] <= total_words <= aggregate_range["maximum"]
    scene_count = len(revision.script.scenes)
    assert len(extraction_requests) == scene_count + 1
    assert len(review_requests) == scene_count + 1
    assert extraction_requests[-1].input_data["spoken_text"].startswith("Verified")
    assert review_requests[-1].input_data["scene_spoken_text"].startswith("Verified")
    assert all(
        set(request.output_schema["properties"]) == {"claims"}
        for request in extraction_requests
    )
    initial_texts = [
        request.input_data["spoken_text"] for request in extraction_requests[:scene_count]
    ]
    assert extraction_requests[-1].input_data["spoken_text"] != initial_texts[0]
    assert [scene.spoken_text for scene in revision.script.scenes[1:]] == initial_texts[1:]
