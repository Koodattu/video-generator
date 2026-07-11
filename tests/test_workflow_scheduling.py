from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from video_generator.cli import _completed_call_counts
from video_generator.contracts import (
    CreativeBrief,
    OutputLanguage,
    Quality,
    StructuredTextResult,
    VisualReviewItem,
)
from video_generator.executor import StructuredExecution
from video_generator.profiles import PROFILES
from video_generator.prompting import build_frozen_assets
from video_generator.provenance import build_runtime_snapshot
from video_generator.run_store import RunStore
from video_generator.workflow import VisualReviewBundle, WorkflowEngine


pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="workflow scheduling tests require FFmpeg and ffprobe",
)


def _store(
    tmp_path: Path,
    resolved_config,
    *,
    quality: Quality,
    music_enabled: bool,
) -> RunStore:
    config = resolved_config.model_copy(
        update={
            "profile": "deterministic-test",
            "project_root": str(tmp_path),
            "task_bindings": dict(PROFILES["deterministic-test"]),
            "output_language": OutputLanguage.ENGLISH,
            "duration_seconds": 10,
            "quality": quality,
            "offline": True,
            "idea_candidates": 2,
            "research_query_limit": 0,
            "research_source_limit": 2,
            "visual_target_seconds": 5,
            "visual_min_seconds": 4,
            "visual_max_seconds": 8,
            "music_enabled": music_enabled,
            "captions_enabled": False,
            "animated_captions": False,
        }
    )
    frozen_assets = build_frozen_assets(config)
    frozen_assets["runtime_snapshot"] = build_runtime_snapshot(config)
    return RunStore.create(
        project_root=tmp_path,
        config=config,
        brief=CreativeBrief(idea_direction="A tiny mystery on a snowy path"),
        frozen_assets=frozen_assets,
    )


def _review_result(scene_id: str, *, passed: bool) -> StructuredExecution:
    item = VisualReviewItem(
        scene_id=scene_id,
        passed=passed,
        scores={
            "subject_action": 5,
            "style_match": 5,
            "identity": 5 if passed else 3,
            "composition": 5,
            "text_logo_free": 5,
            "audience_safety": 5,
        },
        failures=[] if passed else ["character identity drifted"],
        regeneration_instruction="" if passed else "Restore the red scarf and blue lantern.",
    )
    return StructuredExecution(
        artifact=item,
        result=StructuredTextResult(data=item.model_dump(mode="json")),
        prompt_version="test",
        schema_hash="test",
    )


def test_music_brief_preparation_requires_the_same_text_backend() -> None:
    workflow = object.__new__(WorkflowEngine)
    workflow.config = SimpleNamespace(
        music_enabled=True,
        task_bindings={
            "image_prompt_compile": "local:llama-server",
            "music_brief": "openai:gpt-5.4-mini",
        },
    )
    workflow.stop_after = None

    assert not workflow._should_prepare_music_brief()


def test_visual_review_batches_regeneration_before_second_pass(
    tmp_path: Path,
    resolved_config,
) -> None:
    store = _store(
        tmp_path,
        resolved_config,
        quality=Quality.FINAL,
        music_enabled=True,
    )
    events: list[str] = []
    first_pass_scenes: list[str] = []
    failed_scenes: list[str] = []

    with WorkflowEngine(store=store, environment={}, stop_after="visual-review") as workflow:
        original_structured = workflow.executor.structured
        original_image = workflow.executor.image

        def structured(task_id, input_data, output_model, **kwargs):
            if task_id != "visual_review":
                if task_id == "music_brief":
                    events.append("music-brief")
                return original_structured(task_id, input_data, output_model, **kwargs)
            scene_id = str(input_data["scene_id"])
            pass_number = int(input_data["pass_number"])
            events.append(f"review:{scene_id}:{pass_number}")
            if pass_number == 1:
                first_pass_scenes.append(scene_id)
                passed = len(first_pass_scenes) > 2
                if not passed:
                    failed_scenes.append(scene_id)
            else:
                passed = True
            return _review_result(scene_id, passed=passed)

        def image(request, output_path):
            if "Targeted correction." in request.prompt:
                events.append(f"regenerate:{request.scene_id}")
            return original_image(request, output_path)

        workflow.executor.structured = structured
        workflow.executor.image = image
        delivery = workflow.run()

    assert delivery is None
    assert len(failed_scenes) == 2
    relevant = [event for event in events if event.startswith(("review:", "regenerate:"))]
    assert relevant == [
        *(f"review:{scene_id}:1" for scene_id in first_pass_scenes),
        *(f"regenerate:{scene_id}" for scene_id in failed_scenes),
        *(f"review:{scene_id}:2" for scene_id in failed_scenes),
    ]
    record = store.stage_record("visual-review")
    assert record is not None and record.status == "complete"
    bundle = store.load_artifact(record, VisualReviewBundle)
    assert bundle.report is not None and bundle.report.pass_number == 2
    assert [item.generated.scene_id for item in bundle.images.items] == first_pass_scenes
    assert "music-brief" not in events
    assert store.stage_record("music-brief") is None
    assert store.completed_item_ids("music-brief") == []


def test_music_brief_is_prepared_in_text_batch_and_finalized_in_stage_order(
    tmp_path: Path,
    resolved_config,
) -> None:
    store = _store(
        tmp_path,
        resolved_config,
        quality=Quality.DRAFT,
        music_enabled=True,
    )
    events: list[str] = []

    with WorkflowEngine(store=store, environment={}, stop_after="music-brief") as workflow:
        original_structured = workflow.executor.structured
        original_images = workflow._images

        def structured(task_id, input_data, output_model, **kwargs):
            if task_id in {"image_prompt_compile", "music_brief"}:
                events.append(task_id)
            return original_structured(task_id, input_data, output_model, **kwargs)

        def images(requests):
            events.append("images")
            assert store.stage_record("music-brief") is None
            assert store.completed_item_ids("music-brief") == ["brief"]
            assert _completed_call_counts(store)["music_brief"] == 1
            return original_images(requests)

        workflow.executor.structured = structured
        workflow._images = images
        delivery = workflow.run()

    assert delivery is None
    assert events.count("music_brief") == 1
    assert max(index for index, event in enumerate(events) if event == "image_prompt_compile") < events.index(
        "music_brief"
    ) < events.index("images")
    record = store.stage_record("music-brief")
    assert record is not None and record.status == "complete"
    assert _completed_call_counts(store)["music_brief"] == 1
