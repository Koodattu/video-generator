from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from video_generator.contracts import CreativeBrief, OutputLanguage
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
