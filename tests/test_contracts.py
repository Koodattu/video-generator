from __future__ import annotations

import pytest
from pydantic import ValidationError

from video_generator.contracts import MusicBrief, RenderPlan, RenderScene


def test_music_sections_must_cover_requested_duration() -> None:
    with pytest.raises(ValidationError, match="requested duration"):
        MusicBrief(
            prompt="quiet instrumental",
            requested_duration_seconds=30,
            tempo_range_bpm="60-70",
            instrumentation=["piano"],
            texture="sparse",
            exclusions=["lyrics"],
            sections=[{"start_seconds": 0, "end_seconds": 20, "mood": "warm", "energy": "low"}],
        )


def test_music_prompt_respects_local_backend_limit() -> None:
    with pytest.raises(ValidationError, match="at most 512 characters"):
        MusicBrief(
            prompt="x" * 513,
            requested_duration_seconds=30,
            tempo_range_bpm="60-70",
            instrumentation=["piano"],
            texture="sparse",
            exclusions=["lyrics"],
            sections=[
                {"start_seconds": 0, "end_seconds": 30, "mood": "warm", "energy": "low"}
            ],
        )


def test_render_scenes_must_be_contiguous() -> None:
    with pytest.raises(ValidationError, match="contiguous"):
        RenderPlan(
            scenes=[
                RenderScene(scene_id="scene-001", image_path="one.png", start_seconds=0, end_seconds=2),
                RenderScene(scene_id="scene-002", image_path="two.png", start_seconds=3, end_seconds=5),
            ],
            narration_path="narration.wav",
            width=1280,
            height=720,
            fps=30,
            duration_seconds=5,
        )
