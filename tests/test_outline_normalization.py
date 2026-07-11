from __future__ import annotations

from video_generator.contracts import OutlineScene, StoryOutline
from video_generator.workflow import WorkflowEngine


def test_outline_duration_weights_are_normalized_to_budget() -> None:
    outline = StoryOutline(
        title="Fixture",
        concept_summary="A short winter story.",
        scenes=[
            OutlineScene(
                scene_id=f"scene-{index:03d}",
                narrative_purpose="Advance the story.",
                change="The situation changes.",
                emotional_beat="Curiosity",
                visual_opportunity="A clear snowy action.",
                provisional_seconds=14,
            )
            for index in range(1, 9)
        ],
    )

    WorkflowEngine._normalize_outline_durations(outline, 120)

    assert sum(scene.provisional_seconds for scene in outline.scenes) == 120
    assert [scene.provisional_seconds for scene in outline.scenes] == [15] * 8


def test_outline_duration_normalization_respects_scene_bounds() -> None:
    raw_durations = [20, 20, 20, 20, 20, 15, 3, 2]
    outline = StoryOutline(
        title="Fixture",
        concept_summary="A short winter story.",
        scenes=[
            OutlineScene(
                scene_id=f"scene-{index:03d}",
                narrative_purpose="Advance the story.",
                change="The situation changes.",
                emotional_beat="Curiosity",
                visual_opportunity="A clear snowy action.",
                provisional_seconds=duration,
            )
            for index, duration in enumerate(raw_durations, start=1)
        ],
    )

    WorkflowEngine._normalize_outline_durations(
        outline,
        120,
        minimum_seconds=8,
        maximum_seconds=20,
    )

    durations = [scene.provisional_seconds for scene in outline.scenes]
    assert sum(durations) == 120
    assert 4 <= durations[0] <= 20
    assert all(8 <= duration <= 20 for duration in durations[1:-1])
    assert 4 <= durations[-1] <= 20
