from __future__ import annotations

import pytest

from types import SimpleNamespace

from video_generator.contracts import (
    NarrationScript,
    OutlineScene,
    OutputLanguage,
    ScriptScene,
    StoryOutline,
)
from video_generator.errors import BackendError
from video_generator.workflow import WorkflowEngine


@pytest.mark.parametrize("count", [6, 14])
def test_script_word_range_rejects_under_and_over_length_scripts(count: int) -> None:
    script = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(
                scene_id="scene-001",
                spoken_text=" ".join(["word"] * count),
                pause_after_seconds=0,
            )
        ],
    )

    with pytest.raises(BackendError, match=f"has {count} words.*8-12"):
        WorkflowEngine._validate_script_word_range(
            script,
            minimum_words=8,
            maximum_words=12,
        )


def test_script_word_range_accepts_inclusive_bounds() -> None:
    script = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(
                scene_id="scene-001",
                spoken_text="one two three four five six seven eight",
                pause_after_seconds=0,
            )
        ],
    )

    WorkflowEngine._validate_script_word_range(
        script,
        minimum_words=8,
        maximum_words=12,
    )


@pytest.mark.parametrize(
    ("language", "expected_minimum", "expected_target", "expected_maximum"),
    [
        (OutputLanguage.ENGLISH, 260, 291, 306),
        (OutputLanguage.FINNISH, 64, 222, 246),
    ],
)
def test_script_word_plan_minimum_matches_duration_acceptance(
    language: OutputLanguage,
    expected_minimum: int,
    expected_target: int,
    expected_maximum: int,
) -> None:
    engine = object.__new__(WorkflowEngine)
    engine.config = SimpleNamespace(output_language=language, duration_seconds=120)
    outline = StoryOutline(
        title="Fixture",
        concept_summary="A winter story.",
        scenes=[
            OutlineScene(
                scene_id=f"scene-{index:03d}",
                narrative_purpose="Advance the story.",
                change="The situation changes.",
                emotional_beat="Curiosity",
                visual_opportunity="A clear snowy action.",
                provisional_seconds=15,
            )
            for index in range(1, 9)
        ],
    )

    plan = engine._script_word_plan(outline)

    assert plan["minimum_total_word_count"] == expected_minimum
    assert plan["target_total_word_count"] == expected_target
    assert plan["maximum_total_word_count"] == expected_maximum
    expected_sentence_bounds = (2, 3) if language is OutputLanguage.ENGLISH else (3, 4)
    assert {
        (scene["minimum_sentence_count"], scene["maximum_sentence_count"])
        for scene in plan["scene_word_targets"]
    } == {expected_sentence_bounds}
