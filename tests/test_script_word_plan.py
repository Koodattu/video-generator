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


@pytest.mark.parametrize(
    ("count", "expected_target"),
    [(6, 8), (14, 12)],
)
def test_script_word_range_rejects_under_and_over_length_scripts(
    count: int,
    expected_target: int,
) -> None:
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

    with pytest.raises(BackendError, match=f"has {count} words.*8-12") as caught:
        WorkflowEngine._validate_script_word_range(
            script,
            minimum_words=8,
            maximum_words=12,
        )

    assert caught.value.details == {
        "actual_word_count": count,
        "minimum_word_count": 8,
        "maximum_word_count": 12,
        "target_word_count": expected_target,
        "word_delta": expected_target - count,
        "count_method": "whitespace-separated words across every spoken_text field",
    }


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
    (
        "language",
        "backend_id",
        "expected_minimum",
        "expected_target",
        "expected_maximum",
        "expected_sentence_bounds",
    ),
    [
        (OutputLanguage.ENGLISH, "openai:fixture", 260, 291, 306, (2, 3)),
        (OutputLanguage.ENGLISH, "local:fixture", 199, 291, 306, (3, 4)),
        (OutputLanguage.FINNISH, "local:fixture", 64, 222, 246, (3, 4)),
    ],
)
def test_script_word_plan_minimum_matches_duration_acceptance(
    language: OutputLanguage,
    backend_id: str,
    expected_minimum: int,
    expected_target: int,
    expected_maximum: int,
    expected_sentence_bounds: tuple[int, int],
) -> None:
    engine = object.__new__(WorkflowEngine)
    engine.config = SimpleNamespace(
        output_language=language,
        duration_seconds=120,
        task_bindings={"script_draft": backend_id},
    )
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
    assert {
        (scene["minimum_sentence_count"], scene["maximum_sentence_count"])
        for scene in plan["scene_word_targets"]
    } == {expected_sentence_bounds}
    assert sum(
        int(scene["minimum_word_count"])
        for scene in plan["scene_word_targets"]
    ) == plan["minimum_total_word_count"]
    assert sum(
        int(scene["target_word_count"])
        for scene in plan["scene_word_targets"]
    ) == plan["target_total_word_count"]
    assert sum(
        int(scene["maximum_word_count"])
        for scene in plan["scene_word_targets"]
    ) == plan["maximum_total_word_count"]
    assert all(
        int(scene["minimum_word_count"])
        <= int(scene["target_word_count"])
        <= int(scene["maximum_word_count"])
        for scene in plan["scene_word_targets"]
    )


def test_policy_v3_local_draft_leaves_room_for_measured_audio_repair() -> None:
    engine = object.__new__(WorkflowEngine)
    engine.workflow_policy_version = 3
    engine.config = SimpleNamespace(
        output_language=OutputLanguage.ENGLISH,
        duration_seconds=32,
        task_bindings={"script_draft": "local:fixture"},
        narration_delivery_spec=SimpleNamespace(target_words_per_second=2.193),
    )
    outline = StoryOutline(
        title="Fixture",
        concept_summary="A short local story.",
        scenes=[
            OutlineScene(
                scene_id=f"scene-{index:03d}",
                narrative_purpose="Advance the story.",
                change="The situation changes.",
                emotional_beat="Curiosity",
                visual_opportunity="A clear action.",
                provisional_seconds=8,
            )
            for index in range(1, 5)
        ],
    )

    plan = engine._script_word_plan(outline)

    assert plan["minimum_total_word_count"] == 53
    assert plan["target_total_word_count"] == 67
    assert plan["maximum_total_word_count"] == 70


def test_policy_v10_local_draft_defers_more_duration_fit_to_measured_audio() -> None:
    engine = object.__new__(WorkflowEngine)
    engine.workflow_policy_version = 10
    engine.config = SimpleNamespace(
        output_language=OutputLanguage.ENGLISH,
        duration_seconds=32,
        task_bindings={"script_draft": "local:fixture"},
        narration_delivery_spec=SimpleNamespace(target_words_per_second=2.193),
    )
    outline = StoryOutline(
        title="Fixture",
        concept_summary="A short local story.",
        scenes=[
            OutlineScene(
                scene_id=f"scene-{index:03d}",
                narrative_purpose="Advance the story.",
                change="The situation changes.",
                emotional_beat="Curiosity",
                visual_opportunity="A clear action.",
                provisional_seconds=8,
            )
            for index in range(1, 5)
        ],
    )

    plan = engine._script_word_plan(outline)

    assert plan["minimum_total_word_count"] == 46
    assert plan["target_total_word_count"] == 67
    assert plan["maximum_total_word_count"] == 70


def test_policy_v11_local_draft_restores_strict_aggregate_envelope() -> None:
    engine = object.__new__(WorkflowEngine)
    engine.workflow_policy_version = 11
    engine.config = SimpleNamespace(
        output_language=OutputLanguage.ENGLISH,
        duration_seconds=32,
        task_bindings={"script_draft": "local:fixture"},
        narration_delivery_spec=SimpleNamespace(target_words_per_second=2.193),
    )
    outline = StoryOutline(
        title="Fixture",
        concept_summary="A short local story.",
        scenes=[
            OutlineScene(
                scene_id=f"scene-{index:03d}",
                narrative_purpose="Advance the story.",
                change="The situation changes.",
                emotional_beat="Curiosity",
                visual_opportunity="A clear action.",
                provisional_seconds=8,
            )
            for index in range(1, 5)
        ],
    )

    plan = engine._script_word_plan(outline)

    assert plan["minimum_total_word_count"] == 53
    assert plan["target_total_word_count"] == 67
    assert plan["maximum_total_word_count"] == 70
