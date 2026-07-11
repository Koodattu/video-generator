from __future__ import annotations

import pytest

from video_generator.contracts import NarrationScript, ScriptScene
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
