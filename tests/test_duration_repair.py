from __future__ import annotations

import math

import pytest

from video_generator.contracts import (
    MediaReference,
    NarrationScript,
    NarrationTimeline,
    RevisedScript,
    ScriptScene,
    TimelineScene,
)
from video_generator.errors import BackendError
from video_generator.media import duration_is_accepted
from video_generator.workflow import WorkflowEngine


def test_duration_repair_rejects_a_no_op_word_count() -> None:
    original = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(
                scene_id="scene-001",
                spoken_text=" ".join(["one"] * 50),
                pause_after_seconds=1.5,
            ),
            ScriptScene(
                scene_id="scene-002",
                spoken_text=" ".join(["two"] * 50),
                pause_after_seconds=0,
            ),
        ],
    )
    revision = RevisedScript(script=original, dispositions=[])

    targets = [
        {
            "scene_id": "scene-001",
            "original_word_count": 50,
            "target_word_count": 38,
            "minimum_word_count": 36,
            "maximum_word_count": 40,
        },
        {
            "scene_id": "scene-002",
            "original_word_count": 50,
            "target_word_count": 38,
            "minimum_word_count": 36,
            "maximum_word_count": 40,
        },
    ]

    with pytest.raises(BackendError, match="scene-001 got 50.*remove 10-14 words"):
        WorkflowEngine._validate_duration_revision(
            revision,
            original,
            {"scene-001", "scene-002"},
            scene_repair_targets=targets,
        )


def test_duration_repair_rejects_per_scene_violation_when_total_passes() -> None:
    original = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(scene_id="scene-001", spoken_text="one two three four", pause_after_seconds=1),
            ScriptScene(scene_id="scene-002", spoken_text="five six seven eight", pause_after_seconds=0),
        ],
    )
    revision = RevisedScript(
        script=NarrationScript(
            title="Fixture",
            scenes=[
                ScriptScene(scene_id="scene-001", spoken_text="one", pause_after_seconds=1),
                ScriptScene(
                    scene_id="scene-002",
                    spoken_text="two three four five six seven eight",
                    pause_after_seconds=0,
                ),
            ],
        ),
        dispositions=[],
    )
    targets = [
        {
            "scene_id": scene_id,
            "original_word_count": 4,
            "target_word_count": 4,
            "minimum_word_count": 3,
            "maximum_word_count": 5,
        }
        for scene_id in ("scene-001", "scene-002")
    ]

    with pytest.raises(BackendError, match="scene-001 got 1.*scene-002 got 7.*redistribute"):
        WorkflowEngine._validate_duration_revision(
            revision,
            original,
            {"scene-001", "scene-002"},
            scene_repair_targets=targets,
        )


def test_duration_repair_allows_one_word_scene_drift_when_total_passes() -> None:
    original = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(scene_id="scene-001", spoken_text="one two three four", pause_after_seconds=1),
            ScriptScene(scene_id="scene-002", spoken_text="five six seven eight", pause_after_seconds=0),
        ],
    )
    revision = RevisedScript(
        script=NarrationScript(
            title="Fixture",
            scenes=[
                ScriptScene(scene_id="scene-001", spoken_text="one two", pause_after_seconds=1),
                ScriptScene(
                    scene_id="scene-002",
                    spoken_text="three four five six seven eight",
                    pause_after_seconds=0,
                ),
            ],
        ),
        dispositions=[],
    )
    targets = [
        {
            "scene_id": scene_id,
            "original_word_count": 4,
            "target_word_count": 4,
            "minimum_word_count": 3,
            "maximum_word_count": 5,
        }
        for scene_id in ("scene-001", "scene-002")
    ]

    WorkflowEngine._validate_duration_revision(
        revision,
        original,
        {"scene-001", "scene-002"},
        scene_repair_targets=targets,
    )


def test_duration_repair_allows_one_word_aggregate_drift() -> None:
    original = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(
                scene_id="scene-001",
                spoken_text="one two three four five six",
                pause_after_seconds=0,
            )
        ],
    )
    revision = RevisedScript(
        script=NarrationScript(
            title="Fixture",
            scenes=[
                ScriptScene(
                    scene_id="scene-001",
                    spoken_text="one two three four",
                    pause_after_seconds=0,
                )
            ],
        ),
        dispositions=[],
    )

    WorkflowEngine._validate_duration_revision(
        revision,
        original,
        {"scene-001"},
        scene_repair_targets=[
            {
                "scene_id": "scene-001",
                "original_word_count": 6,
                "target_word_count": 7,
                "minimum_word_count": 5,
                "maximum_word_count": 9,
            }
        ],
    )


def test_duration_repair_scale_uses_midpoint_speech_window_without_fixed_pause() -> None:
    audio = MediaReference(path="fixture.wav", sha256="0" * 64, mime_type="audio/wav")
    timeline = NarrationTimeline(
        narration_audio=audio,
        duration_seconds=32.56,
        delivery_duration_seconds=32.5666666667,
        scenes=[
            TimelineScene(
                scene_id="scene-001",
                audio=audio,
                start_seconds=0,
                speech_end_seconds=14.24,
                end_seconds=16.24,
            ),
            TimelineScene(
                scene_id="scene-002",
                audio=audio,
                start_seconds=16.24,
                speech_end_seconds=32.56,
                end_seconds=32.56,
            ),
        ],
    )

    scale = WorkflowEngine._duration_repair_scale(
        timeline,
        target_seconds=28.5,
        selected_scene_ids={"scene-001", "scene-002"},
    )

    assert scale == pytest.approx((28.5 - 2.0) / (14.24 + 16.32))
    assert round(47 * scale) == 41
    assert round(55 * scale) == 48


def test_duration_acceptance_keeps_an_exact_ceiling_with_an_eighty_five_percent_floor() -> None:
    audio = MediaReference(path="fixture.wav", sha256="0" * 64, mime_type="audio/wav")

    def timeline(duration: float) -> NarrationTimeline:
        return NarrationTimeline(
            narration_audio=audio,
            duration_seconds=duration,
            delivery_duration_seconds=math.ceil(duration * 30) / 30,
            scenes=[
                TimelineScene(
                    scene_id="scene-001",
                    audio=audio,
                    start_seconds=0,
                    speech_end_seconds=duration,
                    end_seconds=duration,
                )
            ],
        )

    assert duration_is_accepted(timeline(26.16), 30)
    assert not duration_is_accepted(timeline(25.4), 30)
    assert not duration_is_accepted(timeline(30.04), 30)
