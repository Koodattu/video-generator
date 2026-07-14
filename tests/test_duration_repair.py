from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from video_generator.contracts import (
    ContentMode,
    MediaReference,
    NarrationDeliverySpec,
    NarrationScript,
    NarrationTimeline,
    OutputLanguage,
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


def test_scene_local_duration_repair_returns_only_replacement_text() -> None:
    audio = MediaReference(path="fixture.wav", sha256="0" * 64, mime_type="audio/wav")
    script = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(
                scene_id=f"scene-{index:03d}",
                spoken_text="one two three four five six seven eight nine ten",
                pause_after_seconds=0.15 if index == 1 else 0,
            )
            for index in range(1, 3)
        ],
    )
    timeline = NarrationTimeline(
        narration_audio=audio,
        duration_seconds=10,
        delivery_duration_seconds=10,
        scenes=[
            TimelineScene(
                scene_id=f"scene-{index:03d}",
                audio=audio,
                start_seconds=(index - 1) * 5,
                speech_end_seconds=index * 5 - (0.15 if index == 1 else 0),
                end_seconds=index * 5,
            )
            for index in range(1, 3)
        ],
    )
    targets = [
        {
            "scene_id": f"scene-{index:03d}",
            "original_word_count": 10,
            "target_word_count": 8,
            "minimum_word_count": 7,
            "maximum_word_count": 9,
        }
        for index in range(1, 3)
    ]
    engine = object.__new__(WorkflowEngine)
    engine.config = SimpleNamespace(
        content_mode=ContentMode.FICTION,
        output_language=OutputLanguage.ENGLISH,
    )
    requests = []

    def structured_item(**kwargs):
        requests.append(kwargs)
        words = kwargs["input_data"]["spoken_text"].split()
        replacement = kwargs["output_model"](
            spoken_text=" ".join(words[: kwargs["input_data"]["target_word_count"]]) + "."
        )
        kwargs["invariant"](replacement)
        return replacement, []

    engine._structured_item = structured_item

    repaired, usage, response = engine._repair_duration_by_scene(
        script=script,
        measured_timeline=timeline,
        duration_scale=0.8,
        scene_repair_targets=targets,
        selected_scene_ids={"scene-001", "scene-002"},
        factual_research=None,
    )

    assert usage == []
    assert response["repair_strategy"] == "per-scene-text-v3"
    assert len(requests) == 2
    assert all(
        set(request["output_model"].model_json_schema()["properties"])
        == {"spoken_text"}
        for request in requests
    )
    assert all("script" not in request["input_data"] for request in requests)
    assert [scene.scene_id for scene in repaired.script.scenes] == [
        "scene-001",
        "scene-002",
    ]
    assert [scene.pause_after_seconds for scene in repaired.script.scenes] == [0.15, 0]
    assert all(len(scene.spoken_text.split()) == 8 for scene in repaired.script.scenes)


def test_policy_v11_duration_repair_allows_scene_counts_to_compensate() -> None:
    audio = MediaReference(path="fixture.wav", sha256="0" * 64, mime_type="audio/wav")
    script = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(
                scene_id=f"scene-{index:03d}",
                spoken_text="one two three four five six seven eight nine ten",
                pause_after_seconds=0.15 if index == 1 else 0,
            )
            for index in range(1, 3)
        ],
    )
    timeline = NarrationTimeline(
        narration_audio=audio,
        duration_seconds=10,
        delivery_duration_seconds=10,
        scenes=[
            TimelineScene(
                scene_id=f"scene-{index:03d}",
                audio=audio,
                start_seconds=(index - 1) * 5,
                speech_end_seconds=index * 5 - (0.15 if index == 1 else 0),
                end_seconds=index * 5,
            )
            for index in range(1, 3)
        ],
    )
    targets = [
        {
            "scene_id": f"scene-{index:03d}",
            "original_word_count": 10,
            "target_word_count": 8,
            "minimum_word_count": 7,
            "maximum_word_count": 9,
        }
        for index in range(1, 3)
    ]
    engine = object.__new__(WorkflowEngine)
    engine.workflow_policy_version = 11
    engine.config = SimpleNamespace(
        content_mode=ContentMode.FICTION,
        output_language=OutputLanguage.ENGLISH,
    )
    requests = []

    def structured_item(**kwargs):
        requests.append(kwargs)
        count = 5 if len(requests) == 1 else 9
        replacement = kwargs["output_model"](spoken_text=" ".join(["word"] * count))
        kwargs["invariant"](replacement)
        return replacement, []

    engine._structured_item = structured_item

    repaired, usage, response = engine._repair_duration_by_scene(
        script=script,
        measured_timeline=timeline,
        duration_scale=0.8,
        scene_repair_targets=targets,
        selected_scene_ids={"scene-001", "scene-002"},
        factual_research=None,
    )

    assert usage == []
    assert response["repair_strategy"] == "per-scene-text-v4-host-aggregate-fit"
    assert response["word_fit_items"] == []
    assert len(requests) == 2
    assert [len(scene.spoken_text.split()) for scene in repaired.script.scenes] == [5, 9]
    assert [scene.scene_id for scene in repaired.script.scenes] == ["scene-001", "scene-002"]
    assert [scene.pause_after_seconds for scene in repaired.script.scenes] == [0.15, 0]


def test_policy_v11_duration_repair_fits_only_one_scene_when_aggregate_is_short() -> None:
    audio = MediaReference(path="fixture.wav", sha256="0" * 64, mime_type="audio/wav")
    script = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(
                scene_id=f"scene-{index:03d}",
                spoken_text="one two three four five six seven eight nine ten",
                pause_after_seconds=0.15 if index == 1 else 0,
            )
            for index in range(1, 3)
        ],
    )
    timeline = NarrationTimeline(
        narration_audio=audio,
        duration_seconds=10,
        delivery_duration_seconds=10,
        scenes=[
            TimelineScene(
                scene_id=f"scene-{index:03d}",
                audio=audio,
                start_seconds=(index - 1) * 5,
                speech_end_seconds=index * 5 - (0.15 if index == 1 else 0),
                end_seconds=index * 5,
            )
            for index in range(1, 3)
        ],
    )
    targets = [
        {
            "scene_id": f"scene-{index:03d}",
            "original_word_count": 10,
            "target_word_count": 8,
            "minimum_word_count": 7,
            "maximum_word_count": 9,
        }
        for index in range(1, 3)
    ]
    engine = object.__new__(WorkflowEngine)
    engine.workflow_policy_version = 11
    engine.config = SimpleNamespace(
        content_mode=ContentMode.FICTION,
        output_language=OutputLanguage.ENGLISH,
    )
    requests = []

    def structured_item(**kwargs):
        requests.append(kwargs)
        input_data = kwargs["input_data"]
        count = (
            int(input_data["target_word_count"])
            if input_data["repair_strategy"] == "single-scene-word-fit-v1"
            else 5
        )
        replacement = kwargs["output_model"](spoken_text=" ".join(["word"] * count))
        kwargs["invariant"](replacement)
        return replacement, []

    engine._structured_item = structured_item

    repaired, usage, response = engine._repair_duration_by_scene(
        script=script,
        measured_timeline=timeline,
        duration_scale=0.8,
        scene_repair_targets=targets,
        selected_scene_ids={"scene-001", "scene-002"},
        factual_research=None,
    )

    assert usage == []
    assert len(requests) == 3
    fit_request = requests[-1]
    assert fit_request["input_data"]["repair_strategy"] == "single-scene-word-fit-v1"
    assert set(fit_request["output_model"].model_json_schema()["properties"]) == {
        "spoken_text"
    }
    assert "script" not in fit_request["input_data"]
    assert response["word_fit_items"] == [
        {"scene_id": "scene-001", "item_id": "duration-word-fit-scene-001"}
    ]
    counts = [len(scene.spoken_text.split()) for scene in repaired.script.scenes]
    assert 14 <= sum(counts) <= 18
    assert counts[1] == 5
    assert [scene.scene_id for scene in repaired.script.scenes] == ["scene-001", "scene-002"]
    assert [scene.pause_after_seconds for scene in repaired.script.scenes] == [0.15, 0]


def test_post_factual_duration_fit_uses_distinct_item_and_can_edit_second_scene() -> None:
    original = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(
                scene_id=f"scene-{index:03d}",
                spoken_text="one two three four five six seven eight nine ten",
                pause_after_seconds=0.15 if index == 1 else 0,
            )
            for index in range(1, 3)
        ],
    )
    repaired = NarrationScript(
        title="Fixture",
        scenes=[
            original.scenes[0].model_copy(
                update={"spoken_text": "one two three four five six seven"}
            ),
            original.scenes[1].model_copy(update={"spoken_text": "Verified."}),
        ],
    )
    targets = [
        {
            "scene_id": f"scene-{index:03d}",
            "original_word_count": 10,
            "target_word_count": 8,
            "minimum_word_count": 7,
            "maximum_word_count": 9,
        }
        for index in range(1, 3)
    ]
    engine = object.__new__(WorkflowEngine)
    engine.config = SimpleNamespace(
        content_mode=ContentMode.FACTUAL,
        output_language=OutputLanguage.ENGLISH,
    )
    requests = []

    def structured_item(**kwargs):
        requests.append(kwargs)
        count = int(kwargs["input_data"]["target_word_count"])
        replacement = kwargs["output_model"](spoken_text=" ".join(["word"] * count))
        kwargs["invariant"](replacement)
        return replacement, []

    engine._structured_item = structured_item

    fitted, usage, fit_items = engine._fit_duration_repair_word_range(
        script=repaired,
        original=original,
        scene_repair_targets=targets,
        selected_scene_ids={"scene-001", "scene-002"},
        factual_research=None,
        outline=None,
        item_prefix="duration-factual-word-fit",
    )

    assert usage == []
    assert len(requests) == 1
    request = requests[0]
    assert request["item_id"] == "duration-factual-word-fit-scene-002"
    assert request["input_data"]["spoken_text"] == "Verified."
    assert set(request["output_model"].model_json_schema()["properties"]) == {
        "spoken_text"
    }
    assert fit_items == [
        {
            "scene_id": "scene-002",
            "item_id": "duration-factual-word-fit-scene-002",
        }
    ]
    counts = [len(scene.spoken_text.split()) for scene in fitted.scenes]
    assert counts[0] == 7
    assert 14 <= sum(counts) <= 18


def test_pause_fit_removes_only_excess_silence_when_speech_fits_budget() -> None:
    audio = MediaReference(path="fixture.wav", sha256="0" * 64, mime_type="audio/wav")
    script = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(
                scene_id="scene-001",
                spoken_text="The first sentence remains unchanged.",
                pause_after_seconds=2,
            ),
            ScriptScene(
                scene_id="scene-002",
                spoken_text="The second sentence remains unchanged.",
                pause_after_seconds=0,
            ),
        ],
    )
    timeline = NarrationTimeline(
        narration_audio=audio,
        duration_seconds=121,
        delivery_duration_seconds=121,
        scenes=[
            TimelineScene(
                scene_id="scene-001",
                audio=audio,
                start_seconds=0,
                speech_end_seconds=60,
                end_seconds=62,
            ),
            TimelineScene(
                scene_id="scene-002",
                audio=audio,
                start_seconds=62,
                speech_end_seconds=121,
                end_seconds=121,
            ),
        ],
    )

    fitted = WorkflowEngine._fit_pauses_to_budget(script, timeline, 120)

    assert fitted is not None
    assert [scene.spoken_text for scene in fitted.scenes] == [
        scene.spoken_text for scene in script.scenes
    ]
    assert sum(scene.pause_after_seconds for scene in fitted.scenes) == pytest.approx(1, abs=0.001)


def test_pause_fit_defers_to_script_repair_when_speech_exceeds_budget() -> None:
    audio = MediaReference(path="fixture.wav", sha256="0" * 64, mime_type="audio/wav")
    script = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(
                scene_id="scene-001",
                spoken_text="The first sentence remains unchanged.",
                pause_after_seconds=1,
            ),
            ScriptScene(
                scene_id="scene-002",
                spoken_text="The second sentence remains unchanged.",
                pause_after_seconds=0,
            ),
        ],
    )
    timeline = NarrationTimeline(
        narration_audio=audio,
        duration_seconds=122,
        delivery_duration_seconds=122,
        scenes=[
            TimelineScene(
                scene_id="scene-001",
                audio=audio,
                start_seconds=0,
                speech_end_seconds=61,
                end_seconds=62,
            ),
            TimelineScene(
                scene_id="scene-002",
                audio=audio,
                start_seconds=62,
                speech_end_seconds=122,
                end_seconds=122,
            ),
        ],
    )

    assert WorkflowEngine._fit_pauses_to_budget(script, timeline, 120) is None


def test_pause_fit_does_not_pad_short_narration_with_silence() -> None:
    audio = MediaReference(path="fixture.wav", sha256="0" * 64, mime_type="audio/wav")
    script = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(
                scene_id=f"scene-{index:03d}",
                spoken_text=f"Scene {index} remains unchanged.",
                pause_after_seconds=0.5 if index < 5 else 0,
            )
            for index in range(1, 6)
        ],
    )
    spans = [
        (0, 18, 18.5),
        (18.5, 36.5, 37),
        (37, 55, 55.5),
        (55.5, 73.5, 74),
        (74, 92, 92),
    ]
    timeline = NarrationTimeline(
        narration_audio=audio,
        duration_seconds=92,
        delivery_duration_seconds=92,
        scenes=[
            TimelineScene(
                scene_id=f"scene-{index:03d}",
                audio=audio,
                start_seconds=start,
                speech_end_seconds=speech_end,
                end_seconds=end,
            )
            for index, (start, speech_end, end) in enumerate(spans, start=1)
        ],
    )

    assert WorkflowEngine._fit_pauses_to_budget(script, timeline, 120) is None


def test_legacy_pause_fit_can_expand_silence_for_resume_compatibility() -> None:
    audio = MediaReference(path="fixture.wav", sha256="0" * 64, mime_type="audio/wav")
    script = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(
                scene_id=f"scene-{index:03d}",
                spoken_text=f"Scene {index} remains unchanged.",
                pause_after_seconds=0.5 if index < 5 else 0,
            )
            for index in range(1, 6)
        ],
    )
    spans = [
        (0, 18, 18.5),
        (18.5, 36.5, 37),
        (37, 55, 55.5),
        (55.5, 73.5, 74),
        (74, 92, 92),
    ]
    timeline = NarrationTimeline(
        narration_audio=audio,
        duration_seconds=92,
        delivery_duration_seconds=92,
        scenes=[
            TimelineScene(
                scene_id=f"scene-{index:03d}",
                audio=audio,
                start_seconds=start,
                speech_end_seconds=speech_end,
                end_seconds=end,
            )
            for index, (start, speech_end, end) in enumerate(spans, start=1)
        ],
    )

    fitted = WorkflowEngine._fit_pauses_to_budget(
        script,
        timeline,
        120,
        allow_expansion=True,
    )

    assert fitted is not None
    assert sum(scene.pause_after_seconds for scene in fitted.scenes) == pytest.approx(13, abs=0.001)
    assert all(scene.pause_after_seconds <= 3.25 for scene in fitted.scenes)


def test_legacy_pause_fit_preserves_exact_over_budget_rounding() -> None:
    audio = MediaReference(path="fixture.wav", sha256="0" * 64, mime_type="audio/wav")
    pauses = [2.9, 1.3, 2.3, 2.4, 2.4, 2.5, 0]
    script = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(
                scene_id=f"scene-{index:03d}",
                spoken_text=f"Scene {index} remains unchanged.",
                pause_after_seconds=pause,
            )
            for index, pause in enumerate(pauses, start=1)
        ],
    )
    cursor = 0.0
    speech_duration = 45.632 / len(pauses)
    timeline_scenes = []
    for index, pause in enumerate(pauses, start=1):
        start = cursor
        speech_end = start + speech_duration
        cursor = speech_end + pause
        timeline_scenes.append(
            TimelineScene(
                scene_id=f"scene-{index:03d}",
                audio=audio,
                start_seconds=start,
                speech_end_seconds=speech_end,
                end_seconds=cursor,
            )
        )
    timeline = NarrationTimeline(
        narration_audio=audio,
        duration_seconds=cursor,
        delivery_duration_seconds=cursor,
        scenes=timeline_scenes,
    )

    fitted = WorkflowEngine._fit_pauses_to_budget(
        script,
        timeline,
        51,
        allow_expansion=True,
    )

    assert fitted is not None
    assert [scene.pause_after_seconds for scene in fitted.scenes] == [
        1.1282,
        0.5057,
        0.8946,
        0.9335,
        0.9335,
        0.9724,
        0,
    ]


def test_legacy_tempo_fit_preserves_resume_calculation() -> None:
    assert WorkflowEngine._legacy_tempo_fit_rate(
        speech_seconds=88,
        scene_count=5,
        budget_seconds=120,
    ) == pytest.approx(88 / 95)


def test_legacy_narration_item_fit_selects_legacy_rate(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = object.__new__(WorkflowEngine)
    engine.continuity_policy_enabled = False
    calls = []

    def legacy_rate(**kwargs):
        calls.append(kwargs)
        return None

    def current_rate(**kwargs):
        raise AssertionError("current pacing policy should not run for a legacy resume")

    monkeypatch.setattr(engine, "_legacy_tempo_fit_rate", legacy_rate)
    monkeypatch.setattr(engine, "_tempo_fit_rate", current_rate)

    adjusted, tempo = engine._tempo_fit_narration_items(
        [SimpleNamespace(normalized_duration_seconds=88.0)],
        pause_seconds=4,
        budget_seconds=120,
        output_root=tmp_path,
    )

    assert adjusted is None
    assert tempo is None
    assert calls == [{"speech_seconds": 88.0, "scene_count": 1, "budget_seconds": 120}]


def test_new_scripts_reject_long_authored_pauses() -> None:
    script = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(
                scene_id="scene-001",
                spoken_text="The first event changes the situation.",
                pause_after_seconds=1.2,
            ),
            ScriptScene(
                scene_id="scene-002",
                spoken_text="The second event resolves it.",
                pause_after_seconds=0,
            ),
        ],
    )

    with pytest.raises(BackendError, match="0.75-second production maximum"):
        WorkflowEngine._validate_authored_pauses(script)


def test_duration_lengthening_repairs_each_scene_independently() -> None:
    audio = MediaReference(path="fixture.wav", sha256="0" * 64, mime_type="audio/wav")
    script = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(
                scene_id="scene-001",
                spoken_text=" ".join(["one"] * 10),
                pause_after_seconds=1,
            ),
            ScriptScene(
                scene_id="scene-002",
                spoken_text=" ".join(["two"] * 10),
                pause_after_seconds=0,
            ),
        ],
    )
    timeline = NarrationTimeline(
        narration_audio=audio,
        duration_seconds=21,
        delivery_duration_seconds=21,
        scenes=[
            TimelineScene(
                scene_id="scene-001",
                audio=audio,
                start_seconds=0,
                speech_end_seconds=10,
                end_seconds=11,
            ),
            TimelineScene(
                scene_id="scene-002",
                audio=audio,
                start_seconds=11,
                speech_end_seconds=21,
                end_seconds=21,
            ),
        ],
    )
    targets = [
        {
            "scene_id": scene_id,
            "original_word_count": 10,
            "target_word_count": 15,
            "minimum_word_count": 13,
            "maximum_word_count": 17,
            "minimum_word_delta": 3,
            "target_word_delta": 5,
            "maximum_word_delta": 7,
        }
        for scene_id in ("scene-001", "scene-002")
    ]

    class Executor:
        def __init__(self) -> None:
            self.requests = []

        def structured(
            self,
            task_id,
            input_data,
            output_model,
            *,
            invariant,
            instruction_suffix,
        ):
            self.requests.append(input_data)
            assert "whitespace-separated words" in instruction_suffix
            target_words = int(input_data["scene_repair_targets"][0]["target_word_count"])
            artifact = output_model(
                scene_id="scene-001",
                spoken_text=" ".join(["word"] * target_words),
            )
            invariant(artifact)
            return SimpleNamespace(
                artifact=artifact,
                result=SimpleNamespace(usage=None, raw_response={"target_words": target_words}),
            )

    engine = object.__new__(WorkflowEngine)
    engine.config = SimpleNamespace(output_language=OutputLanguage.FINNISH)
    engine.executor = Executor()

    revision, usage, responses = engine._lengthen_duration_by_scene(
        script=script,
        measured_timeline=timeline,
        duration_scale=1.5,
        scene_repair_targets=targets,
        selected_scene_ids={"scene-001", "scene-002"},
    )

    assert len(engine.executor.requests) == 2
    assert [scene.scene_id for scene in revision.script.scenes] == ["scene-001", "scene-002"]
    assert [len(scene.spoken_text.split()) for scene in revision.script.scenes] == [15, 15]
    assert usage == []
    assert set(responses) == {"scene-001", "scene-002"}


def test_tempo_fit_uses_small_pitch_preserving_slowdown() -> None:
    tempo = WorkflowEngine._tempo_fit_rate(
        speech_seconds=92,
        pause_seconds=4,
        budget_seconds=120,
    )

    assert tempo == pytest.approx(92 / (108 - 4))
    assert 0.88 < tempo < 0.89


def test_tempo_fit_uses_small_pitch_preserving_speedup() -> None:
    tempo = WorkflowEngine._tempo_fit_rate(
        speech_seconds=29.6,
        pause_seconds=0.75,
        budget_seconds=28,
    )

    assert tempo == pytest.approx(29.6 / (28 * 0.98 - 0.75))
    assert 1.10 < tempo < 1.15


def test_tempo_fit_rejects_large_speedup() -> None:
    assert (
        WorkflowEngine._tempo_fit_rate(
            speech_seconds=35,
            pause_seconds=1,
            budget_seconds=28,
        )
        is None
    )


def test_tempo_fit_rejects_large_slowdown() -> None:
    assert (
        WorkflowEngine._tempo_fit_rate(
            speech_seconds=60,
            pause_seconds=4,
            budget_seconds=120,
        )
        is None
    )


def test_tempo_fit_uses_minimum_rate_when_it_reaches_accepted_floor() -> None:
    assert WorkflowEngine._tempo_fit_rate(
        speech_seconds=84,
        pause_seconds=4,
        budget_seconds=120,
    ) == pytest.approx(0.85)


def test_delivery_tempo_fit_corrects_small_rate_shortfall() -> None:
    tempo = WorkflowEngine._delivery_tempo_fit_rate(
        achieved_words_per_second=2.020,
        minimum_words_per_second=2.025,
        maximum_words_per_second=2.577,
    )

    assert tempo == pytest.approx((2.025 * 1.002) / 2.020)
    assert 1.0 < tempo < 1.01


def test_delivery_tempo_fit_leaves_in_range_delivery_unchanged() -> None:
    assert (
        WorkflowEngine._delivery_tempo_fit_rate(
            achieved_words_per_second=2.2,
            minimum_words_per_second=2.025,
            maximum_words_per_second=2.577,
        )
        is None
    )


def test_delivery_tempo_fit_rejects_large_rate_correction() -> None:
    assert (
        WorkflowEngine._delivery_tempo_fit_rate(
            achieved_words_per_second=1.5,
            minimum_words_per_second=2.025,
            maximum_words_per_second=2.577,
        )
        is None
    )


def test_policy_twenty_four_delivery_word_range_reserves_pauses_and_unselected_words() -> None:
    script = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(
                scene_id="scene-001",
                spoken_text=" ".join(["selected"] * 50),
                pause_after_seconds=0.24,
            ),
            ScriptScene(
                scene_id="scene-002",
                spoken_text=" ".join(["fixed"] * 10),
                pause_after_seconds=0,
            ),
        ],
    )
    engine = object.__new__(WorkflowEngine)
    engine.workflow_policy_version = 24
    engine.config = SimpleNamespace(
        duration_seconds=24,
        fps=30,
        narration_delivery_spec=NarrationDeliverySpec(
            target_words_per_second=2.301,
            minimum_words_per_second=2.025,
            maximum_words_per_second=2.577,
            target_pause_seconds=0.08,
            maximum_pause_seconds=0.75,
        ),
    )

    aggregate_range = engine._duration_repair_aggregate_word_range(
        script=script,
        scene_repair_targets=[
            {
                "scene_id": "scene-001",
                "minimum_word_count": 45,
                "target_word_count": 50,
                "maximum_word_count": 60,
            }
        ],
        selected_scene_ids={"scene-001"},
    )

    assert aggregate_range == (45, 50, 51)


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
