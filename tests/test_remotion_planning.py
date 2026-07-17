from __future__ import annotations

from types import SimpleNamespace

import pytest

from video_generator.contracts import (
    AnchoredWord,
    ContentFormat,
    ContentMode,
    OutputLanguage,
    RemotionAssetKind,
    RemotionAttentionLevel,
    RemotionBeatFunction,
    RemotionEditPlan,
    RemotionEditShot,
    RemotionMotionPreset,
    RemotionRhythmBeat,
    RemotionRhythmPlan,
    RemotionSfxPreset,
    RemotionTemplate,
    RemotionTransitionPreset,
)
from video_generator.remotion_planning import assess_remotion_edit_plan_quality
from video_generator.remotion_renderer import remotion_motion_for_template
from video_generator.workflow import WorkflowEngine


def _shot_content(
    template: RemotionTemplate,
) -> tuple[list[str], RemotionAssetKind, str, list[str]]:
    if template is RemotionTemplate.CODE_REVEAL:
        return ["const value = 1", "return value"], RemotionAssetKind.NONE, "", []
    if template is RemotionTemplate.DIAGRAM_FLOW:
        return ["Input", "Output"], RemotionAssetKind.NONE, "", []
    if template is RemotionTemplate.COMPARISON_SPLIT:
        return ["Before", "After"], RemotionAssetKind.NONE, "", []
    if template is RemotionTemplate.SOURCE_SCREENSHOT:
        return [], RemotionAssetKind.SOURCE_SCREENSHOT, "", ["source-001"]
    if template is RemotionTemplate.MEME_CUTAWAY:
        return [], RemotionAssetKind.MEME, "surprised server reaction", []
    return [], RemotionAssetKind.NONE, "", []


def _plan(
    templates: list[RemotionTemplate],
    *,
    durations: list[float] | None = None,
    evidence_shot: int | None = None,
    section_shot: int | None = None,
    transition_shot: int | None = None,
    functions: list[RemotionBeatFunction] | None = None,
    attentions: list[RemotionAttentionLevel] | None = None,
) -> RemotionEditPlan:
    durations = durations or [3.0] * len(templates)
    beats = []
    words = []
    shots = []
    cursor = 0.0
    for index, (template, duration) in enumerate(zip(templates, durations, strict=True), start=1):
        shot_id = f"shot-{index:03d}"
        beat_id = f"beat-{index:03d}"
        word_id = f"word-{index:06d}"
        scene_id = f"scene-{index:03d}"
        function = (
            functions[index - 1]
            if functions is not None
            else RemotionBeatFunction.HOOK
            if index == 1
            else RemotionBeatFunction.LANDING
            if index == len(templates)
            else RemotionBeatFunction.EVIDENCE
            if index == evidence_shot
            else RemotionBeatFunction.EXPLANATION
        )
        beats.append(
            RemotionRhythmBeat(
                beat_id=beat_id,
                shot_id=shot_id,
                function=function,
                attention=attentions[index - 1]
                if attentions is not None
                else (
                    RemotionAttentionLevel.HIGH
                    if function
                    in {
                        RemotionBeatFunction.HOOK,
                        RemotionBeatFunction.EVIDENCE,
                        RemotionBeatFunction.LANDING,
                    }
                    else RemotionAttentionLevel.MEDIUM
                ),
                evidence_required=index == evidence_shot,
                section_start=index == section_shot,
            )
        )
        end = cursor + duration
        words.append(
            AnchoredWord(
                word_id=word_id,
                scene_id=scene_id,
                text=f"word{index}",
                start_seconds=cursor,
                end_seconds=end,
            )
        )
        body_lines, asset_kind, asset_query, source_ids = _shot_content(template)
        shots.append(
            RemotionEditShot(
                shot_id=shot_id,
                scene_id=scene_id,
                narration_excerpt=f"Narration {index}",
                start_word_id=word_id,
                end_word_id=word_id,
                start_seconds=cursor,
                end_seconds=end,
                start_frame=round(cursor * 30),
                end_frame=round(end * 30),
                template=template,
                purpose=function.value,
                headline=f"Headline {index}",
                supporting_text="",
                body_lines=body_lines,
                asset_kind=asset_kind,
                asset_query=asset_query,
                motion=remotion_motion_for_template(template),
                transition_in=(
                    RemotionTransitionPreset.SECTION_WIPE
                    if index == transition_shot
                    else RemotionTransitionPreset.HARD_CUT
                ),
                sfx=RemotionSfxPreset.NONE,
                source_screenshot_source_ids=source_ids,
            )
        )
        cursor = end
    return RemotionEditPlan(
        title="Fixture",
        width=1920,
        height=1080,
        fps=30,
        duration_seconds=cursor,
        duration_frames=round(cursor * 30),
        words=words,
        shots=shots,
        rhythm=RemotionRhythmPlan(beats=beats),
    )


def test_quality_policy_accepts_grounded_mid_plan_section() -> None:
    plan = _plan(
        [
            RemotionTemplate.KINETIC_HOOK,
            RemotionTemplate.HEADLINE_ZOOM,
            RemotionTemplate.SOURCE_SCREENSHOT,
            RemotionTemplate.CONCLUSION,
        ],
        evidence_shot=3,
        section_shot=3,
        transition_shot=3,
        attentions=[
            RemotionAttentionLevel.HIGH,
            RemotionAttentionLevel.MEDIUM,
            RemotionAttentionLevel.MEDIUM,
            RemotionAttentionLevel.HIGH,
        ],
    )

    report = assess_remotion_edit_plan_quality(plan)

    assert report.passed
    assert report.findings == []


@pytest.mark.parametrize(
    ("plan", "expected_code"),
    [
        (
            _plan(
                [
                    RemotionTemplate.KINETIC_HOOK,
                    RemotionTemplate.CODE_REVEAL,
                    RemotionTemplate.CONCLUSION,
                ],
                durations=[3, 2, 3],
            ),
            "readable_dwell",
        ),
        (
            _plan(
                [
                    RemotionTemplate.KINETIC_HOOK,
                    RemotionTemplate.HEADLINE_ZOOM,
                    RemotionTemplate.HEADLINE_ZOOM,
                    RemotionTemplate.HEADLINE_ZOOM,
                    RemotionTemplate.CONCLUSION,
                ]
            ),
            "template_repetition",
        ),
        (
            _plan(
                [
                    RemotionTemplate.KINETIC_HOOK,
                    RemotionTemplate.SOURCE_SCREENSHOT,
                    RemotionTemplate.SOURCE_SCREENSHOT,
                    RemotionTemplate.CONCLUSION,
                ]
            ),
            "template_repetition",
        ),
        (
            _plan(
                [
                    RemotionTemplate.KINETIC_HOOK,
                    RemotionTemplate.HEADLINE_ZOOM,
                    RemotionTemplate.DIAGRAM_FLOW,
                    RemotionTemplate.MEME_CUTAWAY,
                    RemotionTemplate.CONCLUSION,
                ],
                durations=[1, 1, 1, 1, 3],
            ),
            "rapid_cut_density",
        ),
        (
            _plan(
                [
                    RemotionTemplate.KINETIC_HOOK,
                    RemotionTemplate.HEADLINE_ZOOM,
                    RemotionTemplate.CONCLUSION,
                ],
                evidence_shot=2,
                attentions=[
                    RemotionAttentionLevel.HIGH,
                    RemotionAttentionLevel.MEDIUM,
                    RemotionAttentionLevel.HIGH,
                ],
            ),
            "visible_evidence",
        ),
    ],
)
def test_quality_policy_reports_blocking_editing_defects(
    plan: RemotionEditPlan,
    expected_code: str,
) -> None:
    report = assess_remotion_edit_plan_quality(plan)

    assert not report.passed
    assert expected_code in {finding.code for finding in report.findings}


def test_plan_rejects_transition_outside_declared_section_boundary() -> None:
    with pytest.raises(ValueError, match="must match rhythm section starts"):
        _plan(
            [
                RemotionTemplate.KINETIC_HOOK,
                RemotionTemplate.HEADLINE_ZOOM,
                RemotionTemplate.DIAGRAM_FLOW,
                RemotionTemplate.CONCLUSION,
            ],
            section_shot=3,
            transition_shot=2,
        )


def test_rhythm_plan_rejects_noncanonical_shot_coverage() -> None:
    with pytest.raises(ValueError, match="contiguous ordered Shot IDs"):
        RemotionRhythmPlan(
            beats=[
                RemotionRhythmBeat(
                    beat_id="beat-001",
                    shot_id="shot-001",
                    function=RemotionBeatFunction.HOOK,
                    attention=RemotionAttentionLevel.HIGH,
                ),
                RemotionRhythmBeat(
                    beat_id="beat-002",
                    shot_id="shot-003",
                    function=RemotionBeatFunction.LANDING,
                    attention=RemotionAttentionLevel.HIGH,
                ),
            ]
        )


def test_rhythm_plan_rejects_flat_function_and_attention_patterns() -> None:
    templates = [
        RemotionTemplate.KINETIC_HOOK,
        RemotionTemplate.HEADLINE_ZOOM,
        RemotionTemplate.DIAGRAM_FLOW,
        RemotionTemplate.CODE_REVEAL,
        RemotionTemplate.COMPARISON_SPLIT,
        RemotionTemplate.CONCLUSION,
    ]
    with pytest.raises(ValueError, match="one editorial function across four"):
        _plan(
            templates,
            functions=[
                RemotionBeatFunction.HOOK,
                RemotionBeatFunction.EXPLANATION,
                RemotionBeatFunction.EXPLANATION,
                RemotionBeatFunction.EXPLANATION,
                RemotionBeatFunction.EXPLANATION,
                RemotionBeatFunction.LANDING,
            ],
        )
    with pytest.raises(ValueError, match="high-attention budget"):
        _plan(
            templates,
            functions=[
                RemotionBeatFunction.HOOK,
                RemotionBeatFunction.SETUP,
                RemotionBeatFunction.EXPLANATION,
                RemotionBeatFunction.EXAMPLE,
                RemotionBeatFunction.SYNTHESIS,
                RemotionBeatFunction.LANDING,
            ],
            attentions=[RemotionAttentionLevel.HIGH] * len(templates),
        )


def test_quality_policy_requires_breathing_room_in_long_form_plan() -> None:
    plan = _plan(
        [
            RemotionTemplate.KINETIC_HOOK,
            RemotionTemplate.HEADLINE_ZOOM,
            RemotionTemplate.DIAGRAM_FLOW,
            RemotionTemplate.CODE_REVEAL,
            RemotionTemplate.COMPARISON_SPLIT,
            RemotionTemplate.MEME_CUTAWAY,
            RemotionTemplate.DIAGRAM_FLOW,
            RemotionTemplate.CONCLUSION,
        ],
        durations=[6] * 8,
        functions=[
            RemotionBeatFunction.HOOK,
            RemotionBeatFunction.SETUP,
            RemotionBeatFunction.EXPLANATION,
            RemotionBeatFunction.EXAMPLE,
            RemotionBeatFunction.CONTRAST,
            RemotionBeatFunction.SYNTHESIS,
            RemotionBeatFunction.EXPLANATION,
            RemotionBeatFunction.LANDING,
        ],
        attentions=[
            RemotionAttentionLevel.HIGH,
            RemotionAttentionLevel.MEDIUM,
            RemotionAttentionLevel.MEDIUM,
            RemotionAttentionLevel.MEDIUM,
            RemotionAttentionLevel.MEDIUM,
            RemotionAttentionLevel.MEDIUM,
            RemotionAttentionLevel.MEDIUM,
            RemotionAttentionLevel.HIGH,
        ],
    )

    report = assess_remotion_edit_plan_quality(plan)

    assert not report.passed
    assert "rhythm_balance" in {finding.code for finding in report.findings}


def test_rhythm_evidence_is_offered_only_with_readable_interior_dwell() -> None:
    class _OutlineScene:
        def __init__(self, scene_id: str, arc_role: str) -> None:
            self.scene_id = scene_id
            self.arc_role = arc_role

        def model_dump(self, *, mode: str) -> dict[str, str]:
            assert mode == "json"
            return {"scene_id": self.scene_id, "arc_role": self.arc_role}

    engine = object.__new__(WorkflowEngine)
    engine.config = SimpleNamespace(
        content_mode=ContentMode.FACTUAL,
        content_format=ContentFormat.EXPLAINER,
        output_language=OutputLanguage.ENGLISH,
    )
    engine._scene_grounded_source_ids = lambda *_args: ["source-001"]
    captured_input: dict[str, object] = {}

    def structured_item(**kwargs):
        captured_input.update(kwargs["input_data"])
        plan = RemotionRhythmPlan(
            beats=[
                RemotionRhythmBeat(
                    beat_id="beat-001",
                    shot_id="shot-001",
                    function=RemotionBeatFunction.HOOK,
                    attention=RemotionAttentionLevel.HIGH,
                ),
                RemotionRhythmBeat(
                    beat_id="beat-002",
                    shot_id="shot-002",
                    function=RemotionBeatFunction.EVIDENCE,
                    attention=RemotionAttentionLevel.MEDIUM,
                ),
                RemotionRhythmBeat(
                    beat_id="beat-003",
                    shot_id="shot-003",
                    function=RemotionBeatFunction.EVIDENCE,
                    attention=RemotionAttentionLevel.MEDIUM,
                    evidence_required=True,
                ),
                RemotionRhythmBeat(
                    beat_id="beat-004",
                    shot_id="shot-004",
                    function=RemotionBeatFunction.LANDING,
                    attention=RemotionAttentionLevel.HIGH,
                ),
            ]
        )
        kwargs["invariant"](plan)
        return plan, []

    engine._structured_item = structured_item
    schedule = [
        {
            "shot_id": "shot-001",
            "scene_id": "scene-001",
            "narration_excerpt": "Hook.",
            "start_seconds": 0,
            "end_seconds": 3,
        },
        {
            "shot_id": "shot-002",
            "scene_id": "scene-002",
            "narration_excerpt": "Short evidence.",
            "start_seconds": 3,
            "end_seconds": 5.4,
        },
        {
            "shot_id": "shot-003",
            "scene_id": "scene-002",
            "narration_excerpt": "Readable evidence.",
            "start_seconds": 5.4,
            "end_seconds": 8.4,
        },
        {
            "shot_id": "shot-004",
            "scene_id": "scene-003",
            "narration_excerpt": "Landing.",
            "start_seconds": 8.4,
            "end_seconds": 11.4,
        },
    ]

    plan, _usage = engine._remotion_rhythm_plan(
        schedule=schedule,
        outline=SimpleNamespace(
            scenes=[
                _OutlineScene("scene-001", "modern_hook"),
                _OutlineScene("scene-002", "evidence"),
                _OutlineScene("scene-003", "landing"),
            ]
        ),
        research=SimpleNamespace(),
        factual_grounding=None,
        source_options=[{"source_id": "source-001"}],
    )

    rhythm_schedule = captured_input["canonical_shot_schedule"]
    assert [item["evidence_available"] for item in rhythm_schedule] == [
        False,
        False,
        True,
        False,
    ]
    assert plan.beats[2].evidence_required
