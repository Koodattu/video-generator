from __future__ import annotations

import re
from collections.abc import Sequence

from .contracts import (
    RemotionAssetKind,
    RemotionEditPlan,
    RemotionMotionPreset,
    RemotionQualityFinding,
    RemotionQualityReport,
    RemotionTemplate,
    RemotionTransitionPreset,
    WordTiming,
)


_STRONG_BOUNDARY = re.compile(r"""[.!?…]+["'”’)\]]*$""")
_WEAK_BOUNDARY = re.compile(r"""[,;:]+["'”’)\]]*$""")
_COMMON_ABBREVIATIONS = frozenset(
    {
        "e.g.",
        "i.e.",
        "etc.",
        "mr.",
        "mrs.",
        "ms.",
        "dr.",
        "prof.",
        "esim.",
        "jne.",
        "mm.",
        "n.",
    }
)


def _boundary_priority(text: str) -> int | None:
    normalized = text.strip().casefold()
    if not normalized:
        return None
    if normalized in _COMMON_ABBREVIATIONS or re.fullmatch(r"(?:[a-z]\.){2,}", normalized):
        return None
    if _STRONG_BOUNDARY.search(normalized):
        return 0
    if _WEAK_BOUNDARY.search(normalized):
        return 1
    return None


def semantic_shot_end_frames(
    *,
    scene_start_frame: int,
    scene_end_frame: int,
    shot_count: int,
    minimum_frames: int,
    maximum_frames: int,
    scene_start_seconds: float,
    words: Sequence[WordTiming],
    fps: int,
) -> list[int]:
    """Return frame-aligned Shot ends, preferring legal sentence and clause boundaries."""

    if shot_count <= 0:
        raise ValueError("shot_count must be positive")
    if scene_end_frame <= scene_start_frame:
        raise ValueError("scene_end_frame must follow scene_start_frame")
    if shot_count == 1:
        return [scene_end_frame]

    candidates: list[tuple[int, int]] = []
    for word in words:
        priority = _boundary_priority(word.text)
        if priority is None:
            continue
        frame = round((scene_start_seconds + float(word.end_seconds)) * fps)
        if scene_start_frame < frame < scene_end_frame:
            candidates.append((frame, priority))

    ends: list[int] = []
    cursor = scene_start_frame
    scene_frames = scene_end_frame - scene_start_frame
    for boundary_index in range(1, shot_count):
        remaining_shots = shot_count - boundary_index
        minimum_end = max(
            cursor + minimum_frames,
            scene_end_frame - remaining_shots * maximum_frames,
        )
        maximum_end = min(
            cursor + maximum_frames,
            scene_end_frame - remaining_shots * minimum_frames,
        )
        if minimum_end > maximum_end:
            raise ValueError("Shot bounds cannot fit the requested Scene")
        desired = round(
            scene_start_frame + scene_frames * boundary_index / shot_count
        )
        desired = min(max(desired, minimum_end), maximum_end)
        eligible = [
            (priority, abs(frame - desired), frame)
            for frame, priority in candidates
            if minimum_end <= frame <= maximum_end
        ]
        selected = min(eligible)[2] if eligible else desired
        ends.append(selected)
        cursor = selected
    ends.append(scene_end_frame)
    return ends


def assess_remotion_edit_plan_quality(plan: RemotionEditPlan) -> RemotionQualityReport:
    findings: list[RemotionQualityFinding] = []
    beats_by_shot = (
        {beat.shot_id: beat for beat in plan.rhythm.beats}
        if plan.rhythm is not None
        else {}
    )

    for shot in plan.shots:
        duration = shot.end_seconds - shot.start_seconds
        if shot.template in {
            RemotionTemplate.CODE_REVEAL,
            RemotionTemplate.SOURCE_SCREENSHOT,
        } and duration < 2.5:
            findings.append(
                RemotionQualityFinding(
                    code="readable_dwell",
                    shot_ids=[shot.shot_id],
                    message=(
                        f"{shot.template.value} needs at least 2.5 seconds of readable dwell"
                    ),
                )
            )
        beat = beats_by_shot.get(shot.shot_id)
        if beat is not None and beat.evidence_required and (
            shot.template is not RemotionTemplate.SOURCE_SCREENSHOT
            or shot.asset_kind is not RemotionAssetKind.SOURCE_SCREENSHOT
            or not shot.source_screenshot_source_ids
        ):
            findings.append(
                RemotionQualityFinding(
                    code="visible_evidence",
                    shot_ids=[shot.shot_id],
                    message=(
                        "an evidence-required Beat needs a scene-grounded source screenshot"
                    ),
                )
            )

    for index in range(2, len(plan.shots)):
        window = plan.shots[index - 2 : index + 1]
        if len({shot.template for shot in window}) == 1:
            findings.append(
                RemotionQualityFinding(
                    code="template_repetition",
                    shot_ids=[shot.shot_id for shot in window],
                    message="the same Remotion template cannot repeat across three Shots",
                )
            )
    for previous, current in zip(plan.shots, plan.shots[1:]):
        if (
            previous.template is RemotionTemplate.SOURCE_SCREENSHOT
            and current.template is RemotionTemplate.SOURCE_SCREENSHOT
        ):
            findings.append(
                RemotionQualityFinding(
                    code="template_repetition",
                    shot_ids=[previous.shot_id, current.shot_id],
                    message="full-screen source screenshots cannot be consecutive",
                )
            )

    for index in range(3, len(plan.shots)):
        window = plan.shots[index - 3 : index + 1]
        if len({shot.motion for shot in window}) == 1 and window[0].motion is not RemotionMotionPreset.HOLD:
            findings.append(
                RemotionQualityFinding(
                    code="motion_repetition",
                    shot_ids=[shot.shot_id for shot in window],
                    message="the same motion cannot repeat across four consecutive Shots",
                )
            )

    rapid = [
        shot
        for shot in plan.shots
        if shot.end_seconds - shot.start_seconds < 1.5
    ]
    for index, shot in enumerate(rapid):
        window = [
            candidate
            for candidate in rapid[index:]
            if candidate.start_seconds < shot.start_seconds + 10
        ]
        if len(window) > 3:
            findings.append(
                RemotionQualityFinding(
                    code="rapid_cut_density",
                    shot_ids=[candidate.shot_id for candidate in window[:4]],
                    message="more than three rapid cutaways occur within ten seconds",
                )
            )
            break

    if plan.rhythm is not None:
        if (
            plan.duration_seconds >= 45
            and len(plan.rhythm.beats) >= 8
            and not any(
                beat.attention.value == "low"
                or beat.function.value == "breathing_room"
                for beat in plan.rhythm.beats[1:-1]
            )
        ):
            findings.append(
                RemotionQualityFinding(
                    code="rhythm_balance",
                    shot_ids=[plan.rhythm.beats[len(plan.rhythm.beats) // 2].shot_id],
                    message=(
                        "a long-form Remotion plan needs an intentional low-attention "
                        "or breathing-room Beat"
                    ),
                )
            )
        expected = {
            beat.shot_id for beat in plan.rhythm.beats if beat.section_start
        }
        actual = {
            shot.shot_id
            for shot in plan.shots
            if shot.transition_in is RemotionTransitionPreset.SECTION_WIPE
        }
        if expected != actual:
            findings.append(
                RemotionQualityFinding(
                    code="section_transition",
                    shot_ids=sorted(expected.symmetric_difference(actual)) or [plan.shots[0].shot_id],
                    message="section wipes must occur only on declared rhythm section starts",
                )
            )
        transition_shots = [
            shot
            for shot in plan.shots
            if shot.transition_in is RemotionTransitionPreset.SECTION_WIPE
        ]
        if len(transition_shots) > 2:
            findings.append(
                RemotionQualityFinding(
                    code="section_transition",
                    shot_ids=[shot.shot_id for shot in transition_shots[:3]],
                    message="a Remotion plan may use at most two section wipes",
                )
            )
        for previous, current in zip(transition_shots, transition_shots[1:]):
            if current.start_seconds - previous.start_seconds < 10:
                findings.append(
                    RemotionQualityFinding(
                        code="section_transition",
                        shot_ids=[previous.shot_id, current.shot_id],
                        message="section wipes must be separated by at least ten seconds",
                    )
                )

    return RemotionQualityReport(passed=not findings, findings=findings)
