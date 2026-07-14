from __future__ import annotations

from video_generator.contracts import ReviewFinding, ReviewReport
from video_generator.workflow import WorkflowEngine


def _report(review_type: str) -> ReviewReport:
    return ReviewReport(
        review_type=review_type,
        passed=False,
        findings=[
            ReviewFinding(
                finding_id="finding-001",
                severity="major",
                scene_id="scene-001",
                evidence="First issue",
                recommendation="Fix the first issue",
            ),
            ReviewFinding(
                finding_id="finding-002",
                severity="minor",
                scene_id="scene-002",
                evidence="Second issue",
                recommendation="Fix the second issue",
            ),
        ],
    )


def test_review_finding_ids_are_canonical_and_unique_across_roles() -> None:
    story = _report("story")
    spoken = _report("spoken")

    scene_ids = {"scene-001", "scene-002"}
    WorkflowEngine._validate_review(story, "story", "review_story", scene_ids)
    WorkflowEngine._validate_review(spoken, "spoken", "review_spoken", scene_ids)

    assert [finding.finding_id for finding in story.findings] == [
        "story:finding-001",
        "story:finding-002",
    ]
    assert [finding.finding_id for finding in spoken.findings] == [
        "spoken:finding-001",
        "spoken:finding-002",
    ]
    assert len({finding.finding_id for report in (story, spoken) for finding in report.findings}) == 4
