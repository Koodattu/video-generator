from __future__ import annotations

import pytest
from pydantic import ValidationError

from video_generator.contracts import (
    EvidenceRecord,
    EvidenceRecordDraft,
    FactualResearchPack,
    FactualResearchSynthesis,
    MusicBrief,
    NarrationScript,
    RenderPlan,
    RenderScene,
    ResearchSource,
    ScriptScene,
)


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


@pytest.mark.parametrize(
    "spoken_text",
    [
        "A fox found a lantern. pause_after_seconds 0.35",
        'A fox found a lantern. "pause_after_seconds": 0.35',
        "A fox found a lantern. scene_id scene-001",
    ],
)
def test_narration_rejects_host_field_labels_inside_spoken_text(
    spoken_text: str,
) -> None:
    with pytest.raises(ValidationError, match="host schema field fragment"):
        NarrationScript(
            title="Lantern",
            scenes=[ScriptScene(scene_id="scene-001", spoken_text=spoken_text)],
        )


def test_narration_allows_ordinary_words_that_resemble_field_names() -> None:
    script = NarrationScript(
        title="Lantern",
        scenes=[
            ScriptScene(
                scene_id="scene-001",
                spoken_text="The scene ID is visible, and the spoken text remains natural.",
            )
        ],
    )

    assert script.scenes[0].spoken_text.endswith("natural.")


@pytest.mark.parametrize(
    "spoken_text",
    [
        "Set schema_version: 1 before the migration runs.",
        'The code reads scene_id = "scene-001" from the request.',
        "The count_method len(items) is shown on screen.",
    ],
)
def test_narration_allows_host_like_identifiers_in_technical_explanations(
    spoken_text: str,
) -> None:
    script = NarrationScript(
        title="Technical explainer",
        scenes=[ScriptScene(scene_id="scene-001", spoken_text=spoken_text)],
    )

    assert script.scenes[0].spoken_text == spoken_text


@pytest.mark.parametrize(
    "updates",
    [
        {"supported_statement": "x" * 601},
        {"limitations": ["bounded"] * 5},
        {"limitations": ["x" * 241]},
    ],
)
def test_evidence_draft_bounds_local_model_payload(updates: dict[str, object]) -> None:
    values: dict[str, object] = {
        "supported_statement": "A bounded statement.",
        "source_ids": ["source-001"],
        "confidence": "high",
    }
    values.update(updates)

    with pytest.raises(ValidationError):
        EvidenceRecordDraft.model_validate(values)


def test_evidence_record_uses_the_same_statement_bound() -> None:
    with pytest.raises(ValidationError):
        EvidenceRecord(
            evidence_id="evidence-001",
            supported_statement="x" * 601,
            source_ids=["source-001"],
            confidence="high",
        )


def test_factual_research_synthesis_accepts_at_most_twelve_records() -> None:
    record = EvidenceRecordDraft(
        supported_statement="A bounded statement.",
        source_ids=["source-001"],
        confidence="high",
    )

    assert len(FactualResearchSynthesis(evidence=[record] * 12).evidence) == 12
    with pytest.raises(ValidationError):
        FactualResearchSynthesis(evidence=[record] * 13)


def test_factual_research_pack_accepts_at_most_twelve_records() -> None:
    source = ResearchSource(
        source_id="source-001",
        url="https://example.com/source",
        title="Bounded source",
    )
    evidence = [
        EvidenceRecord(
            evidence_id=f"evidence-{index:03d}",
            supported_statement=f"Bounded statement {index}.",
            source_ids=[source.source_id],
            confidence="high",
        )
        for index in range(1, 14)
    ]

    assert len(FactualResearchPack(sources=[source], evidence=evidence[:12]).evidence) == 12
    with pytest.raises(ValidationError):
        FactualResearchPack(sources=[source], evidence=evidence)
