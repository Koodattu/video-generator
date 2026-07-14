from __future__ import annotations

import pytest

from types import SimpleNamespace

from video_generator.contracts import (
    CandidateSet,
    EvidenceRecord,
    EvidenceRecordDraft,
    FactualResearchPack,
    OutputLanguage,
    ResearchFinding,
    ResearchPack,
    ResearchSource,
    StoryCandidate,
)
from video_generator.errors import BackendError
from video_generator.workflow import (
    EvidenceGroundingDecision,
    SourceAdmissionDecision,
    WorkflowEngine,
)


def test_research_pack_rejects_fabricated_source_ids() -> None:
    bounded_source = ResearchSource(
        source_id="source-001",
        url="https://example.com/source",
        title="Bounded source",
    )
    fabricated_source = ResearchSource(
        source_id="src_001",
        url="https://example.com/fabricated",
        title="Fabricated source",
    )
    pack = ResearchPack(
        sources=[fabricated_source],
        findings=[
            ResearchFinding(
                finding_id="finding-001",
                summary="Unsupported detail",
                source_ids=["src_001"],
            )
        ],
    )

    with pytest.raises(BackendError, match="outside the bounded search results: src_001"):
        WorkflowEngine._validate_research_source_references(pack, [bounded_source])


def test_empty_research_pack_is_valid_without_sources_or_findings() -> None:
    pack = ResearchPack(queries=["a bounded query"])

    WorkflowEngine._validate_research_source_references(pack, [])

    assert pack.sources == []
    assert pack.findings == []


def test_research_finding_requires_bounded_source_attribution() -> None:
    bounded_source = ResearchSource(
        source_id="source-001",
        url="https://example.com/source",
        title="Bounded source",
    )
    pack = ResearchPack(
        sources=[bounded_source],
        findings=[
            ResearchFinding(
                finding_id="finding-001",
                summary="An unattributed detail",
                source_ids=[],
            )
        ],
    )

    with pytest.raises(BackendError, match="requires at least one bounded Source ID"):
        WorkflowEngine._validate_research_source_references(pack, [bounded_source])


def test_candidate_source_id_is_mapped_to_its_single_research_finding() -> None:
    source = ResearchSource(
        source_id="source-001",
        url="https://example.com/source",
        title="Bounded source",
    )
    research = ResearchPack(
        sources=[source],
        findings=[
            ResearchFinding(
                finding_id="finding-001",
                summary="A bounded detail",
                source_ids=["source-001"],
            )
        ],
    )
    candidates = CandidateSet(
        candidates=[
            StoryCandidate(
                candidate_id="candidate-001",
                title="Fixture",
                premise="A premise",
                protagonist_desire="Find shelter",
                obstacle="A blocked path",
                turn="The light changes direction",
                ending_direction="The traveler chooses a new path",
                emotional_promise="Quiet suspense",
                research_inspiration_ids=["source-001"],
                duration_fit="Fits",
            )
        ]
    )
    engine = object.__new__(WorkflowEngine)
    engine.config = SimpleNamespace(idea_candidates=1)

    engine._validate_candidates(candidates, research)

    assert candidates.candidates[0].research_inspiration_ids == ["finding-001"]


def test_candidate_source_id_maps_to_all_findings_from_that_source() -> None:
    source = ResearchSource(
        source_id="source-001",
        url="https://example.com/source",
        title="Bounded source",
    )
    research = ResearchPack(
        sources=[source],
        findings=[
            ResearchFinding(
                finding_id=f"finding-{index:03d}",
                summary=f"Bounded detail {index}",
                source_ids=["source-001"],
            )
            for index in (1, 2)
        ],
    )
    candidate = StoryCandidate(
        candidate_id="candidate-001",
        title="Fixture",
        premise="A premise",
        protagonist_desire="Find shelter",
        obstacle="A blocked path",
        turn="The light changes direction",
        ending_direction="The traveler chooses a new path",
        emotional_promise="Quiet suspense",
        research_inspiration_ids=["source-001"],
        duration_fit="Fits",
    )
    candidates = CandidateSet(candidates=[candidate])
    engine = object.__new__(WorkflowEngine)
    engine.config = SimpleNamespace(idea_candidates=1)

    engine._validate_candidates(candidates, research)

    assert candidate.research_inspiration_ids == ["finding-001", "finding-002"]


def test_factual_candidate_source_id_is_mapped_to_admitted_evidence() -> None:
    source = ResearchSource(
        source_id="source-001",
        url="https://example.com/source",
        title="Bounded source",
    )
    research = FactualResearchPack(
        sources=[source],
        evidence=[
            EvidenceRecord(
                evidence_id="evidence-001",
                supported_statement="A bounded statement.",
                source_ids=[source.source_id],
                confidence="high",
            )
        ],
    )
    candidate = StoryCandidate(
        candidate_id="candidate-001",
        title="Fixture",
        premise="A premise",
        protagonist_desire="Understand the evidence",
        obstacle="A common assumption",
        turn="The evidence changes the explanation",
        ending_direction="Return to the opening observation",
        emotional_promise="Clear curiosity",
        research_inspiration_ids=[source.source_id],
        duration_fit="Fits",
    )
    engine = object.__new__(WorkflowEngine)
    engine.config = SimpleNamespace(idea_candidates=1)

    engine._validate_candidates(CandidateSet(candidates=[candidate]), research)

    assert candidate.research_inspiration_ids == ["evidence-001"]
    assert set(WorkflowEngine._authoring_research_payload(research)) == {"evidence"}
    assert WorkflowEngine._outline_scene_evidence_ids(
        SimpleNamespace(scene_id="scene-001"),
        research,
    ) == ["evidence-001"]


def test_compact_authoring_evidence_preserves_time_sensitivity() -> None:
    evidence = EvidenceRecord(
        evidence_id="evidence-001",
        supported_statement="A time-bounded statement.",
        source_ids=["source-001"],
        confidence="medium",
        time_sensitive=True,
        limitations=["Verified only for the stated period."],
    )

    assert WorkflowEngine._authoring_evidence_payload(evidence) == {
        "evidence_id": "evidence-001",
        "supported_statement": "A time-bounded statement.",
        "confidence": "medium",
        "time_sensitive": True,
        "limitations": ["Verified only for the stated period."],
    }


def test_source_entailment_review_excludes_unsupported_evidence_candidate() -> None:
    source = ResearchSource(
        source_id="source-001",
        url="https://example.com/source",
        title="Bounded source",
        excerpt="The source directly supports only the first statement.",
    )
    candidates = [
        EvidenceRecordDraft(
            supported_statement="The source supports the first statement.",
            source_ids=[source.source_id],
            confidence="high",
        ),
        EvidenceRecordDraft(
            supported_statement="A stronger unsupported causal statement.",
            source_ids=[source.source_id],
            confidence="high",
        ),
    ]
    engine = object.__new__(WorkflowEngine)
    engine.config = SimpleNamespace(output_language=OutputLanguage.ENGLISH)

    def structured_item(**kwargs):
        verdict = (
            "entailed"
            if kwargs["input_data"]["candidate_statement"].startswith("The source")
            else "not_entailed"
        )
        return (
            EvidenceGroundingDecision(
                verdict=verdict,
                rationale="The linked excerpt determines whether this exact statement is supported.",
            ),
            [],
        )

    engine._structured_item = structured_item

    admitted, usage, warnings = engine._ground_factual_evidence_candidates(
        candidates,
        [source],
    )

    assert admitted == candidates[:1]
    assert usage == []
    assert warnings == [
        "Excluded factual Evidence candidate 2 after source-entailment review."
    ]


def test_source_admission_rejects_content_farm_signal_before_model_review() -> None:
    content_farm = ResearchSource(
        source_id="source-001",
        url="https://example.com/translated-article",
        title="Tieteellinen Ja Suosittu Multimedian Portaali 💫",
        excerpt="A machine-translated scientific-sounding excerpt.",
    )
    reference = ResearchSource(
        source_id="source-002",
        url="https://en.wikipedia.org/wiki/Fixture",
        title="Fixture – Wikipedia",
        excerpt="A substantive, attributable reference excerpt.",
    )
    reviewed_sources: list[str] = []
    engine = object.__new__(WorkflowEngine)
    engine.config = SimpleNamespace(output_language=OutputLanguage.ENGLISH)

    def structured_item(**kwargs):
        assert kwargs["output_model"] is SourceAdmissionDecision
        reviewed_sources.append(kwargs["input_data"]["source"]["source_id"])
        return (
            SourceAdmissionDecision(
                verdict="admit",
                rationale="The transparent reference has bounded provenance and a substantive excerpt.",
            ),
            [],
        )

    engine._structured_item = structured_item

    admitted, usage, warnings = engine._admit_factual_sources(
        [content_farm, reference]
    )

    assert admitted == [reference]
    assert usage == []
    assert reviewed_sources == [reference.source_id]
    assert warnings == [
        "Excluded factual Source source-001 because its title contains content-farm or "
        "promotional-portal signals."
    ]


def test_source_admission_uses_one_bounded_decision_per_source() -> None:
    admitted_source = ResearchSource(
        source_id="source-001",
        url="https://example.edu/research",
        title="University research",
        excerpt="A substantive institutional excerpt.",
    )
    rejected_source = ResearchSource(
        source_id="source-002",
        url="https://example.com/blog",
        title="Anonymous blog",
        excerpt="An unattributed blog excerpt.",
    )
    engine = object.__new__(WorkflowEngine)
    engine.config = SimpleNamespace(output_language=OutputLanguage.ENGLISH)
    requests: list[dict[str, object]] = []

    def structured_item(**kwargs):
        requests.append(kwargs)
        source_id = kwargs["input_data"]["source"]["source_id"]
        verdict = "admit" if source_id == admitted_source.source_id else "reject"
        return (
            SourceAdmissionDecision(
                verdict=verdict,
                rationale="This bounded decision follows the source's visible provenance signals.",
            ),
            [],
        )

    engine._structured_item = structured_item

    admitted, usage, warnings = engine._admit_factual_sources(
        [admitted_source, rejected_source]
    )

    assert admitted == [admitted_source]
    assert usage == []
    assert len(requests) == 2
    assert all(
        request["input_data"]["review_strategy"] == "single-source-admission-v1"
        for request in requests
    )
    assert warnings == [
        "Excluded factual Source source-002 after bounded source-admission review."
    ]


def test_source_entailment_review_blocks_when_every_candidate_is_rejected() -> None:
    source = ResearchSource(
        source_id="source-001",
        url="https://example.com/source",
        title="Bounded source",
        excerpt="A narrow bounded excerpt.",
    )
    candidate = EvidenceRecordDraft(
        supported_statement="An unsupported generalization.",
        source_ids=[source.source_id],
        confidence="high",
    )
    engine = object.__new__(WorkflowEngine)
    engine.config = SimpleNamespace(output_language=OutputLanguage.ENGLISH)
    engine._structured_item = lambda **kwargs: (
        EvidenceGroundingDecision(
            verdict="not_entailed",
            rationale="The candidate is broader than the supplied linked source excerpt.",
        ),
        [],
    )

    with pytest.raises(BackendError, match="no Evidence candidate directly entailed"):
        engine._ground_factual_evidence_candidates([candidate], [source])


def test_source_entailment_excludes_non_high_confidence_candidate() -> None:
    source = ResearchSource(
        source_id="source-001",
        url="https://example.com/source",
        title="Bounded source",
        excerpt="The source directly supports both bounded statements.",
    )
    medium = EvidenceRecordDraft(
        supported_statement="A medium-confidence bounded statement.",
        source_ids=[source.source_id],
        confidence="medium",
    )
    high = EvidenceRecordDraft(
        supported_statement="A high-confidence bounded statement.",
        source_ids=[source.source_id],
        confidence="high",
    )
    engine = object.__new__(WorkflowEngine)
    engine.config = SimpleNamespace(output_language=OutputLanguage.ENGLISH)
    engine._structured_item = lambda **kwargs: (
        EvidenceGroundingDecision(
            verdict="entailed",
            rationale="The linked excerpt directly entails this exact bounded statement.",
        ),
        [],
    )

    admitted, usage, warnings = engine._ground_factual_evidence_candidates(
        [medium, high],
        [source],
    )

    assert admitted == [high]
    assert usage == []
    assert warnings == [
        "Excluded factual Evidence candidate 1 because authoring requires high-confidence "
        "bounded evidence."
    ]


def test_source_entailment_excludes_candidate_with_any_empty_linked_excerpt() -> None:
    bounded_source = ResearchSource(
        source_id="source-001",
        url="https://example.com/bounded",
        title="Bounded source",
        excerpt="This excerpt supports the bounded statement.",
    )
    empty_source = ResearchSource(
        source_id="source-002",
        url="https://example.com/empty",
        title="Empty source",
        excerpt="   ",
    )
    excluded = EvidenceRecordDraft(
        supported_statement="A candidate linked partly to missing source content.",
        source_ids=[bounded_source.source_id, empty_source.source_id],
        confidence="low",
    )
    admitted_candidate = EvidenceRecordDraft(
        supported_statement="This excerpt supports the bounded statement.",
        source_ids=[bounded_source.source_id],
        confidence="high",
    )
    reviewed_statements: list[str] = []
    engine = object.__new__(WorkflowEngine)
    engine.config = SimpleNamespace(output_language=OutputLanguage.ENGLISH)

    def structured_item(**kwargs):
        reviewed_statements.append(kwargs["input_data"]["candidate_statement"])
        return (
            EvidenceGroundingDecision(
                verdict="entailed",
                rationale="The supplied nonempty excerpt directly entails this exact statement.",
            ),
            [],
        )

    engine._structured_item = structured_item

    admitted, usage, warnings = engine._ground_factual_evidence_candidates(
        [excluded, admitted_candidate],
        [bounded_source, empty_source],
    )

    assert admitted == [admitted_candidate]
    assert usage == []
    assert reviewed_statements == [admitted_candidate.supported_statement]
    assert warnings == [
        "Excluded factual Evidence candidate 1 because linked Source excerpts were empty: "
        "source-002."
    ]


def test_source_entailment_blocks_when_all_linked_excerpts_are_empty() -> None:
    source = ResearchSource(
        source_id="source-001",
        url="https://example.com/empty",
        title="Empty source",
    )
    candidate = EvidenceRecordDraft(
        supported_statement="A statement without bounded source content.",
        source_ids=[source.source_id],
        confidence="low",
    )
    engine = object.__new__(WorkflowEngine)
    engine.config = SimpleNamespace(output_language=OutputLanguage.ENGLISH)
    engine._structured_item = lambda **kwargs: pytest.fail(
        "Empty excerpts must be rejected before model review"
    )

    with pytest.raises(BackendError, match="no Evidence candidate directly entailed"):
        engine._ground_factual_evidence_candidates([candidate], [source])
