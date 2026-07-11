from __future__ import annotations

import pytest

from types import SimpleNamespace

from video_generator.contracts import (
    CandidateSet,
    ResearchFinding,
    ResearchPack,
    ResearchSource,
    StoryCandidate,
)
from video_generator.errors import BackendError
from video_generator.workflow import WorkflowEngine


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
