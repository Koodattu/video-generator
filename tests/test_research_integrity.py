from __future__ import annotations

import pytest

from video_generator.contracts import ResearchFinding, ResearchPack, ResearchSource
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
