from __future__ import annotations

import pytest

from ddgs.exceptions import DDGSException

from video_generator.backends.ddgs import DDGSSearchBackend
from video_generator.contracts import OutputLanguage, ResearchSource, SearchRequest, SourceFetchRequest
from video_generator.errors import BackendError, ErrorKind


class StubDDGS:
    def __init__(self, results=None, error: Exception | None = None) -> None:
        self.results = results or []
        self.error = error
        self.calls = []

    def text(self, query, **kwargs):
        self.calls.append((query, kwargs))
        if self.error:
            raise self.error
        return self.results


@pytest.mark.parametrize(
    ("language", "region"),
    [
        (OutputLanguage.ENGLISH, "us-en"),
        (OutputLanguage.FINNISH, "fi-fi"),
    ],
)
def test_ddgs_uses_bounded_duckduckgo_region(language: OutputLanguage, region: str) -> None:
    client = StubDDGS(
        [{"title": "Result", "href": "https://example.com/item", "body": "Useful detail."}]
    )
    backend = DDGSSearchBackend(client=client)

    result = backend.search(SearchRequest(query="winter detail", max_results=3, language=language))

    assert len(result.sources) == 1
    assert client.calls == [
        (
            "winter detail",
            {
                "region": region,
                "safesearch": "moderate",
                "max_results": 3,
                "page": 1,
                "backend": "duckduckgo",
            },
        )
    ]


def test_ddgs_filters_urls_deduplicates_and_bounds_excerpt() -> None:
    client = StubDDGS(
        [
            {"title": " First  result ", "href": "https://example.com/item", "body": "a  b"},
            {"title": "Duplicate", "href": "https://example.com/item", "body": "ignored"},
            {"title": "Credentials", "href": "https://user@example.com/private", "body": "ignored"},
            {"title": "Local", "href": "file:///tmp/item", "body": "ignored"},
            {"title": "Second", "href": "http://example.org/second", "body": "x" * 2500},
        ]
    )
    result = DDGSSearchBackend(client=client).search(
        SearchRequest(query="winter detail", max_results=5, language=OutputLanguage.ENGLISH)
    )

    assert [source.url for source in result.sources] == [
        "https://example.com/item",
        "http://example.org/second",
    ]
    assert result.sources[0].title == "First result"
    assert result.sources[0].excerpt == "a b"
    assert len(result.sources[1].excerpt) == 2000


def test_ddgs_maps_library_failures_to_transient_error() -> None:
    backend = DDGSSearchBackend(client=StubDDGS(error=DDGSException("rate limited")))

    with pytest.raises(BackendError, match="DuckDuckGo search failed") as caught:
        backend.search(
            SearchRequest(query="winter detail", max_results=1, language=OutputLanguage.ENGLISH)
        )

    assert caught.value.kind is ErrorKind.TRANSIENT


def test_ddgs_treats_no_results_as_an_empty_search() -> None:
    backend = DDGSSearchBackend(client=StubDDGS(error=DDGSException("No results found.")))

    result = backend.search(
        SearchRequest(query="very narrow query", max_results=1, language=OutputLanguage.ENGLISH)
    )

    assert result.sources == []


def test_ddgs_does_not_fetch_arbitrary_result_pages() -> None:
    backend = DDGSSearchBackend(client=StubDDGS())
    request = SourceFetchRequest(
        source=ResearchSource(
            source_id="source-001",
            url="https://example.com/item",
            title="Example",
        )
    )

    with pytest.raises(BackendError, match="direct arbitrary-URL") as caught:
        backend.fetch(request)

    assert caught.value.kind is ErrorKind.UNSUPPORTED
