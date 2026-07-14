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


def test_ddgs_falls_back_to_bounded_keyless_engines_when_duckduckgo_is_empty() -> None:
    class FallbackDDGS:
        def __init__(self) -> None:
            self.calls = []

        def text(self, query, **kwargs):
            self.calls.append((query, kwargs))
            if kwargs["backend"] == "duckduckgo":
                raise DDGSException("No results found.")
            return [
                {
                    "title": "Thermal conductivity",
                    "href": "https://en.wikipedia.org/wiki/Thermal_conductivity",
                    "body": "A bounded fallback result.",
                }
            ]

    client = FallbackDDGS()
    result = DDGSSearchBackend(client=client).search(
        SearchRequest(
            query="thermal conductivity",
            max_results=3,
            language=OutputLanguage.ENGLISH,
        )
    )

    assert len(result.sources) == 1
    assert [call[1]["backend"] for call in client.calls] == [
        "duckduckgo",
        "wikipedia,brave",
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

    with pytest.raises(BackendError, match="DDGS search failed") as caught:
        backend.search(
            SearchRequest(query="winter detail", max_results=1, language=OutputLanguage.ENGLISH)
        )

    assert caught.value.kind is ErrorKind.TRANSIENT


def test_ddgs_treats_no_results_as_an_empty_search() -> None:
    backend = DDGSSearchBackend(
        client=StubDDGS(error=DDGSException("No results found.")),
        wikipedia_search=lambda *args, **kwargs: [],
    )

    result = backend.search(
        SearchRequest(query="very narrow query", max_results=1, language=OutputLanguage.ENGLISH)
    )

    assert result.sources == []


def test_ddgs_uses_fixed_host_wikipedia_fallback_when_all_engines_are_empty() -> None:
    calls = []

    def wikipedia_search(query, *, language, max_results):
        calls.append((query, language, max_results))
        return [
            {
                "title": "Thermal conductivity",
                "href": "https://en.wikipedia.org/wiki/Thermal_conductivity",
                "body": "A bounded MediaWiki search excerpt.",
            }
        ]

    backend = DDGSSearchBackend(
        client=StubDDGS(error=DDGSException("No results found.")),
        wikipedia_search=wikipedia_search,
    )
    result = backend.search(
        SearchRequest(
            query="thermal conductivity",
            max_results=2,
            language=OutputLanguage.ENGLISH,
        )
    )

    assert calls == [("thermal conductivity", OutputLanguage.ENGLISH, 2)]
    assert [source.url for source in result.sources] == [
        "https://en.wikipedia.org/wiki/Thermal_conductivity"
    ]


def test_wikipedia_excerpt_selects_the_query_relevant_sentence_window() -> None:
    excerpt = DDGSSearchBackend._relevant_excerpt(
        "The opening defines a general property. "
        "Human skin responds to the inward or outward flow of heat. "
        "At similar room temperatures, a metal object can feel cool while wood feels warmer. "
        "A final section discusses an unrelated measurement method.",
        "human skin metal wood room temperature",
    )

    assert "Human skin responds" in excerpt
    assert "metal object can feel cool" in excerpt
    assert "unrelated measurement" in excerpt
    assert "opening defines" not in excerpt


def test_wikipedia_fallback_relaxes_over_specific_query_and_keeps_relevant_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_queries = []

    def mediawiki_json(host, params):
        assert host == "en.wikipedia.org"
        if params["action"] == "parse":
            return {
                "parse": {
                    "text": (
                        "<h2>Heat sensed by human skin</h2>"
                        "<p>Thermal effusivity affects how materials feel to human skin.</p>"
                        "<table><tr><td>Pine wood</td><td>0.36</td></tr>"
                        "<tr><td>Stainless steel</td><td>7.23</td></tr></table>"
                    )
                }
            }
        search_queries.append(params["srsearch"])
        if len(search_queries) == 1:
            return {"query": {"search": []}}
        return {
            "query": {
                "search": [
                    {
                        "title": "Thermal effusivity",
                        "snippet": "A material property.",
                    }
                ]
            }
        }

    monkeypatch.setattr(
        DDGSSearchBackend,
        "_mediawiki_json",
        staticmethod(mediawiki_json),
    )

    results = DDGSSearchBackend._wikipedia_text(
        "thermal effusivity pine wood stainless steel human skin",
        language=OutputLanguage.ENGLISH,
        max_results=3,
    )

    assert search_queries == [
        "thermal effusivity pine wood stainless steel human skin",
        "thermal effusivity",
    ]
    assert len(results) == 1
    assert "human skin" in results[0]["body"]
    assert "Pine wood" in results[0]["body"]
    assert "Stainless steel" in results[0]["body"]


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
