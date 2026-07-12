from __future__ import annotations

import urllib.parse

from ..contracts import ProbeItem, ProbeReport, SearchRequest, SearchResult, SourceDocument, SourceFetchRequest, UsageRecord
from ..net import HttpClient, SafeSourceFetcher, source_from_search
from ..profiles import BACKEND_DESCRIPTORS
from .base import Backend


class BraveSearchBackend(Backend):
    descriptor = BACKEND_DESCRIPTORS["brave:web"]

    def __init__(self, api_key: str, *, http: HttpClient | None = None) -> None:
        self.api_key = api_key
        self.http = http or HttpClient(timeout_seconds=20, max_response_bytes=2_000_000)
        self.fetcher = SafeSourceFetcher()

    def probe(self, *, live: bool = False) -> ProbeReport:
        configured = bool(self.api_key)
        return ProbeReport(
            backend_id=self.descriptor.backend_id,
            ready=configured,
            items=[
                ProbeItem(
                    name="credential",
                    ready=configured,
                    detail="BRAVE_SEARCH_API_KEY is configured" if configured else "BRAVE_SEARCH_API_KEY is missing",
                    action=None if configured else "Add BRAVE_SEARCH_API_KEY to .env.",
                )
            ],
        )

    def search(self, request: SearchRequest) -> SearchResult:
        query = urllib.parse.urlencode(
            {"q": request.query, "count": request.max_results, "search_lang": request.language.value}
        )
        response = self.http.request(
            "GET",
            f"https://api.search.brave.com/res/v1/web/search?{query}",
            headers={"Accept": "application/json", "X-Subscription-Token": self.api_key},
        )
        payload = response.json()
        results = payload.get("web", {}).get("results", [])
        sources = []
        for index, item in enumerate(results[: request.max_results], start=1):
            if not isinstance(item, dict) or not item.get("url"):
                continue
            parsed = urllib.parse.urlparse(str(item["url"]))
            sources.append(
                source_from_search(
                    source_id=f"source-{index:03d}",
                    url=str(item["url"]),
                    title=str(item.get("title") or parsed.netloc),
                    publisher=parsed.netloc,
                    excerpt=str(item.get("description") or "")[:2000],
                    language=request.language.value,
                )
            )
        request_id = response.headers.get("x-request-id", "")
        return SearchResult(
            query=request.query,
            sources=sources,
            provider_request_id=request_id,
            usage=UsageRecord(
                task_id="search",
                backend_id=self.descriptor.backend_id,
                provider_request_id=request_id,
                input_units=1,
                billable_units={"search_queries": 1},
                reserved_usd=self.descriptor.reservation_usd,
            ),
        )

    def fetch(self, request: SourceFetchRequest) -> SourceDocument:
        return self.fetcher.fetch(request)
