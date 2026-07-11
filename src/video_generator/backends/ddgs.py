from __future__ import annotations

import urllib.parse
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

from ..contracts import (
    OutputLanguage,
    ProbeItem,
    ProbeReport,
    SearchRequest,
    SearchResult,
    SourceDocument,
    SourceFetchRequest,
    UsageRecord,
)
from ..errors import BackendError, ErrorKind
from ..net import SafeSourceFetcher, source_from_search
from ..profiles import BACKEND_DESCRIPTORS
from .base import Backend


DDGS_VERSION = "9.14.4"
REGIONS = {
    OutputLanguage.ENGLISH: "us-en",
    OutputLanguage.FINNISH: "fi-fi",
}


class DDGSSearchBackend(Backend):
    descriptor = BACKEND_DESCRIPTORS["ddgs:duckduckgo"]

    def __init__(self, *, client: Any | None = None) -> None:
        self.client = client or DDGS(timeout=10)
        self.fetcher = SafeSourceFetcher()

    @staticmethod
    def _installed_version() -> str:
        try:
            return version("ddgs")
        except PackageNotFoundError:
            return ""

    def _text(self, query: str, *, language: OutputLanguage, max_results: int) -> list[dict[str, Any]]:
        try:
            values = self.client.text(
                query,
                region=REGIONS[language],
                safesearch="moderate",
                max_results=max_results,
                page=1,
                backend="duckduckgo",
            )
        except (TimeoutException, RatelimitException) as exc:
            raise BackendError(
                f"DuckDuckGo search failed: {exc}",
                kind=ErrorKind.TRANSIENT,
            ) from exc
        except DDGSException as exc:
            if str(exc).strip().rstrip(".").lower() == "no results found":
                return []
            raise BackendError(
                f"DuckDuckGo search failed: {exc}",
                kind=ErrorKind.TRANSIENT,
            ) from exc
        return [item for item in values if isinstance(item, dict)]

    def probe(self, *, live: bool = False) -> ProbeReport:
        installed = self._installed_version()
        items = [
            ProbeItem(
                name="package",
                ready=installed == DDGS_VERSION,
                detail=(
                    f"ddgs {installed} installed"
                    if installed
                    else "ddgs is not installed"
                ),
                action=(
                    None
                    if installed == DDGS_VERSION
                    else "Run uv sync --active --all-extras from the project virtual environment."
                ),
            )
        ]
        if installed == DDGS_VERSION and live:
            try:
                results = self._text(
                    "winter storytelling detail",
                    language=OutputLanguage.ENGLISH,
                    max_results=1,
                )
                items.append(
                    ProbeItem(
                        name="live_search",
                        ready=True,
                        detail=(
                            "DuckDuckGo returned a result"
                            if results
                            else "DuckDuckGo request completed with no results"
                        ),
                    )
                )
            except BackendError as exc:
                items.append(
                    ProbeItem(
                        name="live_search",
                        ready=False,
                        detail=exc.message,
                        action="Retry later or disable research search for this Run.",
                    )
                )
        return ProbeReport(
            backend_id=self.descriptor.backend_id,
            ready=all(item.ready for item in items),
            items=items,
        )

    def search(self, request: SearchRequest) -> SearchResult:
        raw_results = self._text(
            request.query,
            language=request.language,
            max_results=request.max_results,
        )
        sources = []
        seen_urls: set[str] = set()
        for item in raw_results:
            url = str(item.get("href") or "").strip()
            parsed = urllib.parse.urlparse(url)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or url in seen_urls
            ):
                continue
            seen_urls.add(url)
            excerpt = " ".join(str(item.get("body") or "").split())[:2000]
            title = " ".join(str(item.get("title") or parsed.hostname).split())
            sources.append(
                source_from_search(
                    source_id=f"source-{len(sources) + 1:03d}",
                    url=url,
                    title=title or parsed.hostname,
                    publisher=parsed.netloc,
                    excerpt=excerpt,
                    language=request.language.value,
                )
            )
            if len(sources) >= request.max_results:
                break
        return SearchResult(
            query=request.query,
            sources=sources,
            usage=UsageRecord(
                task_id="search",
                backend_id=self.descriptor.backend_id,
                input_units=1,
            ),
        )

    def fetch(self, request: SourceFetchRequest) -> SourceDocument:
        return self.fetcher.fetch(request)
