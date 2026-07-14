from __future__ import annotations

import html
import json
import re
import urllib.parse
import urllib.request
from collections.abc import Callable
from html.parser import HTMLParser
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
TEXT_BACKENDS = ("duckduckgo", "wikipedia,brave")
WIKIPEDIA_HOSTS = {
    OutputLanguage.ENGLISH: "en.wikipedia.org",
    OutputLanguage.FINNISH: "fi.wikipedia.org",
}
MAX_WIKIPEDIA_RESPONSE_BYTES = 2_000_000
WIKIPEDIA_BLOCK_TAGS = {"p", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._skip_depth += 1
        elif tag in WIKIPEDIA_BLOCK_TAGS and not self._skip_depth:
            self.parts.append(". ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip_depth:
            self._skip_depth -= 1
        elif tag in WIKIPEDIA_BLOCK_TAGS and not self._skip_depth:
            self.parts.append(". ")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)


class DDGSSearchBackend(Backend):
    descriptor = BACKEND_DESCRIPTORS["ddgs:duckduckgo"]

    def __init__(
        self,
        *,
        client: Any | None = None,
        wikipedia_search: Callable[..., list[dict[str, Any]]] | None = None,
    ) -> None:
        self.client = client or DDGS(timeout=10)
        self.wikipedia_search = wikipedia_search or self._wikipedia_text
        self.fetcher = SafeSourceFetcher()

    @staticmethod
    def _installed_version() -> str:
        try:
            return version("ddgs")
        except PackageNotFoundError:
            return ""

    def _text(self, query: str, *, language: OutputLanguage, max_results: int) -> list[dict[str, Any]]:
        for backend in TEXT_BACKENDS:
            try:
                values = self.client.text(
                    query,
                    region=REGIONS[language],
                    safesearch="moderate",
                    max_results=max_results,
                    page=1,
                    backend=backend,
                )
            except (TimeoutException, RatelimitException) as exc:
                raise BackendError(
                    f"DDGS search failed: {exc}",
                    kind=ErrorKind.TRANSIENT,
                ) from exc
            except DDGSException as exc:
                if str(exc).strip().rstrip(".").lower() == "no results found":
                    values = []
                else:
                    raise BackendError(
                        f"DDGS search failed: {exc}",
                        kind=ErrorKind.TRANSIENT,
                    ) from exc
            results = [item for item in values if isinstance(item, dict)]
            if results:
                return results
        return []

    @staticmethod
    def _mediawiki_json(host: str, params: dict[str, str | int]) -> dict[str, Any]:
        encoded = urllib.parse.urlencode(params)
        request = urllib.request.Request(
            f"https://{host}/w/api.php?{encoded}",
            headers={"User-Agent": "video-generator/0.1 (bounded factual research fallback)"},
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                payload = response.read(MAX_WIKIPEDIA_RESPONSE_BYTES + 1)
        except (OSError, TimeoutError) as exc:
            raise BackendError(
                f"Wikipedia search fallback failed: {exc}",
                kind=ErrorKind.TRANSIENT,
            ) from exc
        if len(payload) > MAX_WIKIPEDIA_RESPONSE_BYTES:
            raise BackendError(
                "Wikipedia search fallback response exceeded the bounded size limit",
                kind=ErrorKind.INVALID_OUTPUT,
            )
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackendError(
                "Wikipedia search fallback returned invalid JSON",
                kind=ErrorKind.TRANSIENT,
            ) from exc
        if not isinstance(value, dict):
            raise BackendError(
                "Wikipedia search fallback returned an invalid payload",
                kind=ErrorKind.TRANSIENT,
            )
        return value

    @staticmethod
    def _relevant_excerpt(page_text: str, query: str, *, maximum_chars: int = 2000) -> str:
        terms = {
            token.casefold()
            for token in re.findall(r"[^\W\d_]{3,}", query, flags=re.UNICODE)
        }
        sentences = [
            " ".join(sentence.split())
            for sentence in re.split(r"(?<=[.!?])\s+", page_text)
            if sentence.strip()
        ]
        if not sentences:
            return ""
        scores = [
            sum(term in sentence.casefold() for term in terms)
            for sentence in sentences
        ]
        best_index = max(
            range(len(sentences)),
            key=lambda index: (scores[index], -index),
        )
        selected = set(
            range(
                max(0, best_index - 1),
                min(len(sentences), best_index + 2),
            )
        )
        covered_terms = {
            term
            for index in selected
            for term in terms
            if term in sentences[index].casefold()
        }
        while len(selected) < 8:
            uncovered = terms - covered_terms
            if not uncovered:
                break
            candidates = [index for index in range(len(sentences)) if index not in selected]
            if not candidates:
                break
            index = max(
                candidates,
                key=lambda item: (
                    sum(term in sentences[item].casefold() for term in uncovered),
                    scores[item],
                    -item,
                ),
            )
            newly_covered = {
                term for term in uncovered if term in sentences[index].casefold()
            }
            if not newly_covered:
                break
            selected.add(index)
            covered_terms.update(newly_covered)
        return " ".join(sentences[index] for index in sorted(selected))[
            :maximum_chars
        ].strip()

    @classmethod
    def _wikipedia_page_excerpt(cls, host: str, title: str, query: str) -> str:
        data = cls._mediawiki_json(
            host,
            {
                "action": "parse",
                "format": "json",
                "formatversion": "2",
                "page": title,
                "prop": "text",
                "disabletoc": "1",
            },
        )
        raw_html = data.get("parse", {}).get("text", "")
        if not isinstance(raw_html, str):
            return ""
        parser = _VisibleTextParser()
        parser.feed(raw_html)
        return cls._relevant_excerpt(" ".join(parser.parts), query)

    @classmethod
    def _wikipedia_text(
        cls,
        query: str,
        *,
        language: OutputLanguage,
        max_results: int,
    ) -> list[dict[str, Any]]:
        host = WIKIPEDIA_HOSTS[language]
        data = cls._mediawiki_json(
            host,
            {
                "action": "query",
                "format": "json",
                "formatversion": "2",
                "list": "search",
                "srnamespace": "0",
                "srsearch": query,
                "srlimit": max_results,
                "utf8": "1",
            },
        )
        values = data.get("query", {}).get("search", []) if isinstance(data, dict) else []
        if not values:
            terms = list(
                dict.fromkeys(
                    token.casefold()
                    for token in re.findall(r"[^\W\d_]{3,}", query, flags=re.UNICODE)
                )
            )
            relaxed_query = " ".join(terms[:2])
            if relaxed_query and relaxed_query != query.casefold():
                data = cls._mediawiki_json(
                    host,
                    {
                        "action": "query",
                        "format": "json",
                        "formatversion": "2",
                        "list": "search",
                        "srnamespace": "0",
                        "srsearch": relaxed_query,
                        "srlimit": max_results,
                        "utf8": "1",
                    },
                )
                values = (
                    data.get("query", {}).get("search", [])
                    if isinstance(data, dict)
                    else []
                )
        results = []
        for result_index, item in enumerate(values):
            if not isinstance(item, dict):
                continue
            title = " ".join(str(item.get("title") or "").split())
            if not title:
                continue
            snippet = html.unescape(re.sub(r"<[^>]+>", " ", str(item.get("snippet") or "")))
            excerpt = " ".join(snippet.split())
            if result_index < 2:
                try:
                    excerpt = cls._wikipedia_page_excerpt(host, title, query) or excerpt
                except BackendError:
                    pass
            results.append(
                {
                    "title": title,
                    "href": (
                        f"https://{host}/wiki/"
                        + urllib.parse.quote(title.replace(" ", "_"), safe="()_-")
                    ),
                    "body": excerpt,
                }
            )
        return results[:max_results]

    def _search_results(
        self,
        query: str,
        *,
        language: OutputLanguage,
        max_results: int,
    ) -> list[dict[str, Any]]:
        results = self._text(query, language=language, max_results=max_results)
        if results:
            return results
        return self.wikipedia_search(
            query,
            language=language,
            max_results=max_results,
        )

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
                results = self._search_results(
                    "winter storytelling detail",
                    language=OutputLanguage.ENGLISH,
                    max_results=1,
                )
                items.append(
                    ProbeItem(
                        name="live_search",
                        ready=True,
                        detail=(
                            "Public search returned a result"
                            if results
                            else "Public search completed with no results"
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
        raw_results = self._search_results(
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
