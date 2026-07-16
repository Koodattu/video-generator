from __future__ import annotations

import hashlib
import ipaddress
import json
import mimetypes
import socket
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Mapping

from .contracts import ResearchSource, SourceDocument, SourceFetchRequest
from .errors import BackendError, ErrorKind


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes

    def json_value(self) -> Any:
        try:
            return json.loads(self.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackendError("provider returned invalid JSON", kind=ErrorKind.INVALID_OUTPUT) from exc

    def json(self) -> dict[str, Any]:
        value = self.json_value()
        if not isinstance(value, dict):
            raise BackendError("provider returned a non-object JSON response", kind=ErrorKind.INVALID_OUTPUT)
        return value


class HttpClient:
    def __init__(self, *, timeout_seconds: float = 60, max_response_bytes: int = 20_000_000) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_response_bytes = max_response_bytes
        self._opener = urllib.request.build_opener(_SecureRedirect())

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        json_body: Mapping[str, Any] | None = None,
        body: bytes | None = None,
        timeout_seconds: float | None = None,
        max_response_bytes: int | None = None,
        allowed_redirect_hosts: set[str] | frozenset[str] | None = None,
    ) -> HttpResponse:
        request_headers = {"User-Agent": "video-generator/0.1"}
        request_headers.update(headers or {})
        if json_body is not None:
            if body is not None:
                raise ValueError("json_body and body are mutually exclusive")
            body = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
            request_headers.setdefault("Content-Type", "application/json")
        request = urllib.request.Request(url, data=body, headers=request_headers, method=method.upper())
        maximum = max_response_bytes or self.max_response_bytes
        opener = (
            self._opener
            if allowed_redirect_hosts is None
            else urllib.request.build_opener(
                _AllowlistedRedirect(frozenset(allowed_redirect_hosts))
            )
        )
        try:
            with opener.open(request, timeout=timeout_seconds or self.timeout_seconds) as response:
                response_body = response.read(maximum + 1)
                if len(response_body) > maximum:
                    raise BackendError("provider response exceeded the byte limit", kind=ErrorKind.INVALID_OUTPUT)
                return HttpResponse(response.status, dict(response.headers.items()), response_body)
        except urllib.error.HTTPError as exc:
            error_body = exc.read(min(maximum, 1_000_000))
            self._raise_http_error(exc.code, error_body, dict(exc.headers.items()))
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            raise BackendError(f"network request failed: {exc}", kind=ErrorKind.TRANSIENT) from exc
        raise AssertionError("unreachable")

    @staticmethod
    def _raise_http_error(status: int, body: bytes, headers: Mapping[str, str]) -> None:
        request_id = headers.get("x-request-id") or headers.get("request-id") or ""
        message = f"provider HTTP {status}"
        code = ""
        try:
            payload = json.loads(body.decode("utf-8"))
            error = payload.get("error", payload) if isinstance(payload, dict) else {}
            if isinstance(error, dict):
                message = str(error.get("message") or message)
                code = str(error.get("code") or error.get("type") or "")
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
        if status in {408, 409, 425, 429} or status >= 500:
            kind = ErrorKind.TRANSIENT
        elif status in {401, 403}:
            kind = ErrorKind.NOT_READY
        elif code in {"moderation_blocked", "content_policy_violation"}:
            kind = ErrorKind.POLICY_REFUSAL
        elif status == 402:
            kind = ErrorKind.BUDGET_EXCEEDED
        else:
            kind = ErrorKind.INVALID_OUTPUT
        raise BackendError(message, kind=kind, details={"status": status, "code": code, "request_id": request_id})


def multipart_body(
    fields: Mapping[str, str], files: list[tuple[str, Path, str | None]]
) -> tuple[bytes, str]:
    boundary = f"video-generator-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )
    for name, path, explicit_mime in files:
        mime_type = explicit_mime or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                (
                    f'Content-Disposition: form-data; name="{name}"; filename="{path.name}"\r\n'
                ).encode(),
                f"Content-Type: {mime_type}\r\n\r\n".encode(),
                path.read_bytes(),
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.lower()
        if lowered in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        elif lowered == "title":
            self._in_title = True
        elif lowered in {"p", "br", "li", "h1", "h2", "h3", "blockquote"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
        elif lowered == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = " ".join(data.split())
        if not text:
            return
        if self._in_title:
            self.title = (self.title + " " + text).strip()
        self.parts.append(text)

    def text(self) -> str:
        lines = [" ".join(line.split()) for line in " ".join(self.parts).splitlines()]
        return "\n".join(line for line in lines if line)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


class _SecureRedirect(urllib.request.HTTPRedirectHandler):
    _FORWARDED_HEADERS = {
        "accept",
        "accept-encoding",
        "accept-language",
        "user-agent",
    }

    @staticmethod
    def _reject_https_downgrade(req: Any, newurl: str) -> None:
        source_url = getattr(req, "full_url", "")
        source = urllib.parse.urlparse(source_url)
        target = urllib.parse.urlparse(newurl)
        if source.scheme.casefold() == "https" and target.scheme.casefold() != "https":
            raise BackendError(
                "HTTPS redirects may not downgrade to an insecure scheme",
                kind=ErrorKind.UNSUPPORTED,
            )

    @classmethod
    def _strip_sensitive_headers(cls, request: Any) -> Any:
        if request is None:
            return None
        for values in (request.headers, request.unredirected_hdrs):
            for name in list(values):
                if name.casefold() not in cls._FORWARDED_HEADERS:
                    del values[name]
        return request

    def _build_redirect_request(
        self,
        req: Any,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Any:
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        return self._strip_sensitive_headers(redirected)

    def redirect_request(
        self,
        req: Any,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Any:
        self._reject_https_downgrade(req, newurl)
        validate_public_http_url(newurl)
        return self._build_redirect_request(req, fp, code, msg, headers, newurl)


class _AllowlistedRedirect(_SecureRedirect):
    def __init__(self, allowed_hosts: frozenset[str]) -> None:
        super().__init__()
        self.allowed_hosts = frozenset(host.casefold() for host in allowed_hosts)

    def redirect_request(
        self,
        req: Any,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Any:
        self._reject_https_downgrade(req, newurl)
        parsed = validate_public_http_url(newurl)
        if (parsed.hostname or "").casefold() not in self.allowed_hosts:
            raise BackendError(
                f"redirect target is outside the provider host allowlist: {parsed.hostname}",
                kind=ErrorKind.UNSUPPORTED,
            )
        return self._build_redirect_request(req, fp, code, msg, headers, newurl)


def validate_public_http_url(url: str) -> urllib.parse.ParseResult:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise BackendError("source URL must use public HTTP(S)", kind=ErrorKind.UNSUPPORTED)
    if parsed.username or parsed.password:
        raise BackendError("source URLs may not contain credentials", kind=ErrorKind.UNSUPPORTED)
    try:
        answers = socket.getaddrinfo(
            parsed.hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
        )
    except socket.gaierror as exc:
        raise BackendError(
            f"could not resolve source host: {parsed.hostname}", kind=ErrorKind.TRANSIENT
        ) from exc
    for answer in answers:
        address = ipaddress.ip_address(answer[4][0].split("%", 1)[0])
        if not address.is_global:
            raise BackendError(
                f"source resolves to a non-public address: {address}",
                kind=ErrorKind.UNSUPPORTED,
            )
    return parsed


class SafeSourceFetcher:
    ALLOWED_MIME = {"text/html", "text/plain", "application/json", "application/xhtml+xml"}

    def __init__(self) -> None:
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    @staticmethod
    def _validate_url(url: str) -> urllib.parse.ParseResult:
        return validate_public_http_url(url)

    def fetch(self, request: SourceFetchRequest) -> SourceDocument:
        raise BackendError(
            "direct arbitrary-URL source fetching is disabled in v0; research uses provider-grounded excerpts",
            kind=ErrorKind.UNSUPPORTED,
        )


def source_from_search(
    *, source_id: str, url: str, title: str, publisher: str = "", excerpt: str = "", language: str = ""
) -> ResearchSource:
    return ResearchSource(
        source_id=source_id,
        url=url,
        title=title,
        publisher=publisher,
        excerpt=excerpt,
        language=language,
        content_sha256=hashlib.sha256(excerpt.encode("utf-8")).hexdigest() if excerpt else "",
    )
