from __future__ import annotations

import html
import json
import re
import shutil
import urllib.parse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    AssetRights,
    MediaReference,
    OutputLanguage,
    RemotionAsset,
    RemotionAssetKind,
    RemotionAssetPolicy,
    RemotionAssetRequest,
    utc_now,
)
from .errors import BackendError, ErrorKind, MediaError
from .media import MediaTools, normalize_image
from .net import HttpClient, validate_public_http_url
from .util import atomic_write_json, atomic_write_text, relative_path, sha256_file


WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
PEXELS_PHOTO_API = "https://api.pexels.com/v1/search"
PEXELS_VIDEO_API = "https://api.pexels.com/videos/search"
MAX_ASSET_BYTES = 80_000_000
MAX_VIDEO_SECONDS = 12


def _plain_text(value: Any) -> str:
    raw = str(value.get("value", "") if isinstance(value, dict) else value or "")
    without_tags = re.sub(r"<[^>]+>", " ", raw)
    return " ".join(html.unescape(without_tags).split())


def _truthy_metadata(value: Any) -> bool:
    return _plain_text(value).casefold() in {"1", "true", "yes"}


@dataclass(frozen=True)
class AssetCandidate:
    candidate_id: str
    provider: str
    provider_asset_id: str
    media_kind: str
    mime_type: str
    title: str
    description: str
    source_page_url: str
    download_url: str
    creator_name: str
    creator_url: str
    width: int
    height: int
    duration_seconds: float
    rights: AssetRights
    raw_metadata: dict[str, Any]
    source_path: Path | None = None

    def selection_payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "provider": self.provider,
            "media_kind": self.media_kind,
            "title": self.title,
            "description": self.description,
            "width": self.width,
            "height": self.height,
            "duration_seconds": self.duration_seconds,
            "rights_status": self.rights.review_status,
            "license_name": self.rights.license_name,
        }


def _renumber(candidates: list[AssetCandidate]) -> list[AssetCandidate]:
    return [
        replace(candidate, candidate_id=f"candidate-{index:03d}")
        for index, candidate in enumerate(candidates, start=1)
    ]


def _local_rights(path: Path) -> AssetRights:
    sidecars = [
        path.with_suffix(path.suffix + ".license.json"),
        path.with_suffix(".license.json"),
    ]
    sidecar = next((candidate for candidate in sidecars if candidate.is_file()), None)
    if sidecar is None:
        return AssetRights(
            license_id="user-supplied",
            license_name="User-supplied authorized media",
            attribution_required=False,
            review_status="approved",
            review_reason=(
                "The file is in media-library/; the operator is responsible for placing only "
                "owned or explicitly authorized media there."
            ),
        )
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        return AssetRights.model_validate(payload)
    except (OSError, ValueError, TypeError) as exc:
        raise BackendError(
            f"invalid local media rights sidecar: {sidecar}",
            kind=ErrorKind.INVALID_OUTPUT,
        ) from exc


def search_local_media(
    root: Path,
    request: RemotionAssetRequest,
    *,
    limit: int = 6,
) -> list[AssetCandidate]:
    if not root.is_dir():
        return []
    image_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".avif"}
    video_suffixes = {".mp4", ".mov", ".webm", ".m4v"}
    desired_suffixes = (
        {".gif"}
        if request.kind is RemotionAssetKind.GIF
        else video_suffixes
        if request.kind is RemotionAssetKind.STOCK_VIDEO
        else image_suffixes | ({".gif"} if request.kind is RemotionAssetKind.MEME else set())
    )
    query_tokens = {
        token for token in re.findall(r"[a-z0-9]+", request.query.casefold()) if len(token) > 1
    }
    ranked: list[tuple[int, str, Path]] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.casefold() not in desired_suffixes:
            continue
        searchable = " ".join(
            re.findall(
                r"[a-z0-9]+",
                relative_path(path, root).casefold(),
            )
        )
        score = sum(1 for token in query_tokens if token in searchable)
        if query_tokens and score == 0:
            continue
        ranked.append((-score, path.as_posix().casefold(), path))
    candidates = []
    for _, _, path in sorted(ranked)[:limit]:
        is_motion = path.suffix.casefold() in video_suffixes | {".gif"}
        mime_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".avif": "image/avif",
            ".gif": "image/gif",
            ".mp4": "video/mp4",
            ".mov": "video/quicktime",
            ".webm": "video/webm",
            ".m4v": "video/x-m4v",
        }[path.suffix.casefold()]
        candidates.append(
            AssetCandidate(
                candidate_id="candidate-000",
                provider="local",
                provider_asset_id=relative_path(path, root),
                media_kind="video" if is_motion else "image",
                mime_type=mime_type,
                title=path.stem.replace("_", " ").replace("-", " "),
                description="User-supplied local media library asset.",
                source_page_url="",
                download_url="",
                creator_name="",
                creator_url="",
                width=0,
                height=0,
                duration_seconds=0,
                rights=_local_rights(path),
                raw_metadata={"relative_path": relative_path(path, root)},
                source_path=path.resolve(),
            )
        )
    return candidates


def _wikimedia_rights(
    metadata: Mapping[str, Any],
    *,
    allow_share_alike: bool,
) -> AssetRights | None:
    if _truthy_metadata(metadata.get("NonFree")):
        return None
    short_name = _plain_text(metadata.get("LicenseShortName"))
    license_url = _plain_text(metadata.get("LicenseUrl"))
    usage_terms = _plain_text(metadata.get("UsageTerms"))
    lowered = " ".join((short_name, usage_terms, license_url)).casefold()
    if any(
        marker in lowered
        for marker in (
            "by-nc",
            "by-nd",
            "noncommercial",
            "non-commercial",
            "no derivatives",
            "noderivatives",
        )
    ):
        return None
    share_alike = "by-sa" in lowered or "share alike" in lowered
    if "cc0" in lowered:
        license_id = "CC0"
    elif "public domain" in lowered or "publicdomain" in lowered or "pdm" in lowered:
        license_id = "Public Domain"
    elif share_alike and re.search(r"\bcc\s*by", lowered):
        if not allow_share_alike:
            return None
        license_id = short_name or "CC BY-SA"
    elif re.search(r"\bcc\s*by", lowered):
        license_id = short_name or "CC BY"
    else:
        return None
    creator = _plain_text(metadata.get("Artist"))
    credit = _plain_text(metadata.get("Credit"))
    attribution_required = _truthy_metadata(metadata.get("AttributionRequired")) or (
        "CC BY" in license_id.upper()
    )
    attribution = credit or creator
    if attribution_required and not attribution:
        return None
    return AssetRights(
        license_id=license_id[:120],
        license_name=(short_name or usage_terms or license_id)[:240],
        license_url=license_url[:1000],
        terms_url="https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia",
        attribution_required=attribution_required,
        attribution_text=attribution[:1000],
        share_alike=share_alike,
        review_status="approved",
        review_reason=(
            "Host policy parsed an allowlisted public-domain or Creative Commons license from "
            "Wikimedia metadata. The raw metadata is retained because Commons provides no warranty."
        ),
    )


class WikimediaClient:
    def __init__(self, *, http: HttpClient, user_agent: str, allow_share_alike: bool) -> None:
        self.http = http
        self.user_agent = user_agent.strip()
        self.allow_share_alike = allow_share_alike

    def search(self, request: RemotionAssetRequest, *, limit: int = 6) -> list[AssetCandidate]:
        if not self.user_agent:
            return []
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "generator": "search",
            "gsrnamespace": "6",
            "gsrsearch": request.query,
            "gsrlimit": str(min(12, max(1, limit * 2))),
            "prop": "imageinfo",
            "iiprop": "url|mime|size|sha1|extmetadata",
            "iiurlwidth": "1920",
        }
        response = self.http.request(
            "GET",
            WIKIMEDIA_API + "?" + urllib.parse.urlencode(params),
            headers={"User-Agent": self.user_agent},
            max_response_bytes=5_000_000,
            allowed_redirect_hosts={"commons.wikimedia.org"},
        ).json()
        pages = response.get("query", {}).get("pages", [])
        if not isinstance(pages, list):
            return []
        candidates: list[AssetCandidate] = []
        for page in pages:
            if not isinstance(page, dict):
                continue
            imageinfo = page.get("imageinfo")
            info = imageinfo[0] if isinstance(imageinfo, list) and imageinfo else None
            if not isinstance(info, dict):
                continue
            mime_type = str(info.get("mime") or "").casefold()
            is_gif = mime_type == "image/gif"
            is_video = mime_type.startswith("video/")
            if request.kind is RemotionAssetKind.GIF and not is_gif:
                continue
            if request.kind is RemotionAssetKind.STOCK_VIDEO and not is_video:
                continue
            if request.kind in {RemotionAssetKind.STOCK_IMAGE, RemotionAssetKind.MEME} and (
                is_video or (is_gif and request.kind is RemotionAssetKind.STOCK_IMAGE)
            ):
                continue
            if not (mime_type.startswith("image/") or is_video):
                continue
            metadata = info.get("extmetadata") if isinstance(info.get("extmetadata"), dict) else {}
            rights = _wikimedia_rights(
                metadata,
                allow_share_alike=self.allow_share_alike,
            )
            if rights is None:
                continue
            original_url = str(info.get("url") or "")
            thumb_url = str(info.get("thumburl") or "")
            download_url = original_url if is_gif or is_video else thumb_url or original_url
            source_url = str(info.get("descriptionurl") or "")
            if not download_url.startswith("https://upload.wikimedia.org/"):
                continue
            candidates.append(
                AssetCandidate(
                    candidate_id="candidate-000",
                    provider="wikimedia",
                    provider_asset_id=str(page.get("pageid") or page.get("title") or ""),
                    media_kind="video" if is_gif or is_video else "image",
                    mime_type=mime_type,
                    title=str(page.get("title") or "").removeprefix("File:"),
                    description=_plain_text(metadata.get("ImageDescription"))[:500],
                    source_page_url=source_url,
                    download_url=download_url,
                    creator_name=_plain_text(metadata.get("Artist"))[:300],
                    creator_url="",
                    width=int(info.get("thumbwidth") or info.get("width") or 0),
                    height=int(info.get("thumbheight") or info.get("height") or 0),
                    duration_seconds=0,
                    rights=rights,
                    raw_metadata={
                        "pageid": page.get("pageid"),
                        "title": page.get("title"),
                        "mime": mime_type,
                        "sha1": info.get("sha1"),
                        "width": info.get("width"),
                        "height": info.get("height"),
                        "extmetadata": metadata,
                    },
                )
            )
            if len(candidates) >= limit:
                break
        return candidates


def _pexels_rights(creator: str, creator_url: str) -> AssetRights:
    attribution = f"{creator} / Pexels" if creator else "Pexels"
    return AssetRights(
        license_id="Pexels",
        license_name="Pexels License",
        license_url="https://www.pexels.com/license/",
        terms_url="https://www.pexels.com/api/terms-of-service/",
        attribution_required=False,
        attribution_text=attribution,
        review_status="approved",
        review_reason=(
            "Pexels permits free use and modification. Provider links and creator credit are still "
            "recorded, and identifiable people must not be shown in a bad light or as endorsing a claim."
        ),
    )


class PexelsClient:
    def __init__(self, *, http: HttpClient, api_key: str) -> None:
        self.http = http
        self.api_key = api_key.strip()

    def search(
        self,
        request: RemotionAssetRequest,
        *,
        language: OutputLanguage,
        limit: int = 6,
    ) -> list[AssetCandidate]:
        if not self.api_key or request.kind not in {
            RemotionAssetKind.STOCK_IMAGE,
            RemotionAssetKind.STOCK_VIDEO,
        }:
            return []
        is_video = request.kind is RemotionAssetKind.STOCK_VIDEO
        params = {
            "query": request.query,
            "per_page": str(min(15, max(1, limit))),
            "orientation": "landscape",
            # Direction calls deliberately produce English search queries for every output language.
            "locale": "en-US",
        }
        response = self.http.request(
            "GET",
            (PEXELS_VIDEO_API if is_video else PEXELS_PHOTO_API)
            + "?"
            + urllib.parse.urlencode(params),
            headers={"Authorization": self.api_key},
            max_response_bytes=5_000_000,
            allowed_redirect_hosts={"api.pexels.com"},
        ).json()
        items = response.get("videos" if is_video else "photos", [])
        if not isinstance(items, list):
            return []
        candidates: list[AssetCandidate] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            user = item.get("user") if isinstance(item.get("user"), dict) else {}
            creator = str((user.get("name") if is_video else item.get("photographer")) or "")
            creator_url = str(
                (user.get("url") if is_video else item.get("photographer_url")) or ""
            )
            if is_video:
                files = [
                    value
                    for value in item.get("video_files", [])
                    if isinstance(value, dict)
                    and str(value.get("file_type") or "").casefold() == "video/mp4"
                    and str(value.get("link") or "").startswith("https://")
                ]
                files.sort(
                    key=lambda value: (
                        int(value.get("height") or 0) > 1080,
                        -int(value.get("height") or 0),
                    )
                )
                chosen = next(
                    (value for value in files if 480 <= int(value.get("height") or 0) <= 1080),
                    files[0] if files else None,
                )
                if chosen is None:
                    continue
                download_url = str(chosen["link"])
                width = int(chosen.get("width") or item.get("width") or 0)
                height = int(chosen.get("height") or item.get("height") or 0)
            else:
                sources = item.get("src") if isinstance(item.get("src"), dict) else {}
                download_url = str(sources.get("large2x") or sources.get("landscape") or "")
                width = int(item.get("width") or 0)
                height = int(item.get("height") or 0)
            host = urllib.parse.urlparse(download_url).hostname or ""
            if host not in {"images.pexels.com", "videos.pexels.com"}:
                continue
            candidates.append(
                AssetCandidate(
                    candidate_id="candidate-000",
                    provider="pexels",
                    provider_asset_id=str(item.get("id") or ""),
                    media_kind="video" if is_video else "image",
                    mime_type="video/mp4" if is_video else "image/jpeg",
                    title=str(item.get("alt") or request.query),
                    description=str(item.get("alt") or request.query)[:500],
                    source_page_url=str(item.get("url") or ""),
                    download_url=download_url,
                    creator_name=creator,
                    creator_url=creator_url,
                    width=width,
                    height=height,
                    duration_seconds=float(item.get("duration") or 0),
                    rights=_pexels_rights(creator, creator_url),
                    raw_metadata={
                        "id": item.get("id"),
                        "url": item.get("url"),
                        "creator": creator,
                        "creator_url": creator_url,
                        "width": width,
                        "height": height,
                        "duration": item.get("duration"),
                    },
                )
            )
        return candidates


def find_asset_candidates(
    request: RemotionAssetRequest,
    *,
    project_root: Path,
    policy: RemotionAssetPolicy,
    language: OutputLanguage,
    allow_share_alike: bool,
    environment: Mapping[str, str],
    http: HttpClient | None = None,
) -> list[AssetCandidate]:
    client = http or HttpClient(timeout_seconds=30, max_response_bytes=MAX_ASSET_BYTES)
    candidates = search_local_media(project_root / "media-library", request)
    if policy is RemotionAssetPolicy.LOCAL_ONLY:
        return _renumber(candidates)
    wikimedia = WikimediaClient(
        http=client,
        user_agent=environment.get("WIKIMEDIA_USER_AGENT", ""),
        allow_share_alike=allow_share_alike,
    )
    provider_errors: list[str] = []
    try:
        candidates.extend(wikimedia.search(request))
    except BackendError as exc:
        provider_errors.append(f"Wikimedia: {exc.message}")
    pexels = PexelsClient(http=client, api_key=environment.get("PEXELS_API_KEY", ""))
    try:
        candidates.extend(pexels.search(request, language=language))
    except BackendError as exc:
        provider_errors.append(f"Pexels: {exc.message}")
    if not candidates and provider_errors:
        raise BackendError(
            "asset providers failed: " + "; ".join(provider_errors),
            kind=ErrorKind.TRANSIENT,
        )
    return _renumber(candidates[:12])


def _extension_for_mime(mime_type: str) -> str:
    return {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/avif": ".avif",
        "image/gif": ".gif",
        "video/mp4": ".mp4",
        "video/webm": ".webm",
        "video/quicktime": ".mov",
    }.get(mime_type.casefold(), ".bin")


def _response_mime_type(headers: Mapping[str, str]) -> str:
    value = next(
        (
            str(header_value)
            for name, header_value in headers.items()
            if name.casefold() == "content-type"
        ),
        "",
    )
    mime_type = value.split(";", 1)[0].strip().casefold()
    if not mime_type:
        raise BackendError(
            "asset download did not declare a Content-Type",
            kind=ErrorKind.INVALID_OUTPUT,
        )
    if _extension_for_mime(mime_type) == ".bin":
        raise BackendError(
            f"asset download returned unsupported MIME type {mime_type}",
            kind=ErrorKind.INVALID_OUTPUT,
        )
    return mime_type


def materialize_candidate(
    candidate: AssetCandidate,
    *,
    destination_stem: Path,
    http: HttpClient | None = None,
) -> tuple[Path, str]:
    if candidate.source_path is not None:
        destination = destination_stem.with_suffix(candidate.source_path.suffix)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate.source_path, destination)
        return destination, candidate.mime_type
    parsed = urllib.parse.urlparse(candidate.download_url)
    allowed_hosts = (
        {"upload.wikimedia.org"}
        if candidate.provider == "wikimedia"
        else {"images.pexels.com", "videos.pexels.com"}
        if candidate.provider == "pexels"
        else set()
    )
    if parsed.scheme != "https" or (parsed.hostname or "") not in allowed_hosts:
        raise BackendError(
            f"asset provider returned a disallowed download host: {parsed.hostname or '<missing>'}",
            kind=ErrorKind.UNSUPPORTED,
        )
    validate_public_http_url(candidate.download_url)
    response = (http or HttpClient(timeout_seconds=90, max_response_bytes=MAX_ASSET_BYTES)).request(
        "GET",
        candidate.download_url,
        max_response_bytes=MAX_ASSET_BYTES,
        allowed_redirect_hosts=allowed_hosts,
    )
    content_type = _response_mime_type(response.headers)
    destination = destination_stem.with_suffix(_extension_for_mime(content_type))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.body)
    return destination, content_type


def normalize_asset_media(
    tools: MediaTools,
    source: Path,
    destination_dir: Path,
    *,
    width: int,
    height: int,
    fps: int,
) -> tuple[Path, str, int, int, float, str]:
    payload = tools.probe_json(source)
    streams = payload.get("streams", []) if isinstance(payload.get("streams"), list) else []
    video_stream = next(
        (stream for stream in streams if isinstance(stream, dict) and stream.get("codec_type") == "video"),
        None,
    )
    if not isinstance(video_stream, dict):
        raise MediaError(f"asset has no visual stream: {source}", kind=ErrorKind.INVALID_OUTPUT)
    codec = str(video_stream.get("codec_name") or "").casefold()
    is_motion = codec not in {"png", "mjpeg", "jpeg", "webp", "av1"} or source.suffix.casefold() in {
        ".gif",
        ".mp4",
        ".mov",
        ".webm",
        ".m4v",
    }
    destination_dir.mkdir(parents=True, exist_ok=True)
    if not is_motion:
        destination = destination_dir / "normalized.png"
        normalize_image(tools, source, destination, width=width, height=height)
        return destination, "image", width, height, 0.0, (
            f"ffmpeg scale/crop to {width}x{height}, square pixels, one PNG frame"
        )
    destination = destination_dir / "normalized.mp4"
    filter_value = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},setsar=1,fps={fps}"
    )
    tools.run(
        [
            tools.ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            "-t",
            str(MAX_VIDEO_SECONDS),
            "-an",
            "-vf",
            filter_value,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        timeout=900,
    )
    normalized = tools.probe_json(destination)
    normalized_stream = next(
        stream
        for stream in normalized.get("streams", [])
        if isinstance(stream, dict) and stream.get("codec_type") == "video"
    )
    duration_raw = normalized_stream.get("duration") or normalized.get("format", {}).get("duration") or 0
    duration = float(duration_raw)
    if duration <= 0:
        raise MediaError("normalized motion asset has no duration", kind=ErrorKind.INVALID_OUTPUT)
    return destination, "video", width, height, duration, (
        f"ffmpeg silent H.264 yuv420p CFR {fps}, scale/crop {width}x{height}, capped at {MAX_VIDEO_SECONDS}s"
    )


def build_asset_record(
    request: RemotionAssetRequest,
    candidate: AssetCandidate,
    *,
    original: Path,
    original_mime_type: str,
    normalized: Path,
    media_kind: str,
    width: int,
    height: int,
    duration_seconds: float,
    transform: str,
    project_root: Path,
    warnings: list[str] | None = None,
) -> RemotionAsset:
    normalized_mime = "video/mp4" if media_kind == "video" else "image/png"
    return RemotionAsset(
        asset_id=request.asset_id,
        shot_id=request.shot_id,
        provider=candidate.provider,
        provider_asset_id=candidate.provider_asset_id,
        media_kind=media_kind,
        search_query=request.query,
        source_page_url=candidate.source_page_url,
        creator_name=candidate.creator_name,
        creator_url=candidate.creator_url,
        rights=candidate.rights,
        original=MediaReference(
            path=relative_path(original, project_root),
            sha256=sha256_file(original),
            mime_type=original_mime_type,
        ),
        normalized=MediaReference(
            path=relative_path(normalized, project_root),
            sha256=sha256_file(normalized),
            mime_type=normalized_mime,
        ),
        width=width,
        height=height,
        duration_seconds=duration_seconds,
        transform=transform,
        retrieved_at=utc_now(),
        warnings=warnings or [],
    )


def write_asset_credits(
    assets: list[RemotionAsset],
    *,
    output_dir: Path,
    project_root: Path,
) -> tuple[MediaReference, MediaReference]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "credits.json"
    markdown_path = output_dir / "credits.md"
    records = []
    lines = ["# Media credits", ""]

    def markdown_text(value: str) -> str:
        return re.sub(r"([\\`*_{}\[\]()#+.!|>-])", r"\\\1", value)

    for asset in assets:
        record = {
            "asset_id": asset.asset_id,
            "shot_id": asset.shot_id,
            "provider": asset.provider,
            "source_page_url": asset.source_page_url,
            "creator_name": asset.creator_name,
            "creator_url": asset.creator_url,
            "license_id": asset.rights.license_id,
            "license_name": asset.rights.license_name,
            "license_url": asset.rights.license_url,
            "attribution": asset.rights.attribution_text,
            "attribution_required": asset.rights.attribution_required,
            "share_alike": asset.rights.share_alike,
            "review_status": asset.rights.review_status,
            "review_reason": asset.rights.review_reason,
            "sha256": asset.original.sha256,
            "retrieved_at": asset.retrieved_at.isoformat(),
            "transform": asset.transform,
        }
        records.append(record)
        creator = asset.creator_name or asset.provider.replace("_", " ").title()
        attribution = asset.rights.attribution_text or creator
        source = f" ([source]({asset.source_page_url}))" if asset.source_page_url else ""
        license_link = (
            f"[{asset.rights.license_name}]({asset.rights.license_url})"
            if asset.rights.license_url
            else asset.rights.license_name
        )
        lines.append(
            f"- `{asset.asset_id}` / `{asset.shot_id}`: {markdown_text(attribution)}{source}; "
            f"{license_link}. Modified: {markdown_text(asset.transform)}."
        )
        if asset.rights.share_alike:
            lines.append("  - ShareAlike requirements apply to publication and adaptations.")
        if asset.rights.review_status == "editorial_context":
            lines.append(
                "  - Editorial-context review required: "
                + markdown_text(asset.rights.review_reason)
            )
    atomic_write_json(json_path, {"schema_version": 1, "assets": records})
    atomic_write_text(markdown_path, "\n".join(lines) + "\n")
    return (
        MediaReference(
            path=relative_path(json_path, project_root),
            sha256=sha256_file(json_path),
            mime_type="application/json",
        ),
        MediaReference(
            path=relative_path(markdown_path, project_root),
            sha256=sha256_file(markdown_path),
            mime_type="text/markdown",
        ),
    )
