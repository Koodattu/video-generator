from __future__ import annotations

import base64
import re
import urllib.parse
from pathlib import Path
from typing import Any

from ..contracts import (
    AlignmentRequest,
    AlignmentResult,
    MediaReference,
    MusicAsset,
    MusicRequest,
    MusicResult,
    ProbeItem,
    ProbeReport,
    SpeechAsset,
    SpeechRequest,
    SpeechResult,
    UsageRecord,
    WordTiming,
)
from ..errors import BackendError, ErrorKind
from ..media import MediaTools
from ..net import HttpClient, multipart_body
from ..profiles import BACKEND_DESCRIPTORS
from ..util import atomic_write_bytes, relative_path, sha256_file
from .base import Backend


def character_alignment_to_words(text: str, alignment: dict[str, Any]) -> list[WordTiming]:
    characters = alignment.get("characters")
    starts = alignment.get("character_start_times_seconds")
    ends = alignment.get("character_end_times_seconds")
    if not isinstance(characters, list) or not isinstance(starts, list) or not isinstance(ends, list):
        raise BackendError("TTS alignment arrays are missing", kind=ErrorKind.INVALID_OUTPUT)
    if not (len(characters) == len(starts) == len(ends)):
        raise BackendError("TTS alignment arrays have different lengths", kind=ErrorKind.INVALID_OUTPUT)
    aligned_text = "".join(str(character) for character in characters)
    if aligned_text != text:
        raise BackendError(
            "TTS character alignment does not match the canonical Scene text",
            kind=ErrorKind.INVALID_OUTPUT,
        )
    result = []
    for match in re.finditer(r"\S+", text, flags=re.UNICODE):
        first, last = match.start(), match.end() - 1
        try:
            start = float(starts[first])
            end = float(ends[last])
        except (IndexError, TypeError, ValueError) as exc:
            raise BackendError("TTS alignment contains invalid timings", kind=ErrorKind.INVALID_OUTPUT) from exc
        if end < start:
            raise BackendError("TTS alignment is not monotonic", kind=ErrorKind.INVALID_OUTPUT)
        result.append(WordTiming(text=match.group(0), start_seconds=start, end_seconds=end, confidence=1.0))
    return result


class _ElevenLabsClient(Backend):
    api_base = "https://api.elevenlabs.io/v1"

    def __init__(
        self,
        api_key: str,
        workspace_root: Path,
        run_root: Path,
        *,
        configured_voice_id: str = "",
        http: HttpClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.workspace_root = workspace_root.resolve()
        self.run_root = run_root.resolve()
        self.configured_voice_id = configured_voice_id
        self.http = http or HttpClient(timeout_seconds=600, max_response_bytes=500_000_000)
        self.media = MediaTools.discover()

    @property
    def headers(self) -> dict[str, str]:
        return {"xi-api-key": self.api_key}

    def _probe(self, model_id: str, *, live: bool, voice_id: str | None = None) -> ProbeReport:
        configured = bool(self.api_key)
        items = [
            ProbeItem(
                name="credential",
                ready=configured,
                detail="ELEVENLABS_API_KEY is configured" if configured else "ELEVENLABS_API_KEY is missing",
                action=None if configured else "Add ELEVENLABS_API_KEY to .env.",
            )
        ]
        if configured and live:
            try:
                payload = self.http.request("GET", f"{self.api_base}/models", headers=self.headers).json_value()
                models = payload if isinstance(payload, list) else payload.get("models", [])
                found = any(
                    isinstance(item, dict) and item.get("model_id") == model_id for item in models
                )
                items.append(
                    ProbeItem(
                        name="model_access",
                        ready=found,
                        detail=f"model access confirmed for {model_id}" if found else f"model {model_id} was not listed",
                        action=None if found else f"Confirm that the ElevenLabs account can access {model_id}.",
                    )
                )
            except BackendError as exc:
                items.append(
                    ProbeItem(name="model_access", ready=False, detail=exc.message, action="Check the API key and account access.")
                )
            if voice_id is not None:
                if not voice_id:
                    items.append(
                        ProbeItem(
                            name="voice_access",
                            ready=False,
                            detail="ElevenLabs voice ID is missing",
                            action=(
                                "Set ELEVENLABS_VOICE_ID in .env or "
                                "voice.elevenlabs_voice_id in config.toml."
                            ),
                        )
                    )
                else:
                    try:
                        encoded_voice_id = urllib.parse.quote(voice_id, safe="")
                        payload = self.http.request(
                            "GET",
                            f"{self.api_base}/voices/{encoded_voice_id}",
                            headers=self.headers,
                        ).json()
                        returned_voice_id = str(payload.get("voice_id") or "")
                        items.append(
                            ProbeItem(
                                name="voice_access",
                                ready=returned_voice_id == voice_id,
                                detail=(
                                    f"voice access confirmed for {voice_id}"
                                    if returned_voice_id == voice_id
                                    else "ElevenLabs returned a different voice identity"
                                ),
                                action=(
                                    None
                                    if returned_voice_id == voice_id
                                    else "Confirm voice.elevenlabs_voice_id and account access."
                                ),
                            )
                        )
                    except BackendError as exc:
                        items.append(
                            ProbeItem(
                                name="voice_access",
                                ready=False,
                                detail=exc.message,
                                action="Confirm voice.elevenlabs_voice_id and account access.",
                            )
                        )
        return ProbeReport(
            backend_id=self.descriptor.backend_id,
            ready=all(item.ready for item in items),
            items=items,
        )


class ElevenLabsSpeechBackend(_ElevenLabsClient):
    descriptor = BACKEND_DESCRIPTORS["elevenlabs:eleven_multilingual_v2"]

    def probe(self, *, live: bool = False) -> ProbeReport:
        return self._probe(
            self.descriptor.model_id,
            live=live,
            voice_id=self.configured_voice_id,
        )

    def synthesize(self, request: SpeechRequest) -> SpeechResult:
        voice_id = request.voice.elevenlabs_voice_id
        if not voice_id:
            raise BackendError("ElevenLabs voice ID is not configured", kind=ErrorKind.NOT_READY)
        encoded_voice_id = urllib.parse.quote(voice_id, safe="")
        body: dict[str, Any] = {
            "text": request.text,
            "model_id": self.descriptor.model_id,
            "language_code": request.output_language.value,
        }
        if request.preceding_text:
            body["previous_text"] = request.preceding_text
        if request.following_text:
            body["next_text"] = request.following_text
        response = self.http.request(
            "POST",
            f"{self.api_base}/text-to-speech/{encoded_voice_id}/with-timestamps?output_format=mp3_44100_128",
            headers=self.headers,
            json_body=body,
            max_response_bytes=100_000_000,
        )
        payload = response.json()
        try:
            audio_bytes = base64.b64decode(payload["audio_base64"], validate=True)
        except (KeyError, TypeError, ValueError) as exc:
            raise BackendError("ElevenLabs TTS response did not contain valid audio", kind=ErrorKind.INVALID_OUTPUT) from exc
        alignment = payload.get("alignment")
        if not isinstance(alignment, dict):
            raise BackendError("ElevenLabs TTS did not return character timing", kind=ErrorKind.INVALID_OUTPUT)
        words = character_alignment_to_words(request.text, alignment)
        if not words:
            raise BackendError("ElevenLabs TTS returned no timed words", kind=ErrorKind.INVALID_OUTPUT)
        output_path = (self.workspace_root / request.output_path).resolve()
        try:
            output_path.relative_to(self.run_root)
        except ValueError as exc:
            raise BackendError("speech output is outside the Run workspace", kind=ErrorKind.UNSUPPORTED) from exc
        atomic_write_bytes(output_path, audio_bytes)
        probe = self.media.probe_audio(output_path)
        alignment_duration = max(float(value) for value in alignment["character_end_times_seconds"])
        if alignment_duration > probe.duration_seconds + 0.08:
            raise BackendError("TTS alignment extends beyond the written audio", kind=ErrorKind.INVALID_OUTPUT)
        duration = probe.duration_seconds
        provider_request_id = response.headers.get("request-id") or response.headers.get("x-request-id") or ""
        character_cost = response.headers.get("character-cost")
        return SpeechResult(
            asset=SpeechAsset(
                scene_id=request.scene_id,
                audio=MediaReference(
                    path=relative_path(output_path, self.workspace_root),
                    sha256=sha256_file(output_path),
                    mime_type="audio/mpeg",
                ),
                duration_seconds=duration,
                sample_rate=probe.sample_rate,
                channels=probe.channels,
                word_timings=words,
                timing_precision="character",
                provider_request_id=provider_request_id,
            ),
            usage=UsageRecord(
                task_id="narration_synthesis",
                backend_id=self.descriptor.backend_id,
                provider_request_id=provider_request_id,
                input_units=float(character_cost or len(request.text)),
                output_units=duration,
                reserved_usd=self.descriptor.reservation_usd,
            ),
        )


class ElevenLabsAlignmentBackend(_ElevenLabsClient):
    descriptor = BACKEND_DESCRIPTORS["elevenlabs:forced-alignment"]

    def probe(self, *, live: bool = False) -> ProbeReport:
        configured = bool(self.api_key)
        return ProbeReport(
            backend_id=self.descriptor.backend_id,
            ready=configured,
            items=[
                ProbeItem(
                    name="credential",
                    ready=configured,
                    detail="ELEVENLABS_API_KEY is configured" if configured else "ELEVENLABS_API_KEY is missing",
                    action=None if configured else "Add ELEVENLABS_API_KEY to .env.",
                )
            ],
        )

    def align(self, request: AlignmentRequest) -> AlignmentResult:
        audio_path = (self.workspace_root / request.audio_path).resolve()
        try:
            audio_path.relative_to(self.run_root)
        except ValueError as exc:
            raise BackendError("alignment input is outside the Run workspace", kind=ErrorKind.UNSUPPORTED) from exc
        body, content_type = multipart_body({"text": request.transcript}, [("file", audio_path, None)])
        response = self.http.request(
            "POST",
            f"{self.api_base}/forced-alignment",
            headers={**self.headers, "Content-Type": content_type},
            body=body,
            max_response_bytes=20_000_000,
        )
        payload = response.json()
        words = []
        for item in payload.get("words", []):
            if not isinstance(item, dict):
                continue
            try:
                text = str(item["text"])
                start = float(item["start"])
                end = float(item["end"])
                loss = float(item.get("loss") or 0)
            except (KeyError, TypeError, ValueError) as exc:
                raise BackendError("forced alignment returned malformed word timing", kind=ErrorKind.INVALID_OUTPUT) from exc
            if not text or end < start:
                raise BackendError("forced alignment returned invalid word timing", kind=ErrorKind.INVALID_OUTPUT)
            words.append(
                WordTiming(
                    text=text,
                    start_seconds=start,
                    end_seconds=end,
                    confidence=max(0.0, min(1.0, 1.0 - loss)),
                )
            )
        if not words:
            raise BackendError("forced alignment returned no words", kind=ErrorKind.INVALID_OUTPUT)
        provider_request_id = response.headers.get("request-id") or response.headers.get("x-request-id") or ""
        return AlignmentResult(
            recognized_words=words,
            provider_request_id=provider_request_id,
            usage=UsageRecord(
                task_id="caption_alignment",
                backend_id=self.descriptor.backend_id,
                provider_request_id=provider_request_id,
                input_units=len(request.transcript),
                reserved_usd=self.descriptor.reservation_usd,
            ),
        )


class ElevenLabsMusicBackend(_ElevenLabsClient):
    descriptor = BACKEND_DESCRIPTORS["elevenlabs:music_v2"]

    def probe(self, *, live: bool = False) -> ProbeReport:
        return self._probe(self.descriptor.model_id, live=live)

    def generate(self, request: MusicRequest) -> MusicResult:
        duration = request.brief.requested_duration_seconds
        maximum = float(self.descriptor.max_duration_seconds or 600)
        if duration > maximum:
            raise BackendError(
                f"the built-in ElevenLabs Music descriptor is capped at {maximum:.0f} seconds",
                kind=ErrorKind.UNSUPPORTED,
                action="Use a shorter loop segment or another explicitly configured Music Backend.",
            )
        response = self.http.request(
            "POST",
            f"{self.api_base}/music?output_format=mp3_48000_192",
            headers=self.headers,
            json_body={
                "prompt": request.brief.prompt,
                "music_length_ms": round(duration * 1000),
                "model_id": self.descriptor.model_id,
                "force_instrumental": True,
                "sign_with_c2pa": False,
            },
            max_response_bytes=300_000_000,
        )
        output_path = (self.workspace_root / request.output_path).resolve()
        try:
            output_path.relative_to(self.run_root)
        except ValueError as exc:
            raise BackendError("music output is outside the Run workspace", kind=ErrorKind.UNSUPPORTED) from exc
        atomic_write_bytes(output_path, response.body)
        probe = self.media.probe_audio(output_path)
        provider_request_id = response.headers.get("request-id") or response.headers.get("song-id") or ""
        return MusicResult(
            asset=MusicAsset(
                audio=MediaReference(
                    path=relative_path(output_path, self.workspace_root),
                    sha256=sha256_file(output_path),
                    mime_type="audio/mpeg",
                ),
                duration_seconds=probe.duration_seconds,
                provider_request_id=provider_request_id,
            ),
            usage=UsageRecord(
                task_id="music_generate",
                backend_id=self.descriptor.backend_id,
                provider_request_id=provider_request_id,
                output_units=probe.duration_seconds,
                reserved_usd=self.descriptor.reservation_usd,
            ),
        )
