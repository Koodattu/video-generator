from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..contracts import (
    AlignmentRequest,
    AlignmentResult,
    ImageAsset,
    ImageRequest,
    ImageResult,
    MediaReference,
    MusicAsset,
    MusicRequest,
    MusicResult,
    ProbeReport,
    SpeechAsset,
    SpeechRequest,
    SpeechResult,
    StructuredTextRequest,
    StructuredTextResult,
    UsageRecord,
    WordTiming,
)
from ..errors import BackendError, ErrorKind
from ..profiles import BACKEND_DESCRIPTORS
from ..runners import RunnerManager
from ..util import image_dimensions, relative_path, sha256_file
from .base import Backend


def _usage(
    value: Any,
    *,
    task_id: str,
    backend_id: str,
    provider_request_id: str = "",
) -> UsageRecord:
    data = value if isinstance(value, dict) else {}
    return UsageRecord(
        task_id=task_id,
        backend_id=backend_id,
        provider_request_id=provider_request_id,
        input_units=float(data.get("input_units") or data.get("input_tokens") or 0),
        output_units=float(data.get("output_units") or data.get("output_tokens") or 0),
        elapsed_seconds=float(data.get("elapsed_seconds") or 0),
        peak_vram_mb=float(data["peak_vram_mb"]) if data.get("peak_vram_mb") is not None else None,
        warnings=[str(item) for item in data.get("warnings", [])],
    )


class _LocalBackend(Backend):
    def __init__(self, backend_id: str, manager: RunnerManager) -> None:
        self.descriptor = BACKEND_DESCRIPTORS[backend_id]
        self.manager = manager
        self.project_root = manager.project_root
        self.run_root = manager.run_root

    def probe(self, *, live: bool = False) -> ProbeReport:
        return self.manager.probe(self.descriptor.backend_id, live=live)

    def _media_path(self, value: str) -> Path:
        path = (self.project_root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
        try:
            path.relative_to(self.run_root)
        except ValueError as exc:
            raise BackendError("runner returned media outside the Run workspace", kind=ErrorKind.INVALID_OUTPUT) from exc
        if not path.is_file():
            raise BackendError(f"runner did not create expected media: {path}", kind=ErrorKind.INVALID_OUTPUT)
        return path

    def _relative(self, value: str | Path) -> str:
        raw = Path(value)
        path = raw.resolve() if raw.is_absolute() else (self.project_root / raw).resolve()
        try:
            path.relative_to(self.run_root)
        except ValueError as exc:
            raise BackendError("runner media input is outside the Run workspace", kind=ErrorKind.UNSUPPORTED) from exc
        if not path.is_file():
            raise BackendError(f"runner media input does not exist: {path}", kind=ErrorKind.NOT_READY)
        return relative_path(path, self.project_root)

    def _output_relative(self, value: str | Path) -> str:
        raw = Path(value)
        path = raw.resolve() if raw.is_absolute() else (self.project_root / raw).resolve()
        try:
            path.relative_to(self.run_root)
        except ValueError as exc:
            raise BackendError("runner output is outside the Run workspace", kind=ErrorKind.UNSUPPORTED) from exc
        return relative_path(path, self.project_root)

    def _voice_path(self, value: str) -> str:
        if not value:
            return ""
        path = (self.project_root / value).resolve()
        try:
            path.relative_to(self.project_root / "private")
        except ValueError as exc:
            raise BackendError("voice reference is outside private/", kind=ErrorKind.UNSUPPORTED) from exc
        if not path.is_file():
            raise BackendError(f"voice reference does not exist: {path}", kind=ErrorKind.NOT_READY)
        return relative_path(path, self.project_root)


class LocalStructuredTextBackend(_LocalBackend):
    def complete(self, request: StructuredTextRequest) -> StructuredTextResult:
        media_inputs = [self._relative(value) for value in request.media_inputs]
        result = self.manager.invoke(
            self.descriptor.backend_id,
            "structured_text.complete",
            {
                **request.model_dump(mode="json", exclude={"media_inputs"}),
                "media_inputs": media_inputs,
            },
        )
        data = result.get("data")
        if not isinstance(data, dict):
            raise BackendError("local text runner returned no structured data", kind=ErrorKind.INVALID_OUTPUT)
        provider_request_id = str(result.get("provider_request_id") or "")
        return StructuredTextResult(
            data=data,
            raw_response={
                "runtime_revision": result.get("runtime_revision"),
                "model_revision": result.get("model_revision"),
                "model_id": result.get("model_id"),
                "profile_id": result.get("profile_id"),
                "context_size": result.get("context_size"),
                "speculation": result.get("speculation"),
                "startup_elapsed_seconds": result.get("startup_elapsed_seconds"),
                "server_timings": result.get("server_timings"),
            },
            provider_request_id=provider_request_id,
            usage=_usage(
                result.get("usage"),
                task_id=request.task_id,
                backend_id=self.descriptor.backend_id,
                provider_request_id=provider_request_id,
            ),
        )


class LocalSpeechBackend(_LocalBackend):
    def synthesize(self, request: SpeechRequest) -> SpeechResult:
        safe_request = request.model_copy(
            update={
                "output_path": self._output_relative(request.output_path),
                "voice": request.voice.model_copy(
                    update={
                        "reference_audio": self._voice_path(request.voice.reference_audio),
                        "reference_transcript": self._voice_path(request.voice.reference_transcript),
                    }
                ),
            }
        )
        result = self.manager.invoke(
            self.descriptor.backend_id,
            "speech.synthesize",
            safe_request.model_dump(mode="json"),
        )
        path = self._media_path(str(result.get("audio_path") or request.output_path))
        try:
            words = [WordTiming.model_validate(item) for item in result.get("word_timings", [])]
            asset = SpeechAsset(
                scene_id=request.scene_id,
                audio=MediaReference(
                    path=relative_path(path, self.project_root),
                    sha256=sha256_file(path),
                    mime_type=str(result.get("mime_type") or "audio/wav"),
                ),
                duration_seconds=float(result["duration_seconds"]),
                sample_rate=int(result["sample_rate"]),
                channels=int(result["channels"]),
                word_timings=words,
                timing_precision=str(result.get("timing_precision") or "none"),
                provider_request_id=str(result.get("provider_request_id") or ""),
            )
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            raise BackendError("local speech runner returned invalid metadata", kind=ErrorKind.INVALID_OUTPUT) from exc
        return SpeechResult(
            asset=asset,
            usage=_usage(
                result.get("usage"),
                task_id="narration_synthesis",
                backend_id=self.descriptor.backend_id,
                provider_request_id=asset.provider_request_id,
            ),
        )


class LocalAlignmentBackend(_LocalBackend):
    def align(self, request: AlignmentRequest) -> AlignmentResult:
        safe_request = request.model_copy(update={"audio_path": self._relative(request.audio_path)})
        result = self.manager.invoke(
            self.descriptor.backend_id,
            "alignment.align",
            safe_request.model_dump(mode="json"),
        )
        try:
            words = [WordTiming.model_validate(item) for item in result["recognized_words"]]
        except (KeyError, TypeError, ValidationError) as exc:
            raise BackendError("local alignment runner returned invalid word timings", kind=ErrorKind.INVALID_OUTPUT) from exc
        provider_request_id = str(result.get("provider_request_id") or "")
        return AlignmentResult(
            recognized_words=words,
            provider_request_id=provider_request_id,
            usage=_usage(
                result.get("usage"),
                task_id="caption_alignment",
                backend_id=self.descriptor.backend_id,
                provider_request_id=provider_request_id,
            ),
        )


class LocalImageBackend(_LocalBackend):
    def generate(self, request: ImageRequest, output_path: Path) -> ImageResult:
        output_path = output_path.resolve()
        output_path.relative_to(self.run_root)
        payload = request.model_dump(mode="json")
        payload["output_path"] = relative_path(output_path, self.project_root)
        payload["reference_paths"] = [self._relative(value) for value in request.reference_paths]
        result = self.manager.invoke(self.descriptor.backend_id, "image.generate", payload)
        path = self._media_path(str(result.get("image_path") or payload["output_path"]))
        try:
            width, height = image_dimensions(path)
        except ValueError as exc:
            raise BackendError(str(exc), kind=ErrorKind.INVALID_OUTPUT) from exc
        provider_request_id = str(result.get("provider_request_id") or "")
        asset = ImageAsset(
            scene_id=request.scene_id,
            shot_id=getattr(request, "shot_id", None),
            image=MediaReference(
                path=relative_path(path, self.project_root),
                sha256=sha256_file(path),
                mime_type=str(result.get("mime_type") or "image/png"),
            ),
            width=width,
            height=height,
            generation_settings=result.get("generation_settings")
            if isinstance(result.get("generation_settings"), dict)
            else {},
            provider_request_id=provider_request_id,
        )
        return ImageResult(
            asset=asset,
            usage=_usage(
                result.get("usage"),
                task_id="image_generate",
                backend_id=self.descriptor.backend_id,
                provider_request_id=provider_request_id,
            ),
        )


class LocalMusicBackend(_LocalBackend):
    def generate(self, request: MusicRequest) -> MusicResult:
        safe_request = request.model_copy(update={"output_path": self._output_relative(request.output_path)})
        result = self.manager.invoke(
            self.descriptor.backend_id,
            "music.generate",
            safe_request.model_dump(mode="json"),
        )
        path = self._media_path(str(result.get("audio_path") or request.output_path))
        try:
            asset = MusicAsset(
                audio=MediaReference(
                    path=relative_path(path, self.project_root),
                    sha256=sha256_file(path),
                    mime_type=str(result.get("mime_type") or "audio/wav"),
                ),
                duration_seconds=float(result["duration_seconds"]),
                looped=bool(result.get("looped", False)),
                provider_request_id=str(result.get("provider_request_id") or ""),
            )
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            raise BackendError("local music runner returned invalid metadata", kind=ErrorKind.INVALID_OUTPUT) from exc
        return MusicResult(
            asset=asset,
            usage=_usage(
                result.get("usage"),
                task_id="music_generate",
                backend_id=self.descriptor.backend_id,
                provider_request_id=asset.provider_request_id,
            ),
        )
