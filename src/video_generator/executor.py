from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, ValidationError

from .contracts import (
    AlignmentRequest,
    AlignmentResult,
    ImageRequest,
    ImageResult,
    MusicRequest,
    MusicResult,
    SearchRequest,
    SearchResult,
    SpeechRequest,
    SpeechResult,
    StructuredTextRequest,
    StructuredTextResult,
    UsageRecord,
)
from .errors import BackendError, ErrorKind
from .prompting import PromptLibrary
from .registry import BackendRegistry
from .run_store import RunStore
from .util import hash_value


T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)


@dataclass(frozen=True)
class StructuredExecution:
    artifact: BaseModel
    result: StructuredTextResult
    prompt_version: str
    schema_hash: str


class TaskExecutor:
    def __init__(
        self,
        *,
        registry: BackendRegistry,
        store: RunStore,
        prompts: PromptLibrary,
    ) -> None:
        self.registry = registry
        self.store = store
        self.prompts = prompts
        self.config = store.config

    def _reserve(self, task_id: str, backend_id: str, amount: float | None = None) -> float:
        descriptor = self.registry.descriptor(backend_id)
        reservation = descriptor.reservation_usd if amount is None else amount
        if reservation > 0:
            self.store.reserve_cost(reservation, task_id=task_id, backend_id=backend_id)
        return reservation

    def _call(
        self,
        task_id: str,
        backend_id: str,
        callback: Callable[[], T],
        *,
        invalid_retries: int = 0,
        transient_retries: int = 2,
        reservation: float | None = None,
    ) -> T:
        invalid_attempts = 0
        transient_attempts = 0
        total_reserved = 0.0
        attempts = 0
        while True:
            total_reserved += self._reserve(task_id, backend_id, reservation)
            attempts += 1
            try:
                result = callback()
                usage = getattr(result, "usage", None)
                if isinstance(usage, UsageRecord):
                    usage.reserved_usd = total_reserved
                    if attempts > 1:
                        usage.warnings.append(f"completed after {attempts} provider attempts")
                return result
            except BackendError as exc:
                if exc.kind is ErrorKind.INVALID_OUTPUT and invalid_attempts < invalid_retries:
                    invalid_attempts += 1
                    continue
                if exc.kind is ErrorKind.TRANSIENT and transient_attempts < transient_retries:
                    delay = min(8.0, 0.75 * (2**transient_attempts))
                    transient_attempts += 1
                    time.sleep(delay)
                    continue
                raise

    def structured(
        self,
        task_id: str,
        input_data: dict[str, Any],
        output_model: type[M],
        *,
        media_inputs: list[Path] | None = None,
        target_image_backend: str | None = None,
        max_output_tokens: int = 8000,
    ) -> StructuredExecution:
        backend_id = self.config.task_bindings[task_id]
        backend = self.registry.get(backend_id)
        prompt = self.prompts.get(
            task_id,
            language=self.config.output_language,
            target_image_backend=target_image_backend,
        )
        schema = self.prompts.schema(task_id)
        schema_hash = hash_value(schema)
        request = StructuredTextRequest(
            task_id=task_id,
            instructions=prompt.instructions,
            input_data=input_data,
            output_schema=schema,
            output_language=self.config.output_language,
            max_output_tokens=max_output_tokens,
            media_inputs=[str(path.resolve()) for path in media_inputs or []],
        )
        descriptor = self.registry.descriptor(backend_id)
        text_reservation = descriptor.reservation_usd * max(
            0.5,
            max_output_tokens / 8000 + len(str(input_data)) / 50_000 + 0.25 * len(media_inputs or []),
        )
        result = self._call(
            task_id,
            backend_id,
            lambda: backend.complete(request),
            invalid_retries=1,
            reservation=text_reservation,
        )
        try:
            artifact = output_model.model_validate(result.data)
        except ValidationError as first_error:
            first_usage = result.usage.model_copy(deep=True) if result.usage else None
            repair_request = request.model_copy(
                update={
                    "instructions": (
                        request.instructions
                        + "\n\nThe prior response failed schema or invariant validation. Repair only the output; "
                        "do not change the task or add commentary."
                    ),
                    "input_data": {
                        "original_input": input_data,
                        "invalid_output": result.data,
                        "validation_errors": first_error.errors(include_url=False),
                    },
                }
            )
            result = self._call(
                task_id,
                backend_id,
                lambda: backend.complete(repair_request),
                invalid_retries=0,
                reservation=text_reservation,
            )
            try:
                artifact = output_model.model_validate(result.data)
            except ValidationError as second_error:
                raise BackendError(
                    f"{task_id} output failed validation after one repair: {second_error}",
                    kind=ErrorKind.INVALID_OUTPUT,
                ) from second_error
            if first_usage and result.usage:
                result.usage.input_units += first_usage.input_units
                result.usage.output_units += first_usage.output_units
                result.usage.reserved_usd += first_usage.reserved_usd
                result.usage.warnings.append(
                    "usage includes the schema-invalid response and one structured repair"
                )
        return StructuredExecution(artifact, result, prompt.version, schema_hash)

    def search(self, request: SearchRequest) -> SearchResult:
        backend_id = self.config.task_bindings["search"]
        backend = self.registry.get(backend_id)
        return self._call("search", backend_id, lambda: backend.search(request), transient_retries=2)

    def speech(self, request: SpeechRequest) -> SpeechResult:
        backend_id = self.config.task_bindings["narration_synthesis"]
        backend = self.registry.get(backend_id)
        character_factor = max(1.0, len(request.text) / 1000)
        reservation = self.registry.descriptor(backend_id).reservation_usd * character_factor
        return self._call(
            "narration_synthesis",
            backend_id,
            lambda: backend.synthesize(request),
            transient_retries=1,
            reservation=reservation,
        )

    def align(self, request: AlignmentRequest) -> AlignmentResult:
        backend_id = self.config.task_bindings["caption_alignment"]
        backend = self.registry.get(backend_id)
        return self._call(
            "caption_alignment",
            backend_id,
            lambda: backend.align(request),
            transient_retries=1,
        )

    def image(self, request: ImageRequest, output_path: Path) -> ImageResult:
        backend_id = self.config.task_bindings["image_generate"]
        backend = self.registry.get(backend_id)
        descriptor = self.registry.descriptor(backend_id)
        quality_factor = {"low": 0.6, "medium": 1.0, "high": 2.0}.get(request.quality, 1.0)
        pixel_factor = max(1.0, request.width * request.height / (2048 * 1152))
        reference_factor = 1.0 + 0.25 * len(request.reference_paths)
        reservation = descriptor.reservation_usd * quality_factor * pixel_factor * reference_factor
        return self._call(
            "image_generate",
            backend_id,
            lambda: backend.generate(request, output_path),
            transient_retries=1,
            reservation=reservation,
        )

    def music(self, request: MusicRequest) -> MusicResult:
        backend_id = self.config.task_bindings["music_generate"]
        backend = self.registry.get(backend_id)
        duration_factor = max(1.0, request.brief.requested_duration_seconds / 60)
        reservation = self.registry.descriptor(backend_id).reservation_usd * duration_factor
        return self._call(
            "music_generate",
            backend_id,
            lambda: backend.generate(request),
            transient_retries=1,
            reservation=reservation,
        )


def result_usage(result: Any, *, task_id: str, backend_id: str) -> UsageRecord:
    value = getattr(result, "usage", None)
    if isinstance(value, UsageRecord):
        return value
    return UsageRecord(task_id=task_id, backend_id=backend_id)
