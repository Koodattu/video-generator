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
from .costs import calculate_list_price
from .errors import BackendError, ErrorKind
from .prompting import PromptLibrary, task_output_language
from .registry import BackendRegistry
from .run_store import RunStore
from .schema import restricted_json_schema
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
        frozen_assets = getattr(store, "frozen_assets", {})
        profile = frozen_assets.get("profile") if isinstance(frozen_assets, dict) else None
        self.pricing_catalog = (
            profile.get("pricing_catalog")
            if isinstance(profile, dict) and isinstance(profile.get("pricing_catalog"), dict)
            else None
        )

    def _price_usage(self, usage: UsageRecord) -> UsageRecord:
        if self.pricing_catalog is None:
            return usage.model_copy(
                update={
                    "estimated_usd": None,
                    "cost_status": "unpriced",
                    "pricing_snapshot": self.config.pricing_snapshot,
                    "cost_basis": "legacy Run has no frozen pricing catalog",
                    "warnings": list(
                        dict.fromkeys(
                            [*usage.warnings, "cost unavailable because this Run predates frozen pricing tables"]
                        )
                    ),
                }
            )
        return calculate_list_price(usage, catalog=self.pricing_catalog)

    def _reserve(
        self, task_id: str, backend_id: str, amount: float | None = None
    ) -> tuple[float, str]:
        descriptor = self.registry.descriptor(backend_id)
        reservation = descriptor.reservation_usd if amount is None else amount
        call_id = ""
        if getattr(descriptor, "cloud", False):
            call_id = self.store.reserve_cost(
                reservation, task_id=task_id, backend_id=backend_id
            )
        return reservation, call_id

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
        attempts = 0
        while True:
            attempt_reserved, call_id = self._reserve(task_id, backend_id, reservation)
            attempts += 1
            started = time.perf_counter()
            try:
                result = callback()
                elapsed = time.perf_counter() - started
                usage = getattr(result, "usage", None)
                if isinstance(usage, UsageRecord):
                    usage = usage.model_copy(
                        update={
                            "call_id": call_id,
                            "reserved_usd": attempt_reserved,
                            "elapsed_seconds": usage.elapsed_seconds or elapsed,
                        }
                    )
                    if call_id:
                        usage = self._price_usage(usage)
                    if attempts > 1:
                        usage.warnings.append(f"completed after {attempts} provider attempts")
                    if call_id:
                        self.store.settle_cost(call_id, usage)
                    setattr(result, "usage", usage)
                elif call_id:
                    usage = self._price_usage(
                        UsageRecord(
                            task_id=task_id,
                            backend_id=backend_id,
                            call_id=call_id,
                            reserved_usd=attempt_reserved,
                            elapsed_seconds=elapsed,
                            warnings=["provider response did not include usage metadata"],
                        )
                    )
                    self.store.settle_cost(call_id, usage)
                    if hasattr(result, "usage"):
                        setattr(result, "usage", usage)
                return result
            except BaseException as exc:
                if call_id:
                    self.store.mark_cost_unresolved(
                        call_id,
                        elapsed_seconds=time.perf_counter() - started,
                        error=exc,
                    )
                if (
                    isinstance(exc, BackendError)
                    and exc.kind is ErrorKind.INVALID_OUTPUT
                    and invalid_attempts < invalid_retries
                ):
                    invalid_attempts += 1
                    continue
                if (
                    isinstance(exc, BackendError)
                    and exc.kind is ErrorKind.TRANSIENT
                    and transient_attempts < transient_retries
                ):
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
        invariant: Callable[[M], None] | None = None,
        instruction_suffix: str = "",
    ) -> StructuredExecution:
        backend_id = self.config.task_bindings[task_id]
        backend = self.registry.get(backend_id)
        output_language = (
            self.prompts.output_language(task_id, self.config.output_language)
            if hasattr(self.prompts, "output_language")
            else task_output_language(task_id, self.config.output_language)
        )
        prompt = self.prompts.get(
            task_id,
            language=self.config.output_language,
            target_image_backend=target_image_backend,
        )
        schema = restricted_json_schema(output_model.model_json_schema())
        schema_hash = hash_value(schema)
        request = StructuredTextRequest(
            task_id=task_id,
            instructions=(
                prompt.instructions
                + ("\n\n" + instruction_suffix.strip() if instruction_suffix.strip() else "")
            ),
            input_data=input_data,
            output_schema=schema,
            output_language=output_language,
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
        def validate(data: dict[str, Any]) -> M:
            artifact = output_model.model_validate(data)
            if invariant:
                invariant(artifact)
            return artifact

        def validation_errors(error: ValidationError | BackendError) -> list[dict[str, Any]]:
            if isinstance(error, ValidationError):
                return error.errors(
                    include_url=False,
                    include_context=False,
                    include_input=False,
                )
            if error.kind is not ErrorKind.INVALID_OUTPUT:
                raise error
            return [{"type": "invariant", "msg": error.message, "loc": []}]

        cloud_length_sensitive_tasks = {
            "script_draft",
            "script_revision",
            "duration_repair",
        }
        maximum_validation_repairs = (
            1
            if getattr(descriptor, "cloud", False)
            and task_id not in cloud_length_sensitive_tasks
            else 2
        )
        prior_usage: list[UsageRecord] = []
        repair_count = 0
        while True:
            try:
                artifact = validate(result.data)
                break
            except (ValidationError, BackendError) as error:
                errors = validation_errors(error)
                if repair_count >= maximum_validation_repairs:
                    repair_label = (
                        "one repair"
                        if maximum_validation_repairs == 1
                        else f"{maximum_validation_repairs} repairs"
                    )
                    raise BackendError(
                        f"{task_id} output failed validation after {repair_label}: {error}",
                        kind=ErrorKind.INVALID_OUTPUT,
                    ) from error
                if result.usage:
                    prior_usage.append(result.usage.model_copy(deep=True))
                repair_count += 1
            repair_request = request.model_copy(
                update={
                    "instructions": (
                        request.instructions
                        + "\n\nThe prior response failed schema or invariant validation. Repair only the output; "
                        "do not change the task or add commentary. Treat every validation error as a "
                        "hard constraint. If an error gives an inclusive numeric range, count using the "
                        "stated method and return a value comfortably inside that range."
                    ),
                    "input_data": {
                        "original_input": input_data,
                        "invalid_output": result.data,
                        "validation_errors": errors,
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
        if prior_usage and result.usage:
            for usage in prior_usage:
                result.usage.input_units += usage.input_units
                result.usage.output_units += usage.output_units
                result.usage.reserved_usd += usage.reserved_usd
                result.usage.elapsed_seconds += usage.elapsed_seconds
                for unit_name, amount in usage.billable_units.items():
                    result.usage.billable_units[unit_name] = (
                        result.usage.billable_units.get(unit_name, 0) + amount
                    )
                if usage.estimated_usd is not None:
                    result.usage.estimated_usd = (
                        (result.usage.estimated_usd or 0) + usage.estimated_usd
                    )
                if usage.actual_usd is not None:
                    result.usage.actual_usd = (
                        (result.usage.actual_usd or 0) + usage.actual_usd
                    )
            result.usage.call_id = ""
            result.usage.warnings.append(
                f"usage includes {len(prior_usage)} invalid structured response(s)"
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
