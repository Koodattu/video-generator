from __future__ import annotations

import copy
import re
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


def _bounded_remotion_text(value: str, maximum: int) -> str:
    compact = " ".join(value.split())
    if len(compact) <= maximum:
        return compact
    prefix = compact[:maximum]
    if " " in prefix:
        prefix = prefix.rsplit(" ", 1)[0]
    return prefix.rstrip(" ,;:-")


def _is_contiguous_remotion_copy(value: str, narration_excerpt: str) -> bool:
    candidate = re.findall(r"\w+", value.casefold(), flags=re.UNICODE)
    source = re.findall(r"\w+", narration_excerpt.casefold(), flags=re.UNICODE)
    if not candidate:
        return True
    width = len(candidate)
    return any(
        source[index : index + width] == candidate
        for index in range(len(source) - width + 1)
    )


def _canonicalize_remotion_direction(
    input_data: dict[str, Any],
    data: dict[str, Any],
) -> dict[str, Any]:
    normalized = copy.deepcopy(data)
    templates = {
        "kinetic_hook",
        "headline_zoom",
        "source_screenshot",
        "code_reveal",
        "diagram_flow",
        "comparison_split",
        "meme_cutaway",
        "conclusion",
    }
    asset_kinds = {
        "none",
        "stock_image",
        "stock_video",
        "gif",
        "meme",
        "source_screenshot",
        "generated_image",
    }
    template = normalized.get("template")
    asset_kind = normalized.get("asset_kind")
    headline = normalized.get("headline")
    supporting_text = normalized.get("supporting_text")
    body_lines = normalized.get("body_lines")
    asset_query = normalized.get("asset_query")
    if (
        template not in templates
        or asset_kind not in asset_kinds
        or not isinstance(headline, str)
        or not isinstance(supporting_text, str)
        or not isinstance(body_lines, list)
        or not all(isinstance(line, str) for line in body_lines)
        or not isinstance(asset_query, str)
    ):
        return normalized

    shot_position = input_data.get("shot_position")
    shot_count = input_data.get("shot_count")
    positioned = (
        isinstance(shot_position, int)
        and isinstance(shot_count, int)
        and shot_count >= 2
        and 1 <= shot_position <= shot_count
    )
    if positioned and shot_position == 1:
        template = "kinetic_hook"
    elif positioned and shot_position == shot_count:
        template = "conclusion"
    elif positioned and template in {"kinetic_hook", "conclusion"}:
        template = "headline_zoom"

    headline = _bounded_remotion_text(headline, 80)
    supporting_text = _bounded_remotion_text(supporting_text, 160)
    body_lines = [
        bounded
        for line in body_lines
        if line.strip()
        if (bounded := _bounded_remotion_text(line, 120))
    ]
    asset_query = _bounded_remotion_text(asset_query, 180)
    narration_excerpt = str(input_data.get("narration_excerpt", ""))
    if not headline:
        headline = _bounded_remotion_text(narration_excerpt, 80)

    def downgrade_to_headline() -> None:
        nonlocal template, body_lines, asset_kind, asset_query
        template = "headline_zoom"
        body_lines = []
        asset_kind = "none"
        asset_query = ""

    if template == "code_reveal":
        if len(body_lines) < 2:
            downgrade_to_headline()
        else:
            body_lines = [_bounded_remotion_text(line, 70) for line in body_lines[:8]]
    elif template == "diagram_flow":
        if len(body_lines) < 2:
            downgrade_to_headline()
        else:
            body_lines = [_bounded_remotion_text(line, 32) for line in body_lines[:5]]
    elif template == "comparison_split":
        if len(body_lines) < 2:
            downgrade_to_headline()
        else:
            body_lines = [_bounded_remotion_text(line, 60) for line in body_lines[:2]]
    elif template == "source_screenshot":
        if input_data.get("content_mode") != "factual" or not input_data.get("source_options"):
            downgrade_to_headline()
        else:
            asset_kind = "source_screenshot"
            asset_query = ""
    elif template == "meme_cutaway":
        if asset_kind not in {"meme", "gif", "stock_image"} or not asset_query:
            downgrade_to_headline()

    body_templates = {"code_reveal", "diagram_flow", "comparison_split"}
    supporting_templates = {
        "kinetic_hook",
        "headline_zoom",
        "diagram_flow",
        "meme_cutaway",
        "conclusion",
    }
    asset_templates = {
        "kinetic_hook",
        "headline_zoom",
        "source_screenshot",
        "meme_cutaway",
    }
    if template not in body_templates:
        body_lines = []
    if template not in supporting_templates:
        supporting_text = ""
    if template not in asset_templates:
        asset_kind = "none"
        asset_query = ""
    elif asset_kind == "source_screenshot" and template != "source_screenshot":
        asset_kind = "none"
        asset_query = ""
    elif asset_kind in {"none", "source_screenshot"}:
        asset_query = ""
    elif not asset_query:
        if template == "meme_cutaway":
            downgrade_to_headline()
        else:
            asset_kind = "none"

    if input_data.get("content_mode") == "factual":
        if not _is_contiguous_remotion_copy(headline, narration_excerpt):
            headline = _bounded_remotion_text(narration_excerpt, 80)
        if supporting_text and not _is_contiguous_remotion_copy(
            supporting_text, narration_excerpt
        ):
            supporting_text = ""
        if template in body_templates and any(
            not _is_contiguous_remotion_copy(line, narration_excerpt)
            for line in body_lines
        ):
            downgrade_to_headline()

    normalized.update(
        template=template,
        headline=headline,
        supporting_text=supporting_text,
        body_lines=body_lines,
        asset_kind=asset_kind,
        asset_query=asset_query,
    )
    return normalized


def _canonicalize_host_owned_fields(
    task_id: str,
    input_data: dict[str, Any],
    data: dict[str, Any],
) -> dict[str, Any]:
    normalized = copy.deepcopy(data)

    if task_id == "remotion_direction":
        normalized = _canonicalize_remotion_direction(input_data, normalized)

    if task_id == "ideate":
        candidates = normalized.get("candidates")
        if isinstance(candidates, list):
            for index, candidate in enumerate(candidates, start=1):
                if isinstance(candidate, dict):
                    candidate["candidate_id"] = f"candidate-{index:03d}"

    if task_id == "outline":
        scenes = normalized.get("scenes")
        if isinstance(scenes, list):
            research_pack = input_data.get("research_pack")
            known_evidence_ids: set[str] | None = None
            if isinstance(research_pack, dict):
                known_evidence_ids = {
                    str(item[id_field])
                    for collection_name, id_field in (
                        ("evidence", "evidence_id"),
                        ("findings", "finding_id"),
                    )
                    for item in research_pack.get(collection_name, [])
                    if isinstance(item, dict) and item.get(id_field)
                }
            for index, scene in enumerate(scenes, start=1):
                if isinstance(scene, dict):
                    scene["scene_id"] = f"scene-{index:03d}"
                    if known_evidence_ids is not None and isinstance(
                        scene.get("evidence_ids"), list
                    ):
                        scene["evidence_ids"] = [
                            evidence_id
                            for evidence_id in scene["evidence_ids"]
                            if evidence_id in known_evidence_ids
                        ]

    review_types = {
        "review_story": "story",
        "review_spoken": "spoken",
        "review_constraints": "constraints",
    }
    if (
        task_id in review_types
        and input_data.get("review_strategy")
        not in {
            "single-finding-resolution-v1",
            "single-brief-constraint-v1",
            "single-remotion-plan-constraint-v1",
        }
    ):
        review_type = review_types[task_id]
        normalized["review_type"] = review_type
        findings = normalized.get("findings")
        if isinstance(findings, list):
            for index, finding in enumerate(findings, start=1):
                if isinstance(finding, dict):
                    finding["finding_id"] = f"{review_type}:finding-{index:03d}"
            normalized["passed"] = not findings

    if task_id == "claim_inventory":
        coverage_check = (
            input_data.get("coverage_strategy") == "single-scene-claim-coverage-v1"
        )
        claims_field = "missing_claims" if coverage_check else "claims"
        claims = normalized.get(claims_field)
        scene_extraction = (
            input_data.get("inventory_strategy") == "single-scene-claim-extraction-v2"
            or coverage_check
        )
        source_script = input_data.get("script")
        script_scenes = source_script.get("scenes") if isinstance(source_script, dict) else None
        research_pack = input_data.get("research_pack")
        known_evidence_ids = {
            str(item["evidence_id"])
            for item in research_pack.get("evidence", [])
            if isinstance(item, dict) and item.get("evidence_id")
        } if isinstance(research_pack, dict) else set()
        if isinstance(claims, list):
            for index, claim in enumerate(claims, start=1):
                if not isinstance(claim, dict):
                    continue
                if not scene_extraction:
                    claim["claim_id"] = f"claim-{index:03d}"
                if scene_extraction:
                    claim.pop("evidence_ids", None)
                elif isinstance(claim.get("evidence_ids"), list):
                    claim["evidence_ids"] = [
                        evidence_id
                        for evidence_id in claim["evidence_ids"]
                        if evidence_id in known_evidence_ids
                    ]
                exact_text = claim.get("exact_text")
                if (
                    not scene_extraction
                    and isinstance(exact_text, str)
                    and isinstance(script_scenes, list)
                ):
                    matching_scene_ids = [
                        scene.get("scene_id")
                        for scene in script_scenes
                        if isinstance(scene, dict)
                        and isinstance(scene.get("spoken_text"), str)
                        and exact_text in scene["spoken_text"]
                    ]
                    if len(matching_scene_ids) == 1:
                        claim["scene_id"] = matching_scene_ids[0]

    if task_id == "factual_review":
        reviews = normalized.get("claims")
        inventory = input_data.get("claim_inventory")
        inventory_claims = inventory.get("claims") if isinstance(inventory, dict) else None
        research_pack = input_data.get("research_pack")
        evidence_records = input_data.get("evidence_records")
        evidence_items = (
            evidence_records
            if isinstance(evidence_records, list)
            else research_pack.get("evidence", [])
            if isinstance(research_pack, dict)
            else []
        )
        known_evidence_ids = {
            str(item["evidence_id"])
            for item in evidence_items
            if isinstance(item, dict) and item.get("evidence_id")
        }
        if isinstance(normalized.get("evidence_ids"), list):
            normalized["evidence_ids"] = [
                evidence_id
                for evidence_id in normalized["evidence_ids"]
                if evidence_id in known_evidence_ids
            ]
        if isinstance(reviews, list) and isinstance(inventory_claims, list):
            for expected, review in zip(inventory_claims, reviews, strict=False):
                if not isinstance(expected, dict) or not isinstance(review, dict):
                    continue
                if expected.get("claim_id"):
                    review["claim_id"] = expected["claim_id"]
                if isinstance(review.get("evidence_ids"), list):
                    review["evidence_ids"] = [
                        evidence_id
                        for evidence_id in review["evidence_ids"]
                        if evidence_id in known_evidence_ids
                    ]
        if "passed" in normalized or "claims" in normalized:
            uncovered = normalized.get("uncovered_claims")
            normalized["passed"] = (
                isinstance(reviews, list)
                and len(reviews) == len(inventory_claims or [])
                and not uncovered
                and all(
                    isinstance(review, dict)
                    and review.get("verdict") in {"supported", "not_a_factual_claim"}
                    for review in reviews
                )
            )

    expected_script_scenes: Any = None
    output_script_scenes: Any = None
    if task_id == "script_draft":
        outline = input_data.get("outline")
        expected_script_scenes = outline.get("scenes") if isinstance(outline, dict) else None
        output_script_scenes = normalized.get("scenes")
    elif task_id == "script_revision":
        source_script = input_data.get("script")
        expected_script_scenes = (
            source_script.get("scenes") if isinstance(source_script, dict) else None
        )
        revised_script = normalized.get("script")
        output_script_scenes = (
            revised_script.get("scenes") if isinstance(revised_script, dict) else None
        )
    if isinstance(expected_script_scenes, list) and isinstance(output_script_scenes, list):
        for expected, output in zip(expected_script_scenes, output_script_scenes, strict=False):
            if isinstance(expected, dict) and isinstance(output, dict) and "scene_id" in expected:
                output["scene_id"] = expected["scene_id"]

    if task_id == "visual_plan":
        schedule = input_data.get("canonical_shot_schedule")
        shots = normalized.get("shots")
        if isinstance(schedule, list) and schedule and isinstance(shots, list):
            canonical_fields = (
                "shot_id",
                "scene_id",
                "narration_excerpt",
                "start_seconds",
                "end_seconds",
            )
            for expected, shot in zip(schedule, shots, strict=False):
                if not isinstance(expected, dict) or not isinstance(shot, dict):
                    continue
                for field_name in canonical_fields:
                    if field_name in expected:
                        shot[field_name] = expected[field_name]
            final = schedule[-1]
            if isinstance(final, dict) and "end_seconds" in final:
                normalized["duration_seconds"] = final["end_seconds"]
        else:
            source_script = input_data.get("script")
            expected_scenes = (
                source_script.get("scenes") if isinstance(source_script, dict) else None
            )
            visual_scenes = normalized.get("scenes")
            if isinstance(expected_scenes, list) and isinstance(visual_scenes, list):
                for expected, visual in zip(expected_scenes, visual_scenes, strict=False):
                    if isinstance(expected, dict) and isinstance(visual, dict) and "scene_id" in expected:
                        visual["scene_id"] = expected["scene_id"]

    return normalized


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
        required_finding_ids = input_data.get("required_finding_ids")
        if task_id == "script_revision" and isinstance(required_finding_ids, list):
            if not required_finding_ids:
                schema.get("properties", {})["dispositions"] = {"const": []}
            else:
                dispositions_schema = schema.get("properties", {}).get("dispositions")
                finding_id_schema = (
                    schema.get("$defs", {})
                    .get("RevisionDisposition", {})
                    .get("properties", {})
                    .get("finding_id")
                )
                if isinstance(dispositions_schema, dict):
                    dispositions_schema["minItems"] = len(required_finding_ids)
                    dispositions_schema["maxItems"] = len(required_finding_ids)
                if isinstance(finding_id_schema, dict):
                    finding_id_schema["enum"] = required_finding_ids
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
            artifact = output_model.model_validate(
                _canonicalize_host_owned_fields(task_id, input_data, data)
            )
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
            diagnostic: dict[str, Any] = {
                "type": "invariant",
                "msg": error.message,
                "loc": [],
            }
            if error.details:
                diagnostic["details"] = error.details
            return [diagnostic]

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
            word_count_guidance = ""
            for validation_error in errors:
                details = validation_error.get("details", {})
                word_delta = details.get("word_delta") if isinstance(details, dict) else None
                actual_word_count = (
                    details.get("actual_word_count") if isinstance(details, dict) else None
                )
                minimum_word_count = (
                    details.get("minimum_word_count") if isinstance(details, dict) else None
                )
                maximum_word_count = (
                    details.get("maximum_word_count") if isinstance(details, dict) else None
                )
                target_word_count = (
                    details.get("target_word_count") if isinstance(details, dict) else None
                )
                if (
                    isinstance(actual_word_count, int)
                    and isinstance(minimum_word_count, int)
                    and isinstance(maximum_word_count, int)
                    and minimum_word_count <= maximum_word_count
                ):
                    target_guidance = (
                        f", aiming near {target_word_count}"
                        if isinstance(target_word_count, int)
                        and minimum_word_count <= target_word_count <= maximum_word_count
                        else ""
                    )
                    word_count_guidance = (
                        f" The invalid script has {actual_word_count} words. Return between "
                        f"{minimum_word_count} and {maximum_word_count} whitespace-separated words "
                        f"inclusive{target_guidance}; recount before returning."
                    )
                    break
                if (
                    isinstance(word_delta, int)
                    and word_delta != 0
                    and isinstance(actual_word_count, int)
                    and isinstance(target_word_count, int)
                ):
                    direction = "add" if word_delta > 0 else "remove"
                    word_count_guidance = (
                        f" The invalid script has {actual_word_count} words. "
                        f"{direction.capitalize()} exactly {abs(word_delta)} whitespace-separated "
                        f"words across spoken_text so the repaired total is "
                        f"{target_word_count}; recount before returning."
                    )
                    break
            if word_count_guidance:
                output_fields = set(request.output_schema.get("properties", {}))
                if output_fields == {"spoken_text"}:
                    repair_instructions = (
                        "Repair only the supplied invalid spoken_text and return corrected JSON with "
                        "exactly the single spoken_text field. Preserve its facts, events, intent, and "
                        "unrelated wording. Returning unchanged text is invalid. For additions, use only "
                        "neutral clarification or connective wording and do not introduce a new claim, "
                        "event, quotation, or direction."
                        + word_count_guidance
                    )
                else:
                    repair_instructions = (
                        "Repair only the supplied invalid structured output and return only corrected "
                        "JSON. Preserve its title, Scene IDs and order, pause values, facts, events, "
                        "intent, and all unrelated wording exactly. Edit spoken_text; returning the "
                        "unchanged script is invalid. For additions, use only neutral clarification or "
                        "connective wording and do not introduce a new claim, event, quotation, or "
                        "direction."
                        + word_count_guidance
                    )
                repair_input_data = {
                    "invalid_output": result.data,
                    "validation_errors": errors,
                }
            elif (
                set(request.output_schema.get("properties", {})) == {"spoken_text"}
                and any(
                    validation_error.get("type") == "spoken_text_host_field"
                    for validation_error in errors
                )
            ):
                repair_instructions = (
                    "Repair only the supplied invalid spoken_text and return corrected JSON with "
                    "exactly the single spoken_text field. Remove every host schema field label and "
                    "its leaked value. Preserve all actual narration words, facts, events, intent, "
                    "language, and order. Return only words that should be spoken aloud."
                )
                repair_input_data = {
                    "invalid_output": result.data,
                    "validation_errors": errors,
                }
            else:
                repair_instructions = (
                    request.instructions
                    + "\n\nThe prior response failed schema or invariant validation. Repair only the output; "
                    "do not change the task or add commentary. Treat every validation error as a "
                    "hard constraint. If an error gives an inclusive numeric range, count using the "
                    "stated method and return a value comfortably inside that range."
                )
                repair_input_data = {
                    "original_input": input_data,
                    "invalid_output": result.data,
                    "validation_errors": errors,
                }
            repair_request = request.model_copy(
                update={
                    "instructions": repair_instructions,
                    "input_data": repair_input_data,
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
