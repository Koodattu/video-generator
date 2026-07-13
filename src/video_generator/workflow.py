from __future__ import annotations

import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from langdetect import DetectorFactory, LangDetectException, detect
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import (
    AlignmentRequest,
    BackendDescriptor,
    CaptionTrack,
    CandidateSet,
    ClaimInventory,
    ContentFormat,
    ContentMode,
    CreativeBrief,
    DeliveryManifest,
    ExplainerCandidateSet,
    ExplainerOutline,
    ExplainerSelectionReport,
    FailurePolicy,
    FactualResearchPack,
    FactualRevisedScript,
    FactualReviewReport,
    ImageAsset,
    ImageRequest,
    MediaReference,
    MAX_CADENCED_SHOTS,
    MusicAsset,
    MusicBrief,
    MusicRequest,
    NarrationScript,
    NarrationTimeline,
    PUBLIC_STAGES,
    Quality,
    RenderPlan,
    RenderScene,
    ResearchPack,
    ResearchSource,
    ReviewReport,
    RevisedScript,
    SearchRequest,
    SelectionReport,
    SpeechAsset,
    SpeechRequest,
    StoryOutline,
    TimedImageRequest,
    TimedVisualPlan,
    TimedVisualReviewItem,
    UsageRecord,
    VisualPlan,
    VisualReviewItem,
    VisualReviewReport,
    VisualShotMode,
    WordTiming,
)
from .errors import BackendError, ErrorKind, MediaError, VideoGeneratorError
from .executor import StructuredExecution, TaskExecutor, result_usage
from .media import (
    AudioProbe,
    MediaTools,
    adjust_audio_tempo,
    build_timeline,
    caption_track_from_timeline,
    concatenate_audio,
    delivery_ceiling,
    delivery_manifest,
    duration_is_accepted,
    fit_music,
    normalize_audio,
    normalize_image,
    qc_video,
    reconcile_word_timings,
    render_video,
    write_ass,
    write_srt,
)
from .prompting import PromptLibrary
from .profiles import image_generation_dimensions
from .provenance import verify_runtime_snapshot
from .registry import BackendRegistry
from .run_store import RunStore
from .schema import restricted_json_schema
from .util import (
    atomic_write_json,
    hash_run_input,
    hash_value,
    relative_path,
    replace_path,
    sha256_file,
)


INTERNAL_REVISION = "media-workflow-v9"
MULTI_FORMAT_INTERNAL_REVISION = "media-workflow-v10"
LEGACY_INTERNAL_REVISION = "media-workflow-v8"
MAX_AUTHORED_SCENE_PAUSE_SECONDS = 0.75
DetectorFactory.seed = 0

CandidateSetLike = CandidateSet | ExplainerCandidateSet
SelectionReportLike = SelectionReport | ExplainerSelectionReport
OutlineLike = StoryOutline | ExplainerOutline
VisualPlanLike = VisualPlan | TimedVisualPlan
ImageRequestLike = ImageRequest | TimedImageRequest
VisualReviewItemLike = VisualReviewItem | TimedVisualReviewItem


def _raw_image_extension(backend_id: str) -> str:
    if backend_id == "deterministic:stick":
        return ".ppm"
    if backend_id == "gemini:gemini-3.1-flash-image":
        return ".jpg"
    return ".png"


class WorkflowModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1


class NarrationItem(WorkflowModel):
    speech: SpeechAsset
    normalized_audio: MediaReference
    normalized_duration_seconds: float
    normalized_sample_rate: int
    normalized_channels: int


class NarrationBundle(WorkflowModel):
    script: NarrationScript
    timeline: NarrationTimeline
    items: list[NarrationItem]
    duration_repaired: bool = False
    tempo_adjustment: float = 1.0


class AlignedSceneWords(WorkflowModel):
    scene_id: str
    words: list[WordTiming]
    coverage: float


class ExpandedSceneText(WorkflowModel):
    scene_id: str
    spoken_text: str = Field(min_length=1, max_length=10000)


class CaptionBundle(WorkflowModel):
    enabled: bool
    track: CaptionTrack | None = None
    srt: MediaReference | None = None
    ass: MediaReference | None = None
    scene_words: dict[str, list[WordTiming]] = Field(default_factory=dict)


class ImageRequestSet(WorkflowModel):
    requests: list[ImageRequest | TimedImageRequest]
    character_ids_by_scene: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_visual_identities(self) -> "ImageRequestSet":
        visual_ids = [
            request.shot_id
            if isinstance(request, TimedImageRequest)
            else request.scene_id
            for request in self.requests
        ]
        if not visual_ids or len(visual_ids) != len(set(visual_ids)):
            raise ValueError("Image Request visual IDs must be nonempty and unique")
        unknown_character_keys = sorted(set(self.character_ids_by_scene) - set(visual_ids))
        if unknown_character_keys:
            raise ValueError(
                "Image Request character mapping references unknown visual IDs: "
                + ", ".join(unknown_character_keys)
            )
        return self


class ImageItem(WorkflowModel):
    generated: ImageAsset
    normalized_image: MediaReference
    request: ImageRequest | TimedImageRequest | None = None


class ImageSet(WorkflowModel):
    items: list[ImageItem]

    @model_validator(mode="after")
    def validate_visual_identities(self) -> "ImageSet":
        visual_ids = [item.generated.shot_id or item.generated.scene_id for item in self.items]
        if not visual_ids or len(visual_ids) != len(set(visual_ids)):
            raise ValueError("Generated-image visual IDs must be nonempty and unique")
        mismatches = []
        for item in self.items:
            if item.request is None:
                continue
            request_id = (
                item.request.shot_id
                if isinstance(item.request, TimedImageRequest)
                else item.request.scene_id
            )
            generated_id = item.generated.shot_id or item.generated.scene_id
            if request_id != generated_id or item.request.scene_id != item.generated.scene_id:
                mismatches.append(generated_id)
        if mismatches:
            raise ValueError(
                "Generated images do not match their requests: " + ", ".join(mismatches)
            )
        return self


class VisualReviewBundle(WorkflowModel):
    reviewed: bool
    report: VisualReviewReport | None = None
    images: ImageSet


class MusicBriefBundle(WorkflowModel):
    enabled: bool
    brief: MusicBrief | None = None


class ResearchQueryBundle(WorkflowModel):
    query: str
    sources: list[ResearchSource]


class MusicBundle(WorkflowModel):
    enabled: bool
    generated: MusicAsset | None = None
    fitted_audio: MediaReference | None = None
    warning: str = ""


class RenderBundle(WorkflowModel):
    plan: RenderPlan
    primary_video: MediaReference
    burned_video: MediaReference | None = None


def _usage_list(values: Iterable[UsageRecord | None]) -> list[UsageRecord]:
    return [value for value in values if value is not None]


class WorkflowEngine:
    def __init__(
        self,
        *,
        store: RunStore,
        environment: dict[str, str],
        stop_after: str | None = None,
    ) -> None:
        if stop_after is not None and stop_after not in PUBLIC_STAGES:
            raise ValueError(f"unknown stop-after stage: {stop_after}")
        self.store = store
        self.config = store.config
        self.brief = store.brief
        verify_runtime_snapshot(self.config, store.frozen_assets)
        self.project_root = Path(self.config.project_root).resolve()
        self.stop_after = stop_after
        self.prompts = PromptLibrary(store.frozen_assets)
        self.workflow_policy_version = self.prompts.workflow_policy_version
        self.continuity_policy_enabled = self.workflow_policy_version >= 2
        self.internal_revision = (
            MULTI_FORMAT_INTERNAL_REVISION
            if self.workflow_policy_version >= 3
            else INTERNAL_REVISION
            if self.continuity_policy_enabled
            else LEGACY_INTERNAL_REVISION
        )
        self.tools = MediaTools.discover()
        raw_descriptors = (
            store.frozen_assets.get("profile", {}).get("backend_descriptors", {})
            if isinstance(store.frozen_assets.get("profile"), dict)
            else {}
        )
        frozen_descriptors = {
            backend_id: BackendDescriptor.model_validate(value)
            for backend_id, value in raw_descriptors.items()
            if isinstance(value, dict)
        }
        self.registry = BackendRegistry(
            config=self.config,
            environment=environment,
            run_root=store.root,
            descriptors=frozen_descriptors or None,
        )
        self.executor = TaskExecutor(registry=self.registry, store=store, prompts=self.prompts)

    def close(self) -> None:
        self.registry.close()

    def __enter__(self) -> "WorkflowEngine":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def run(self) -> DeliveryManifest | None:
        self.store.set_status("running")
        try:
            research = self._research()
            if self._stop("research"):
                return None
            explainer_format = self.config.content_format is not ContentFormat.NARRATIVE
            candidate_model: type[CandidateSetLike] = (
                ExplainerCandidateSet if explainer_format else CandidateSet
            )
            candidates = self._structured_stage(
                "ideate",
                "ideate",
                {
                    "brief": self.brief.model_dump(mode="json"),
                    "research_pack": research.model_dump(mode="json"),
                    "candidate_count": self.config.idea_candidates,
                    "duration_seconds": self.config.duration_seconds,
                    "content_mode": self.config.content_mode.value,
                    "content_format": self.config.content_format.value,
                },
                candidate_model,
                invariant=lambda value: self._validate_candidate_set(value, research),
            )
            if self._stop("ideate"):
                return None
            selection_model: type[SelectionReportLike] = (
                ExplainerSelectionReport if explainer_format else SelectionReport
            )
            selection_input = {
                "candidate_set": candidates.model_dump(mode="json"),
                "duration_seconds": self.config.duration_seconds,
                "content_mode": self.config.content_mode.value,
                "content_format": self.config.content_format.value,
            }
            if self.workflow_policy_version >= 3:
                selection_input.update(
                    {
                        "brief": self.brief.model_dump(mode="json"),
                        "research_pack": research.model_dump(mode="json"),
                        "audience": self.config.audience,
                    }
                )
            selection = self._structured_stage(
                "select",
                "select",
                selection_input,
                selection_model,
                invariant=lambda value: self._validate_selection(value, candidates),
            )
            if self._stop("select"):
                return None
            chosen = next(
                candidate for candidate in candidates.candidates if candidate.candidate_id == selection.chosen_candidate_id
            )
            target_scene_count, minimum_scene_count, maximum_scene_count = (
                self._outline_scene_count_bounds()
            )
            outline_model: type[OutlineLike] = ExplainerOutline if explainer_format else StoryOutline
            outline_input = {
                "brief": self.brief.model_dump(mode="json"),
                "story_concept": chosen.model_dump(mode="json"),
                "selection": selection.model_dump(mode="json"),
                "duration_seconds": self.config.duration_seconds,
                "visual_target_seconds": self.config.visual_target_seconds,
                "visual_min_seconds": self.config.visual_min_seconds,
                "visual_max_seconds": self.config.visual_max_seconds,
                "target_scene_count": target_scene_count,
                "minimum_scene_count": minimum_scene_count,
                "maximum_scene_count": maximum_scene_count,
                "content_mode": self.config.content_mode.value,
                "content_format": self.config.content_format.value,
            }
            if self.workflow_policy_version >= 3:
                outline_input["research_pack"] = research.model_dump(mode="json")
            outline = self._structured_stage(
                "outline",
                "outline",
                outline_input,
                outline_model,
                invariant=lambda value: self._validate_outline(value, research=research),
            )
            if self._stop("outline"):
                return None
            script_word_plan = self._script_word_plan(outline)
            draft_input = {
                "brief": self.brief.model_dump(mode="json"),
                "outline": outline.model_dump(mode="json"),
                "output_language": self.config.output_language.value,
                "duration_seconds": self.config.duration_seconds,
                "estimated_words_per_second": self._target_words_per_second(),
                "narration_delivery": self._delivery_payload(),
                "content_mode": self.config.content_mode.value,
                "content_format": self.config.content_format.value,
                **script_word_plan,
            }
            if self.config.content_mode is ContentMode.FACTUAL:
                draft_input["research_pack"] = research.model_dump(mode="json")
            draft = self._structured_stage(
                "script-draft",
                "script_draft",
                draft_input,
                NarrationScript,
                invariant=lambda value: self._validate_draft(
                    value,
                    outline,
                    minimum_words=int(script_word_plan["minimum_total_word_count"]),
                    maximum_words=int(script_word_plan["maximum_total_word_count"]),
                    enforce_pause_limit=self.continuity_policy_enabled,
                    maximum_pause_seconds=self._maximum_authored_pause_seconds(),
                ),
            )
            if self._stop("script-draft"):
                return None
            story_review = self._review_stage(
                "review-story", "review_story", "story", draft, outline, research=research
            )
            if self._stop("review-story"):
                return None
            spoken_review = self._review_stage(
                "review-spoken", "review_spoken", "spoken", draft, outline, research=research
            )
            if self._stop("review-spoken"):
                return None
            constraint_review = self._review_stage(
                "review-constraints",
                "review_constraints",
                "constraints",
                draft,
                outline,
                research=research,
            )
            if self._stop("review-constraints"):
                return None
            reviews = [story_review, spoken_review, constraint_review]
            revision_input = {
                "brief": self.brief.model_dump(mode="json"),
                "outline": outline.model_dump(mode="json"),
                "script": draft.model_dump(mode="json"),
                "review_reports": [review.model_dump(mode="json") for review in reviews],
                "required_finding_ids": sorted(
                    finding.finding_id for review in reviews for finding in review.findings
                ),
                "duration_seconds": self.config.duration_seconds,
                "output_language": self.config.output_language.value,
                "narration_delivery": self._delivery_payload(),
                "content_mode": self.config.content_mode.value,
                "content_format": self.config.content_format.value,
                **script_word_plan,
            }
            if self.config.content_mode is ContentMode.FACTUAL:
                revision_input["research_pack"] = research.model_dump(mode="json")
            def revision_invariant(value: RevisedScript) -> None:
                self._validate_revision(
                    value,
                    reviews,
                    [scene.scene_id for scene in outline.scenes],
                    minimum_words=int(script_word_plan["minimum_total_word_count"]),
                    maximum_words=int(script_word_plan["maximum_total_word_count"]),
                    enforce_pause_limit=self.continuity_policy_enabled,
                    maximum_pause_seconds=self._maximum_authored_pause_seconds(),
                )
            if self.config.content_mode is ContentMode.FACTUAL:
                if not isinstance(research, FactualResearchPack):
                    raise BackendError(
                        "factual Run did not produce an evidence-backed Research Pack",
                        kind=ErrorKind.INVALID_OUTPUT,
                    )
                revised: RevisedScript | FactualRevisedScript = self._factual_revision_stage(
                    revision_input,
                    research,
                    invariant=revision_invariant,
                )
            else:
                revised = self._structured_stage(
                    "script-revision",
                    "script_revision",
                    revision_input,
                    RevisedScript,
                    invariant=revision_invariant,
                )
            final_script = revised.script
            if self._stop("script-revision"):
                return None
            narration = self._narration(
                final_script,
                factual_research=research if isinstance(research, FactualResearchPack) else None,
            )
            if self._stop("narration"):
                return None
            captions = self._captions(narration)
            if self._stop("captions"):
                return None
            visual_plan_input = {
                "script": narration.script.model_dump(mode="json"),
                "timeline": narration.timeline.model_dump(mode="json"),
                "style_id": self.config.style,
                "style_description": self.config.style_description,
                "audience": self.config.audience,
                "content_mode": self.config.content_mode.value,
                "content_format": self.config.content_format.value,
                "delivery": {
                    "width": self.config.delivery_width,
                    "height": self.config.delivery_height,
                    "aspect_ratio": "16:9",
                },
            }
            if self.continuity_policy_enabled:
                visual_plan_input = {
                    "brief": self.brief.model_dump(mode="json"),
                    "outline": outline.model_dump(mode="json"),
                    **visual_plan_input,
                }
            timed_visuals = self.config.visual_shot_mode is VisualShotMode.CADENCED
            canonical_shot_schedule = (
                self._build_shot_schedule(narration, captions) if timed_visuals else []
            )
            if timed_visuals:
                visual_plan_input["canonical_shot_schedule"] = canonical_shot_schedule
                visual_plan_input["shot_cadence"] = {
                    "target_seconds": self.config.shot_target_seconds,
                    "minimum_seconds": self.config.shot_min_seconds,
                    "maximum_seconds": self.config.shot_max_seconds,
                    "frame_rate": self.config.fps,
                }
            visual_plan_model: type[VisualPlanLike] = (
                TimedVisualPlan if timed_visuals else VisualPlan
            )
            if timed_visuals:
                def visual_plan_invariant(value: TimedVisualPlan) -> None:
                    self._validate_timed_visual_plan(
                        value,
                        schedule=canonical_shot_schedule,
                        script=narration.script,
                    )
            else:
                def visual_plan_invariant(value: VisualPlan) -> None:
                    self._validate_visual_plan(
                        value,
                        outline=outline,
                        script=narration.script,
                        require_continuity=self.continuity_policy_enabled,
                        require_character_continuity=(
                            self.config.content_format is ContentFormat.NARRATIVE
                        ),
                    )
            visual_plan = self._structured_stage(
                "visual-plan",
                "visual_plan",
                visual_plan_input,
                visual_plan_model,
                invariant=visual_plan_invariant,
                max_output_tokens=16000 if timed_visuals else 8000,
            )
            if self._stop("visual-plan"):
                return None
            image_requests = self._image_prompts(visual_plan)
            if self._stop("image-prompt-compile"):
                return None
            prepared_music_brief = None
            if self._should_prepare_music_brief():
                prepared_music_brief = self._prepare_music_brief(narration)
            images = self._images(image_requests)
            if self._stop("images"):
                return None
            reviewed = self._visual_review(visual_plan, image_requests, images)
            if self._stop("visual-review"):
                return None
            music_brief = self._music_brief(narration, prepared=prepared_music_brief)
            if self._stop("music-brief"):
                return None
            music = self._music(music_brief, narration.timeline)
            if self._stop("music"):
                return None
            self.registry.release_local_workers()
            rendered = self._render(
                narration.timeline,
                captions,
                reviewed.images,
                music,
                visual_plan,
            )
            if self._stop("render"):
                return None
            delivery = self._delivery(rendered, captions)
            self.store.set_status("complete")
            return delivery
        except VideoGeneratorError as exc:
            active = next(
                (
                    stage
                    for stage in reversed(PUBLIC_STAGES)
                    if self.store.manifest.stages.get(stage)
                    and self.store.manifest.stages[stage].status == "running"
                ),
                None,
            )
            if active:
                self.store.fail_stage(active, exc)
            else:
                self.store.set_status("failed", exc)
            raise
        except Exception as exc:
            wrapped = VideoGeneratorError(str(exc), kind=ErrorKind.INTERNAL)
            active = next(
                (
                    stage
                    for stage in reversed(PUBLIC_STAGES)
                    if self.store.manifest.stages.get(stage)
                    and self.store.manifest.stages[stage].status == "running"
                ),
                None,
            )
            if active:
                self.store.fail_stage(active, wrapped)
            else:
                self.store.set_status("failed", wrapped)
            raise wrapped from exc

    def _stop(self, stage: str) -> bool:
        if self.stop_after == stage:
            self.store.stop_after(stage)
            return True
        return False

    def _should_prepare_music_brief(self) -> bool:
        return (
            self.config.music_enabled
            and self.config.task_bindings["music_brief"]
            == self.config.task_bindings["image_prompt_compile"]
            and (
                self.stop_after is None
                or PUBLIC_STAGES.index(self.stop_after) >= PUBLIC_STAGES.index("music-brief")
            )
        )

    def _stage_metadata(
        self,
        *,
        stage: str,
        task_id: str | None,
        input_data: Any,
        target_image_backend: str | None = None,
    ) -> dict[str, str]:
        if task_id:
            backend_id = self.config.task_bindings[task_id]
            descriptor = self.registry.descriptor(backend_id)
            output_language = self.prompts.output_language(
                task_id, self.config.output_language
            )
            prompt = self.prompts.get(
                task_id,
                language=self.config.output_language,
                target_image_backend=target_image_backend,
            )
            schema = self.prompts.schema(task_id)
            return {
                "input_hash": hash_run_input(input_data),
                "config_hash": hash_value(
                    {
                        "task_id": task_id,
                        "backend_id": backend_id,
                        "language": output_language.value,
                    }
                ),
                "backend_id": backend_id,
                "backend_revision": descriptor.revision,
                "prompt_version": prompt.version,
                "schema_hash": hash_value(schema),
            }
        return {
            "input_hash": hash_run_input(input_data),
            "config_hash": hash_value(self._internal_stage_config(stage)),
            "backend_id": "internal:media",
            "backend_revision": self.internal_revision,
            "prompt_version": "",
            "schema_hash": "",
        }

    def _internal_stage_config(self, stage: str) -> dict[str, Any]:
        direct: dict[str, Any] = {"stage": stage, "revision": self.internal_revision}
        if stage == "narration":
            direct.update(
                duration=self.config.duration_seconds,
                fps=self.config.fps,
                voice=self.config.voice.model_dump(mode="json"),
                speech_backend=self.config.task_bindings["narration_synthesis"],
                repair_backend=self.config.task_bindings["duration_repair"],
            )
        elif stage == "captions":
            direct.update(
                enabled=self.config.captions_enabled,
                animated=self.config.animated_captions,
                visual_shot_mode=self.config.visual_shot_mode.value,
                alignment_backend=self.config.task_bindings["caption_alignment"],
                width=self.config.delivery_width,
                height=self.config.delivery_height,
            )
        elif stage in {"images", "visual-review"}:
            direct.update(
                quality=self.config.quality.value,
                width=self.config.delivery_width,
                height=self.config.delivery_height,
                image_backend=self.config.task_bindings["image_generate"],
                review_backend=self.config.task_bindings["visual_review"],
            )
        elif stage in {"music-brief", "music"}:
            direct.update(
                enabled=self.config.music_enabled,
                failure_policy=self.config.failure_policy.value,
                music_backend=self.config.task_bindings["music_generate"],
            )
        elif stage in {"render", "delivery"}:
            direct.update(
                width=self.config.delivery_width,
                height=self.config.delivery_height,
                fps=self.config.fps,
                budget=self.config.duration_seconds,
                animated=self.config.animated_captions,
            )
        return direct

    def _structured_stage(
        self,
        stage: str,
        task_id: str,
        input_data: dict[str, Any],
        output_model: type[BaseModel],
        *,
        media_inputs: list[Path] | None = None,
        target_image_backend: str | None = None,
        invariant: Callable[[Any], None] | None = None,
        max_output_tokens: int = 8000,
    ) -> Any:
        metadata = self._stage_metadata(
            stage=stage,
            task_id=task_id,
            input_data=input_data,
            target_image_backend=target_image_backend,
        )
        reusable = self.store.reusable_record(stage, **metadata)
        if reusable:
            artifact = self.store.load_artifact(reusable, output_model)
            if invariant:
                invariant(artifact)
            return artifact
        workspace = self.store.workspace(stage)
        self.store.begin_stage(stage, attempt=workspace.attempt, **metadata)
        execution = self.executor.structured(
            task_id,
            input_data,
            output_model,
            media_inputs=media_inputs,
            target_image_backend=target_image_backend,
            max_output_tokens=max_output_tokens,
            invariant=invariant,
        )
        atomic_write_json(workspace.work_dir / "provider-response.json", execution.result.raw_response)
        promoted = self.store.promote_stage(
            workspace,
            execution.artifact,
            usage=_usage_list([execution.result.usage]),
        )
        return output_model.model_validate(promoted)

    def _review_stage(
        self,
        stage: str,
        task_id: str,
        expected_type: str,
        script: NarrationScript,
        outline: OutlineLike,
        *,
        research: ResearchPack | FactualResearchPack,
    ) -> ReviewReport:
        input_data = {
            "brief": self.brief.model_dump(mode="json"),
            "outline": outline.model_dump(mode="json"),
            "script": script.model_dump(mode="json"),
            "duration_seconds": self.config.duration_seconds,
            "audience": self.config.audience,
            "output_language": self.config.output_language.value,
        }
        if self.config.content_mode is ContentMode.FACTUAL:
            input_data["research_pack"] = research.model_dump(mode="json")
        report = self._structured_stage(
            stage,
            task_id,
            input_data,
            ReviewReport,
            invariant=lambda value: self._validate_review(value, expected_type, task_id),
        )
        return report

    @staticmethod
    def _validate_review(report: ReviewReport, expected_type: str, task_id: str) -> None:
        WorkflowEngine._require(
            report.review_type == expected_type,
            (
                f"{task_id} must set review_type to {expected_type!r}; "
                f"got {report.review_type!r}"
            ),
        )
        prefix = f"{expected_type}:"
        for finding in report.findings:
            if not finding.finding_id.startswith(prefix):
                finding.finding_id = prefix + finding.finding_id

    @staticmethod
    def _require(condition: bool, message: str) -> None:
        if not condition:
            raise BackendError(message, kind=ErrorKind.INVALID_OUTPUT)

    @staticmethod
    def _validate_selection(
        selection: SelectionReportLike,
        candidates: CandidateSetLike,
    ) -> None:
        candidate_ids = {candidate.candidate_id for candidate in candidates.candidates}
        if selection.chosen_candidate_id not in candidate_ids:
            raise BackendError("selector chose an unknown candidate ID", kind=ErrorKind.INVALID_OUTPUT)
        if {score.candidate_id for score in selection.scores} != candidate_ids:
            raise BackendError(
                "selector did not score every candidate exactly once",
                kind=ErrorKind.INVALID_OUTPUT,
            )

    def _validate_candidates(self, candidates: CandidateSet, research: ResearchPack) -> None:
        self._require(
            len(candidates.candidates) == self.config.idea_candidates,
            "ideation did not return the configured candidate count",
        )
        finding_ids = {finding.finding_id for finding in research.findings}
        source_to_findings: dict[str, list[str]] = {}
        for finding in research.findings:
            for source_id in finding.source_ids:
                source_to_findings.setdefault(source_id, []).append(finding.finding_id)
        for candidate in candidates.candidates:
            normalized_ids = []
            for reference_id in candidate.research_inspiration_ids:
                replacements = source_to_findings.get(reference_id, [])
                for normalized_id in replacements or [reference_id]:
                    if normalized_id not in normalized_ids:
                        normalized_ids.append(normalized_id)
            candidate.research_inspiration_ids = normalized_ids
        unknown = sorted(
            {
                finding_id
                for candidate in candidates.candidates
                for finding_id in candidate.research_inspiration_ids
                if finding_id not in finding_ids
            }
        )
        self._require(
            not unknown,
            "Story Candidates reference unknown Research Finding IDs: " + ", ".join(unknown),
        )

    def _validate_candidate_set(
        self,
        candidates: CandidateSetLike,
        research: ResearchPack | FactualResearchPack,
    ) -> None:
        if isinstance(candidates, CandidateSet):
            self._validate_candidates(candidates, research)
            return
        self._require(
            len(candidates.candidates) == self.config.idea_candidates,
            "ideation did not return the configured candidate count",
        )
        known_ids = (
            {item.evidence_id for item in research.evidence}
            if isinstance(research, FactualResearchPack)
            else {item.finding_id for item in research.findings}
        )
        unknown = sorted(
            {
                evidence_id
                for candidate in candidates.candidates
                for evidence_id in candidate.evidence_ids
                if evidence_id not in known_ids
            }
        )
        self._require(
            not unknown,
            "Explainer Candidates reference unknown Evidence IDs: " + ", ".join(unknown),
        )

    def _outline_scene_count_bounds(self) -> tuple[int, int, int]:
        target_count = max(
            1,
            math.ceil(self.config.duration_seconds / self.config.visual_target_seconds),
        )
        return target_count, max(1, target_count - 1), target_count + 1

    def _target_words_per_second(self) -> float:
        delivery = getattr(self.config, "narration_delivery_spec", None)
        if delivery is not None:
            return float(delivery.target_words_per_second)
        return 2.55 if self.config.output_language.value == "en" else 1.95

    def _maximum_authored_pause_seconds(self) -> float:
        delivery = getattr(self.config, "narration_delivery_spec", None)
        if getattr(self, "workflow_policy_version", 2) >= 3 and delivery is not None:
            return float(delivery.maximum_pause_seconds)
        return MAX_AUTHORED_SCENE_PAUSE_SECONDS

    def _delivery_payload(self) -> dict[str, Any]:
        delivery = getattr(self.config, "narration_delivery_spec", None)
        return delivery.model_dump(mode="json") if delivery is not None else {}

    def _script_word_plan(self, outline: OutlineLike) -> dict[str, Any]:
        words_per_second = self._target_words_per_second()
        script_backend_id = getattr(self.config, "task_bindings", {}).get("script_draft", "")
        local_script_backend = script_backend_id.startswith("local:")
        if getattr(self, "workflow_policy_version", 2) >= 3:
            minimum_duration_fraction = 0.85
            maximum_word_tolerance = 1.0
        else:
            minimum_duration_fraction = (
                0.85
                if self.config.output_language.value == "en" and not local_script_backend
                else 0.65
                if self.config.output_language.value == "en"
                else 0.0
            )
            maximum_word_tolerance = 1.0 if self.config.output_language.value == "en" else 1.05
        use_short_english_envelope = (
            self.config.output_language.value == "en"
            and not local_script_backend
        )
        target_total = max(
            len(outline.scenes) * 8,
            round(self.config.duration_seconds * 0.95 * words_per_second),
        )
        scene_targets: list[dict[str, int | str]] = []
        allocated = 0
        for index, scene in enumerate(outline.scenes):
            if index == len(outline.scenes) - 1:
                target = target_total - allocated
            else:
                target = max(
                    4,
                    round(target_total * scene.provisional_seconds / self.config.duration_seconds),
                )
                allocated += target
            scene_targets.append(
                {
                    "scene_id": scene.scene_id,
                    "target_word_count": target,
                    "minimum_sentence_count": (
                        2
                        if self.config.duration_seconds >= 90
                        and use_short_english_envelope
                        else 3
                        if self.config.duration_seconds >= 90
                        else 1
                    ),
                    "maximum_sentence_count": (
                        3
                        if self.config.duration_seconds >= 90
                        and use_short_english_envelope
                        else 4
                        if self.config.duration_seconds >= 90
                        else 2
                    ),
                }
            )
        return {
            "target_total_word_count": target_total,
            "minimum_total_word_count": max(
                len(outline.scenes) * 8,
                round(
                    self.config.duration_seconds
                    * minimum_duration_fraction
                    * words_per_second
                ),
            ),
            "maximum_total_word_count": max(
                len(outline.scenes) * 8,
                round(
                    self.config.duration_seconds
                    * words_per_second
                    * maximum_word_tolerance
                ),
            ),
            "scene_word_targets": scene_targets,
        }

    @classmethod
    def _validate_draft(
        cls,
        draft: NarrationScript,
        outline: OutlineLike,
        *,
        minimum_words: int,
        maximum_words: int,
        enforce_pause_limit: bool = True,
        maximum_pause_seconds: float = MAX_AUTHORED_SCENE_PAUSE_SECONDS,
    ) -> None:
        cls._require(
            [scene.scene_id for scene in draft.scenes]
            == [scene.scene_id for scene in outline.scenes],
            "draft changed the outline Scene IDs",
        )
        cls._validate_script_word_range(
            draft,
            minimum_words=minimum_words,
            maximum_words=maximum_words,
        )
        if enforce_pause_limit:
            cls._validate_authored_pauses(draft, maximum_pause_seconds=maximum_pause_seconds)

    @staticmethod
    def _validate_authored_pauses(
        script: NarrationScript,
        *,
        maximum_pause_seconds: float = MAX_AUTHORED_SCENE_PAUSE_SECONDS,
    ) -> None:
        excessive = [
            scene.scene_id
            for scene in script.scenes[:-1]
            if scene.pause_after_seconds > maximum_pause_seconds
        ]
        if excessive:
            raise BackendError(
                (
                    f"Narration Script pauses exceed the {maximum_pause_seconds:.2f}-second "
                    "production maximum: "
                    + ", ".join(excessive)
                ),
                kind=ErrorKind.INVALID_OUTPUT,
            )

    @staticmethod
    def _validate_script_word_range(
        script: NarrationScript,
        *,
        minimum_words: int,
        maximum_words: int,
    ) -> None:
        actual = sum(len(scene.spoken_text.split()) for scene in script.scenes)
        if not minimum_words <= actual <= maximum_words:
            raise BackendError(
                (
                    f"Narration Script has {actual} words; required inclusive range is "
                    f"{minimum_words}-{maximum_words}"
                ),
                kind=ErrorKind.INVALID_OUTPUT,
            )

    def _validate_outline(
        self,
        outline: OutlineLike,
        *,
        research: ResearchPack | FactualResearchPack | None = None,
    ) -> None:
        self._normalize_outline_durations(
            outline,
            self.config.duration_seconds,
            minimum_seconds=self.config.visual_min_seconds,
            maximum_seconds=self.config.visual_max_seconds,
        )
        self._require(
            abs(sum(scene.provisional_seconds for scene in outline.scenes) - self.config.duration_seconds)
            <= 0.05,
            "outline Scene allocations do not equal the Duration Budget",
        )
        _, minimum_count, maximum_count = self._outline_scene_count_bounds()
        self._require(
            minimum_count <= len(outline.scenes) <= maximum_count,
            f"outline must contain {minimum_count}-{maximum_count} Scenes for the configured visual cadence",
        )
        for index, scene in enumerate(outline.scenes):
            edge_scene = index in {0, len(outline.scenes) - 1}
            minimum = self.config.visual_min_seconds / 2 if edge_scene else self.config.visual_min_seconds
            self._require(
                minimum <= scene.provisional_seconds <= self.config.visual_max_seconds,
                f"{scene.scene_id} is outside the configured Scene duration bounds",
            )
        if isinstance(outline, ExplainerOutline) and research is not None:
            known_ids = (
                {item.evidence_id for item in research.evidence}
                if isinstance(research, FactualResearchPack)
                else {item.finding_id for item in research.findings}
            )
            unknown = sorted(
                {
                    evidence_id
                    for scene in outline.scenes
                    for evidence_id in scene.evidence_ids
                    if evidence_id not in known_ids
                }
            )
            self._require(
                not unknown,
                "Explainer Outline references unknown Evidence IDs: " + ", ".join(unknown),
            )

    @staticmethod
    def _validate_visual_plan(
        visual_plan: VisualPlan,
        *,
        outline: OutlineLike,
        script: NarrationScript,
        require_continuity: bool = True,
        require_character_continuity: bool = True,
    ) -> None:
        expected = [scene.scene_id for scene in script.scenes]
        WorkflowEngine._require(
            [scene.scene_id for scene in visual_plan.scenes] == expected,
            "Visual Plan does not cover every Scene in order",
        )
        WorkflowEngine._require(
            [scene.scene_id for scene in outline.scenes] == expected,
            "Visual Plan input Scene IDs do not match the approved Story Outline",
        )
        if not require_continuity:
            return
        if not require_character_continuity and not visual_plan.characters:
            return
        WorkflowEngine._require(
            bool(visual_plan.characters),
            "Visual Plan must define at least one recurring Character Identity for continuity",
        )
        character_scene_counts = {
            character.character_id: sum(
                character.character_id in scene.character_ids for scene in visual_plan.scenes
            )
            for character in visual_plan.characters
        }
        minimum_scene_count = min(2, len(visual_plan.scenes))
        WorkflowEngine._require(
            any(count >= minimum_scene_count for count in character_scene_counts.values()),
            "Visual Plan must map at least one Character Identity across Scenes for continuity",
        )
        incomplete_characters = [
            character.character_id
            for character in visual_plan.characters
            if not character.body_form.strip()
            or not character.proportions
            or not character.face_and_markings
            or not character.identity_constraints
        ]
        WorkflowEngine._require(
            not incomplete_characters,
            "Visual Plan Character Identities need fixed body form, proportions, face/markings, "
            "and identity constraints: "
            + ", ".join(incomplete_characters),
        )
        incomplete_scenes = [
            scene.scene_id
            for scene in visual_plan.scenes
            if not scene.continuity_from_previous
            or not scene.state_after_scene
            or (scene.character_ids and not scene.identity_requirements)
        ]
        WorkflowEngine._require(
            not incomplete_scenes,
            "Visual Briefs need incoming state, resulting state, and character identity locks: "
            + ", ".join(incomplete_scenes),
        )

    def _build_shot_schedule(
        self,
        narration: NarrationBundle,
        captions: CaptionBundle,
    ) -> list[dict[str, Any]]:
        fps = self.config.fps
        target_frames = max(1, round(self.config.shot_target_seconds * fps))
        minimum_frames = max(1, round(self.config.shot_min_seconds * fps))
        maximum_frames = max(minimum_frames, round(self.config.shot_max_seconds * fps))
        total_frames = round(narration.timeline.delivery_duration_seconds * fps)
        script_by_id = {scene.scene_id: scene for scene in narration.script.scenes}
        schedule: list[dict[str, Any]] = []
        scene_start_frame = 0
        for scene_index, timeline_scene in enumerate(narration.timeline.scenes):
            scene_end_frame = (
                total_frames
                if scene_index == len(narration.timeline.scenes) - 1
                else round(narration.timeline.scenes[scene_index + 1].start_seconds * fps)
            )
            frame_count = max(1, scene_end_frame - scene_start_frame)
            minimum_count = max(1, math.ceil(frame_count / maximum_frames))
            maximum_count = frame_count // minimum_frames
            if minimum_count > maximum_count:
                raise BackendError(
                    (
                        f"{timeline_scene.scene_id} cannot be divided into Shots within the "
                        f"configured {self.config.shot_min_seconds:g}-"
                        f"{self.config.shot_max_seconds:g}s bounds"
                    ),
                    kind=ErrorKind.INVALID_OUTPUT,
                    action="Widen the Shot bounds or rerun from outline with longer editorial Scenes.",
                )
            desired_count = max(1, round(frame_count / target_frames))
            shot_count = min(max(desired_count, minimum_count), maximum_count)
            base_frames, remainder = divmod(frame_count, shot_count)
            words = captions.scene_words.get(timeline_scene.scene_id) or timeline_scene.words
            script_words = script_by_id[timeline_scene.scene_id].spoken_text.split()
            cursor = scene_start_frame
            for local_index in range(shot_count):
                duration_frames = base_frames + (1 if local_index < remainder else 0)
                end_frame = cursor + duration_frames
                local_start = cursor / fps - timeline_scene.start_seconds
                local_end = end_frame / fps - timeline_scene.start_seconds
                excerpt_words = [
                    word.text
                    for word in words
                    if local_start <= (word.start_seconds + word.end_seconds) / 2 < local_end
                ]
                if not excerpt_words:
                    word_start = math.floor(local_index * len(script_words) / shot_count)
                    word_end = math.floor((local_index + 1) * len(script_words) / shot_count)
                    excerpt_words = script_words[word_start : max(word_start + 1, word_end)]
                schedule.append(
                    {
                        "shot_id": f"shot-{len(schedule) + 1:03d}",
                        "scene_id": timeline_scene.scene_id,
                        "narration_excerpt": " ".join(excerpt_words).strip(),
                        "start_seconds": round(cursor / fps, 6),
                        "end_seconds": round(end_frame / fps, 6),
                    }
                )
                cursor = end_frame
            scene_start_frame = scene_end_frame
        if len(schedule) > MAX_CADENCED_SHOTS:
            raise BackendError(
                f"cadenced schedule contains {len(schedule)} Shots; the current limit is "
                f"{MAX_CADENCED_SHOTS}",
                kind=ErrorKind.UNSUPPORTED,
                action="Increase shot_target_seconds, shorten the Run, or use scene_locked mode.",
            )
        return schedule

    @staticmethod
    def _validate_timed_visual_plan(
        visual_plan: TimedVisualPlan,
        *,
        schedule: Sequence[dict[str, Any]],
        script: NarrationScript,
    ) -> None:
        WorkflowEngine._require(
            bool(schedule),
            "canonical Shot schedule is empty",
        )
        WorkflowEngine._require(
            len(visual_plan.shots) == len(schedule),
            "Timed Visual Plan does not cover every canonical Shot",
        )
        known_scene_ids = {scene.scene_id for scene in script.scenes}
        for shot, expected in zip(visual_plan.shots, schedule, strict=True):
            for field_name in (
                "shot_id",
                "scene_id",
                "narration_excerpt",
                "start_seconds",
                "end_seconds",
            ):
                actual = getattr(shot, field_name)
                target = expected[field_name]
                if isinstance(target, float):
                    matched = abs(float(actual) - target) <= 0.0005
                else:
                    matched = actual == target
                WorkflowEngine._require(
                    matched,
                    f"Timed Visual Plan changed canonical {field_name} for {expected['shot_id']}",
                )
            WorkflowEngine._require(
                shot.scene_id in known_scene_ids,
                f"{shot.shot_id} references an unknown parent Scene",
            )
        WorkflowEngine._require(
            abs(visual_plan.duration_seconds - float(schedule[-1]["end_seconds"])) <= 0.0005,
            "Timed Visual Plan changed the canonical delivery duration",
        )
        incomplete_characters = [
            character.character_id
            for character in visual_plan.characters
            if not character.body_form.strip()
            or not character.proportions
            or not character.face_and_markings
            or not character.identity_constraints
        ]
        WorkflowEngine._require(
            not incomplete_characters,
            "Timed Visual Plan Character Identities are incomplete: "
            + ", ".join(incomplete_characters),
        )

    @staticmethod
    def _normalize_outline_durations(
        outline: OutlineLike,
        budget_seconds: float,
        *,
        minimum_seconds: float = 0,
        maximum_seconds: float = math.inf,
    ) -> None:
        if not outline.scenes:
            return

        last_index = len(outline.scenes) - 1
        lower_bounds = [
            minimum_seconds / 2 if index in {0, last_index} else minimum_seconds
            for index in range(len(outline.scenes))
        ]
        upper_bounds = [maximum_seconds] * len(outline.scenes)
        if sum(lower_bounds) > budget_seconds + 0.05 or sum(upper_bounds) < budget_seconds - 0.05:
            raise BackendError(
                "configured Scene duration bounds cannot fit the Duration Budget",
                kind=ErrorKind.INVALID_OUTPUT,
            )

        normalized = list(lower_bounds)
        remaining = budget_seconds - sum(normalized)
        active = set(range(len(outline.scenes)))
        weights = [max(scene.provisional_seconds, 0.001) for scene in outline.scenes]
        while remaining > 1e-9 and active:
            weight_total = sum(weights[index] for index in active)
            proposed = {
                index: remaining * weights[index] / weight_total for index in active
            }
            saturated = [
                index
                for index in active
                if proposed[index] > upper_bounds[index] - normalized[index]
            ]
            if not saturated:
                for index in active:
                    normalized[index] += proposed[index]
                remaining = 0
                break
            for index in saturated:
                capacity = upper_bounds[index] - normalized[index]
                normalized[index] += capacity
                remaining -= capacity
                active.remove(index)

        normalized = [round(value, 3) for value in normalized]
        delta = round(budget_seconds - sum(normalized), 3)
        for index, value in enumerate(normalized):
            if abs(delta) <= 0.0005:
                break
            if delta > 0:
                adjustment = min(delta, upper_bounds[index] - value)
            else:
                adjustment = max(delta, lower_bounds[index] - value)
            normalized[index] = round(value + adjustment, 3)
            delta = round(delta - adjustment, 3)
        for scene, duration in zip(outline.scenes, normalized, strict=True):
            scene.provisional_seconds = duration

    @staticmethod
    def _validate_revision(
        revision: RevisedScript,
        reviews: Sequence[ReviewReport],
        expected_scene_ids: list[str],
        *,
        minimum_words: int | None = None,
        maximum_words: int | None = None,
        enforce_pause_limit: bool = True,
        maximum_pause_seconds: float = MAX_AUTHORED_SCENE_PAUSE_SECONDS,
    ) -> None:
        if [scene.scene_id for scene in revision.script.scenes] != expected_scene_ids:
            raise BackendError("revision changed Scene IDs or order", kind=ErrorKind.INVALID_OUTPUT)
        findings = {
            finding.finding_id: finding
            for review in reviews
            for finding in review.findings
        }
        if len(findings) != sum(len(review.findings) for review in reviews):
            raise BackendError(
                "review Finding IDs must be unique across all review roles",
                kind=ErrorKind.INVALID_OUTPUT,
            )
        dispositions = {item.finding_id: item for item in revision.dispositions}
        if set(dispositions) != set(findings):
            missing = sorted(set(findings) - set(dispositions))
            unexpected = sorted(set(dispositions) - set(findings))
            detail = []
            if missing:
                detail.append("missing: " + ", ".join(missing))
            if unexpected:
                detail.append("unexpected: " + ", ".join(unexpected))
            raise BackendError(
                "revision dispositions must match required Finding IDs (" + "; ".join(detail) + ")",
                kind=ErrorKind.INVALID_OUTPUT,
            )
        unresolved_blocking = [
            finding_id
            for finding_id, finding in findings.items()
            if finding.severity == "blocking"
            and dispositions[finding_id].disposition != "applied"
        ]
        if unresolved_blocking:
            raise BackendError(
                "revision left blocking Findings unresolved: " + ", ".join(sorted(unresolved_blocking)),
                kind=ErrorKind.INVALID_OUTPUT,
            )
        if minimum_words is not None and maximum_words is not None:
            WorkflowEngine._validate_script_word_range(
                revision.script,
                minimum_words=minimum_words,
                maximum_words=maximum_words,
            )
        if enforce_pause_limit:
            WorkflowEngine._validate_authored_pauses(
                revision.script,
                maximum_pause_seconds=maximum_pause_seconds,
            )

    def _structured_item(
        self,
        *,
        stage: str,
        item_id: str,
        task_id: str,
        input_data: dict[str, Any],
        output_model: type[BaseModel],
        invariant: Callable[[Any], None] | None = None,
    ) -> tuple[Any, list[UsageRecord]]:
        metadata = self._stage_metadata(stage=stage, task_id=task_id, input_data=input_data)
        reusable = self.store.reusable_item(stage, item_id, **metadata)
        if reusable:
            artifact = self.store.load_item_artifact(reusable, output_model)
            if invariant:
                invariant(artifact)
            return artifact, reusable.usage
        workspace = self.store.workspace(stage, item_id=item_id)
        execution = self.executor.structured(
            task_id,
            input_data,
            output_model,
            invariant=invariant,
        )
        atomic_write_json(workspace.work_dir / "provider-response.json", execution.result.raw_response)
        usage = _usage_list([execution.result.usage])
        promoted = self.store.promote_item(
            workspace,
            execution.artifact,
            usage=usage,
            **metadata,
        )
        return output_model.model_validate(promoted), usage

    def _factual_audit(
        self,
        *,
        stage: str,
        item_prefix: str,
        script: NarrationScript,
        research: FactualResearchPack,
    ) -> tuple[ClaimInventory, FactualReviewReport, list[UsageRecord]]:
        evidence_ids = {item.evidence_id for item in research.evidence}
        script_by_id = {scene.scene_id: scene for scene in script.scenes}

        def validate_inventory(inventory: ClaimInventory) -> None:
            for claim in inventory.claims:
                scene = script_by_id.get(claim.scene_id)
                self._require(scene is not None, f"Claim {claim.claim_id} references an unknown Scene")
                self._require(
                    claim.exact_text in scene.spoken_text,
                    f"Claim {claim.claim_id} does not preserve exact Script wording",
                )
                unknown = sorted(set(claim.evidence_ids) - evidence_ids)
                self._require(
                    not unknown,
                    f"Claim {claim.claim_id} references unknown Evidence IDs: {', '.join(unknown)}",
                )

        inventory_input = {
            "script": script.model_dump(mode="json"),
            "research_pack": research.model_dump(mode="json"),
            "content_format": self.config.content_format.value,
        }
        inventory, inventory_usage = self._structured_item(
            stage=stage,
            item_id=f"{item_prefix}-claim-inventory",
            task_id="claim_inventory",
            input_data=inventory_input,
            output_model=ClaimInventory,
            invariant=validate_inventory,
        )
        review_input = {
            **inventory_input,
            "claim_inventory": inventory.model_dump(mode="json"),
        }

        def validate_review(review: FactualReviewReport) -> None:
            inventory_ids = {claim.claim_id for claim in inventory.claims}
            reviewed_ids = {claim.claim_id for claim in review.claims}
            self._require(
                reviewed_ids == inventory_ids,
                "Factual Review must cover every inventoried Claim exactly once",
            )
            unknown = sorted(
                {
                    evidence_id
                    for claim in review.claims
                    for evidence_id in claim.evidence_ids
                    if evidence_id not in evidence_ids
                }
            )
            self._require(
                not unknown,
                "Factual Review references unknown Evidence IDs: " + ", ".join(unknown),
            )
            unsupported_citations = [
                claim.claim_id
                for claim in review.claims
                if claim.verdict == "supported" and not claim.evidence_ids
            ]
            self._require(
                not unsupported_citations,
                "Supported Claims require direct Evidence IDs: "
                + ", ".join(unsupported_citations),
            )

        review, review_usage = self._structured_item(
            stage=stage,
            item_id=f"{item_prefix}-factual-review",
            task_id="factual_review",
            input_data=review_input,
            output_model=FactualReviewReport,
            invariant=validate_review,
        )
        if not review.passed:
            failed_claims = [
                claim.claim_id for claim in review.claims if claim.verdict != "supported"
            ]
            detail = ", ".join([*failed_claims, *review.uncovered_claims]) or "review did not pass"
            raise BackendError(
                "factual accuracy gate blocked narration: " + detail,
                kind=ErrorKind.INVALID_OUTPUT,
                action="Revise the Script or improve bounded research evidence before narration.",
            )
        return inventory, review, [*inventory_usage, *review_usage]

    def _factual_revision_stage(
        self,
        revision_input: dict[str, Any],
        research: FactualResearchPack,
        *,
        invariant: Callable[[RevisedScript], None],
    ) -> FactualRevisedScript:
        aggregate_input = {
            "revision": revision_input,
            "research_pack": research.model_dump(mode="json"),
        }
        revision_metadata = self._stage_metadata(
            stage="script-revision",
            task_id="script_revision",
            input_data=aggregate_input,
        )
        revision_metadata["config_hash"] = hash_value(
            {
                "content_mode": self.config.content_mode.value,
                "tasks": {
                    task_id: {
                        "backend_id": self.config.task_bindings[task_id],
                        "backend_revision": self.registry.descriptor(
                            self.config.task_bindings[task_id]
                        ).revision,
                    }
                    for task_id in ("script_revision", "claim_inventory", "factual_review")
                },
            }
        )
        revision_metadata["prompt_version"] = str(
            self.store.frozen_assets.get("prompt_set_version", "")
        )
        revision_metadata["schema_hash"] = hash_value(
            {
                task_id: self.prompts.schema(task_id)
                for task_id in ("script_revision", "claim_inventory", "factual_review")
            }
        )
        reusable = self.store.reusable_record("script-revision", **revision_metadata)
        if reusable:
            artifact = self.store.load_artifact(reusable, FactualRevisedScript)
            invariant(artifact)
            if not artifact.factual_review.passed:
                raise BackendError(
                    "cached factual accuracy gate did not pass",
                    kind=ErrorKind.INVALID_OUTPUT,
                )
            return artifact
        attempt = self.store.next_attempt("script-revision")
        self.store.begin_stage("script-revision", attempt=attempt, **revision_metadata)
        revision, revision_usage = self._structured_item(
            stage="script-revision",
            item_id="revision",
            task_id="script_revision",
            input_data=revision_input,
            output_model=RevisedScript,
            invariant=invariant,
        )
        inventory, review, audit_usage = self._factual_audit(
            stage="script-revision",
            item_prefix="approved-script",
            script=revision.script,
            research=research,
        )
        artifact = FactualRevisedScript(
            script=revision.script,
            dispositions=revision.dispositions,
            claim_inventory=inventory,
            factual_review=review,
        )
        promoted = self.store.complete_fanout_stage(
            "script-revision",
            artifact,
            usage=[*revision_usage, *audit_usage],
        )
        return FactualRevisedScript.model_validate(promoted)

    def _research(self) -> ResearchPack | FactualResearchPack:
        queries = []
        if not self.config.offline and self.config.research_query_limit:
            candidates = self.brief.research_focus or [
                self.brief.central_question
                or self.brief.idea_direction
                or "unusual family-safe story inspiration"
            ]
            for focus in candidates:
                query = focus.strip()
                if query and query not in queries:
                    queries.append(query)
                if len(queries) >= self.config.research_query_limit:
                    break
        input_seed = {
            "brief": self.brief.model_dump(mode="json"),
            "queries": queries,
            "offline": self.config.offline,
            "query_limit": self.config.research_query_limit,
            "source_limit": self.config.research_source_limit,
            "content_mode": self.config.content_mode.value,
            "content_format": self.config.content_format.value,
        }
        metadata = self._stage_metadata(stage="research", task_id="research", input_data=input_seed)
        pack_model: type[ResearchPack | FactualResearchPack] = (
            FactualResearchPack
            if self.config.content_mode is ContentMode.FACTUAL
            else ResearchPack
        )
        reusable = self.store.reusable_record("research", **metadata)
        if reusable:
            return self.store.load_artifact(reusable, pack_model)
        workspace = self.store.workspace("research")
        self.store.begin_stage("research", attempt=workspace.attempt, **metadata)
        sources: list[ResearchSource] = []
        usage: list[UsageRecord] = []
        warnings = []
        if queries:
            search_backend_id = self.config.task_bindings["search"]
            search_descriptor = self.registry.descriptor(search_backend_id)
            for query_index, query in enumerate(queries, start=1):
                if len(sources) >= self.config.research_source_limit:
                    break
                request = SearchRequest(
                    query=query,
                    max_results=min(5, self.config.research_source_limit - len(sources)),
                    language=self.config.output_language,
                )
                item_id = f"query-{query_index:03d}"
                item_input = request.model_dump(mode="json")
                input_hash = hash_run_input(item_input)
                config_hash = hash_value({"backend": search_backend_id})
                reusable_item = self.store.reusable_item(
                    "research",
                    item_id,
                    input_hash=input_hash,
                    config_hash=config_hash,
                    backend_id=search_backend_id,
                    backend_revision=search_descriptor.revision,
                )
                if reusable_item:
                    query_bundle = self.store.load_item_artifact(reusable_item, ResearchQueryBundle)
                    usage.extend(reusable_item.usage)
                    warnings.extend(reusable_item.warnings)
                else:
                    item_workspace = self.store.workspace("research", item_id=item_id)
                    result = self.executor.search(request)
                    item_sources: list[ResearchSource] = []
                    item_warnings: list[str] = []
                    for source_index, source in enumerate(result.sources, start=1):
                        item_sources.append(
                            source.model_copy(update={"source_id": f"result-{source_index:03d}"})
                        )
                    query_bundle = ResearchQueryBundle(query=query, sources=item_sources)
                    item_usage = _usage_list([result.usage])
                    promoted_item = self.store.promote_item(
                        item_workspace,
                        query_bundle,
                        input_hash=input_hash,
                        config_hash=config_hash,
                        backend_id=search_backend_id,
                        backend_revision=search_descriptor.revision,
                        usage=item_usage,
                        warnings=item_warnings,
                    )
                    query_bundle = ResearchQueryBundle.model_validate(promoted_item)
                    usage.extend(item_usage)
                    warnings.extend(item_warnings)
                for source in query_bundle.sources:
                    if len(sources) >= self.config.research_source_limit:
                        break
                    sources.append(
                        source.model_copy(update={"source_id": f"source-{len(sources) + 1:03d}"})
                    )
        if not sources:
            if self.config.content_mode is ContentMode.FACTUAL:
                raise BackendError(
                    "factual research returned no usable bounded sources",
                    kind=ErrorKind.INVALID_OUTPUT,
                    action="Refine research_focus or use a search Backend that returns attributable sources.",
                )
            if queries:
                warnings.append(
                    "Search returned no usable bounded sources; continuing with an empty Research Pack."
                )
            pack = ResearchPack(queries=queries)
            promoted = self.store.promote_stage(workspace, pack, usage=usage, warnings=warnings)
            return ResearchPack.model_validate(promoted)
        task_input = {
            **input_seed,
            "sources": [source.model_dump(mode="json") for source in sources],
        }
        execution = self.executor.structured(
            "research",
            task_input,
            pack_model,
            invariant=lambda value: self._validate_research_source_references(value, sources),
        )
        model_pack = pack_model.model_validate(execution.artifact)
        pack_data = model_pack.model_dump(mode="json")
        pack_data["queries"] = queries
        pack_data["sources"] = [source.model_dump(mode="json") for source in sources]
        pack = pack_model.model_validate(pack_data)
        if len(pack.queries) > self.config.research_query_limit or len(pack.sources) > self.config.research_source_limit:
            raise BackendError("Research Pack exceeded configured query/source limits", kind=ErrorKind.INVALID_OUTPUT)
        atomic_write_json(workspace.work_dir / "provider-response.json", execution.result.raw_response)
        usage.extend(_usage_list([execution.result.usage]))
        promoted = self.store.promote_stage(workspace, pack, usage=usage, warnings=warnings)
        return pack_model.model_validate(promoted)

    @staticmethod
    def _validate_research_source_references(
        pack: ResearchPack,
        sources: Sequence[ResearchSource],
    ) -> None:
        expected_ids = {source.source_id for source in sources}
        unknown = sorted(
            {
                source_id
                for finding in pack.findings
                for source_id in finding.source_ids
                if source_id not in expected_ids
            }
        )
        WorkflowEngine._require(
            not unknown,
            "Research Findings reference sources outside the bounded search results: "
            + ", ".join(unknown),
        )

    def _narration(
        self,
        script: NarrationScript,
        *,
        factual_research: FactualResearchPack | None = None,
    ) -> NarrationBundle:
        input_data = {
            "script": script.model_dump(mode="json"),
            "voice": self.config.voice.model_dump(mode="json"),
            "duration_seconds": self.config.duration_seconds,
            "backend": self.config.task_bindings["narration_synthesis"],
            "delivery": self._delivery_payload(),
        }
        metadata = self._stage_metadata(stage="narration", task_id=None, input_data=input_data)
        metadata["backend_id"] = self.config.task_bindings["narration_synthesis"]
        metadata["backend_revision"] = self.registry.descriptor(metadata["backend_id"]).revision
        reusable = self.store.reusable_record("narration", **metadata)
        if reusable:
            return self.store.load_artifact(reusable, NarrationBundle)
        aggregate_workspace = self.store.workspace("narration")
        self.store.begin_stage("narration", attempt=aggregate_workspace.attempt, **metadata)
        items, usage = self._synthesize_script(script, repair=False)
        bundle = self._assemble_narration(script, items, duration_repaired=False)
        if not duration_is_accepted(bundle.timeline, self.config.duration_seconds):
            pause_fitted_script = self._fit_pauses_to_budget(
                script,
                bundle.timeline,
                self.config.duration_seconds,
                allow_expansion=not self.continuity_policy_enabled,
            )
            if pause_fitted_script is not None:
                pause_fitted_bundle = self._assemble_narration(
                    pause_fitted_script,
                    items,
                    duration_repaired=not self.continuity_policy_enabled,
                )
                if duration_is_accepted(
                    pause_fitted_bundle.timeline,
                    self.config.duration_seconds,
                ):
                    bundle = pause_fitted_bundle
        if not duration_is_accepted(bundle.timeline, self.config.duration_seconds):
            target = self.config.duration_seconds * 0.95
            selected = [scene.scene_id for scene in script.scenes]
            scale = self._duration_repair_scale(
                bundle.timeline,
                target_seconds=target,
                selected_scene_ids=set(selected),
            )
            repaired_bundle, llm_repair_usage = self._duration_repair(
                script=script,
                measured_timeline=bundle.timeline,
                target_seconds=target,
                duration_scale=scale,
                selected_scene_ids=selected,
            )
            repaired = repaired_bundle.script
            if factual_research is not None:
                _, _, factual_usage = self._factual_audit(
                    stage="narration",
                    item_prefix="duration-repaired-script",
                    script=repaired,
                    research=factual_research,
                )
                usage.extend(factual_usage)
            repair_items, tts_repair_usage = self._synthesize_script(
                repaired, repair=True, selected_scene_ids=set(selected), existing_items=items
            )
            usage.extend(llm_repair_usage)
            usage.extend(tts_repair_usage)
            bundle = self._assemble_narration(repaired, repair_items, duration_repaired=True)
            if not duration_is_accepted(bundle.timeline, self.config.duration_seconds):
                pause_fitted_repaired = self._fit_pauses_to_budget(
                    repaired,
                    bundle.timeline,
                    self.config.duration_seconds,
                    allow_expansion=not self.continuity_policy_enabled,
                )
                if pause_fitted_repaired is not None:
                    pause_fitted_bundle = self._assemble_narration(
                        pause_fitted_repaired,
                        repair_items,
                        duration_repaired=True,
                    )
                    if duration_is_accepted(
                        pause_fitted_bundle.timeline,
                        self.config.duration_seconds,
                    ):
                        bundle = pause_fitted_bundle
            if not duration_is_accepted(bundle.timeline, self.config.duration_seconds):
                tempo_items, tempo = self._tempo_fit_narration_items(
                    repair_items,
                    pause_seconds=sum(
                        scene.pause_after_seconds for scene in repaired.scenes
                    ),
                    budget_seconds=self.config.duration_seconds,
                    output_root=aggregate_workspace.work_dir / "tempo-adjusted",
                )
                if tempo_items is not None and tempo is not None:
                    tempo_bundle = self._assemble_narration(
                        repaired,
                        tempo_items,
                        duration_repaired=True,
                        tempo_adjustment=tempo,
                    )
                    tempo_pause_script = self._fit_pauses_to_budget(
                        repaired,
                        tempo_bundle.timeline,
                        self.config.duration_seconds,
                        allow_expansion=not self.continuity_policy_enabled,
                    )
                    if tempo_pause_script is not None:
                        tempo_bundle = self._assemble_narration(
                            tempo_pause_script,
                            tempo_items,
                            duration_repaired=True,
                            tempo_adjustment=tempo,
                        )
                    if duration_is_accepted(
                        tempo_bundle.timeline,
                        self.config.duration_seconds,
                    ):
                        bundle = tempo_bundle
            if not duration_is_accepted(bundle.timeline, self.config.duration_seconds):
                raise MediaError(
                    (
                        f"narration is {bundle.timeline.duration_seconds:.2f}s after the single Duration Repair; "
                        f"required range is {self.config.duration_seconds * 0.85:.2f}-"
                        f"{delivery_ceiling(self.config.duration_seconds, self.config.fps):.2f}s"
                    ),
                    kind=ErrorKind.INVALID_OUTPUT,
                    action="Inspect the repaired script and explicitly rerun from narration with adjusted duration or voice settings.",
                )
        if self.workflow_policy_version >= 3:
            self._validate_narration_delivery(bundle)
        workspace = aggregate_workspace
        narration_path = self.project_root / bundle.timeline.narration_audio.path
        copied = workspace.work_dir / "narration.wav"
        shutil.copy2(narration_path, copied)
        timeline = bundle.timeline.model_copy(
            update={
                "narration_audio": MediaReference(
                    path=relative_path(copied, self.project_root),
                    sha256=sha256_file(copied),
                    mime_type="audio/wav",
                )
            }
        )
        bundle = bundle.model_copy(update={"timeline": timeline})
        promoted = self.store.promote_stage(workspace, bundle, usage=usage)
        return NarrationBundle.model_validate(promoted)

    def _validate_narration_delivery(self, bundle: NarrationBundle) -> None:
        delivery = self.config.narration_delivery_spec
        if delivery is None:
            return
        speech_seconds = sum(
            scene.speech_end_seconds - scene.start_seconds for scene in bundle.timeline.scenes
        )
        word_count = sum(len(scene.spoken_text.split()) for scene in bundle.script.scenes)
        self._require(speech_seconds > 0, "Narration delivery has no measurable speech")
        achieved = word_count / speech_seconds
        self._require(
            delivery.minimum_words_per_second <= achieved <= delivery.maximum_words_per_second,
            (
                f"Narration delivery measured {achieved:.3f} words/second; configured range is "
                f"{delivery.minimum_words_per_second:.3f}-{delivery.maximum_words_per_second:.3f}"
            ),
        )
        self._validate_authored_pauses(
            bundle.script,
            maximum_pause_seconds=float(delivery.maximum_pause_seconds),
        )

    @staticmethod
    def _fit_pauses_to_budget(
        script: NarrationScript,
        timeline: NarrationTimeline,
        budget_seconds: float,
        *,
        allow_expansion: bool = False,
    ) -> NarrationScript | None:
        if allow_expansion:
            return WorkflowEngine._legacy_fit_pauses_to_budget(
                script,
                timeline,
                budget_seconds,
            )
        ceiling = delivery_ceiling(budget_seconds, timeline.fps)
        speech_seconds = sum(
            scene.speech_end_seconds - scene.start_seconds for scene in timeline.scenes
        )
        if speech_seconds > ceiling + 1e-6:
            return None
        # Silence is never added to make short narration satisfy the duration floor. Authored pauses
        # are only reduced when they alone push otherwise valid speech over the delivery ceiling.
        if timeline.delivery_duration_seconds <= ceiling + 1e-6:
            return None
        eligible_indexes = [
            index
            for index in range(max(0, len(script.scenes) - 1))
            if script.scenes[index].pause_after_seconds > 0
        ]
        current_pause_seconds = sum(
            script.scenes[index].pause_after_seconds for index in eligible_indexes
        )
        allowed_pause_seconds = max(0.0, ceiling - speech_seconds - 0.0001)
        if not eligible_indexes or allowed_pause_seconds >= current_pause_seconds - 0.0005:
            return None
        scale = allowed_pause_seconds / current_pause_seconds
        rounded = {
            index: round(script.scenes[index].pause_after_seconds * scale, 4)
            for index in eligible_indexes
        }
        delta = round(allowed_pause_seconds - sum(rounded.values()), 4)
        for index in eligible_indexes:
            if abs(delta) <= 0.00005:
                break
            adjustment = (
                min(delta, script.scenes[index].pause_after_seconds - rounded[index])
                if delta > 0
                else max(delta, -rounded[index])
            )
            rounded[index] = round(rounded[index] + adjustment, 4)
            delta = round(delta - adjustment, 4)

        fitted_scenes = []
        for index, scene in enumerate(script.scenes):
            pause = rounded.get(index, 0.0)
            fitted_scenes.append(scene.model_copy(update={"pause_after_seconds": pause}))
        return NarrationScript.model_validate(
            script.model_copy(update={"scenes": fitted_scenes}).model_dump(mode="json")
        )

    @staticmethod
    def _legacy_fit_pauses_to_budget(
        script: NarrationScript,
        timeline: NarrationTimeline,
        budget_seconds: float,
    ) -> NarrationScript | None:
        ceiling = delivery_ceiling(budget_seconds, timeline.fps)
        minimum = budget_seconds * 0.85
        speech_seconds = sum(
            scene.speech_end_seconds - scene.start_seconds for scene in timeline.scenes
        )
        current_pause_seconds = sum(scene.pause_after_seconds for scene in script.scenes)
        if speech_seconds > ceiling + 1e-6:
            return None

        eligible_indexes = list(range(max(0, len(script.scenes) - 1)))
        maximum_pause_seconds = 3.25 * len(eligible_indexes)
        if timeline.delivery_duration_seconds < minimum:
            if speech_seconds + maximum_pause_seconds < minimum - 1e-6:
                return None
            desired_total = min(
                budget_seconds * 0.95,
                speech_seconds + maximum_pause_seconds,
                ceiling - 0.0001,
            )
            allowed_pause_seconds = max(0.0, desired_total - speech_seconds)
            weights = [
                max(script.scenes[index].pause_after_seconds, 0.15)
                for index in eligible_indexes
            ]
        else:
            if current_pause_seconds <= 0:
                return None
            allowed_pause_seconds = max(0.0, ceiling - speech_seconds - 0.0001)
            eligible_indexes = [
                index
                for index in eligible_indexes
                if script.scenes[index].pause_after_seconds > 0
            ]
            weights = [script.scenes[index].pause_after_seconds for index in eligible_indexes]

        if not eligible_indexes or abs(allowed_pause_seconds - current_pause_seconds) <= 0.0005:
            return None

        allocated = {index: 0.0 for index in eligible_indexes}
        remaining = allowed_pause_seconds
        active = set(eligible_indexes)
        weight_by_index = dict(zip(eligible_indexes, weights, strict=True))
        while remaining > 1e-9 and active:
            weight_total = sum(weight_by_index[index] for index in active)
            proposed = {
                index: remaining * weight_by_index[index] / weight_total for index in active
            }
            saturated = [
                index for index in active if proposed[index] > 3.25 - allocated[index]
            ]
            if not saturated:
                for index in active:
                    allocated[index] += proposed[index]
                remaining = 0
                break
            for index in saturated:
                capacity = 3.25 - allocated[index]
                allocated[index] += capacity
                remaining -= capacity
                active.remove(index)

        rounded = {index: round(value, 4) for index, value in allocated.items()}
        delta = round(allowed_pause_seconds - sum(rounded.values()), 4)
        for index in eligible_indexes:
            if abs(delta) <= 0.00005:
                break
            adjustment = (
                min(delta, 3.25 - rounded[index])
                if delta > 0
                else max(delta, -rounded[index])
            )
            rounded[index] = round(rounded[index] + adjustment, 4)
            delta = round(delta - adjustment, 4)

        fitted_scenes = []
        for index, scene in enumerate(script.scenes):
            pause = rounded.get(index, 0.0)
            fitted_scenes.append(scene.model_copy(update={"pause_after_seconds": pause}))
        return NarrationScript.model_validate(
            script.model_copy(update={"scenes": fitted_scenes}).model_dump(mode="json")
        )

    @staticmethod
    def _tempo_fit_rate(
        *,
        speech_seconds: float,
        pause_seconds: float,
        budget_seconds: float,
    ) -> float | None:
        desired_speech_seconds = budget_seconds * 0.90 - pause_seconds
        if desired_speech_seconds <= speech_seconds:
            return None
        maximum_speech_seconds = speech_seconds / 0.85
        desired_speech_seconds = min(desired_speech_seconds, maximum_speech_seconds)
        if desired_speech_seconds + pause_seconds < budget_seconds * 0.85 - 1e-6:
            return None
        tempo = speech_seconds / desired_speech_seconds
        if not 0.85 <= tempo < 1.0:
            return None
        return tempo

    @staticmethod
    def _legacy_tempo_fit_rate(
        *,
        speech_seconds: float,
        scene_count: int,
        budget_seconds: float,
    ) -> float | None:
        maximum_pause_seconds = 3.25 * max(0, scene_count - 1)
        desired_speech_seconds = budget_seconds * 0.90 - maximum_pause_seconds
        if desired_speech_seconds <= speech_seconds:
            return None
        tempo = speech_seconds / desired_speech_seconds
        if not 0.85 <= tempo < 1.0:
            return None
        return tempo

    def _tempo_fit_narration_items(
        self,
        items: Sequence[NarrationItem],
        *,
        pause_seconds: float,
        budget_seconds: float,
        output_root: Path,
    ) -> tuple[list[NarrationItem] | None, float | None]:
        speech_seconds = sum(item.normalized_duration_seconds for item in items)
        tempo = (
            self._tempo_fit_rate(
                speech_seconds=speech_seconds,
                pause_seconds=pause_seconds,
                budget_seconds=budget_seconds,
            )
            if self.continuity_policy_enabled
            else self._legacy_tempo_fit_rate(
                speech_seconds=speech_seconds,
                scene_count=len(items),
                budget_seconds=budget_seconds,
            )
        )
        if tempo is None:
            return None, None
        adjusted_items = []
        for item in items:
            destination = output_root / f"{item.speech.scene_id}.wav"
            probe = adjust_audio_tempo(
                self.tools,
                self.project_root / item.normalized_audio.path,
                destination,
                tempo=tempo,
            )
            adjusted_items.append(
                item.model_copy(
                    update={
                        "normalized_audio": MediaReference(
                            path=relative_path(destination, self.project_root),
                            sha256=sha256_file(destination),
                            mime_type="audio/wav",
                        ),
                        "normalized_duration_seconds": probe.duration_seconds,
                        "normalized_sample_rate": probe.sample_rate,
                        "normalized_channels": probe.channels,
                    }
                )
            )
        return adjusted_items, tempo

    def _duration_repair(
        self,
        *,
        script: NarrationScript,
        measured_timeline: NarrationTimeline,
        target_seconds: float,
        duration_scale: float,
        selected_scene_ids: list[str],
    ) -> tuple[RevisedScript, list[UsageRecord]]:
        task_id = "duration_repair"
        backend_id = self.config.task_bindings[task_id]
        descriptor = self.registry.descriptor(backend_id)
        prompt = self.prompts.get(task_id, language=self.config.output_language)
        schema_hash = hash_value(
            restricted_json_schema(ExpandedSceneText.model_json_schema())
            if duration_scale > 1
            else self.prompts.schema(task_id)
        )
        selected = set(selected_scene_ids)
        scene_repair_targets = []
        for scene in script.scenes:
            if scene.scene_id not in selected:
                continue
            original_words = len(scene.spoken_text.split())
            target_words = max(1, round(original_words * duration_scale))
            word_tolerance = 6 if duration_scale > 1 else 2
            minimum_words = (
                max(original_words + 1, target_words - word_tolerance)
                if duration_scale > 1
                else max(1, target_words - word_tolerance)
            )
            scene_repair_targets.append(
                {
                    "scene_id": scene.scene_id,
                    "original_word_count": original_words,
                    "target_word_count": target_words,
                    "minimum_word_count": minimum_words,
                    "maximum_word_count": target_words + word_tolerance,
                    "minimum_word_delta": minimum_words - original_words,
                    "target_word_delta": target_words - original_words,
                    "maximum_word_delta": target_words + word_tolerance - original_words,
                }
            )
        item_input = {
            "script": script.model_dump(mode="json"),
            "measured_timeline": measured_timeline.model_dump(
                mode="json", exclude={"narration_audio"}
            ),
            "target_seconds": target_seconds,
            "duration_scale": duration_scale,
            "selected_scene_ids": selected_scene_ids,
            "scene_repair_targets": scene_repair_targets,
            "output_language": self.config.output_language.value,
            "repair_strategy": (
                "per-scene-lengthening-v1" if duration_scale > 1 else "full-script-v1"
            ),
        }
        item_id = "duration-repair-script"
        input_hash = hash_run_input(item_input)
        config_hash = hash_value(
            {
                "backend": backend_id,
                "language": self.config.output_language.value,
                "duration_seconds": self.config.duration_seconds,
            }
        )
        reusable = self.store.reusable_item(
            "narration",
            item_id,
            input_hash=input_hash,
            config_hash=config_hash,
            backend_id=backend_id,
            backend_revision=descriptor.revision,
            prompt_version=prompt.version,
            schema_hash=schema_hash,
        )
        if reusable:
            repaired = self.store.load_item_artifact(reusable, RevisedScript)
            self._validate_duration_revision(
                repaired,
                script,
                selected,
                scene_repair_targets=scene_repair_targets,
            )
            return repaired, reusable.usage
        workspace = self.store.workspace("narration", item_id=item_id)
        if duration_scale > 1:
            repaired, item_usage, provider_response = self._lengthen_duration_by_scene(
                script=script,
                measured_timeline=measured_timeline,
                duration_scale=duration_scale,
                scene_repair_targets=scene_repair_targets,
                selected_scene_ids=selected,
            )
        else:
            execution = self.executor.structured(
                task_id,
                item_input,
                RevisedScript,
                invariant=lambda value: self._validate_duration_revision(
                    value,
                    script,
                    selected,
                    scene_repair_targets=scene_repair_targets,
                ),
            )
            repaired = RevisedScript.model_validate(execution.artifact)
            item_usage = _usage_list([execution.result.usage])
            provider_response = execution.result.raw_response
        atomic_write_json(workspace.work_dir / "provider-response.json", provider_response)
        promoted = self.store.promote_item(
            workspace,
            repaired,
            input_hash=input_hash,
            config_hash=config_hash,
            backend_id=backend_id,
            backend_revision=descriptor.revision,
            prompt_version=prompt.version,
            schema_hash=schema_hash,
            usage=item_usage,
        )
        return RevisedScript.model_validate(promoted), item_usage

    def _lengthen_duration_by_scene(
        self,
        *,
        script: NarrationScript,
        measured_timeline: NarrationTimeline,
        duration_scale: float,
        scene_repair_targets: list[dict[str, int | str]],
        selected_scene_ids: set[str],
    ) -> tuple[RevisedScript, list[UsageRecord], dict[str, Any]]:
        targets = {str(target["scene_id"]): target for target in scene_repair_targets}
        timeline = {scene.scene_id: scene for scene in measured_timeline.scenes}
        repaired_scenes = []
        usage: list[UsageRecord] = []
        responses: dict[str, Any] = {}
        for scene in script.scenes:
            if scene.scene_id not in selected_scene_ids:
                repaired_scenes.append(scene)
                continue
            target = targets[scene.scene_id]
            single_scene = scene.model_copy(update={"scene_id": "scene-001"})
            single_script = NarrationScript(title=script.title, scenes=[single_scene])
            single_target = {**target, "scene_id": "scene-001"}
            measured_scene = timeline[scene.scene_id]
            speech_seconds = measured_scene.speech_end_seconds - measured_scene.start_seconds
            pause_seconds = measured_scene.end_seconds - measured_scene.speech_end_seconds
            single_input = {
                "script": single_script.model_dump(mode="json"),
                "measured_timeline": {
                    "scene_id": "scene-001",
                    "speech_seconds": speech_seconds,
                    "pause_seconds": pause_seconds,
                },
                "target_seconds": speech_seconds * duration_scale + pause_seconds,
                "duration_scale": duration_scale,
                "selected_scene_ids": ["scene-001"],
                "scene_repair_targets": [single_target],
                "output_language": self.config.output_language.value,
                "source_scene_id": scene.scene_id,
                "repair_strategy": "single-scene-lengthening-v2",
            }
            original_word_count = int(single_target["original_word_count"])
            minimum_word_count = int(single_target["minimum_word_count"])
            maximum_word_count = int(single_target["maximum_word_count"])
            execution = self.executor.structured(
                "duration_repair",
                single_input,
                ExpandedSceneText,
                invariant=lambda value, repair_target=single_target: (
                    self._validate_expanded_scene(value, repair_target)
                ),
                instruction_suffix=(
                    "This is exactly one Scene expansion. The original spoken_text has "
                    f"{original_word_count} whitespace-separated words. Return scene_id "
                    f"'scene-001' and a complete spoken_text with {minimum_word_count}-"
                    f"{maximum_word_count} words inclusive. Preserve the useful original "
                    "sentences, then add enough concrete causal action or consequence to reach "
                    "that range. Count the final words before returning; copying the original "
                    "unchanged is invalid."
                ),
            )
            expanded = ExpandedSceneText.model_validate(execution.artifact)
            repaired_scene = scene.model_copy(
                update={"spoken_text": expanded.spoken_text}
            )
            repaired_scenes.append(repaired_scene)
            usage.extend(_usage_list([execution.result.usage]))
            responses[scene.scene_id] = execution.result.raw_response
        revision = RevisedScript(
            script=NarrationScript(title=script.title, scenes=repaired_scenes),
            dispositions=[],
        )
        self._validate_duration_revision(
            revision,
            script,
            selected_scene_ids,
            scene_repair_targets=scene_repair_targets,
        )
        return revision, usage, responses

    @staticmethod
    def _validate_expanded_scene(
        expanded: ExpandedSceneText,
        target: dict[str, int | str],
    ) -> None:
        if expanded.scene_id != str(target["scene_id"]):
            raise BackendError(
                "single-Scene Duration Repair changed the Scene ID",
                kind=ErrorKind.INVALID_OUTPUT,
            )
        actual = len(expanded.spoken_text.split())
        minimum = int(target["minimum_word_count"])
        maximum = int(target["maximum_word_count"])
        if not minimum <= actual <= maximum:
            raise BackendError(
                (
                    f"single-Scene Duration Repair has {actual} words; required inclusive range "
                    f"is {minimum}-{maximum}"
                ),
                kind=ErrorKind.INVALID_OUTPUT,
            )

    @staticmethod
    def _validate_duration_revision(
        revision: RevisedScript,
        original: NarrationScript,
        selected_scene_ids: set[str],
        *,
        scene_repair_targets: list[dict[str, int | str]] | None = None,
    ) -> None:
        repaired = revision.script
        if [scene.scene_id for scene in repaired.scenes] != [scene.scene_id for scene in original.scenes]:
            raise BackendError(
                "Duration Repair changed Scene IDs or order",
                kind=ErrorKind.INVALID_OUTPUT,
            )
        old_by_id = {scene.scene_id: scene for scene in original.scenes}
        for scene in repaired.scenes:
            original_scene = old_by_id[scene.scene_id]
            if (
                scene.scene_id not in selected_scene_ids
                and scene.model_dump(mode="json") != original_scene.model_dump(mode="json")
            ):
                raise BackendError(
                    "Duration Repair changed an unselected Scene",
                    kind=ErrorKind.INVALID_OUTPUT,
                )
            if (
                scene.scene_id in selected_scene_ids
                and scene.pause_after_seconds != original_scene.pause_after_seconds
            ):
                raise BackendError(
                    "Duration Repair changed a Scene pause",
                    kind=ErrorKind.INVALID_OUTPUT,
                )
        if scene_repair_targets is not None:
            repaired_by_id = {scene.scene_id: scene for scene in repaired.scenes}
            violations: list[str] = []
            total_actual = 0
            total_minimum = 0
            total_maximum = 0
            severe_scene_error = False
            for target in scene_repair_targets:
                scene_id = str(target["scene_id"])
                actual = len(repaired_by_id[scene_id].spoken_text.split())
                minimum = int(target["minimum_word_count"])
                desired = int(target["target_word_count"])
                maximum = int(target["maximum_word_count"])
                total_actual += actual
                total_minimum += minimum
                total_maximum += maximum
                if actual < minimum - 1 or actual > maximum + 1:
                    severe_scene_error = True
                if actual < minimum:
                    delta = f"add {minimum - actual}-{maximum - actual} words"
                elif actual > maximum:
                    delta = f"remove {actual - maximum}-{actual - minimum} words"
                else:
                    continue
                violations.append(
                    f"{scene_id} got {actual}, required {minimum}-{maximum} "
                    f"(target {desired}; {delta})"
                )
            total_invalid = not total_minimum - 1 <= total_actual <= total_maximum + 1
            if violations and (total_invalid or severe_scene_error):
                if total_actual < total_minimum:
                    total_delta = (
                        f"add {total_minimum - total_actual}-{total_maximum - total_actual} words"
                    )
                elif total_actual > total_maximum:
                    total_delta = (
                        f"remove {total_actual - total_maximum}-{total_actual - total_minimum} words"
                    )
                else:
                    total_delta = "redistribute words between Scenes"
                details = "; ".join(violations)
                raise BackendError(
                    (
                        f"Duration Repair word counts are invalid: {details}; total got {total_actual}, "
                        f"required {total_minimum}-{total_maximum} ({total_delta})"
                    ),
                    kind=ErrorKind.INVALID_OUTPUT,
                )

    @staticmethod
    def _duration_repair_scale(
        measured_timeline: NarrationTimeline,
        *,
        target_seconds: float,
        selected_scene_ids: set[str],
    ) -> float:
        editable_speech_seconds = 0.0
        fixed_seconds = 0.0
        for scene in measured_timeline.scenes:
            if scene.scene_id in selected_scene_ids:
                editable_speech_seconds += scene.speech_end_seconds - scene.start_seconds
                fixed_seconds += scene.end_seconds - scene.speech_end_seconds
            else:
                fixed_seconds += scene.end_seconds - scene.start_seconds
        target_speech_seconds = target_seconds - fixed_seconds
        if editable_speech_seconds <= 0 or target_speech_seconds <= 0:
            raise MediaError(
                "Duration Repair has no positive editable speech window",
                kind=ErrorKind.INVALID_OUTPUT,
            )
        return target_speech_seconds / editable_speech_seconds

    def _synthesize_script(
        self,
        script: NarrationScript,
        *,
        repair: bool,
        selected_scene_ids: set[str] | None = None,
        existing_items: list[NarrationItem] | None = None,
    ) -> tuple[list[NarrationItem], list[UsageRecord]]:
        selected_scene_ids = selected_scene_ids or {scene.scene_id for scene in script.scenes}
        existing = {item.speech.scene_id: item for item in existing_items or []}
        output = []
        usage = []
        backend_id = self.config.task_bindings["narration_synthesis"]
        descriptor = self.registry.descriptor(backend_id)
        for index, scene in enumerate(script.scenes):
            if scene.scene_id not in selected_scene_ids and scene.scene_id in existing:
                output.append(existing[scene.scene_id])
                continue
            item_id = f"{scene.scene_id}-repair" if repair else scene.scene_id
            item_input = {
                "scene": scene.model_dump(mode="json"),
                "voice": self.config.voice.model_dump(mode="json"),
                "language": self.config.output_language.value,
                "delivery": self._delivery_payload(),
            }
            input_hash = hash_run_input(item_input)
            config_hash = hash_value(
                {
                    "backend": backend_id,
                    "voice": self.config.voice.name,
                    "delivery": self._delivery_payload(),
                }
            )
            reusable = self.store.reusable_item(
                "narration",
                item_id,
                input_hash=input_hash,
                config_hash=config_hash,
                backend_id=backend_id,
                backend_revision=descriptor.revision,
            )
            if reusable:
                item = self.store.load_item_artifact(reusable, NarrationItem)
                output.append(item)
                usage.extend(reusable.usage)
                continue
            workspace = self.store.workspace("narration", item_id=item_id)
            extension = ".mp3" if descriptor.provider == "elevenlabs" else ".wav"
            raw_path = workspace.work_dir / f"speech{extension}"
            normalized_path = workspace.work_dir / "normalized.wav"
            request = SpeechRequest(
                scene_id=scene.scene_id,
                text=scene.spoken_text,
                output_language=self.config.output_language,
                voice=self.config.voice,
                delivery=self.config.narration_delivery_spec,
                output_path=relative_path(raw_path, self.project_root),
                preceding_text=script.scenes[index - 1].spoken_text[-500:] if index else "",
                following_text=script.scenes[index + 1].spoken_text[:500] if index + 1 < len(script.scenes) else "",
            )
            result = self.executor.speech(request)
            if result.asset.scene_id != scene.scene_id:
                raise BackendError("Speech Backend changed the Scene ID", kind=ErrorKind.INVALID_OUTPUT)
            probe = normalize_audio(self.tools, self.project_root / result.asset.audio.path, normalized_path)
            delivery = self.config.narration_delivery_spec
            if (
                self.workflow_policy_version >= 3
                and delivery is not None
                and abs(float(delivery.tempo_multiplier) - 1.0) > 0.0005
            ):
                delivery_path = workspace.work_dir / "delivery.wav"
                probe = adjust_audio_tempo(
                    self.tools,
                    normalized_path,
                    delivery_path,
                    tempo=float(delivery.tempo_multiplier),
                )
                normalized_path = delivery_path
            item = NarrationItem(
                speech=result.asset,
                normalized_audio=MediaReference(
                    path=relative_path(normalized_path, self.project_root),
                    sha256=sha256_file(normalized_path),
                    mime_type="audio/wav",
                ),
                normalized_duration_seconds=probe.duration_seconds,
                normalized_sample_rate=probe.sample_rate,
                normalized_channels=probe.channels,
            )
            item_usage = _usage_list([result.usage])
            promoted = self.store.promote_item(
                workspace,
                item,
                input_hash=input_hash,
                config_hash=config_hash,
                backend_id=backend_id,
                backend_revision=descriptor.revision,
                usage=item_usage,
            )
            output.append(NarrationItem.model_validate(promoted))
            usage.extend(item_usage)
        by_id = {item.speech.scene_id: item for item in output}
        return [by_id[scene.scene_id] for scene in script.scenes], usage

    def _assemble_narration(
        self,
        script: NarrationScript,
        items: Sequence[NarrationItem],
        *,
        duration_repaired: bool,
        tempo_adjustment: float = 1.0,
    ) -> NarrationBundle:
        assembly_root = self.store.root / "work" / "narration" / f"assembly-{os.urandom(4).hex()}"
        assembly_root.mkdir(parents=True, exist_ok=False)
        clips = [self.project_root / item.normalized_audio.path for item in items]
        durations = [item.normalized_duration_seconds for item in items]
        pauses = [scene.pause_after_seconds for scene in script.scenes]
        master = assembly_root / "master.wav"
        master_probe = concatenate_audio(self.tools, clips, durations, pauses, master)
        probes = [
            AudioProbe(
                item.normalized_duration_seconds,
                item.normalized_sample_rate,
                item.normalized_channels,
                "pcm_s16le",
            )
            for item in items
        ]
        timeline = build_timeline(
            script=script,
            source_assets=[item.speech for item in items],
            normalized_paths=clips,
            normalized_probes=probes,
            narration_path=master,
            narration_probe=master_probe,
            workspace_root=self.project_root,
            fps=self.config.fps,
        )
        return NarrationBundle(
            script=script,
            timeline=timeline,
            items=list(items),
            duration_repaired=duration_repaired,
            tempo_adjustment=tempo_adjustment,
        )

    def _captions(self, narration: NarrationBundle) -> CaptionBundle:
        input_data = {
            "script": narration.script.model_dump(mode="json"),
            "timeline": narration.timeline.model_dump(mode="json"),
            "enabled": self.config.captions_enabled,
            "animated": self.config.animated_captions,
            "visual_shot_mode": self.config.visual_shot_mode.value,
        }
        metadata = self._stage_metadata(stage="captions", task_id=None, input_data=input_data)
        timing_required = (
            self.config.captions_enabled
            or self.config.visual_shot_mode is VisualShotMode.CADENCED
        )
        if timing_required and not all(scene.words for scene in narration.timeline.scenes):
            backend_id = self.config.task_bindings["caption_alignment"]
            metadata["backend_id"] = backend_id
            metadata["backend_revision"] = self.registry.descriptor(backend_id).revision
        reusable = self.store.reusable_record("captions", **metadata)
        if reusable:
            return self.store.load_artifact(reusable, CaptionBundle)
        workspace = self.store.workspace("captions")
        self.store.begin_stage("captions", attempt=workspace.attempt, **metadata)
        if not timing_required:
            promoted = self.store.promote_stage(workspace, CaptionBundle(enabled=False))
            return CaptionBundle.model_validate(promoted)
        scene_words: dict[str, Sequence[WordTiming]] = {}
        coverage: dict[str, float] = {}
        usage = []
        script_by_id = {scene.scene_id: scene for scene in narration.script.scenes}
        for timeline_scene in narration.timeline.scenes:
            canonical = re.findall(r"\S+", script_by_id[timeline_scene.scene_id].spoken_text, flags=re.UNICODE)
            if timeline_scene.words and [word.text for word in timeline_scene.words] == canonical:
                scene_words[timeline_scene.scene_id] = timeline_scene.words
                coverage[timeline_scene.scene_id] = 1.0
                continue
            backend_id = self.config.task_bindings["caption_alignment"]
            descriptor = self.registry.descriptor(backend_id)
            item_input = {
                "scene_id": timeline_scene.scene_id,
                "transcript": script_by_id[timeline_scene.scene_id].spoken_text,
                "audio": timeline_scene.audio.model_dump(mode="json"),
            }
            item_hash = hash_run_input(item_input)
            item_config_hash = hash_value({"backend": backend_id, "language": self.config.output_language.value})
            reusable_item = self.store.reusable_item(
                "captions",
                timeline_scene.scene_id,
                input_hash=item_hash,
                config_hash=item_config_hash,
                backend_id=backend_id,
                backend_revision=descriptor.revision,
            )
            if reusable_item:
                aligned = self.store.load_item_artifact(reusable_item, AlignedSceneWords)
                usage.extend(reusable_item.usage)
            else:
                item_workspace = self.store.workspace("captions", item_id=timeline_scene.scene_id)
                result = self.executor.align(
                    AlignmentRequest(
                        scene_id=timeline_scene.scene_id,
                        transcript=script_by_id[timeline_scene.scene_id].spoken_text,
                        audio_path=timeline_scene.audio.path,
                        output_language=self.config.output_language,
                    )
                )
                words, score = reconcile_word_timings(
                    script_by_id[timeline_scene.scene_id].spoken_text,
                    result.recognized_words,
                    scene_duration=timeline_scene.speech_end_seconds - timeline_scene.start_seconds,
                )
                aligned = AlignedSceneWords(scene_id=timeline_scene.scene_id, words=words, coverage=score)
                item_usage = _usage_list([result.usage])
                promoted = self.store.promote_item(
                    item_workspace,
                    aligned,
                    input_hash=item_hash,
                    config_hash=item_config_hash,
                    backend_id=backend_id,
                    backend_revision=descriptor.revision,
                    usage=item_usage,
                )
                aligned = AlignedSceneWords.model_validate(promoted)
                usage.extend(item_usage)
            scene_words[timeline_scene.scene_id] = aligned.words
            coverage[timeline_scene.scene_id] = aligned.coverage
        track = caption_track_from_timeline(
            narration.timeline,
            narration.script,
            scene_words=scene_words,
            coverage_by_scene=coverage,
            language=self.config.output_language,
        )
        srt_ref = None
        ass_ref = None
        if self.config.captions_enabled:
            srt_path = workspace.work_dir / f"captions.{self.config.output_language.value}.srt"
            write_srt(track, srt_path)
            srt_ref = MediaReference(
                path=relative_path(srt_path, self.project_root),
                sha256=sha256_file(srt_path),
                mime_type="application/x-subrip",
            )
        if self.config.captions_enabled and self.config.animated_captions:
            ass_path = workspace.work_dir / f"captions.{self.config.output_language.value}.ass"
            write_ass(track, ass_path, width=self.config.delivery_width, height=self.config.delivery_height)
            ass_ref = MediaReference(
                path=relative_path(ass_path, self.project_root),
                sha256=sha256_file(ass_path),
                mime_type="text/x-ass",
            )
        bundle = CaptionBundle(
            enabled=self.config.captions_enabled,
            track=track,
            srt=srt_ref,
            ass=ass_ref,
            scene_words={scene_id: list(words) for scene_id, words in scene_words.items()},
        )
        promoted = self.store.promote_stage(workspace, bundle, usage=usage)
        return CaptionBundle.model_validate(promoted)

    @staticmethod
    def _visual_briefs(visual_plan: VisualPlanLike) -> list[Any]:
        return list(visual_plan.shots if isinstance(visual_plan, TimedVisualPlan) else visual_plan.scenes)

    @staticmethod
    def _visual_key(value: Any) -> str:
        shot_id = getattr(value, "shot_id", None)
        return str(shot_id or value.scene_id)

    def _visual_plan_payload(self, visual_plan: VisualPlanLike) -> dict[str, Any]:
        payload = visual_plan.model_dump(mode="json")
        if self.continuity_policy_enabled:
            return payload
        for character in payload.get("characters", []):
            for field_name in (
                "body_form",
                "proportions",
                "face_and_markings",
                "wardrobe",
                "identity_constraints",
            ):
                character.pop(field_name, None)
        for scene in payload.get("scenes", []):
            for field_name in (
                "continuity_from_previous",
                "state_after_scene",
                "identity_requirements",
                "persistent_elements",
            ):
                scene.pop(field_name, None)
        return payload

    def _image_prompts(self, visual_plan: VisualPlanLike) -> ImageRequestSet:
        target_backend_id = self.config.task_bindings["image_generate"]
        target_descriptor = self.registry.descriptor(target_backend_id)
        input_data = {
            "visual_plan": self._visual_plan_payload(visual_plan),
            "target_backend_id": target_backend_id,
            "target_revision": target_descriptor.revision,
        }
        metadata = self._stage_metadata(
            stage="image-prompt-compile",
            task_id="image_prompt_compile",
            input_data=input_data,
            target_image_backend=target_backend_id,
        )
        reusable = self.store.reusable_record("image-prompt-compile", **metadata)
        if reusable:
            return self.store.load_artifact(reusable, ImageRequestSet)
        self.store.begin_stage(
            "image-prompt-compile", attempt=self.store.next_attempt("image-prompt-compile"), **metadata
        )
        requests = []
        usage = []
        generation_width, generation_height = image_generation_dimensions(
            target_backend_id,
            delivery_width=self.config.delivery_width,
            delivery_height=self.config.delivery_height,
        )
        characters_by_id = {
            character.character_id: character for character in visual_plan.characters
        }
        visual_briefs = self._visual_briefs(visual_plan)
        request_model: type[ImageRequestLike] = (
            TimedImageRequest if isinstance(visual_plan, TimedVisualPlan) else ImageRequest
        )
        for index, visual_brief in enumerate(visual_briefs):
            visual_id = self._visual_key(visual_brief)
            previous_brief = visual_briefs[index - 1] if index > 0 else None
            next_brief = (
                visual_briefs[index + 1]
                if index + 1 < len(visual_briefs)
                else None
            )
            relevant_characters = (
                [
                    characters_by_id[character_id].model_dump(mode="json")
                    for character_id in visual_brief.character_ids
                ]
                if self.continuity_policy_enabled
                else [
                    character
                    for character in self._visual_plan_payload(visual_plan)["characters"]
                ]
            )
            item_input = {
                "visual_shot" if isinstance(visual_plan, TimedVisualPlan) else "visual_brief": (
                    visual_brief.model_dump(mode="json")
                    if self.continuity_policy_enabled
                    else self._visual_plan_payload(visual_plan)["scenes"][index]
                ),
                "style_profile": visual_plan.style_profile.model_dump(mode="json"),
                "characters": relevant_characters,
                "target_backend_id": target_backend_id,
                "target_descriptor": {
                    "model_id": target_descriptor.model_id,
                    "revision": target_descriptor.revision,
                    "supports_reference_images": target_descriptor.supports_reference_images,
                },
                "generation_width": generation_width,
                "generation_height": generation_height,
                "image_quality": "low" if self.config.quality is Quality.DRAFT else "high",
                "reference_paths": [],
            }
            if self.continuity_policy_enabled:
                item_input["continuity_context"] = {
                    "previous_brief": (
                        previous_brief.model_dump(mode="json") if previous_brief else None
                    ),
                    "next_brief": next_brief.model_dump(mode="json") if next_brief else None,
                    "rule": "Context only: depict the current Visual Brief, never an adjacent action.",
                }
            item_hash = hash_run_input(item_input)
            item_config_hash = hash_value(
                {
                    "compiler": self.config.task_bindings["image_prompt_compile"],
                    "target": target_backend_id,
                    "language": self.prompts.output_language(
                        "image_prompt_compile", self.config.output_language
                    ).value,
                }
            )
            compiler_id = self.config.task_bindings["image_prompt_compile"]
            compiler_descriptor = self.registry.descriptor(compiler_id)
            prompt = self.prompts.get(
                "image_prompt_compile",
                language=self.config.output_language,
                target_image_backend=target_backend_id,
            )
            schema_hash = hash_value(self.prompts.schema("image_prompt_compile"))
            reusable_item = self.store.reusable_item(
                "image-prompt-compile",
                visual_id,
                input_hash=item_hash,
                config_hash=item_config_hash,
                backend_id=compiler_id,
                backend_revision=compiler_descriptor.revision,
                prompt_version=prompt.version,
                schema_hash=schema_hash,
            )
            if reusable_item:
                image_request = self.store.load_item_artifact(reusable_item, request_model)
                self._validate_image_request_language(image_request)
                usage.extend(reusable_item.usage)
            else:
                item_workspace = self.store.workspace("image-prompt-compile", item_id=visual_id)
                execution = self.executor.structured(
                    "image_prompt_compile",
                    item_input,
                    request_model,
                    target_image_backend=target_backend_id,
                    invariant=self._validate_image_request_language,
                )
                image_request = request_model.model_validate(execution.artifact)
                image_request = self._canonical_image_request(
                    image_request,
                    scene_id=visual_brief.scene_id,
                    shot_id=getattr(visual_brief, "shot_id", None),
                    target_backend_id=target_backend_id,
                    width=generation_width,
                    height=generation_height,
                    quality=item_input["image_quality"],
                    reference_paths=item_input["reference_paths"],
                )
                atomic_write_json(item_workspace.work_dir / "provider-response.json", execution.result.raw_response)
                item_usage = _usage_list([execution.result.usage])
                promoted = self.store.promote_item(
                    item_workspace,
                    image_request,
                    input_hash=item_hash,
                    config_hash=item_config_hash,
                    backend_id=compiler_id,
                    backend_revision=compiler_descriptor.revision,
                    prompt_version=prompt.version,
                    schema_hash=schema_hash,
                    usage=item_usage,
                )
                image_request = request_model.model_validate(promoted)
                usage.extend(item_usage)
            requests.append(image_request)
        bundle = ImageRequestSet(
            requests=requests,
            character_ids_by_scene=(
                {
                    self._visual_key(brief): list(brief.character_ids)
                    for brief in visual_briefs
                }
                if self.continuity_policy_enabled
                else {}
            ),
        )
        promoted = self.store.complete_fanout_stage("image-prompt-compile", bundle, usage=usage)
        return ImageRequestSet.model_validate(promoted)

    @staticmethod
    def _canonical_image_request(
        image_request: ImageRequestLike,
        *,
        scene_id: str,
        target_backend_id: str,
        width: int,
        height: int,
        quality: str,
        reference_paths: list[str],
        shot_id: str | None = None,
    ) -> ImageRequestLike:
        settings = image_request.settings
        if target_backend_id == "local:flux.2-klein-4b":
            settings = settings.model_copy(
                update={"inference_steps": 4, "guidance_scale": 1.0}
            )
        elif target_backend_id == "gemini:gemini-3.1-flash-image":
            settings = settings.model_copy(
                update={
                    "output_format": "jpeg",
                    "aspect_ratio": "16:9",
                    "image_size": "2K" if max(width, height) >= 1600 else "1K",
                }
            )
        updates: dict[str, Any] = {
                "scene_id": scene_id,
                "target_backend_id": target_backend_id,
                "width": width,
                "height": height,
                "quality": quality,
                "reference_paths": reference_paths,
                "settings": settings,
        }
        if isinstance(image_request, TimedImageRequest):
            updates["shot_id"] = shot_id
        return image_request.model_copy(update=updates)

    @staticmethod
    def _validate_image_request_language(image_request: ImageRequestLike) -> None:
        for field_name in ("prompt", "negative_prompt"):
            value = getattr(image_request, field_name).strip()
            if not value:
                continue
            try:
                language = detect(value)
            except LangDetectException as exc:
                raise BackendError(
                    f"ImageRequest.{field_name} language could not be verified as English",
                    kind=ErrorKind.INVALID_OUTPUT,
                ) from exc
            if language != "en":
                raise BackendError(
                    f"ImageRequest.{field_name} must be English; detected {language}",
                    kind=ErrorKind.INVALID_OUTPUT,
                )

    @staticmethod
    def _with_continuity_references(
        request: ImageRequestLike,
        *,
        character_ids: Sequence[str],
        character_reference_paths: dict[str, str],
        supports_reference_images: bool,
    ) -> ImageRequestLike:
        reference_paths = list(request.reference_paths)
        if supports_reference_images:
            for character_id in character_ids:
                known_path = character_reference_paths.get(character_id)
                if known_path and known_path not in reference_paths:
                    reference_paths.append(known_path)
                if len(reference_paths) >= 3:
                    break
        if not reference_paths:
            return request
        return request.model_copy(
            update={
                "reference_paths": reference_paths,
                "prompt": (
                    request.prompt
                    + "\n\nContinuity references are identity/style evidence only. Preserve "
                    "the same body form, proportions, face, markings, colors, wardrobe, and "
                    "recurring props. Do not copy a reference pose, framing, expression, action, "
                    "or background; the current scene instructions win."
                ),
            }
        )

    def _images(self, request_set: ImageRequestSet) -> ImageSet:
        input_data = request_set.model_dump(mode="json")
        metadata = self._stage_metadata(stage="images", task_id=None, input_data=input_data)
        backend_id = self.config.task_bindings["image_generate"]
        descriptor = self.registry.descriptor(backend_id)
        metadata["backend_id"] = backend_id
        metadata["backend_revision"] = descriptor.revision
        reusable = self.store.reusable_record("images", **metadata)
        if reusable:
            return self.store.load_artifact(reusable, ImageSet)
        self.store.begin_stage("images", attempt=self.store.next_attempt("images"), **metadata)
        items = []
        usage = []
        character_reference_paths: dict[str, str] = {}
        for request in request_set.requests:
            visual_id = self._visual_key(request)
            effective_request = self._with_continuity_references(
                request,
                character_ids=request_set.character_ids_by_scene.get(visual_id, []),
                character_reference_paths=character_reference_paths,
                supports_reference_images=descriptor.supports_reference_images,
            )
            item_input = effective_request.model_dump(mode="json")
            if effective_request.reference_paths:
                item_input = {
                    "request": item_input,
                    "reference_sha256": [
                        sha256_file(self.project_root / reference_path)
                        for reference_path in effective_request.reference_paths
                    ],
                }
            item_hash = hash_run_input(item_input)
            config_hash = hash_value(
                {"backend": backend_id, "width": self.config.delivery_width, "height": self.config.delivery_height}
            )
            reusable_item = self.store.reusable_item(
                "images",
                visual_id,
                input_hash=item_hash,
                config_hash=config_hash,
                backend_id=backend_id,
                backend_revision=descriptor.revision,
            )
            if reusable_item:
                item = self.store.load_item_artifact(reusable_item, ImageItem)
                if item.request is None:
                    item = item.model_copy(update={"request": effective_request})
                usage.extend(reusable_item.usage)
            else:
                workspace = self.store.workspace("images", item_id=visual_id)
                extension = _raw_image_extension(backend_id)
                raw_path = workspace.work_dir / f"generated{extension}"
                normalized_path = workspace.work_dir / "normalized.png"
                result = self.executor.image(effective_request, raw_path)
                if result.asset.scene_id != effective_request.scene_id:
                    raise BackendError("Image Backend changed the Scene ID", kind=ErrorKind.INVALID_OUTPUT)
                if result.asset.shot_id != getattr(effective_request, "shot_id", None):
                    raise BackendError("Image Backend changed the Shot ID", kind=ErrorKind.INVALID_OUTPUT)
                normalize_image(
                    self.tools,
                    self.project_root / result.asset.image.path,
                    normalized_path,
                    width=self.config.delivery_width,
                    height=self.config.delivery_height,
                )
                item = ImageItem(
                    generated=result.asset,
                    normalized_image=MediaReference(
                        path=relative_path(normalized_path, self.project_root),
                        sha256=sha256_file(normalized_path),
                        mime_type="image/png",
                    ),
                    request=effective_request,
                )
                item_usage = _usage_list([result.usage])
                promoted = self.store.promote_item(
                    workspace,
                    item,
                    input_hash=item_hash,
                    config_hash=config_hash,
                    backend_id=backend_id,
                    backend_revision=descriptor.revision,
                    usage=item_usage,
                )
                item = ImageItem.model_validate(promoted)
                usage.extend(item_usage)
            for character_id in request_set.character_ids_by_scene.get(visual_id, []):
                character_reference_paths.setdefault(
                    character_id, item.normalized_image.path
                )
            items.append(item)
        bundle = ImageSet(items=items)
        promoted = self.store.complete_fanout_stage("images", bundle, usage=usage)
        return ImageSet.model_validate(promoted)

    def _regenerate_visual_review_image(
        self,
        *,
        item_id: str,
        request: ImageRequestLike,
        reason: str,
    ) -> tuple[ImageItem, list[UsageRecord]]:
        backend_id = self.config.task_bindings["image_generate"]
        descriptor = self.registry.descriptor(backend_id)
        reference_hashes = [
            sha256_file(self.project_root / reference_path)
            for reference_path in request.reference_paths
        ]
        regeneration_input_hash = hash_run_input(
            {
                "request": request.model_dump(mode="json"),
                "reference_sha256": reference_hashes,
            }
        )
        regeneration_config_hash = hash_value(
            {"backend": backend_id, "regeneration": reason}
        )
        reusable_item = self.store.reusable_item(
            "visual-review",
            item_id,
            input_hash=regeneration_input_hash,
            config_hash=regeneration_config_hash,
            backend_id=backend_id,
            backend_revision=descriptor.revision,
        )
        if reusable_item:
            replacement = self.store.load_item_artifact(reusable_item, ImageItem)
            if replacement.request is None:
                replacement = replacement.model_copy(update={"request": request})
            return replacement, reusable_item.usage

        workspace = self.store.workspace("visual-review", item_id=item_id)
        extension = _raw_image_extension(backend_id)
        raw_path = workspace.work_dir / f"generated{extension}"
        normalized_path = workspace.work_dir / "normalized.png"
        result = self.executor.image(request, raw_path)
        if result.asset.scene_id != request.scene_id:
            raise BackendError("Image Backend changed the Scene ID", kind=ErrorKind.INVALID_OUTPUT)
        if result.asset.shot_id != getattr(request, "shot_id", None):
            raise BackendError("Image Backend changed the Shot ID", kind=ErrorKind.INVALID_OUTPUT)
        normalize_image(
            self.tools,
            self.project_root / result.asset.image.path,
            normalized_path,
            width=self.config.delivery_width,
            height=self.config.delivery_height,
        )
        replacement = ImageItem(
            generated=result.asset,
            normalized_image=MediaReference(
                path=relative_path(normalized_path, self.project_root),
                sha256=sha256_file(normalized_path),
                mime_type="image/png",
            ),
            request=request,
        )
        item_usage = _usage_list([result.usage])
        promoted = self.store.promote_item(
            workspace,
            replacement,
            input_hash=regeneration_input_hash,
            config_hash=regeneration_config_hash,
            backend_id=backend_id,
            backend_revision=descriptor.revision,
            usage=item_usage,
        )
        return ImageItem.model_validate(promoted), item_usage

    def _visual_review(
        self, visual_plan: VisualPlanLike, requests: ImageRequestSet, images: ImageSet
    ) -> VisualReviewBundle:
        input_data = {
            "visual_plan": visual_plan.model_dump(mode="json"),
            "images": images.model_dump(mode="json"),
            "quality": self.config.quality.value,
        }
        metadata = self._stage_metadata(stage="visual-review", task_id=None, input_data=input_data)
        if self.config.quality is Quality.FINAL:
            backend_id = self.config.task_bindings["visual_review"]
            metadata["backend_id"] = backend_id
            metadata["backend_revision"] = self.registry.descriptor(backend_id).revision
        reusable = self.store.reusable_record("visual-review", **metadata)
        if reusable:
            return self.store.load_artifact(reusable, VisualReviewBundle)
        aggregate_workspace = self.store.workspace("visual-review")
        self.store.begin_stage("visual-review", attempt=aggregate_workspace.attempt, **metadata)
        if self.config.quality is Quality.DRAFT:
            bundle = VisualReviewBundle(reviewed=False, images=images)
            promoted = self.store.promote_stage(aggregate_workspace, bundle)
            return VisualReviewBundle.model_validate(promoted)
        visual_briefs = self._visual_briefs(visual_plan)
        briefs = {self._visual_key(brief): brief for brief in visual_briefs}
        request_by_id = {self._visual_key(request): request for request in requests.requests}
        final_items = {self._visual_key(item.generated): item for item in images.items}
        expected_visual_ids = set(briefs)
        self._require(
            set(request_by_id) == expected_visual_ids,
            "Image Requests do not cover every Visual Brief exactly once",
        )
        self._require(
            set(final_items) == expected_visual_ids,
            "Generated images do not cover every Visual Brief exactly once",
        )
        reviews: list[VisualReviewItemLike] = []
        usage = []
        failed = []
        for item in images.items:
            visual_id = self._visual_key(item.generated)
            review = self._review_image(briefs[visual_id], visual_plan, item, pass_number=1)
            reviews.append(review[0])
            usage.extend(review[1])
            if not review[0].passed:
                failed.append(review[0])
        second_reviews: list[VisualReviewItemLike] = []
        if failed:
            replacements: dict[str, ImageItem] = {}
            regeneration_usage: dict[str, list[UsageRecord]] = {}
            reference_replacements: dict[str, str] = {}
            for failure in failed:
                visual_id = self._visual_key(failure)
                original_item = final_items[visual_id]
                original = original_item.request or request_by_id[visual_id]
                remapped_references = [
                    reference_replacements.get(path, path)
                    for path in original.reference_paths
                ]
                corrected = original.model_copy(
                    update={
                        "reference_paths": remapped_references,
                        "prompt": (
                            original.prompt
                            + "\n\nTargeted correction. Preserve all otherwise correct content. "
                            + failure.regeneration_instruction
                        )
                    }
                )
                regeneration_id = f"{visual_id}-regeneration"
                replacement, item_usage = self._regenerate_visual_review_image(
                    item_id=regeneration_id,
                    request=corrected,
                    reason="targeted-correction-v2",
                )
                replacements[visual_id] = replacement
                regeneration_usage[visual_id] = item_usage
                final_items[visual_id] = replacement
                reference_replacements[
                    original_item.normalized_image.path
                ] = replacement.normalized_image.path

            propagated_visual_ids: list[str] = []
            failed_visual_ids = {self._visual_key(failure) for failure in failed}
            for brief in visual_briefs:
                visual_id = self._visual_key(brief)
                if visual_id in failed_visual_ids:
                    continue
                original_item = final_items[visual_id]
                original = original_item.request or request_by_id[visual_id]
                remapped_references = [
                    reference_replacements.get(path, path)
                    for path in original.reference_paths
                ]
                if remapped_references == original.reference_paths:
                    continue
                corrected = original.model_copy(
                    update={
                        "reference_paths": remapped_references,
                        "prompt": (
                            original.prompt
                            + "\n\nAn upstream identity reference was corrected. Recreate this "
                            "same scene using the replacement identity evidence while preserving "
                            "the current action, setting, composition, and emotional beat."
                        ),
                    }
                )
                replacement, item_usage = self._regenerate_visual_review_image(
                    item_id=f"{visual_id}-continuity-regeneration",
                    request=corrected,
                    reason="replacement-reference-v1",
                )
                final_items[visual_id] = replacement
                replacements[visual_id] = replacement
                regeneration_usage[visual_id] = item_usage
                reference_replacements[
                    original_item.normalized_image.path
                ] = replacement.normalized_image.path
                propagated_visual_ids.append(visual_id)

            review_visual_ids = [self._visual_key(failure) for failure in failed] + propagated_visual_ids
            for visual_id in review_visual_ids:
                replacement = replacements[visual_id]
                usage.extend(regeneration_usage[visual_id])
                second = self._review_image(briefs[visual_id], visual_plan, replacement, pass_number=2)
                second_reviews.append(second[0])
                usage.extend(second[1])
                if not second[0].passed:
                    raise BackendError(
                        f"regenerated image for {visual_id} failed its final re-review",
                        kind=ErrorKind.INVALID_OUTPUT,
                        action="Inspect the Visual Review and explicitly rerun from image-prompt-compile or images.",
                    )
        ordered_images = ImageSet(items=[final_items[self._visual_key(brief)] for brief in visual_briefs])
        report = VisualReviewReport(items=reviews + second_reviews, pass_number=2 if second_reviews else 1)
        bundle = VisualReviewBundle(reviewed=True, report=report, images=ordered_images)
        promoted = self.store.promote_stage(aggregate_workspace, bundle, usage=usage)
        return VisualReviewBundle.model_validate(promoted)

    def _review_image(
        self,
        brief: Any,
        visual_plan: VisualPlanLike,
        image: ImageItem,
        *,
        pass_number: int,
    ) -> tuple[VisualReviewItemLike, list[UsageRecord]]:
        backend_id = self.config.task_bindings["visual_review"]
        descriptor = self.registry.descriptor(backend_id)
        visual_id = self._visual_key(brief)
        item_id = f"{visual_id}-review-{pass_number}"
        character_ids = set(brief.character_ids)
        relevant_characters = [
            character.model_dump(mode="json")
            for character in visual_plan.characters
            if character.character_id in character_ids
        ]
        reference_paths = list(image.request.reference_paths) if image.request else []
        reference_hashes = [
            sha256_file(self.project_root / reference_path)
            for reference_path in reference_paths
        ]
        item_input = {
            "scene_id": brief.scene_id,
            "shot_id": getattr(brief, "shot_id", None),
            "visual_brief": brief.model_dump(mode="json"),
            "style_profile": visual_plan.style_profile.model_dump(mode="json"),
            "characters": relevant_characters,
            "audience": self.config.audience,
            "pass_number": pass_number,
            "minimum_score": 4,
            "image_sha256": image.normalized_image.sha256,
            "media_order": ["current_scene", *["identity_reference"] * len(reference_paths)],
            "reference_image_sha256": reference_hashes,
        }
        prompt = self.prompts.get("visual_review", language=self.config.output_language)
        schema_hash = hash_value(self.prompts.schema("visual_review"))
        item_hash = hash_run_input(item_input)
        item_config_hash = hash_value({"backend": backend_id, "quality": self.config.quality.value})
        reusable = self.store.reusable_item(
            "visual-review",
            item_id,
            input_hash=item_hash,
            config_hash=item_config_hash,
            backend_id=backend_id,
            backend_revision=descriptor.revision,
            prompt_version=prompt.version,
            schema_hash=schema_hash,
        )
        review_model: type[VisualReviewItemLike] = (
            TimedVisualReviewItem if isinstance(visual_plan, TimedVisualPlan) else VisualReviewItem
        )
        if reusable:
            return self.store.load_item_artifact(reusable, review_model), reusable.usage
        workspace = self.store.workspace("visual-review", item_id=item_id)
        execution = self.executor.structured(
            "visual_review",
            item_input,
            review_model,
            media_inputs=[
                self.project_root / image.normalized_image.path,
                *[self.project_root / path for path in reference_paths],
            ],
        )
        item = review_model.model_validate(execution.artifact)
        if item.scene_id != brief.scene_id:
            raise BackendError("Visual Review changed the Scene ID", kind=ErrorKind.INVALID_OUTPUT)
        if isinstance(item, TimedVisualReviewItem) and item.shot_id != visual_id:
            raise BackendError("Visual Review changed the Shot ID", kind=ErrorKind.INVALID_OUTPUT)
        if pass_number == 1 and not item.passed and not item.regeneration_instruction.strip():
            raise BackendError(
                "failed first-pass Visual Review requires a targeted regeneration instruction",
                kind=ErrorKind.INVALID_OUTPUT,
            )
        if pass_number == 2 and item.regeneration_instruction:
            item.regeneration_instruction = ""
        atomic_write_json(workspace.work_dir / "provider-response.json", execution.result.raw_response)
        item_usage = _usage_list([execution.result.usage])
        promoted = self.store.promote_item(
            workspace,
            item,
            input_hash=item_hash,
            config_hash=item_config_hash,
            backend_id=backend_id,
            backend_revision=descriptor.revision,
            prompt_version=prompt.version,
            schema_hash=schema_hash,
            usage=item_usage,
        )
        return review_model.model_validate(promoted), item_usage

    def _music_brief_context(
        self, narration: NarrationBundle
    ) -> tuple[BackendDescriptor, float, float, dict[str, Any], dict[str, str]]:
        music_descriptor = self.registry.descriptor(self.config.task_bindings["music_generate"])
        timeline_duration = narration.timeline.duration_seconds
        generation_duration = min(
            timeline_duration,
            float(music_descriptor.max_duration_seconds or timeline_duration),
        )
        input_data = {
            "enabled": self.config.music_enabled,
            "duration_seconds": generation_duration,
            "generation_duration_seconds": generation_duration,
            "timeline_duration_seconds": timeline_duration,
            "script": narration.script.model_dump(mode="json"),
            "timeline": narration.timeline.model_dump(mode="json"),
        }
        metadata = self._stage_metadata(
            stage="music-brief",
            task_id="music_brief" if self.config.music_enabled else None,
            input_data=input_data,
        )
        return music_descriptor, timeline_duration, generation_duration, input_data, metadata

    def _prepare_music_brief(
        self, narration: NarrationBundle
    ) -> tuple[MusicBriefBundle, list[UsageRecord]]:
        if not self.config.music_enabled:
            raise ValueError("music brief preparation requires music to be enabled")
        music_descriptor, timeline_duration, generation_duration, input_data, metadata = (
            self._music_brief_context(narration)
        )
        reusable = self.store.reusable_record("music-brief", **metadata)
        if reusable:
            return self.store.load_artifact(reusable, MusicBriefBundle), reusable.usage
        reusable_item = self.store.reusable_item("music-brief", "brief", **metadata)
        if reusable_item:
            return (
                self.store.load_item_artifact(reusable_item, MusicBriefBundle),
                reusable_item.usage,
            )
        workspace = self.store.workspace("music-brief", item_id="brief")
        execution = self.executor.structured("music_brief", input_data, MusicBrief)
        brief = MusicBrief.model_validate(execution.artifact)
        if abs(brief.requested_duration_seconds - generation_duration) > 0.01:
            raise BackendError(
                "Music Brief duration does not match the Backend generation duration",
                kind=ErrorKind.INVALID_OUTPUT,
            )
        if generation_duration + 0.01 < timeline_duration and (
            not music_descriptor.supports_looping or not brief.seamless_loop_preferred
        ):
            raise BackendError(
                "Music Brief must request a seamless loop for a duration-limited Backend",
                kind=ErrorKind.INVALID_OUTPUT,
            )
        atomic_write_json(workspace.work_dir / "provider-response.json", execution.result.raw_response)
        bundle = MusicBriefBundle(enabled=True, brief=brief)
        item_usage = _usage_list([execution.result.usage])
        promoted = self.store.promote_item(
            workspace,
            bundle,
            usage=item_usage,
            **metadata,
        )
        return MusicBriefBundle.model_validate(promoted), item_usage

    def _music_brief(
        self,
        narration: NarrationBundle,
        *,
        prepared: tuple[MusicBriefBundle, list[UsageRecord]] | None = None,
    ) -> MusicBriefBundle:
        _, _, _, _, metadata = self._music_brief_context(narration)
        reusable = self.store.reusable_record("music-brief", **metadata)
        if reusable:
            return self.store.load_artifact(reusable, MusicBriefBundle)
        workspace = self.store.workspace("music-brief")
        self.store.begin_stage("music-brief", attempt=workspace.attempt, **metadata)
        if not self.config.music_enabled:
            promoted = self.store.promote_stage(workspace, MusicBriefBundle(enabled=False))
            return MusicBriefBundle.model_validate(promoted)
        bundle, usage = prepared or self._prepare_music_brief(narration)
        promoted = self.store.promote_stage(workspace, bundle, usage=usage)
        return MusicBriefBundle.model_validate(promoted)

    def _music(self, brief_bundle: MusicBriefBundle, timeline: NarrationTimeline) -> MusicBundle:
        input_data = {
            "brief": brief_bundle.model_dump(mode="json"),
            "timeline_duration": timeline.duration_seconds,
        }
        metadata = self._stage_metadata(stage="music", task_id=None, input_data=input_data)
        if brief_bundle.enabled:
            backend_id = self.config.task_bindings["music_generate"]
            metadata["backend_id"] = backend_id
            metadata["backend_revision"] = self.registry.descriptor(backend_id).revision
        reusable = self.store.reusable_record("music", **metadata)
        if reusable:
            return self.store.load_artifact(reusable, MusicBundle)
        workspace = self.store.workspace("music")
        self.store.begin_stage("music", attempt=workspace.attempt, **metadata)
        if not brief_bundle.enabled or brief_bundle.brief is None:
            promoted = self.store.promote_stage(workspace, MusicBundle(enabled=False))
            return MusicBundle.model_validate(promoted)
        raw_path = workspace.work_dir / "generated-music.mp3"
        if self.config.task_bindings["music_generate"].startswith("local:"):
            raw_path = raw_path.with_suffix(".wav")
        fitted_path = workspace.work_dir / "music-bed.wav"
        try:
            result = self.executor.music(
                MusicRequest(
                    brief=brief_bundle.brief,
                    output_path=relative_path(raw_path, self.project_root),
                    output_language=self.config.output_language,
                )
            )
            expected_music_duration = brief_bundle.brief.requested_duration_seconds
            if abs(result.asset.duration_seconds - expected_music_duration) > max(
                1.0, expected_music_duration * 0.01
            ):
                raise BackendError(
                    "Music Backend silently changed the requested generation duration",
                    kind=ErrorKind.INVALID_OUTPUT,
                )
            allow_loop = bool(brief_bundle.brief.seamless_loop_preferred)
            fit_music(
                self.tools,
                self.project_root / result.asset.audio.path,
                fitted_path,
                duration=timeline.duration_seconds,
                allow_loop=allow_loop,
            )
            bundle = MusicBundle(
                enabled=True,
                generated=result.asset,
                fitted_audio=MediaReference(
                    path=relative_path(fitted_path, self.project_root),
                    sha256=sha256_file(fitted_path),
                    mime_type="audio/wav",
                ),
            )
            promoted = self.store.promote_stage(
                workspace, bundle, usage=_usage_list([result.usage])
            )
            return MusicBundle.model_validate(promoted)
        except VideoGeneratorError as exc:
            if self.config.failure_policy is FailurePolicy.STRICT:
                raise
            warning = f"music omitted by Failure Policy: {exc.message}"
            bundle = MusicBundle(enabled=False, warning=warning)
            promoted = self.store.promote_stage(workspace, bundle, warnings=[warning])
            self.store.add_warning(warning)
            return MusicBundle.model_validate(promoted)

    def _render(
        self,
        timeline: NarrationTimeline,
        captions: CaptionBundle,
        images: ImageSet,
        music: MusicBundle,
        visual_plan: VisualPlanLike,
    ) -> RenderBundle:
        input_data = {
            "timeline": timeline.model_dump(mode="json"),
            "captions": captions.model_dump(mode="json"),
            "images": images.model_dump(mode="json"),
            "music": music.model_dump(mode="json"),
            "visual_plan": visual_plan.model_dump(mode="json"),
        }
        metadata = self._stage_metadata(stage="render", task_id=None, input_data=input_data)
        reusable = self.store.reusable_record("render", **metadata)
        if reusable:
            return self.store.load_artifact(reusable, RenderBundle)
        workspace = self.store.workspace("render")
        self.store.begin_stage("render", attempt=workspace.attempt, **metadata)
        image_by_visual = {self._visual_key(item.generated): item for item in images.items}
        expected_visual_ids = {
            self._visual_key(visual)
            for visual in self._visual_briefs(visual_plan)
        }
        self._require(
            set(image_by_visual) == expected_visual_ids,
            "Render input does not contain exactly one image per planned visual",
        )
        if isinstance(visual_plan, TimedVisualPlan):
            render_scenes = [
                RenderScene(
                    scene_id=shot.scene_id,
                    shot_id=shot.shot_id,
                    image_path=image_by_visual[shot.shot_id].normalized_image.path,
                    start_seconds=shot.start_seconds,
                    end_seconds=shot.end_seconds,
                )
                for shot in visual_plan.shots
            ]
        else:
            render_scenes = [
                RenderScene(
                    scene_id=scene.scene_id,
                    image_path=image_by_visual[scene.scene_id].normalized_image.path,
                    start_seconds=scene.start_seconds,
                    end_seconds=scene.end_seconds,
                )
                for scene in timeline.scenes
            ]
        render_scenes[-1].end_seconds = timeline.delivery_duration_seconds
        plan = RenderPlan(
            scenes=render_scenes,
            narration_path=timeline.narration_audio.path,
            music_path=music.fitted_audio.path if music.enabled and music.fitted_audio else None,
            caption_srt_path=captions.srt.path if captions.enabled and captions.srt else None,
            caption_ass_path=captions.ass.path if captions.enabled and captions.ass else None,
            caption_language=self.config.output_language if captions.enabled else None,
            width=self.config.delivery_width,
            height=self.config.delivery_height,
            fps=self.config.fps,
            duration_seconds=timeline.delivery_duration_seconds,
            caption_mode=(
                "selectable_and_burned"
                if captions.enabled and captions.ass
                else "selectable"
                if captions.enabled
                else "none"
            ),
        )
        base = workspace.work_dir / "base.mp4"
        primary = workspace.work_dir / "video.mp4"
        burned = workspace.work_dir / "video-captioned.mp4" if captions.ass else None
        outputs = render_video(
            self.tools,
            plan,
            workspace_root=self.project_root,
            base_path=base,
            output_path=primary,
            burned_output_path=burned,
        )
        primary_checks = qc_video(
            self.tools,
            primary,
            width=self.config.delivery_width,
            height=self.config.delivery_height,
            fps=self.config.fps,
            expected_duration=plan.duration_seconds,
            budget=self.config.duration_seconds,
            captions_expected=captions.enabled,
        )
        if not all(check.passed for check in primary_checks):
            failures = "; ".join(
                f"{check.name}: {check.detail}" for check in primary_checks if not check.passed
            )
            raise MediaError(f"rendered media QC failed: {failures}", kind=ErrorKind.INVALID_OUTPUT)
        if burned and burned in outputs:
            burned_checks = qc_video(
                self.tools,
                burned,
                width=self.config.delivery_width,
                height=self.config.delivery_height,
                fps=self.config.fps,
                expected_duration=plan.duration_seconds,
                budget=self.config.duration_seconds,
                captions_expected=False,
            )
            if not all(check.passed for check in burned_checks):
                failures = "; ".join(
                    f"{check.name}: {check.detail}" for check in burned_checks if not check.passed
                )
                raise MediaError(
                    f"burned-caption render QC failed: {failures}",
                    kind=ErrorKind.INVALID_OUTPUT,
                )
        bundle = RenderBundle(
            plan=plan,
            primary_video=MediaReference(
                path=relative_path(primary, self.project_root),
                sha256=sha256_file(primary),
                mime_type="video/mp4",
            ),
            burned_video=(
                MediaReference(
                    path=relative_path(burned, self.project_root),
                    sha256=sha256_file(burned),
                    mime_type="video/mp4",
                )
                if burned and burned in outputs
                else None
            ),
        )
        promoted = self.store.promote_stage(workspace, bundle)
        return RenderBundle.model_validate(promoted)

    def _delivery(self, rendered: RenderBundle, captions: CaptionBundle) -> DeliveryManifest:
        input_data = rendered.model_dump(mode="json")
        metadata = self._stage_metadata(stage="delivery", task_id=None, input_data=input_data)
        reusable = self.store.reusable_record("delivery", **metadata)
        if reusable:
            return self.store.load_artifact(reusable, DeliveryManifest)
        workspace = self.store.workspace("delivery")
        self.store.begin_stage("delivery", attempt=workspace.attempt, **metadata)
        outputs_dir = self.store.root / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        def copy_atomic(source: Path, destination: Path) -> None:
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            shutil.copy2(source, temporary)
            replace_path(temporary, destination)

        source_primary = self.project_root / rendered.primary_video.path
        checks = qc_video(
            self.tools,
            source_primary,
            width=self.config.delivery_width,
            height=self.config.delivery_height,
            fps=self.config.fps,
            expected_duration=rendered.plan.duration_seconds,
            budget=self.config.duration_seconds,
            captions_expected=captions.enabled,
        )
        if not all(check.passed for check in checks):
            failures = "; ".join(f"{check.name}: {check.detail}" for check in checks if not check.passed)
            raise MediaError(f"delivery media QC failed: {failures}", kind=ErrorKind.INVALID_OUTPUT)
        primary = outputs_dir / "video.mp4"
        copy_atomic(source_primary, primary)
        output_files: list[tuple[str, Path, str]] = [("primary_video", primary, "video/mp4")]
        if rendered.burned_video:
            source_burned = self.project_root / rendered.burned_video.path
            burned_checks = qc_video(
                self.tools,
                source_burned,
                width=self.config.delivery_width,
                height=self.config.delivery_height,
                fps=self.config.fps,
                expected_duration=rendered.plan.duration_seconds,
                budget=self.config.duration_seconds,
                captions_expected=False,
            )
            checks.extend(
                check.model_copy(update={"name": f"burned_{check.name}"})
                for check in burned_checks
            )
            if not all(check.passed for check in burned_checks):
                failures = "; ".join(
                    f"{check.name}: {check.detail}" for check in burned_checks if not check.passed
                )
                raise MediaError(
                    f"burned-caption media QC failed: {failures}",
                    kind=ErrorKind.INVALID_OUTPUT,
                )
            burned = outputs_dir / "video-captioned.mp4"
            copy_atomic(source_burned, burned)
            output_files.append(("burned_caption_video", burned, "video/mp4"))
        if captions.enabled and captions.srt:
            srt = outputs_dir / "captions.srt"
            copy_atomic(self.project_root / captions.srt.path, srt)
            output_files.append(("caption_sidecar", srt, "application/x-subrip"))
        manifest = delivery_manifest(
            run_id=self.store.manifest.run_id,
            output_files=output_files,
            workspace_root=self.project_root,
            duration=rendered.plan.duration_seconds,
            checks=checks,
            warnings=self.store.manifest.warnings,
        )
        atomic_write_json(workspace.work_dir / "delivery-manifest-copy.json", manifest.model_dump(mode="json"))
        output_manifest = outputs_dir / "delivery-manifest.json"
        atomic_write_json(output_manifest, manifest.model_dump(mode="json"))
        promoted = self.store.promote_stage(
            workspace,
            manifest,
            extra_files=[*[path for _, path, _ in output_files], output_manifest],
        )
        final_manifest = DeliveryManifest.model_validate(promoted)
        return final_manifest
