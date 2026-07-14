from __future__ import annotations

import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Sequence

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
    FactualClaimReview,
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
    RevisionDisposition,
    RevisedScript,
    SearchRequest,
    SelectionReport,
    SceneClaimExtraction,
    ScriptClaim,
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
    CAPTION_RECONCILIATION_REVISION,
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
MULTI_FORMAT_INTERNAL_REVISION = "media-workflow-v26"
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


class ReplacementText(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spoken_text: str = Field(min_length=1, max_length=10000)


class VisualStyleContent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str = Field(min_length=1, max_length=4000)
    palette: list[str] = Field(min_length=1, max_length=20)
    line_style: str = Field(min_length=1, max_length=1000)
    background: str = Field(min_length=1, max_length=1000)
    must_avoid: list[str] = Field(min_length=1, max_length=30)


class VisualCharacterContent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=200)
    signature_traits: list[str] = Field(min_length=1, max_length=30)
    color_anchors: list[str] = Field(default_factory=list, max_length=20)
    recurring_props: list[str] = Field(default_factory=list, max_length=20)
    body_form: str = Field(min_length=1, max_length=1000)
    proportions: list[str] = Field(min_length=1, max_length=20)
    face_and_markings: list[str] = Field(min_length=1, max_length=20)
    wardrobe: list[str] = Field(default_factory=list, max_length=20)
    identity_constraints: list[str] = Field(min_length=1, max_length=30)


class VisualFoundation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    style: VisualStyleContent
    characters: list[VisualCharacterContent] = Field(default_factory=list, max_length=20)


class VisualBriefContent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    story_moment: str = Field(min_length=1, max_length=2000)
    subjects: list[str] = Field(min_length=1, max_length=20)
    action: str = Field(min_length=1, max_length=2000)
    emotion: str = Field(min_length=1, max_length=1000)
    environment: str = Field(min_length=1, max_length=2000)
    composition: str = Field(min_length=1, max_length=2000)
    must_show: list[str] = Field(min_length=1, max_length=30)
    must_avoid: list[str] = Field(min_length=1, max_length=30)
    character_ids: list[str] = Field(default_factory=list, max_length=20)
    continuity_from_previous: list[str] = Field(default_factory=list, max_length=30)
    state_after_scene: list[str] = Field(default_factory=list, max_length=30)
    identity_requirements: list[str] = Field(default_factory=list, max_length=30)
    persistent_elements: list[str] = Field(default_factory=list, max_length=30)


class ImagePromptContent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=12000)
    negative_prompt: str = Field(default="", max_length=12000)


class ClaimReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdict: Literal[
        "supported",
        "needs_qualification",
        "unsupported",
        "not_a_factual_claim",
    ]
    evidence_ids: list[str] = Field(default_factory=list, max_length=20)
    rationale: str = Field(min_length=1, max_length=4000)


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
            def draft_invariant(value: NarrationScript) -> None:
                self._validate_draft(
                    value,
                    outline,
                    minimum_words=int(script_word_plan["minimum_total_word_count"]),
                    maximum_words=int(script_word_plan["maximum_total_word_count"]),
                    enforce_pause_limit=self.continuity_policy_enabled,
                    maximum_pause_seconds=self._maximum_authored_pause_seconds(),
                )

            if self.workflow_policy_version >= 5:
                draft = self._scene_local_draft_stage(
                    draft_input,
                    outline,
                    research,
                    invariant=draft_invariant,
                )
            else:
                draft = self._structured_stage(
                    "script-draft",
                    "script_draft",
                    draft_input,
                    NarrationScript,
                    invariant=draft_invariant,
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
            if self.workflow_policy_version >= 4:
                revised = self._scene_local_revision_stage(
                    revision_input,
                    reviews,
                    outline,
                    research,
                    invariant=revision_invariant,
                )
            elif self.config.content_mode is ContentMode.FACTUAL:
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
                outline=outline,
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
            if self.workflow_policy_version >= 7:
                visual_plan = self._split_visual_plan_stage(
                    visual_plan_input,
                    outline=outline,
                    script=narration.script,
                    schedule=canonical_shot_schedule,
                    timed_visuals=timed_visuals,
                    output_model=visual_plan_model,
                    invariant=visual_plan_invariant,
                )
            else:
                visual_plan = self._structured_stage(
                    "visual-plan",
                    "visual_plan",
                    visual_plan_input,
                    visual_plan_model,
                    invariant=visual_plan_invariant,
                    max_output_tokens=(
                        7000
                        if timed_visuals
                        and self.config.task_bindings["visual_plan"].startswith("local:")
                        else 16000
                        if timed_visuals
                        else 8000
                    ),
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
                        "target_image_backend": target_image_backend,
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
                reconciliation_revision=CAPTION_RECONCILIATION_REVISION,
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
            invariant=lambda value: self._validate_review(
                value,
                expected_type,
                task_id,
                {scene.scene_id for scene in script.scenes},
            ),
        )
        return report

    @staticmethod
    def _validate_review(
        report: ReviewReport,
        expected_type: str,
        task_id: str,
        expected_scene_ids: set[str],
    ) -> None:
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
            WorkflowEngine._require(
                finding.scene_id in expected_scene_ids,
                f"{task_id} Finding {finding.finding_id} must identify one supplied Scene ID",
            )

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
        if (
            getattr(self, "workflow_policy_version", 2) >= 12
            and self.config.visual_shot_mode is VisualShotMode.CADENCED
            and self.config.content_format is not ContentFormat.NARRATIVE
        ):
            editorial_target = (
                5 if self.config.content_format is ContentFormat.MYTHBUSTER else 4
            )
            feasible_minimum = max(
                1,
                math.ceil(self.config.duration_seconds / self.config.visual_max_seconds),
            )
            feasible_maximum = max(
                feasible_minimum,
                math.floor(self.config.duration_seconds / self.config.visual_min_seconds) + 1,
            )
            target_count = min(
                max(editorial_target, feasible_minimum),
                feasible_maximum,
            )
            return (
                target_count,
                max(feasible_minimum, target_count - 1),
                min(feasible_maximum, target_count + 1),
            )
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
            minimum_duration_fraction = (
                0.65
                if local_script_backend
                and getattr(self, "workflow_policy_version", 2) == 10
                else 0.75
                if local_script_backend
                else 0.85
            )
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
        minimum_total = max(
            len(outline.scenes) * 8,
            round(
                self.config.duration_seconds
                * minimum_duration_fraction
                * words_per_second
            ),
        )
        maximum_total = max(
            len(outline.scenes) * 8,
            round(
                self.config.duration_seconds
                * words_per_second
                * maximum_word_tolerance
            ),
        )

        def proportional_allocation(total: int) -> list[int]:
            weights = [int(item["target_word_count"]) for item in scene_targets]
            weight_total = sum(weights)
            raw = [total * weight / weight_total for weight in weights]
            result = [math.floor(value) for value in raw]
            remainder = total - sum(result)
            order = sorted(
                range(len(raw)),
                key=lambda index: (raw[index] - result[index], -index),
                reverse=True,
            )
            for index in order[:remainder]:
                result[index] += 1
            return result

        scene_minimums = proportional_allocation(minimum_total)
        scene_maximums = proportional_allocation(maximum_total)
        for scene_target, minimum, maximum in zip(
            scene_targets,
            scene_minimums,
            scene_maximums,
            strict=True,
        ):
            scene_target["minimum_word_count"] = minimum
            scene_target["maximum_word_count"] = maximum
        return {
            "target_total_word_count": target_total,
            "minimum_total_word_count": minimum_total,
            "maximum_total_word_count": maximum_total,
            "scene_word_targets": scene_targets,
        }

    def _scene_local_draft_stage(
        self,
        draft_input: dict[str, Any],
        outline: OutlineLike,
        research: ResearchPack | FactualResearchPack,
        *,
        invariant: Callable[[NarrationScript], None],
    ) -> NarrationScript:
        advisory_scene_words = self.workflow_policy_version >= 11
        aggregate_input = {
            "draft_strategy": "single-scene-v1",
            "scene_word_policy": (
                "advisory-with-host-aggregate-fit-v1"
                if advisory_scene_words
                else "strict-per-scene-v1"
            ),
            "draft": draft_input,
        }
        metadata = self._stage_metadata(
            stage="script-draft",
            task_id="script_draft",
            input_data=aggregate_input,
        )
        metadata["config_hash"] = hash_value(
            {
                "strategy": (
                    "single-scene-v2-host-aggregate-fit"
                    if advisory_scene_words
                    else "single-scene-v1"
                ),
                "backend_id": metadata["backend_id"],
                "backend_revision": metadata["backend_revision"],
                "language": self.config.output_language.value,
            }
        )
        metadata["schema_hash"] = hash_value(
            restricted_json_schema(ReplacementText.model_json_schema(mode="validation"))
        )
        reusable = self.store.reusable_record("script-draft", **metadata)
        if reusable:
            artifact = self.store.load_artifact(reusable, NarrationScript)
            invariant(artifact)
            return artifact

        attempt = self.store.next_attempt("script-draft")
        self.store.begin_stage("script-draft", attempt=attempt, **metadata)
        target_by_scene = {
            str(item["scene_id"]): item
            for item in draft_input["scene_word_targets"]
        }
        outline_context = outline.model_dump(mode="json", exclude={"scenes"})
        factual_evidence = (
            [
                {
                    "evidence_id": item.evidence_id,
                    "supported_statement": item.supported_statement,
                    "confidence": item.confidence,
                    "limitations": item.limitations,
                }
                for item in research.evidence
            ]
            if isinstance(research, FactualResearchPack)
            else []
        )
        delivery = getattr(self.config, "narration_delivery_spec", None)
        authored_pause = min(
            float(getattr(delivery, "target_pause_seconds", 0.15)),
            self._maximum_authored_pause_seconds(),
        )
        scenes: list[dict[str, Any]] = []
        usage: list[UsageRecord] = []
        for index, outline_scene in enumerate(outline.scenes):
            scene_target = target_by_scene[outline_scene.scene_id]
            minimum_words = int(scene_target["minimum_word_count"])
            target_words = int(scene_target["target_word_count"])
            maximum_words = int(scene_target["maximum_word_count"])
            preferred_evidence_ids = list(getattr(outline_scene, "evidence_ids", []))
            available_scene_evidence = (
                [
                    evidence
                    for evidence in factual_evidence
                    if evidence["evidence_id"] in preferred_evidence_ids
                ]
                if advisory_scene_words and isinstance(research, FactualResearchPack)
                else factual_evidence
            )
            item_input = {
                "draft_strategy": "single-scene-v1",
                "scene_word_policy": (
                    "advisory-with-host-aggregate-fit-v1"
                    if advisory_scene_words
                    else "strict-per-scene-v1"
                ),
                "outline_context": outline_context,
                "outline_scene": outline_scene.model_dump(mode="json"),
                "scene_position": index + 1,
                "scene_count": len(outline.scenes),
                "adjacent_context": {
                    "previous_spoken_text": (
                        str(scenes[-1]["spoken_text"]) if scenes else ""
                    ),
                    "next_outline_scene": (
                        outline.scenes[index + 1].model_dump(mode="json")
                        if index + 1 < len(outline.scenes)
                        else None
                    ),
                },
                "brief_constraints": {
                    "idea_direction": self.brief.idea_direction,
                    "tone": self.brief.tone,
                    "themes": self.brief.themes,
                    "must_include": self.brief.must_include,
                    "avoid": self.brief.avoid,
                    "audience": self.config.audience,
                },
                "available_factual_evidence": available_scene_evidence,
                "preferred_evidence_ids": preferred_evidence_ids,
                "output_language": self.config.output_language.value,
                "content_mode": self.config.content_mode.value,
                "content_format": self.config.content_format.value,
                "narration_delivery": self._delivery_payload(),
                "minimum_word_count": minimum_words,
                "target_word_count": target_words,
                "maximum_word_count": maximum_words,
                "minimum_sentence_count": int(scene_target["minimum_sentence_count"]),
                "maximum_sentence_count": int(scene_target["maximum_sentence_count"]),
                "count_method": "len(spoken_text.split())",
            }

            def validate_scene_text(
                replacement: ReplacementText,
                *,
                minimum: int = minimum_words,
                maximum: int = maximum_words,
            ) -> None:
                NarrationScript.model_validate(
                    {
                        "schema_version": 1,
                        "title": "Draft Scene",
                        "scenes": [
                            {
                                "scene_id": "scene-001",
                                "spoken_text": replacement.spoken_text,
                                "pause_after_seconds": 0,
                            }
                        ],
                    }
                )
                actual = len(replacement.spoken_text.split())
                if not advisory_scene_words and not minimum <= actual <= maximum:
                    target = minimum if actual < minimum else maximum
                    raise BackendError(
                        (
                            f"Draft Scene has {actual} words; required inclusive range "
                            f"is {minimum}-{maximum}"
                        ),
                        kind=ErrorKind.INVALID_OUTPUT,
                        details={
                            "actual_word_count": actual,
                            "minimum_word_count": minimum,
                            "maximum_word_count": maximum,
                            "target_word_count": target,
                            "word_delta": target - actual,
                            "count_method": "len(spoken_text.split())",
                        },
                    )

            replacement, item_usage = self._structured_item(
                stage="script-draft",
                item_id=f"draft-{outline_scene.scene_id}",
                task_id="script_draft",
                input_data=item_input,
                output_model=ReplacementText,
                invariant=validate_scene_text,
                max_output_tokens=800,
                instruction_suffix=((
                    (
                        "Return exactly one complete spoken_text for this Scene. Aim near "
                        f"{target_words} words, using {minimum_words}-{maximum_words} only as "
                        "planning guidance; Python reconciles the completed Script's aggregate "
                        "length. Prefer useful natural language over filler or mechanical padding. "
                    )
                    if advisory_scene_words
                    else (
                        "Return exactly one complete spoken_text for this Scene. "
                        f"It must contain {minimum_words}-{maximum_words} whitespace-separated "
                        "words inclusive. "
                    )
                ) + "Do not return any host-owned field."),
            )
            scenes.append(
                {
                    "scene_id": outline_scene.scene_id,
                    "spoken_text": replacement.spoken_text,
                    "pause_after_seconds": (
                        0 if index == len(outline.scenes) - 1 else round(authored_pause, 3)
                    ),
                }
            )
            usage.extend(item_usage)

        script = NarrationScript(title=outline.title, scenes=scenes)
        if advisory_scene_words:
            script, fit_usage = self._fit_scene_local_script_word_range(
                script=script,
                outline=outline,
                research=research,
                scene_word_targets=draft_input["scene_word_targets"],
                minimum_total=int(draft_input["minimum_total_word_count"]),
                target_total=int(draft_input["target_total_word_count"]),
                maximum_total=int(draft_input["maximum_total_word_count"]),
            )
            usage.extend(fit_usage)
        invariant(script)
        promoted = self.store.complete_fanout_stage(
            "script-draft",
            script,
            usage=usage,
        )
        return NarrationScript.model_validate(promoted)

    def _fit_scene_local_script_word_range(
        self,
        *,
        script: NarrationScript,
        outline: OutlineLike,
        research: ResearchPack | FactualResearchPack,
        scene_word_targets: list[dict[str, Any]],
        minimum_total: int,
        target_total: int,
        maximum_total: int,
        stage: str = "script-draft",
        task_id: str = "script_draft",
        strategy_field: str = "draft_strategy",
        item_prefix: str = "word-fit",
    ) -> tuple[NarrationScript, list[UsageRecord]]:
        def total_words(scenes: Sequence[Any]) -> int:
            return sum(len(scene.spoken_text.split()) for scene in scenes)

        working_scenes = [scene.model_copy(deep=True) for scene in script.scenes]
        total = total_words(working_scenes)
        if minimum_total <= total <= maximum_total:
            return script, []

        targets = {str(item["scene_id"]): item for item in scene_word_targets}
        outline_by_id = {scene.scene_id: scene for scene in outline.scenes}
        factual_evidence = (
            {
                evidence.evidence_id: {
                    "evidence_id": evidence.evidence_id,
                    "supported_statement": evidence.supported_statement,
                    "confidence": evidence.confidence,
                    "limitations": evidence.limitations,
                }
                for evidence in research.evidence
            }
            if isinstance(research, FactualResearchPack)
            else {}
        )
        usage: list[UsageRecord] = []
        fitted_scene_ids: set[str] = set()
        while not minimum_total <= total <= maximum_total:
            direction = "lengthen" if total < minimum_total else "shorten"
            candidates = [
                scene
                for scene in working_scenes
                if scene.scene_id not in fitted_scene_ids
                and (direction == "lengthen" or len(scene.spoken_text.split()) > 1)
            ]
            self._require(
                bool(candidates),
                "Host aggregate Script fitting exhausted all editable Scenes",
            )

            def rank(scene: Any) -> tuple[int, int, int]:
                actual = len(scene.spoken_text.split())
                target = int(targets[scene.scene_id]["target_word_count"])
                evidence_backed = int(
                    bool(getattr(outline_by_id[scene.scene_id], "evidence_ids", []))
                )
                if direction == "lengthen":
                    return evidence_backed, target - actual, actual
                return 0, actual - target, actual

            selected = max(candidates, key=rank)
            selected_index = next(
                index
                for index, scene in enumerate(working_scenes)
                if scene.scene_id == selected.scene_id
            )
            actual_words = len(selected.spoken_text.split())
            unchanged_total = total - actual_words
            if direction == "lengthen":
                minimum_words = max(actual_words + 1, minimum_total - unchanged_total)
                maximum_words = max(minimum_words, maximum_total - unchanged_total)
            else:
                feasible_maximum = maximum_total - unchanged_total
                if feasible_maximum < 1:
                    minimum_words = maximum_words = 1
                else:
                    minimum_words = max(1, minimum_total - unchanged_total)
                    maximum_words = min(actual_words - 1, feasible_maximum)
                    self._require(
                        minimum_words <= maximum_words,
                        "Host aggregate Script fitting calculated an empty replacement range",
                    )
            target_words = min(
                max(target_total - unchanged_total, minimum_words),
                maximum_words,
            )
            outline_scene = outline_by_id[selected.scene_id]
            preferred_evidence_ids = list(getattr(outline_scene, "evidence_ids", []))
            item_input = {
                strategy_field: "single-scene-word-fit-v1",
                "direction": direction,
                "spoken_text": selected.spoken_text,
                "adjacent_context": {
                    "previous_spoken_text": (
                        working_scenes[selected_index - 1].spoken_text
                        if selected_index > 0
                        else ""
                    ),
                    "next_spoken_text": (
                        working_scenes[selected_index + 1].spoken_text
                        if selected_index + 1 < len(working_scenes)
                        else ""
                    ),
                },
                "outline_scene": outline_scene.model_dump(mode="json"),
                "available_factual_evidence": [
                    factual_evidence[evidence_id]
                    for evidence_id in preferred_evidence_ids
                    if evidence_id in factual_evidence
                ],
                "output_language": self.config.output_language.value,
                "content_mode": self.config.content_mode.value,
                "minimum_word_count": minimum_words,
                "target_word_count": target_words,
                "maximum_word_count": maximum_words,
                "aggregate_word_counts": {
                    "current": total,
                    "minimum": minimum_total,
                    "target": target_total,
                    "maximum": maximum_total,
                },
                "count_method": "len(spoken_text.split())",
            }

            def validate_fit(
                replacement: ReplacementText,
                *,
                minimum: int = minimum_words,
                maximum: int = maximum_words,
            ) -> None:
                NarrationScript.model_validate(
                    {
                        "schema_version": 1,
                        "title": "Script word fit",
                        "scenes": [
                            {
                                "scene_id": "scene-001",
                                "spoken_text": replacement.spoken_text,
                                "pause_after_seconds": 0,
                            }
                        ],
                    }
                )
                actual = len(replacement.spoken_text.split())
                if not minimum <= actual <= maximum:
                    boundary = minimum if actual < minimum else maximum
                    raise BackendError(
                        (
                            f"Script word fit has {actual} words; required inclusive range "
                            f"is {minimum}-{maximum}"
                        ),
                        kind=ErrorKind.INVALID_OUTPUT,
                        details={
                            "actual_word_count": actual,
                            "minimum_word_count": minimum,
                            "maximum_word_count": maximum,
                            "target_word_count": boundary,
                            "word_delta": boundary - actual,
                            "count_method": "len(spoken_text.split())",
                        },
                    )

            editing_instruction = (
                "Expand with useful concrete explanation or a natural connective already supported "
                "by the supplied Scene context. In factual mode, any added assertion must be directly "
                "supported by available_factual_evidence. Do not pad with filler."
                if direction == "lengthen"
                else "Use a deletion-first edit that preserves the Scene's essential meaning and facts."
            )
            replacement, item_usage = self._structured_item(
                stage=stage,
                item_id=f"{item_prefix}-{selected.scene_id}",
                task_id=task_id,
                input_data=item_input,
                output_model=ReplacementText,
                invariant=validate_fit,
                max_output_tokens=800,
                instruction_suffix=(
                    "Return only one complete replacement spoken_text. Python selected this Scene and "
                    "computed a feasible aggregate residual range; do not return an ID, title, pause, "
                    f"word count, or explanation. Use {minimum_words}-{maximum_words} words inclusive, "
                    f"aiming near {target_words}. {editing_instruction}"
                ),
            )
            working_scenes[selected_index] = selected.model_copy(
                update={"spoken_text": replacement.spoken_text}
            )
            fitted_scene_ids.add(selected.scene_id)
            usage.extend(item_usage)
            total = total_words(working_scenes)

        return NarrationScript(title=script.title, scenes=working_scenes), usage

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
            target_words = minimum_words if actual < minimum_words else maximum_words
            raise BackendError(
                (
                    f"Narration Script has {actual} words; required inclusive range is "
                    f"{minimum_words}-{maximum_words}"
                ),
                kind=ErrorKind.INVALID_OUTPUT,
                details={
                    "actual_word_count": actual,
                    "minimum_word_count": minimum_words,
                    "maximum_word_count": maximum_words,
                    "target_word_count": target_words,
                    "word_delta": target_words - actual,
                    "count_method": "whitespace-separated words across every spoken_text field",
                },
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
            f"outline must contain {minimum_count}-{maximum_count} Scenes for the configured editorial cadence",
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
            for index, scene in enumerate(visual_plan.scenes)
            if (index > 0 and not scene.continuity_from_previous)
            or not scene.state_after_scene
            or (scene.character_ids and not scene.identity_requirements)
        ]
        WorkflowEngine._require(
            not incomplete_scenes,
            "Visual Briefs need post-opening incoming state, resulting state, and character identity "
            "locks: "
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

    def _split_visual_plan_stage(
        self,
        visual_plan_input: dict[str, Any],
        *,
        outline: OutlineLike,
        script: NarrationScript,
        schedule: list[dict[str, Any]],
        timed_visuals: bool,
        output_model: type[VisualPlanLike],
        invariant: Callable[[Any], None],
    ) -> VisualPlanLike:
        aggregate_input = {
            "visual_strategy": "foundation-and-items-v1",
            "visual_plan": visual_plan_input,
        }
        metadata = self._stage_metadata(
            stage="visual-plan",
            task_id="visual_plan",
            input_data=aggregate_input,
        )
        metadata["config_hash"] = hash_value(
            {
                "strategy": "foundation-and-items-v1",
                "backend_id": metadata["backend_id"],
                "backend_revision": metadata["backend_revision"],
                "language": self.prompts.output_language(
                    "visual_plan", self.config.output_language
                ).value,
                "style_id": self.config.style,
                "timed_visuals": timed_visuals,
            }
        )
        metadata["schema_hash"] = hash_value(
            {
                "foundation": restricted_json_schema(
                    VisualFoundation.model_json_schema(mode="validation")
                ),
                "visual_content": restricted_json_schema(
                    VisualBriefContent.model_json_schema(mode="validation")
                ),
                "aggregate": restricted_json_schema(
                    output_model.model_json_schema(mode="validation")
                ),
            }
        )
        reusable = self.store.reusable_record("visual-plan", **metadata)
        if reusable:
            artifact = self.store.load_artifact(reusable, output_model)
            invariant(artifact)
            return artifact

        attempt = self.store.next_attempt("visual-plan")
        self.store.begin_stage("visual-plan", attempt=attempt, **metadata)
        foundation_input = {
            "visual_strategy": "foundation-v1",
            "brief": self.brief.model_dump(mode="json"),
            "outline": outline.model_dump(mode="json"),
            "script": script.model_dump(mode="json"),
            "style_id": self.config.style,
            "style_description": self.config.style_description,
            "audience": self.config.audience,
            "content_mode": self.config.content_mode.value,
            "content_format": self.config.content_format.value,
            "delivery": visual_plan_input["delivery"],
            "visual_count": len(schedule) if timed_visuals else len(script.scenes),
        }

        def validate_foundation(value: VisualFoundation) -> None:
            names = [character.name.casefold().strip() for character in value.characters]
            self._require(
                len(names) == len(set(names)),
                "Visual Foundation contains duplicate Character names",
            )
            if self.config.content_format is ContentFormat.NARRATIVE:
                self._require(
                    bool(value.characters),
                    "Narrative Visual Foundation requires at least one recurring Character",
                )

        foundation, foundation_usage = self._structured_item(
            stage="visual-plan",
            item_id="foundation",
            task_id="visual_plan",
            input_data=foundation_input,
            output_model=VisualFoundation,
            invariant=validate_foundation,
            max_output_tokens=2400,
            instruction_suffix=(
                "Return only the reusable style content and complete recurring Character "
                "identity content requested by the schema. Do not return style_id, Character "
                "IDs, Shot IDs, Scene IDs, timing, narration excerpts, or visual items."
            ),
        )
        style_data = foundation.style.model_dump(mode="json")
        style_data["style_id"] = self.config.style
        style_data["must_avoid"] = list(
            dict.fromkeys(
                [
                    *style_data["must_avoid"],
                    "written text",
                    "captions",
                    "labels",
                    "logos",
                    "watermarks",
                ]
            )
        )
        characters = [
            {
                "character_id": f"character-{index:03d}",
                **character.model_dump(mode="json"),
            }
            for index, character in enumerate(foundation.characters, start=1)
        ]
        known_character_ids = {item["character_id"] for item in characters}
        outline_by_id = {scene.scene_id: scene for scene in outline.scenes}
        script_by_id = {scene.scene_id: scene for scene in script.scenes}
        targets = (
            schedule
            if timed_visuals
            else [
                {
                    "scene_id": scene.scene_id,
                    "narration_excerpt": scene.spoken_text,
                }
                for scene in script.scenes
            ]
        )
        visual_items: list[dict[str, Any]] = []
        item_usage = list(foundation_usage)
        for index, target in enumerate(targets):
            scene_id = str(target["scene_id"])
            visual_id = str(target.get("shot_id") or scene_id)
            next_target = targets[index + 1] if index + 1 < len(targets) else None
            item_input = {
                "visual_strategy": "single-visual-v1",
                "visual_target": target,
                "parent_outline_scene": outline_by_id[scene_id].model_dump(mode="json"),
                "parent_scene_spoken_text": script_by_id[scene_id].spoken_text,
                "previous_visual": visual_items[-1] if visual_items else None,
                "next_visual_target": next_target,
                "style_profile": style_data,
                "character_identities": characters,
                "brief_constraints": {
                    "tone": self.brief.tone,
                    "must_include": self.brief.must_include,
                    "avoid": self.brief.avoid,
                    "audience": self.config.audience,
                },
                "content_mode": self.config.content_mode.value,
                "content_format": self.config.content_format.value,
                "delivery": visual_plan_input["delivery"],
                "rule": (
                    "Depict only visual_target.narration_excerpt. Adjacent inputs are read-only "
                    "continuity context and must not leak future actions into this image."
                ),
            }

            def validate_visual_content(
                content: VisualBriefContent,
                *,
                item_index: int = index,
                current_visual_id: str = visual_id,
            ) -> None:
                unknown = sorted(set(content.character_ids) - known_character_ids)
                self._require(
                    not unknown,
                    f"{current_visual_id} references unknown Character IDs: "
                    + ", ".join(unknown),
                )
                self._require(
                    not content.character_ids or bool(content.identity_requirements),
                    f"{current_visual_id} uses a Character without identity requirements",
                )
                if (
                    not timed_visuals
                    and self.config.content_format is ContentFormat.NARRATIVE
                ):
                    self._require(
                        bool(content.state_after_scene),
                        f"{current_visual_id} requires a resulting visible state",
                    )
                    if item_index > 0:
                        self._require(
                            bool(content.continuity_from_previous),
                            f"{current_visual_id} requires incoming continuity state",
                        )

            content, usage = self._structured_item(
                stage="visual-plan",
                item_id=f"content-{visual_id}",
                task_id="visual_plan",
                input_data=item_input,
                output_model=VisualBriefContent,
                invariant=validate_visual_content,
                max_output_tokens=1800,
                instruction_suffix=(
                    "Return only the visual content fields requested by the schema for this one "
                    "image. Do not return a Shot ID, Scene ID, narration excerpt, timestamp, "
                    "duration, Style Profile, Character definition, or surrounding visual item."
                ),
            )
            content_data = content.model_dump(mode="json")
            if timed_visuals:
                visual_items.append({**content_data, **target})
            else:
                visual_items.append({"scene_id": scene_id, **content_data})
            item_usage.extend(usage)

        if timed_visuals:
            plan_data = {
                "schema_version": 1,
                "style_profile": style_data,
                "characters": characters,
                "duration_seconds": float(schedule[-1]["end_seconds"]),
                "shots": visual_items,
            }
        else:
            plan_data = {
                "schema_version": 1,
                "style_profile": style_data,
                "characters": characters,
                "scenes": visual_items,
            }
        plan = output_model.model_validate(plan_data)
        invariant(plan)
        promoted = self.store.complete_fanout_stage(
            "visual-plan",
            plan,
            usage=item_usage,
        )
        return output_model.model_validate(promoted)

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
        max_output_tokens: int = 8000,
        instruction_suffix: str = "",
        target_image_backend: str | None = None,
    ) -> tuple[Any, list[UsageRecord]]:
        metadata = self._stage_metadata(
            stage=stage,
            task_id=task_id,
            input_data=input_data,
            target_image_backend=target_image_backend,
        )
        metadata["schema_hash"] = hash_value(
            restricted_json_schema(output_model.model_json_schema(mode="validation"))
        )
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
            max_output_tokens=max_output_tokens,
            invariant=invariant,
            instruction_suffix=instruction_suffix,
            target_image_backend=target_image_backend,
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
        raise_on_failure: bool = True,
        prior_inventory: ClaimInventory | None = None,
        prior_review: FactualReviewReport | None = None,
        changed_scene_ids: set[str] | None = None,
    ) -> tuple[ClaimInventory, FactualReviewReport, list[UsageRecord]]:
        if self.workflow_policy_version >= 13:
            return self._factual_audit_by_scene(
                stage=stage,
                item_prefix=item_prefix,
                script=script,
                research=research,
                raise_on_failure=raise_on_failure,
                prior_inventory=prior_inventory,
                prior_review=prior_review,
                changed_scene_ids=changed_scene_ids,
            )
        if self.workflow_policy_version >= 4:
            return self._factual_audit_by_claim(
                stage=stage,
                item_prefix=item_prefix,
                script=script,
                research=research,
                raise_on_failure=raise_on_failure,
            )
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
        if not review.passed and raise_on_failure:
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

    def _factual_audit_by_scene(
        self,
        *,
        stage: str,
        item_prefix: str,
        script: NarrationScript,
        research: FactualResearchPack,
        raise_on_failure: bool,
        prior_inventory: ClaimInventory | None,
        prior_review: FactualReviewReport | None,
        changed_scene_ids: set[str] | None,
    ) -> tuple[ClaimInventory, FactualReviewReport, list[UsageRecord]]:
        evidence_by_id = {item.evidence_id: item for item in research.evidence}
        script_by_id = {scene.scene_id: scene for scene in script.scenes}
        self._require(
            (prior_inventory is None) == (prior_review is None),
            "Prior factual inventory and review must be supplied together",
        )
        if prior_inventory is not None:
            self._require(
                changed_scene_ids is not None,
                "A partial factual re-audit requires explicit changed Scene IDs",
            )
        changed = set(script_by_id) if prior_inventory is None else set(changed_scene_ids or set())
        unknown_changed = sorted(changed - set(script_by_id))
        self._require(
            not unknown_changed,
            "Factual re-audit references unknown Scenes: " + ", ".join(unknown_changed),
        )

        prior_claims_by_scene: dict[str, list[ScriptClaim]] = {}
        prior_decisions: dict[str, FactualClaimReview] = {}
        if prior_inventory is not None and prior_review is not None:
            prior_decisions = {item.claim_id: item for item in prior_review.claims}
            self._require(
                set(prior_decisions) == {item.claim_id for item in prior_inventory.claims},
                "Prior factual review must cover every prior Claim exactly once",
            )
            for claim in prior_inventory.claims:
                scene = script_by_id.get(claim.scene_id)
                self._require(
                    scene is not None,
                    f"Prior Claim {claim.claim_id} references an unknown Scene",
                )
                if claim.scene_id not in changed:
                    self._require(
                        claim.exact_text in scene.spoken_text,
                        f"Unchanged Claim {claim.claim_id} no longer preserves exact Script wording",
                    )
                    unknown = sorted(set(claim.evidence_ids) - set(evidence_by_id))
                    self._require(
                        not unknown,
                        f"Prior Claim {claim.claim_id} references unknown Evidence IDs: "
                        + ", ".join(unknown),
                    )
                    decision = prior_decisions[claim.claim_id]
                    unsupported_citations = sorted(
                        set(decision.evidence_ids) - set(claim.evidence_ids)
                    )
                    self._require(
                        not unsupported_citations,
                        f"Prior review for {claim.claim_id} cites Evidence not linked by its Claim: "
                        + ", ".join(unsupported_citations),
                    )
                    if decision.verdict == "supported":
                        self._require(
                            bool(decision.evidence_ids),
                            f"Prior supported Claim {claim.claim_id} has no direct Evidence IDs",
                        )
                    if decision.verdict == "not_a_factual_claim":
                        self._require(
                            not decision.evidence_ids,
                            f"Prior non-factual Claim {claim.claim_id} cites Evidence IDs",
                        )
                    prior_claims_by_scene.setdefault(claim.scene_id, []).append(claim)

        claims_with_prior: list[tuple[ScriptClaim, FactualClaimReview | None]] = []
        usage: list[UsageRecord] = []
        for scene in script.scenes:
            if prior_inventory is not None and scene.scene_id not in changed:
                claims_with_prior.extend(
                    (claim, prior_decisions[claim.claim_id])
                    for claim in prior_claims_by_scene.get(scene.scene_id, [])
                )
                continue

            inventory_input = {
                "inventory_strategy": "single-scene-claim-extraction-v2",
                "spoken_text": scene.spoken_text,
                "research_pack": {
                    "evidence": [item.model_dump(mode="json") for item in research.evidence]
                },
                "output_language": self.config.output_language.value,
                "content_format": self.config.content_format.value,
            }

            def validate_extraction(extraction: SceneClaimExtraction) -> None:
                exact_texts: set[str] = set()
                for claim in extraction.claims:
                    self._require(
                        claim.exact_text in scene.spoken_text,
                        "Extracted Claim does not preserve exact Scene wording",
                    )
                    self._require(
                        claim.exact_text not in exact_texts,
                        "Scene Claim extraction contains duplicate exact wording",
                    )
                    exact_texts.add(claim.exact_text)
                    unknown = sorted(set(claim.evidence_ids) - set(evidence_by_id))
                    self._require(
                        not unknown,
                        "Extracted Claim references unknown Evidence IDs: "
                        + ", ".join(unknown),
                    )

            extraction, extraction_usage = self._structured_item(
                stage=stage,
                item_id=f"{item_prefix}-claim-inventory-{scene.scene_id}",
                task_id="claim_inventory",
                input_data=inventory_input,
                output_model=SceneClaimExtraction,
                invariant=validate_extraction,
                max_output_tokens=1200,
                instruction_suffix=(
                    "Extract Claims only from the supplied spoken_text. Return only claims with "
                    "exact_text, evidence_ids, and qualification. An empty claims list is valid. "
                    "Do not return a Scene ID, Claim ID, coverage note, or narration edit."
                ),
            )
            usage.extend(extraction_usage)
            claims_with_prior.extend(
                (
                    ScriptClaim(
                        claim_id=f"{scene.scene_id}-claim-{index:03d}",
                        scene_id=scene.scene_id,
                        exact_text=claim.exact_text,
                        evidence_ids=claim.evidence_ids,
                        qualification=claim.qualification,
                    ),
                    None,
                )
                for index, claim in enumerate(extraction.claims, start=1)
            )

        claims = [claim for claim, _ in claims_with_prior]
        self._require(
            bool(claims),
            "Factual Script contains no externally verifiable Claims",
        )
        inventory = ClaimInventory(
            claims=claims,
            coverage_notes=(
                "Claims were extracted Scene by Scene; Python assigned stable Claim and Scene IDs."
            ),
        )
        claim_reviews: list[FactualClaimReview] = []
        for claim, prior_decision in claims_with_prior:
            if prior_decision is not None:
                claim_reviews.append(prior_decision.model_copy(update={"claim_id": claim.claim_id}))
                continue
            allowed_evidence_ids = set(claim.evidence_ids)
            allowed_evidence = [
                evidence_by_id[evidence_id].model_dump(mode="json")
                for evidence_id in claim.evidence_ids
            ]
            review_input = {
                "review_strategy": "single-claim-v1",
                "claim": claim.model_dump(mode="json"),
                "scene_spoken_text": script_by_id[claim.scene_id].spoken_text,
                "evidence_records": allowed_evidence,
                "research_pack": {"evidence": allowed_evidence},
                "output_language": self.config.output_language.value,
            }

            def validate_decision(decision: ClaimReviewDecision) -> None:
                unknown = sorted(set(decision.evidence_ids) - allowed_evidence_ids)
                self._require(
                    not unknown,
                    "Factual Claim Review cites Evidence not linked by extraction: "
                    + ", ".join(unknown),
                )
                if decision.verdict == "supported":
                    self._require(
                        bool(decision.evidence_ids),
                        "A supported Claim requires at least one direct Evidence ID",
                    )
                if decision.verdict == "not_a_factual_claim":
                    self._require(
                        not decision.evidence_ids,
                        "A non-factual Claim classification must not cite Evidence IDs",
                    )

            decision, decision_usage = self._structured_item(
                stage=stage,
                item_id=f"{item_prefix}-factual-review-{claim.claim_id}",
                task_id="factual_review",
                input_data=review_input,
                output_model=ClaimReviewDecision,
                invariant=validate_decision,
                max_output_tokens=800,
                instruction_suffix=(
                    "Review exactly the one supplied Claim against only evidence_records. Return "
                    "only verdict, evidence_ids, and rationale. Use not_a_factual_claim only for "
                    "text with no independently asserted factual proposition."
                ),
            )
            usage.extend(decision_usage)
            claim_reviews.append(
                FactualClaimReview(
                    claim_id=claim.claim_id,
                    verdict=decision.verdict,
                    evidence_ids=decision.evidence_ids,
                    rationale=decision.rationale,
                )
            )

        accepted_verdicts = {"supported", "not_a_factual_claim"}
        passed = all(item.verdict in accepted_verdicts for item in claim_reviews)
        review = FactualReviewReport(
            passed=passed,
            claims=claim_reviews,
            uncovered_claims=[],
            summary=(
                "Every extracted Claim passed its bounded individual review."
                if passed
                else "One or more extracted Claims require revision or evidence."
            ),
        )
        if not passed and raise_on_failure:
            failed_claims = [
                item.claim_id for item in claim_reviews if item.verdict not in accepted_verdicts
            ]
            raise BackendError(
                "factual accuracy gate blocked narration: " + ", ".join(failed_claims),
                kind=ErrorKind.INVALID_OUTPUT,
                action="Revise the Script or improve bounded research evidence before narration.",
            )
        return inventory, review, usage

    def _factual_audit_by_claim(
        self,
        *,
        stage: str,
        item_prefix: str,
        script: NarrationScript,
        research: FactualResearchPack,
        raise_on_failure: bool = True,
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
            "inventory_strategy": "script-claim-extraction-v1",
        }
        inventory, inventory_usage = self._structured_item(
            stage=stage,
            item_id=f"{item_prefix}-claim-inventory",
            task_id="claim_inventory",
            input_data=inventory_input,
            output_model=ClaimInventory,
            invariant=validate_inventory,
            max_output_tokens=2400,
        )
        usage = list(inventory_usage)
        claim_reviews: list[FactualClaimReview] = []
        for claim in inventory.claims:
            review_input = {
                "review_strategy": "single-claim-v1",
                "claim": claim.model_dump(mode="json"),
                "scene_spoken_text": script_by_id[claim.scene_id].spoken_text,
                "evidence_records": [
                    item.model_dump(mode="json") for item in research.evidence
                ],
                "research_pack": {
                    "evidence": [
                        item.model_dump(mode="json") for item in research.evidence
                    ]
                },
                "output_language": self.config.output_language.value,
            }

            def validate_decision(decision: ClaimReviewDecision) -> None:
                unknown = sorted(set(decision.evidence_ids) - evidence_ids)
                self._require(
                    not unknown,
                    "Factual Claim Review references unknown Evidence IDs: "
                    + ", ".join(unknown),
                )
                if decision.verdict == "supported":
                    self._require(
                        bool(decision.evidence_ids),
                        "A supported Claim requires at least one direct Evidence ID",
                    )
                if decision.verdict == "not_a_factual_claim":
                    self._require(
                        not decision.evidence_ids,
                        "A non-factual Claim classification must not cite Evidence IDs",
                    )

            decision, decision_usage = self._structured_item(
                stage=stage,
                item_id=f"{item_prefix}-factual-review-{claim.claim_id}",
                task_id="factual_review",
                input_data=review_input,
                output_model=ClaimReviewDecision,
                invariant=validate_decision,
                max_output_tokens=800,
                instruction_suffix=(
                    "Review exactly the one supplied Claim. Return only verdict, evidence_ids, "
                    "and rationale. Use not_a_factual_claim only for text with no independently "
                    "asserted factual proposition."
                ),
            )
            usage.extend(decision_usage)
            claim_reviews.append(
                FactualClaimReview(
                    claim_id=claim.claim_id,
                    verdict=decision.verdict,
                    evidence_ids=decision.evidence_ids,
                    rationale=decision.rationale,
                )
            )
        accepted_verdicts = {"supported", "not_a_factual_claim"}
        passed = all(item.verdict in accepted_verdicts for item in claim_reviews)
        review = FactualReviewReport(
            passed=passed,
            claims=claim_reviews,
            uncovered_claims=[],
            summary=(
                "Every extracted Claim passed its bounded individual review."
                if passed
                else "One or more extracted Claims require revision or evidence."
            ),
        )
        if not review.passed and raise_on_failure:
            failed_claims = [
                claim.claim_id
                for claim in review.claims
                if claim.verdict not in accepted_verdicts
            ]
            raise BackendError(
                "factual accuracy gate blocked narration: " + ", ".join(failed_claims),
                kind=ErrorKind.INVALID_OUTPUT,
                action="Revise the Script or improve bounded research evidence before narration.",
            )
        return inventory, review, usage

    def _repair_factual_script_once(
        self,
        *,
        stage: str,
        item_prefix: str,
        script: NarrationScript,
        research: FactualResearchPack,
        inventory: ClaimInventory,
        review: FactualReviewReport,
    ) -> tuple[NarrationScript, list[UsageRecord]]:
        accepted_verdicts = {"supported", "not_a_factual_claim"}
        claims_by_id = {claim.claim_id: claim for claim in inventory.claims}
        failures_by_scene: dict[str, list[tuple[Any, FactualClaimReview]]] = {}
        for decision in review.claims:
            if decision.verdict in accepted_verdicts:
                continue
            claim = claims_by_id.get(decision.claim_id)
            self._require(
                claim is not None,
                f"Factual Review references unknown Claim {decision.claim_id}",
            )
            failures_by_scene.setdefault(claim.scene_id, []).append((claim, decision))

        self._require(
            bool(failures_by_scene),
            "Factual repair requires at least one failed Claim",
        )
        working_scenes = [scene.model_copy(deep=True) for scene in script.scenes]
        usage: list[UsageRecord] = []
        for scene in working_scenes:
            failures = failures_by_scene.get(scene.scene_id)
            if not failures:
                continue
            allowed_evidence_ids = {
                evidence_id
                for claim, decision in failures
                for evidence_id in [*claim.evidence_ids, *decision.evidence_ids]
            }
            item_input = {
                "repair_strategy": "factual-claim-repair-v1",
                "spoken_text": scene.spoken_text,
                "failed_claims": [
                    {
                        "exact_text": claim.exact_text,
                        "qualification": claim.qualification,
                        "verdict": decision.verdict,
                        "rationale": decision.rationale,
                        "evidence_ids": decision.evidence_ids,
                    }
                    for claim, decision in failures
                ],
                "allowed_factual_evidence": [
                    evidence.model_dump(mode="json")
                    for evidence in research.evidence
                    if evidence.evidence_id in allowed_evidence_ids
                ],
                "output_language": self.config.output_language.value,
                "content_format": self.config.content_format.value,
            }

            def validate_repair(replacement: ReplacementText) -> None:
                NarrationScript.model_validate(
                    {
                        "schema_version": 1,
                        "title": "Factual scene repair",
                        "scenes": [
                            {
                                "scene_id": "scene-001",
                                "spoken_text": replacement.spoken_text,
                                "pause_after_seconds": 0,
                            }
                        ],
                    }
                )

            replacement, repair_usage = self._structured_item(
                stage=stage,
                item_id=f"{item_prefix}-{scene.scene_id}",
                task_id="script_revision",
                input_data=item_input,
                output_model=ReplacementText,
                invariant=validate_repair,
                max_output_tokens=800,
                instruction_suffix=(
                    "Rewrite only the supplied Scene text so every factual assertion is directly "
                    "supported by allowed_factual_evidence. Remove or narrow unsupported wording; "
                    "do not invent a bridge claim. When no evidence is supplied, remove the factual "
                    "assertion. Preserve the Scene's intent and output language. "
                    "Keep the correction concise and naturally close to the original pacing; Python "
                    "will reconcile the completed Script's aggregate length. Return only the single "
                    "spoken_text field and no host-owned field."
                ),
            )
            scene.spoken_text = replacement.spoken_text
            usage.extend(repair_usage)

        return NarrationScript(title=script.title, scenes=working_scenes), usage

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

    def _scene_local_revision_stage(
        self,
        revision_input: dict[str, Any],
        reviews: Sequence[ReviewReport],
        outline: OutlineLike,
        research: ResearchPack | FactualResearchPack,
        *,
        invariant: Callable[[RevisedScript], None],
    ) -> RevisedScript | FactualRevisedScript:
        draft = NarrationScript.model_validate(revision_input["script"])
        aggregate_input = {
            "revision": revision_input,
            "revision_strategy": "single-scene-replacement-v1",
        }
        metadata = self._stage_metadata(
            stage="script-revision",
            task_id="script_revision",
            input_data=aggregate_input,
        )
        task_ids = ["script_revision"]
        if isinstance(research, FactualResearchPack):
            task_ids.extend(["claim_inventory", "factual_review"])
        metadata["config_hash"] = hash_value(
            {
                "strategy": "single-scene-replacement-v1",
                "factual_repair_strategy": (
                    "factual-claim-repair-v2-partial-scene-reaudit"
                    if isinstance(research, FactualResearchPack)
                    and self.workflow_policy_version >= 13
                    else "factual-claim-repair-v1"
                    if isinstance(research, FactualResearchPack)
                    else None
                ),
                "factual_aggregate_fit_strategy": (
                    "single-scene-word-fit-v1"
                    if isinstance(research, FactualResearchPack)
                    and self.workflow_policy_version >= 11
                    else None
                ),
                "claim_inventory_strategy": (
                    "single-scene-claim-extraction-v2"
                    if isinstance(research, FactualResearchPack)
                    and self.workflow_policy_version >= 13
                    else None
                ),
                "tasks": {
                    task_id: {
                        "backend_id": self.config.task_bindings[task_id],
                        "backend_revision": self.registry.descriptor(
                            self.config.task_bindings[task_id]
                        ).revision,
                    }
                    for task_id in task_ids
                },
            }
        )
        metadata["prompt_version"] = str(
            self.store.frozen_assets.get("prompt_set_version", "")
        )
        schema_payload: dict[str, Any] = {
            "scene_replacement": restricted_json_schema(
                ReplacementText.model_json_schema(mode="validation")
            )
        }
        if isinstance(research, FactualResearchPack):
            schema_payload.update(
                {
                    "factual_scene_repair": restricted_json_schema(
                        ReplacementText.model_json_schema(mode="validation")
                    ),
                    "claim_inventory": (
                        restricted_json_schema(
                            SceneClaimExtraction.model_json_schema(mode="validation")
                        )
                        if self.workflow_policy_version >= 13
                        else self.prompts.schema("claim_inventory")
                    ),
                    "claim_review_decision": restricted_json_schema(
                        ClaimReviewDecision.model_json_schema(mode="validation")
                    ),
                }
            )
        metadata["schema_hash"] = hash_value(schema_payload)
        artifact_model: type[RevisedScript | FactualRevisedScript] = (
            FactualRevisedScript
            if isinstance(research, FactualResearchPack)
            else RevisedScript
        )
        reusable = self.store.reusable_record("script-revision", **metadata)
        if reusable:
            artifact = self.store.load_artifact(reusable, artifact_model)
            invariant(artifact)
            if isinstance(artifact, FactualRevisedScript) and not artifact.factual_review.passed:
                raise BackendError(
                    "cached factual accuracy gate did not pass",
                    kind=ErrorKind.INVALID_OUTPUT,
                )
            return artifact

        attempt = self.store.next_attempt("script-revision")
        self.store.begin_stage("script-revision", attempt=attempt, **metadata)
        findings_by_scene: dict[str, list[Any]] = {}
        for review in reviews:
            for finding in review.findings:
                if finding.scene_id is None:
                    raise BackendError(
                        f"Review Finding {finding.finding_id} has no Scene ID",
                        kind=ErrorKind.INVALID_OUTPUT,
                    )
                findings_by_scene.setdefault(finding.scene_id, []).append(finding)

        minimum_total = int(revision_input["minimum_total_word_count"])
        maximum_total = int(revision_input["maximum_total_word_count"])
        outline_by_id = {scene.scene_id: scene for scene in outline.scenes}
        working_scenes = [scene.model_copy(deep=True) for scene in draft.scenes]
        dispositions: list[RevisionDisposition] = []
        item_usage: list[UsageRecord] = []
        for index, scene in enumerate(working_scenes):
            findings = findings_by_scene.get(scene.scene_id, [])
            if not findings:
                continue
            current_total = sum(len(item.spoken_text.split()) for item in working_scenes)
            original_count = len(scene.spoken_text.split())
            unchanged_total = current_total - original_count
            minimum_words = max(1, minimum_total - unchanged_total)
            maximum_words = max(minimum_words, maximum_total - unchanged_total)
            maximum_words = min(maximum_words, max(minimum_words, original_count + 8))
            target_words = min(max(original_count, minimum_words), maximum_words)
            outline_scene = outline_by_id[scene.scene_id]
            scene_evidence_ids = set(getattr(outline_scene, "evidence_ids", []))
            allowed_evidence = []
            if isinstance(research, FactualResearchPack):
                allowed_evidence = [
                    evidence.model_dump(mode="json")
                    for evidence in research.evidence
                    if evidence.evidence_id in scene_evidence_ids
                ]
            item_input = {
                "revision_strategy": "single-scene-replacement-v1",
                "spoken_text": scene.spoken_text,
                "adjacent_context": {
                    "previous_spoken_text": (
                        working_scenes[index - 1].spoken_text if index > 0 else ""
                    ),
                    "next_spoken_text": (
                        working_scenes[index + 1].spoken_text
                        if index + 1 < len(working_scenes)
                        else ""
                    ),
                },
                "outline_scene": outline_scene.model_dump(mode="json"),
                "findings": [
                    {
                        "severity": finding.severity,
                        "evidence": finding.evidence,
                        "recommendation": finding.recommendation,
                    }
                    for finding in findings
                ],
                "brief_constraints": {
                    "tone": self.brief.tone,
                    "must_include": self.brief.must_include,
                    "avoid": self.brief.avoid,
                },
                "allowed_factual_evidence": allowed_evidence,
                "output_language": self.config.output_language.value,
                "content_mode": self.config.content_mode.value,
                "content_format": self.config.content_format.value,
                "minimum_word_count": minimum_words,
                "target_word_count": target_words,
                "maximum_word_count": maximum_words,
                "count_method": "len(spoken_text.split())",
            }

            def validate_replacement(
                replacement: ReplacementText,
                *,
                minimum: int = minimum_words,
                maximum: int = maximum_words,
            ) -> None:
                NarrationScript.model_validate(
                    {
                        "schema_version": 1,
                        "title": "Scene replacement",
                        "scenes": [
                            {
                                "scene_id": "scene-001",
                                "spoken_text": replacement.spoken_text,
                                "pause_after_seconds": 0,
                            }
                        ],
                    }
                )
                actual = len(replacement.spoken_text.split())
                if not minimum <= actual <= maximum:
                    target = minimum if actual < minimum else maximum
                    raise BackendError(
                        (
                            f"Scene replacement has {actual} words; required inclusive range "
                            f"is {minimum}-{maximum}"
                        ),
                        kind=ErrorKind.INVALID_OUTPUT,
                        details={
                            "actual_word_count": actual,
                            "minimum_word_count": minimum,
                            "maximum_word_count": maximum,
                            "target_word_count": target,
                            "word_delta": target - actual,
                            "count_method": "len(spoken_text.split())",
                        },
                    )

            original_text = scene.spoken_text
            replacement, usage = self._structured_item(
                stage="script-revision",
                item_id=f"replacement-{scene.scene_id}",
                task_id="script_revision",
                input_data=item_input,
                output_model=ReplacementText,
                invariant=validate_replacement,
                max_output_tokens=1200,
                instruction_suffix=(
                    "Return exactly one complete replacement spoken_text for this Scene. "
                    f"It must contain {minimum_words}-{maximum_words} whitespace-separated "
                    "words inclusive. Do not return any host-owned field."
                ),
            )
            scene.spoken_text = replacement.spoken_text
            item_usage.extend(usage)
            changed = replacement.spoken_text.strip() != original_text.strip()
            dispositions.extend(
                RevisionDisposition(
                    finding_id=finding.finding_id,
                    disposition="applied" if changed else "rejected",
                    explanation=(
                        "Applied through a bounded Scene text replacement."
                        if changed
                        else "The bounded replacement returned no textual change."
                    ),
                )
                for finding in findings
            )

        revision = RevisedScript(
            script=NarrationScript(title=draft.title, scenes=working_scenes),
            dispositions=dispositions,
        )
        invariant(revision)
        if isinstance(research, FactualResearchPack):
            inventory, review, audit_usage = self._factual_audit(
                stage="script-revision",
                item_prefix="approved-script",
                script=revision.script,
                research=research,
                raise_on_failure=self.workflow_policy_version < 9,
            )
            item_usage.extend(audit_usage)
            if not review.passed:
                audited_script = revision.script
                repaired_script, repair_usage = self._repair_factual_script_once(
                    stage="script-revision",
                    item_prefix="factual-repair",
                    script=revision.script,
                    research=research,
                    inventory=inventory,
                    review=review,
                )
                if self.workflow_policy_version >= 11:
                    repaired_script, fit_usage = self._fit_scene_local_script_word_range(
                        script=repaired_script,
                        outline=outline,
                        research=research,
                        scene_word_targets=revision_input["scene_word_targets"],
                        minimum_total=int(revision_input["minimum_total_word_count"]),
                        target_total=int(revision_input["target_total_word_count"]),
                        maximum_total=int(revision_input["maximum_total_word_count"]),
                        stage="script-revision",
                        task_id="script_revision",
                        strategy_field="revision_strategy",
                        item_prefix="factual-aggregate-word-fit",
                    )
                    repair_usage.extend(fit_usage)
                revision = RevisedScript(
                    script=repaired_script,
                    dispositions=revision.dispositions,
                )
                invariant(revision)
                changed_scene_ids = {
                    before.scene_id
                    for before, after in zip(
                        audited_script.scenes,
                        repaired_script.scenes,
                        strict=True,
                    )
                    if before.spoken_text != after.spoken_text
                }
                inventory, review, repaired_audit_usage = self._factual_audit(
                    stage="script-revision",
                    item_prefix="repaired-script",
                    script=revision.script,
                    research=research,
                    prior_inventory=inventory,
                    prior_review=review,
                    changed_scene_ids=changed_scene_ids,
                )
                item_usage.extend(repair_usage)
                item_usage.extend(repaired_audit_usage)
            artifact: RevisedScript | FactualRevisedScript = FactualRevisedScript(
                script=revision.script,
                dispositions=revision.dispositions,
                claim_inventory=inventory,
                factual_review=review,
            )
        else:
            artifact = revision
        promoted = self.store.complete_fanout_stage(
            "script-revision",
            artifact,
            usage=item_usage,
        )
        return artifact_model.model_validate(promoted)

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
                remaining_slots = self.config.research_source_limit - len(sources)
                remaining_queries = len(queries) - query_index + 1
                request = SearchRequest(
                    query=query,
                    max_results=min(
                        5,
                        max(1, math.ceil(remaining_slots / remaining_queries)),
                    ),
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
        outline: OutlineLike | None = None,
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
        if (
            self.workflow_policy_version >= 6
            and not duration_is_accepted(bundle.timeline, self.config.duration_seconds)
        ):
            tempo_items, tempo = self._tempo_fit_narration_items(
                items,
                pause_seconds=sum(scene.pause_after_seconds for scene in script.scenes),
                budget_seconds=self.config.duration_seconds,
                output_root=aggregate_workspace.work_dir / "tempo-adjusted-initial",
            )
            if tempo_items is not None and tempo is not None:
                tempo_bundle = self._assemble_narration(
                    script,
                    tempo_items,
                    duration_repaired=True,
                    tempo_adjustment=tempo,
                )
                tempo_pause_script = self._fit_pauses_to_budget(
                    script,
                    tempo_bundle.timeline,
                    self.config.duration_seconds,
                    allow_expansion=False,
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
            target = self.config.duration_seconds * 0.95
            selected = [scene.scene_id for scene in script.scenes]
            scale = self._duration_repair_scale(
                bundle.timeline,
                target_seconds=target,
                selected_scene_ids=set(selected),
            )
            repaired_revision, llm_repair_usage, scene_repair_targets = self._duration_repair(
                script=script,
                measured_timeline=bundle.timeline,
                target_seconds=target,
                duration_scale=scale,
                selected_scene_ids=selected,
                factual_research=factual_research,
                outline=outline,
            )
            repaired = repaired_revision.script
            if factual_research is not None:
                inventory, review, factual_usage = self._factual_audit(
                    stage="narration",
                    item_prefix="duration-repaired-script",
                    script=repaired,
                    research=factual_research,
                    raise_on_failure=self.workflow_policy_version < 9,
                )
                usage.extend(factual_usage)
                if not review.passed:
                    audited_repaired_script = repaired
                    repaired, repair_usage = self._repair_factual_script_once(
                        stage="narration",
                        item_prefix="duration-factual-repair",
                        script=repaired,
                        research=factual_research,
                        inventory=inventory,
                        review=review,
                    )
                    if self.workflow_policy_version >= 11:
                        repaired, fit_usage, _ = self._fit_duration_repair_word_range(
                            script=repaired,
                            original=script,
                            scene_repair_targets=scene_repair_targets,
                            selected_scene_ids=set(selected),
                            factual_research=factual_research,
                            outline=outline,
                            item_prefix="duration-factual-word-fit",
                        )
                        repair_usage.extend(fit_usage)
                    self._validate_duration_revision(
                        RevisedScript(script=repaired, dispositions=[]),
                        script,
                        set(selected),
                        scene_repair_targets=scene_repair_targets,
                        enforce_scene_ranges=self.workflow_policy_version < 11,
                    )
                    changed_scene_ids = {
                        before.scene_id
                        for before, after in zip(
                            audited_repaired_script.scenes,
                            repaired.scenes,
                            strict=True,
                        )
                        if before.spoken_text != after.spoken_text
                    }
                    _, _, repaired_factual_usage = self._factual_audit(
                        stage="narration",
                        item_prefix="duration-repaired-script-recheck",
                        script=repaired,
                        research=factual_research,
                        prior_inventory=inventory,
                        prior_review=review,
                        changed_scene_ids=changed_scene_ids,
                    )
                    usage.extend(repair_usage)
                    usage.extend(repaired_factual_usage)
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
        if speech_seconds + pause_seconds > budget_seconds:
            desired_speech_seconds = budget_seconds * 0.98 - pause_seconds
            if desired_speech_seconds <= 0:
                return None
            tempo = speech_seconds / desired_speech_seconds
            return tempo if 1.0 < tempo <= 1.15 else None
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
        factual_research: FactualResearchPack | None = None,
        outline: OutlineLike | None = None,
    ) -> tuple[RevisedScript, list[UsageRecord], list[dict[str, int | str]]]:
        task_id = "duration_repair"
        backend_id = self.config.task_bindings[task_id]
        descriptor = self.registry.descriptor(backend_id)
        prompt = self.prompts.get(task_id, language=self.config.output_language)
        scene_local_repair = self.workflow_policy_version >= 6
        schema_hash = (
            hash_value(
                {
                    "scene_output": restricted_json_schema(
                        ReplacementText.model_json_schema(mode="validation")
                    ),
                    "aggregate": restricted_json_schema(
                        RevisedScript.model_json_schema(mode="validation")
                    ),
                }
            )
            if scene_local_repair
            else hash_value(
                restricted_json_schema(ExpandedSceneText.model_json_schema())
                if duration_scale > 1
                else self.prompts.schema(task_id)
            )
        )
        selected = set(selected_scene_ids)
        scene_repair_targets = []
        for scene in script.scenes:
            if scene.scene_id not in selected:
                continue
            original_words = len(scene.spoken_text.split())
            target_words = max(1, round(original_words * duration_scale))
            word_tolerance = 2 if scene_local_repair else 6 if duration_scale > 1 else 2
            if duration_scale > 1:
                minimum_words = max(original_words + 1, target_words - word_tolerance)
                maximum_words = max(minimum_words, target_words + word_tolerance)
            elif scene_local_repair and original_words > 1:
                maximum_words = min(original_words - 1, target_words + word_tolerance)
                minimum_words = min(maximum_words, max(1, target_words - word_tolerance))
            else:
                minimum_words = max(1, target_words - word_tolerance)
                maximum_words = target_words + word_tolerance
            scene_repair_targets.append(
                {
                    "scene_id": scene.scene_id,
                    "original_word_count": original_words,
                    "target_word_count": target_words,
                    "minimum_word_count": minimum_words,
                    "maximum_word_count": maximum_words,
                    "minimum_word_delta": minimum_words - original_words,
                    "target_word_delta": target_words - original_words,
                    "maximum_word_delta": maximum_words - original_words,
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
                "per-scene-text-v4-host-aggregate-fit"
                if scene_local_repair and self.workflow_policy_version >= 11
                else "per-scene-text-v3"
                if scene_local_repair
                else "per-scene-lengthening-v1"
                if duration_scale > 1
                else "full-script-v1"
            ),
        }
        if factual_research is not None:
            item_input["available_factual_evidence"] = [
                {
                    "evidence_id": item.evidence_id,
                    "supported_statement": item.supported_statement,
                    "confidence": item.confidence,
                    "limitations": item.limitations,
                }
                for item in factual_research.evidence
            ]
        item_id = "duration-repair-script"
        input_hash = hash_run_input(item_input)
        config_hash = hash_value(
            {
                "backend": backend_id,
                "language": self.config.output_language.value,
                "duration_seconds": self.config.duration_seconds,
                "strategy": item_input["repair_strategy"],
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
                enforce_scene_ranges=self.workflow_policy_version < 11,
            )
            return repaired, reusable.usage, scene_repair_targets
        workspace = self.store.workspace("narration", item_id=item_id)
        if scene_local_repair:
            repaired, item_usage, provider_response = self._repair_duration_by_scene(
                script=script,
                measured_timeline=measured_timeline,
                duration_scale=duration_scale,
                scene_repair_targets=scene_repair_targets,
                selected_scene_ids=selected,
                factual_research=factual_research,
                outline=outline,
            )
        elif duration_scale > 1:
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
        return RevisedScript.model_validate(promoted), item_usage, scene_repair_targets

    def _repair_duration_by_scene(
        self,
        *,
        script: NarrationScript,
        measured_timeline: NarrationTimeline,
        duration_scale: float,
        scene_repair_targets: list[dict[str, int | str]],
        selected_scene_ids: set[str],
        factual_research: FactualResearchPack | None,
        outline: OutlineLike | None = None,
    ) -> tuple[RevisedScript, list[UsageRecord], dict[str, Any]]:
        advisory_scene_words = getattr(self, "workflow_policy_version", 6) >= 11
        targets = {str(target["scene_id"]): target for target in scene_repair_targets}
        timeline = {scene.scene_id: scene for scene in measured_timeline.scenes}
        evidence_by_id = (
            {
                item.evidence_id: {
                    "evidence_id": item.evidence_id,
                    "supported_statement": item.supported_statement,
                    "confidence": item.confidence,
                    "limitations": item.limitations,
                }
                for item in factual_research.evidence
            }
            if factual_research is not None
            else {}
        )
        outline_by_id = (
            {scene.scene_id: scene for scene in outline.scenes}
            if outline is not None
            else {}
        )
        repaired_scenes = []
        usage: list[UsageRecord] = []
        response_index: dict[str, Any] = {
            "repair_strategy": (
                "per-scene-text-v4-host-aggregate-fit"
                if advisory_scene_words
                else "per-scene-text-v3"
            ),
            "items": {},
        }
        for index, scene in enumerate(script.scenes):
            if scene.scene_id not in selected_scene_ids:
                repaired_scenes.append(scene)
                continue
            target = targets[scene.scene_id]
            measured_scene = timeline[scene.scene_id]
            speech_seconds = measured_scene.speech_end_seconds - measured_scene.start_seconds
            pause_seconds = measured_scene.end_seconds - measured_scene.speech_end_seconds
            minimum_words = int(target["minimum_word_count"])
            target_words = int(target["target_word_count"])
            maximum_words = int(target["maximum_word_count"])
            original_words = int(target["original_word_count"])
            direction = "lengthen" if duration_scale > 1 else "shorten"
            outline_scene = outline_by_id.get(scene.scene_id)
            preferred_evidence_ids = list(
                getattr(outline_scene, "evidence_ids", [])
            )
            available_scene_evidence = (
                [
                    evidence_by_id[evidence_id]
                    for evidence_id in preferred_evidence_ids
                    if evidence_id in evidence_by_id
                ]
                if advisory_scene_words and outline is not None
                else list(evidence_by_id.values())
            )
            item_input = {
                "repair_strategy": "single-scene-text-v3",
                "scene_word_policy": (
                    "advisory-with-host-aggregate-fit-v1"
                    if advisory_scene_words
                    else "strict-per-scene-v1"
                ),
                "direction": direction,
                "spoken_text": scene.spoken_text,
                "adjacent_context": {
                    "previous_spoken_text": (
                        script.scenes[index - 1].spoken_text if index > 0 else ""
                    ),
                    "next_spoken_text": (
                        script.scenes[index + 1].spoken_text
                        if index + 1 < len(script.scenes)
                        else ""
                    ),
                },
                "measured_timing": {
                    "speech_seconds": speech_seconds,
                    "pause_seconds": pause_seconds,
                    "target_seconds": speech_seconds * duration_scale + pause_seconds,
                    "duration_scale": duration_scale,
                },
                "available_factual_evidence": available_scene_evidence,
                "content_mode": self.config.content_mode.value,
                "output_language": self.config.output_language.value,
                "original_word_count": original_words,
                "minimum_word_count": minimum_words,
                "target_word_count": target_words,
                "maximum_word_count": maximum_words,
                "count_method": "len(spoken_text.split())",
            }

            def validate_repair_text(
                replacement: ReplacementText,
                *,
                minimum: int = minimum_words,
                maximum: int = maximum_words,
            ) -> None:
                NarrationScript.model_validate(
                    {
                        "schema_version": 1,
                        "title": "Duration Repair Scene",
                        "scenes": [
                            {
                                "scene_id": "scene-001",
                                "spoken_text": replacement.spoken_text,
                                "pause_after_seconds": 0,
                            }
                        ],
                    }
                )
                actual = len(replacement.spoken_text.split())
                if not advisory_scene_words and not minimum <= actual <= maximum:
                    boundary = minimum if actual < minimum else maximum
                    raise BackendError(
                        (
                            f"single-Scene Duration Repair has {actual} words; required "
                            f"inclusive range is {minimum}-{maximum}"
                        ),
                        kind=ErrorKind.INVALID_OUTPUT,
                        details={
                            "actual_word_count": actual,
                            "minimum_word_count": minimum,
                            "maximum_word_count": maximum,
                            "target_word_count": boundary,
                            "word_delta": boundary - actual,
                            "count_method": "len(spoken_text.split())",
                        },
                    )

            if direction == "lengthen" and advisory_scene_words:
                editing_instruction = (
                    "Add only useful explanation that belongs in this Scene; the target is guidance, "
                    "not permission to pad. In factual mode, any added assertion must be directly "
                    "supported by available_factual_evidence. Returning a naturally shorter Scene is "
                    "allowed because Python reconciles the complete Script."
                )
            elif direction == "lengthen":
                editing_instruction = (
                    "Preserve the useful original wording and sentence order, then add only enough "
                    "complete spoken wording to reach the range. In factual mode, every added assertion "
                    "must be directly supported by available_factual_evidence; otherwise add only a "
                    "non-factual connective or clarification already implicit in the original."
                )
            else:
                editing_instruction = (
                    "Make a deletion-first minimal edit. Preserve the original sentence order and "
                    "wording, removing only enough nonessential wording to reach the range. Do not "
                    "introduce or paraphrase any claim."
                )
            item_id = f"duration-text-{scene.scene_id}"
            replacement, item_usage = self._structured_item(
                stage="narration",
                item_id=item_id,
                task_id="duration_repair",
                input_data=item_input,
                output_model=ReplacementText,
                invariant=validate_repair_text,
                max_output_tokens=800,
                instruction_suffix=(
                    (
                        "Return exactly one complete replacement spoken_text and no host-owned field. "
                        f"Aim naturally near {target_words} words; {minimum_words}-{maximum_words} is "
                        "per-Scene planning guidance while Python enforces the aggregate range. "
                    )
                    if advisory_scene_words
                    else (
                        "Return exactly one complete replacement spoken_text and no host-owned field. "
                        f"It must contain {minimum_words}-{maximum_words} whitespace-separated words "
                        f"inclusive, aiming for {target_words}. "
                    )
                ) + editing_instruction,
            )
            repaired_scenes.append(
                scene.model_copy(update={"spoken_text": replacement.spoken_text})
            )
            usage.extend(item_usage)
            response_index["items"][scene.scene_id] = {"item_id": item_id}

        revision = RevisedScript(
            script=NarrationScript(title=script.title, scenes=repaired_scenes),
            dispositions=[],
        )
        if advisory_scene_words:
            fitted_script, fit_usage, fit_items = self._fit_duration_repair_word_range(
                script=revision.script,
                original=script,
                scene_repair_targets=scene_repair_targets,
                selected_scene_ids=selected_scene_ids,
                factual_research=factual_research,
                outline=outline,
            )
            revision = RevisedScript(script=fitted_script, dispositions=[])
            usage.extend(fit_usage)
            response_index["word_fit_items"] = fit_items
        self._validate_duration_revision(
            revision,
            script,
            selected_scene_ids,
            scene_repair_targets=scene_repair_targets,
            enforce_scene_ranges=not advisory_scene_words,
        )
        return revision, usage, response_index

    def _fit_duration_repair_word_range(
        self,
        *,
        script: NarrationScript,
        original: NarrationScript,
        scene_repair_targets: list[dict[str, int | str]],
        selected_scene_ids: set[str],
        factual_research: FactualResearchPack | None,
        outline: OutlineLike | None,
        item_prefix: str = "duration-word-fit",
    ) -> tuple[NarrationScript, list[UsageRecord], list[dict[str, str]]]:
        targets = {str(item["scene_id"]): item for item in scene_repair_targets}
        minimum_total = sum(int(item["minimum_word_count"]) for item in scene_repair_targets)
        target_total = sum(int(item["target_word_count"]) for item in scene_repair_targets)
        maximum_total = sum(int(item["maximum_word_count"]) for item in scene_repair_targets)
        working_scenes = [scene.model_copy(deep=True) for scene in script.scenes]

        def selected_total() -> int:
            return sum(
                len(scene.spoken_text.split())
                for scene in working_scenes
                if scene.scene_id in selected_scene_ids
            )

        total = selected_total()
        if minimum_total <= total <= maximum_total:
            return script, [], []

        original_by_id = {scene.scene_id: scene for scene in original.scenes}
        outline_by_id = (
            {scene.scene_id: scene for scene in outline.scenes}
            if outline is not None
            else {}
        )
        evidence_by_id = (
            {
                evidence.evidence_id: {
                    "evidence_id": evidence.evidence_id,
                    "supported_statement": evidence.supported_statement,
                    "confidence": evidence.confidence,
                    "limitations": evidence.limitations,
                }
                for evidence in factual_research.evidence
            }
            if factual_research is not None
            else {}
        )
        fitted_scene_ids: set[str] = set()
        usage: list[UsageRecord] = []
        fit_items: list[dict[str, str]] = []
        while not minimum_total <= total <= maximum_total:
            direction = "lengthen" if total < minimum_total else "shorten"
            candidates = [
                scene
                for scene in working_scenes
                if scene.scene_id in selected_scene_ids
                and scene.scene_id not in fitted_scene_ids
                and (direction == "lengthen" or len(scene.spoken_text.split()) > 1)
            ]
            self._require(
                bool(candidates),
                "Host aggregate Duration Repair fitting exhausted all editable Scenes",
            )

            def rank(scene: Any) -> tuple[int, int, int]:
                actual = len(scene.spoken_text.split())
                desired = int(targets[scene.scene_id]["target_word_count"])
                evidence_backed = int(
                    bool(getattr(outline_by_id.get(scene.scene_id), "evidence_ids", []))
                )
                if direction == "lengthen":
                    return evidence_backed, desired - actual, actual
                return 0, actual - desired, actual

            selected = max(candidates, key=rank)
            selected_index = next(
                index
                for index, scene in enumerate(working_scenes)
                if scene.scene_id == selected.scene_id
            )
            actual_words = len(selected.spoken_text.split())
            unchanged_total = total - actual_words
            if direction == "lengthen":
                minimum_words = max(actual_words + 1, minimum_total - unchanged_total)
                maximum_words = max(minimum_words, maximum_total - unchanged_total)
            else:
                feasible_maximum = maximum_total - unchanged_total
                if feasible_maximum < 1:
                    minimum_words = maximum_words = 1
                else:
                    minimum_words = max(1, minimum_total - unchanged_total)
                    maximum_words = min(actual_words - 1, feasible_maximum)
                    self._require(
                        minimum_words <= maximum_words,
                        "Host Duration Repair fitting calculated an empty replacement range",
                    )
            target_words = min(
                max(target_total - unchanged_total, minimum_words),
                maximum_words,
            )
            outline_scene = outline_by_id.get(selected.scene_id)
            preferred_evidence_ids = list(
                getattr(outline_scene, "evidence_ids", [])
            )
            item_input = {
                "repair_strategy": "single-scene-word-fit-v1",
                "direction": direction,
                "spoken_text": selected.spoken_text,
                "original_spoken_text": original_by_id[selected.scene_id].spoken_text,
                "adjacent_context": {
                    "previous_spoken_text": (
                        working_scenes[selected_index - 1].spoken_text
                        if selected_index > 0
                        else ""
                    ),
                    "next_spoken_text": (
                        working_scenes[selected_index + 1].spoken_text
                        if selected_index + 1 < len(working_scenes)
                        else ""
                    ),
                },
                "available_factual_evidence": [
                    evidence_by_id[evidence_id]
                    for evidence_id in preferred_evidence_ids
                    if evidence_id in evidence_by_id
                ],
                "content_mode": self.config.content_mode.value,
                "output_language": self.config.output_language.value,
                "minimum_word_count": minimum_words,
                "target_word_count": target_words,
                "maximum_word_count": maximum_words,
                "aggregate_word_counts": {
                    "current": total,
                    "minimum": minimum_total,
                    "target": target_total,
                    "maximum": maximum_total,
                },
                "count_method": "len(spoken_text.split())",
            }

            def validate_fit(
                replacement: ReplacementText,
                *,
                minimum: int = minimum_words,
                maximum: int = maximum_words,
            ) -> None:
                NarrationScript.model_validate(
                    {
                        "schema_version": 1,
                        "title": "Duration word fit",
                        "scenes": [
                            {
                                "scene_id": "scene-001",
                                "spoken_text": replacement.spoken_text,
                                "pause_after_seconds": 0,
                            }
                        ],
                    }
                )
                actual = len(replacement.spoken_text.split())
                if not minimum <= actual <= maximum:
                    boundary = minimum if actual < minimum else maximum
                    raise BackendError(
                        (
                            f"Duration word fit has {actual} words; required inclusive range "
                            f"is {minimum}-{maximum}"
                        ),
                        kind=ErrorKind.INVALID_OUTPUT,
                        details={
                            "actual_word_count": actual,
                            "minimum_word_count": minimum,
                            "maximum_word_count": maximum,
                            "target_word_count": boundary,
                            "word_delta": boundary - actual,
                            "count_method": "len(spoken_text.split())",
                        },
                    )

            editing_instruction = (
                "Add only useful explanation that belongs in this Scene. In factual mode, every added "
                "assertion must be directly supported by available_factual_evidence; do not pad with "
                "filler."
                if direction == "lengthen"
                else "Use a deletion-first edit and preserve the Scene's essential meaning and facts."
            )
            item_id = f"{item_prefix}-{selected.scene_id}"
            replacement, item_usage = self._structured_item(
                stage="narration",
                item_id=item_id,
                task_id="duration_repair",
                input_data=item_input,
                output_model=ReplacementText,
                invariant=validate_fit,
                max_output_tokens=800,
                instruction_suffix=(
                    "Return only one complete replacement spoken_text. Python selected this Scene and "
                    "computed a feasible aggregate residual range; do not return an ID, title, pause, "
                    f"word count, or explanation. Use {minimum_words}-{maximum_words} words inclusive, "
                    f"aiming near {target_words}. {editing_instruction}"
                ),
            )
            working_scenes[selected_index] = selected.model_copy(
                update={"spoken_text": replacement.spoken_text}
            )
            fitted_scene_ids.add(selected.scene_id)
            usage.extend(item_usage)
            fit_items.append({"scene_id": selected.scene_id, "item_id": item_id})
            total = selected_total()

        return NarrationScript(title=script.title, scenes=working_scenes), usage, fit_items

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
        enforce_scene_ranges: bool = True,
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
            if violations and (
                total_invalid or (enforce_scene_ranges and severe_scene_error)
            ):
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
            item_config_hash = hash_value(
                {
                    "backend": backend_id,
                    "language": self.config.output_language.value,
                    "reconciliation_revision": CAPTION_RECONCILIATION_REVISION,
                }
            )
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
        if self.workflow_policy_version >= 8:
            metadata["config_hash"] = hash_value(
                {
                    "strategy": "prompt-content-v1",
                    "compiler_backend": metadata["backend_id"],
                    "compiler_revision": metadata["backend_revision"],
                    "target_backend": target_backend_id,
                    "target_revision": target_descriptor.revision,
                }
            )
            metadata["schema_hash"] = hash_value(
                {
                    "prompt_content": restricted_json_schema(
                        ImagePromptContent.model_json_schema(mode="validation")
                    ),
                    "aggregate": restricted_json_schema(
                        ImageRequestSet.model_json_schema(mode="validation")
                    ),
                }
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
            if self.workflow_policy_version >= 8:
                item_input["compiler_strategy"] = "prompt-content-v1"
            if self.continuity_policy_enabled:
                item_input["continuity_context"] = {
                    "previous_brief": (
                        previous_brief.model_dump(mode="json") if previous_brief else None
                    ),
                    "next_brief": next_brief.model_dump(mode="json") if next_brief else None,
                    "rule": "Context only: depict the current Visual Brief, never an adjacent action.",
                }
            if self.workflow_policy_version >= 8:
                prompt_content, item_usage = self._structured_item(
                    stage="image-prompt-compile",
                    item_id=visual_id,
                    task_id="image_prompt_compile",
                    input_data=item_input,
                    output_model=ImagePromptContent,
                    invariant=self._validate_image_request_language,
                    max_output_tokens=1600,
                    instruction_suffix=(
                        "Return only prompt and negative_prompt for this one image. Python owns "
                        "the Scene/Shot identity, target Backend, dimensions, quality, seed, "
                        "reference paths, and generation settings; do not return any of them."
                    ),
                    target_image_backend=target_backend_id,
                )
                request_data = {
                    "schema_version": 1,
                    "scene_id": visual_brief.scene_id,
                    "target_backend_id": target_backend_id,
                    "prompt": prompt_content.prompt,
                    "negative_prompt": prompt_content.negative_prompt,
                    "width": generation_width,
                    "height": generation_height,
                    "quality": item_input["image_quality"],
                    "reference_paths": item_input["reference_paths"],
                    "settings": {},
                }
                if isinstance(visual_plan, TimedVisualPlan):
                    request_data["shot_id"] = visual_id
                image_request = request_model.model_validate(request_data)
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
                requests.append(image_request)
                usage.extend(item_usage)
                continue
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
