from __future__ import annotations

import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from langdetect import DetectorFactory, LangDetectException, detect
from pydantic import BaseModel, ConfigDict, Field

from .contracts import (
    AlignmentRequest,
    BackendDescriptor,
    CaptionTrack,
    CandidateSet,
    CreativeBrief,
    DeliveryManifest,
    FailurePolicy,
    ImageAsset,
    ImageRequest,
    MediaReference,
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
    UsageRecord,
    VisualPlan,
    VisualReviewItem,
    VisualReviewReport,
    WordTiming,
)
from .errors import BackendError, ErrorKind, MediaError, VideoGeneratorError
from .executor import StructuredExecution, TaskExecutor, result_usage
from .media import (
    AudioProbe,
    MediaTools,
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
from .prompting import PromptLibrary, task_output_language
from .provenance import verify_runtime_snapshot
from .registry import BackendRegistry
from .run_store import RunStore
from .task_models import TASK_OUTPUT_MODELS
from .util import (
    atomic_write_json,
    hash_run_input,
    hash_value,
    relative_path,
    replace_path,
    sha256_file,
)


INTERNAL_REVISION = "media-workflow-v2"
DetectorFactory.seed = 0


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


class AlignedSceneWords(WorkflowModel):
    scene_id: str
    words: list[WordTiming]
    coverage: float


class CaptionBundle(WorkflowModel):
    enabled: bool
    track: CaptionTrack | None = None
    srt: MediaReference | None = None
    ass: MediaReference | None = None


class ImageRequestSet(WorkflowModel):
    requests: list[ImageRequest]


class ImageItem(WorkflowModel):
    generated: ImageAsset
    normalized_image: MediaReference


class ImageSet(WorkflowModel):
    items: list[ImageItem]


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
            candidates = self._structured_stage(
                "ideate",
                "ideate",
                {
                    "brief": self.brief.model_dump(mode="json"),
                    "research_pack": research.model_dump(mode="json"),
                    "candidate_count": self.config.idea_candidates,
                    "duration_seconds": self.config.duration_seconds,
                },
                CandidateSet,
                invariant=lambda value: self._validate_candidates(value, research),
            )
            if self._stop("ideate"):
                return None
            selection = self._structured_stage(
                "select",
                "select",
                {"candidate_set": candidates.model_dump(mode="json"), "duration_seconds": self.config.duration_seconds},
                SelectionReport,
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
            outline = self._structured_stage(
                "outline",
                "outline",
                {
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
                },
                StoryOutline,
                invariant=self._validate_outline,
            )
            if self._stop("outline"):
                return None
            script_word_plan = self._script_word_plan(outline)
            draft = self._structured_stage(
                "script-draft",
                "script_draft",
                {
                    "brief": self.brief.model_dump(mode="json"),
                    "outline": outline.model_dump(mode="json"),
                    "output_language": self.config.output_language.value,
                    "duration_seconds": self.config.duration_seconds,
                    "estimated_words_per_second": (
                        2.55 if self.config.output_language.value == "en" else 1.95
                    ),
                    **script_word_plan,
                },
                NarrationScript,
                invariant=lambda value: self._validate_draft(
                    value,
                    outline,
                    minimum_words=int(script_word_plan["minimum_total_word_count"]),
                    maximum_words=int(script_word_plan["maximum_total_word_count"]),
                ),
            )
            if self._stop("script-draft"):
                return None
            story_review = self._review_stage("review-story", "review_story", "story", draft, outline)
            if self._stop("review-story"):
                return None
            spoken_review = self._review_stage("review-spoken", "review_spoken", "spoken", draft, outline)
            if self._stop("review-spoken"):
                return None
            constraint_review = self._review_stage(
                "review-constraints", "review_constraints", "constraints", draft, outline
            )
            if self._stop("review-constraints"):
                return None
            revised = self._structured_stage(
                "script-revision",
                "script_revision",
                {
                    "brief": self.brief.model_dump(mode="json"),
                    "outline": outline.model_dump(mode="json"),
                    "script": draft.model_dump(mode="json"),
                    "review_reports": [
                        story_review.model_dump(mode="json"),
                        spoken_review.model_dump(mode="json"),
                        constraint_review.model_dump(mode="json"),
                    ],
                    "required_finding_ids": sorted(
                        finding.finding_id
                        for review in (story_review, spoken_review, constraint_review)
                        for finding in review.findings
                    ),
                    "duration_seconds": self.config.duration_seconds,
                    "output_language": self.config.output_language.value,
                    **script_word_plan,
                },
                RevisedScript,
                invariant=lambda value: self._validate_revision(
                    value,
                    [story_review, spoken_review, constraint_review],
                    [scene.scene_id for scene in outline.scenes],
                    minimum_words=int(script_word_plan["minimum_total_word_count"]),
                    maximum_words=int(script_word_plan["maximum_total_word_count"]),
                ),
            )
            final_script = revised.script
            if self._stop("script-revision"):
                return None
            narration = self._narration(final_script)
            if self._stop("narration"):
                return None
            captions = self._captions(narration)
            if self._stop("captions"):
                return None
            visual_plan = self._structured_stage(
                "visual-plan",
                "visual_plan",
                {
                    "script": narration.script.model_dump(mode="json"),
                    "timeline": narration.timeline.model_dump(mode="json"),
                    "style_id": self.config.style,
                    "style_description": self.config.style_description,
                    "audience": self.config.audience,
                    "delivery": {
                        "width": self.config.delivery_width,
                        "height": self.config.delivery_height,
                        "aspect_ratio": "16:9",
                    },
                },
                VisualPlan,
                invariant=lambda value: self._require(
                    [scene.scene_id for scene in value.scenes]
                    == [scene.scene_id for scene in narration.script.scenes],
                    "Visual Plan does not cover every Scene in order",
                ),
            )
            if self._stop("visual-plan"):
                return None
            image_requests = self._image_prompts(visual_plan)
            if self._stop("image-prompt-compile"):
                return None
            images = self._images(image_requests)
            if self._stop("images"):
                return None
            reviewed = self._visual_review(visual_plan, image_requests, images)
            if self._stop("visual-review"):
                return None
            music_brief = self._music_brief(narration)
            if self._stop("music-brief"):
                return None
            music = self._music(music_brief, narration.timeline)
            if self._stop("music"):
                return None
            self.registry.release_local_workers()
            rendered = self._render(narration.timeline, captions, reviewed.images, music)
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
            output_language = task_output_language(task_id, self.config.output_language)
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
            "backend_revision": INTERNAL_REVISION,
            "prompt_version": "",
            "schema_hash": "",
        }

    def _internal_stage_config(self, stage: str) -> dict[str, Any]:
        direct: dict[str, Any] = {"stage": stage, "revision": INTERNAL_REVISION}
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
        outline: StoryOutline,
    ) -> ReviewReport:
        report = self._structured_stage(
            stage,
            task_id,
            {
                "brief": self.brief.model_dump(mode="json"),
                "outline": outline.model_dump(mode="json"),
                "script": script.model_dump(mode="json"),
                "duration_seconds": self.config.duration_seconds,
                "audience": self.config.audience,
                "output_language": self.config.output_language.value,
            },
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
    def _validate_selection(selection: SelectionReport, candidates: CandidateSet) -> None:
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

    def _outline_scene_count_bounds(self) -> tuple[int, int, int]:
        target_count = max(
            1,
            math.ceil(self.config.duration_seconds / self.config.visual_target_seconds),
        )
        return target_count, max(1, target_count - 1), target_count + 1

    def _script_word_plan(self, outline: StoryOutline) -> dict[str, Any]:
        words_per_second = 2.55 if self.config.output_language.value == "en" else 1.95
        minimum_duration_fraction = 0.85 if self.config.output_language.value == "en" else 0.60
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
            scene_targets.append({"scene_id": scene.scene_id, "target_word_count": target})
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
                round(self.config.duration_seconds * words_per_second),
            ),
            "scene_word_targets": scene_targets,
        }

    @classmethod
    def _validate_draft(
        cls,
        draft: NarrationScript,
        outline: StoryOutline,
        *,
        minimum_words: int,
        maximum_words: int,
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

    def _validate_outline(self, outline: StoryOutline) -> None:
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

    @staticmethod
    def _normalize_outline_durations(
        outline: StoryOutline,
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

    def _research(self) -> ResearchPack:
        queries = []
        if not self.config.offline and self.config.research_query_limit:
            candidates = self.brief.research_focus or [self.brief.idea_direction or "unusual family-safe story inspiration"]
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
        }
        metadata = self._stage_metadata(stage="research", task_id="research", input_data=input_seed)
        reusable = self.store.reusable_record("research", **metadata)
        if reusable:
            return self.store.load_artifact(reusable, ResearchPack)
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
            ResearchPack,
            invariant=lambda value: self._validate_research_source_references(value, sources),
        )
        model_pack = ResearchPack.model_validate(execution.artifact)
        pack_data = model_pack.model_dump(mode="json")
        pack_data["queries"] = queries
        pack_data["sources"] = [source.model_dump(mode="json") for source in sources]
        pack = ResearchPack.model_validate(pack_data)
        if len(pack.queries) > self.config.research_query_limit or len(pack.sources) > self.config.research_source_limit:
            raise BackendError("Research Pack exceeded configured query/source limits", kind=ErrorKind.INVALID_OUTPUT)
        atomic_write_json(workspace.work_dir / "provider-response.json", execution.result.raw_response)
        usage.extend(_usage_list([execution.result.usage]))
        promoted = self.store.promote_stage(workspace, pack, usage=usage, warnings=warnings)
        return ResearchPack.model_validate(promoted)

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

    def _narration(self, script: NarrationScript) -> NarrationBundle:
        input_data = {
            "script": script.model_dump(mode="json"),
            "voice": self.config.voice.model_dump(mode="json"),
            "duration_seconds": self.config.duration_seconds,
            "backend": self.config.task_bindings["narration_synthesis"],
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
            )
            if pause_fitted_script is not None:
                pause_fitted_bundle = self._assemble_narration(
                    pause_fitted_script,
                    items,
                    duration_repaired=True,
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
            repair_items, tts_repair_usage = self._synthesize_script(
                repaired, repair=True, selected_scene_ids=set(selected), existing_items=items
            )
            usage.extend(llm_repair_usage)
            usage.extend(tts_repair_usage)
            bundle = self._assemble_narration(repaired, repair_items, duration_repaired=True)
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

    @staticmethod
    def _fit_pauses_to_budget(
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
        maximum_pause_seconds = 3.0 * len(eligible_indexes)
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
                index for index in active if proposed[index] > 3.0 - allocated[index]
            ]
            if not saturated:
                for index in active:
                    allocated[index] += proposed[index]
                remaining = 0
                break
            for index in saturated:
                capacity = 3.0 - allocated[index]
                allocated[index] += capacity
                remaining -= capacity
                active.remove(index)

        rounded = {index: round(value, 4) for index, value in allocated.items()}
        delta = round(allowed_pause_seconds - sum(rounded.values()), 4)
        for index in eligible_indexes:
            if abs(delta) <= 0.00005:
                break
            adjustment = min(delta, 3.0 - rounded[index]) if delta > 0 else max(delta, -rounded[index])
            rounded[index] = round(rounded[index] + adjustment, 4)
            delta = round(delta - adjustment, 4)

        fitted_scenes = []
        for index, scene in enumerate(script.scenes):
            pause = rounded.get(index, 0.0)
            fitted_scenes.append(scene.model_copy(update={"pause_after_seconds": pause}))
        return NarrationScript.model_validate(
            script.model_copy(update={"scenes": fitted_scenes}).model_dump(mode="json")
        )

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
        schema_hash = hash_value(self.prompts.schema(task_id))
        selected = set(selected_scene_ids)
        scene_repair_targets = []
        for scene in script.scenes:
            if scene.scene_id not in selected:
                continue
            original_words = len(scene.spoken_text.split())
            target_words = max(1, round(original_words * duration_scale))
            scene_repair_targets.append(
                {
                    "scene_id": scene.scene_id,
                    "original_word_count": original_words,
                    "target_word_count": target_words,
                    "minimum_word_count": max(1, target_words - 2),
                    "maximum_word_count": target_words + 2,
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
        atomic_write_json(workspace.work_dir / "provider-response.json", execution.result.raw_response)
        item_usage = _usage_list([execution.result.usage])
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
            }
            input_hash = hash_run_input(item_input)
            config_hash = hash_value({"backend": backend_id, "voice": self.config.voice.name})
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
                output_path=relative_path(raw_path, self.project_root),
                preceding_text=script.scenes[index - 1].spoken_text[-500:] if index else "",
                following_text=script.scenes[index + 1].spoken_text[:500] if index + 1 < len(script.scenes) else "",
            )
            result = self.executor.speech(request)
            if result.asset.scene_id != scene.scene_id:
                raise BackendError("Speech Backend changed the Scene ID", kind=ErrorKind.INVALID_OUTPUT)
            probe = normalize_audio(self.tools, self.project_root / result.asset.audio.path, normalized_path)
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
        self, script: NarrationScript, items: Sequence[NarrationItem], *, duration_repaired: bool
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
        return NarrationBundle(script=script, timeline=timeline, items=list(items), duration_repaired=duration_repaired)

    def _captions(self, narration: NarrationBundle) -> CaptionBundle:
        input_data = {
            "script": narration.script.model_dump(mode="json"),
            "timeline": narration.timeline.model_dump(mode="json"),
            "enabled": self.config.captions_enabled,
            "animated": self.config.animated_captions,
        }
        metadata = self._stage_metadata(stage="captions", task_id=None, input_data=input_data)
        if self.config.captions_enabled and not all(scene.words for scene in narration.timeline.scenes):
            backend_id = self.config.task_bindings["caption_alignment"]
            metadata["backend_id"] = backend_id
            metadata["backend_revision"] = self.registry.descriptor(backend_id).revision
        reusable = self.store.reusable_record("captions", **metadata)
        if reusable:
            return self.store.load_artifact(reusable, CaptionBundle)
        workspace = self.store.workspace("captions")
        self.store.begin_stage("captions", attempt=workspace.attempt, **metadata)
        if not self.config.captions_enabled:
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
        srt_path = workspace.work_dir / f"captions.{self.config.output_language.value}.srt"
        write_srt(track, srt_path)
        ass_ref = None
        if self.config.animated_captions:
            ass_path = workspace.work_dir / f"captions.{self.config.output_language.value}.ass"
            write_ass(track, ass_path, width=self.config.delivery_width, height=self.config.delivery_height)
            ass_ref = MediaReference(
                path=relative_path(ass_path, self.project_root),
                sha256=sha256_file(ass_path),
                mime_type="text/x-ass",
            )
        bundle = CaptionBundle(
            enabled=True,
            track=track,
            srt=MediaReference(
                path=relative_path(srt_path, self.project_root),
                sha256=sha256_file(srt_path),
                mime_type="application/x-subrip",
            ),
            ass=ass_ref,
        )
        promoted = self.store.promote_stage(workspace, bundle, usage=usage)
        return CaptionBundle.model_validate(promoted)

    def _image_prompts(self, visual_plan: VisualPlan) -> ImageRequestSet:
        target_backend_id = self.config.task_bindings["image_generate"]
        target_descriptor = self.registry.descriptor(target_backend_id)
        input_data = {
            "visual_plan": visual_plan.model_dump(mode="json"),
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
        if target_backend_id == "local:flux.2-klein-4b":
            generation_width, generation_height = 1024, 576
        elif target_backend_id == "deterministic:stick":
            generation_width, generation_height = self.config.delivery_width, self.config.delivery_height
        else:
            generation_width, generation_height = 2048, 1152
        characters = [character.model_dump(mode="json") for character in visual_plan.characters]
        for visual_brief in visual_plan.scenes:
            item_input = {
                "visual_brief": visual_brief.model_dump(mode="json"),
                "style_profile": visual_plan.style_profile.model_dump(mode="json"),
                "characters": characters,
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
            item_hash = hash_run_input(item_input)
            item_config_hash = hash_value(
                {
                    "compiler": self.config.task_bindings["image_prompt_compile"],
                    "target": target_backend_id,
                    "language": task_output_language(
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
                visual_brief.scene_id,
                input_hash=item_hash,
                config_hash=item_config_hash,
                backend_id=compiler_id,
                backend_revision=compiler_descriptor.revision,
                prompt_version=prompt.version,
                schema_hash=schema_hash,
            )
            if reusable_item:
                image_request = self.store.load_item_artifact(reusable_item, ImageRequest)
                self._validate_image_request_language(image_request)
                usage.extend(reusable_item.usage)
            else:
                item_workspace = self.store.workspace("image-prompt-compile", item_id=visual_brief.scene_id)
                execution = self.executor.structured(
                    "image_prompt_compile",
                    item_input,
                    ImageRequest,
                    target_image_backend=target_backend_id,
                    invariant=self._validate_image_request_language,
                )
                image_request = ImageRequest.model_validate(execution.artifact)
                image_request = self._canonical_image_request(
                    image_request,
                    scene_id=visual_brief.scene_id,
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
                image_request = ImageRequest.model_validate(promoted)
                usage.extend(item_usage)
            requests.append(image_request)
        bundle = ImageRequestSet(requests=requests)
        promoted = self.store.complete_fanout_stage("image-prompt-compile", bundle, usage=usage)
        return ImageRequestSet.model_validate(promoted)

    @staticmethod
    def _canonical_image_request(
        image_request: ImageRequest,
        *,
        scene_id: str,
        target_backend_id: str,
        width: int,
        height: int,
        quality: str,
        reference_paths: list[str],
    ) -> ImageRequest:
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
        return image_request.model_copy(
            update={
                "scene_id": scene_id,
                "target_backend_id": target_backend_id,
                "width": width,
                "height": height,
                "quality": quality,
                "reference_paths": reference_paths,
                "settings": settings,
            }
        )

    @staticmethod
    def _validate_image_request_language(image_request: ImageRequest) -> None:
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
        for request in request_set.requests:
            item_input = request.model_dump(mode="json")
            item_hash = hash_run_input(item_input)
            config_hash = hash_value(
                {"backend": backend_id, "width": self.config.delivery_width, "height": self.config.delivery_height}
            )
            reusable_item = self.store.reusable_item(
                "images",
                request.scene_id,
                input_hash=item_hash,
                config_hash=config_hash,
                backend_id=backend_id,
                backend_revision=descriptor.revision,
            )
            if reusable_item:
                item = self.store.load_item_artifact(reusable_item, ImageItem)
                usage.extend(reusable_item.usage)
            else:
                workspace = self.store.workspace("images", item_id=request.scene_id)
                extension = _raw_image_extension(backend_id)
                raw_path = workspace.work_dir / f"generated{extension}"
                normalized_path = workspace.work_dir / "normalized.png"
                result = self.executor.image(request, raw_path)
                if result.asset.scene_id != request.scene_id:
                    raise BackendError("Image Backend changed the Scene ID", kind=ErrorKind.INVALID_OUTPUT)
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
            items.append(item)
        bundle = ImageSet(items=items)
        promoted = self.store.complete_fanout_stage("images", bundle, usage=usage)
        return ImageSet.model_validate(promoted)

    def _visual_review(
        self, visual_plan: VisualPlan, requests: ImageRequestSet, images: ImageSet
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
        briefs = {brief.scene_id: brief for brief in visual_plan.scenes}
        request_by_id = {request.scene_id: request for request in requests.requests}
        final_items = {item.generated.scene_id: item for item in images.items}
        reviews: list[VisualReviewItem] = []
        usage = []
        failed = []
        for item in images.items:
            review = self._review_image(briefs[item.generated.scene_id], visual_plan, item, pass_number=1)
            reviews.append(review[0])
            usage.extend(review[1])
            if not review[0].passed:
                failed.append(review[0])
        second_reviews: list[VisualReviewItem] = []
        if failed:
            image_backend_id = self.config.task_bindings["image_generate"]
            image_descriptor = self.registry.descriptor(image_backend_id)
            for failure in failed:
                original = request_by_id[failure.scene_id]
                corrected = original.model_copy(
                    update={
                        "prompt": (
                            original.prompt
                            + "\n\nTargeted correction. Preserve all otherwise correct content. "
                            + failure.regeneration_instruction
                        )
                    }
                )
                regeneration_id = f"{failure.scene_id}-regeneration"
                regeneration_input_hash = hash_run_input(corrected.model_dump(mode="json"))
                regeneration_config_hash = hash_value({"backend": image_backend_id, "regeneration": 1})
                reusable_item = self.store.reusable_item(
                    "visual-review",
                    regeneration_id,
                    input_hash=regeneration_input_hash,
                    config_hash=regeneration_config_hash,
                    backend_id=image_backend_id,
                    backend_revision=image_descriptor.revision,
                )
                if reusable_item:
                    replacement = self.store.load_item_artifact(reusable_item, ImageItem)
                    usage.extend(reusable_item.usage)
                else:
                    workspace = self.store.workspace("visual-review", item_id=regeneration_id)
                    extension = _raw_image_extension(image_backend_id)
                    raw_path = workspace.work_dir / f"generated{extension}"
                    normalized_path = workspace.work_dir / "normalized.png"
                    result = self.executor.image(corrected, raw_path)
                    if result.asset.scene_id != corrected.scene_id:
                        raise BackendError("Image Backend changed the Scene ID", kind=ErrorKind.INVALID_OUTPUT)
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
                    )
                    item_usage = _usage_list([result.usage])
                    promoted = self.store.promote_item(
                        workspace,
                        replacement,
                        input_hash=regeneration_input_hash,
                        config_hash=regeneration_config_hash,
                        backend_id=image_backend_id,
                        backend_revision=image_descriptor.revision,
                        usage=item_usage,
                    )
                    replacement = ImageItem.model_validate(promoted)
                    usage.extend(item_usage)
                final_items[failure.scene_id] = replacement
                second = self._review_image(briefs[failure.scene_id], visual_plan, replacement, pass_number=2)
                second_reviews.append(second[0])
                usage.extend(second[1])
                if not second[0].passed:
                    raise BackendError(
                        f"regenerated image for {failure.scene_id} failed its final re-review",
                        kind=ErrorKind.INVALID_OUTPUT,
                        action="Inspect the Visual Review and explicitly rerun from image-prompt-compile or images.",
                    )
        ordered_images = ImageSet(items=[final_items[brief.scene_id] for brief in visual_plan.scenes])
        report = VisualReviewReport(items=reviews + second_reviews, pass_number=2 if second_reviews else 1)
        bundle = VisualReviewBundle(reviewed=True, report=report, images=ordered_images)
        promoted = self.store.promote_stage(aggregate_workspace, bundle, usage=usage)
        return VisualReviewBundle.model_validate(promoted)

    def _review_image(
        self,
        brief: Any,
        visual_plan: VisualPlan,
        image: ImageItem,
        *,
        pass_number: int,
    ) -> tuple[VisualReviewItem, list[UsageRecord]]:
        backend_id = self.config.task_bindings["visual_review"]
        descriptor = self.registry.descriptor(backend_id)
        item_id = f"{brief.scene_id}-review-{pass_number}"
        item_input = {
            "scene_id": brief.scene_id,
            "visual_brief": brief.model_dump(mode="json"),
            "style_profile": visual_plan.style_profile.model_dump(mode="json"),
            "characters": [character.model_dump(mode="json") for character in visual_plan.characters],
            "audience": self.config.audience,
            "pass_number": pass_number,
            "minimum_score": 4,
            "image_sha256": image.normalized_image.sha256,
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
        if reusable:
            return self.store.load_item_artifact(reusable, VisualReviewItem), reusable.usage
        workspace = self.store.workspace("visual-review", item_id=item_id)
        execution = self.executor.structured(
            "visual_review",
            item_input,
            VisualReviewItem,
            media_inputs=[self.project_root / image.normalized_image.path],
        )
        item = VisualReviewItem.model_validate(execution.artifact)
        if item.scene_id != brief.scene_id:
            raise BackendError("Visual Review changed the Scene ID", kind=ErrorKind.INVALID_OUTPUT)
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
        return VisualReviewItem.model_validate(promoted), item_usage

    def _music_brief(self, narration: NarrationBundle) -> MusicBriefBundle:
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
        metadata = self._stage_metadata(stage="music-brief", task_id=None, input_data=input_data)
        if self.config.music_enabled:
            task_metadata = self._stage_metadata(stage="music-brief", task_id="music_brief", input_data=input_data)
            metadata = task_metadata
        reusable = self.store.reusable_record("music-brief", **metadata)
        if reusable:
            return self.store.load_artifact(reusable, MusicBriefBundle)
        workspace = self.store.workspace("music-brief")
        self.store.begin_stage("music-brief", attempt=workspace.attempt, **metadata)
        if not self.config.music_enabled:
            promoted = self.store.promote_stage(workspace, MusicBriefBundle(enabled=False))
            return MusicBriefBundle.model_validate(promoted)
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
        promoted = self.store.promote_stage(
            workspace, bundle, usage=_usage_list([execution.result.usage])
        )
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
    ) -> RenderBundle:
        input_data = {
            "timeline": timeline.model_dump(mode="json"),
            "captions": captions.model_dump(mode="json"),
            "images": images.model_dump(mode="json"),
            "music": music.model_dump(mode="json"),
        }
        metadata = self._stage_metadata(stage="render", task_id=None, input_data=input_data)
        reusable = self.store.reusable_record("render", **metadata)
        if reusable:
            return self.store.load_artifact(reusable, RenderBundle)
        workspace = self.store.workspace("render")
        self.store.begin_stage("render", attempt=workspace.attempt, **metadata)
        image_by_scene = {item.generated.scene_id: item for item in images.items}
        render_scenes = [
            RenderScene(
                scene_id=scene.scene_id,
                image_path=image_by_scene[scene.scene_id].normalized_image.path,
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
