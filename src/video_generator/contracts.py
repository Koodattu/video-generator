from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, PositiveInt, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class VersionedContract(ContractModel):
    schema_version: Literal[1] = 1


class OutputLanguage(StrEnum):
    ENGLISH = "en"
    FINNISH = "fi"


class Quality(StrEnum):
    DRAFT = "draft"
    FINAL = "final"


class ContentMode(StrEnum):
    FICTION = "fiction"
    FACTUAL = "factual"


class FailurePolicy(StrEnum):
    STRICT = "strict"
    OMIT_WITH_WARNING = "omit_with_warning"


class ProtocolName(StrEnum):
    SEARCH = "search"
    STRUCTURED_TEXT = "structured_text"
    SPEECH = "speech"
    ALIGNMENT = "alignment"
    IMAGE = "image"
    MUSIC = "music"


TASK_IDS: tuple[str, ...] = (
    "search",
    "research",
    "ideate",
    "select",
    "outline",
    "script_draft",
    "review_story",
    "review_spoken",
    "review_constraints",
    "script_revision",
    "factual_review",
    "narration_synthesis",
    "duration_repair",
    "caption_alignment",
    "visual_plan",
    "image_prompt_compile",
    "image_generate",
    "visual_review",
    "music_brief",
    "music_generate",
)


PUBLIC_STAGES: tuple[str, ...] = (
    "research",
    "ideate",
    "select",
    "outline",
    "script-draft",
    "review-story",
    "review-spoken",
    "review-constraints",
    "script-revision",
    "narration",
    "captions",
    "visual-plan",
    "image-prompt-compile",
    "images",
    "visual-review",
    "music-brief",
    "music",
    "render",
    "delivery",
)


TASK_PROTOCOL: dict[str, ProtocolName] = {
    "search": ProtocolName.SEARCH,
    "research": ProtocolName.STRUCTURED_TEXT,
    "ideate": ProtocolName.STRUCTURED_TEXT,
    "select": ProtocolName.STRUCTURED_TEXT,
    "outline": ProtocolName.STRUCTURED_TEXT,
    "script_draft": ProtocolName.STRUCTURED_TEXT,
    "review_story": ProtocolName.STRUCTURED_TEXT,
    "review_spoken": ProtocolName.STRUCTURED_TEXT,
    "review_constraints": ProtocolName.STRUCTURED_TEXT,
    "script_revision": ProtocolName.STRUCTURED_TEXT,
    "factual_review": ProtocolName.STRUCTURED_TEXT,
    "narration_synthesis": ProtocolName.SPEECH,
    "duration_repair": ProtocolName.STRUCTURED_TEXT,
    "caption_alignment": ProtocolName.ALIGNMENT,
    "visual_plan": ProtocolName.STRUCTURED_TEXT,
    "image_prompt_compile": ProtocolName.STRUCTURED_TEXT,
    "image_generate": ProtocolName.IMAGE,
    "visual_review": ProtocolName.STRUCTURED_TEXT,
    "music_brief": ProtocolName.STRUCTURED_TEXT,
    "music_generate": ProtocolName.MUSIC,
}


class VoiceSettings(ContractModel):
    name: Annotated[str, Field(min_length=1, max_length=120)]
    reference_audio: str = ""
    reference_transcript: str = ""
    elevenlabs_voice_id: str = ""
    authorization: Literal["self", "explicit_permission"] = "self"


class RawRunConfig(VersionedContract):
    profile: Literal[
        "local",
        "cloud-openai",
        "cloud-gemini",
        "cloud-openai-gemini",
        "hybrid-local-first",
    ] = "local"
    output_language: OutputLanguage = OutputLanguage.FINNISH
    duration_seconds: Annotated[FiniteFloat, Field(ge=10, le=3600)] = 90
    quality: Quality = Quality.DRAFT
    content_mode: ContentMode = ContentMode.FICTION
    audience: Literal["family_safe_general"] = "family_safe_general"
    style: Annotated[str, Field(min_length=1, max_length=120)] = "ms_paint_stick"
    style_description: Annotated[str, Field(max_length=1000)] = ""
    motion_style: Literal["static_cuts"] = "static_cuts"
    offline: bool = False
    cost_ceiling_usd: Annotated[FiniteFloat, Field(ge=0, le=10000)] = 10.0
    failure_policy: FailurePolicy = FailurePolicy.STRICT
    usage_purpose: Literal["personal_noncommercial"] = "personal_noncommercial"
    idea_candidates: Annotated[int, Field(ge=1, le=10)] = 5
    research_query_limit: Annotated[int, Field(ge=0, le=20)] = 5
    research_source_limit: Annotated[int, Field(ge=0, le=50)] = 10
    visual_target_seconds: Annotated[FiniteFloat, Field(gt=0, le=120)] = 15
    visual_min_seconds: Annotated[FiniteFloat, Field(gt=0, le=120)] = 8
    visual_max_seconds: Annotated[FiniteFloat, Field(gt=0, le=180)] = 25
    music_enabled: bool = False
    captions_enabled: bool = True
    animated_captions: bool = False
    voice: VoiceSettings
    task_overrides: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_combinations(self) -> "RawRunConfig":
        if self.content_mode is ContentMode.FACTUAL:
            raise ValueError(
                "factual mode is not available until claim/evidence capture and factual review are implemented"
            )
        if self.visual_min_seconds > self.visual_target_seconds:
            raise ValueError("visual_min_seconds must not exceed visual_target_seconds")
        if self.visual_target_seconds > self.visual_max_seconds:
            raise ValueError("visual_target_seconds must not exceed visual_max_seconds")
        if self.animated_captions and not self.captions_enabled:
            raise ValueError("animated_captions requires captions_enabled")
        unknown = sorted(set(self.task_overrides) - set(TASK_IDS))
        if unknown:
            raise ValueError(f"unknown task override IDs: {', '.join(unknown)}")
        return self


class CreativeBrief(VersionedContract):
    idea_direction: Annotated[str, Field(max_length=2000)] = ""
    surprise_me: bool = False
    tone: Annotated[str, Field(max_length=500)] = ""
    themes: Annotated[list[str], Field(max_length=20)] = Field(default_factory=list)
    must_include: Annotated[list[str], Field(max_length=20)] = Field(default_factory=list)
    avoid: Annotated[list[str], Field(max_length=20)] = Field(default_factory=list)
    research_focus: Annotated[list[str], Field(max_length=20)] = Field(default_factory=list)

    @field_validator("themes", "must_include", "avoid", "research_focus")
    @classmethod
    def validate_list_items(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value or len(value) > 300 for value in cleaned):
            raise ValueError("list values must contain 1-300 characters")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("list values must be unique")
        return cleaned

    @model_validator(mode="after")
    def validate_direction(self) -> "CreativeBrief":
        cleaned_direction = self.idea_direction.strip()
        object.__setattr__(self, "idea_direction", cleaned_direction)
        if not cleaned_direction and not self.surprise_me:
            raise ValueError("idea_direction is required unless surprise_me is true")
        conflict = sorted(set(self.must_include) & set(self.avoid))
        if conflict:
            raise ValueError(f"the same value cannot be required and avoided: {', '.join(conflict)}")
        return self


class ResolvedRunConfig(VersionedContract):
    profile: str
    profile_version: str
    output_language: OutputLanguage
    duration_seconds: FiniteFloat
    quality: Quality
    content_mode: ContentMode
    audience: str
    style: str
    style_description: str
    motion_style: str
    offline: bool
    cost_ceiling_usd: FiniteFloat
    failure_policy: FailurePolicy
    usage_purpose: str
    idea_candidates: int
    research_query_limit: int
    research_source_limit: int
    visual_target_seconds: FiniteFloat
    visual_min_seconds: FiniteFloat
    visual_max_seconds: FiniteFloat
    music_enabled: bool
    captions_enabled: bool
    animated_captions: bool
    voice: VoiceSettings
    task_bindings: dict[str, str]
    delivery_width: PositiveInt
    delivery_height: PositiveInt
    fps: Literal[30] = 30
    project_root: str
    pricing_snapshot: str
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("task_bindings")
    @classmethod
    def validate_bindings(cls, bindings: dict[str, str]) -> dict[str, str]:
        unknown = sorted(set(bindings) - set(TASK_IDS))
        missing = sorted(set(TASK_IDS) - set(bindings))
        if unknown or missing:
            parts = []
            if unknown:
                parts.append(f"unknown: {', '.join(unknown)}")
            if missing:
                parts.append(f"missing: {', '.join(missing)}")
            raise ValueError("invalid task bindings (" + "; ".join(parts) + ")")
        return bindings


class ResearchSource(ContractModel):
    source_id: str
    url: str
    title: str
    publisher: str = ""
    retrieved_at: datetime = Field(default_factory=utc_now)
    language: str = ""
    excerpt: str = ""
    content_sha256: str = ""


class ResearchFinding(ContractModel):
    finding_id: str
    summary: str
    source_ids: list[str] = Field(default_factory=list)


class ResearchPack(VersionedContract):
    queries: list[str] = Field(default_factory=list)
    sources: list[ResearchSource] = Field(default_factory=list)
    findings: list[ResearchFinding] = Field(default_factory=list)
    motifs: list[str] = Field(default_factory=list)
    setting_details: list[str] = Field(default_factory=list)
    vocabulary: list[str] = Field(default_factory=list)
    cultural_cautions: list[str] = Field(default_factory=list)
    cliches_to_avoid: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_research_references(self) -> "ResearchPack":
        source_ids = [source.source_id for source in self.sources]
        finding_ids = [finding.finding_id for finding in self.findings]
        if len(source_ids) != len(set(source_ids)) or any(not item for item in source_ids):
            raise ValueError("Research Source IDs must be nonempty and unique")
        if len(finding_ids) != len(set(finding_ids)) or any(not item for item in finding_ids):
            raise ValueError("Research Finding IDs must be nonempty and unique")
        unknown = sorted(
            {
                source_id
                for finding in self.findings
                for source_id in finding.source_ids
                if source_id not in set(source_ids)
            }
        )
        if unknown:
            raise ValueError(f"Research Findings reference unknown Source IDs: {', '.join(unknown)}")
        return self


class StoryCandidate(ContractModel):
    candidate_id: str
    title: str
    premise: str
    protagonist_desire: str
    obstacle: str
    turn: str
    ending_direction: str
    emotional_promise: str
    research_inspiration_ids: list[str] = Field(default_factory=list)
    visual_opportunities: list[str] = Field(default_factory=list)
    originality_risks: list[str] = Field(default_factory=list)
    duration_fit: str


class CandidateSet(VersionedContract):
    candidates: list[StoryCandidate]

    @field_validator("candidates")
    @classmethod
    def unique_candidates(cls, values: list[StoryCandidate]) -> list[StoryCandidate]:
        ids = [item.candidate_id.strip() for item in values]
        if not values or any(not item_id for item_id in ids) or len(ids) != len(set(ids)):
            raise ValueError("candidate IDs must be nonempty and unique")
        return values


class CandidateScore(ContractModel):
    candidate_id: str
    duration_fit: Annotated[int, Field(ge=1, le=5)]
    originality: Annotated[int, Field(ge=1, le=5)]
    story_potential: Annotated[int, Field(ge=1, le=5)]
    visual_strength: Annotated[int, Field(ge=1, le=5)]
    spoken_suitability: Annotated[int, Field(ge=1, le=5)]
    audience_fit: Annotated[int, Field(ge=1, le=5)]
    research_responsibility: Annotated[int, Field(ge=1, le=5)]
    rationale: str


class SelectionReport(VersionedContract):
    scores: list[CandidateScore]
    chosen_candidate_id: str
    rationale: str

    @model_validator(mode="after")
    def validate_selection(self) -> "SelectionReport":
        score_ids = [score.candidate_id for score in self.scores]
        if not score_ids or len(score_ids) != len(set(score_ids)):
            raise ValueError("selection scores must contain unique candidate IDs")
        if self.chosen_candidate_id not in score_ids:
            raise ValueError("chosen candidate must have a score")
        return self


class OutlineScene(ContractModel):
    scene_id: str
    narrative_purpose: str
    change: str
    emotional_beat: str
    visual_opportunity: str
    provisional_seconds: Annotated[FiniteFloat, Field(gt=0)]
    continuity_obligations: list[str] = Field(default_factory=list)


class StoryOutline(VersionedContract):
    title: str
    concept_summary: str
    scenes: list[OutlineScene]

    @field_validator("scenes")
    @classmethod
    def validate_scene_ids(cls, scenes: list[OutlineScene]) -> list[OutlineScene]:
        expected = [f"scene-{index:03d}" for index in range(1, len(scenes) + 1)]
        actual = [scene.scene_id for scene in scenes]
        if not scenes or actual != expected:
            raise ValueError(f"scene IDs must be contiguous and ordered: {expected}")
        return scenes


class ScriptScene(ContractModel):
    scene_id: str
    spoken_text: Annotated[str, Field(min_length=1, max_length=10000)]
    pause_after_seconds: Annotated[FiniteFloat, Field(ge=0, le=3.25)] = 0.15


class NarrationScript(VersionedContract):
    title: str
    scenes: list[ScriptScene]

    @field_validator("scenes")
    @classmethod
    def validate_script_scenes(cls, scenes: list[ScriptScene]) -> list[ScriptScene]:
        expected = [f"scene-{index:03d}" for index in range(1, len(scenes) + 1)]
        if [scene.scene_id for scene in scenes] != expected:
            raise ValueError("script Scene IDs must remain contiguous and ordered")
        forbidden = ("```", "**", "[SFX", "[VISUAL", "# ")
        for scene in scenes:
            if any(marker.lower() in scene.spoken_text.lower() for marker in forbidden):
                raise ValueError(f"{scene.scene_id} contains non-spoken formatting or directions")
        if scenes:
            scenes[-1].pause_after_seconds = 0
        return scenes


class ReviewFinding(ContractModel):
    finding_id: str
    severity: Literal["info", "minor", "major", "blocking"]
    scene_id: str | None = None
    evidence: str
    recommendation: str


class ReviewReport(VersionedContract):
    review_type: Literal["story", "spoken", "constraints", "factual"]
    passed: bool
    findings: list[ReviewFinding] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_findings(self) -> "ReviewReport":
        ids = [finding.finding_id for finding in self.findings]
        if any(not item for item in ids) or len(ids) != len(set(ids)):
            raise ValueError("review Finding IDs must be nonempty and unique")
        if self.passed and any(finding.severity == "blocking" for finding in self.findings):
            raise ValueError("a review with blocking findings cannot pass")
        return self


class RevisionDisposition(ContractModel):
    finding_id: str
    disposition: Literal["applied", "partially_applied", "rejected"]
    explanation: str


class RevisedScript(VersionedContract):
    script: NarrationScript
    dispositions: list[RevisionDisposition] = Field(default_factory=list)

    @field_validator("dispositions")
    @classmethod
    def unique_dispositions(cls, values: list[RevisionDisposition]) -> list[RevisionDisposition]:
        ids = [item.finding_id for item in values]
        if any(not item for item in ids) or len(ids) != len(set(ids)):
            raise ValueError("revision dispositions must have nonempty unique Finding IDs")
        return values


class MediaReference(ContractModel):
    path: str
    sha256: str
    mime_type: str


class WordTiming(ContractModel):
    text: str
    start_seconds: Annotated[FiniteFloat, Field(ge=0)]
    end_seconds: Annotated[FiniteFloat, Field(ge=0)]
    confidence: Annotated[FiniteFloat, Field(ge=0, le=1)] | None = None

    @model_validator(mode="after")
    def validate_span(self) -> "WordTiming":
        if self.end_seconds < self.start_seconds:
            raise ValueError("word end must not precede its start")
        return self


class SpeechAsset(VersionedContract):
    scene_id: str
    audio: MediaReference
    duration_seconds: Annotated[FiniteFloat, Field(gt=0)]
    sample_rate: PositiveInt
    channels: PositiveInt
    word_timings: list[WordTiming] = Field(default_factory=list)
    timing_precision: Literal["none", "character", "word", "aligned"] = "none"
    provider_request_id: str = ""


class TimelineScene(ContractModel):
    scene_id: str
    audio: MediaReference
    start_seconds: Annotated[FiniteFloat, Field(ge=0)]
    speech_end_seconds: Annotated[FiniteFloat, Field(ge=0)]
    end_seconds: Annotated[FiniteFloat, Field(ge=0)]
    words: list[WordTiming] = Field(default_factory=list)


class NarrationTimeline(VersionedContract):
    narration_audio: MediaReference
    duration_seconds: Annotated[FiniteFloat, Field(gt=0)]
    delivery_duration_seconds: Annotated[FiniteFloat, Field(gt=0)]
    fps: PositiveInt = 30
    scenes: list[TimelineScene]

    @model_validator(mode="after")
    def validate_timeline(self) -> "NarrationTimeline":
        previous = 0.0
        for scene in self.scenes:
            if abs(scene.start_seconds - previous) > 0.002:
                raise ValueError("Timeline Scenes must be contiguous")
            if not (scene.start_seconds <= scene.speech_end_seconds <= scene.end_seconds):
                raise ValueError("invalid Scene timing span")
            previous = scene.end_seconds
        if abs(previous - self.duration_seconds) > 0.002:
            raise ValueError("Timeline duration must equal the final Scene end")
        if self.delivery_duration_seconds + 1e-6 < self.duration_seconds:
            raise ValueError("delivery duration cannot cut narration")
        return self


class CharacterIdentity(ContractModel):
    character_id: str
    name: str
    signature_traits: list[str]
    color_anchors: list[str] = Field(default_factory=list)
    recurring_props: list[str] = Field(default_factory=list)


class StyleProfile(ContractModel):
    style_id: str
    description: str
    palette: list[str]
    line_style: str
    background: str
    must_avoid: list[str]


class VisualBrief(ContractModel):
    scene_id: str
    story_moment: str
    subjects: list[str]
    action: str
    emotion: str
    environment: str
    composition: str
    must_show: list[str]
    must_avoid: list[str]
    character_ids: list[str] = Field(default_factory=list)


class VisualPlan(VersionedContract):
    style_profile: StyleProfile
    characters: list[CharacterIdentity] = Field(default_factory=list)
    scenes: list[VisualBrief]

    @model_validator(mode="after")
    def validate_visual_plan(self) -> "VisualPlan":
        expected = [f"scene-{index:03d}" for index in range(1, len(self.scenes) + 1)]
        if not self.scenes or [scene.scene_id for scene in self.scenes] != expected:
            raise ValueError("Visual Plan Scene IDs must be contiguous and ordered")
        character_ids = [character.character_id for character in self.characters]
        if len(set(character_ids)) != len(character_ids):
            raise ValueError("Character Identity IDs must be unique")
        unknown = sorted(
            {
                character_id
                for scene in self.scenes
                for character_id in scene.character_ids
                if character_id not in set(character_ids)
            }
        )
        if unknown:
            raise ValueError(f"Visual Briefs reference unknown Character IDs: {', '.join(unknown)}")
        return self


class ImageGenerationSettings(ContractModel):
    inference_steps: Annotated[int, Field(ge=1, le=100)] | None = None
    guidance_scale: Annotated[FiniteFloat, Field(ge=0, le=30)] | None = None
    moderation: Literal["auto", "low"] | None = None
    output_format: Literal["png", "jpeg", "webp"] = "png"
    aspect_ratio: Literal["16:9"] = "16:9"
    image_size: Literal["1K", "2K", "4K"] | None = None
    cpu_offload: bool | None = None


class ImageRequest(VersionedContract):
    scene_id: str
    target_backend_id: str
    prompt: Annotated[str, Field(min_length=1, max_length=12000)]
    negative_prompt: str = ""
    width: PositiveInt
    height: PositiveInt
    quality: Literal["low", "medium", "high"] = "medium"
    seed: int | None = None
    reference_paths: list[str] = Field(default_factory=list)
    settings: ImageGenerationSettings = Field(default_factory=ImageGenerationSettings)

    @model_validator(mode="after")
    def validate_image_request(self) -> "ImageRequest":
        if abs(self.width / self.height - 16 / 9) > 0.002:
            raise ValueError("image dimensions must use a 16:9 aspect ratio")
        if len(set(self.reference_paths)) != len(self.reference_paths):
            raise ValueError("image reference paths must be unique")
        return self


class ImageAsset(VersionedContract):
    scene_id: str
    image: MediaReference
    width: PositiveInt
    height: PositiveInt
    generation_settings: dict[str, Any] = Field(default_factory=dict)
    provider_request_id: str = ""


class VisualScore(ContractModel):
    subject_action: Annotated[int, Field(ge=1, le=5)]
    style_match: Annotated[int, Field(ge=1, le=5)]
    identity: Annotated[int, Field(ge=1, le=5)]
    composition: Annotated[int, Field(ge=1, le=5)]
    text_logo_free: Annotated[int, Field(ge=1, le=5)]
    audience_safety: Annotated[int, Field(ge=1, le=5)]


class VisualReviewItem(ContractModel):
    scene_id: str
    passed: bool
    hard_failure: bool = False
    scores: VisualScore
    failures: list[str] = Field(default_factory=list)
    regeneration_instruction: str = ""

    @model_validator(mode="after")
    def validate_result(self) -> "VisualReviewItem":
        minimum_score = min(self.scores.model_dump().values())
        if self.passed and (
            self.hard_failure or self.failures or self.regeneration_instruction or minimum_score < 4
        ):
            raise ValueError("a passed Visual Review cannot contain failures or a score below four")
        if not self.passed and not self.failures:
            raise ValueError("a failed Visual Review requires at least one explicit failure")
        return self


class VisualReviewReport(VersionedContract):
    items: list[VisualReviewItem]
    pass_number: Annotated[int, Field(ge=1, le=2)] = 1


class CaptionTrack(VersionedContract):
    language: OutputLanguage
    words: list[WordTiming]
    reconciliation_coverage: Annotated[FiniteFloat, Field(ge=0, le=1)] = 1.0

    @field_validator("words")
    @classmethod
    def monotonic_words(cls, words: list[WordTiming]) -> list[WordTiming]:
        previous = 0.0
        for word in words:
            if word.start_seconds + 0.002 < previous:
                raise ValueError("caption words must be monotonic")
            previous = word.end_seconds
        return words


class MusicSection(ContractModel):
    start_seconds: Annotated[FiniteFloat, Field(ge=0)]
    end_seconds: Annotated[FiniteFloat, Field(gt=0)]
    mood: str
    energy: str

    @model_validator(mode="after")
    def validate_span(self) -> "MusicSection":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("music section end must follow its start")
        return self


class MusicBrief(VersionedContract):
    prompt: Annotated[str, Field(min_length=1, max_length=512)]
    requested_duration_seconds: Annotated[FiniteFloat, Field(gt=0)]
    tempo_range_bpm: str
    instrumentation: list[str]
    texture: str
    exclusions: list[str]
    sections: list[MusicSection]
    seamless_loop_preferred: bool = False

    @model_validator(mode="after")
    def validate_sections(self) -> "MusicBrief":
        if not self.sections:
            raise ValueError("music brief requires at least one timed section")
        previous = 0.0
        for section in self.sections:
            if abs(section.start_seconds - previous) > 0.02:
                raise ValueError("music sections must be contiguous and start at zero")
            previous = section.end_seconds
        if abs(previous - self.requested_duration_seconds) > 0.02:
            raise ValueError("music sections must end at the requested duration")
        return self


class MusicAsset(VersionedContract):
    audio: MediaReference
    duration_seconds: Annotated[FiniteFloat, Field(gt=0)]
    looped: bool = False
    provider_request_id: str = ""


class RenderScene(ContractModel):
    scene_id: str
    image_path: str
    start_seconds: Annotated[FiniteFloat, Field(ge=0)]
    end_seconds: Annotated[FiniteFloat, Field(gt=0)]

    @model_validator(mode="after")
    def validate_span(self) -> "RenderScene":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("render Scene end must follow its start")
        return self


class RenderPlan(VersionedContract):
    scenes: list[RenderScene]
    narration_path: str
    music_path: str | None = None
    caption_srt_path: str | None = None
    caption_ass_path: str | None = None
    caption_language: OutputLanguage | None = None
    width: PositiveInt
    height: PositiveInt
    fps: PositiveInt
    duration_seconds: Annotated[FiniteFloat, Field(gt=0)]
    video_codec: Literal["libx264"] = "libx264"
    audio_codec: Literal["aac"] = "aac"
    caption_mode: Literal["none", "selectable", "selectable_and_burned"] = "none"

    @model_validator(mode="after")
    def validate_plan(self) -> "RenderPlan":
        if not self.scenes:
            raise ValueError("Render Plan requires at least one Scene")
        previous = 0.0
        for scene in self.scenes:
            if abs(scene.start_seconds - previous) > 0.002:
                raise ValueError("Render Plan Scenes must be contiguous and start at zero")
            previous = scene.end_seconds
        if abs(previous - self.duration_seconds) > 0.002:
            raise ValueError("Render Plan duration must equal its final Scene end")
        if self.caption_mode == "none" and (self.caption_srt_path or self.caption_ass_path):
            raise ValueError("caption paths require a caption mode")
        if self.caption_mode != "none" and not self.caption_srt_path:
            raise ValueError("selectable captions require an SRT path")
        if self.caption_mode != "none" and self.caption_language is None:
            raise ValueError("selectable captions require a caption language")
        if self.caption_mode == "none" and self.caption_language is not None:
            raise ValueError("caption language requires a caption mode")
        if self.caption_mode == "selectable_and_burned" and not self.caption_ass_path:
            raise ValueError("burned captions require an ASS path")
        return self


class QCCheck(ContractModel):
    name: str
    passed: bool
    detail: str


class DeliveryFile(ContractModel):
    role: str
    media: MediaReference


class DeliveryManifest(VersionedContract):
    run_id: str
    outputs: list[DeliveryFile]
    duration_seconds: FiniteFloat
    checks: list[QCCheck]
    warnings: list[str] = Field(default_factory=list)
    completed_at: datetime = Field(default_factory=utc_now)


class BackendDescriptor(VersionedContract):
    backend_id: str
    provider: str
    model_id: str
    revision: str
    protocols: set[ProtocolName]
    cloud: bool
    runner: Literal["in_process", "native", "wsl", "either"] = "in_process"
    languages: set[OutputLanguage]
    required_env: list[str] = Field(default_factory=list)
    required_assets: list[str] = Field(default_factory=list)
    supports_vision: bool = False
    supports_word_timing: bool = False
    supports_voice_cloning: bool = False
    supports_reference_images: bool = False
    max_duration_seconds: Annotated[FiniteFloat, Field(gt=0)] | None = None
    supports_looping: bool = False
    exclusive_gpu: bool = False
    reservation_usd: Annotated[FiniteFloat, Field(ge=0)] = 0
    license_name: str
    allowed_usage_purposes: set[str] = Field(default_factory=lambda: {"personal_noncommercial"})
    notes: str = ""


class ProbeItem(ContractModel):
    name: str
    ready: bool
    detail: str
    action: str | None = None


class ProbeReport(VersionedContract):
    backend_id: str
    ready: bool
    items: list[ProbeItem]
    probed_at: datetime = Field(default_factory=utc_now)


class CostEstimate(VersionedContract):
    estimated_usd: Annotated[FiniteFloat, Field(ge=0)]
    already_reserved_usd: Annotated[FiniteFloat, Field(ge=0)] = 0
    projected_total_usd: Annotated[FiniteFloat, Field(ge=0)]
    ceiling_usd: Annotated[FiniteFloat, Field(ge=0)]
    scene_count: PositiveInt
    line_items: dict[str, Annotated[FiniteFloat, Field(ge=0)]] = Field(default_factory=dict)
    basis: str


class PreflightReport(VersionedContract):
    ready: bool
    profile: str
    output_language: OutputLanguage
    active_backends: list[str]
    checks: list[ProbeItem]
    backend_reports: list[ProbeReport]
    cost: CostEstimate
    live: bool = False
    warnings: list[str] = Field(default_factory=list)
    probed_at: datetime = Field(default_factory=utc_now)


class UsageRecord(VersionedContract):
    task_id: str
    backend_id: str
    provider_request_id: str = ""
    input_units: Annotated[FiniteFloat, Field(ge=0)] = 0
    output_units: Annotated[FiniteFloat, Field(ge=0)] = 0
    reserved_usd: Annotated[FiniteFloat, Field(ge=0)] = 0
    actual_usd: Annotated[FiniteFloat, Field(ge=0)] | None = None
    elapsed_seconds: Annotated[FiniteFloat, Field(ge=0)] = 0
    peak_vram_mb: Annotated[FiniteFloat, Field(ge=0)] | None = None
    warnings: list[str] = Field(default_factory=list)


class StageRecord(ContractModel):
    stage: str
    status: Literal["pending", "running", "complete", "failed", "stopped"] = "pending"
    attempt: PositiveInt = 1
    input_hash: str = ""
    config_hash: str = ""
    backend_id: str = ""
    backend_revision: str = ""
    prompt_version: str = ""
    schema_hash: str = ""
    output_paths: list[str] = Field(default_factory=list)
    output_hashes: dict[str, str] = Field(default_factory=dict)
    item_ids: list[str] = Field(default_factory=list)
    usage: list[UsageRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: dict[str, Any] | None = None


class RunManifest(VersionedContract):
    run_id: str
    status: Literal["created", "running", "stopped", "failed", "complete"] = "created"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    parent_run_id: str | None = None
    fork_stage: str | None = None
    config_hash: str
    brief_hash: str
    frozen_assets_hash: str
    stages: dict[str, StageRecord] = Field(default_factory=dict)
    reserved_cost_usd: FiniteFloat = 0
    cost_reservations: list[UsageRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SearchRequest(ContractModel):
    query: str
    max_results: Annotated[int, Field(ge=1, le=20)] = 5
    language: OutputLanguage


class SearchResult(ContractModel):
    query: str
    sources: list[ResearchSource]
    provider_request_id: str = ""
    usage: UsageRecord | None = None


class SourceFetchRequest(ContractModel):
    source: ResearchSource
    max_bytes: Annotated[int, Field(ge=1024, le=5_000_000)] = 1_000_000
    max_text_chars: Annotated[int, Field(ge=100, le=100_000)] = 12_000
    timeout_seconds: Annotated[FiniteFloat, Field(gt=0, le=60)] = 12
    redirect_limit: Annotated[int, Field(ge=0, le=10)] = 4


class SourceDocument(ContractModel):
    source_id: str
    final_url: str
    title: str
    text: str
    content_sha256: str
    mime_type: str


class StructuredTextRequest(ContractModel):
    task_id: str
    instructions: str
    input_data: dict[str, Any]
    output_schema: dict[str, Any]
    output_language: OutputLanguage
    max_output_tokens: Annotated[int, Field(ge=128, le=32000)] = 8000
    media_inputs: list[str] = Field(default_factory=list)

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, task_id: str) -> str:
        if task_id not in TASK_IDS:
            raise ValueError(f"unknown Workflow Task ID: {task_id}")
        return task_id


class StructuredTextResult(ContractModel):
    data: dict[str, Any]
    raw_response: dict[str, Any] = Field(default_factory=dict)
    provider_request_id: str = ""
    usage: UsageRecord | None = None


class SpeechRequest(ContractModel):
    scene_id: str
    text: str
    output_language: OutputLanguage
    voice: VoiceSettings
    output_path: str
    preceding_text: str = ""
    following_text: str = ""


class SpeechResult(ContractModel):
    asset: SpeechAsset
    usage: UsageRecord | None = None


class AlignmentRequest(ContractModel):
    scene_id: str
    transcript: str
    audio_path: str
    output_language: OutputLanguage


class AlignmentResult(ContractModel):
    recognized_words: list[WordTiming]
    provider_request_id: str = ""
    usage: UsageRecord | None = None


class ImageResult(ContractModel):
    asset: ImageAsset
    usage: UsageRecord | None = None


class MusicRequest(ContractModel):
    brief: MusicBrief
    output_path: str
    output_language: OutputLanguage


class MusicResult(ContractModel):
    asset: MusicAsset
    usage: UsageRecord | None = None


def contract_schema(model: type[BaseModel]) -> dict[str, Any]:
    return model.model_json_schema(mode="validation")


def ensure_workspace_path(path: str | Path, roots: list[Path]) -> Path:
    candidate = Path(path).resolve()
    for root in roots:
        try:
            candidate.relative_to(root.resolve())
            return candidate
        except ValueError:
            continue
    raise ValueError(f"path is outside allowed workspace roots: {candidate}")
