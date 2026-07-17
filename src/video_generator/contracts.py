from __future__ import annotations

import ipaddress
import math
import re
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, PositiveInt, field_validator, model_validator
from pydantic_core import PydanticCustomError


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


class ContentFormat(StrEnum):
    NARRATIVE = "narrative"
    EXPLAINER = "explainer"
    MYTHBUSTER = "mythbuster"


class NarrationPace(StrEnum):
    SLOW = "slow"
    STANDARD = "standard"
    FAST = "fast"


class VisualShotMode(StrEnum):
    SCENE_LOCKED = "scene_locked"
    CADENCED = "cadenced"


class VideoStyle(StrEnum):
    STILL_IMAGE = "still_image"
    REMOTION_EXPLAINER = "remotion_explainer"


class VideoOrientation(StrEnum):
    LANDSCAPE = "landscape"
    PORTRAIT = "portrait"

    @property
    def aspect_ratio(self) -> Literal["16:9", "9:16"]:
        return "9:16" if self is VideoOrientation.PORTRAIT else "16:9"


class RemotionAssetPolicy(StrEnum):
    LOCAL_ONLY = "local_only"
    STOCK_PREFERRED = "stock_preferred"


MAX_CADENCED_SHOTS = 72


_SPOKEN_TEXT_HOST_FIELD_PATTERN = re.compile(
    r"""
    (?:^|[.!?]\s+)
    [\"']?
    (?P<field>
        pause_after_seconds
        | schema_version
        | scene_id
        | spoken_text
        | minimum_word_count
        | target_word_count
        | maximum_word_count
        | count_method
        | draft_strategy
        | revision_strategy
        | repair_strategy
    )
    [\"']?
    (?:\s*[:=]\s*|\s+)
    (?:
        [-+]?(?:\d+(?:\.\d+)?|\.\d+)\b
        | true\b
        | false\b
        | null\b
        | none\b
        | scene-\d+\b
        | single-scene[\w-]*\b
        | len\([^\r\n]{1,200}\)
        | [\"'][^\"'\r\n]{0,200}[\"']
        | \{[^\r\n]{0,200}\}
        | \[[^\r\n]{0,200}\]
    )
    \s*[,;.]?\s*$
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


def validate_spoken_text_only(value: str) -> str:
    match = _SPOKEN_TEXT_HOST_FIELD_PATTERN.search(value)
    if match is not None:
        raise PydanticCustomError(
            "spoken_text_host_field",
            (
                "spoken_text ends with the host schema field fragment '{field}'; "
                "return only words intended for narration"
            ),
            {"field": match.group("field")},
        )
    return value


def estimated_cadenced_shot_count(
    duration_seconds: float,
    visual_target_seconds: float,
    shot_target_seconds: float,
) -> int:
    scene_count = max(1, math.ceil(duration_seconds / visual_target_seconds)) + 1
    return max(
        scene_count,
        math.ceil(duration_seconds / shot_target_seconds) + math.ceil(scene_count / 2),
    )


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
    "claim_inventory",
    "factual_review",
    "narration_synthesis",
    "duration_repair",
    "caption_alignment",
    "visual_plan",
    "remotion_direction",
    "remotion_asset_select",
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
    "claim_inventory": ProtocolName.STRUCTURED_TEXT,
    "factual_review": ProtocolName.STRUCTURED_TEXT,
    "narration_synthesis": ProtocolName.SPEECH,
    "duration_repair": ProtocolName.STRUCTURED_TEXT,
    "caption_alignment": ProtocolName.ALIGNMENT,
    "visual_plan": ProtocolName.STRUCTURED_TEXT,
    "remotion_direction": ProtocolName.STRUCTURED_TEXT,
    "remotion_asset_select": ProtocolName.STRUCTURED_TEXT,
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
    reference_language: OutputLanguage = OutputLanguage.ENGLISH
    elevenlabs_voice_id: str = ""
    authorization: Literal["self", "explicit_permission"] = "self"


class NarrationDeliverySpec(ContractModel):
    pace: NarrationPace = NarrationPace.STANDARD
    target_words_per_second: Annotated[FiniteFloat, Field(gt=0, le=8)]
    minimum_words_per_second: Annotated[FiniteFloat, Field(gt=0, le=8)]
    maximum_words_per_second: Annotated[FiniteFloat, Field(gt=0, le=8)]
    target_pause_seconds: Annotated[FiniteFloat, Field(ge=0, le=3)]
    maximum_pause_seconds: Annotated[FiniteFloat, Field(ge=0, le=3)]
    tempo_multiplier: Annotated[FiniteFloat, Field(ge=0.75, le=1.35)] = 1.0
    description: Annotated[str, Field(max_length=500)] = ""

    @model_validator(mode="after")
    def validate_ranges(self) -> "NarrationDeliverySpec":
        if self.minimum_words_per_second > self.target_words_per_second:
            raise ValueError("minimum narration rate must not exceed the target rate")
        if self.target_words_per_second > self.maximum_words_per_second:
            raise ValueError("target narration rate must not exceed the maximum rate")
        if self.target_pause_seconds > self.maximum_pause_seconds:
            raise ValueError("target narration pause must not exceed the maximum pause")
        return self


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
    content_format: ContentFormat = ContentFormat.NARRATIVE
    narration_pace: NarrationPace = NarrationPace.STANDARD
    narration_delivery: Annotated[str, Field(max_length=500)] = ""
    audience: Literal["family_safe_general"] = "family_safe_general"
    orientation: VideoOrientation = VideoOrientation.LANDSCAPE
    video_style: VideoStyle = VideoStyle.STILL_IMAGE
    style: Annotated[str, Field(min_length=1, max_length=120)] = "ms_paint_stick"
    style_description: Annotated[str, Field(max_length=1000)] = ""
    motion_style: Literal["static_cuts"] = "static_cuts"
    remotion_asset_policy: RemotionAssetPolicy = RemotionAssetPolicy.STOCK_PREFERRED
    remotion_allow_share_alike: bool = False
    remotion_require_asset_approval: bool = False
    remotion_source_screenshot_hosts: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=253)]], Field(max_length=50)
    ] = Field(default_factory=list)
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
    visual_shot_mode: VisualShotMode = VisualShotMode.SCENE_LOCKED
    shot_target_seconds: Annotated[FiniteFloat, Field(gt=0, le=120)] = 3
    shot_min_seconds: Annotated[FiniteFloat, Field(gt=0, le=120)] = 2
    shot_max_seconds: Annotated[FiniteFloat, Field(gt=0, le=180)] = 5
    music_enabled: bool = False
    captions_enabled: bool = True
    animated_captions: bool = False
    voice: VoiceSettings
    task_overrides: dict[str, str] = Field(default_factory=dict)

    @field_validator("remotion_source_screenshot_hosts")
    @classmethod
    def validate_remotion_source_screenshot_hosts(cls, values: list[str]) -> list[str]:
        normalized = []
        for value in values:
            host = value.strip().rstrip(".").casefold()
            if "://" in host or "/" in host or ":" in host or "." not in host:
                raise ValueError("source screenshot hosts must be DNS hostnames such as wikipedia.org")
            try:
                ipaddress.ip_address(host)
            except ValueError:
                pass
            else:
                raise ValueError("source screenshot hosts must not be IP addresses")
            try:
                host = host.encode("idna").decode("ascii")
            except UnicodeError as exc:
                raise ValueError("source screenshot host is not a valid IDNA hostname") from exc
            labels = host.split(".")
            if len(host) > 253 or any(
                re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label) is None
                for label in labels
            ):
                raise ValueError("source screenshot host is not a valid DNS hostname")
            if host not in normalized:
                normalized.append(host)
        return normalized

    @model_validator(mode="after")
    def validate_combinations(self) -> "RawRunConfig":
        if self.offline and self.remotion_source_screenshot_hosts:
            raise ValueError(
                "offline mode cannot configure remotion_source_screenshot_hosts"
            )
        if self.content_format is ContentFormat.MYTHBUSTER and self.content_mode is not ContentMode.FACTUAL:
            raise ValueError("mythbuster format requires content_mode = 'factual'")
        if self.content_mode is ContentMode.FACTUAL and (
            self.offline or self.research_query_limit == 0 or self.research_source_limit == 0
        ):
            raise ValueError(
                "factual mode requires live bounded research with nonzero query and source limits"
            )
        if self.visual_min_seconds > self.visual_target_seconds:
            raise ValueError("visual_min_seconds must not exceed visual_target_seconds")
        if self.visual_target_seconds > self.visual_max_seconds:
            raise ValueError("visual_target_seconds must not exceed visual_max_seconds")
        if self.shot_min_seconds > self.shot_target_seconds:
            raise ValueError("shot_min_seconds must not exceed shot_target_seconds")
        if self.shot_target_seconds > self.shot_max_seconds:
            raise ValueError("shot_target_seconds must not exceed shot_max_seconds")
        if (
            self.visual_shot_mode is VisualShotMode.CADENCED
            or self.video_style is VideoStyle.REMOTION_EXPLAINER
        ):
            estimated_shots = estimated_cadenced_shot_count(
                float(self.duration_seconds),
                float(self.visual_target_seconds),
                float(self.shot_target_seconds),
            )
            if estimated_shots > MAX_CADENCED_SHOTS:
                raise ValueError(
                    f"cadenced mode estimates {estimated_shots} visual Shots; the current "
                    f"single-plan limit is {MAX_CADENCED_SHOTS}. Increase shot_target_seconds, "
                    "shorten the Run, or use scene_locked mode"
                )
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
    modern_anchor: Annotated[str, Field(max_length=500)] = ""
    central_question: Annotated[str, Field(max_length=1000)] = ""
    misconception: Annotated[str, Field(max_length=1000)] = ""
    desired_takeaway: Annotated[str, Field(max_length=1000)] = ""

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
    content_format: ContentFormat = ContentFormat.NARRATIVE
    narration_pace: NarrationPace = NarrationPace.STANDARD
    narration_delivery: str = ""
    narration_delivery_spec: NarrationDeliverySpec | None = None
    audience: str
    orientation: VideoOrientation = VideoOrientation.LANDSCAPE
    video_style: VideoStyle = VideoStyle.STILL_IMAGE
    style: str
    style_description: str
    motion_style: str
    remotion_asset_policy: RemotionAssetPolicy = RemotionAssetPolicy.STOCK_PREFERRED
    remotion_allow_share_alike: bool = False
    remotion_require_asset_approval: bool = False
    remotion_source_screenshot_hosts: list[str] = Field(default_factory=list)
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
    visual_shot_mode: VisualShotMode = VisualShotMode.SCENE_LOCKED
    shot_target_seconds: FiniteFloat = 3
    shot_min_seconds: FiniteFloat = 2
    shot_max_seconds: FiniteFloat = 5
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


EvidenceStatement = Annotated[str, Field(min_length=1, max_length=600)]
EvidenceLimitation = Annotated[str, Field(min_length=1, max_length=240)]
EvidenceLimitations = Annotated[list[EvidenceLimitation], Field(max_length=4)]


class EvidenceRecord(ContractModel):
    evidence_id: Annotated[str, Field(min_length=1, max_length=120)]
    supported_statement: EvidenceStatement
    source_ids: Annotated[list[str], Field(min_length=1, max_length=20)]
    confidence: Literal["low", "medium", "high"]
    time_sensitive: bool = False
    limitations: EvidenceLimitations = Field(default_factory=list)


class ResearchFindingDraft(ContractModel):
    summary: Annotated[str, Field(min_length=1, max_length=4000)]
    source_ids: Annotated[list[str], Field(min_length=1, max_length=20)]


class EvidenceRecordDraft(ContractModel):
    supported_statement: EvidenceStatement
    source_ids: Annotated[list[str], Field(min_length=1, max_length=20)]
    confidence: Literal["low", "medium", "high"]
    time_sensitive: bool = False
    limitations: EvidenceLimitations = Field(default_factory=list)


class ResearchSynthesis(ContractModel):
    findings: Annotated[list[ResearchFindingDraft], Field(max_length=100)] = Field(
        default_factory=list
    )
    motifs: list[str] = Field(default_factory=list)
    setting_details: list[str] = Field(default_factory=list)
    vocabulary: list[str] = Field(default_factory=list)
    cultural_cautions: list[str] = Field(default_factory=list)
    cliches_to_avoid: list[str] = Field(default_factory=list)


class FactualResearchSynthesis(ContractModel):
    evidence: Annotated[list[EvidenceRecordDraft], Field(min_length=1, max_length=12)]


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


class FactualResearchPack(ResearchPack):
    evidence: Annotated[list[EvidenceRecord], Field(min_length=1, max_length=12)]

    @model_validator(mode="after")
    def validate_evidence_references(self) -> "FactualResearchPack":
        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("Evidence IDs must be unique")
        source_ids = {source.source_id for source in self.sources}
        unknown = sorted(
            {
                source_id
                for item in self.evidence
                for source_id in item.source_ids
                if source_id not in source_ids
            }
        )
        if unknown:
            raise ValueError(f"Evidence references unknown Source IDs: {', '.join(unknown)}")
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


class ExplainerCandidate(ContractModel):
    candidate_id: str
    title: str
    modern_anchor: str
    central_question: str
    misconception: str = ""
    thesis: str
    evidence_ids: list[str] = Field(default_factory=list)
    evidence_ladder: list[str] = Field(default_factory=list)
    human_angle: str
    landing_direction: str
    visual_opportunities: list[str] = Field(default_factory=list)
    accuracy_risks: list[str] = Field(default_factory=list)
    duration_fit: str


class ExplainerCandidateSet(VersionedContract):
    candidates: list[ExplainerCandidate]

    @field_validator("candidates")
    @classmethod
    def unique_candidates(cls, values: list[ExplainerCandidate]) -> list[ExplainerCandidate]:
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


class ExplainerCandidateScore(ContractModel):
    candidate_id: str
    duration_fit: Annotated[int, Field(ge=1, le=5)]
    hook_strength: Annotated[int, Field(ge=1, le=5)]
    evidence_strength: Annotated[int, Field(ge=1, le=5)]
    progression: Annotated[int, Field(ge=1, le=5)]
    visual_strength: Annotated[int, Field(ge=1, le=5)]
    spoken_suitability: Annotated[int, Field(ge=1, le=5)]
    audience_fit: Annotated[int, Field(ge=1, le=5)]
    rationale: str


class ExplainerSelectionReport(VersionedContract):
    scores: list[ExplainerCandidateScore]
    chosen_candidate_id: str
    rationale: str

    @model_validator(mode="after")
    def validate_selection(self) -> "ExplainerSelectionReport":
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


class ExplainerOutlineScene(ContractModel):
    scene_id: str
    arc_role: Literal[
        "modern_hook",
        "extreme_contrast",
        "question",
        "misconception",
        "correction",
        "evidence",
        "human_tangent",
        "synthesis",
        "landing",
    ]
    purpose: str
    key_point: str
    evidence_ids: list[str] = Field(default_factory=list)
    visual_opportunity: str
    provisional_seconds: Annotated[FiniteFloat, Field(gt=0)]
    continuity_obligations: list[str] = Field(default_factory=list)


class ExplainerOutline(VersionedContract):
    title: str
    thesis: str
    modern_anchor: str
    misconception: str = ""
    landing_callback: str
    scenes: list[ExplainerOutlineScene]

    @field_validator("scenes")
    @classmethod
    def validate_scene_ids(cls, scenes: list[ExplainerOutlineScene]) -> list[ExplainerOutlineScene]:
        expected = [f"scene-{index:03d}" for index in range(1, len(scenes) + 1)]
        actual = [scene.scene_id for scene in scenes]
        if not scenes or actual != expected:
            raise ValueError(f"scene IDs must be contiguous and ordered: {expected}")
        return scenes


class ScriptScene(ContractModel):
    scene_id: str
    spoken_text: Annotated[str, Field(min_length=1, max_length=10000)]
    pause_after_seconds: Annotated[FiniteFloat, Field(ge=0, le=3.25)] = 0.15

    @field_validator("spoken_text")
    @classmethod
    def validate_spoken_text(cls, value: str) -> str:
        return validate_spoken_text_only(value)


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


class BriefConstraintAssessment(ContractModel):
    satisfied: bool
    scene_id: str | None = None
    evidence: Annotated[str, Field(max_length=2000)] = ""
    recommendation: Annotated[str, Field(max_length=2000)] = ""

    @model_validator(mode="after")
    def validate_unsatisfied_assessment(self) -> "BriefConstraintAssessment":
        if not self.satisfied and (
            not self.scene_id
            or not self.evidence.strip()
            or not self.recommendation.strip()
        ):
            raise ValueError(
                "an unsatisfied brief constraint requires a Scene ID, evidence, and recommendation"
            )
        return self


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


class ScriptClaim(ContractModel):
    claim_id: Annotated[str, Field(min_length=1, max_length=120)]
    scene_id: Annotated[str, Field(pattern=r"^scene-\d{3}$")]
    exact_text: Annotated[str, Field(min_length=1, max_length=4000)]
    evidence_ids: Annotated[list[str], Field(max_length=20)] = Field(default_factory=list)
    qualification: Annotated[str, Field(max_length=2000)] = ""


class ExtractedClaim(ContractModel):
    exact_text: Annotated[str, Field(min_length=1, max_length=4000)]
    qualification: Annotated[str, Field(max_length=2000)] = ""


class SceneClaimExtraction(ContractModel):
    claims: Annotated[list[ExtractedClaim], Field(max_length=20)] = Field(default_factory=list)


class SceneClaimCoverage(ContractModel):
    missing_claims: Annotated[list[ExtractedClaim], Field(max_length=20)] = Field(
        default_factory=list
    )


class ClaimInventory(VersionedContract):
    claims: Annotated[list[ScriptClaim], Field(min_length=1)]
    coverage_notes: Annotated[str, Field(max_length=4000)] = ""

    @field_validator("claims")
    @classmethod
    def unique_claims(cls, values: list[ScriptClaim]) -> list[ScriptClaim]:
        ids = [item.claim_id for item in values]
        if any(not item for item in ids) or len(ids) != len(set(ids)):
            raise ValueError("claim IDs must be nonempty and unique")
        return values


class FactualClaimReview(ContractModel):
    claim_id: str
    verdict: Literal[
        "supported",
        "needs_qualification",
        "unsupported",
        "not_a_factual_claim",
    ]
    evidence_ids: list[str] = Field(default_factory=list)
    rationale: Annotated[str, Field(min_length=1, max_length=4000)]


class FactualReviewReport(VersionedContract):
    passed: bool
    claims: list[FactualClaimReview]
    uncovered_claims: list[str] = Field(default_factory=list)
    summary: Annotated[str, Field(max_length=4000)] = ""

    @model_validator(mode="after")
    def validate_review(self) -> "FactualReviewReport":
        ids = [item.claim_id for item in self.claims]
        if len(ids) != len(set(ids)):
            raise ValueError("factual review Claim IDs must be unique")
        accepted_verdicts = {"supported", "not_a_factual_claim"}
        if self.passed and (
            self.uncovered_claims
            or any(item.verdict not in accepted_verdicts for item in self.claims)
        ):
            raise ValueError("a passed factual review cannot contain uncovered or unsupported claims")
        return self


class FactualRevisedScript(RevisedScript):
    claim_inventory: ClaimInventory
    factual_review: FactualReviewReport

    @model_validator(mode="after")
    def validate_claim_review_coverage(self) -> "FactualRevisedScript":
        inventory_ids = {item.claim_id for item in self.claim_inventory.claims}
        review_ids = {item.claim_id for item in self.factual_review.claims}
        if inventory_ids != review_ids:
            raise ValueError("factual review must cover every inventoried claim exactly once")
        return self


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
    body_form: str = ""
    proportions: list[str] = Field(default_factory=list)
    face_and_markings: list[str] = Field(default_factory=list)
    wardrobe: list[str] = Field(default_factory=list)
    identity_constraints: list[str] = Field(default_factory=list)


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
    continuity_from_previous: list[str] = Field(default_factory=list)
    state_after_scene: list[str] = Field(default_factory=list)
    identity_requirements: list[str] = Field(default_factory=list)
    persistent_elements: list[str] = Field(default_factory=list)


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


class TimedVisualShot(VisualBrief):
    shot_id: Annotated[str, Field(pattern=r"^shot-\d{3,}$")]
    narration_excerpt: Annotated[str, Field(min_length=1, max_length=4000)]
    start_seconds: Annotated[FiniteFloat, Field(ge=0)]
    end_seconds: Annotated[FiniteFloat, Field(gt=0)]

    @model_validator(mode="after")
    def validate_span(self) -> "TimedVisualShot":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("visual Shot end must follow its start")
        return self


class TimedVisualPlan(VersionedContract):
    style_profile: StyleProfile
    characters: list[CharacterIdentity] = Field(default_factory=list)
    duration_seconds: Annotated[FiniteFloat, Field(gt=0)]
    shots: list[TimedVisualShot]

    @model_validator(mode="after")
    def validate_visual_plan(self) -> "TimedVisualPlan":
        expected = [f"shot-{index:03d}" for index in range(1, len(self.shots) + 1)]
        if not self.shots or [shot.shot_id for shot in self.shots] != expected:
            raise ValueError("Timed Visual Plan Shot IDs must be contiguous and ordered")
        character_ids = [character.character_id for character in self.characters]
        if len(set(character_ids)) != len(character_ids):
            raise ValueError("Character Identity IDs must be unique")
        known_characters = set(character_ids)
        unknown = sorted(
            {
                character_id
                for shot in self.shots
                for character_id in shot.character_ids
                if character_id not in known_characters
            }
        )
        if unknown:
            raise ValueError(f"Visual Shots reference unknown Character IDs: {', '.join(unknown)}")
        previous = 0.0
        for shot in self.shots:
            if abs(shot.start_seconds - previous) > 0.002:
                raise ValueError("Timed Visual Plan Shots must be contiguous and start at zero")
            previous = shot.end_seconds
        if abs(previous - self.duration_seconds) > 0.002:
            raise ValueError("Timed Visual Plan duration must equal its final Shot end")
        return self


class RemotionTemplate(StrEnum):
    KINETIC_HOOK = "kinetic_hook"
    HEADLINE_ZOOM = "headline_zoom"
    SOURCE_SCREENSHOT = "source_screenshot"
    CODE_REVEAL = "code_reveal"
    DIAGRAM_FLOW = "diagram_flow"
    COMPARISON_SPLIT = "comparison_split"
    MEME_CUTAWAY = "meme_cutaway"
    CONCLUSION = "conclusion"


class RemotionAssetKind(StrEnum):
    NONE = "none"
    STOCK_IMAGE = "stock_image"
    STOCK_VIDEO = "stock_video"
    GIF = "gif"
    MEME = "meme"
    SOURCE_SCREENSHOT = "source_screenshot"
    GENERATED_IMAGE = "generated_image"


class RemotionMotionPreset(StrEnum):
    PUNCH_IN = "punch_in"
    SLIDE_UP = "slide_up"
    PAN = "pan"
    TYPE_ON = "type_on"
    BUILD = "build"
    HOLD = "hold"


class RemotionSfxPreset(StrEnum):
    NONE = "none"
    CLICK = "click"
    POP = "pop"
    WHOOSH = "whoosh"


class RemotionTransitionPreset(StrEnum):
    HARD_CUT = "hard_cut"
    SECTION_WIPE = "section_wipe"


class RemotionShotDirection(ContractModel):
    template: RemotionTemplate
    headline: Annotated[str, Field(min_length=1, max_length=120)]
    supporting_text: Annotated[str, Field(max_length=240)] = ""
    body_lines: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=120)]],
        Field(max_length=8),
    ] = Field(default_factory=list)
    asset_kind: RemotionAssetKind = RemotionAssetKind.NONE
    asset_query: Annotated[str, Field(max_length=180)] = ""
    sfx: RemotionSfxPreset = RemotionSfxPreset.NONE

    @model_validator(mode="after")
    def validate_template_content(self) -> "RemotionShotDirection":
        if len(self.headline) > 80 or len(self.supporting_text) > 160:
            raise ValueError("Remotion headline/supporting text exceeds the fixed layout budget")
        if self.template in {RemotionTemplate.CODE_REVEAL, RemotionTemplate.DIAGRAM_FLOW} and len(
            self.body_lines
        ) < 2:
            raise ValueError(f"{self.template.value} requires at least two body lines")
        if self.template is RemotionTemplate.COMPARISON_SPLIT and len(self.body_lines) != 2:
            raise ValueError("comparison_split requires exactly two body lines")
        if self.template is RemotionTemplate.DIAGRAM_FLOW and (
            len(self.body_lines) > 5 or any(len(line) > 32 for line in self.body_lines)
        ):
            raise ValueError("diagram_flow supports two to five labels of at most 32 characters")
        if self.template is RemotionTemplate.CODE_REVEAL and any(
            len(line) > 70 for line in self.body_lines
        ):
            raise ValueError("code_reveal lines may contain at most 70 characters")
        if self.template is RemotionTemplate.COMPARISON_SPLIT and any(
            len(line) > 60 for line in self.body_lines
        ):
            raise ValueError("comparison_split statements may contain at most 60 characters")
        if self.template not in {
            RemotionTemplate.CODE_REVEAL,
            RemotionTemplate.DIAGRAM_FLOW,
            RemotionTemplate.COMPARISON_SPLIT,
        } and self.body_lines:
            raise ValueError(f"{self.template.value} does not render body_lines")
        if self.template is RemotionTemplate.SOURCE_SCREENSHOT and (
            self.asset_kind is not RemotionAssetKind.SOURCE_SCREENSHOT
        ):
            raise ValueError("source_screenshot requires a source_screenshot asset")
        if self.asset_kind is RemotionAssetKind.SOURCE_SCREENSHOT and (
            self.template is not RemotionTemplate.SOURCE_SCREENSHOT
        ):
            raise ValueError("source_screenshot assets require the source_screenshot template")
        if self.template is RemotionTemplate.MEME_CUTAWAY and self.asset_kind not in {
            RemotionAssetKind.MEME,
            RemotionAssetKind.GIF,
            RemotionAssetKind.STOCK_IMAGE,
        }:
            raise ValueError("meme_cutaway requires a meme, GIF, or stock image")
        if self.template not in {
            RemotionTemplate.KINETIC_HOOK,
            RemotionTemplate.HEADLINE_ZOOM,
            RemotionTemplate.SOURCE_SCREENSHOT,
            RemotionTemplate.MEME_CUTAWAY,
        } and self.asset_kind is not RemotionAssetKind.NONE:
            raise ValueError(f"{self.template.value} does not render an external asset")
        searched_kinds = {
            RemotionAssetKind.STOCK_IMAGE,
            RemotionAssetKind.STOCK_VIDEO,
            RemotionAssetKind.GIF,
            RemotionAssetKind.MEME,
            RemotionAssetKind.GENERATED_IMAGE,
        }
        if self.asset_kind in searched_kinds and not self.asset_query.strip():
            raise ValueError(f"{self.asset_kind.value} requires an English asset query")
        if self.asset_kind in {RemotionAssetKind.NONE, RemotionAssetKind.SOURCE_SCREENSHOT} and (
            self.asset_query.strip()
        ):
            raise ValueError(f"{self.asset_kind.value} must not include an asset query")
        return self


class RemotionAssetChoice(ContractModel):
    candidate_id: Annotated[str, Field(pattern=r"^candidate-\d{3,}$")]


class AnchoredWord(ContractModel):
    word_id: Annotated[str, Field(pattern=r"^word-\d{6,}$")]
    scene_id: Annotated[str, Field(pattern=r"^scene-\d{3,}$")]
    text: Annotated[str, Field(min_length=1, max_length=200)]
    start_seconds: Annotated[FiniteFloat, Field(ge=0)]
    end_seconds: Annotated[FiniteFloat, Field(ge=0)]

    @model_validator(mode="after")
    def validate_span(self) -> "AnchoredWord":
        if self.end_seconds < self.start_seconds:
            raise ValueError("anchored word end must not precede its start")
        return self


class RemotionEditShot(RemotionShotDirection):
    purpose: Annotated[str, Field(min_length=1, max_length=240)]
    motion: RemotionMotionPreset
    transition_in: RemotionTransitionPreset = RemotionTransitionPreset.HARD_CUT
    shot_id: Annotated[str, Field(pattern=r"^shot-\d{3,}$")]
    scene_id: Annotated[str, Field(pattern=r"^scene-\d{3,}$")]
    narration_excerpt: Annotated[str, Field(min_length=1, max_length=4000)]
    start_word_id: Annotated[str, Field(pattern=r"^word-\d{6,}$")]
    end_word_id: Annotated[str, Field(pattern=r"^word-\d{6,}$")]
    start_seconds: Annotated[FiniteFloat, Field(ge=0)]
    end_seconds: Annotated[FiniteFloat, Field(gt=0)]
    start_frame: Annotated[int, Field(ge=0)]
    end_frame: PositiveInt
    source_screenshot_source_ids: Annotated[
        list[Annotated[str, Field(pattern=r"^source-\d{3,}$")]], Field(max_length=8)
    ] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_span(self) -> "RemotionEditShot":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Remotion edit Shot end must follow its start")
        if self.end_frame <= self.start_frame:
            raise ValueError("Remotion edit Shot must contain at least one frame")
        if self.template is RemotionTemplate.CODE_REVEAL and (
            self.motion is not RemotionMotionPreset.TYPE_ON
        ):
            raise ValueError("code_reveal requires the type_on motion preset")
        if self.template is RemotionTemplate.DIAGRAM_FLOW and (
            self.motion is not RemotionMotionPreset.BUILD
        ):
            raise ValueError("diagram_flow requires the build motion preset")
        return self


class RemotionEditPlan(VersionedContract):
    renderer: Literal["remotion"] = "remotion"
    title: Annotated[str, Field(min_length=1, max_length=200)]
    width: PositiveInt
    height: PositiveInt
    fps: PositiveInt
    duration_seconds: Annotated[FiniteFloat, Field(gt=0)]
    duration_frames: PositiveInt
    words: list[AnchoredWord]
    shots: list[RemotionEditShot]

    @model_validator(mode="after")
    def validate_plan(self) -> "RemotionEditPlan":
        if not self.words or not self.shots:
            raise ValueError("Remotion Edit Plan requires words and Shots")
        expected_words = [f"word-{index:06d}" for index in range(1, len(self.words) + 1)]
        if [word.word_id for word in self.words] != expected_words:
            raise ValueError("Remotion word IDs must be contiguous and ordered")
        expected_shots = [f"shot-{index:03d}" for index in range(1, len(self.shots) + 1)]
        if [shot.shot_id for shot in self.shots] != expected_shots:
            raise ValueError("Remotion Shot IDs must be contiguous and ordered")
        word_positions = {word.word_id: index for index, word in enumerate(self.words)}
        previous_frame = 0
        previous_seconds = 0.0
        previous_end_word = -1
        section_transitions = 0
        for index, shot in enumerate(self.shots):
            if shot.start_frame != previous_frame:
                raise ValueError("Remotion Shots must be frame-contiguous and start at zero")
            if abs(shot.start_seconds - previous_seconds) > 0.002:
                raise ValueError("Remotion Shot times must be contiguous and start at zero")
            if shot.start_word_id not in word_positions or shot.end_word_id not in word_positions:
                raise ValueError("Remotion Shot references an unknown word anchor")
            start_word = word_positions[shot.start_word_id]
            end_word = word_positions[shot.end_word_id]
            if start_word > end_word or start_word < previous_end_word:
                raise ValueError("Remotion Shot word anchors are not monotonic")
            previous_end_word = end_word
            if shot.transition_in is RemotionTransitionPreset.SECTION_WIPE:
                if index == 0:
                    raise ValueError("the first Remotion Shot cannot have a section transition")
                section_transitions += 1
            previous_frame = shot.end_frame
            previous_seconds = shot.end_seconds
        if previous_frame != self.duration_frames:
            raise ValueError("Remotion duration must equal the final Shot frame")
        if abs(previous_seconds - self.duration_seconds) > 0.002:
            raise ValueError("Remotion duration must equal the final Shot time")
        if abs(self.duration_frames / self.fps - self.duration_seconds) > 1 / self.fps + 0.002:
            raise ValueError("Remotion frame and second durations disagree")
        if len(self.shots) > 1 and section_transitions != 1:
            raise ValueError("Remotion plans require exactly one section_wipe transition")
        return self


class RemotionAssetRequest(ContractModel):
    asset_id: Annotated[str, Field(pattern=r"^asset-\d{3,}$")]
    shot_id: Annotated[str, Field(pattern=r"^shot-\d{3,}$")]
    kind: RemotionAssetKind
    query: Annotated[str, Field(max_length=180)] = ""
    source_id: Annotated[str, Field(max_length=120)] = ""
    generated_prompt: Annotated[str, Field(max_length=2000)] = ""

    @model_validator(mode="after")
    def validate_request(self) -> "RemotionAssetRequest":
        if self.kind in {RemotionAssetKind.NONE, RemotionAssetKind.SOURCE_SCREENSHOT}:
            if self.kind is RemotionAssetKind.SOURCE_SCREENSHOT and not self.source_id:
                raise ValueError("source screenshot requires a host-owned source ID")
        elif not self.query.strip():
            raise ValueError("searched or generated asset requires a query")
        if self.kind is RemotionAssetKind.GENERATED_IMAGE and not self.generated_prompt.strip():
            raise ValueError("generated image requires a host-compiled prompt")
        return self


class RemotionAssetRequestSet(VersionedContract):
    requests: list[RemotionAssetRequest]

    @model_validator(mode="after")
    def validate_requests(self) -> "RemotionAssetRequestSet":
        asset_ids = [request.asset_id for request in self.requests]
        shot_ids = [request.shot_id for request in self.requests]
        if len(asset_ids) != len(set(asset_ids)) or len(shot_ids) != len(set(shot_ids)):
            raise ValueError("Remotion asset requests must have unique asset and Shot IDs")
        return self


class AssetRights(ContractModel):
    license_id: Annotated[str, Field(min_length=1, max_length=120)]
    license_name: Annotated[str, Field(min_length=1, max_length=240)]
    license_url: Annotated[str, Field(max_length=1000)] = ""
    terms_url: Annotated[str, Field(max_length=1000)] = ""
    attribution_required: bool = False
    attribution_text: Annotated[str, Field(max_length=1000)] = ""
    share_alike: bool = False
    review_status: Literal["approved", "editorial_context"]
    review_reason: Annotated[str, Field(min_length=1, max_length=1000)]

    @model_validator(mode="after")
    def validate_attribution(self) -> "AssetRights":
        if self.attribution_required and not self.attribution_text.strip():
            raise ValueError("rights requiring attribution must include attribution_text")
        return self


class RemotionAsset(VersionedContract):
    asset_id: Annotated[str, Field(pattern=r"^asset-\d{3,}$")]
    shot_id: Annotated[str, Field(pattern=r"^shot-\d{3,}$")]
    provider: Literal["local", "wikimedia", "pexels", "source_screenshot", "generated"]
    provider_asset_id: Annotated[str, Field(max_length=240)] = ""
    media_kind: Literal["image", "video"]
    search_query: Annotated[str, Field(max_length=180)] = ""
    source_page_url: Annotated[str, Field(max_length=2000)] = ""
    creator_name: Annotated[str, Field(max_length=300)] = ""
    creator_url: Annotated[str, Field(max_length=1000)] = ""
    rights: AssetRights
    original: MediaReference
    normalized: MediaReference
    width: PositiveInt
    height: PositiveInt
    duration_seconds: Annotated[FiniteFloat, Field(ge=0)] = 0
    transform: Annotated[str, Field(min_length=1, max_length=2000)]
    retrieved_at: datetime
    warnings: list[Annotated[str, Field(min_length=1, max_length=1000)]] = Field(
        default_factory=list
    )


class RemotionAssetBundle(VersionedContract):
    assets: list[RemotionAsset]
    credits_json: MediaReference
    credits_markdown: MediaReference

    @model_validator(mode="after")
    def validate_assets(self) -> "RemotionAssetBundle":
        asset_ids = [asset.asset_id for asset in self.assets]
        shot_ids = [asset.shot_id for asset in self.assets]
        if len(asset_ids) != len(set(asset_ids)) or len(shot_ids) != len(set(shot_ids)):
            raise ValueError("resolved Remotion assets must have unique asset and Shot IDs")
        return self


class ImageGenerationSettings(ContractModel):
    inference_steps: Annotated[int, Field(ge=1, le=100)] | None = None
    guidance_scale: Annotated[FiniteFloat, Field(ge=0, le=30)] | None = None
    moderation: Literal["auto", "low"] | None = None
    output_format: Literal["png", "jpeg", "webp"] = "png"
    aspect_ratio: Literal["16:9", "9:16"] = "16:9"
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
        expected_ratio = 16 / 9 if self.settings.aspect_ratio == "16:9" else 9 / 16
        qwen_native_size = (
            self.target_backend_id == "local:qwen-image-2512-nf4"
            and (self.width, self.height)
            == (
                (1664, 928)
                if self.settings.aspect_ratio == "16:9"
                else (928, 1664)
            )
        )
        if not qwen_native_size and abs(self.width / self.height - expected_ratio) > 0.002:
            raise ValueError(
                "image dimensions must match settings.aspect_ratio or use the matching "
                "Qwen native preset"
            )
        if len(set(self.reference_paths)) != len(self.reference_paths):
            raise ValueError("image reference paths must be unique")
        return self


class TimedImageRequest(ImageRequest):
    shot_id: Annotated[str, Field(pattern=r"^shot-\d{3,}$")]


class ImageAsset(VersionedContract):
    scene_id: str
    shot_id: Annotated[str, Field(pattern=r"^shot-\d{3,}$")] | None = None
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


class TimedVisualReviewItem(VisualReviewItem):
    shot_id: Annotated[str, Field(pattern=r"^shot-\d{3,}$")]


class VisualReviewReport(VersionedContract):
    items: list[VisualReviewItem | TimedVisualReviewItem]
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
    shot_id: Annotated[str, Field(pattern=r"^shot-\d{3,}$")] | None = None
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
        visual_ids = []
        for scene in self.scenes:
            if abs(scene.start_seconds - previous) > 0.002:
                raise ValueError("Render Plan Scenes must be contiguous and start at zero")
            previous = scene.end_seconds
            visual_ids.append(scene.shot_id or scene.scene_id)
        if len(visual_ids) != len(set(visual_ids)):
            raise ValueError("Render Plan visual IDs must be unique")
        if any(scene.shot_id for scene in self.scenes) and not all(
            scene.shot_id for scene in self.scenes
        ):
            raise ValueError("Render Plan cannot mix Scene and Shot visual identities")
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


class RemotionRenderPlan(VersionedContract):
    renderer: Literal["remotion"] = "remotion"
    edit_plan: RemotionEditPlan
    assets: list[RemotionAsset]
    credits_json: MediaReference
    credits_markdown: MediaReference
    render_manifest: MediaReference
    narration_path: str
    music_path: str | None = None
    caption_srt_path: str | None = None
    caption_ass_path: str | None = None
    caption_language: OutputLanguage | None = None
    width: PositiveInt
    height: PositiveInt
    fps: PositiveInt
    duration_seconds: Annotated[FiniteFloat, Field(gt=0)]
    caption_mode: Literal[
        "none",
        "selectable",
        "selectable_and_burned",
        "selectable_and_composited",
    ] = "none"

    @model_validator(mode="after")
    def validate_plan(self) -> "RemotionRenderPlan":
        if self.width != self.edit_plan.width or self.height != self.edit_plan.height:
            raise ValueError("Remotion render dimensions must match the Edit Plan")
        if self.fps != self.edit_plan.fps:
            raise ValueError("Remotion render FPS must match the Edit Plan")
        if abs(self.duration_seconds - self.edit_plan.duration_seconds) > 0.002:
            raise ValueError("Remotion render duration must match the Edit Plan")
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
    runner: Literal["in_process", "native", "wsl", "docker", "either"] = "in_process"
    languages: set[OutputLanguage]
    required_env: list[str] = Field(default_factory=list)
    required_assets: list[str] = Field(default_factory=list)
    supports_vision: bool = False
    supports_word_timing: bool = False
    supports_voice_cloning: bool = False
    requires_reference_transcript: bool = False
    requires_reference_language: bool = False
    supports_reference_images: bool = False
    supports_negative_prompt: bool = False
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
    visual_shot_count: PositiveInt | None = None
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
    call_id: str = ""
    provider_request_id: str = ""
    input_units: Annotated[FiniteFloat, Field(ge=0)] = 0
    output_units: Annotated[FiniteFloat, Field(ge=0)] = 0
    billable_units: dict[str, Annotated[FiniteFloat, Field(ge=0)]] = Field(default_factory=dict)
    reserved_usd: Annotated[FiniteFloat, Field(ge=0)] = 0
    estimated_usd: Annotated[FiniteFloat, Field(ge=0)] | None = None
    actual_usd: Annotated[FiniteFloat, Field(ge=0)] | None = None
    cost_status: Literal["not_applicable", "estimated", "reported", "unpriced"] = "unpriced"
    pricing_snapshot: str = ""
    cost_basis: str = ""
    elapsed_seconds: Annotated[FiniteFloat, Field(ge=0)] = 0
    peak_vram_mb: Annotated[FiniteFloat, Field(ge=0)] | None = None
    warnings: list[str] = Field(default_factory=list)


class CloudCallRecord(ContractModel):
    call_id: str
    task_id: str
    backend_id: str
    stage: str = ""
    status: Literal["reserved", "settled", "unresolved"] = "reserved"
    provider_request_id: str = ""
    reserved_usd: Annotated[FiniteFloat, Field(ge=0)] = 0
    estimated_usd: Annotated[FiniteFloat, Field(ge=0)] | None = None
    actual_usd: Annotated[FiniteFloat, Field(ge=0)] | None = None
    billable_units: dict[str, Annotated[FiniteFloat, Field(ge=0)]] = Field(default_factory=dict)
    pricing_snapshot: str = ""
    cost_basis: str = ""
    incurred_in_run_id: str = ""
    inherited: bool = False
    legacy: bool = False
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
    elapsed_seconds: Annotated[FiniteFloat, Field(ge=0)] = 0
    error: dict[str, Any] | None = None
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
    cloud_calls: list[CloudCallRecord] = Field(default_factory=list)
    cloud_cost_ledger_version: Annotated[int, Field(ge=0)] = 0
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
    delivery: NarrationDeliverySpec | None = None
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
