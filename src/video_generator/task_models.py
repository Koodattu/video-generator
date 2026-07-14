from __future__ import annotations

from pydantic import BaseModel

from .contracts import (
    CandidateSet,
    ContentFormat,
    ContentMode,
    ExplainerCandidateSet,
    ExplainerOutline,
    ExplainerSelectionReport,
    FactualResearchSynthesis,
    FactualReviewReport,
    ImageRequest,
    MusicBrief,
    NarrationScript,
    ResearchSynthesis,
    ResolvedRunConfig,
    ReviewReport,
    RevisedScript,
    SceneClaimExtraction,
    SelectionReport,
    StoryOutline,
    TimedImageRequest,
    TimedVisualPlan,
    TimedVisualReviewItem,
    VisualPlan,
    VisualReviewItem,
    VisualShotMode,
)


def task_output_models(config: ResolvedRunConfig | None = None) -> dict[str, type[BaseModel]]:
    models: dict[str, type[BaseModel]] = {
        "research": ResearchSynthesis,
        "ideate": CandidateSet,
        "select": SelectionReport,
        "outline": StoryOutline,
        "script_draft": NarrationScript,
        "review_story": ReviewReport,
        "review_spoken": ReviewReport,
        "review_constraints": ReviewReport,
        "script_revision": RevisedScript,
        "factual_review": ReviewReport,
        "duration_repair": RevisedScript,
        "visual_plan": VisualPlan,
        "image_prompt_compile": ImageRequest,
        "visual_review": VisualReviewItem,
        "music_brief": MusicBrief,
    }
    if config is None:
        return models
    if config.content_mode is ContentMode.FACTUAL:
        models.update(
            {
                "research": FactualResearchSynthesis,
                "claim_inventory": SceneClaimExtraction,
                "factual_review": FactualReviewReport,
            }
        )
    if config.content_format is not ContentFormat.NARRATIVE:
        models.update(
            {
                "ideate": ExplainerCandidateSet,
                "select": ExplainerSelectionReport,
                "outline": ExplainerOutline,
            }
        )
    if config.visual_shot_mode is VisualShotMode.CADENCED:
        models.update(
            {
                "visual_plan": TimedVisualPlan,
                "image_prompt_compile": TimedImageRequest,
                "visual_review": TimedVisualReviewItem,
            }
        )
    return models


TASK_OUTPUT_MODELS = task_output_models()
