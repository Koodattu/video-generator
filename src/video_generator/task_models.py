from __future__ import annotations

from pydantic import BaseModel

from .contracts import (
    CandidateSet,
    ImageRequest,
    MusicBrief,
    NarrationScript,
    ResearchPack,
    ReviewReport,
    RevisedScript,
    SelectionReport,
    StoryOutline,
    VisualPlan,
    VisualReviewItem,
)


TASK_OUTPUT_MODELS: dict[str, type[BaseModel]] = {
    "research": ResearchPack,
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

