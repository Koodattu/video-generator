from __future__ import annotations

from video_generator.contracts import OutputLanguage
from video_generator.prompting import (
    PROMPT_SET_VERSION,
    SHARED_RULES,
    PromptLibrary,
    build_frozen_assets,
    task_output_language,
)


def test_image_prompt_compiler_always_uses_english() -> None:
    prompt = PromptLibrary().get(
        "image_prompt_compile",
        language=OutputLanguage.FINNISH,
        target_image_backend="local:flux.2-klein-4b",
    )

    assert PROMPT_SET_VERSION == "2026-07-12.v14"
    assert task_output_language("image_prompt_compile", OutputLanguage.FINNISH) is OutputLanguage.ENGLISH
    assert "Source artifact language: fi" in prompt.instructions
    assert "Required ImageRequest prompt language: English" in prompt.instructions
    assert "Selected Output Language: fi" not in prompt.instructions
    assert "ImageRequest.negative_prompt entirely in English" in prompt.instructions


def test_ordinary_tasks_keep_the_run_language() -> None:
    prompt = PromptLibrary().get("script_draft", language=OutputLanguage.FINNISH)

    assert task_output_language("script_draft", OutputLanguage.FINNISH) is OutputLanguage.FINNISH
    assert prompt.instructions.endswith("Selected Output Language: fi.")


def test_legacy_frozen_visual_plan_keeps_original_run_language() -> None:
    assets = build_frozen_assets()
    assets.pop("workflow_policy_version")
    prompts = PromptLibrary(assets)

    assert prompts.output_language("visual_plan", OutputLanguage.FINNISH) is OutputLanguage.FINNISH
    assert prompts.output_language("image_prompt_compile", OutputLanguage.FINNISH) is OutputLanguage.ENGLISH


def test_default_config_keeps_the_legacy_prompt_and_schema_pack(resolved_config) -> None:
    baseline = build_frozen_assets()
    configured = build_frozen_assets(resolved_config)

    assert configured["prompt_set_version"] == baseline["prompt_set_version"]
    assert configured["workflow_policy_version"] == baseline["workflow_policy_version"]
    assert configured["prompts"] == baseline["prompts"]
    assert configured["schemas"] == baseline["schemas"]
    assert "claim_inventory" not in baseline["prompts"]
    assert "review_type" in baseline["schemas"]["factual_review"]["properties"]


def test_craft_rules_remain_task_specific() -> None:
    assert "midpoint" not in SHARED_RULES
    outline = PromptLibrary().get("outline", language=OutputLanguage.ENGLISH)
    assert "renew or complicate that uncertainty once around the middle" in " ".join(
        outline.instructions.split()
    )
