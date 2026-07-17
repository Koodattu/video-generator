from __future__ import annotations

from video_generator.contracts import ContentFormat, ContentMode, OutputLanguage, ProtocolName
from video_generator.profiles import BACKEND_DESCRIPTORS
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


def test_image_prompt_compiler_supports_every_local_image_backend() -> None:
    expected_markers = {
        "local:flux.2-klein-4b": "FLUX.2 Klein 4B",
        "local:z-image-turbo": "Z-Image Turbo",
        "local:ideogram-4-nf4": "Ideogram 4 NF4",
        "local:qwen-image-2512-nf4": "Qwen-Image-2512 NF4",
    }

    for backend_id, marker in expected_markers.items():
        prompt = PromptLibrary().get(
            "image_prompt_compile",
            language=OutputLanguage.ENGLISH,
            target_image_backend=backend_id,
        )
        assert marker in prompt.instructions


def test_image_prompt_compiler_supports_every_registered_image_backend() -> None:
    image_backends = {
        backend_id
        for backend_id, descriptor in BACKEND_DESCRIPTORS.items()
        if ProtocolName.IMAGE in descriptor.protocols
    }

    for backend_id in image_backends:
        PromptLibrary().get(
            "image_prompt_compile",
            language=OutputLanguage.ENGLISH,
            target_image_backend=backend_id,
        )


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


def test_factual_review_prompt_contains_only_exclusive_small_decision_strategies(
    resolved_config,
) -> None:
    config = resolved_config.model_copy(
        update={
            "content_mode": ContentMode.FACTUAL,
            "content_format": ContentFormat.MYTHBUSTER,
        }
    )
    instructions = build_frozen_assets(config)["prompts"]["factual_review"][
        "instructions"
    ]

    assert "exactly one bounded decision" in instructions
    assert "mutually exclusive" in instructions
    assert "uncovered_claims" not in instructions
    assert "neither expands nor narrows direct authorization" in instructions
    assert "matching unlabeled measurement or threshold markers" in instructions
