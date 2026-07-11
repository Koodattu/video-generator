from __future__ import annotations

import pytest

from video_generator.contracts import ImageGenerationSettings, ImageRequest
from video_generator.errors import BackendError, ErrorKind
from video_generator.workflow import WorkflowEngine, _raw_image_extension


def test_flux_request_uses_host_owned_dimensions_and_sampler_settings() -> None:
    compiled = ImageRequest(
        scene_id="wrong-scene",
        target_backend_id="wrong-backend",
        prompt="A fox beside a tiny amber lantern.",
        width=2048,
        height=1152,
        quality="high",
        reference_paths=["wrong.png"],
        settings=ImageGenerationSettings(inference_steps=20, guidance_scale=3.5),
    )

    request = WorkflowEngine._canonical_image_request(
        compiled,
        scene_id="scene-001",
        target_backend_id="local:flux.2-klein-4b",
        width=1024,
        height=576,
        quality="low",
        reference_paths=[],
    )

    assert request.scene_id == "scene-001"
    assert request.target_backend_id == "local:flux.2-klein-4b"
    assert (request.width, request.height, request.quality) == (1024, 576, "low")
    assert request.reference_paths == []
    assert request.settings.inference_steps == 4
    assert request.settings.guidance_scale == 1.0


def test_raw_image_extension_matches_backend_output_format() -> None:
    assert _raw_image_extension("gemini:gemini-3.1-flash-image") == ".jpg"
    assert _raw_image_extension("local:flux.2-klein-4b") == ".png"
    assert _raw_image_extension("deterministic:stick") == ".ppm"


def test_gemini_request_uses_host_owned_jpeg_format() -> None:
    compiled = ImageRequest(
        scene_id="scene-001",
        target_backend_id="gemini:gemini-3.1-flash-image",
        prompt="A small orange fox shields an amber lantern from falling snow in a wide forest view.",
        negative_prompt="No text, letters, logos, watermarks, gradients, or photorealistic details.",
        width=2048,
        height=1152,
        settings=ImageGenerationSettings(output_format="png", image_size="1K"),
    )

    request = WorkflowEngine._canonical_image_request(
        compiled,
        scene_id="scene-001",
        target_backend_id="gemini:gemini-3.1-flash-image",
        width=2048,
        height=1152,
        quality="low",
        reference_paths=[],
    )

    assert request.settings.output_format == "jpeg"
    assert request.settings.image_size == "2K"
    assert request.settings.aspect_ratio == "16:9"


def test_image_prompt_language_validation_rejects_finnish() -> None:
    request = ImageRequest(
        scene_id="scene-001",
        target_backend_id="local:flux.2-klein-4b",
        prompt=(
            "Pieni oranssi kettu suojaa meripihkaista lyhtyä lumelta sinisessä talvimetsässä, "
            "leveä rauhallinen sommitelma ilman kirjoitettua tekstiä."
        ),
        width=1024,
        height=576,
    )

    with pytest.raises(BackendError, match="must be English") as caught:
        WorkflowEngine._validate_image_request_language(request)

    assert caught.value.kind is ErrorKind.INVALID_OUTPUT


def test_image_prompt_language_validation_accepts_english() -> None:
    request = ImageRequest(
        scene_id="scene-001",
        target_backend_id="local:flux.2-klein-4b",
        prompt=(
            "A small orange fox shields an amber lantern from falling snow in a sparse blue winter "
            "forest, shown in a wide and readable composition."
        ),
        negative_prompt="No text, letters, logos, watermarks, gradients, or photorealistic details.",
        width=1024,
        height=576,
    )

    WorkflowEngine._validate_image_request_language(request)
