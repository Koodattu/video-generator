from __future__ import annotations

from video_generator.contracts import ImageGenerationSettings, ImageRequest
from video_generator.workflow import WorkflowEngine


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
