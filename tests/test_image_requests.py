from __future__ import annotations

import pytest

from video_generator.contracts import (
    CharacterIdentity,
    ImageGenerationSettings,
    ImageRequest,
    NarrationScript,
    OutlineScene,
    ScriptScene,
    StyleProfile,
    StoryOutline,
    VisualBrief,
    VisualPlan,
)
from video_generator.errors import BackendError, ErrorKind
from video_generator.workflow import WorkflowEngine, _raw_image_extension


def _continuity_inputs() -> tuple[StoryOutline, NarrationScript]:
    outline = StoryOutline(
        title="The lantern",
        concept_summary="Aino carries a lantern through the snow.",
        scenes=[
            OutlineScene(
                scene_id=f"scene-{index:03d}",
                narrative_purpose="Advance the journey.",
                change="Aino moves closer to shelter.",
                emotional_beat="Hope grows.",
                visual_opportunity="A red scarf and blue lantern cross the snow.",
                provisional_seconds=10,
                continuity_obligations=["Keep Aino's scarf and lantern unchanged."],
            )
            for index in range(1, 3)
        ],
    )
    script = NarrationScript(
        title="The lantern",
        scenes=[
            ScriptScene(
                scene_id=f"scene-{index:03d}",
                spoken_text="Aino follows the lantern through the snow.",
            )
            for index in range(1, 3)
        ],
    )
    return outline, script


def _visual_briefs(*, character_ids: list[str]) -> list[VisualBrief]:
    return [
        VisualBrief(
            scene_id=f"scene-{index:03d}",
            story_moment="Aino follows the lantern.",
            subjects=["Aino", "blue lantern"],
            action="Aino walks through the snow.",
            emotion="hopeful",
            environment="snowy path",
            composition="wide",
            must_show=["red scarf", "blue lantern"],
            must_avoid=["text"],
            character_ids=character_ids,
            continuity_from_previous=["Aino has the red scarf and blue lantern."],
            state_after_scene=["Aino advances along the path."],
            identity_requirements=(
                ["small upright figure with a red scarf"] if character_ids else []
            ),
        )
        for index in range(1, 3)
    ]


def _style_profile() -> StyleProfile:
    return StyleProfile(
        style_id="ink",
        description="Simple ink",
        palette=["black", "white", "red", "blue"],
        line_style="loose",
        background="paper",
        must_avoid=["text"],
    )


def _aino_identity() -> CharacterIdentity:
    return CharacterIdentity(
        character_id="character-aino",
        name="Aino",
        signature_traits=["red scarf"],
        body_form="small upright figure",
        proportions=["round head", "short limbs"],
        face_and_markings=["two black dot eyes"],
        identity_constraints=["never remove or recolor the scarf"],
    )


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


def test_continuity_visual_plan_requires_a_character_identity() -> None:
    outline, script = _continuity_inputs()
    plan = VisualPlan(
        style_profile=_style_profile(),
        characters=[],
        scenes=_visual_briefs(character_ids=[]),
    )

    with pytest.raises(BackendError, match="at least one recurring Character Identity"):
        WorkflowEngine._validate_visual_plan(plan, outline=outline, script=script)


def test_continuity_visual_plan_requires_a_cross_scene_identity_mapping() -> None:
    outline, script = _continuity_inputs()
    scenes = _visual_briefs(character_ids=[])
    scenes[0].character_ids = ["character-aino"]
    scenes[0].identity_requirements = ["small upright figure with a red scarf"]
    plan = VisualPlan(
        style_profile=_style_profile(),
        characters=[_aino_identity()],
        scenes=scenes,
    )

    with pytest.raises(BackendError, match="map at least one Character Identity across Scenes"):
        WorkflowEngine._validate_visual_plan(plan, outline=outline, script=script)


def test_continuity_visual_plan_accepts_a_recurring_identity_mapping() -> None:
    outline, script = _continuity_inputs()
    plan = VisualPlan(
        style_profile=_style_profile(),
        characters=[_aino_identity()],
        scenes=_visual_briefs(character_ids=["character-aino"]),
    )

    WorkflowEngine._validate_visual_plan(plan, outline=outline, script=script)


def test_character_reference_is_carried_forward_without_copying_composition() -> None:
    request = ImageRequest(
        scene_id="scene-002",
        target_backend_id="local:flux.2-klein-4b",
        prompt="The same small orange fox lifts a lantern beside a frozen stream.",
        width=1024,
        height=576,
    )

    effective = WorkflowEngine._with_continuity_references(
        request,
        character_ids=["fox"],
        character_reference_paths={"fox": "runs/run/stages/images/scene-001.png"},
        supports_reference_images=True,
    )

    assert effective.reference_paths == ["runs/run/stages/images/scene-001.png"]
    assert "identity/style evidence only" in effective.prompt
    assert "Do not copy a reference pose" in effective.prompt


def test_character_reference_is_not_added_for_unsupported_backend() -> None:
    request = ImageRequest(
        scene_id="scene-002",
        target_backend_id="deterministic:stick",
        prompt="A stick figure lifts a lantern beside a frozen stream.",
        width=1280,
        height=720,
    )

    assert WorkflowEngine._with_continuity_references(
        request,
        character_ids=["hero"],
        character_reference_paths={"hero": "first.png"},
        supports_reference_images=False,
    ) == request


def test_legacy_visual_payload_omits_v14_continuity_fields() -> None:
    engine = object.__new__(WorkflowEngine)
    engine.continuity_policy_enabled = False
    plan = VisualPlan(
        style_profile=StyleProfile(
            style_id="ink",
            description="Simple ink",
            palette=["black", "white"],
            line_style="loose",
            background="paper",
            must_avoid=["text"],
        ),
        characters=[
            CharacterIdentity(
                character_id="fox",
                name="Fox",
                signature_traits=["small"],
                body_form="quadruped",
                proportions=["short legs"],
                face_and_markings=["white muzzle"],
                identity_constraints=["never bipedal"],
            )
        ],
        scenes=[
            VisualBrief(
                scene_id="scene-001",
                story_moment="The fox finds a light.",
                subjects=["fox", "lantern"],
                action="The fox approaches.",
                emotion="curious",
                environment="snow",
                composition="wide",
                must_show=["lantern"],
                must_avoid=["text"],
                character_ids=["fox"],
                continuity_from_previous=["opening state"],
                state_after_scene=["lantern found"],
                identity_requirements=["quadruped"],
            )
        ],
    )

    payload = engine._visual_plan_payload(plan)

    assert "body_form" not in payload["characters"][0]
    assert "continuity_from_previous" not in payload["scenes"][0]
    assert payload["scenes"][0]["character_ids"] == ["fox"]
