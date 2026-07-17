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
    VideoOrientation,
    VisualBrief,
    VisualPlan,
)
from video_generator.errors import BackendError, ErrorKind
from video_generator.profiles import image_generation_dimensions
from video_generator.workflow import ImagePromptContent, WorkflowEngine, _raw_image_extension


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


@pytest.mark.parametrize(
    ("backend_id", "quality", "steps", "guidance", "cpu_offload"),
    [
        ("local:z-image-turbo", "high", 9, 0.0, False),
        ("local:ideogram-4-nf4", "low", 12, None, True),
        ("local:ideogram-4-nf4", "medium", 20, None, True),
        ("local:ideogram-4-nf4", "high", 48, None, True),
        ("local:qwen-image-2512-nf4", "low", 50, 4.0, True),
        ("local:qwen-image-2512-nf4", "medium", 50, 4.0, True),
        ("local:qwen-image-2512-nf4", "high", 50, 4.0, True),
    ],
)
def test_challenger_image_requests_use_host_owned_runtime_settings(
    backend_id: str,
    quality: str,
    steps: int,
    guidance: float | None,
    cpu_offload: bool,
) -> None:
    compiled = ImageRequest(
        scene_id="scene-001",
        target_backend_id="wrong-backend",
        prompt="A fox beside a tiny amber lantern.",
        width=2048,
        height=1152,
        settings=ImageGenerationSettings(
            inference_steps=99,
            guidance_scale=19.0,
            cpu_offload=not cpu_offload,
        ),
    )

    request = WorkflowEngine._canonical_image_request(
        compiled,
        scene_id="scene-001",
        target_backend_id=backend_id,
        width=1024,
        height=576,
        quality=quality,
        reference_paths=[],
    )

    assert request.settings.inference_steps == steps
    assert request.settings.guidance_scale == guidance
    assert request.settings.cpu_offload is cpu_offload


def test_qwen_image_uses_its_documented_native_generation_size() -> None:
    width, height = image_generation_dimensions(
        "local:qwen-image-2512-nf4",
        delivery_width=1280,
        delivery_height=720,
    )
    request = ImageRequest(
        scene_id="scene-001",
        target_backend_id="local:qwen-image-2512-nf4",
        prompt="A clear winter shelter diagram in a restrained blue and amber palette.",
        width=width,
        height=height,
    )

    assert (request.width, request.height) == (1664, 928)
    assert image_generation_dimensions(
        "local:flux.2-klein-4b",
        delivery_width=1280,
        delivery_height=720,
    ) == (1024, 576)


@pytest.mark.parametrize(
    ("backend_id", "expected"),
    [
        ("openai:gpt-image-2", (1152, 2048)),
        ("gemini:gemini-3.1-flash-image", (1152, 2048)),
        ("local:flux.2-klein-4b", (576, 1024)),
        ("local:z-image-turbo", (576, 1024)),
        ("local:ideogram-4-nf4", (576, 1024)),
        ("local:qwen-image-2512-nf4", (928, 1664)),
        ("deterministic:stick", (720, 1280)),
    ],
)
def test_portrait_generation_dimensions_are_backend_native(
    backend_id: str,
    expected: tuple[int, int],
) -> None:
    assert image_generation_dimensions(
        backend_id,
        delivery_width=720,
        delivery_height=1280,
        orientation=VideoOrientation.PORTRAIT,
    ) == expected


def test_portrait_image_request_requires_matching_aspect_setting() -> None:
    request = ImageRequest(
        scene_id="scene-001",
        target_backend_id="local:flux.2-klein-4b",
        prompt="A clear vertical winter shelter diagram.",
        width=576,
        height=1024,
        settings=ImageGenerationSettings(aspect_ratio="9:16"),
    )

    assert request.settings.aspect_ratio == "9:16"
    qwen_request = ImageRequest(
        scene_id="scene-001",
        target_backend_id="local:qwen-image-2512-nf4",
        prompt="A clear vertical winter shelter diagram.",
        width=928,
        height=1664,
        settings=ImageGenerationSettings(aspect_ratio="9:16"),
    )
    assert (qwen_request.width, qwen_request.height) == (928, 1664)

    with pytest.raises(ValueError, match="match settings.aspect_ratio"):
        ImageRequest(
            scene_id="scene-001",
            target_backend_id="local:flux.2-klein-4b",
            prompt="A clear vertical winter shelter diagram.",
            width=576,
            height=1024,
        )


def test_non_qwen_image_request_still_requires_the_selected_aspect_ratio() -> None:
    with pytest.raises(ValueError, match="match settings.aspect_ratio"):
        ImageRequest(
            scene_id="scene-001",
            target_backend_id="local:flux.2-klein-4b",
            prompt="A clear winter shelter diagram.",
            width=1664,
            height=928,
        )


def test_qwen_negative_prompt_drops_approved_palette_and_style_conflicts() -> None:
    visual_brief = VisualBrief(
        scene_id="scene-001",
        story_moment="A cyan shelter glows beside an amber lantern.",
        subjects=["cyan shelter", "amber lantern"],
        action="The lantern illuminates the shelter.",
        emotion="focused",
        environment="snowy night",
        composition="centered wide view",
        must_show=["cyan shelter", "amber lantern"],
        must_avoid=["text", "logos"],
        continuity_from_previous=[],
        state_after_scene=["The shelter remains visible."],
    )
    style_profile = StyleProfile(
        style_id="paper-cut",
        description="Simple paper-cut collage with cyan and amber accents.",
        palette=["cyan", "amber", "black"],
        line_style="rough paper-cut edges",
        background="dark paper texture",
        must_avoid=["text"],
    )
    content = ImagePromptContent(
        prompt=(
            "A simple paper-cut collage of a cyan shelter and amber lantern on dark paper."
        ),
        negative_prompt=(
            "text, logos, no cyan or amber colors, without paper-cut collage, black lantern, "
            "blurry anatomy"
        ),
    )

    result = WorkflowEngine._deconflict_qwen_negative_prompt(
        content,
        visual_brief=visual_brief,
        style_profile=style_profile,
    )

    assert result.negative_prompt == "text, logos, black lantern, blurry anatomy"


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


def test_canonical_portrait_request_uses_host_owned_aspect_ratio() -> None:
    compiled = ImageRequest(
        scene_id="scene-001",
        target_backend_id="gemini:gemini-3.1-flash-image",
        prompt="A fox beside an amber lantern in a tall forest frame.",
        width=2048,
        height=1152,
    )

    request = WorkflowEngine._canonical_image_request(
        compiled,
        scene_id="scene-001",
        target_backend_id="gemini:gemini-3.1-flash-image",
        width=1152,
        height=2048,
        quality="low",
        reference_paths=[],
    )

    assert (request.width, request.height) == (1152, 2048)
    assert request.settings.aspect_ratio == "9:16"


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


def test_continuity_visual_plan_allows_no_previous_state_for_opening_scene() -> None:
    outline, script = _continuity_inputs()
    scenes = _visual_briefs(character_ids=["character-aino"])
    scenes[0].continuity_from_previous = []
    plan = VisualPlan(
        style_profile=_style_profile(),
        characters=[_aino_identity()],
        scenes=scenes,
    )

    WorkflowEngine._validate_visual_plan(plan, outline=outline, script=script)


def test_continuity_visual_plan_requires_previous_state_after_opening_scene() -> None:
    outline, script = _continuity_inputs()
    scenes = _visual_briefs(character_ids=["character-aino"])
    scenes[1].continuity_from_previous = []
    plan = VisualPlan(
        style_profile=_style_profile(),
        characters=[_aino_identity()],
        scenes=scenes,
    )

    with pytest.raises(BackendError, match="post-opening incoming state.*scene-002"):
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
