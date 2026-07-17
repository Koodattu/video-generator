from __future__ import annotations

from types import SimpleNamespace

import pytest

from video_generator.config import resolve_narration_delivery
from video_generator.contracts import (
    CaptionTrack,
    ContentFormat,
    ContentMode,
    CreativeBrief,
    MediaReference,
    NarrationPace,
    NarrationScript,
    NarrationTimeline,
    OutputLanguage,
    OutlineScene,
    RawRunConfig,
    ReviewFinding,
    ScriptScene,
    StoryOutline,
    TimelineScene,
    TimedImageRequest,
    TimedVisualPlan,
    VideoStyle,
    VisualShotMode,
    VoiceSettings,
    WordTiming,
)
from video_generator.errors import BackendError
from video_generator.profiles import BACKEND_DESCRIPTORS, PROFILES
from video_generator.prompting import MULTI_FORMAT_PROMPT_SET_VERSION, build_frozen_assets
from video_generator.task_models import task_output_models
from video_generator.workflow import CaptionBundle, ImageRequestSet, NarrationBundle, WorkflowEngine


def test_production_profiles_never_select_the_programmatic_image_backend() -> None:
    for profile_name, bindings in PROFILES.items():
        if profile_name == "deterministic-test":
            continue
        image_backend = BACKEND_DESCRIPTORS[bindings["image_generate"]]
        assert image_backend.provider != "deterministic", profile_name


def test_cadenced_mythbuster_scene_count_follows_editorial_arc_not_image_target() -> None:
    engine = object.__new__(WorkflowEngine)
    engine.workflow_policy_version = 12
    engine.config = SimpleNamespace(
        duration_seconds=24,
        visual_target_seconds=10,
        visual_min_seconds=6,
        visual_max_seconds=14,
        visual_shot_mode=VisualShotMode.CADENCED,
        content_format=ContentFormat.MYTHBUSTER,
    )

    assert engine._outline_scene_count_bounds() == (5, 4, 5)
    engine.config.content_format = ContentFormat.EXPLAINER
    assert engine._outline_scene_count_bounds() == (4, 3, 5)


def test_multi_format_prompt_pack_selects_timed_contracts(resolved_config) -> None:
    delivery = resolve_narration_delivery(OutputLanguage.ENGLISH, NarrationPace.FAST)
    bindings = dict(resolved_config.task_bindings)
    bindings["narration_synthesis"] = "local:omnivoice"
    config = resolved_config.model_copy(
        update={
            "content_mode": ContentMode.FACTUAL,
            "content_format": ContentFormat.MYTHBUSTER,
            "narration_pace": NarrationPace.FAST,
            "narration_delivery_spec": delivery,
            "visual_shot_mode": VisualShotMode.CADENCED,
            "task_bindings": bindings,
        }
    )

    models = task_output_models(config)
    assets = build_frozen_assets(config)

    assert models["visual_plan"] is TimedVisualPlan
    assert models["image_prompt_compile"] is TimedImageRequest
    assert assets["prompt_set_version"] == MULTI_FORMAT_PROMPT_SET_VERSION
    assert assets["workflow_policy_version"] == 42
    assert assets["prompts"]["research"]["version"] == (
        f"{MULTI_FORMAT_PROMPT_SET_VERSION}:research"
    )
    assert assets["prompts"]["script_draft"]["version"].endswith(
        ":spoken-text-only-v1"
    )
    assert assets["prompts"]["review_story"]["version"].endswith(
        ":spoken-script-scope-and-resolution-v1"
    )
    assert assets["prompts"]["review_spoken"]["version"].endswith(
        ":spoken-script-scope-and-resolution-v1"
    )
    assert assets["prompts"]["review_constraints"]["version"].endswith(
        ":scope-aware-brief-and-remotion-plan-v2"
    )
    assert "complete revised_script" in assets["prompts"]["review_constraints"][
        "instructions"
    ]
    claim_properties = assets["schemas"]["claim_inventory"]["$defs"]["ExtractedClaim"][
        "properties"
    ]
    assert set(claim_properties) == {"exact_text", "qualification"}
    assert "shots" in assets["schemas"]["visual_plan"]["properties"]
    assert set(assets["schemas"]["research"]["properties"]) == {"evidence"}
    evidence_schema = assets["schemas"]["research"]["properties"]["evidence"]
    assert evidence_schema["maxItems"] == 12
    evidence_properties = assets["schemas"]["research"]["$defs"][
        "EvidenceRecordDraft"
    ]["properties"]
    assert "evidence_id" not in evidence_properties
    assert evidence_properties["supported_statement"]["maxLength"] == 600
    assert evidence_properties["limitations"]["maxItems"] == 4
    assert evidence_properties["limitations"]["items"]["maxLength"] == 240
    assert "ResearchFindingDraft" not in assets["schemas"]["research"]["$defs"]
    assert "urgent" in assets["prompts"]["script_draft"]["instructions"].lower()


def test_visual_brief_constraints_are_routed_out_of_spoken_script() -> None:
    engine = object.__new__(WorkflowEngine)
    engine.brief = CreativeBrief(
        idea_direction="A fox finds a lantern.",
        must_include=["a tiny amber lantern", "one quiet visual joke"],
    )

    assert engine._script_brief_must_include() == ["a tiny amber lantern"]
    assert engine._script_brief_avoid() == []
    assert WorkflowEngine._is_visual_brief_constraint("ruudulla näkyvä meemi")
    assert not WorkflowEngine._is_visual_brief_constraint("an unexpected friendship")
    assert not WorkflowEngine._is_visual_brief_constraint(
        "make the cost visible in the fox's choice"
    )
    assert not WorkflowEngine._is_visual_brief_constraint("a visibly nervous fox")
    assert not WorkflowEngine._is_visual_brief_constraint(
        "tee seuraukset näkyviksi ketun valinnassa"
    )
    assert not WorkflowEngine._is_visual_brief_constraint("one last shot at redemption")
    assert WorkflowEngine._is_visual_brief_constraint("one reaction shot")


@pytest.mark.parametrize(
    ("value", "narrative", "visual"),
    [
        (
            "Tell the fox story and show a reaction GIF",
            "Tell the fox story",
            "show a reaction GIF",
        ),
        (
            "Kerro ketun tarina ja näytä reaktio-GIF",
            "Kerro ketun tarina",
            "näytä reaktio-GIF",
        ),
        (
            "Tell the fox story and show how friendship changes him",
            "Tell the fox story and show how friendship changes him",
            None,
        ),
        ("Show a reaction GIF", "Show a reaction GIF", None),
        ("A visibly nervous fox finds courage", "A visibly nervous fox finds courage", None),
    ],
)
def test_remotion_idea_direction_splits_only_a_trailing_visual_command(
    value: str,
    narrative: str,
    visual: str | None,
) -> None:
    assert WorkflowEngine._split_remotion_idea_direction(value) == (narrative, visual)


def test_remotion_routes_a_mixed_idea_direction_to_script_and_visual_plan() -> None:
    engine = object.__new__(WorkflowEngine)
    engine.config = SimpleNamespace(video_style=VideoStyle.REMOTION_EXPLAINER)
    engine.brief = CreativeBrief(
        idea_direction="Tell the fox story and show a reaction GIF",
        must_include=["one quiet visual joke", "a tiny lantern"],
        avoid=["avoid stock footage", "chosen-one plots"],
    )

    assert engine._script_idea_direction() == "Tell the fox story"
    assert engine._script_review_brief_payload()["idea_direction"] == (
        "Tell the fox story"
    )
    assert engine._editorial_brief_payload()["idea_direction"] == "Tell the fox story"
    assert "show a reaction GIF" not in str(engine._editorial_brief_payload())
    assert engine._research_query_candidates() == ["Tell the fox story"]
    assert engine._remotion_visual_brief_constraints() == [
        ("must-include", 0, "show a reaction GIF"),
        ("must-include", 1, "one quiet visual joke"),
        ("avoid", 1, "avoid stock footage"),
    ]

    engine.config.video_style = VideoStyle.STILL_IMAGE
    assert engine._script_idea_direction() == engine.brief.idea_direction
    assert engine._editorial_brief_payload()["idea_direction"] == engine.brief.idea_direction
    assert engine._research_query_candidates() == [engine.brief.idea_direction]
    assert engine._remotion_visual_brief_constraints() == []


def test_story_and_spoken_review_payloads_exclude_visual_only_requirements() -> None:
    engine = object.__new__(WorkflowEngine)
    engine.config = SimpleNamespace(video_style=VideoStyle.REMOTION_EXPLAINER)
    engine.brief = CreativeBrief(
        idea_direction="A fox finds a lantern.",
        must_include=["a tiny amber lantern", "one visible visual joke"],
        avoid=["chosen-one plots", "avoid stock footage"],
    )
    outline = StoryOutline(
        title="Fixture",
        concept_summary="A winter story.",
        scenes=[
            OutlineScene(
                scene_id="scene-001",
                narrative_purpose="Introduce the fox.",
                change="The fox finds warmth.",
                emotional_beat="Relief",
                visual_opportunity="The fox hides the lantern under a tiny leaf.",
                provisional_seconds=6,
            )
        ],
    )

    brief_payload = engine._script_review_brief_payload()
    outline_payload = engine._script_review_outline_payload(outline)

    assert brief_payload["must_include"] == ["a tiny amber lantern"]
    assert brief_payload["avoid"] == ["chosen-one plots"]
    assert "visual_opportunity" not in outline_payload["scenes"][0]


def test_brief_constraint_resolution_rechecks_only_the_complete_revised_script() -> None:
    engine = object.__new__(WorkflowEngine)
    engine.workflow_policy_version = 39
    engine.config = SimpleNamespace(
        output_language=OutputLanguage.ENGLISH,
        video_style=VideoStyle.REMOTION_EXPLAINER,
    )
    engine.brief = CreativeBrief(
        idea_direction="A shy fox discovers a persistent lantern on a winter night."
    )
    script = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(
                scene_id="scene-001",
                spoken_text=(
                    "On the coldest night of winter, a shy fox discovered a tiny lantern "
                    "that refused to go out."
                ),
                pause_after_seconds=0,
            ),
            ScriptScene(
                scene_id="scene-002",
                spoken_text="A shivering owl descended beside the fox.",
                pause_after_seconds=0,
            ),
        ],
    )
    brief_finding = ReviewFinding(
        finding_id="constraints:brief-idea-direction-001",
        severity="blocking",
        scene_id="scene-001",
        evidence="The shy fox is missing.",
        recommendation="Introduce the shy fox and the persistent lantern.",
    )

    brief_payload = engine._revision_finding_resolution_input(
        finding=brief_finding,
        original_spoken_text="A lantern appeared.",
        revised_script=script,
        scene_index=0,
        allowed_factual_evidence=[],
    )

    assert brief_payload["resolution_scope"] == "complete-script-brief-constraint-v1"
    assert "adjacent_context" not in brief_payload
    assert "original_spoken_text" not in brief_payload
    assert [
        scene["scene_id"] for scene in brief_payload["revised_script"]["scenes"]
    ] == ["scene-001", "scene-002"]

    story_payload = engine._revision_finding_resolution_input(
        finding=ReviewFinding(
            finding_id="story:finding-001",
            severity="minor",
            scene_id="scene-001",
            evidence="The transition is abrupt.",
            recommendation="Smooth the transition.",
        ),
        original_spoken_text="A lantern appeared.",
        revised_script=script,
        scene_index=0,
        allowed_factual_evidence=[],
    )
    assert story_payload["resolution_scope"] == "single-scene-finding-v1"
    assert story_payload["adjacent_context"]["next_spoken_text"].startswith(
        "A shivering owl"
    )
    assert "revised_script" not in story_payload


def test_brief_constraint_repair_targets_a_feasible_explicit_replacement() -> None:
    engine = object.__new__(WorkflowEngine)
    engine.config = SimpleNamespace(video_style=VideoStyle.REMOTION_EXPLAINER)
    engine.brief = CreativeBrief(
        idea_direction=(
            "On the coldest night of winter, a shy fox finds a tiny lantern that refuses "
            "to go out"
        )
    )
    finding = ReviewFinding(
        finding_id="constraints:brief-idea-direction-001",
        severity="blocking",
        scene_id="scene-001",
        evidence="The persistent light is missing.",
        recommendation=(
            "Revise scene-001 to: 'On the coldest night of winter, a shy fox found a tiny "
            "amber lantern that refused to go out.'"
        ),
    )

    assert engine._revision_finding_payload(finding)["brief_constraint"] == {
        "kind": "idea-direction",
        "constraint": engine.brief.idea_direction,
    }
    assert (
        engine._revision_target_word_count(
            findings=[finding],
            scene_id="scene-001",
            current_words=13,
            minimum_words=8,
            maximum_words=16,
        )
        == 16
    )


def test_subtractive_finding_does_not_expand_to_the_residual_maximum() -> None:
    engine = object.__new__(WorkflowEngine)
    engine.brief = CreativeBrief(idea_direction="A fox finds a persistent lantern.")
    finding = ReviewFinding(
        finding_id="story:finding-001",
        severity="major",
        scene_id="scene-001",
        evidence="The sentence repeats the same point.",
        recommendation="Remove the repetition.",
    )

    assert (
        engine._revision_target_word_count(
            findings=[finding],
            scene_id="scene-001",
            current_words=13,
            minimum_words=8,
            maximum_words=16,
        )
        == 13
    )


def test_explicit_brief_replacement_parser_is_bounded_and_quote_tolerant() -> None:
    curly = ReviewFinding(
        finding_id="constraints:brief-idea-direction-001",
        severity="blocking",
        scene_id="scene-001",
        evidence="The persistent light is missing.",
        recommendation="Correction: revise scene-001 to: “The lantern stayed lit all night.”",
    )
    mismatched_scene = curly.model_copy(
        update={"recommendation": "Revise scene-002 to: ‘The lantern stayed lit.’"}
    )
    ambiguous = curly.model_copy(
        update={"recommendation": "Revise scene-001 to: 'One line' or 'another line'."}
    )

    assert (
        WorkflowEngine._explicit_scene_replacement_word_count(
            curly,
            scene_id="scene-001",
        )
        == 6
    )
    assert (
        WorkflowEngine._explicit_scene_replacement_word_count(
            mismatched_scene,
            scene_id="scene-001",
        )
        is None
    )
    assert (
        WorkflowEngine._explicit_scene_replacement_word_count(
            ambiguous,
            scene_id="scene-001",
        )
        is None
    )


def test_host_requires_explicit_framing_when_fiction_brief_requests_it() -> None:
    engine = object.__new__(WorkflowEngine)
    engine.config = SimpleNamespace(
        content_mode=ContentMode.FICTION,
        content_format=ContentFormat.EXPLAINER,
        output_language=OutputLanguage.FINNISH,
    )
    engine.brief = CreativeBrief(
        idea_direction="Nopea ja selvästi kuvitteellinen selitysvideo.",
        avoid=[
            "Do not mention dragons.",
            "kuvitteellisen mekanismin esittäminen todellisena tietona",
        ],
    )
    literal = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(
                scene_id="scene-001",
                spoken_text="Todellisuudessa pieni miehistö käyttää vipuja keittimen sisällä.",
                pause_after_seconds=0,
            )
        ],
    )

    finding = engine._host_fiction_framing_finding(literal)

    assert finding is not None
    assert finding.finding_id == "constraints:host-explicit-fiction-framing"
    assert finding.scene_id == "scene-001"

    unrelated_avoid_finding = ReviewFinding(
        finding_id="constraints:brief-avoid-001",
        severity="blocking",
        scene_id="scene-001",
        evidence="The script contains an unrelated prohibited detail.",
        recommendation="Remove that detail.",
    )
    assert not engine._has_brief_fiction_framing_finding([unrelated_avoid_finding])

    framing_avoid_finding = unrelated_avoid_finding.model_copy(
        update={"finding_id": "constraints:brief-avoid-002"}
    )
    assert engine._has_brief_fiction_framing_finding([framing_avoid_finding])

    framed = literal.model_copy(deep=True)
    framed.scenes[0].spoken_text = (
        "Kuvittele pieni miehistö käyttämässä vipuja keittimen sisällä."
    )
    assert engine._host_fiction_framing_finding(framed) is None


def test_shot_schedule_is_frame_aligned_and_keeps_parent_scene_ids() -> None:
    engine = object.__new__(WorkflowEngine)
    engine.config = SimpleNamespace(
        fps=30,
        shot_target_seconds=3,
        shot_min_seconds=2,
        shot_max_seconds=5,
    )
    script = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(
                scene_id="scene-001",
                spoken_text="One two three four five six seven eight nine ten eleven twelve.",
                pause_after_seconds=0,
            )
        ],
    )
    media = MediaReference(path="fixture.wav", sha256="0" * 64, mime_type="audio/wav")
    timeline = NarrationTimeline(
        narration_audio=media,
        duration_seconds=10,
        delivery_duration_seconds=10,
        fps=30,
        scenes=[
            TimelineScene(
                scene_id="scene-001",
                audio=media,
                start_seconds=0,
                speech_end_seconds=10,
                end_seconds=10,
            )
        ],
    )
    words = [
        WordTiming(text=word, start_seconds=index * 0.8, end_seconds=index * 0.8 + 0.6)
        for index, word in enumerate(script.scenes[0].spoken_text.split())
    ]
    captions = CaptionBundle(
        enabled=False,
        track=CaptionTrack(language=OutputLanguage.ENGLISH, words=words),
        scene_words={"scene-001": words},
    )
    narration = NarrationBundle(script=script, timeline=timeline, items=[])

    schedule = engine._build_shot_schedule(narration, captions)

    assert [shot["shot_id"] for shot in schedule] == ["shot-001", "shot-002", "shot-003"]
    assert {shot["scene_id"] for shot in schedule} == {"scene-001"}
    assert schedule[0]["start_seconds"] == 0
    assert schedule[-1]["end_seconds"] == 10
    assert all(
        abs(round(shot["start_seconds"] * 30) - shot["start_seconds"] * 30) < 0.0001
        for shot in schedule
    )
    assert all(
        abs(round(shot["end_seconds"] * 30) - shot["end_seconds"] * 30) < 0.0001
        for shot in schedule
    )


def test_shot_schedule_prefers_sentence_and_clause_boundaries() -> None:
    engine = object.__new__(WorkflowEngine)
    engine.config = SimpleNamespace(
        fps=30,
        shot_target_seconds=3,
        shot_min_seconds=2,
        shot_max_seconds=5,
    )
    text = "Alpha beta gamma. Delta epsilon zeta; Eta theta iota."
    script = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(
                scene_id="scene-001",
                spoken_text=text,
                pause_after_seconds=0,
            )
        ],
    )
    media = MediaReference(path="fixture.wav", sha256="0" * 64, mime_type="audio/wav")
    timeline = NarrationTimeline(
        narration_audio=media,
        duration_seconds=10,
        delivery_duration_seconds=10,
        fps=30,
        scenes=[
            TimelineScene(
                scene_id="scene-001",
                audio=media,
                start_seconds=0,
                speech_end_seconds=10,
                end_seconds=10,
            )
        ],
    )
    words = [
        WordTiming(text=word, start_seconds=index, end_seconds=index + 0.8)
        for index, word in enumerate(text.split())
    ]
    narration = NarrationBundle(script=script, timeline=timeline, items=[])
    captions = CaptionBundle(
        enabled=False,
        track=CaptionTrack(language=OutputLanguage.ENGLISH, words=words),
        scene_words={"scene-001": words},
    )

    schedule = engine._build_shot_schedule(narration, captions)

    assert [shot["end_seconds"] for shot in schedule] == [2.8, 5.8, 10.0]
    assert [shot["narration_excerpt"] for shot in schedule] == [
        "Alpha beta gamma.",
        "Delta epsilon zeta;",
        "Eta theta iota.",
    ]


def test_delivery_presets_are_quantitative() -> None:
    slow = resolve_narration_delivery(OutputLanguage.ENGLISH, NarrationPace.SLOW)
    fast = resolve_narration_delivery(OutputLanguage.ENGLISH, NarrationPace.FAST)

    assert slow.target_words_per_second < fast.target_words_per_second
    assert slow.maximum_pause_seconds > fast.maximum_pause_seconds
    assert slow.tempo_multiplier < 1 < fast.tempo_multiplier


def test_factual_narrative_prompt_does_not_fall_back_to_fiction(resolved_config) -> None:
    config = resolved_config.model_copy(
        update={
            "content_mode": ContentMode.FACTUAL,
            "content_format": ContentFormat.NARRATIVE,
        }
    )

    assets = build_frozen_assets(config)
    ideate = assets["prompts"]["ideate"]["instructions"].lower()
    outline = assets["prompts"]["outline"]["instructions"].lower()

    assert "factual narrative" in ideate
    assert "loose inspiration" not in ideate
    assert "factual research pack" in outline


def test_oversized_cadenced_plan_is_rejected_before_a_run_starts() -> None:
    with pytest.raises(ValueError, match="single-plan limit"):
        RawRunConfig(
            duration_seconds=600,
            visual_shot_mode=VisualShotMode.CADENCED,
            voice=VoiceSettings(name="fixture"),
        )


def test_infeasible_scene_shot_bounds_are_rejected() -> None:
    engine = object.__new__(WorkflowEngine)
    engine.config = SimpleNamespace(
        fps=30,
        shot_target_seconds=4.5,
        shot_min_seconds=4,
        shot_max_seconds=5,
    )
    script = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(
                scene_id="scene-001",
                spoken_text="One two three four five six.",
                pause_after_seconds=0,
            )
        ],
    )
    media = MediaReference(path="fixture.wav", sha256="0" * 64, mime_type="audio/wav")
    timeline = NarrationTimeline(
        narration_audio=media,
        duration_seconds=6,
        delivery_duration_seconds=6,
        fps=30,
        scenes=[
            TimelineScene(
                scene_id="scene-001",
                audio=media,
                start_seconds=0,
                speech_end_seconds=6,
                end_seconds=6,
            )
        ],
    )
    narration = NarrationBundle(script=script, timeline=timeline, items=[])

    with pytest.raises(BackendError, match="cannot be divided into Shots"):
        engine._build_shot_schedule(narration, CaptionBundle(enabled=False))


def test_short_scene_uses_one_boundary_limited_shot() -> None:
    engine = object.__new__(WorkflowEngine)
    engine.config = SimpleNamespace(
        fps=30,
        shot_target_seconds=3,
        shot_min_seconds=2,
        shot_max_seconds=5,
    )
    script = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(
                scene_id="scene-001",
                spoken_text="What is happening?",
                pause_after_seconds=0,
            )
        ],
    )
    media = MediaReference(path="fixture.wav", sha256="0" * 64, mime_type="audio/wav")
    timeline = NarrationTimeline(
        narration_audio=media,
        duration_seconds=1.8,
        delivery_duration_seconds=1.8,
        fps=30,
        scenes=[
            TimelineScene(
                scene_id="scene-001",
                audio=media,
                start_seconds=0,
                speech_end_seconds=1.8,
                end_seconds=1.8,
            )
        ],
    )
    narration = NarrationBundle(script=script, timeline=timeline, items=[])

    schedule = engine._build_shot_schedule(narration, CaptionBundle(enabled=False))

    assert schedule == [
        {
            "shot_id": "shot-001",
            "scene_id": "scene-001",
            "narration_excerpt": "What is happening?",
            "start_seconds": 0.0,
            "end_seconds": 1.8,
        }
    ]


def test_duplicate_visual_request_ids_are_rejected() -> None:
    request = TimedImageRequest(
        scene_id="scene-001",
        shot_id="shot-001",
        target_backend_id="local:flux.2-klein-4b",
        prompt="A blue boot compresses white snow crystals, simple diagram, no text.",
        width=1024,
        height=576,
    )

    with pytest.raises(ValueError, match="visual IDs must be nonempty and unique"):
        ImageRequestSet(requests=[request, request])
