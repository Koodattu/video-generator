from __future__ import annotations

from types import SimpleNamespace

import pytest

from video_generator.contracts import (
    ClaimInventory,
    ContentFormat,
    ContentMode,
    EvidenceRecord,
    ExplainerOutline,
    ExplainerOutlineScene,
    ExtractedClaim,
    FactualClaimReview,
    FactualResearchPack,
    FactualRevisedScript,
    FactualReviewReport,
    NarrationDeliverySpec,
    NarrationScript,
    OutputLanguage,
    ResearchSource,
    SceneClaimCoverage,
    SceneClaimExtraction,
    ScriptClaim,
    ScriptScene,
    StyleProfile,
    VisualBrief,
)
from video_generator.errors import BackendError
from video_generator.workflow import (
    FactualVisualDecision,
    FactualVisualDepiction,
    HOST_LEXICAL_POLICY_PREFIX,
    HOST_SELF_CONTAINED_POLICY_PREFIX,
    ClaimReviewDecision,
    ReplacementText,
    VisualBriefContent,
    WorkflowEngine,
)


def test_factual_repair_protects_supported_text_and_handles_evidence_free_scene() -> None:
    script = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(
                scene_id="scene-001",
                spoken_text="Unsupported wording. Supported wording.",
            ),
            ScriptScene(
                scene_id="scene-002",
                spoken_text="Unsupported landing.",
                pause_after_seconds=0,
            ),
        ],
    )
    inventory = ClaimInventory(
        claims=[
            ScriptClaim(
                claim_id="scene-001-claim-001",
                scene_id="scene-001",
                exact_text="Unsupported wording.",
                evidence_ids=["evidence-001"],
            ),
            ScriptClaim(
                claim_id="scene-001-claim-002",
                scene_id="scene-001",
                exact_text="Supported wording.",
                evidence_ids=["evidence-002"],
            ),
            ScriptClaim(
                claim_id="scene-002-claim-001",
                scene_id="scene-002",
                exact_text="Unsupported landing.",
                evidence_ids=[],
            ),
        ]
    )
    review = FactualReviewReport(
        passed=False,
        claims=[
            FactualClaimReview(
                claim_id="scene-001-claim-001",
                verdict="needs_qualification",
                evidence_ids=["evidence-001"],
                rationale="Narrow the wording.",
            ),
            FactualClaimReview(
                claim_id="scene-001-claim-002",
                verdict="supported",
                evidence_ids=["evidence-002"],
                rationale="Directly supported.",
            ),
            FactualClaimReview(
                claim_id="scene-002-claim-001",
                verdict="unsupported",
                evidence_ids=[],
                rationale="No evidence supports this landing.",
            ),
        ],
        summary="Two claims need repair.",
    )
    source = ResearchSource(
        source_id="source-001",
        url="https://example.com/source",
        title="Fixture source",
    )
    research = FactualResearchPack(
        sources=[source],
        evidence=[
            EvidenceRecord(
                evidence_id="evidence-001",
                supported_statement="Qualified wording.",
                source_ids=[source.source_id],
                confidence="high",
            ),
            EvidenceRecord(
                evidence_id="evidence-002",
                supported_statement="Supported wording.",
                source_ids=[source.source_id],
                confidence="high",
            ),
        ],
    )
    engine = object.__new__(WorkflowEngine)
    engine.config = SimpleNamespace(
        output_language=OutputLanguage.FINNISH,
        content_format=ContentFormat.MYTHBUSTER,
    )
    requests = []

    def structured_item(**kwargs):
        requests.append(kwargs)
        replacement = kwargs["output_model"](
            spoken_text="Qualified wording. Supported wording."
        )
        kwargs["invariant"](replacement)
        return replacement, []

    engine._structured_item = structured_item

    repaired, usage = engine._repair_factual_script_once(
        stage="script-revision",
        item_prefix="factual-repair",
        script=script,
        research=research,
        inventory=inventory,
        review=review,
    )

    assert usage == []
    assert len(requests) == 1
    assert requests[0]["input_data"]["protected_exact_texts"] == [
        "Supported wording."
    ]
    assert repaired.scenes[0].spoken_text == "Qualified wording. Supported wording."
    assert repaired.scenes[1].spoken_text == (
        "Palaa nyt alkuun ja katso kokonaisuutta uudelleen."
    )
    assert "jäistä" not in repaired.scenes[1].spoken_text


def test_host_replaces_orphaned_supported_fragment_with_canonical_evidence() -> None:
    source = ResearchSource(
        source_id="source-001",
        url="https://example.com/source",
        title="Fixture source",
    )
    research = FactualResearchPack(
        sources=[source],
        evidence=[
            EvidenceRecord(
                evidence_id="evidence-001",
                supported_statement="Higher salinity lowers the freezing point of water.",
                source_ids=[source.source_id],
                confidence="high",
            )
        ],
    )
    script = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(
                scene_id="scene-001",
                spoken_text="Many people believe salt creates heat, but that is false.",
            )
        ],
    )
    inventory = ClaimInventory(
        claims=[
            ScriptClaim(
                claim_id="scene-001-claim-001",
                scene_id="scene-001",
                exact_text="Many people believe salt creates heat",
            ),
            ScriptClaim(
                claim_id="scene-001-claim-002",
                scene_id="scene-001",
                exact_text="that is false",
                evidence_ids=["evidence-001"],
            ),
        ]
    )
    review = FactualReviewReport(
        passed=False,
        claims=[
            FactualClaimReview(
                claim_id="scene-001-claim-001",
                verdict="unsupported",
                evidence_ids=[],
                rationale="No evidence supports the reported prevalence.",
            ),
            FactualClaimReview(
                claim_id="scene-001-claim-002",
                verdict="supported",
                evidence_ids=["evidence-001"],
                rationale="The bounded record supports the intended correction.",
            ),
        ],
        summary="The prevalence wording requires repair.",
    )
    engine = object.__new__(WorkflowEngine)
    engine.workflow_policy_version = 15
    engine.config = SimpleNamespace(
        output_language=OutputLanguage.ENGLISH,
        content_format=ContentFormat.MYTHBUSTER,
    )
    engine._structured_item = lambda **kwargs: pytest.fail(
        "Evidence-free unsupported wording must be repaired by the host"
    )

    repaired, usage = engine._repair_factual_script_once(
        stage="script-revision",
        item_prefix="factual-repair",
        script=script,
        research=research,
        inventory=inventory,
        review=review,
    )

    assert usage == []
    assert repaired.scenes[0].spoken_text == research.evidence[0].supported_statement
    assert "that is false" not in repaired.scenes[0].spoken_text


@pytest.mark.parametrize(
    "spoken_text,exact_texts",
    [
        (
            "Salt lowers the freezing point.",
            ["Salt lowers the freezing point", "lowers the freezing point"],
        ),
        (
            "No evidence shows salt lowers the freezing point.",
            [
                "No evidence shows salt lowers the freezing point",
                "salt lowers the freezing point",
            ],
        ),
    ],
)
def test_nested_claim_spans_are_rejected_before_review(
    spoken_text: str,
    exact_texts: list[str],
) -> None:
    with pytest.raises(BackendError, match="overlapping semantic spans"):
        WorkflowEngine._normalize_non_overlapping_claim_spans(
            spoken_text,
            [ExtractedClaim(exact_text=exact_text) for exact_text in exact_texts],
        )


def test_exact_duplicate_claim_spans_are_deduplicated() -> None:
    normalized = WorkflowEngine._normalize_non_overlapping_claim_spans(
        "Salt lowers the freezing point.",
        [
            ExtractedClaim(exact_text="Salt lowers the freezing point"),
            ExtractedClaim(exact_text="Salt lowers the freezing point"),
        ],
    )

    assert [claim.exact_text for claim in normalized] == [
        "Salt lowers the freezing point"
    ]


def test_host_builds_complete_sentence_claim_spans_without_model_copying() -> None:
    claims = WorkflowEngine._programmatic_sentence_claims(
        'No evidence shows salt creates heat. Seawater freezes near -1.8 °C. "What next?"'
    )

    assert [claim.exact_text for claim in claims] == [
        "No evidence shows salt creates heat.",
        "Seawater freezes near -1.8 °C.",
        '"What next?"',
    ]


def test_coverage_overlap_is_rejected_instead_of_creating_a_claim_fragment() -> None:
    with pytest.raises(BackendError, match="overlaps an existing exact Claim span"):
        WorkflowEngine._normalize_non_overlapping_claim_spans(
            "Salt lowers the freezing point by changing how water freezes.",
            [
                ExtractedClaim(
                    exact_text=(
                        "Salt lowers the freezing point by changing how water freezes."
                    )
                )
            ],
            occupied_exact_texts=["Salt lowers the freezing point"],
        )


@pytest.mark.parametrize(
    "exact_text,qualification",
    [
        ("häiritsemällä molekyylien järjestäytymistä", ""),
        ("Moni luulee, että suola tuottaa lämpöä.", "Signposted misconception"),
        ("Many people think salt produces heat.", "fictional_framing"),
        ("Imagine salt lowers the freezing point.", "hypothetical_example"),
        ("The illustration shows salt causes melting.", "illustration_description"),
        ("Notice vaccines protect children.", ""),
        ("The illustration shows Earth orbits the Sun.", ""),
        ("What should you inspect next?", ""),
        ("Imagine an icy front step.", ""),
    ],
)
def test_nonfactual_verdict_rejects_assertive_or_prevalence_text(
    exact_text: str,
    qualification: str,
) -> None:
    claim = ScriptClaim(
        claim_id="scene-001-claim-001",
        scene_id="scene-001",
        exact_text=exact_text,
        qualification=qualification,
    )

    assert not WorkflowEngine._claim_text_allows_nonfactual_verdict(claim)


@pytest.mark.parametrize(
    "exact_text",
    [
        "Palaa nyt alkuun ja katso kokonaisuutta uudelleen.",
        "Keskity nyt seuraavaan konkreettiseen kohtaan.",
        "Siirry nyt seuraavaan konkreettiseen kohtaan.",
        "Now return to the opening and reconsider the whole picture.",
        "Focus now on the next concrete point.",
        "Move now to the next concrete point.",
    ],
)
def test_nonfactual_verdict_accepts_only_short_explicit_nonassertive_text(
    exact_text: str,
) -> None:
    claim = ScriptClaim(
        claim_id="scene-001-claim-001",
        scene_id="scene-001",
        exact_text=exact_text,
    )

    assert WorkflowEngine._claim_text_allows_nonfactual_verdict(claim)


def test_disallowed_nonfactual_verdict_is_converted_to_unsupported() -> None:
    claim = ScriptClaim(
        claim_id="scene-001-claim-001",
        scene_id="scene-001",
        exact_text="Salt disrupts molecular organization.",
    )
    decision = ClaimReviewDecision(
        verdict="not_a_factual_claim",
        evidence_ids=[],
        rationale="The reviewer incorrectly treated the mechanism as non-factual framing.",
    )

    normalized = WorkflowEngine._apply_nonfactual_claim_policy(claim, decision)

    assert normalized.verdict == "unsupported"
    assert normalized.evidence_ids == []
    assert "Host policy" in normalized.rationale


def test_allowed_nonfactual_verdict_is_preserved() -> None:
    claim = ScriptClaim(
        claim_id="scene-001-claim-001",
        scene_id="scene-001",
        exact_text="Focus now on the next concrete point.",
    )
    decision = ClaimReviewDecision(
        verdict="not_a_factual_claim",
        evidence_ids=[],
        rationale="The exact text is one host-owned neutral transition supplied for this review.",
    )

    assert WorkflowEngine._apply_nonfactual_claim_policy(claim, decision) is decision


def test_per_claim_review_sees_all_evidence_and_python_attaches_approved_id() -> None:
    source = ResearchSource(
        source_id="source-001",
        url="https://example.com/source",
        title="Fixture source",
    )
    research = FactualResearchPack(
        sources=[source],
        evidence=[
            EvidenceRecord(
                evidence_id="evidence-001",
                supported_statement="Salinity affects the freezing point.",
                source_ids=[source.source_id],
                confidence="high",
            ),
            EvidenceRecord(
                evidence_id="evidence-002",
                supported_statement="Higher salinity lowers the freezing point.",
                source_ids=[source.source_id],
                confidence="high",
            ),
        ],
    )
    script = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(
                scene_id="scene-001",
                spoken_text="Salt lowers the freezing point.",
                pause_after_seconds=0,
            )
        ],
    )
    engine = object.__new__(WorkflowEngine)
    engine.config = SimpleNamespace(
        output_language=OutputLanguage.ENGLISH,
        content_format=ContentFormat.MYTHBUSTER,
    )
    review_inputs = []

    def structured_item(**kwargs):
        output_model = kwargs["output_model"]
        if output_model is SceneClaimExtraction:
            return (
                SceneClaimExtraction(
                    claims=[ExtractedClaim(exact_text="Salt lowers the freezing point")]
                ),
                [],
            )
        if output_model is SceneClaimCoverage:
            return SceneClaimCoverage(missing_claims=[]), []
        review_inputs.append(kwargs["input_data"])
        decision = ClaimReviewDecision(
            verdict="supported",
            evidence_ids=["evidence-002"],
            rationale="The second Evidence Record directly entails the directional wording.",
        )
        kwargs["invariant"](decision)
        return decision, []

    engine._structured_item = structured_item

    inventory, review, usage = engine._factual_audit_by_scene(
        stage="script-revision",
        item_prefix="fixture",
        script=script,
        research=research,
        raise_on_failure=True,
        prior_inventory=None,
        prior_review=None,
        changed_scene_ids=None,
    )

    assert usage == []
    assert [item["evidence_id"] for item in review_inputs[0]["evidence_records"]] == [
        "evidence-001",
        "evidence-002",
    ]
    assert "scene_spoken_text" not in review_inputs[0]
    assert inventory.claims[0].evidence_ids == ["evidence-002"]
    assert review.claims[0].evidence_ids == ["evidence-002"]


def test_policy_sixteen_uses_host_sentence_inventory_and_only_reviews_semantics() -> None:
    source = ResearchSource(
        source_id="source-001",
        url="https://example.com/source",
        title="Fixture source",
    )
    research = FactualResearchPack(
        sources=[source],
        evidence=[
            EvidenceRecord(
                evidence_id="evidence-001",
                supported_statement="Salt lowers the freezing point.",
                source_ids=[source.source_id],
                confidence="high",
            )
        ],
    )
    script = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(
                scene_id="scene-001",
                spoken_text=(
                    "Salt lowers the freezing point. "
                    "Focus now on the next concrete point."
                ),
            )
        ],
    )
    engine = object.__new__(WorkflowEngine)
    engine.workflow_policy_version = 16
    engine.config = SimpleNamespace(
        output_language=OutputLanguage.ENGLISH,
        content_format=ContentFormat.MYTHBUSTER,
    )
    review_inputs: list[dict[str, object]] = []

    def structured_item(**kwargs):
        assert kwargs["output_model"] is ClaimReviewDecision
        review_inputs.append(kwargs["input_data"])
        exact_text = kwargs["input_data"]["claim"]["exact_text"]
        if exact_text == "Focus now on the next concrete point.":
            decision = ClaimReviewDecision(
                verdict="not_a_factual_claim",
                evidence_ids=[],
                rationale="This is the exact host-owned neutral transition supplied in policy.",
            )
        else:
            decision = ClaimReviewDecision(
                verdict="supported",
                evidence_ids=["evidence-001"],
                rationale="The admitted Evidence statement exactly supports this complete sentence.",
            )
        kwargs["invariant"](decision)
        return decision, []

    engine._structured_item = structured_item

    inventory, review, usage = engine._factual_audit_by_scene(
        stage="script-revision",
        item_prefix="fixture",
        script=script,
        research=research,
        raise_on_failure=True,
        prior_inventory=None,
        prior_review=None,
        changed_scene_ids=None,
    )

    assert usage == []
    assert [claim.exact_text for claim in inventory.claims] == [
        "Salt lowers the freezing point.",
        "Focus now on the next concrete point.",
    ]
    assert [claim.evidence_ids for claim in inventory.claims] == [
        ["evidence-001"],
        [],
    ]
    assert review.passed
    assert len(review_inputs) == 2


def test_factual_aggregate_word_fit_preserves_supported_exact_text() -> None:
    source = ResearchSource(
        source_id="source-001",
        url="https://example.com/source",
        title="Fixture source",
    )
    research = FactualResearchPack(
        sources=[source],
        evidence=[
            EvidenceRecord(
                evidence_id="evidence-001",
                supported_statement="Supported wording.",
                source_ids=[source.source_id],
                confidence="high",
            )
        ],
    )
    outline = ExplainerOutline(
        title="Fixture",
        thesis="A bounded thesis.",
        modern_anchor="A familiar object.",
        landing_callback="Return to the opening question.",
        scenes=[
            ExplainerOutlineScene(
                scene_id="scene-001",
                arc_role="landing",
                purpose="Land the explanation.",
                key_point="Supported wording.",
                evidence_ids=["evidence-001"],
                visual_opportunity="A simple diagram.",
                provisional_seconds=5,
            )
        ],
    )
    script = NarrationScript(
        title="Fixture",
        scenes=[ScriptScene(scene_id="scene-001", spoken_text="Supported wording.")],
    )
    engine = object.__new__(WorkflowEngine)
    engine.config = SimpleNamespace(
        output_language=OutputLanguage.ENGLISH,
        content_mode=ContentMode.FACTUAL,
    )
    requests = []

    def structured_item(**kwargs):
        requests.append(kwargs)
        replacement = ReplacementText(
            spoken_text="Supported wording. This repeats only the bounded supported statement."
        )
        kwargs["invariant"](replacement)
        return replacement, []

    engine._structured_item = structured_item

    fitted, usage = engine._fit_scene_local_script_word_range(
        script=script,
        outline=outline,
        research=research,
        scene_word_targets=[{"scene_id": "scene-001", "target_word_count": 9}],
        minimum_total=8,
        target_total=9,
        maximum_total=10,
        protected_exact_texts_by_scene={"scene-001": ["Supported wording."]},
    )

    assert usage == []
    assert requests[0]["input_data"]["protected_exact_texts"] == ["Supported wording."]
    assert "Supported wording." in fitted.scenes[0].spoken_text


def test_duration_word_fit_rejects_rewriting_protected_exact_text() -> None:
    source = ResearchSource(
        source_id="source-001",
        url="https://example.com/source",
        title="Fixture source",
    )
    research = FactualResearchPack(
        sources=[source],
        evidence=[
            EvidenceRecord(
                evidence_id="evidence-001",
                supported_statement="Supported wording.",
                source_ids=[source.source_id],
                confidence="high",
            )
        ],
    )
    outline = ExplainerOutline(
        title="Fixture",
        thesis="A bounded thesis.",
        modern_anchor="A familiar object.",
        landing_callback="Return to the opening.",
        scenes=[
            ExplainerOutlineScene(
                scene_id="scene-001",
                arc_role="landing",
                purpose="Land the explanation.",
                key_point="Supported wording.",
                evidence_ids=["evidence-001"],
                visual_opportunity="A simple diagram.",
                provisional_seconds=5,
            )
        ],
    )
    script = NarrationScript(
        title="Fixture",
        scenes=[ScriptScene(scene_id="scene-001", spoken_text="Supported wording.")],
    )
    engine = object.__new__(WorkflowEngine)
    engine.config = SimpleNamespace(
        output_language=OutputLanguage.ENGLISH,
        content_mode=ContentMode.FACTUAL,
    )
    requests = []

    def structured_item(**kwargs):
        requests.append(kwargs)
        with pytest.raises(BackendError, match="changed already supported exact wording"):
            kwargs["invariant"](
                ReplacementText(
                    spoken_text="Different words fill this bounded duration target safely."
                )
            )
        replacement = ReplacementText(
            spoken_text="Supported wording. This repeats only the bounded supported statement."
        )
        kwargs["invariant"](replacement)
        return replacement, []

    engine._structured_item = structured_item

    fitted, usage, items = engine._fit_duration_repair_word_range(
        script=script,
        original=script,
        scene_repair_targets=[
            {
                "scene_id": "scene-001",
                "minimum_word_count": 8,
                "target_word_count": 9,
                "maximum_word_count": 10,
            }
        ],
        selected_scene_ids={"scene-001"},
        factual_research=research,
        outline=outline,
        protected_exact_texts_by_scene={"scene-001": ["Supported wording."]},
    )

    assert usage == []
    assert items == [
        {"scene_id": "scene-001", "item_id": "duration-word-fit-scene-001"}
    ]
    assert requests[0]["input_data"]["protected_exact_texts"] == ["Supported wording."]
    assert "Supported wording." in fitted.scenes[0].spoken_text


def test_policy_eighteen_factual_duration_fit_adds_canonical_evidence_without_llm() -> None:
    source = ResearchSource(
        source_id="source-001",
        url="https://example.com/source",
        title="Fixture source",
    )
    supported = EvidenceRecord(
        evidence_id="evidence-001",
        supported_statement="Supported wording.",
        source_ids=[source.source_id],
        confidence="high",
    )
    unused = EvidenceRecord(
        evidence_id="evidence-002",
        supported_statement="Fresh water freezes at zero degrees.",
        source_ids=[source.source_id],
        confidence="high",
    )
    research = FactualResearchPack(sources=[source], evidence=[supported, unused])
    outline = ExplainerOutline(
        title="Fixture",
        thesis=supported.supported_statement,
        modern_anchor="A familiar object.",
        landing_callback="Return to the opening.",
        scenes=[
            ExplainerOutlineScene(
                scene_id="scene-001",
                arc_role="correction",
                purpose="State the bounded facts.",
                key_point=supported.supported_statement,
                evidence_ids=[supported.evidence_id, unused.evidence_id],
                visual_opportunity="A simple diagram.",
                provisional_seconds=5,
            )
        ],
    )
    script = NarrationScript(
        title="Fixture",
        scenes=[ScriptScene(scene_id="scene-001", spoken_text=supported.supported_statement)],
    )
    engine = object.__new__(WorkflowEngine)
    engine.workflow_policy_version = 18
    engine.config = SimpleNamespace(
        output_language=OutputLanguage.ENGLISH,
        content_mode=ContentMode.FACTUAL,
        content_format=ContentFormat.EXPLAINER,
    )
    engine._structured_item = lambda **kwargs: pytest.fail(
        "Policy 18 factual Duration fitting must not call the provider"
    )

    fitted, usage, items = engine._fit_duration_repair_word_range(
        script=script,
        original=script,
        scene_repair_targets=[
            {
                "scene_id": "scene-001",
                "minimum_word_count": 8,
                "target_word_count": 9,
                "maximum_word_count": 10,
            }
        ],
        selected_scene_ids={"scene-001"},
        factual_research=research,
        outline=outline,
        protected_exact_texts_by_scene={"scene-001": [supported.supported_statement]},
    )

    assert usage == []
    assert items == []
    assert fitted.scenes[0].spoken_text == (
        "Supported wording. Fresh water freezes at zero degrees."
    )


def test_policy_eighteen_factual_duration_fit_deletes_unprotected_transition() -> None:
    source = ResearchSource(
        source_id="source-001",
        url="https://example.com/source",
        title="Fixture source",
    )
    evidence = EvidenceRecord(
        evidence_id="evidence-001",
        supported_statement="Supported wording.",
        source_ids=[source.source_id],
        confidence="high",
    )
    research = FactualResearchPack(sources=[source], evidence=[evidence])
    script = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(
                scene_id="scene-001",
                spoken_text="Supported wording. Move now to the next concrete point.",
            )
        ],
    )
    engine = object.__new__(WorkflowEngine)
    engine.workflow_policy_version = 18
    engine.config = SimpleNamespace(
        output_language=OutputLanguage.ENGLISH,
        content_mode=ContentMode.FACTUAL,
        content_format=ContentFormat.EXPLAINER,
    )
    engine._structured_item = lambda **kwargs: pytest.fail(
        "Policy 18 factual Duration fitting must not call the provider"
    )

    fitted, usage, items = engine._fit_duration_repair_word_range(
        script=script,
        original=script,
        scene_repair_targets=[
            {
                "scene_id": "scene-001",
                "minimum_word_count": 2,
                "target_word_count": 3,
                "maximum_word_count": 4,
            }
        ],
        selected_scene_ids={"scene-001"},
        factual_research=research,
        outline=None,
        protected_exact_texts_by_scene={"scene-001": [evidence.supported_statement]},
    )

    assert usage == []
    assert items == []
    assert fitted.scenes[0].spoken_text == evidence.supported_statement


def test_policy_twenty_four_factual_duration_fit_respects_delivery_word_ceiling() -> None:
    source = ResearchSource(
        source_id="source-001",
        url="https://example.com/source",
        title="Fixture source",
    )
    supported_statement = " ".join(
        f"evidence{index}" for index in range(1, 56)
    ) + "."
    evidence = EvidenceRecord(
        evidence_id="evidence-001",
        supported_statement=supported_statement,
        source_ids=[source.source_id],
        confidence="high",
    )
    research = FactualResearchPack(sources=[source], evidence=[evidence])
    transition = "Move now to the next concrete point."
    script = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(
                scene_id="scene-001",
                spoken_text=f"{supported_statement} {transition}",
                pause_after_seconds=0,
            )
        ],
    )
    engine = object.__new__(WorkflowEngine)
    engine.workflow_policy_version = 24
    engine.config = SimpleNamespace(
        output_language=OutputLanguage.ENGLISH,
        content_mode=ContentMode.FACTUAL,
        content_format=ContentFormat.EXPLAINER,
        duration_seconds=24,
        fps=30,
        narration_delivery_spec=NarrationDeliverySpec(
            target_words_per_second=2.301,
            minimum_words_per_second=2.025,
            maximum_words_per_second=2.577,
            target_pause_seconds=0.08,
            maximum_pause_seconds=0.75,
        ),
    )
    engine._structured_item = lambda **kwargs: pytest.fail(
        "Policy 24 factual Duration fitting must not call the provider"
    )

    fitted, usage, items = engine._fit_duration_repair_word_range(
        script=script,
        original=script,
        scene_repair_targets=[
            {
                "scene_id": "scene-001",
                "minimum_word_count": 55,
                "target_word_count": 60,
                "maximum_word_count": 64,
            }
        ],
        selected_scene_ids={"scene-001"},
        factual_research=research,
        outline=None,
        protected_exact_texts_by_scene={"scene-001": [supported_statement]},
    )

    output_words = len(fitted.scenes[0].spoken_text.split())
    assert len(script.scenes[0].spoken_text.split()) == 62
    assert output_words == 55
    assert fitted.scenes[0].spoken_text == supported_statement
    assert 2.025 <= output_words / 24 <= 2.577
    assert usage == []
    assert items == []


def test_small_factual_landing_deficit_uses_topic_neutral_callback() -> None:
    engine = object.__new__(WorkflowEngine)
    engine.config = SimpleNamespace(output_language=OutputLanguage.FINNISH)
    script = NarrationScript(
        title="Navigation fixture",
        scenes=[
            ScriptScene(scene_id="scene-001", spoken_text="Tähdet näyttävät suunnan."),
            ScriptScene(
                scene_id="scene-002",
                spoken_text="Todiste tukee väitettä.",
                pause_after_seconds=0,
            ),
        ],
    )

    fitted = engine._add_neutral_factual_transitions_to_word_floor(
        script=script,
        preferred_scene_ids={"scene-002"},
        minimum_total=9,
        maximum_total=14,
    )

    assert fitted.scenes[1].spoken_text == (
        "Todiste tukee väitettä. Palaa nyt alkuun ja katso kokonaisuutta uudelleen."
    )
    assert "jää" not in fitted.scenes[1].spoken_text


def test_host_can_fill_multiple_factual_scenes_with_neutral_transitions() -> None:
    engine = object.__new__(WorkflowEngine)
    engine.workflow_policy_version = 15
    engine.config = SimpleNamespace(output_language=OutputLanguage.FINNISH)
    script = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(scene_id="scene-001", spoken_text="Ensimmäinen."),
            ScriptScene(scene_id="scene-002", spoken_text="Toinen."),
            ScriptScene(scene_id="scene-003", spoken_text="Kolmas."),
        ],
    )

    fitted = engine._add_neutral_factual_transitions_to_word_floor(
        script=script,
        preferred_scene_ids={"scene-001", "scene-002"},
        minimum_total=12,
        maximum_total=15,
    )

    assert sum(len(scene.spoken_text.split()) for scene in fitted.scenes) == 13
    assert fitted.scenes[0].spoken_text.endswith(
        "Keskity nyt seuraavaan konkreettiseen kohtaan."
    )
    assert fitted.scenes[1].spoken_text.endswith(
        "Siirry nyt seuraavaan konkreettiseen kohtaan."
    )
    assert fitted.scenes[2].spoken_text == "Kolmas."


def test_policy_seventeen_uses_role_transitions_for_evidence_free_format_scenes() -> None:
    source = ResearchSource(
        source_id="source-001",
        url="https://example.com/source",
        title="Fixture source",
    )
    research = FactualResearchPack(
        sources=[source],
        evidence=[
            EvidenceRecord(
                evidence_id="evidence-001",
                supported_statement="Salt lowers the freezing point.",
                source_ids=[source.source_id],
                confidence="high",
            )
        ],
    )
    script = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(scene_id="scene-001", spoken_text="Salt is scattered on an icy step."),
            ScriptScene(
                scene_id="scene-002",
                spoken_text="Many people think salt creates heat.",
                pause_after_seconds=0,
            ),
        ],
    )
    inventory = ClaimInventory(
        claims=[
            ScriptClaim(
                claim_id="scene-001-claim-001",
                scene_id="scene-001",
                exact_text=script.scenes[0].spoken_text,
            ),
            ScriptClaim(
                claim_id="scene-002-claim-001",
                scene_id="scene-002",
                exact_text=script.scenes[1].spoken_text,
            ),
        ]
    )
    review = FactualReviewReport(
        passed=False,
        claims=[
            FactualClaimReview(
                claim_id=claim.claim_id,
                verdict="unsupported",
                rationale="The admitted evidence does not support this sentence.",
            )
            for claim in inventory.claims
        ],
    )
    outline = ExplainerOutline(
        title="Fixture",
        thesis="Salt lowers the freezing point.",
        modern_anchor="An icy step.",
        misconception="Salt creates heat.",
        landing_callback="Return to the step.",
        scenes=[
            ExplainerOutlineScene(
                scene_id="scene-001",
                arc_role="modern_hook",
                purpose="Open on the anchor.",
                key_point="An icy step.",
                visual_opportunity="Salt on ice.",
                provisional_seconds=3,
            ),
            ExplainerOutlineScene(
                scene_id="scene-002",
                arc_role="misconception",
                purpose="Question the obvious explanation.",
                key_point="The first explanation may be wrong.",
                visual_opportunity="A visual question.",
                provisional_seconds=3,
            ),
        ],
    )
    engine = object.__new__(WorkflowEngine)
    engine.workflow_policy_version = 17
    engine.config = SimpleNamespace(
        output_language=OutputLanguage.ENGLISH,
        content_format=ContentFormat.MYTHBUSTER,
    )
    engine._structured_item = lambda **kwargs: pytest.fail(
        "Host-owned role repair must not call the provider"
    )

    repaired, usage = engine._repair_factual_script_once(
        stage="script-revision",
        item_prefix="fixture",
        script=script,
        research=research,
        inventory=inventory,
        review=review,
        outline=outline,
    )

    assert usage == []
    assert [scene.spoken_text for scene in repaired.scenes] == [
        "What is really happening here?",
        "What if the obvious explanation is wrong?",
    ]


def test_policy_seventeen_removes_only_failed_sentence_from_mixed_scene() -> None:
    source = ResearchSource(
        source_id="source-001",
        url="https://example.com/source",
        title="Fixture source",
    )
    evidence = EvidenceRecord(
        evidence_id="evidence-001",
        supported_statement="Salt lowers the freezing point.",
        source_ids=[source.source_id],
        confidence="high",
    )
    research = FactualResearchPack(sources=[source], evidence=[evidence])
    script = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(
                scene_id="scene-001",
                spoken_text="Unsupported bridge. Salt lowers the freezing point.",
                pause_after_seconds=0,
            )
        ],
    )
    inventory = ClaimInventory(
        claims=[
            ScriptClaim(
                claim_id="scene-001-claim-001",
                scene_id="scene-001",
                exact_text="Unsupported bridge.",
            ),
            ScriptClaim(
                claim_id="scene-001-claim-002",
                scene_id="scene-001",
                exact_text=evidence.supported_statement,
                evidence_ids=[evidence.evidence_id],
            ),
        ]
    )
    review = FactualReviewReport(
        passed=False,
        claims=[
            FactualClaimReview(
                claim_id="scene-001-claim-001",
                verdict="unsupported",
                rationale="The bridge has no admitted evidence.",
            ),
            FactualClaimReview(
                claim_id="scene-001-claim-002",
                verdict="supported",
                evidence_ids=[evidence.evidence_id],
                rationale="The evidence directly supports this exact sentence.",
            ),
        ],
    )
    engine = object.__new__(WorkflowEngine)
    engine.workflow_policy_version = 17
    engine.config = SimpleNamespace(
        output_language=OutputLanguage.ENGLISH,
        content_format=ContentFormat.EXPLAINER,
    )
    engine._structured_item = lambda **kwargs: pytest.fail(
        "Sentence-local host repair must not call the provider"
    )

    repaired, usage = engine._repair_factual_script_once(
        stage="script-revision",
        item_prefix="fixture",
        script=script,
        research=research,
        inventory=inventory,
        review=review,
    )

    assert usage == []
    assert repaired.scenes[0].spoken_text == evidence.supported_statement


def test_policy_seventeen_classifies_host_transition_without_provider_call() -> None:
    source = ResearchSource(
        source_id="source-001",
        url="https://example.com/source",
        title="Fixture source",
    )
    research = FactualResearchPack(
        sources=[source],
        evidence=[
            EvidenceRecord(
                evidence_id="evidence-001",
                supported_statement="Salt lowers the freezing point.",
                source_ids=[source.source_id],
                confidence="high",
            )
        ],
    )
    script = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(
                scene_id="scene-001",
                spoken_text="What is really happening here?",
                pause_after_seconds=0,
            )
        ],
    )
    engine = object.__new__(WorkflowEngine)
    engine.workflow_policy_version = 17
    engine.config = SimpleNamespace(
        output_language=OutputLanguage.ENGLISH,
        content_format=ContentFormat.MYTHBUSTER,
    )
    engine._structured_item = lambda **kwargs: pytest.fail(
        "Exact host-owned framing must not call the provider"
    )

    inventory, review, usage = engine._factual_audit_by_scene(
        stage="script-revision",
        item_prefix="fixture",
        script=script,
        research=research,
        raise_on_failure=True,
        prior_inventory=None,
        prior_review=None,
        changed_scene_ids=None,
    )

    assert usage == []
    assert inventory.claims[0].exact_text == script.scenes[0].spoken_text
    assert review.claims[0].verdict == "not_a_factual_claim"


def test_policy_twenty_two_accepts_exact_finnish_evidence_without_provider_call() -> None:
    source = ResearchSource(
        source_id="source-001",
        url="https://example.com/source",
        title="Fixture source",
    )
    statement = "Mitä suolaisempaa vesi on, sitä alhaisempi on sen jäätymispiste."
    research = FactualResearchPack(
        sources=[source],
        evidence=[
            EvidenceRecord(
                evidence_id="evidence-001",
                supported_statement=statement,
                source_ids=[source.source_id],
                confidence="high",
            )
        ],
    )
    script = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(
                scene_id="scene-001",
                spoken_text=statement,
                pause_after_seconds=0,
            )
        ],
    )
    engine = object.__new__(WorkflowEngine)
    engine.workflow_policy_version = 22
    engine.config = SimpleNamespace(
        output_language=OutputLanguage.FINNISH,
        content_format=ContentFormat.MYTHBUSTER,
    )
    engine._structured_item = lambda **kwargs: pytest.fail(
        "An exact admitted Evidence statement must not call the provider"
    )

    inventory, review, usage = engine._factual_audit_by_scene(
        stage="script-revision",
        item_prefix="fixture",
        script=script,
        research=research,
        raise_on_failure=True,
        prior_inventory=None,
        prior_review=None,
        changed_scene_ids=None,
    )

    assert usage == []
    assert inventory.claims[0].exact_text == statement
    assert inventory.claims[0].evidence_ids == ["evidence-001"]
    assert review.passed
    assert review.claims[0].verdict == "supported"
    assert review.claims[0].evidence_ids == ["evidence-001"]


def test_policy_seventeen_requires_self_contained_factual_sentence() -> None:
    source = ResearchSource(
        source_id="source-001",
        url="https://example.com/source",
        title="Fixture source",
    )
    research = FactualResearchPack(
        sources=[source],
        evidence=[
            EvidenceRecord(
                evidence_id="evidence-001",
                supported_statement="Salt disrupts molecular movement.",
                source_ids=[source.source_id],
                confidence="high",
            )
        ],
    )
    script = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(
                scene_id="scene-001",
                spoken_text="This disrupts molecular movement.",
                pause_after_seconds=0,
            )
        ],
    )
    engine = object.__new__(WorkflowEngine)
    engine.workflow_policy_version = 17
    engine.config = SimpleNamespace(
        output_language=OutputLanguage.ENGLISH,
        content_format=ContentFormat.EXPLAINER,
    )

    def structured_item(**kwargs):
        decision = ClaimReviewDecision(
            verdict="supported",
            evidence_ids=["evidence-001"],
            rationale="The supplied evidence describes the same molecular effect directly.",
        )
        kwargs["invariant"](decision)
        return decision, []

    engine._structured_item = structured_item

    _, review, usage = engine._factual_audit_by_scene(
        stage="script-revision",
        item_prefix="fixture",
        script=script,
        research=research,
        raise_on_failure=False,
        prior_inventory=None,
        prior_review=None,
        changed_scene_ids=None,
    )

    assert usage == []
    assert not review.passed
    assert review.claims[0].verdict == "needs_qualification"
    assert "self-contained" in review.claims[0].rationale


def test_policy_seventeen_word_floor_prefers_canonical_evidence() -> None:
    engine = object.__new__(WorkflowEngine)
    engine.workflow_policy_version = 17
    engine.config = SimpleNamespace(output_language=OutputLanguage.ENGLISH)
    script = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(
                scene_id="scene-001",
                spoken_text="What is really happening here?",
                pause_after_seconds=0,
            )
        ],
    )

    fitted = engine._add_neutral_factual_transitions_to_word_floor(
        script=script,
        preferred_scene_ids={"scene-001"},
        minimum_total=10,
        maximum_total=12,
        canonical_evidence_by_scene={
            "scene-001": ["Salt lowers the freezing point of water."]
        },
    )

    assert fitted.scenes[0].spoken_text == (
        "What is really happening here? Salt lowers the freezing point of water."
    )


def test_policy_eighteen_rejects_model_approved_bridge_claim() -> None:
    source = ResearchSource(
        source_id="source-001",
        url="https://example.com/source",
        title="Fixture source",
    )
    research = FactualResearchPack(
        sources=[source],
        evidence=[
            EvidenceRecord(
                evidence_id="evidence-001",
                supported_statement="Salinity lowers the freezing point of water.",
                source_ids=[source.source_id],
                confidence="high",
            )
        ],
    )
    script = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(
                scene_id="scene-001",
                spoken_text="Salt grains melt an icy step.",
                pause_after_seconds=0,
            )
        ],
    )
    engine = object.__new__(WorkflowEngine)
    engine.workflow_policy_version = 18
    engine.config = SimpleNamespace(
        output_language=OutputLanguage.ENGLISH,
        content_format=ContentFormat.MYTHBUSTER,
    )

    def structured_item(**kwargs):
        decision = ClaimReviewDecision(
            verdict="supported",
            evidence_ids=["evidence-001"],
            rationale="The freezing-point evidence appears to support the stated result.",
        )
        kwargs["invariant"](decision)
        return decision, []

    engine._structured_item = structured_item

    _, review, usage = engine._factual_audit_by_scene(
        stage="script-revision",
        item_prefix="fixture",
        script=script,
        research=research,
        raise_on_failure=False,
        prior_inventory=None,
        prior_review=None,
        changed_scene_ids=None,
    )

    assert usage == []
    assert not review.passed
    assert review.claims[0].verdict == "needs_qualification"
    assert review.claims[0].rationale.startswith(HOST_LEXICAL_POLICY_PREFIX)


def test_policy_eighteen_lexical_gate_accepts_canonical_and_close_finnish_text() -> None:
    assert WorkflowEngine._evidence_statement_supports_claim_lexically(
        "Veden suolaisuus alentaa sen jäätymispistettä.",
        "Veden suolaisuus alentaa sen jäätymispistettä.",
    )
    assert WorkflowEngine._evidence_statement_supports_claim_lexically(
        "Todellisuudessa suola liukenee veteen, jolloin vesi toimii liuottimena.",
        "Suola on liukeneva aine ja vesi toimii liuottimena.",
    )


def test_policy_eighteen_requires_words_and_numbers_in_one_evidence_record() -> None:
    source = ResearchSource(
        source_id="source-001",
        url="https://example.com/source",
        title="Fixture source",
    )
    evidence_by_id = {
        "evidence-001": EvidenceRecord(
            evidence_id="evidence-001",
            supported_statement="Seawater freezes at a lower temperature than fresh water.",
            source_ids=[source.source_id],
            confidence="high",
        ),
        "evidence-002": EvidenceRecord(
            evidence_id="evidence-002",
            supported_statement="The measured value is -1,8 °C.",
            source_ids=[source.source_id],
            confidence="high",
        ),
        "evidence-003": EvidenceRecord(
            evidence_id="evidence-003",
            supported_statement="Seawater freezes at about -1,8 °C.",
            source_ids=[source.source_id],
            confidence="high",
        ),
    }
    claim = ScriptClaim(
        claim_id="scene-001-claim-001",
        scene_id="scene-001",
        exact_text="Seawater freezes at about −1.8 °C.",
    )
    split_decision = ClaimReviewDecision(
        verdict="supported",
        evidence_ids=["evidence-001", "evidence-002"],
        rationale="The two records collectively appear to contain the needed details.",
    )
    combined_decision = split_decision.model_copy(
        update={"evidence_ids": ["evidence-003"]}
    )

    rejected = WorkflowEngine._apply_lexical_claim_policy(
        claim,
        split_decision,
        evidence_by_id,
    )
    accepted = WorkflowEngine._apply_lexical_claim_policy(
        claim,
        combined_decision,
        evidence_by_id,
    )

    assert rejected.verdict == "needs_qualification"
    assert accepted.verdict == "supported"


def test_policy_eighteen_marks_split_evidence_qualification_for_host_repair() -> None:
    source = ResearchSource(
        source_id="source-001",
        url="https://example.com/source",
        title="Fixture source",
    )
    evidence_by_id = {
        "evidence-001": EvidenceRecord(
            evidence_id="evidence-001",
            supported_statement="Meriveden jäätymispiste on noin −1,9 °C.",
            source_ids=[source.source_id],
            confidence="high",
        ),
        "evidence-002": EvidenceRecord(
            evidence_id="evidence-002",
            supported_statement="Suolaisempi vesi on raskaampaa kuin makea vesi.",
            source_ids=[source.source_id],
            confidence="high",
        ),
    }
    claim = ScriptClaim(
        claim_id="scene-001-claim-001",
        scene_id="scene-001",
        exact_text=(
            "Meriveden jäätymispiste on noin −1,9 °C, ja suolaisempi vesi on "
            "raskaampaa kuin makea vesi."
        ),
    )
    decision = ClaimReviewDecision(
        verdict="needs_qualification",
        evidence_ids=list(evidence_by_id),
        rationale="The reviewer combined two records when qualifying the compound sentence.",
    )

    normalized = WorkflowEngine._apply_lexical_claim_policy(
        claim,
        decision,
        evidence_by_id,
    )

    assert normalized.verdict == "needs_qualification"
    assert normalized.rationale.startswith(HOST_LEXICAL_POLICY_PREFIX)


def test_policy_eighteen_keeps_evidence_free_qualification_for_bounded_edit() -> None:
    claim = ScriptClaim(
        claim_id="scene-001-claim-001",
        scene_id="scene-001",
        exact_text="Salt always lowers the freezing point.",
    )
    decision = ClaimReviewDecision(
        verdict="needs_qualification",
        evidence_ids=[],
        rationale="The absolute wording needs a narrower claim.",
    )

    normalized = WorkflowEngine._apply_lexical_claim_policy(claim, decision, {})

    assert normalized is decision


def test_policy_eighteen_repairs_split_evidence_compound_without_editor_call() -> None:
    source = ResearchSource(
        source_id="source-001",
        url="https://example.com/source",
        title="Fixture source",
    )
    ion_statement = (
        "Suola liuetessaan veteen irrottaa ionit kidehilasta ja ne pääsevät liikkeelle."
    )
    freezing_statement = "Meriveden jäätymispiste on noin −1,9 °C."
    density_statement = "Suolaisempi vesi on raskaampaa kuin makea vesi."
    evidence = [
        EvidenceRecord(
            evidence_id="evidence-001",
            supported_statement=ion_statement,
            source_ids=[source.source_id],
            confidence="high",
        ),
        EvidenceRecord(
            evidence_id="evidence-002",
            supported_statement=freezing_statement,
            source_ids=[source.source_id],
            confidence="high",
        ),
        EvidenceRecord(
            evidence_id="evidence-003",
            supported_statement=density_statement,
            source_ids=[source.source_id],
            confidence="high",
        ),
    ]
    research = FactualResearchPack(sources=[source], evidence=evidence)
    transition = "Siirry nyt seuraavaan konkreettiseen kohtaan."
    compound = (
        "Meriveden jäätymispiste on noin −1,9 °C, ja suolaisempi vesi on "
        "raskaampaa kuin makea vesi."
    )
    script = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(
                scene_id="scene-001",
                spoken_text=f"{transition} {ion_statement} {compound} {density_statement}",
                pause_after_seconds=0,
            )
        ],
    )
    claims = [
        ScriptClaim(
            claim_id="scene-001-claim-001",
            scene_id="scene-001",
            exact_text=transition,
        ),
        ScriptClaim(
            claim_id="scene-001-claim-002",
            scene_id="scene-001",
            exact_text=ion_statement,
            evidence_ids=["evidence-001"],
        ),
        ScriptClaim(
            claim_id="scene-001-claim-003",
            scene_id="scene-001",
            exact_text=compound,
            evidence_ids=["evidence-002", "evidence-003"],
        ),
        ScriptClaim(
            claim_id="scene-001-claim-004",
            scene_id="scene-001",
            exact_text=density_statement,
            evidence_ids=["evidence-003"],
        ),
    ]
    compound_decision = WorkflowEngine._apply_lexical_claim_policy(
        claims[2],
        ClaimReviewDecision(
            verdict="needs_qualification",
            evidence_ids=["evidence-002", "evidence-003"],
            rationale="The compound combines two separately supported facts.",
        ),
        {item.evidence_id: item for item in evidence},
    )
    review = FactualReviewReport(
        passed=False,
        claims=[
            FactualClaimReview(
                claim_id=claims[0].claim_id,
                verdict="not_a_factual_claim",
                rationale="This exact host-owned transition is non-assertive.",
            ),
            FactualClaimReview(
                claim_id=claims[1].claim_id,
                verdict="supported",
                evidence_ids=["evidence-001"],
                rationale="The canonical Evidence supports the sentence directly.",
            ),
            FactualClaimReview(
                claim_id=claims[2].claim_id,
                verdict=compound_decision.verdict,
                evidence_ids=compound_decision.evidence_ids,
                rationale=compound_decision.rationale,
            ),
            FactualClaimReview(
                claim_id=claims[3].claim_id,
                verdict="supported",
                evidence_ids=["evidence-003"],
                rationale="The canonical Evidence supports the sentence directly.",
            ),
        ],
    )
    engine = object.__new__(WorkflowEngine)
    engine.workflow_policy_version = 18
    engine.config = SimpleNamespace(
        output_language=OutputLanguage.FINNISH,
        content_format=ContentFormat.MYTHBUSTER,
    )
    engine._structured_item = lambda **kwargs: pytest.fail(
        "Split-evidence repair must not call the editor"
    )

    repaired, usage = engine._repair_factual_script_once(
        stage="narration",
        item_prefix="fixture",
        script=script,
        research=research,
        inventory=ClaimInventory(claims=claims),
        review=review,
    )

    assert usage == []
    assert repaired.scenes[0].spoken_text == (
        f"{transition} {ion_statement} {freezing_statement} {density_statement}"
    )


def test_policy_eighteen_repairs_lexical_hook_with_host_transition() -> None:
    source = ResearchSource(
        source_id="source-001",
        url="https://example.com/source",
        title="Fixture source",
    )
    evidence = EvidenceRecord(
        evidence_id="evidence-001",
        supported_statement="Salinity lowers the freezing point of water.",
        source_ids=[source.source_id],
        confidence="high",
    )
    research = FactualResearchPack(sources=[source], evidence=[evidence])
    script = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(
                scene_id="scene-001",
                spoken_text="Salt grains melt an icy step.",
                pause_after_seconds=0,
            )
        ],
    )
    inventory = ClaimInventory(
        claims=[
            ScriptClaim(
                claim_id="scene-001-claim-001",
                scene_id="scene-001",
                exact_text=script.scenes[0].spoken_text,
                evidence_ids=[evidence.evidence_id],
            )
        ]
    )
    review = FactualReviewReport(
        passed=False,
        claims=[
            FactualClaimReview(
                claim_id="scene-001-claim-001",
                verdict="needs_qualification",
                evidence_ids=[evidence.evidence_id],
                rationale=f"{HOST_LEXICAL_POLICY_PREFIX} rejected the bridge claim.",
            )
        ],
    )
    outline = ExplainerOutline(
        title="Fixture",
        thesis=evidence.supported_statement,
        modern_anchor="An icy step.",
        landing_callback="Return to the step.",
        scenes=[
            ExplainerOutlineScene(
                scene_id="scene-001",
                arc_role="modern_hook",
                purpose="Open on the anchor.",
                key_point="An icy step.",
                visual_opportunity="Salt on ice.",
                provisional_seconds=3,
            )
        ],
    )
    engine = object.__new__(WorkflowEngine)
    engine.workflow_policy_version = 18
    engine.config = SimpleNamespace(
        output_language=OutputLanguage.ENGLISH,
        content_format=ContentFormat.MYTHBUSTER,
    )
    engine._structured_item = lambda **kwargs: pytest.fail(
        "Host lexical repair must not call the provider"
    )

    repaired, usage = engine._repair_factual_script_once(
        stage="script-revision",
        item_prefix="fixture",
        script=script,
        research=research,
        inventory=inventory,
        review=review,
        outline=outline,
    )

    assert usage == []
    assert repaired.scenes[0].spoken_text == "What is really happening here?"


def test_policy_eighteen_repairs_self_contained_correction_with_canonical_text() -> None:
    source = ResearchSource(
        source_id="source-001",
        url="https://example.com/source",
        title="Fixture source",
    )
    evidence = EvidenceRecord(
        evidence_id="evidence-001",
        supported_statement="Salinity lowers the freezing point of water.",
        source_ids=[source.source_id],
        confidence="high",
    )
    research = FactualResearchPack(sources=[source], evidence=[evidence])
    script = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(
                scene_id="scene-001",
                spoken_text="This lowers the freezing point of water.",
                pause_after_seconds=0,
            )
        ],
    )
    inventory = ClaimInventory(
        claims=[
            ScriptClaim(
                claim_id="scene-001-claim-001",
                scene_id="scene-001",
                exact_text=script.scenes[0].spoken_text,
                evidence_ids=[evidence.evidence_id],
            )
        ]
    )
    review = FactualReviewReport(
        passed=False,
        claims=[
            FactualClaimReview(
                claim_id="scene-001-claim-001",
                verdict="needs_qualification",
                evidence_ids=[evidence.evidence_id],
                rationale=(
                    f"{HOST_SELF_CONTAINED_POLICY_PREFIX} rejected the unresolved reference."
                ),
            )
        ],
    )
    outline = ExplainerOutline(
        title="Fixture",
        thesis=evidence.supported_statement,
        modern_anchor="An icy step.",
        landing_callback="Return to the step.",
        scenes=[
            ExplainerOutlineScene(
                scene_id="scene-001",
                arc_role="correction",
                purpose="State the correction.",
                key_point=evidence.supported_statement,
                evidence_ids=[evidence.evidence_id],
                visual_opportunity="A freezing-point marker.",
                provisional_seconds=3,
            )
        ],
    )
    engine = object.__new__(WorkflowEngine)
    engine.workflow_policy_version = 18
    engine.config = SimpleNamespace(
        output_language=OutputLanguage.ENGLISH,
        content_format=ContentFormat.MYTHBUSTER,
    )
    engine._structured_item = lambda **kwargs: pytest.fail(
        "Host self-contained repair must not call the provider"
    )

    repaired, usage = engine._repair_factual_script_once(
        stage="script-revision",
        item_prefix="fixture",
        script=script,
        research=research,
        inventory=inventory,
        review=review,
        outline=outline,
    )

    assert usage == []
    assert repaired.scenes[0].spoken_text == evidence.supported_statement


def test_policy_eighteen_deduplicates_canonical_host_repair() -> None:
    source = ResearchSource(
        source_id="source-001",
        url="https://example.com/source",
        title="Fixture source",
    )
    evidence = EvidenceRecord(
        evidence_id="evidence-001",
        supported_statement="Salinity lowers the freezing point of water.",
        source_ids=[source.source_id],
        confidence="high",
    )
    research = FactualResearchPack(sources=[source], evidence=[evidence])
    script = NarrationScript(
        title="Fixture",
        scenes=[
            ScriptScene(
                scene_id="scene-001",
                spoken_text=(
                    "Salt melts ice. Salinity lowers the freezing point of water."
                ),
                pause_after_seconds=0,
            )
        ],
    )
    inventory = ClaimInventory(
        claims=[
            ScriptClaim(
                claim_id="scene-001-claim-001",
                scene_id="scene-001",
                exact_text="Salt melts ice.",
                evidence_ids=[evidence.evidence_id],
            ),
            ScriptClaim(
                claim_id="scene-001-claim-002",
                scene_id="scene-001",
                exact_text=evidence.supported_statement,
                evidence_ids=[evidence.evidence_id],
            ),
        ]
    )
    review = FactualReviewReport(
        passed=False,
        claims=[
            FactualClaimReview(
                claim_id="scene-001-claim-001",
                verdict="needs_qualification",
                evidence_ids=[evidence.evidence_id],
                rationale=f"{HOST_LEXICAL_POLICY_PREFIX} rejected the bridge claim.",
            ),
            FactualClaimReview(
                claim_id="scene-001-claim-002",
                verdict="supported",
                evidence_ids=[evidence.evidence_id],
                rationale="The Evidence exactly matches the canonical sentence.",
            ),
        ],
    )
    engine = object.__new__(WorkflowEngine)
    engine.workflow_policy_version = 18
    engine.config = SimpleNamespace(
        output_language=OutputLanguage.ENGLISH,
        content_format=ContentFormat.MYTHBUSTER,
    )
    engine._structured_item = lambda **kwargs: pytest.fail(
        "Host deduplication must not call the provider"
    )

    repaired, usage = engine._repair_factual_script_once(
        stage="script-revision",
        item_prefix="fixture",
        script=script,
        research=research,
        inventory=inventory,
        review=review,
    )

    assert usage == []
    assert repaired.scenes[0].spoken_text == evidence.supported_statement


def test_policy_eighteen_word_floor_uses_uncited_global_evidence() -> None:
    source = ResearchSource(
        source_id="source-001",
        url="https://example.com/source",
        title="Fixture source",
    )
    used = EvidenceRecord(
        evidence_id="evidence-001",
        supported_statement="Salinity lowers the freezing point of water.",
        source_ids=[source.source_id],
        confidence="high",
    )
    unused = EvidenceRecord(
        evidence_id="evidence-002",
        supported_statement="Fresh water freezes at 0 °C under normal pressure.",
        source_ids=[source.source_id],
        confidence="high",
    )
    research = FactualResearchPack(sources=[source], evidence=[used, unused])
    outline = ExplainerOutline(
        title="Fixture",
        thesis=used.supported_statement,
        modern_anchor="An icy step.",
        landing_callback="Return to the step.",
        scenes=[
            ExplainerOutlineScene(
                scene_id="scene-001",
                arc_role="correction",
                purpose="State the correction.",
                key_point=used.supported_statement,
                evidence_ids=[used.evidence_id],
                visual_opportunity="A freezing-point marker.",
                provisional_seconds=3,
            )
        ],
    )

    statements = WorkflowEngine._canonical_evidence_statements_by_scene(
        outline,
        research,
        excluded_evidence_ids={used.evidence_id},
    )

    assert statements == {"scene-001": [unused.supported_statement]}


def test_factual_visual_grounding_contains_only_reviewed_scene_evidence() -> None:
    source = ResearchSource(
        source_id="source-001",
        url="https://example.com/source",
        title="Fixture source",
    )
    evidence = EvidenceRecord(
        evidence_id="evidence-001",
        supported_statement="Salt lowers the freezing point.",
        source_ids=[source.source_id],
        confidence="high",
    )
    research = FactualResearchPack(sources=[source], evidence=[evidence])
    inventory = ClaimInventory(
        claims=[
            ScriptClaim(
                claim_id="scene-001-claim-001",
                scene_id="scene-001",
                exact_text="What is really happening here?",
            ),
            ScriptClaim(
                claim_id="scene-002-claim-001",
                scene_id="scene-002",
                exact_text=evidence.supported_statement,
                evidence_ids=[evidence.evidence_id],
            ),
        ]
    )
    revised = FactualRevisedScript(
        script=NarrationScript(
            title="Fixture",
            scenes=[
                ScriptScene(
                    scene_id="scene-001",
                    spoken_text="What is really happening here?",
                ),
                ScriptScene(
                    scene_id="scene-002",
                    spoken_text=evidence.supported_statement,
                    pause_after_seconds=0,
                ),
            ],
        ),
        claim_inventory=inventory,
        factual_review=FactualReviewReport(
            passed=True,
            claims=[
                FactualClaimReview(
                    claim_id="scene-001-claim-001",
                    verdict="not_a_factual_claim",
                    rationale="This exact host-owned question is non-assertive framing.",
                ),
                FactualClaimReview(
                    claim_id="scene-002-claim-001",
                    verdict="supported",
                    evidence_ids=[evidence.evidence_id],
                    rationale="The evidence directly supports this exact sentence.",
                ),
            ],
        ),
    )

    grounding = WorkflowEngine._factual_visual_grounding_payload(
        revised.claim_inventory,
        revised.factual_review,
        research,
    )
    first = WorkflowEngine._factual_visual_grounding_for_scene(grounding, "scene-001")
    second = WorkflowEngine._factual_visual_grounding_for_scene(grounding, "scene-002")

    assert first["supported_claims"] == []
    assert first["nonfactual_framing"] == [
        {"scene_id": "scene-001", "exact_text": "What is really happening here?"}
    ]
    assert first["allowed_evidence_records"] == []
    assert second["supported_claims"][0]["exact_text"] == evidence.supported_statement
    assert second["allowed_evidence_records"][0]["evidence_id"] == evidence.evidence_id


def test_factual_visual_grounding_is_limited_to_the_current_shot_excerpt() -> None:
    grounding = {
        "supported_claims": [
            {
                "scene_id": "scene-001",
                "exact_text": "Water molecules penetrate an ionic crystal lattice.",
                "evidence_ids": ["evidence-001"],
            },
            {
                "scene_id": "scene-001",
                "exact_text": "Seawater freezes at minus one point eight degrees Celsius.",
                "evidence_ids": ["evidence-002"],
            },
        ],
        "nonfactual_framing": [],
        "evidence_records": [
            {"evidence_id": "evidence-001", "supported_statement": "Lattice evidence."},
            {"evidence_id": "evidence-002", "supported_statement": "Temperature evidence."},
        ],
    }

    target = WorkflowEngine._factual_visual_grounding_for_target(
        grounding,
        scene_id="scene-001",
        narration_excerpt="Water molecules penetrate the ionic crystal lattice.",
        timed_visuals=True,
    )

    assert [claim["evidence_ids"] for claim in target["supported_claims"]] == [
        ["evidence-001"]
    ]
    assert [item["evidence_id"] for item in target["allowed_evidence_records"]] == [
        "evidence-001"
    ]


def test_factual_visual_grounding_does_not_match_on_a_shared_number_alone() -> None:
    grounding = {
        "supported_claims": [
            {
                "scene_id": "scene-001",
                "exact_text": "The first sample contains 30 percent salt.",
                "evidence_ids": ["evidence-001"],
            }
        ],
        "nonfactual_framing": [],
        "evidence_records": [
            {"evidence_id": "evidence-001", "supported_statement": "Salt evidence."}
        ],
    }

    target = WorkflowEngine._factual_visual_grounding_for_target(
        grounding,
        scene_id="scene-001",
        narration_excerpt="The unrelated crowd contained 30 people.",
        timed_visuals=True,
    )

    assert target["supported_claims"] == []
    assert target["allowed_evidence_records"] == []


def test_policy_twenty_five_assigns_a_split_claim_only_to_its_terminal_shot() -> None:
    claim = (
        "Meriveden jäätymispiste on -1,8 °C ja Itämeren murtoveden "
        "jäätymispiste on noin -0,3 °C."
    )
    grounding = {
        "supported_claims": [
            {
                "scene_id": "scene-003",
                "exact_text": claim,
                "evidence_ids": ["evidence-003"],
            }
        ],
        "nonfactual_framing": [],
        "evidence_records": [
            {
                "evidence_id": "evidence-003",
                "supported_statement": claim,
            }
        ],
    }

    early = WorkflowEngine._factual_visual_grounding_for_target(
        grounding,
        scene_id="scene-003",
        narration_excerpt="jäätymispiste on -1,8 °C ja Itämeren",
        timed_visuals=True,
        terminal_claim_fragments_only=True,
    )
    terminal = WorkflowEngine._factual_visual_grounding_for_target(
        grounding,
        scene_id="scene-003",
        narration_excerpt="murtoveden jäätymispiste on noin -0,3 °C.",
        timed_visuals=True,
        terminal_claim_fragments_only=True,
    )

    assert early["supported_claims"] == []
    assert early["allowed_evidence_records"] == []
    assert terminal["supported_claims"] == grounding["supported_claims"]
    assert terminal["allowed_evidence_records"] == grounding["evidence_records"]


def _visual_content(*, action: str, state_after_scene: list[str]) -> VisualBriefContent:
    return VisualBriefContent(
        story_moment="A neutral view of coarse salt beside a frozen step.",
        subjects=["coarse salt", "frozen step"],
        action=action,
        emotion="curious and focused",
        environment="a sparse paper-cut winter setting",
        composition="a readable centered still-life composition",
        must_show=["coarse salt", "frozen step"],
        must_avoid=["text", "labels"],
        state_after_scene=state_after_scene,
    )


def test_factual_visual_gate_repairs_one_bounded_visual_and_rechecks() -> None:
    engine = object.__new__(WorkflowEngine)
    engine.workflow_policy_version = 20
    initial = _visual_content(
        action="Salt and the frozen step remain motionless in a generic arrangement.",
        state_after_scene=["the generic arrangement remains unchanged"],
    )
    repaired = _visual_content(
        action=(
            "A split comparison shows saline water remaining liquid beside frozen fresh water."
        ),
        state_after_scene=["the two material states remain clearly contrasted"],
    )
    calls: list[dict] = []
    review_verdicts = iter(["underillustrated", "grounded"])

    def structured_item(**kwargs):
        calls.append(kwargs)
        if kwargs["task_id"] == "factual_review":
            value = FactualVisualDecision(
                verdict=next(review_verdicts),
                rationale="The bounded candidate was checked against the supplied visual evidence.",
            )
        else:
            value = repaired
        kwargs["invariant"](value)
        return value, []

    engine._structured_item = structured_item
    result, usage = engine._audit_and_repair_factual_visual_content(
        visual_id="shot-001",
        content=initial,
        grounding={
            "supported_claims": [
                {
                    "scene_id": "scene-001",
                    "exact_text": "Salinity lowers the freezing point of water.",
                    "evidence_ids": ["evidence-001"],
                }
            ],
            "nonfactual_framing": [],
            "allowed_evidence_records": [
                {
                    "evidence_id": "evidence-001",
                    "supported_statement": "Salinity lowers the freezing point of water.",
                }
            ],
        },
        staging_context={"modern_anchor": "salt beside a frozen step"},
        style_profile={"description": "paper cut"},
        character_identities=[],
        invariant=lambda value: None,
    )

    assert usage == []
    assert result == repaired
    assert [call["item_id"] for call in calls] == [
        "audit-content-shot-001",
        "repair-content-shot-001",
        "recheck-content-shot-001",
    ]
    assert calls[0]["input_data"]["review_requirement"] == {
        "claim_depiction_required": True,
        "active_supported_claim_count": 1,
        "numeric_claims_need_no_written_text": True,
    }
    assert calls[1]["output_model"] is VisualBriefContent
    assert calls[1]["input_data"]["visual_requirement"] == {
        "claim_depiction_required": True,
        "active_supported_claim_count": 1,
    }
    assert not {
        "scene_id",
        "shot_id",
        "narration_excerpt",
        "start_seconds",
        "end_seconds",
    } & set(VisualBriefContent.model_fields)


def test_factual_visual_gate_fails_after_one_unsuccessful_repair() -> None:
    engine = object.__new__(WorkflowEngine)
    engine.workflow_policy_version = 20
    content = _visual_content(
        action="Salt melts the ice into pools of liquid water.",
        state_after_scene=["liquid water surrounds the salt"],
    )
    calls: list[dict] = []

    def structured_item(**kwargs):
        calls.append(kwargs)
        if kwargs["task_id"] == "factual_review":
            value = FactualVisualDecision(
                verdict="unsupported",
                rationale="The candidate still depicts a result absent from the supplied evidence.",
            )
        else:
            value = content
        kwargs["invariant"](value)
        return value, []

    engine._structured_item = structured_item

    with pytest.raises(BackendError, match="after bounded repair"):
        engine._audit_and_repair_factual_visual_content(
            visual_id="shot-001",
            content=content,
            grounding={
                "supported_claims": [],
                "nonfactual_framing": [],
                "allowed_evidence_records": [],
            },
            staging_context={"modern_anchor": "salt beside a frozen step"},
            style_profile={"description": "paper cut"},
            character_identities=[],
            invariant=lambda value: None,
        )

    assert len(calls) == 3


def test_factual_visual_gate_allows_one_distinct_coverage_repair_after_safety_progress() -> None:
    engine = object.__new__(WorkflowEngine)
    engine.workflow_policy_version = 21
    unsupported = _visual_content(
        action="Brackish water begins freezing at a vaguely colder temperature.",
        state_after_scene=["ice crystals appear"],
    )
    safe_but_generic = _visual_content(
        action="Two differently colored water layers remain motionless side by side.",
        state_after_scene=["the layers remain unchanged"],
    )
    grounded = _visual_content(
        action=(
            "Two matching thermometer shapes compare seawater and brackish water, with distinct "
            "ice boundaries showing their different freezing thresholds."
        ),
        state_after_scene=["the two freezing thresholds are visibly contrasted"],
    )
    review_verdicts = iter(["unsupported", "underillustrated", "grounded"])
    repair_results = iter([safe_but_generic, grounded])
    calls: list[dict] = []

    def structured_item(**kwargs):
        calls.append(kwargs)
        if kwargs["task_id"] == "factual_review":
            value = FactualVisualDecision(
                verdict=next(review_verdicts),
                rationale="The candidate was checked against the exact active freezing-point claim.",
            )
        else:
            value = next(repair_results)
        kwargs["invariant"](value)
        return value, []

    engine._structured_item = structured_item
    claim = (
        "Seawater freezes at about minus 1.8 degrees Celsius, while brackish water "
        "freezes at about minus 0.3 degrees Celsius."
    )
    result, usage = engine._audit_and_repair_factual_visual_content(
        visual_id="shot-007",
        content=unsupported,
        grounding={
            "supported_claims": [
                {
                    "scene_id": "scene-003",
                    "exact_text": claim,
                    "evidence_ids": ["evidence-001"],
                }
            ],
            "nonfactual_framing": [],
            "allowed_evidence_records": [
                {
                    "evidence_id": "evidence-001",
                    "supported_statement": claim,
                }
            ],
        },
        staging_context={"modern_anchor": "salt on an icy step"},
        style_profile={"description": "paper cut"},
        character_identities=[],
        invariant=lambda value: None,
    )

    assert usage == []
    assert result == grounded
    assert [call["item_id"] for call in calls] == [
        "audit-content-shot-007",
        "repair-content-shot-007",
        "recheck-content-shot-007",
        "coverage-repair-content-shot-007",
        "final-recheck-content-shot-007",
    ]


def test_policy_twenty_six_no_claim_visual_uses_host_neutral_fallback() -> None:
    engine = object.__new__(WorkflowEngine)
    engine.workflow_policy_version = 26
    initial = _visual_content(
        action="Salt crystals hover above the frozen step.",
        state_after_scene=["the salt is about to land"],
    ).model_copy(
        update={
            "composition": "Salt crystals hover in mid-air above the icy surface.",
            "continuity_from_previous": ["the salt remains suspended"],
            "persistent_elements": ["hovering salt"],
        }
    )
    verdicts = iter(["unsupported", "grounded"])
    calls: list[dict] = []

    def structured_item(**kwargs):
        calls.append(kwargs)
        assert kwargs["task_id"] == "factual_review"
        value = FactualVisualDecision(
            verdict=next(verdicts),
            rationale="The complete assembled visual was checked for unsupported motion and process.",
        )
        kwargs["invariant"](value)
        return value, []

    engine._structured_item = structured_item
    result, usage = engine._audit_and_repair_factual_visual_content(
        visual_id="shot-002",
        content=initial,
        grounding={
            "supported_claims": [],
            "nonfactual_framing": [],
            "allowed_evidence_records": [],
        },
        staging_context={"modern_anchor": "salt beside a frozen step"},
        style_profile={"description": "paper cut"},
        character_identities=[],
        invariant=lambda value: None,
    )

    assert usage == []
    assert [call["item_id"] for call in calls] == [
        "audit-content-shot-002",
        "recheck-content-shot-002",
    ]
    assert result.subjects == initial.subjects
    assert "hover" not in result.action.casefold()
    assert "hover" not in result.composition.casefold()
    assert result.persistent_elements == []
    assert result.state_after_scene == [
        "The neutral staged arrangement remains unchanged."
    ]


@pytest.mark.parametrize("workflow_policy_version", [27, 29, 31])
@pytest.mark.parametrize("initial_verdict", ["underillustrated", "unsupported"])
def test_policy_twenty_seven_and_later_uses_host_threshold_compiler(
    workflow_policy_version: int,
    initial_verdict: str,
) -> None:
    engine = object.__new__(WorkflowEngine)
    engine.workflow_policy_version = workflow_policy_version
    initial = _visual_content(
        action="A solid ice layer sits below a different amount of liquid water.",
        state_after_scene=["the unequal liquid layers remain"],
    ).model_copy(
        update={
            "story_moment": "Pure water visibly freezes into a thick layer of ice.",
            "subjects": ["pure water layer", "solid ice layer"],
            "environment": "a layered frozen surface",
            "composition": "different liquid levels above unequal amounts of ice",
            "must_show": ["thick ice", "thin liquid layer"],
            "must_avoid": ["text", "salt grains"],
            "character_ids": ["character-001"],
            "continuity_from_previous": ["preserve the old ice layer"],
            "identity_requirements": ["preserve the round glasses"],
            "persistent_elements": ["old ice layer"],
        }
    )
    verdicts = iter([initial_verdict, "grounded"])
    calls: list[dict] = []

    def structured_item(**kwargs):
        calls.append(kwargs)
        assert kwargs["task_id"] == "factual_review"
        value = FactualVisualDecision(
            verdict=next(verdicts),
            rationale=(
                "The complete comparison was checked against the supported salinity threshold."
            ),
        )
        kwargs["invariant"](value)
        return value, []

    claim = "Veden jäätymispiste laskee, kun veden suolaisuus kasvaa."
    engine._structured_item = structured_item
    result, usage = engine._audit_and_repair_factual_visual_content(
        visual_id="shot-004",
        content=initial,
        grounding={
            "supported_claims": [
                {
                    "scene_id": "scene-003",
                    "exact_text": claim,
                    "evidence_ids": ["evidence-001"],
                }
            ],
            "nonfactual_framing": [],
            "allowed_evidence_records": [
                {
                    "evidence_id": "evidence-001",
                    "supported_statement": claim,
                }
            ],
        },
        staging_context={
            "narration_excerpt": claim,
            "modern_anchor": "salt beside a frozen step",
        },
        style_profile={"description": "paper cut"},
        character_identities=[{"character_id": "character-001"}],
        invariant=lambda value: None,
    )

    assert usage == []
    assert [call["item_id"] for call in calls] == [
        "audit-content-shot-004",
        "recheck-content-shot-004",
    ]
    assert all(call["output_model"] is FactualVisualDecision for call in calls)

    positive_fields = " ".join(
        [
            result.story_moment,
            *result.subjects,
            result.action,
            result.emotion,
            result.environment,
            result.composition,
            *result.must_show,
            *result.continuity_from_previous,
            *result.state_after_scene,
            *result.persistent_elements,
        ]
    ).casefold()
    assert "lower-salinity water" in positive_fields
    assert "higher-salinity water" in positive_fields
    assert "equal liquid levels" in positive_fields
    assert "marker beside higher-salinity water" in positive_fields
    assert "denser uniform salt-particle pattern" in positive_fields
    assert "solid ice" not in positive_fields
    assert "different amount" not in positive_fields
    assert not any(character.isnumeric() for character in positive_fields)
    assert "°" not in positive_fields
    assert result.must_avoid != initial.must_avoid
    assert "salt grains" not in result.must_avoid
    assert any("phase change" in item for item in result.must_avoid)
    assert result.character_ids == initial.character_ids
    assert result.identity_requirements == initial.identity_requirements


@pytest.mark.parametrize(
    "claim",
    [
        "Meriveden jäätymispiste on noin miinus kaksi astetta.",
        "Basalt has a lower freezing point than granite.",
        "Salinity is 5 percent and the freezing point is minus 2 degrees.",
        "Choose salt rather than basalt because its freezing point is lower.",
        "Salt does not lower the freezing point of water.",
        "Lower salinity decreases the freezing point.",
        "Jäätymispiste ei alene, vaikka suolaisuus kasvaa.",
        "Matala suolaisuus laskee jäätymispistettä.",
        "Salt lowers viscosity, while the freezing point rises.",
        "Higher salinity causes corrosion. The freezing point lowers under pressure.",
        "Suola laskee lämpötilaa, mutta jäätymispiste nousee.",
    ],
)
def test_factual_threshold_compiler_rejects_non_relational_claims(claim: str) -> None:
    grounding = {
        "supported_claims": [
            {
                "scene_id": "scene-001",
                "exact_text": claim,
                "evidence_ids": ["evidence-001"],
            }
        ]
    }

    assert not WorkflowEngine._has_supported_salinity_freezing_threshold_relation(
        grounding
    )


@pytest.mark.parametrize(
    "claim",
    [
        "Veden jäätymispiste laskee, kun veden suolaisuus kasvaa.",
        "Kun suolaisuus kasvaa, veden jäätymispiste laskee.",
        "As salinity increases, the freezing point drops.",
        "The freezing point is lower at higher salinity.",
        "Salt lowers the freezing point of water.",
        "Increasing salinity lowers the freezing point.",
        "Suolaisuuden kasvu laskee jäätymispistettä.",
    ],
)
def test_factual_threshold_compiler_accepts_affirmative_directional_claims(
    claim: str,
) -> None:
    grounding = {
        "supported_claims": [
            {
                "scene_id": "scene-001",
                "exact_text": claim,
                "evidence_ids": ["evidence-001"],
            }
        ]
    }

    assert WorkflowEngine._has_supported_salinity_freezing_threshold_relation(
        grounding
    )


def test_policy_twenty_nine_accepts_safe_underillustrated_visual_without_editing() -> None:
    engine = object.__new__(WorkflowEngine)
    engine.workflow_policy_version = 29
    content = _visual_content(
        action="Salt grains and ice remain separate and motionless.",
        state_after_scene=["the static arrangement remains unchanged"],
    )
    calls: list[dict] = []

    def structured_item(**kwargs):
        calls.append(kwargs)
        assert kwargs["task_id"] == "factual_review"
        value = FactualVisualDecision(
            verdict="underillustrated",
            rationale="The static candidate is safe but does not encode the active relationship.",
        )
        kwargs["invariant"](value)
        return value, []

    engine._structured_item = structured_item
    claim = "Salinity affects water density and freezing point."
    result, usage = engine._audit_and_repair_factual_visual_content(
        visual_id="shot-007",
        content=content,
        grounding={
            "supported_claims": [
                {
                    "scene_id": "scene-004",
                    "exact_text": claim,
                    "evidence_ids": ["evidence-001"],
                }
            ],
            "nonfactual_framing": [],
            "allowed_evidence_records": [
                {
                    "evidence_id": "evidence-001",
                    "supported_statement": claim,
                }
            ],
        },
        staging_context={"modern_anchor": "salt beside a frozen step"},
        style_profile={"description": "paper cut"},
        character_identities=[],
        invariant=lambda value: None,
    )

    assert result is content
    assert usage == []
    assert [call["item_id"] for call in calls] == ["audit-content-shot-007"]
    assert engine._visual_plan_warnings == [
        "shot-007 uses a factually safe but underillustrated visual because bounded "
        "generation and review did not produce a more specific grounded depiction."
    ]


def test_policy_twenty_nine_accepts_safe_bounded_repair_without_coverage_loop() -> None:
    engine = object.__new__(WorkflowEngine)
    engine.workflow_policy_version = 29
    content = _visual_content(
        action="Salt melts the ice into a liquid pool.",
        state_after_scene=["a liquid pool surrounds the salt"],
    )
    decisions = iter(["unsupported", "underillustrated"])
    calls: list[dict] = []
    safe_depiction = "Salt grains and ice remain separate and motionless in a static arrangement."

    def structured_item(**kwargs):
        calls.append(kwargs)
        if kwargs["task_id"] == "factual_review":
            value = FactualVisualDecision(
                verdict=next(decisions),
                rationale="The complete candidate was checked against the vague supported claim.",
            )
        else:
            assert kwargs["output_model"] is FactualVisualDepiction
            value = FactualVisualDepiction(depiction=safe_depiction)
        kwargs["invariant"](value)
        return value, []

    engine._structured_item = structured_item
    claim = "Salinity affects water density and freezing point."
    result, usage = engine._audit_and_repair_factual_visual_content(
        visual_id="shot-007",
        content=content,
        grounding={
            "supported_claims": [
                {
                    "scene_id": "scene-004",
                    "exact_text": claim,
                    "evidence_ids": ["evidence-001"],
                }
            ],
            "nonfactual_framing": [],
            "allowed_evidence_records": [
                {
                    "evidence_id": "evidence-001",
                    "supported_statement": claim,
                }
            ],
        },
        staging_context={"modern_anchor": "salt beside a frozen step"},
        style_profile={"description": "paper cut"},
        character_identities=[],
        invariant=lambda value: None,
    )

    assert usage == []
    assert result.story_moment == safe_depiction
    assert [call["item_id"] for call in calls] == [
        "audit-content-shot-007",
        "repair-content-shot-007",
        "recheck-content-shot-007",
    ]
    assert sum(call["task_id"] == "visual_plan" for call in calls) == 1
    assert engine._visual_plan_warnings == [
        "shot-007 uses a factually safe but underillustrated visual because bounded "
        "generation and review did not produce a more specific grounded depiction."
    ]


def test_policy_twenty_nine_uses_host_fallback_after_unsupported_repair() -> None:
    engine = object.__new__(WorkflowEngine)
    engine.workflow_policy_version = 29
    content = _visual_content(
        action="Salt melts the ice into a liquid pool.",
        state_after_scene=["a liquid pool surrounds the salt"],
    ).model_copy(
        update={
            "subjects": ["salt crystals", "melting ice"],
            "composition": "Salt visibly melts a thick layer of ice.",
            "character_ids": ["character-001"],
            "identity_requirements": ["preserve the round glasses"],
            "persistent_elements": ["melting ice"],
        }
    )
    decisions = iter(["unsupported", "unsupported", "underillustrated"])
    calls: list[dict] = []

    def structured_item(**kwargs):
        calls.append(kwargs)
        if kwargs["task_id"] == "factual_review":
            value = FactualVisualDecision(
                verdict=next(decisions),
                rationale="The complete candidate was checked for unsupported factual semantics.",
            )
        else:
            assert kwargs["output_model"] is FactualVisualDepiction
            value = FactualVisualDepiction(
                depiction="Salt rapidly transforms the ice into liquid water."
            )
        kwargs["invariant"](value)
        return value, []

    engine._structured_item = structured_item
    claim = "Salinity affects water density and freezing point."
    result, usage = engine._audit_and_repair_factual_visual_content(
        visual_id="shot-008",
        content=content,
        grounding={
            "supported_claims": [
                {
                    "scene_id": "scene-004",
                    "exact_text": claim,
                    "evidence_ids": ["evidence-001"],
                }
            ],
            "nonfactual_framing": [],
            "allowed_evidence_records": [
                {
                    "evidence_id": "evidence-001",
                    "supported_statement": claim,
                }
            ],
        },
        staging_context={"modern_anchor": "salt beside a frozen step"},
        style_profile={"description": "paper cut"},
        character_identities=[
            {"character_id": "character-001", "identity": "round glasses"}
        ],
        invariant=lambda value: None,
    )

    assert usage == []
    assert [call["item_id"] for call in calls] == [
        "audit-content-shot-008",
        "repair-content-shot-008",
        "recheck-content-shot-008",
        "fallback-recheck-content-shot-008",
    ]
    assert sum(call["task_id"] == "visual_plan" for call in calls) == 1
    fallback_review = calls[-1]
    assert fallback_review["input_data"]["review_requirement"] == {
        "claim_depiction_required": True,
        "active_supported_claim_count": 1,
        "numeric_claims_need_no_written_text": True,
    }
    assert result.subjects == [
        "the approved recurring characters against an unmarked neutral backdrop"
    ]
    assert result.character_ids == content.character_ids
    assert result.identity_requirements == content.identity_requirements
    assert result.persistent_elements == []
    positive_fields = " ".join(
        [
            result.story_moment,
            *result.subjects,
            result.action,
            result.environment,
            result.composition,
            *result.must_show,
            *result.continuity_from_previous,
            *result.state_after_scene,
            *result.persistent_elements,
        ]
    ).casefold()
    assert "salt" not in positive_fields
    assert "ice" not in positive_fields
    assert "liquid" not in positive_fields
    assert fallback_review["input_data"]["candidate"][
        "visible_character_identities"
    ] == [{"character_id": "character-001", "identity": "round glasses"}]
    assert engine._visual_plan_warnings == [
        "shot-008 uses a prop-free host safety fallback because bounded generation and one "
        "safety repair remained unsupported."
    ]


def test_policy_twenty_nine_rejects_unsupported_host_fallback() -> None:
    engine = object.__new__(WorkflowEngine)
    engine.workflow_policy_version = 29
    content = _visual_content(
        action="Salt melts the ice into a liquid pool.",
        state_after_scene=["a liquid pool surrounds the salt"],
    )
    reviews = iter(
        [
            FactualVisualDecision(
                verdict="unsupported",
                rationale="The initial candidate invents an unsupported physical outcome.",
            ),
            FactualVisualDecision(
                verdict="unsupported",
                rationale="The bounded repair still invents an unsupported physical outcome.",
            ),
            FactualVisualDecision(
                verdict="unsupported",
                rationale="The host fallback unexpectedly retains an unsupported factual assertion.",
            ),
        ]
    )
    engine._review_factual_visual_candidate = lambda **kwargs: (next(reviews), [])

    def structured_item(**kwargs):
        value = FactualVisualDepiction(
            depiction="Salt rapidly transforms the ice into liquid water."
        )
        kwargs["invariant"](value)
        return value, []

    engine._structured_item = structured_item
    claim = "Salinity affects water density and freezing point."

    with pytest.raises(BackendError, match="after host safety fallback"):
        engine._audit_and_repair_factual_visual_content(
            visual_id="shot-009",
            content=content,
            grounding={
                "supported_claims": [
                    {
                        "scene_id": "scene-004",
                        "exact_text": claim,
                        "evidence_ids": ["evidence-001"],
                    }
                ],
                "nonfactual_framing": [],
                "allowed_evidence_records": [
                    {
                        "evidence_id": "evidence-001",
                        "supported_statement": claim,
                    }
                ],
            },
            staging_context={"modern_anchor": "salt beside a frozen step"},
            style_profile={"description": "paper cut"},
            character_identities=[],
            invariant=lambda value: None,
        )

    assert not hasattr(engine, "_visual_plan_warnings")


def test_policy_twenty_three_visual_repair_changes_only_the_depiction_fields() -> None:
    engine = object.__new__(WorkflowEngine)
    engine.workflow_policy_version = 23
    initial = _visual_content(
        action="Salt melts the ice into liquid water.",
        state_after_scene=["the original continuity state remains"],
    ).model_copy(
        update={
            "character_ids": ["character-001"],
            "continuity_from_previous": ["preserve the blue bowl"],
            "identity_requirements": ["preserve the round glasses"],
            "persistent_elements": ["blue bowl"],
        }
    )
    depiction = (
        "A static side-by-side comparison shows saline water and fresh ice as distinct states."
    )
    verdicts = iter(["unsupported", "grounded"])
    calls: list[dict] = []

    def structured_item(**kwargs):
        calls.append(kwargs)
        if kwargs["task_id"] == "factual_review":
            value = FactualVisualDecision(
                verdict=next(verdicts),
                rationale="The candidate was checked against the active supported comparison claim.",
            )
        else:
            value = FactualVisualDepiction(depiction=depiction)
        kwargs["invariant"](value)
        return value, []

    engine._structured_item = structured_item
    result, usage = engine._audit_and_repair_factual_visual_content(
        visual_id="shot-001",
        content=initial,
        grounding={
            "supported_claims": [
                {
                    "scene_id": "scene-001",
                    "exact_text": "Salinity lowers the freezing point of water.",
                    "evidence_ids": ["evidence-001"],
                }
            ],
            "nonfactual_framing": [],
            "allowed_evidence_records": [
                {
                    "evidence_id": "evidence-001",
                    "supported_statement": "Salinity lowers the freezing point of water.",
                }
            ],
        },
        staging_context={"modern_anchor": "salt beside a frozen step"},
        style_profile={"description": "paper cut"},
        character_identities=[{"character_id": "character-001"}],
        invariant=lambda value: None,
    )

    assert usage == []
    assert calls[1]["output_model"] is FactualVisualDepiction
    assert result.story_moment == depiction
    assert result.action == depiction
    assert result.must_show == [depiction]
    for field_name in (
        "subjects",
        "emotion",
        "environment",
        "composition",
        "character_ids",
        "continuity_from_previous",
        "state_after_scene",
        "identity_requirements",
        "persistent_elements",
    ):
        assert getattr(result, field_name) == getattr(initial, field_name)


def test_factual_depiction_repair_merges_host_safety_constraints() -> None:
    initial = _visual_content(
        action="A thermometer shows a labeled value while ice changes phase.",
        state_after_scene=["the original state remains"],
    ).model_copy(
        update={
            "must_avoid": [
                "numbers and units",
                "phase change",
                "a fixture-specific hazard",
            ]
        }
    )

    repaired = WorkflowEngine._apply_factual_visual_depiction(
        initial,
        FactualVisualDepiction(
            depiction="A blank thermometer shape remains beside an unchanged sample."
        ),
    )

    combined = " ".join(repaired.must_avoid).casefold()
    assert "numbers" in combined
    assert "units" in combined
    assert "readable scales" in combined
    assert "phase change" in combined
    assert "a fixture-specific hazard" in repaired.must_avoid


def test_policy_thirty_one_compiles_scalar_temperature_without_written_value() -> None:
    engine = object.__new__(WorkflowEngine)
    engine.workflow_policy_version = 31
    initial = _visual_content(
        action="A labeled thermometer points exactly to −1,9 °C.",
        state_after_scene=["the labeled value remains visible"],
    ).model_copy(
        update={
            "character_ids": ["character-001"],
            "identity_requirements": ["preserve the round glasses"],
        }
    )
    engine._structured_item = lambda **kwargs: pytest.fail(
        "the scalar host compiler must bypass semantic generation and review"
    )
    claim = "Meriveden jäätymispiste on noin −1,9 °C."

    result, usage = engine._audit_and_repair_factual_visual_content(
        visual_id="shot-007",
        content=initial,
        grounding={
            "supported_claims": [
                {
                    "scene_id": "scene-004",
                    "exact_text": claim,
                    "evidence_ids": ["evidence-001"],
                }
            ],
            "nonfactual_framing": [],
            "allowed_evidence_records": [
                {
                    "evidence_id": "evidence-001",
                    "supported_statement": claim,
                }
            ],
        },
        staging_context={"modern_anchor": "salt beside a frozen step"},
        style_profile={"description": "paper cut"},
        character_identities=[{"character_id": "character-001"}],
        invariant=lambda value: None,
    )

    positive = WorkflowEngine._factual_visual_positive_text(result).casefold()
    assert usage == []
    assert "1,9" not in positive
    assert "1.9" not in positive
    assert "°" not in positive
    assert result.character_ids == initial.character_ids
    assert result.identity_requirements == initial.identity_requirements
    combined_avoid = " ".join(result.must_avoid).casefold()
    assert "numbers" in combined_avoid
    assert "units" in combined_avoid
    assert "readable scales" in combined_avoid
    assert engine._visual_plan_warnings == [
        "shot-007 uses a factually safe but underillustrated visual because bounded "
        "generation and review did not produce a more specific grounded depiction."
    ]


def test_compact_factual_depiction_compiler_owns_constraints_and_characters() -> None:
    content = WorkflowEngine._compile_factual_visual_content_from_depiction(
        FactualVisualDepiction(
            depiction="A sealed sample sits beside a blank thermometer shape."
        ),
        style_profile={
            "background": "pale paper",
            "must_avoid": [f"style hazard {index}" for index in range(40)],
        },
        character_identities=[
            {
                "character_id": "character-001",
                "name": "Aino",
                "signature_traits": ["round glasses"],
                "body_form": "small upright figure",
                "identity_constraints": ["keep the blue scarf"],
            }
        ],
        has_previous=False,
    )

    assert content.character_ids == ["character-001"]
    assert content.identity_requirements
    assert content.continuity_from_previous == []
    assert content.state_after_scene
    assert len(content.must_avoid) == 30
    assert "numbers" in content.must_avoid[0]
    assert "unsupported factual mechanisms" in content.must_avoid[1]


def test_policy_twenty_five_visual_repair_uses_compact_input_and_safe_refinement() -> None:
    engine = object.__new__(WorkflowEngine)
    engine.workflow_policy_version = 25
    initial = _visual_content(
        action="Two neutral water samples remain motionless side by side.",
        state_after_scene=["the original neutral samples remain unchanged"],
    ).model_copy(
        update={
            "character_ids": ["character-001"],
            "continuity_from_previous": ["preserve the blue bowl"],
            "identity_requirements": ["preserve the round glasses"],
            "persistent_elements": ["blue bowl"],
        }
    )
    unsupported_proxy = (
        "Dense salt with less ice is compared against sparse salt with more ice."
    )
    safe_threshold = (
        "Two unchanged water samples sit beside matching unlabeled thermometer shapes whose "
        "indicator lines mark two different freezing thresholds."
    )
    verdicts = iter(["underillustrated", "unsupported", "grounded"])
    depictions = iter([unsupported_proxy, safe_threshold])
    calls: list[dict] = []

    def structured_item(**kwargs):
        calls.append(kwargs)
        if kwargs["task_id"] == "factual_review":
            value = FactualVisualDecision(
                verdict=next(verdicts),
                rationale="The candidate was checked against the exact supported threshold comparison.",
            )
        else:
            value = FactualVisualDepiction(depiction=next(depictions))
        kwargs["invariant"](value)
        return value, []

    claim = (
        "Seawater freezes at minus 1.8 degrees Celsius and brackish water freezes at "
        "minus 0.3 degrees Celsius."
    )
    engine._structured_item = structured_item
    result, usage = engine._audit_and_repair_factual_visual_content(
        visual_id="shot-006",
        content=initial,
        grounding={
            "supported_claims": [
                {
                    "scene_id": "scene-003",
                    "exact_text": claim,
                    "evidence_ids": ["evidence-003"],
                }
            ],
            "nonfactual_framing": [],
            "allowed_evidence_records": [
                {
                    "evidence_id": "evidence-003",
                    "supported_statement": claim,
                }
            ],
        },
        staging_context={
            "narration_excerpt": "brackish water freezes at minus 0.3 degrees Celsius.",
            "modern_anchor": "salt beside a frozen step",
        },
        style_profile={"description": "literal cause-and-effect paper cut"},
        character_identities=[{"character_id": "character-001"}],
        invariant=lambda value: None,
    )

    assert usage == []
    assert [call["item_id"] for call in calls] == [
        "audit-content-shot-006",
        "repair-content-shot-006",
        "recheck-content-shot-006",
        "safety-refinement-content-shot-006",
        "final-recheck-content-shot-006",
    ]
    first_repair = calls[1]
    assert first_repair["output_model"] is FactualVisualDepiction
    assert first_repair["input_data"]["repair_mode"] == "underillustrated"
    assert not {
        "previous_visual_content",
        "style_profile",
        "character_identities",
    } & set(first_repair["input_data"])
    assert "correlated proxy" in first_repair["instruction_suffix"]
    assert "thermometer or threshold markers" in first_repair["instruction_suffix"]
    refinement = calls[3]
    assert refinement["input_data"]["repair_mode"] == "claim_coverage_without_proxy"
    assert refinement["input_data"]["previous_depiction"]["action"] == initial.action
    assert result.story_moment == safe_threshold
    assert result.action == safe_threshold
    assert result.must_show == [safe_threshold]
    for field_name in (
        "subjects",
        "emotion",
        "environment",
        "composition",
        "character_ids",
        "continuity_from_previous",
        "state_after_scene",
        "identity_requirements",
        "persistent_elements",
    ):
        assert getattr(result, field_name) == getattr(initial, field_name)


def test_factual_image_prompt_is_assembled_from_approved_fields() -> None:
    engine = object.__new__(WorkflowEngine)
    brief = VisualBrief(
        scene_id="scene-001",
        story_moment="A neutral still life of salt beside a frozen step.",
        subjects=["coarse salt", "frozen step"],
        action="The subjects remain motionless with no visible change.",
        emotion="focused curiosity",
        environment="a sparse pale paper background",
        composition="a centered macro still life with generous empty space",
        must_show=["separate salt grains", "an intact frozen surface"],
        must_avoid=["melting", "liquid water", "text"],
    )
    style = StyleProfile(
        style_id="paper_cut",
        description="Layered paper-cut illustration with visible paper fibers.",
        palette=["icy blue", "white", "deep navy"],
        line_style="clean cut-paper edges",
        background="soft pale-blue paper",
        must_avoid=["photorealism", "gradients"],
    )

    compiled = engine._compile_factual_image_prompt_content(
        visual_brief=brief,
        style_profile=style,
        characters=[],
    )

    assert brief.action in compiled.prompt
    assert style.description in compiled.prompt
    assert "liquid water" not in compiled.prompt
    assert "liquid water" in compiled.negative_prompt


@pytest.mark.parametrize(
    "measurement",
    ["−1,9 °C", "1.9 degrees Celsius", "5 percent", "35 PSU"],
)
def test_factual_image_prompt_rejects_literal_measurements(
    measurement: str,
) -> None:
    engine = object.__new__(WorkflowEngine)
    brief = VisualBrief(
        scene_id="scene-001",
        story_moment=f"A thermometer visibly reads {measurement}.",
        subjects=["thermometer"],
        action=f"The marker points to {measurement}.",
        emotion="neutral",
        environment="a blank paper background",
        composition="one centered object",
        must_show=[f"the exact {measurement} readout"],
        must_avoid=["written text", "numbers", "units"],
    )
    style = StyleProfile(
        style_id="paper_cut",
        description="Layered paper-cut illustration.",
        palette=["icy blue", "white"],
        line_style="clean cut-paper edges",
        background="soft pale-blue paper",
        must_avoid=["photorealism"],
    )

    with pytest.raises(BackendError, match="literal measurement value"):
        engine._compile_factual_image_prompt_content(
            visual_brief=brief,
            style_profile=style,
            characters=[],
        )


def test_literal_measurement_lint_ignores_aspect_ratio_and_style_dimension() -> None:
    assert not WorkflowEngine._contains_literal_measurement_value(
        "A clear 16:9 composition in a flat 2D paper-cut style."
    )
