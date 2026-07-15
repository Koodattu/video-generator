from __future__ import annotations

import math
import re
import struct
import wave
from pathlib import Path
from typing import Any

from ..contracts import (
    AlignmentRequest,
    AlignmentResult,
    ImageAsset,
    ImageRequest,
    ImageResult,
    MediaReference,
    MusicAsset,
    MusicRequest,
    MusicResult,
    OutputLanguage,
    ProbeItem,
    ProbeReport,
    ResearchSource,
    SearchRequest,
    SearchResult,
    SourceDocument,
    SourceFetchRequest,
    SpeechAsset,
    SpeechRequest,
    SpeechResult,
    StructuredTextRequest,
    StructuredTextResult,
    UsageRecord,
    WordTiming,
)
from ..profiles import BACKEND_DESCRIPTORS
from ..util import relative_path, sha256_file
from .base import Backend


def _word_timings(text: str, duration: float) -> list[WordTiming]:
    words = re.findall(r"\S+", text, flags=re.UNICODE)
    if not words:
        return []
    slot = duration / len(words)
    return [
        WordTiming(
            text=word,
            start_seconds=index * slot,
            end_seconds=(index + 0.88) * slot,
            confidence=1.0,
        )
        for index, word in enumerate(words)
    ]


def _fixture_sentence(language: OutputLanguage, index: int) -> str:
    if language is OutputLanguage.FINNISH:
        options = (
            "Lumi narskui saappaiden alla, kun pieni lyhty välähti polun reunassa.",
            "Aino pysähtyi, kuunteli hiljaisuutta ja huomasi valon vastaavan liikkeeseen.",
            "Hän suojasi liekkiä punaisella huivillaan ja jatkoi varovasti eteenpäin.",
            "Kauempana metsä näytti ensin tyhjältä, mutta siniset jäljet muodostivat uuden reitin.",
        )
    else:
        options = (
            "Snow creaked under each boot when a small lantern blinked beside the path.",
            "Aino stopped, listened to the quiet, and saw the light answer her movement.",
            "She sheltered the flame with her red scarf and moved carefully forward.",
            "The distant woods looked empty at first, until blue marks formed a new route.",
        )
    return options[index % len(options)]


def _is_factual_content(data: dict[str, Any]) -> bool:
    return data.get("content_mode") == "factual"


def _factual_evidence(data: dict[str, Any]) -> list[dict[str, Any]]:
    available = data.get("available_factual_evidence")
    if isinstance(available, list):
        return [item for item in available if isinstance(item, dict)]
    research_pack = data.get("research_pack", {})
    if isinstance(research_pack, dict):
        evidence = research_pack.get("evidence", [])
        if isinstance(evidence, list):
            return [item for item in evidence if isinstance(item, dict)]
    return []


def _factual_narrative_text(
    data: dict[str, Any],
    *,
    desired_words: int,
    language: OutputLanguage,
    fallback: str = "",
) -> str:
    evidence = _factual_evidence(data)
    position = max(0, int(data.get("scene_position", 1)) - 1)
    statement = (
        str(evidence[position % len(evidence)].get("supported_statement", "")).strip()
        if evidence
        else fallback.strip()
    )
    if not statement:
        statement = (
            "Tarkastele vain annettua rajattua näyttöä."
            if language is OutputLanguage.FINNISH
            else "Consider only the supplied bounded evidence."
        )
    words = statement.split()
    directives = (
        ["Katso.", "Keskity.", "Jatka.", "Palaa.", "Harkitse."]
        if language is OutputLanguage.FINNISH
        else ["Look.", "Focus.", "Continue.", "Return.", "Reconsider."]
    )
    while len(words) < desired_words:
        words.append(directives[(len(words) - len(statement.split())) % len(directives)])
    return " ".join(words)


def _fake_structured(request: StructuredTextRequest) -> dict[str, Any]:
    data = request.input_data
    task = request.task_id
    if task == "research":
        sources = data.get("sources", [])
        if data.get("content_mode") == "factual":
            evidence = []
            for source in sources:
                excerpt = str(
                    source.get("excerpt")
                    or "The bounded fixture source supports this statement."
                )
                supported_statement = excerpt.split(";", 1)[0].strip()
                if supported_statement and supported_statement[-1] not in ".!?":
                    supported_statement += "."
                evidence.append(
                    {
                        "supported_statement": supported_statement,
                        "source_ids": [str(source.get("source_id"))],
                        "confidence": "high",
                        "time_sensitive": False,
                        "limitations": ["Deterministic test fixture evidence."],
                    }
                )
            return {
                "evidence": evidence
            }
        payload = {
            "findings": [
                {
                    "summary": str(source.get("excerpt") or "A concrete setting detail for original fiction."),
                    "source_ids": [str(source.get("source_id"))],
                }
                for source in sources
            ],
            "motifs": ["a responsive light", "tracks that change direction"],
            "setting_details": ["dry snow squeaks sharply in deep cold"],
            "vocabulary": ["blue hour", "frost smoke"],
            "cultural_cautions": ["avoid treating living traditions as generic magic"],
            "cliches_to_avoid": ["a prophecy that explains the whole plot"],
        }
        return payload
    if task == "ideate":
        count = int(data.get("candidate_count", 5))
        explainer = data.get("content_format") in {"explainer", "mythbuster"}
        factual_narrative = _is_factual_content(data)
        evidence = _factual_evidence(data)
        bounded_statement = (
            str(evidence[0].get("supported_statement", "The bounded evidence is examined."))
            if evidence
            else "The bounded evidence is examined."
        )
        bounded_evidence_id = (
            str(evidence[0].get("evidence_id", "")) if evidence else ""
        )
        candidates = []
        for index in range(1, count + 1):
            if explainer:
                candidates.append(
                    {
                        "candidate_id": f"candidate-{index:03d}",
                        "title": f"Why Winter Paths Sound Different {index}",
                        "modern_anchor": "the sound of a boot on fresh snow",
                        "central_question": "Why does very cold snow squeak?",
                        "misconception": "The sound comes only from heavy boots.",
                        "thesis": "Temperature changes how snow crystals deform and rub together.",
                        "evidence_ids": ["evidence-001"] if data.get("content_mode") == "factual" else [],
                        "evidence_ladder": ["notice the sound", "compare temperatures", "explain crystal friction"],
                        "human_angle": "People have long read snow conditions through sound and feel.",
                        "landing_direction": "hear the next winter step as a tiny material experiment",
                        "visual_opportunities": ["boot and snow crystals", "thermometer", "simple force arrows"],
                        "accuracy_risks": ["overstating a single cause"],
                        "duration_fit": "fits a compact evidence-led explanation",
                    }
                )
                continue
            if factual_narrative:
                candidates.append(
                    {
                        "candidate_id": f"candidate-{index:03d}",
                        "title": f"A Bounded Evidence Path {index}",
                        "premise": bounded_statement,
                        "protagonist_desire": "The viewer wants to understand the bounded evidence.",
                        "obstacle": "The evidence must remain within its stated scope and limits.",
                        "turn": bounded_statement,
                        "ending_direction": "Return to the same bounded statement with its limits intact.",
                        "emotional_promise": "Curiosity resolves into evidence-based understanding.",
                        "research_inspiration_ids": (
                            [bounded_evidence_id] if bounded_evidence_id else []
                        ),
                        "visual_opportunities": ["a literal diagram of the supported statement"],
                        "originality_risks": ["inventing events beyond the supplied evidence"],
                        "duration_fit": "fits a compact evidence-led narrative",
                    }
                )
                continue
            candidates.append(
                {
                    "candidate_id": f"candidate-{index:03d}",
                    "title": f"The Lantern Path {index}",
                    "premise": f"A cautious traveler finds a different impossible signal in the winter dark ({index}).",
                    "protagonist_desire": "reach shelter without abandoning a stranger",
                    "obstacle": "the safe path and the helpful signal point in opposite directions",
                    "turn": "the signal is not asking for rescue but offering directions",
                    "ending_direction": "a small reciprocal act makes both travelers safer",
                    "emotional_promise": "warm curiosity after mild suspense",
                    "research_inspiration_ids": [],
                    "visual_opportunities": ["tiny colored light on white snow", "two sets of tracks"],
                    "originality_risks": ["sentimental rescue ending"],
                    "duration_fit": "fits a compact narrated arc",
                }
            )
        return {"schema_version": 1, "candidates": candidates}
    if task == "select":
        candidates = data.get("candidate_set", {}).get("candidates", [])
        explainer = bool(candidates and "modern_anchor" in candidates[0])
        scores = []
        for index, candidate in enumerate(candidates):
            score = 5 if index == 0 else 4
            score_item = {
                    "candidate_id": candidate["candidate_id"],
                    "duration_fit": score,
                    "visual_strength": 5,
                    "spoken_suitability": score,
                    "audience_fit": 5,
                    "rationale": "The progression is clear and every beat has a simple visual.",
            }
            if explainer:
                score_item.update(
                    hook_strength=score,
                    evidence_strength=score,
                    progression=score,
                )
            else:
                score_item.update(
                    originality=score,
                    story_potential=score,
                    research_responsibility=5,
                )
            scores.append(score_item)
        chosen = candidates[0]["candidate_id"] if candidates else "candidate-001"
        return {
            "schema_version": 1,
            "scores": scores,
            "chosen_candidate_id": chosen,
            "rationale": "The selected concept has the clearest action, turn, and quiet payoff.",
        }
    if task == "outline":
        duration = float(data.get("duration_seconds", 60))
        target = float(data.get("visual_target_seconds", 15))
        scene_count = max(2, round(duration / target))
        per_scene = duration / scene_count
        scenes = []
        explainer = data.get("content_format") in {"explainer", "mythbuster"}
        factual_narrative = _is_factual_content(data)
        evidence = _factual_evidence(data)
        bounded_statement = (
            str(evidence[0].get("supported_statement", "The bounded evidence is examined."))
            if evidence
            else "The bounded evidence is examined."
        )
        arc_roles = [
            "modern_hook",
            "question",
            "misconception",
            "correction",
            "evidence",
            "human_tangent",
            "synthesis",
            "landing",
        ]
        for index in range(1, scene_count + 1):
            if explainer:
                role = arc_roles[min(len(arc_roles) - 1, round((index - 1) * (len(arc_roles) - 1) / max(1, scene_count - 1)))]
                scenes.append(
                    {
                        "scene_id": f"scene-{index:03d}",
                        "arc_role": role,
                        "purpose": "advance the explanation with one concrete step",
                        "key_point": "Cold changes how snow crystals respond under a boot.",
                        "evidence_ids": ["evidence-001"] if data.get("content_mode") == "factual" else [],
                        "visual_opportunity": "a boot, simple snow crystals, and readable arrows",
                        "provisional_seconds": per_scene,
                        "continuity_obligations": ["same blue boot and white snow field"],
                    }
                )
                continue
            if factual_narrative:
                scenes.append(
                    {
                        "scene_id": f"scene-{index:03d}",
                        "narrative_purpose": (
                            "introduce the bounded evidence"
                            if index == 1
                            else "revisit the evidence without expanding its scope"
                        ),
                        "change": bounded_statement,
                        "emotional_beat": "curiosity becomes careful understanding",
                        "visual_opportunity": "a literal diagram of the supported statement",
                        "provisional_seconds": per_scene,
                        "continuity_obligations": ["keep the same evidence subject visible"],
                    }
                )
                continue
            scenes.append(
                {
                    "scene_id": f"scene-{index:03d}",
                    "narrative_purpose": "set up" if index == 1 else "develop and resolve",
                    "change": "the protagonist learns one actionable detail",
                    "emotional_beat": "curiosity becomes cautious trust",
                    "visual_opportunity": "one figure, one colored prop, and clear tracks in snow",
                    "provisional_seconds": per_scene,
                    "continuity_obligations": ["red triangular scarf", "blue tin lantern"],
                }
            )
        if explainer:
            return {
                "schema_version": 1,
                "title": "Why Cold Snow Squeaks",
                "thesis": "Snow sound changes because cold crystals deform and rub differently.",
                "modern_anchor": "the next step you take on fresh winter snow",
                "misconception": "Only boot weight controls the sound.",
                "landing_callback": "listen to that next step as a material experiment",
                "scenes": scenes,
            }
        if factual_narrative:
            return {
                "schema_version": 1,
                "title": "A Bounded Evidence Path",
                "concept_summary": bounded_statement,
                "scenes": scenes,
            }
        return {
            "schema_version": 1,
            "title": "The Lantern That Answered",
            "concept_summary": "A winter traveler follows a strange signal and discovers a reciprocal act of help.",
            "scenes": scenes,
        }
    if task == "script_draft":
        if data.get("draft_strategy") == "single-scene-word-fit-v1":
            desired_words = int(data["target_word_count"])
            text = str(data["spoken_text"])
            language = OutputLanguage(
                data.get("output_language", request.output_language.value)
            )
            if _is_factual_content(data):
                return {
                    "spoken_text": _factual_narrative_text(
                        data,
                        desired_words=desired_words,
                        language=language,
                        fallback=text,
                    )
                }
            sentence_index = 0
            while len(text.split()) < desired_words:
                text += " " + _fixture_sentence(language, sentence_index)
                sentence_index += 1
            return {
                "spoken_text": " ".join(text.split()[:desired_words]).rstrip(".,;:") + "."
            }
        if data.get("draft_strategy") == "single-scene-v1":
            language = OutputLanguage(
                data.get("output_language", request.output_language.value)
            )
            desired_words = int(data.get("target_word_count", 12))
            if _is_factual_content(data):
                return {
                    "spoken_text": _factual_narrative_text(
                        data,
                        desired_words=desired_words,
                        language=language,
                    )
                }
            sentence_index = max(0, int(data.get("scene_position", 1)) - 1)
            text = ""
            while len(text.split()) < desired_words:
                text = (text + " " + _fixture_sentence(language, sentence_index)).strip()
                sentence_index += 1
            text = " ".join(text.split()[:desired_words]).rstrip(".,;:") + "."
            return {"spoken_text": text}
        outline = data.get("outline", {})
        language = OutputLanguage(data.get("output_language", request.output_language.value))
        target_by_scene = {
            item["scene_id"]: int(item["target_word_count"])
            for item in data.get("scene_word_targets", [])
        }
        scenes = []
        for index, scene in enumerate(outline.get("scenes", [])):
            seconds = float(scene.get("provisional_seconds", 15))
            desired_words = target_by_scene.get(scene["scene_id"], max(8, round(seconds / 0.42)))
            if _is_factual_content(data):
                text = _factual_narrative_text(
                    {**data, "scene_position": index + 1},
                    desired_words=desired_words,
                    language=language,
                )
            else:
                text = ""
                sentence_index = index
                while len(text.split()) < desired_words:
                    text = (text + " " + _fixture_sentence(language, sentence_index)).strip()
                    sentence_index += 1
                text = " ".join(text.split()[:desired_words]).rstrip(".,;:") + "."
            scenes.append(
                {
                    "scene_id": scene["scene_id"],
                    "spoken_text": text,
                    "pause_after_seconds": 0 if index == len(outline.get("scenes", [])) - 1 else 0.15,
                }
            )
        return {"schema_version": 1, "title": outline.get("title", "Fixture Story"), "scenes": scenes}
    if task.startswith("review_"):
        if data.get("review_strategy") == "single-brief-constraint-v1":
            return {
                "satisfied": True,
                "scene_id": None,
                "evidence": "",
                "recommendation": "",
            }
        if data.get("review_strategy") == "single-finding-resolution-v1":
            return {
                "resolved": data.get("revised_spoken_text")
                != data.get("original_spoken_text"),
                "explanation": "The deterministic fixture confirms the bounded edit changed the Scene.",
            }
        review_type = {
            "review_story": "story",
            "review_spoken": "spoken",
            "review_constraints": "constraints",
        }[task]
        return {"schema_version": 1, "review_type": review_type, "passed": True, "findings": []}
    if task == "script_revision":
        if data.get("repair_strategy") == "factual-claim-repair-v1":
            text = str(data["spoken_text"])
            evidence = data.get("allowed_factual_evidence", [])
            if not evidence:
                language = OutputLanguage(
                    data.get("output_language", request.output_language.value)
                )
                return {
                    "spoken_text": (
                        "Siirry nyt seuraavaan konkreettiseen kohtaan."
                        if language is OutputLanguage.FINNISH
                        else "Move now to the next concrete point."
                    )
                }
            replacement_statement = str(evidence[0]["supported_statement"])
            for failed in data.get("failed_claims", []):
                text = text.replace(str(failed["exact_text"]), replacement_statement)
            return {"spoken_text": text}
        if data.get("revision_strategy") == "single-scene-word-fit-v1":
            desired_words = int(data["target_word_count"])
            text = str(data["spoken_text"])
            language = OutputLanguage(
                data.get("output_language", request.output_language.value)
            )
            if _is_factual_content(data):
                return {
                    "spoken_text": _factual_narrative_text(
                        data,
                        desired_words=desired_words,
                        language=language,
                        fallback=text,
                    )
                }
            sentence_index = 0
            while len(text.split()) < desired_words:
                text += " " + _fixture_sentence(language, sentence_index)
                sentence_index += 1
            return {
                "spoken_text": " ".join(text.split()[:desired_words]).rstrip(".,;:") + "."
            }
        if data.get("revision_strategy") == "single-scene-replacement-v1":
            return {"spoken_text": data["spoken_text"]}
        if data.get("revision_strategy") == "single-scene-finding-repair-v1":
            return {"spoken_text": data["spoken_text"]}
        return {"schema_version": 1, "script": data["script"], "dispositions": []}
    if task == "claim_inventory":
        if data.get("coverage_strategy") == "single-scene-claim-coverage-v1":
            return {"missing_claims": []}
        if data.get("inventory_strategy") == "single-scene-claim-extraction-v2":
            exact_text = str(data.get("spoken_text", "")).split(".", 1)[0].strip()
            return {
                "claims": (
                    [
                        {
                            "exact_text": exact_text,
                            "qualification": "",
                        }
                    ]
                    if exact_text
                    else []
                )
            }
        script = data["script"]
        evidence = data.get("research_pack", {}).get("evidence", [])
        evidence_ids = [str(item["evidence_id"]) for item in evidence[:1]]
        claims = []
        for index, scene in enumerate(script.get("scenes", []), start=1):
            exact_text = str(scene.get("spoken_text", "")).split(".", 1)[0].strip()
            if not exact_text:
                continue
            claims.append(
                {
                    "claim_id": f"claim-{index:03d}",
                    "scene_id": scene["scene_id"],
                    "exact_text": exact_text,
                    "evidence_ids": evidence_ids,
                    "qualification": "",
                }
            )
        return {"schema_version": 1, "claims": claims, "coverage_notes": "Fixture audit."}
    if task == "factual_review":
        if data.get("review_strategy") == "single-source-admission-v1":
            return {
                "verdict": "admit",
                "rationale": (
                    "The deterministic fixture source has explicit bounded provenance and an excerpt."
                ),
            }
        if data.get("review_strategy") == "single-evidence-source-entailment-v1":
            return {
                "verdict": "entailed",
                "rationale": (
                    "The deterministic fixture directly accepts its linked bounded source excerpt."
                ),
            }
        if data.get("review_strategy") in {
            "single-factual-visual-v1",
            "single-factual-visual-v2",
        }:
            return {
                "verdict": "grounded",
                "rationale": (
                    "The deterministic fixture visual contains only its supplied bounded semantics."
                ),
            }
        if data.get("review_strategy") == "single-claim-v1":
            evidence = data.get("evidence_records", [])
            exact_text = str(data.get("claim", {}).get("exact_text", "")).strip()
            if exact_text in data.get("host_owned_nonfactual_texts", []):
                return {
                    "verdict": "not_a_factual_claim",
                    "evidence_ids": [],
                    "rationale": "The exact text is one supplied host-owned neutral transition.",
                }
            if data.get("content_format") == "narrative":
                claim_text = " ".join(
                    exact_text.split()
                ).rstrip(".!?;:").casefold()
                matched = next(
                    (
                        item
                        for item in evidence
                        if " ".join(str(item.get("supported_statement", "")).split())
                        .rstrip(".!?;:")
                        .casefold()
                        == claim_text
                    ),
                    None,
                )
                return {
                    "verdict": "supported" if matched is not None else "unsupported",
                    "evidence_ids": (
                        [str(matched["evidence_id"])]
                        if matched is not None and matched.get("evidence_id")
                        else []
                    ),
                    "rationale": (
                        "The Claim exactly matches one supplied bounded Evidence statement."
                        if matched is not None
                        else "The Claim does not exactly match any supplied bounded Evidence statement."
                    ),
                }
            return {
                "verdict": "supported",
                "evidence_ids": [
                    str(item["evidence_id"])
                    for item in evidence[:1]
                    if item.get("evidence_id")
                ],
                "rationale": "The deterministic fixture treats its bounded evidence as direct support.",
            }
        claims = data.get("claim_inventory", {}).get("claims", [])
        return {
            "schema_version": 1,
            "passed": True,
            "claims": [
                {
                    "claim_id": claim["claim_id"],
                    "verdict": "supported",
                    "evidence_ids": claim.get("evidence_ids", []),
                    "rationale": "The deterministic fixture treats its bounded evidence as direct support.",
                }
                for claim in claims
            ],
            "uncovered_claims": [],
            "summary": "All fixture claims are covered.",
        }
    if task == "duration_repair":
        if data.get("repair_strategy") == "single-scene-word-fit-v1":
            desired = int(data["target_word_count"])
            text = str(data["spoken_text"])
            language = OutputLanguage(
                data.get("output_language", request.output_language.value)
            )
            if _is_factual_content(data):
                return {
                    "spoken_text": _factual_narrative_text(
                        data,
                        desired_words=desired,
                        language=language,
                        fallback=text,
                    )
                }
            sentence_index = 0
            while len(text.split()) < desired:
                text += " " + _fixture_sentence(language, sentence_index)
                sentence_index += 1
            return {
                "spoken_text": " ".join(text.split()[:desired]).rstrip(".,;:") + "."
            }
        if data.get("repair_strategy") == "single-scene-text-v3":
            desired = int(data["target_word_count"])
            text = str(data["spoken_text"])
            language = OutputLanguage(
                data.get("output_language", request.output_language.value)
            )
            if _is_factual_content(data):
                return {
                    "spoken_text": _factual_narrative_text(
                        data,
                        desired_words=desired,
                        language=language,
                        fallback=text,
                    )
                }
            sentence_index = 0
            while len(text.split()) < desired:
                text += " " + _fixture_sentence(language, sentence_index)
                sentence_index += 1
            return {
                "spoken_text": " ".join(text.split()[:desired]).rstrip(".,;:") + "."
            }
        if data.get("repair_strategy") == "single-scene-lengthening-v2":
            scene = data["script"]["scenes"][0]
            target = data["scene_repair_targets"][0]
            desired = int(target["target_word_count"])
            text = scene["spoken_text"]
            language = OutputLanguage(data.get("output_language", request.output_language.value))
            sentence_index = 0
            while len(text.split()) < desired:
                text += " " + _fixture_sentence(language, sentence_index)
                sentence_index += 1
            return {
                "schema_version": 1,
                "scene_id": "scene-001",
                "spoken_text": " ".join(text.split()[:desired]).rstrip(".,;:") + ".",
            }
        script = data["script"]
        scale = float(data.get("duration_scale", 1.0))
        selected = set(data.get("selected_scene_ids", []))
        for index, scene in enumerate(script.get("scenes", [])):
            if scene["scene_id"] not in selected:
                continue
            words = scene["spoken_text"].split()
            desired = max(4, round(len(words) * scale))
            if desired <= len(words):
                scene["spoken_text"] = " ".join(words[:desired]).rstrip(".,;:") + "."
            else:
                language = OutputLanguage(data.get("output_language", request.output_language.value))
                text = scene["spoken_text"]
                while len(text.split()) < desired:
                    text += " " + _fixture_sentence(language, index + len(text.split()))
                scene["spoken_text"] = " ".join(text.split()[:desired]).rstrip(".,;:") + "."
        return {"schema_version": 1, "script": script, "dispositions": []}
    if task == "visual_plan":
        if data.get("visual_strategy") == "foundation-v1":
            has_character = data.get("content_format", "narrative") == "narrative"
            return {
                "style": {
                    "description": data.get("style_description")
                    or "Naive hand-drawn shapes on a nearly white raster canvas.",
                    "palette": ["black", "white", "red", "blue", "pale gray"],
                    "line_style": "thin, slightly uneven black lines",
                    "background": "sparse naive marks with generous empty space",
                    "must_avoid": [
                        "written words",
                        "watermarks",
                        "photorealism",
                        "3D",
                        "gradients",
                    ],
                },
                "characters": (
                    [
                        {
                            "name": "Aino",
                            "signature_traits": ["round head", "red triangular scarf"],
                            "color_anchors": ["red scarf"],
                            "recurring_props": ["blue tin lantern"],
                            "body_form": "small upright stick figure; always bipedal",
                            "proportions": [
                                "round head",
                                "short straight limbs",
                                "same small scale",
                            ],
                            "face_and_markings": [
                                "two black dot eyes",
                                "no nose",
                                "plain white face",
                            ],
                            "wardrobe": ["red triangular scarf tied at the neck"],
                            "identity_constraints": [
                                "never quadrupedal",
                                "never remove or recolor the scarf",
                            ],
                        }
                    ]
                    if has_character
                    else []
                ),
            }
        if data.get("visual_strategy") == "single-factual-depiction-v1":
            has_supported_claim = bool(
                data.get("factual_grounding", {}).get("supported_claims")
            )
            return {
                "depiction": (
                    "A static literal arrangement shows only the visible subjects and relationship "
                    "authorized by the active supported claim."
                    if has_supported_claim
                    else "A neutral static arrangement shows only the supplied staging subjects."
                )
            }
        if data.get("visual_strategy") == "single-visual-v1":
            target = data["visual_target"]
            characters = data.get("character_identities", [])
            character_ids = [item["character_id"] for item in characters]
            has_previous = data.get("previous_visual") is not None
            factual_grounding = data.get("factual_grounding")
            static_factual_staging = (
                factual_grounding is not None
                and not factual_grounding.get("supported_claims")
            )
            return {
                "story_moment": (
                    "A neutral staged view of the current narrated subjects."
                    if static_factual_staging
                    else "A literal view of the currently supported narrated point."
                    if factual_grounding is not None
                    else target["narration_excerpt"]
                ),
                "subjects": (
                    ["Aino", "blue tin lantern"]
                    if character_ids
                    else ["hand", "literal narrated objects"]
                ),
                "action": (
                    "The supplied subjects remain motionless in a neutral static arrangement."
                    if static_factual_staging
                    else "Show only the literal supported relationship in the current narration."
                    if factual_grounding is not None
                    else "Show the literal visible action described by the current narration."
                ),
                "emotion": "clear focused curiosity",
                "environment": (
                    "a sparse neutral setting from the supplied staging context"
                    if factual_grounding is not None
                    else "a sparse setting consistent with the parent Outline Scene"
                ),
                "composition": (
                    "one static arrangement centered in a readable 16:9 frame"
                    if static_factual_staging
                    else "one large focal action centered in a readable 16:9 frame"
                ),
                "must_show": (
                    ["the staged subjects without a visible change", "clear silhouettes"]
                    if static_factual_staging
                    else ["the current narrated object and action", "clear silhouettes"]
                ),
                "must_avoid": ["written words", "labels", "future events"],
                "character_ids": character_ids,
                "continuity_from_previous": (
                    ["preserve the incoming visible state"] if has_previous else []
                ),
                "state_after_scene": (
                    ["the static arrangement remains unchanged"]
                    if static_factual_staging
                    else ["the current action leaves a visible result"]
                ),
                "identity_requirements": (
                    ["preserve every supplied Aino identity trait exactly"]
                    if character_ids
                    else []
                ),
                "persistent_elements": ["keep the established palette and object design"],
            }
        script = data["script"]
        has_character = data.get("content_format", "narrative") == "narrative"
        style = {
            "style_id": data.get("style_id", "ms_paint_stick"),
            "description": data.get("style_description")
            or "Naive MS Paint-like stick drawing on a nearly white raster canvas.",
            "palette": ["black", "white", "red", "blue", "pale gray"],
            "line_style": "thin, slightly uneven black lines",
            "background": "sparse naive marks with generous empty space",
            "must_avoid": ["written words", "watermarks", "photorealism", "3D", "gradients"],
        }
        characters = [
            {
                "character_id": "character-aino",
                "name": "Aino",
                "signature_traits": ["round head", "red triangular scarf"],
                "color_anchors": ["red scarf"],
                "recurring_props": ["blue tin lantern"],
                "body_form": "small upright stick figure; always bipedal",
                "proportions": ["round head", "short straight limbs", "same small scale"],
                "face_and_markings": ["two black dot eyes", "no nose", "plain white face"],
                "wardrobe": ["red triangular scarf tied at the neck"],
                "identity_constraints": ["never quadrupedal", "never remove or recolor the scarf"],
            }
        ] if has_character else []
        schedule = data.get("canonical_shot_schedule", [])
        if schedule:
            return {
                "schema_version": 1,
                "style_profile": style,
                "characters": characters,
                "duration_seconds": schedule[-1]["end_seconds"],
                "shots": [
                    {
                        **shot,
                        "story_moment": shot["narration_excerpt"],
                        "subjects": ["Aino", "blue tin lantern"] if has_character else ["blue boot", "snow crystals"],
                        "action": "Aino follows the tiny light" if has_character else "the boot compresses snow crystals",
                        "emotion": "curious and cautiously hopeful" if has_character else "clear and informative",
                        "environment": "sparse snowy path at blue hour",
                        "composition": "medium-wide 16:9 view with one clear focal action",
                        "must_show": ["red triangular scarf", "blue tin lantern"] if has_character else ["blue boot", "individual snow crystals"],
                        "must_avoid": ["written words", "crowd", "photorealism"],
                        "character_ids": ["character-aino"] if has_character else [],
                        "continuity_from_previous": ["preserve the same visible objects"],
                        "state_after_scene": ["the visible action advances one step"],
                        "identity_requirements": ["same Aino identity lock"] if has_character else [],
                        "persistent_elements": ["snowy path"],
                    }
                    for shot in schedule
                ],
            }
        return {
            "schema_version": 1,
            "style_profile": style,
            "characters": characters,
            "scenes": [
                {
                    "scene_id": scene["scene_id"],
                    "story_moment": scene["spoken_text"][:240],
                    "subjects": ["Aino", "blue tin lantern"] if has_character else ["blue boot", "snow crystals"],
                    "action": "Aino follows or shields the tiny light" if has_character else "a boot compresses snow crystals",
                    "emotion": "curious and cautiously hopeful" if has_character else "clear and informative",
                    "environment": "sparse snowy path at blue hour",
                    "composition": "medium-wide view with a clear silhouette and empty upper-right space",
                    "must_show": ["red triangular scarf", "blue tin lantern"] if has_character else ["blue boot", "snow crystals"],
                    "must_avoid": ["written words", "crowd", "photorealism"],
                    "character_ids": ["character-aino"] if has_character else [],
                    "continuity_from_previous": ["Aino keeps the red scarf and blue tin lantern"],
                    "state_after_scene": ["Aino and the lantern advance farther along the snowy path"],
                    "identity_requirements": [
                        "small upright bipedal stick figure with round white face and red triangular scarf"
                    ] if has_character else [],
                    "persistent_elements": ["red scarf", "blue tin lantern", "snowy path"],
                }
                for scene in script.get("scenes", [])
            ],
        }
    if task == "image_prompt_compile":
        visual = data.get("visual_shot") or data["visual_brief"]
        style = data["style_profile"]
        prompt = (
            f"A deliberately crude raster drawing on a nearly white 16:9 canvas. {visual['story_moment']} "
            f"Action: {visual['action']}. Composition: {visual['composition']}. "
            f"Use {style['line_style']}; flat limited colors; sparse background; readable silhouettes. "
            f"Must show: {', '.join(visual['must_show'])}. No letters, labels, captions, logos, signatures, "
            "watermarks, photorealism, 3D, gradients, polished vector geometry, or elaborate shading."
        )
        if data.get("compiler_strategy") == "prompt-content-v1":
            return {
                "prompt": prompt,
                "negative_prompt": (
                    "text, watermark, logo, photorealism, 3D, gradient, glossy concept art"
                ),
            }
        payload = {
            "schema_version": 1,
            "scene_id": visual["scene_id"],
            "target_backend_id": data["target_backend_id"],
            "prompt": prompt,
            "negative_prompt": "text, watermark, logo, photorealism, 3D, gradient, glossy concept art",
            "width": int(data.get("generation_width", 2048)),
            "height": int(data.get("generation_height", 1152)),
            "quality": str(data.get("image_quality", "medium")),
            "seed": None,
            "reference_paths": data.get("reference_paths", []),
            "settings": {},
        }
        if visual.get("shot_id"):
            payload["shot_id"] = visual["shot_id"]
        return payload
    if task == "visual_review":
        payload = {
            "scene_id": data["scene_id"],
            "passed": True,
            "hard_failure": False,
            "scores": {
                "subject_action": 5,
                "style_match": 5,
                "identity": 4,
                "composition": 5,
                "text_logo_free": 5,
                "audience_safety": 5,
            },
            "failures": [],
            "regeneration_instruction": "",
        }
        if data.get("shot_id"):
            payload["shot_id"] = data["shot_id"]
        return payload
    if task == "music_brief":
        duration = float(data["duration_seconds"])
        return {
            "schema_version": 1,
            "prompt": "Quiet instrumental winter ambience, sparse felt piano and soft bowed texture, no vocals.",
            "requested_duration_seconds": duration,
            "tempo_range_bpm": "55-70",
            "instrumentation": ["felt piano", "soft strings", "subtle wind texture"],
            "texture": "unobtrusive, spacious, gentle",
            "exclusions": ["lyrics", "speech", "audio logos", "recognizable melodies", "abrupt ending"],
            "sections": [
                {"start_seconds": 0, "end_seconds": duration, "mood": "curious warmth", "energy": "low"}
            ],
            "seamless_loop_preferred": False,
        }
    raise ValueError(f"deterministic Backend has no fixture for task {task}")


class _DeterministicBackend(Backend):
    def probe(self, *, live: bool = False) -> ProbeReport:
        return ProbeReport(
            backend_id=self.descriptor.backend_id,
            ready=True,
            items=[ProbeItem(name="fixture", ready=True, detail="built-in deterministic fixture")],
        )


class DeterministicSearchBackend(_DeterministicBackend):
    descriptor = BACKEND_DESCRIPTORS["deterministic:search"]

    def search(self, request: SearchRequest) -> SearchResult:
        source = ResearchSource(
            source_id="source-001",
            url="https://example.invalid/fixture",
            title="Offline deterministic research fixture",
            publisher="video-generator",
            language=request.language.value,
            excerpt="Dry snow can squeak underfoot; this fixture is creative input, not current evidence.",
        )
        return SearchResult(query=request.query, sources=[source])

    def fetch(self, request: SourceFetchRequest) -> SourceDocument:
        return SourceDocument(
            source_id=request.source.source_id,
            final_url=request.source.url,
            title=request.source.title,
            text=request.source.excerpt,
            content_sha256=request.source.content_sha256,
            mime_type="text/plain",
        )


class DeterministicStructuredTextBackend(_DeterministicBackend):
    descriptor = BACKEND_DESCRIPTORS["deterministic:structured"]

    def complete(self, request: StructuredTextRequest) -> StructuredTextResult:
        data = _fake_structured(request)
        return StructuredTextResult(
            data=data,
            raw_response={"fixture": True, "structured_data": data},
            usage=UsageRecord(task_id=request.task_id, backend_id=self.descriptor.backend_id),
        )


class DeterministicSpeechBackend(_DeterministicBackend):
    descriptor = BACKEND_DESCRIPTORS["deterministic:speech"]

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root.resolve()

    def synthesize(self, request: SpeechRequest) -> SpeechResult:
        duration = max(1.0, len(request.text.split()) * 0.42)
        sample_rate = 16000
        path = (self.workspace_root / request.output_path).resolve()
        path.relative_to(self.workspace_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame_count = round(duration * sample_rate)
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            chunk = bytearray()
            for index in range(frame_count):
                sample = round(900 * math.sin(2 * math.pi * 170 * index / sample_rate))
                chunk.extend(struct.pack("<h", sample))
                if len(chunk) >= 65536:
                    output.writeframesraw(chunk)
                    chunk.clear()
            if chunk:
                output.writeframesraw(chunk)
        return SpeechResult(
            asset=SpeechAsset(
                scene_id=request.scene_id,
                audio=MediaReference(
                    path=relative_path(path, self.workspace_root),
                    sha256=sha256_file(path),
                    mime_type="audio/wav",
                ),
                duration_seconds=duration,
                sample_rate=sample_rate,
                channels=1,
                word_timings=_word_timings(request.text, duration),
                timing_precision="word",
            ),
            usage=UsageRecord(task_id="narration_synthesis", backend_id=self.descriptor.backend_id),
        )


class DeterministicAlignmentBackend(_DeterministicBackend):
    descriptor = BACKEND_DESCRIPTORS["deterministic:alignment"]

    def align(self, request: AlignmentRequest) -> AlignmentResult:
        duration = max(1.0, len(request.transcript.split()) * 0.42)
        return AlignmentResult(
            recognized_words=_word_timings(request.transcript, duration),
            usage=UsageRecord(task_id="caption_alignment", backend_id=self.descriptor.backend_id),
        )


class DeterministicImageBackend(_DeterministicBackend):
    descriptor = BACKEND_DESCRIPTORS["deterministic:stick"]

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root.resolve()

    def generate(self, request: ImageRequest, output_path: Path) -> ImageResult:
        width, height = request.width, request.height
        output_path = output_path.resolve()
        output_path.relative_to(self.workspace_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pixels = bytearray([248, 248, 245]) * (width * height)

        def point(x: int, y: int, color: tuple[int, int, int], radius: int = 2) -> None:
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    px, py = x + dx, y + dy
                    if 0 <= px < width and 0 <= py < height:
                        offset = (py * width + px) * 3
                        pixels[offset : offset + 3] = bytes(color)

        def line(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int], radius: int = 2) -> None:
            steps = max(abs(x1 - x0), abs(y1 - y0), 1)
            for step in range(steps + 1):
                point(round(x0 + (x1 - x0) * step / steps), round(y0 + (y1 - y0) * step / steps), color, radius)

        center_x, center_y = width // 3, height // 2
        black = (25, 25, 25)
        red = (210, 45, 45)
        blue = (40, 100, 210)
        radius = max(15, height // 15)
        for angle in range(0, 360, 3):
            radians = math.radians(angle)
            point(center_x + round(radius * math.cos(radians)), center_y - radius + round(radius * math.sin(radians)), black, 2)
        line(center_x, center_y, center_x, center_y + height // 4, black)
        line(center_x, center_y + height // 12, center_x - width // 12, center_y + height // 6, black)
        line(center_x, center_y + height // 12, center_x + width // 10, center_y + height // 7, black)
        line(center_x, center_y + height // 4, center_x - width // 14, center_y + height // 3, black)
        line(center_x, center_y + height // 4, center_x + width // 14, center_y + height // 3, black)
        line(center_x - width // 20, center_y, center_x + width // 20, center_y + height // 14, red, 4)
        lantern_x, lantern_y = center_x + width // 5, center_y + height // 6
        for y in range(lantern_y - height // 20, lantern_y + height // 20):
            line(lantern_x - width // 40, y, lantern_x + width // 40, y, blue, 1)
        output_path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + pixels)
        return ImageResult(
            asset=ImageAsset(
                scene_id=request.scene_id,
                shot_id=getattr(request, "shot_id", None),
                image=MediaReference(
                    path=relative_path(output_path, self.workspace_root),
                    sha256=sha256_file(output_path),
                    mime_type="image/x-portable-pixmap",
                ),
                width=width,
                height=height,
                generation_settings={"fixture": True},
            ),
            usage=UsageRecord(task_id="image_generate", backend_id=self.descriptor.backend_id),
        )


class DeterministicMusicBackend(_DeterministicBackend):
    descriptor = BACKEND_DESCRIPTORS["deterministic:music"]

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root.resolve()

    def generate(self, request: MusicRequest) -> MusicResult:
        duration = request.brief.requested_duration_seconds
        path = (self.workspace_root / request.output_path).resolve()
        path.relative_to(self.workspace_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        sample_rate = 16000
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            chunk = bytearray()
            for index in range(round(duration * sample_rate)):
                envelope = min(1.0, index / sample_rate) * min(1.0, (duration * sample_rate - index) / sample_rate)
                sample = round(300 * envelope * math.sin(2 * math.pi * 110 * index / sample_rate))
                chunk.extend(struct.pack("<h", sample))
                if len(chunk) >= 65536:
                    output.writeframesraw(chunk)
                    chunk.clear()
            if chunk:
                output.writeframesraw(chunk)
        return MusicResult(
            asset=MusicAsset(
                audio=MediaReference(
                    path=relative_path(path, self.workspace_root),
                    sha256=sha256_file(path),
                    mime_type="audio/wav",
                ),
                duration_seconds=duration,
            ),
            usage=UsageRecord(task_id="music_generate", backend_id=self.descriptor.backend_id),
        )
