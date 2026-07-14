from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, field_validator

from video_generator.contracts import OutputLanguage, RevisedScript, StructuredTextResult
from video_generator.errors import BackendError, ErrorKind
from video_generator.executor import TaskExecutor, _canonicalize_host_owned_fields


class _Output(BaseModel):
    value: int


class _SpokenOutput(BaseModel):
    spoken_text: str


class _ValidatedOutput(BaseModel):
    value: str

    @field_validator("value")
    @classmethod
    def require_ok(cls, value: str) -> str:
        if value != "ok":
            raise ValueError("value must be ok")
        return value


class _Scene(BaseModel):
    scene_id: str


class _SceneList(BaseModel):
    scenes: list[_Scene]


def test_structured_repairs_one_invariant_failure() -> None:
    requests = []

    class Backend:
        def complete(self, request):
            requests.append(request)
            return StructuredTextResult(data={"value": len(requests)})

    backend = Backend()
    registry = SimpleNamespace(
        get=lambda backend_id: backend,
        descriptor=lambda backend_id: SimpleNamespace(reservation_usd=0.0),
    )
    store = SimpleNamespace(
        config=SimpleNamespace(
            task_bindings={"outline": "local:fixture"},
            output_language=OutputLanguage.ENGLISH,
        ),
        reserve_cost=lambda *args, **kwargs: None,
    )
    prompts = SimpleNamespace(
        get=lambda *args, **kwargs: SimpleNamespace(instructions="Return data.", version="fixture"),
        schema=lambda task_id: {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        },
    )
    executor = TaskExecutor(registry=registry, store=store, prompts=prompts)

    def require_two(output: _Output) -> None:
        if output.value != 2:
            raise BackendError("value must be two", kind=ErrorKind.INVALID_OUTPUT)

    execution = executor.structured("outline", {}, _Output, invariant=require_two)

    assert execution.artifact == _Output(value=2)
    assert len(requests) == 2
    assert requests[1].input_data["invalid_output"] == {"value": 1}
    assert requests[1].input_data["validation_errors"] == [
        {"type": "invariant", "msg": "value must be two", "loc": []}
    ]


def test_structured_assigns_outline_scene_ids_without_a_repair_call() -> None:
    requests = []

    class Backend:
        def complete(self, request):
            requests.append(request)
            return StructuredTextResult(
                data={"scenes": [{"scene_id": "scene_001"}, {"scene_id": "anything"}]}
            )

    backend = Backend()
    registry = SimpleNamespace(
        get=lambda backend_id: backend,
        descriptor=lambda backend_id: SimpleNamespace(reservation_usd=0.0),
    )
    store = SimpleNamespace(
        config=SimpleNamespace(
            task_bindings={"outline": "local:fixture"},
            output_language=OutputLanguage.ENGLISH,
        ),
        reserve_cost=lambda *args, **kwargs: None,
    )
    prompts = SimpleNamespace(
        get=lambda *args, **kwargs: SimpleNamespace(instructions="Return data.", version="fixture"),
        schema=lambda task_id: {},
    )

    execution = TaskExecutor(registry=registry, store=store, prompts=prompts).structured(
        "outline",
        {},
        _SceneList,
    )

    assert [scene.scene_id for scene in execution.artifact.scenes] == [
        "scene-001",
        "scene-002",
    ]
    assert len(requests) == 1
    assert execution.result.data["scenes"][0]["scene_id"] == "scene_001"


def test_timed_visual_plan_uses_the_canonical_host_schedule() -> None:
    original = {
        "duration_seconds": 99,
        "shots": [
            {
                "shot_id": "shot_001",
                "scene_id": "scene_999",
                "narration_excerpt": "changed",
                "start_seconds": 12,
                "end_seconds": 13,
            }
        ],
    }
    schedule = [
        {
            "shot_id": "shot-001",
            "scene_id": "scene-001",
            "narration_excerpt": "canonical words",
            "start_seconds": 0.0,
            "end_seconds": 3.2,
        }
    ]

    normalized = _canonicalize_host_owned_fields(
        "visual_plan",
        {"canonical_shot_schedule": schedule},
        original,
    )

    assert normalized == {"duration_seconds": 3.2, "shots": schedule}
    assert original["shots"][0]["shot_id"] == "shot_001"


def test_outline_keeps_only_host_known_evidence_references() -> None:
    normalized = _canonicalize_host_owned_fields(
        "outline",
        {
            "research_pack": {
                "findings": [{"finding_id": "finding-001"}],
                "evidence": [{"evidence_id": "evidence-001"}],
            }
        },
        {
            "scenes": [
                {
                    "scene_id": "wrong",
                    "evidence_ids": [
                        "finding-001",
                        "invented prose instead of an ID",
                        "evidence-001",
                    ],
                }
            ]
        },
    )

    assert normalized["scenes"] == [
        {
            "scene_id": "scene-001",
            "evidence_ids": ["finding-001", "evidence-001"],
        }
    ]


def test_claim_inventory_assigns_ids_scene_and_known_evidence() -> None:
    normalized = _canonicalize_host_owned_fields(
        "claim_inventory",
        {
            "script": {
                "scenes": [
                    {"scene_id": "scene-001", "spoken_text": "Metal carries heat away."}
                ]
            },
            "research_pack": {"evidence": [{"evidence_id": "ev-001"}]},
        },
        {
            "claims": [
                {
                    "claim_id": "invented-id",
                    "scene_id": "wrong-scene",
                    "exact_text": "Metal carries heat away.",
                    "evidence_ids": ["ev-001", "made-up"],
                }
            ]
        },
    )

    assert normalized["claims"] == [
        {
            "claim_id": "claim-001",
            "scene_id": "scene-001",
            "exact_text": "Metal carries heat away.",
            "evidence_ids": ["ev-001"],
        }
    ]


def test_scene_claim_extraction_keeps_only_semantic_fields() -> None:
    normalized = _canonicalize_host_owned_fields(
        "claim_inventory",
        {
            "inventory_strategy": "single-scene-claim-extraction-v2",
            "spoken_text": "Metal carries heat away.",
            "research_pack": {"evidence": [{"evidence_id": "ev-001"}]},
        },
        {
            "claims": [
                {
                    "exact_text": "Metal carries heat away.",
                    "evidence_ids": ["ev-001", "made-up"],
                    "qualification": "At equal starting temperature.",
                }
            ]
        },
    )

    assert normalized == {
        "claims": [
            {
                "exact_text": "Metal carries heat away.",
                "evidence_ids": ["ev-001"],
                "qualification": "At equal starting temperature.",
            }
        ]
    }
    assert _canonicalize_host_owned_fields(
        "claim_inventory",
        {
            "inventory_strategy": "single-scene-claim-extraction-v2",
            "spoken_text": "Look at this spoon.",
            "research_pack": {"evidence": []},
        },
        {"claims": []},
    ) == {"claims": []}


def test_factual_review_assigns_claim_ids_and_derives_pass() -> None:
    normalized = _canonicalize_host_owned_fields(
        "factual_review",
        {
            "claim_inventory": {"claims": [{"claim_id": "claim-001"}]},
            "research_pack": {"evidence": [{"evidence_id": "ev-001"}]},
        },
        {
            "passed": False,
            "claims": [
                {
                    "claim_id": "wrong",
                    "verdict": "supported",
                    "evidence_ids": ["ev-001", "unknown"],
                    "rationale": "Directly supported.",
                }
            ],
            "uncovered_claims": [],
        },
    )

    assert normalized["passed"] is True
    assert normalized["claims"][0]["claim_id"] == "claim-001"
    assert normalized["claims"][0]["evidence_ids"] == ["ev-001"]


def test_review_assigns_host_owned_type_ids_and_pass() -> None:
    normalized = _canonicalize_host_owned_fields(
        "review_spoken",
        {},
        {
            "review_type": "story",
            "passed": True,
            "findings": [
                {
                    "finding_id": "invented",
                    "severity": "minor",
                    "scene_id": "scene-001",
                    "evidence": "Awkward phrase.",
                    "recommendation": "Make it natural.",
                }
            ],
        },
    )

    assert normalized["review_type"] == "spoken"
    assert normalized["passed"] is False
    assert normalized["findings"][0]["finding_id"] == "spoken:finding-001"


def test_structured_repair_uses_word_range_when_available() -> None:
    requests = []

    class Backend:
        def complete(self, request):
            requests.append(request)
            return StructuredTextResult(data={"value": len(requests)})

    backend = Backend()
    registry = SimpleNamespace(
        get=lambda backend_id: backend,
        descriptor=lambda backend_id: SimpleNamespace(reservation_usd=0.0),
    )
    store = SimpleNamespace(
        config=SimpleNamespace(
            task_bindings={"script_draft": "local:fixture"},
            output_language=OutputLanguage.ENGLISH,
        ),
        reserve_cost=lambda *args, **kwargs: None,
    )
    prompts = SimpleNamespace(
        get=lambda *args, **kwargs: SimpleNamespace(instructions="Return data.", version="fixture"),
        schema=lambda task_id: {},
    )
    executor = TaskExecutor(registry=registry, store=store, prompts=prompts)

    def require_two(output: _Output) -> None:
        if output.value != 2:
            raise BackendError(
                "Narration Script has 68 words; required inclusive range is 74-87",
                kind=ErrorKind.INVALID_OUTPUT,
                details={
                    "actual_word_count": 68,
                    "minimum_word_count": 74,
                    "maximum_word_count": 87,
                    "target_word_count": 80,
                    "word_delta": 12,
                },
            )

    execution = executor.structured(
        "script_draft",
        {},
        _Output,
        invariant=require_two,
    )

    assert execution.artifact == _Output(value=2)
    assert "between 74 and 87 whitespace-separated words inclusive" in requests[1].instructions
    assert "aiming near 80" in requests[1].instructions
    assert "Add exactly" not in requests[1].instructions
    assert "returning the unchanged script is invalid" in requests[1].instructions
    assert "original_input" not in requests[1].input_data
    assert requests[1].input_data["validation_errors"][0]["details"]["word_delta"] == 12


def test_structured_repair_tolerates_incomplete_word_count_diagnostics() -> None:
    requests = []

    class Backend:
        def complete(self, request):
            requests.append(request)
            return StructuredTextResult(data={"value": len(requests)})

    backend = Backend()
    registry = SimpleNamespace(
        get=lambda backend_id: backend,
        descriptor=lambda backend_id: SimpleNamespace(reservation_usd=0.0),
    )
    store = SimpleNamespace(
        config=SimpleNamespace(
            task_bindings={"script_draft": "local:fixture"},
            output_language=OutputLanguage.ENGLISH,
        ),
        reserve_cost=lambda *args, **kwargs: None,
    )
    prompts = SimpleNamespace(
        get=lambda *args, **kwargs: SimpleNamespace(instructions="Return data.", version="fixture"),
        schema=lambda task_id: {},
    )
    executor = TaskExecutor(registry=registry, store=store, prompts=prompts)

    def require_two(output: _Output) -> None:
        if output.value != 2:
            raise BackendError(
                "word count is invalid",
                kind=ErrorKind.INVALID_OUTPUT,
                details={"actual_word_count": 13, "word_delta": 2},
            )

    execution = executor.structured("script_draft", {}, _Output, invariant=require_two)

    assert execution.artifact == _Output(value=2)
    assert len(requests) == 2
    assert "Add exactly" not in requests[1].instructions


def test_structured_text_only_word_repair_does_not_request_host_fields() -> None:
    requests = []

    class Backend:
        def complete(self, request):
            requests.append(request)
            count = 13 if len(requests) == 1 else 15
            return StructuredTextResult(data={"spoken_text": " ".join(["word"] * count)})

    backend = Backend()
    registry = SimpleNamespace(
        get=lambda backend_id: backend,
        descriptor=lambda backend_id: SimpleNamespace(reservation_usd=0.0),
    )
    store = SimpleNamespace(
        config=SimpleNamespace(
            task_bindings={"script_draft": "local:fixture"},
            output_language=OutputLanguage.ENGLISH,
        ),
        reserve_cost=lambda *args, **kwargs: None,
    )
    prompts = SimpleNamespace(
        get=lambda *args, **kwargs: SimpleNamespace(
            instructions="Return spoken text.", version="fixture"
        ),
        schema=lambda task_id: {},
    )
    executor = TaskExecutor(registry=registry, store=store, prompts=prompts)

    def require_fifteen(output: _SpokenOutput) -> None:
        actual = len(output.spoken_text.split())
        if actual != 15:
            raise BackendError(
                "Draft Scene has 13 words; required exactly 15",
                kind=ErrorKind.INVALID_OUTPUT,
                details={
                    "actual_word_count": actual,
                    "target_word_count": 15,
                    "word_delta": 15 - actual,
                },
            )

    execution = executor.structured(
        "script_draft",
        {},
        _SpokenOutput,
        invariant=require_fifteen,
    )

    assert len(execution.artifact.spoken_text.split()) == 15
    assert set(requests[1].output_schema["properties"]) == {"spoken_text"}
    assert "exactly the single spoken_text field" in requests[1].instructions
    assert "Scene IDs" not in requests[1].instructions


def test_structured_validation_repair_diagnostics_are_json_serializable() -> None:
    requests = []

    class Backend:
        def complete(self, request):
            requests.append(request)
            if len(requests) == 2:
                json.dumps(request.input_data)
            return StructuredTextResult(data={"value": "bad" if len(requests) == 1 else "ok"})

    backend = Backend()
    registry = SimpleNamespace(
        get=lambda backend_id: backend,
        descriptor=lambda backend_id: SimpleNamespace(reservation_usd=0.0),
    )
    store = SimpleNamespace(
        config=SimpleNamespace(
            task_bindings={"outline": "local:fixture"},
            output_language=OutputLanguage.ENGLISH,
        ),
        reserve_cost=lambda *args, **kwargs: None,
    )
    prompts = SimpleNamespace(
        get=lambda *args, **kwargs: SimpleNamespace(instructions="Return data.", version="fixture"),
        schema=lambda task_id: {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    )
    executor = TaskExecutor(registry=registry, store=store, prompts=prompts)

    execution = executor.structured("outline", {}, _ValidatedOutput)

    assert execution.artifact == _ValidatedOutput(value="ok")
    assert requests[1].input_data["validation_errors"] == [
        {"type": "value_error", "loc": ("value",), "msg": "Value error, value must be ok"}
    ]


@pytest.mark.parametrize("required_finding_ids", [[], ["story-001", "spoken-002"]])
def test_script_revision_schema_constrains_required_findings(
    required_finding_ids: list[str],
) -> None:
    requests = []

    class Backend:
        def complete(self, request):
            requests.append(request)
            return StructuredTextResult(
                data={
                    "schema_version": 1,
                    "script": {
                        "schema_version": 1,
                        "title": "Fixture",
                        "scenes": [
                            {
                                "scene_id": "scene-001",
                                "spoken_text": "A short fixture scene.",
                                "pause_after_seconds": 0,
                            }
                        ],
                    },
                    "dispositions": [
                        {
                            "finding_id": finding_id,
                            "disposition": "applied",
                            "explanation": "Addressed.",
                        }
                        for finding_id in required_finding_ids
                    ],
                }
            )

    backend = Backend()
    registry = SimpleNamespace(
        get=lambda backend_id: backend,
        descriptor=lambda backend_id: SimpleNamespace(reservation_usd=0.0),
    )
    store = SimpleNamespace(
        config=SimpleNamespace(
            task_bindings={"script_revision": "local:fixture"},
            output_language=OutputLanguage.ENGLISH,
        ),
        reserve_cost=lambda *args, **kwargs: None,
    )
    prompts = SimpleNamespace(
        get=lambda *args, **kwargs: SimpleNamespace(instructions="Return data.", version="fixture"),
        schema=lambda task_id: {},
    )

    TaskExecutor(registry=registry, store=store, prompts=prompts).structured(
        "script_revision",
        {"required_finding_ids": required_finding_ids},
        RevisedScript,
    )

    schema = requests[0].output_schema
    dispositions = schema["properties"]["dispositions"]
    finding_id = schema["$defs"]["RevisionDisposition"]["properties"]["finding_id"]
    if required_finding_ids:
        assert dispositions["minItems"] == len(required_finding_ids)
        assert dispositions["maxItems"] == len(required_finding_ids)
        assert finding_id["enum"] == required_finding_ids
    else:
        assert dispositions == {"const": []}
        assert "enum" not in finding_id


def test_local_structured_output_allows_two_validation_repairs() -> None:
    requests = []

    class Backend:
        def complete(self, request):
            requests.append(request)
            return StructuredTextResult(data={"value": "ok" if len(requests) == 3 else "bad"})

    backend = Backend()
    registry = SimpleNamespace(
        get=lambda backend_id: backend,
        descriptor=lambda backend_id: SimpleNamespace(reservation_usd=0.0, cloud=False),
    )
    store = SimpleNamespace(
        config=SimpleNamespace(
            task_bindings={"outline": "local:fixture"},
            output_language=OutputLanguage.ENGLISH,
        ),
        reserve_cost=lambda *args, **kwargs: None,
    )
    prompts = SimpleNamespace(
        get=lambda *args, **kwargs: SimpleNamespace(instructions="Return data.", version="fixture"),
        schema=lambda task_id: {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    )

    execution = TaskExecutor(registry=registry, store=store, prompts=prompts).structured(
        "outline",
        {},
        _ValidatedOutput,
    )

    assert execution.artifact == _ValidatedOutput(value="ok")
    assert len(requests) == 3
    assert requests[2].input_data["invalid_output"] == {"value": "bad"}


def test_cloud_structured_output_remains_capped_at_one_validation_repair() -> None:
    requests = []

    class Backend:
        def complete(self, request):
            requests.append(request)
            return StructuredTextResult(data={"value": "bad"})

    backend = Backend()
    registry = SimpleNamespace(
        get=lambda backend_id: backend,
        descriptor=lambda backend_id: SimpleNamespace(reservation_usd=0.0, cloud=True),
    )
    store = SimpleNamespace(
        config=SimpleNamespace(
            task_bindings={"outline": "cloud:fixture"},
            output_language=OutputLanguage.ENGLISH,
        ),
        reserve_cost=lambda *args, **kwargs: None,
    )
    prompts = SimpleNamespace(
        get=lambda *args, **kwargs: SimpleNamespace(instructions="Return data.", version="fixture"),
        schema=lambda task_id: {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    )

    with pytest.raises(BackendError, match="after one repair"):
        TaskExecutor(registry=registry, store=store, prompts=prompts).structured(
            "outline",
            {},
            _ValidatedOutput,
        )

    assert len(requests) == 2


@pytest.mark.parametrize("task_id", ["script_draft", "script_revision", "duration_repair"])
def test_cloud_length_sensitive_output_allows_two_validation_repairs(task_id: str) -> None:
    requests = []

    class Backend:
        def complete(self, request):
            requests.append(request)
            return StructuredTextResult(data={"value": "ok" if len(requests) == 3 else "bad"})

    backend = Backend()
    registry = SimpleNamespace(
        get=lambda backend_id: backend,
        descriptor=lambda backend_id: SimpleNamespace(reservation_usd=0.0, cloud=True),
    )
    store = SimpleNamespace(
        config=SimpleNamespace(
            task_bindings={task_id: "cloud:fixture"},
            output_language=OutputLanguage.ENGLISH,
        ),
        reserve_cost=lambda *args, **kwargs: None,
    )
    prompts = SimpleNamespace(
        get=lambda *args, **kwargs: SimpleNamespace(instructions="Return data.", version="fixture"),
        schema=lambda task_id: {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    )

    execution = TaskExecutor(registry=registry, store=store, prompts=prompts).structured(
        task_id,
        {},
        _ValidatedOutput,
    )

    assert execution.artifact == _ValidatedOutput(value="ok")
    assert len(requests) == 3
    assert "inclusive numeric range" in requests[1].instructions


def test_image_prompt_compile_request_is_english_for_finnish_run() -> None:
    requests = []

    class Backend:
        def complete(self, request):
            requests.append(request)
            return StructuredTextResult(data={"value": 1})

    backend = Backend()
    registry = SimpleNamespace(
        get=lambda backend_id: backend,
        descriptor=lambda backend_id: SimpleNamespace(reservation_usd=0.0),
    )
    store = SimpleNamespace(
        config=SimpleNamespace(
            task_bindings={"image_prompt_compile": "local:fixture"},
            output_language=OutputLanguage.FINNISH,
        ),
        reserve_cost=lambda *args, **kwargs: None,
    )
    prompts = SimpleNamespace(
        get=lambda *args, **kwargs: SimpleNamespace(instructions="Return data.", version="fixture"),
        schema=lambda task_id: {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        },
    )

    TaskExecutor(registry=registry, store=store, prompts=prompts).structured(
        "image_prompt_compile",
        {},
        _Output,
    )

    assert requests[0].output_language is OutputLanguage.ENGLISH
