from __future__ import annotations

import json
from types import SimpleNamespace

from pydantic import BaseModel, field_validator

from video_generator.contracts import OutputLanguage, StructuredTextResult
from video_generator.errors import BackendError, ErrorKind
from video_generator.executor import TaskExecutor


class _Output(BaseModel):
    value: int


class _ValidatedOutput(BaseModel):
    value: str

    @field_validator("value")
    @classmethod
    def require_ok(cls, value: str) -> str:
        if value != "ok":
            raise ValueError("value must be ok")
        return value


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
