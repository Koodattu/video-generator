from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from video_generator.workers.main import LlamaServerWorker


def test_llama_worker_uses_llama_cpp_nested_json_schema(monkeypatch) -> None:
    calls: list[dict] = []

    class Session:
        baseline = SimpleNamespace(used_mb=100)
        peak_used_mb = 120
        startup_elapsed_seconds = 1.0

        def chat_completion(self, payload: dict) -> dict:
            calls.append(payload)
            return {
                "id": "fixture",
                "choices": [
                    {
                        "message": {"content": '{"schema_version":1}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }

    worker = object.__new__(LlamaServerWorker)
    worker.model_path = Path("model.gguf")
    worker.session = Session()
    schema = {
        "type": "object",
        "properties": {
            "schema_version": {"type": "integer", "const": 1},
            "pause": {"type": "number", "minimum": 0, "maximum": 3},
            "text": {
                "type": "string",
                "minLength": 1,
                "maxLength": 10000,
                "pattern": "^[a-z]+$",
            },
            "items": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 20,
            },
        },
        "required": ["schema_version", "pause", "text", "items"],
        "additionalProperties": False,
    }
    monkeypatch.setenv("VIDEO_GENERATOR_RUNTIME_REVISION", "runtime")
    monkeypatch.setenv("VIDEO_GENERATOR_MODEL_REVISION", "model")

    result = worker.dispatch(
        "structured_text.complete",
        {
            "task_id": "research",
            "instructions": "Return only the requested data.",
            "input_data": {},
            "output_schema": schema,
            "max_output_tokens": 1000,
            "media_inputs": [],
        },
    )

    request = calls[0]
    assert request["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "research",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "schema_version": {"type": "integer", "const": 1},
                    "pause": {"type": "number"},
                    "text": {"type": "string", "minLength": 1},
                    "items": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["schema_version", "pause", "text", "items"],
                "additionalProperties": False,
            },
        },
    }
    assert json.dumps(schema, ensure_ascii=False, separators=(",", ":")) in request["messages"][0]["content"]
    assert request["temperature"] == 0.65
    assert result["data"] == {"schema_version": 1}
    assert result["finish_reason"] == "stop"


def test_llama_worker_uses_low_variance_sampling_for_factual_review() -> None:
    calls: list[dict] = []

    class Session:
        baseline = SimpleNamespace(used_mb=100)
        peak_used_mb = 120
        startup_elapsed_seconds = 1.0

        def chat_completion(self, payload: dict) -> dict:
            calls.append(payload)
            return {
                "id": "fixture",
                "choices": [
                    {
                        "message": {
                            "content": '{"verdict":"grounded","rationale":"The bounded visual is directly grounded."}'
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            }

    worker = object.__new__(LlamaServerWorker)
    worker.model_path = Path("model.gguf")
    worker.session = Session()
    worker.dispatch(
        "structured_text.complete",
        {
            "task_id": "factual_review",
            "instructions": "Return only the requested decision.",
            "input_data": {},
            "output_schema": {
                "type": "object",
                "properties": {
                    "verdict": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["verdict", "rationale"],
                "additionalProperties": False,
            },
            "max_output_tokens": 600,
            "media_inputs": [],
        },
    )

    assert calls[0]["temperature"] == 0.1


def test_llama_worker_rejects_truncated_structured_output() -> None:
    class Session:
        baseline = SimpleNamespace(used_mb=100)
        peak_used_mb = 120
        startup_elapsed_seconds = 1.0

        def chat_completion(self, payload: dict) -> dict:
            return {
                "id": "fixture",
                "choices": [
                    {
                        "message": {"content": '{"schema_version":1}'},
                        "finish_reason": "length",
                    }
                ],
            }

    worker = object.__new__(LlamaServerWorker)
    worker.model_path = Path("model.gguf")
    worker.session = Session()

    with pytest.raises(ValueError, match="did not finish normally: length"):
        worker.dispatch(
            "structured_text.complete",
            {
                "task_id": "research",
                "instructions": "Return only the requested data.",
                "input_data": {},
                "output_schema": {
                    "type": "object",
                    "properties": {"schema_version": {"type": "integer"}},
                    "required": ["schema_version"],
                    "additionalProperties": False,
                },
                "max_output_tokens": 1000,
                "media_inputs": [],
            },
        )
