from __future__ import annotations

import json
import base64
from pathlib import Path
from typing import Any

import pytest

from video_generator.backends.gemini import (
    GeminiImageBackend,
    GeminiSearchBackend,
    GeminiStructuredTextBackend,
    _gemini_usage,
)
from video_generator.backends.openai import OpenAIStructuredTextBackend, OpenAIWebSearchBackend
from video_generator.contracts import ImageRequest, OutputLanguage, SearchRequest, StructuredTextRequest
from video_generator.errors import BackendError, ErrorKind
from video_generator.net import HttpResponse
from video_generator.profiles import BACKEND_DESCRIPTORS, PROFILES
from video_generator.registry import BackendRegistry


class StubHttpClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.requests: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> HttpResponse:
        self.requests.append({"method": method, "url": url, **kwargs})
        return HttpResponse(
            status=200,
            headers={"x-request-id": "request-1"},
            body=json.dumps(self.payload).encode("utf-8"),
        )


def _structured_request() -> StructuredTextRequest:
    return StructuredTextRequest(
        task_id="ideate",
        instructions="Return the requested object.",
        input_data={"topic": "snow"},
        output_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        output_language=OutputLanguage.ENGLISH,
        max_output_tokens=128,
    )


def test_openai_search_requests_provider_source_details() -> None:
    http = StubHttpClient(
        {
            "id": "response-1",
            "status": "completed",
            "output": [
                {
                    "type": "web_search_call",
                    "action": {
                        "sources": [
                            {"url": "https://example.com/source", "title": "Example source"}
                        ]
                    },
                },
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps(
                                {
                                    "sources": [
                                        {
                                            "url": "https://example.com/source",
                                            "title": "Example source",
                                            "publisher": "Example",
                                            "excerpt": "A short grounded summary.",
                                        }
                                    ]
                                }
                            ),
                        }
                    ],
                },
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
    )
    backend = OpenAIWebSearchBackend("test-key", http=http)

    result = backend.search(
        SearchRequest(query="winter detail", max_results=2, language=OutputLanguage.ENGLISH)
    )

    assert result.sources[0].url == "https://example.com/source"
    assert http.requests[0]["json_body"]["include"] == [
        "web_search_call.action.sources"
    ]
    assert http.requests[0]["json_body"]["model"] == "gpt-5.6-terra"


def test_openai_rejects_non_completed_response(tmp_path: Path) -> None:
    http = StubHttpClient({"id": "response-1", "status": "incomplete", "output": []})
    backend = OpenAIStructuredTextBackend(
        "test-key",
        backend_id="openai:gpt-5.6-terra",
        workspace_root=tmp_path,
        http=http,
    )

    with pytest.raises(BackendError, match="status 'incomplete'") as caught:
        backend.complete(_structured_request())

    assert caught.value.kind is ErrorKind.INVALID_OUTPUT


def test_gemini_usage_reads_current_interactions_fields() -> None:
    usage = _gemini_usage(
        {
            "id": "interaction-1",
            "usage": {"total_input_tokens": 123, "total_output_tokens": 45},
        },
        "ideate",
        "gemini:gemini-3.5-flash",
        0.35,
    )

    assert usage.input_units == 123
    assert usage.output_units == 45


def test_gemini_rejects_non_completed_interaction(tmp_path: Path) -> None:
    http = StubHttpClient({"id": "interaction-1", "status": "failed", "steps": []})
    backend = GeminiStructuredTextBackend("test-key", workspace_root=tmp_path, http=http)

    with pytest.raises(BackendError, match="status 'failed'") as caught:
        backend.complete(_structured_request())

    assert caught.value.kind is ErrorKind.INVALID_OUTPUT


def test_gemini_search_rejects_multiple_billable_queries() -> None:
    output = {
        "sources": [
            {
                "url": "https://example.com/source",
                "title": "Example source",
                "publisher": "Example",
                "excerpt": "A short grounded summary.",
            }
        ]
    }
    http = StubHttpClient(
        {
            "id": "interaction-1",
            "status": "completed",
            "steps": [
                {
                    "type": "google_search_call",
                    "arguments": {"queries": ["winter detail", "snow fact"]},
                },
                {
                    "type": "model_output",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(output),
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "url": "https://example.com/source",
                                    "title": "Example source",
                                }
                            ],
                        }
                    ],
                },
            ],
        }
    )
    backend = GeminiSearchBackend("test-key", http=http)

    with pytest.raises(BackendError, match="one-query bound") as caught:
        backend.search(
            SearchRequest(query="winter detail", max_results=2, language=OutputLanguage.ENGLISH)
        )

    assert caught.value.kind is ErrorKind.INVALID_OUTPUT


def test_curated_profiles_use_current_backends() -> None:
    assert PROFILES["cloud-openai"]["script_draft"] == "openai:gpt-5.6-terra"
    assert PROFILES["cloud-openai-gemini"]["script_draft"] == "openai:gpt-5.4-mini"
    assert PROFILES["cloud-openai-gemini"]["image_generate"] == "gemini:gemini-3.1-flash-image"
    assert PROFILES["cloud-openai-gemini"]["search"] == "ddgs:duckduckgo"
    assert PROFILES["local"]["script_draft"] == "local:llama-server"
    assert PROFILES["local"]["search"] == "ddgs:duckduckgo"
    assert BACKEND_DESCRIPTORS["openai:web"].model_id == "gpt-5.6-terra"
    assert BACKEND_DESCRIPTORS["openai:gpt-5.4-mini"].model_id == "gpt-5.4-mini-2026-03-17"
    assert BACKEND_DESCRIPTORS["gemini:search"].model_id == "gemini-3.5-flash"


def test_gemini_image_uses_current_jpeg_response_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    http = StubHttpClient(
        {
            "id": "interaction-image-1",
            "status": "completed",
            "steps": [
                {
                    "type": "model_output",
                    "content": [
                        {
                            "type": "image",
                            "data": base64.b64encode(b"jpeg-bytes").decode("ascii"),
                            "mime_type": "image/jpeg",
                        }
                    ],
                }
            ],
        }
    )
    monkeypatch.setattr("video_generator.backends.gemini.image_dimensions", lambda path: (1024, 576))
    backend = GeminiImageBackend(
        "test-key",
        workspace_root=tmp_path,
        run_root=tmp_path,
        http=http,
    )

    result = backend.generate(
        ImageRequest(
            scene_id="scene-001",
            target_backend_id="gemini:gemini-3.1-flash-image",
            prompt="A fox beside an amber lantern, no text.",
            width=2048,
            height=1152,
            quality="low",
        ),
        tmp_path / "generated.jpg",
    )

    assert http.requests[0]["json_body"]["response_format"] == {
        "type": "image",
        "mime_type": "image/jpeg",
        "aspect_ratio": "16:9",
        "image_size": "2K",
    }
    assert result.asset.image.mime_type == "image/jpeg"
    assert result.asset.image.path.endswith("generated.jpg")


def test_registry_applies_frozen_descriptor_to_adapter(tmp_path: Path, resolved_config) -> None:
    backend_id = "openai:gpt-5.6-terra"
    frozen = BACKEND_DESCRIPTORS[backend_id].model_copy(
        update={"model_id": "frozen-model-snapshot", "revision": "frozen-revision"}
    )

    with BackendRegistry(
        config=resolved_config,
        environment={"OPENAI_API_KEY": "test-key"},
        run_root=tmp_path,
        descriptors={backend_id: frozen},
    ) as registry:
        backend = registry.get(backend_id)

        assert backend.descriptor.model_id == "frozen-model-snapshot"
        assert backend.descriptor.revision == "frozen-revision"
