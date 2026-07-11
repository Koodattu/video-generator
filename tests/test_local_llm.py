from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from video_generator.local_llm import LocalLlmProfile
from video_generator.runners import RunnerManager
from video_generator.setup import prepare_llama_server_backend
from video_generator.util import sha256_file
from video_generator.workers import llama_server
from video_generator.workers.main import LlamaServerWorker, Paths


def _profile_values(**updates):
    values = {
        "profile_id": "fixture-no-mtp",
        "model_id": "fixture/model",
        "model_repo": "fixture/model-GGUF",
        "model_revision": "1" * 40,
        "model_path": "model.gguf",
        "model_sha256": "2" * 64,
        "license_name": "Apache-2.0",
        "llama_cpp_revision": "3" * 40,
        "llama_server_path": "llama-server.exe",
        "llama_server_sha256": "4" * 64,
        "llama_runtime_files": {"llama-server.exe": "4" * 64},
    }
    values.update(updates)
    return values


def test_local_llm_profile_requires_full_nonzero_provenance() -> None:
    with pytest.raises(ValidationError, match="full 40-character commit"):
        LocalLlmProfile.model_validate(_profile_values(model_revision="main"))

    with pytest.raises(ValidationError, match="all draft_model fields"):
        LocalLlmProfile.model_validate(
            _profile_values(speculation="draft-mtp", draft_model_path="draft.gguf")
        )


def test_prepare_llama_server_profile_freezes_runtime_and_launch_settings(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    server = runtime / "llama-server.exe"
    server.write_bytes(b"server")
    dll = runtime / "ggml-cuda.dll"
    dll.write_bytes(b"dll")
    profile_path = tmp_path / "llm.toml"
    profile_path.write_text(
        "\n".join(
            [
                "schema_version = 1",
                'profile_id = "fixture-mtp"',
                'model_id = "fixture/model"',
                'model_repo = "fixture/model-GGUF"',
                f'model_revision = "{"1" * 40}"',
                'model_path = "model.gguf"',
                f'model_sha256 = "{sha256_file(model)}"',
                'license_name = "Apache-2.0"',
                f'llama_cpp_revision = "{"2" * 40}"',
                'llama_server_path = "runtime/llama-server.exe"',
                f'llama_server_sha256 = "{sha256_file(server)}"',
                "context_size = 32768",
                'speculation = "draft-mtp"',
                "speculative_tokens = 2",
                "[llama_runtime_files]",
                f'"llama-server.exe" = "{sha256_file(server)}"',
                f'"ggml-cuda.dll" = "{sha256_file(dll)}"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = prepare_llama_server_backend(tmp_path, profile_path)

    assert result.ready
    manager = RunnerManager(project_root=tmp_path, run_root=tmp_path / "runs" / "fixture")
    spec = manager.load_spec("local:llama-server")
    assert spec.platform == "native"
    assert spec.command[-2:] == ["--kind", "llama-server"]
    arguments = json.loads(spec.environment["VIDEO_GENERATOR_LLAMA_ARGUMENTS"])
    assert arguments[arguments.index("--ctx-size") + 1] == "32768"
    assert arguments[arguments.index("--spec-type") + 1] == "draft-mtp"
    assert spec.metadata["local_llm_profile"]["profile_id"] == "fixture-mtp"
    assert any(path.endswith("llama-server.exe") for path in spec.runtime_files)
    assert any(path.endswith("ggml-cuda.dll") for path in spec.runtime_files)


class _FakeResponse:
    def __init__(self, value: dict) -> None:
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self, maximum: int) -> bytes:
        return json.dumps(self.value).encode("utf-8")


class _FakeOpener:
    def __init__(self, requests: list) -> None:
        self.requests = requests

    def open(self, request, timeout):
        self.requests.append(request)
        if request.full_url.endswith("/health"):
            return _FakeResponse({"status": "ok"})
        return _FakeResponse(
            {
                "id": "local-request",
                "choices": [{"message": {"content": '{"title":"Yö"}'}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            }
        )


class _FakeProcess:
    def __init__(self) -> None:
        self.pid = 4242
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 0

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def kill(self) -> None:
        self.returncode = -9


def test_llama_server_session_is_loopback_authenticated_and_reused(
    tmp_path: Path,
    monkeypatch,
) -> None:
    server = tmp_path / "llama-server.exe"
    model = tmp_path / "model.gguf"
    server.write_bytes(b"server")
    model.write_bytes(b"model")
    requests = []
    popen_calls = []
    process = _FakeProcess()

    monkeypatch.setattr(
        llama_server.urllib.request,
        "build_opener",
        lambda *handlers: _FakeOpener(requests),
    )
    monkeypatch.setattr(
        llama_server.subprocess,
        "Popen",
        lambda command, **kwargs: popen_calls.append((command, kwargs)) or process,
    )
    monkeypatch.setattr(
        llama_server,
        "gpu_snapshot",
        lambda: llama_server.GpuSnapshot(True, 1000, (() if process.poll() is not None else (4242,))),
    )
    session = llama_server.LlamaServerSession(
        executable=server,
        model=model,
        draft_model=None,
        arguments=["--ctx-size", "32768", "--parallel", "1"],
    )

    session.start()
    session.chat_completion({"messages": [{"role": "user", "content": "Hei"}]})
    session.chat_completion({"messages": [{"role": "user", "content": "Yö"}]})
    cleanup = session.close()

    assert len(popen_calls) == 1
    command, kwargs = popen_calls[0]
    assert command.count("127.0.0.1") == 1
    assert session.api_key == ""
    assert not any("Bearer" in value for value in command)
    assert kwargs["env"]["LLAMA_API_KEY"]
    assert cleanup["process_exited"]
    assert cleanup["gpu_process_released"]
    chat_requests = [request for request in requests if request.full_url.endswith("/v1/chat/completions")]
    assert len(chat_requests) == 2
    assert chat_requests[0].get_header("Authorization").startswith("Bearer ")


def test_llama_server_session_rejects_network_and_model_overrides(tmp_path: Path) -> None:
    server = tmp_path / "llama-server.exe"
    model = tmp_path / "model.gguf"
    server.write_bytes(b"server")
    model.write_bytes(b"model")

    with pytest.raises(ValueError, match="managed by the orchestrator"):
        llama_server.LlamaServerSession(
            executable=server,
            model=model,
            draft_model=None,
            arguments=["--host", "0.0.0.0"],
        )


def test_llama_worker_maps_finnish_schema_request(monkeypatch, tmp_path: Path) -> None:
    model = tmp_path / ".cache" / "models" / "llm" / "model.gguf"
    server = tmp_path / ".cache" / "runtimes" / "llama.cpp" / "llama-server.exe"
    model.parent.mkdir(parents=True)
    server.parent.mkdir(parents=True)
    model.write_bytes(b"model")
    server.write_bytes(b"server")
    captured: list[dict] = []

    class FakeSession:
        def __init__(self, **kwargs) -> None:
            self.process = SimpleNamespace(pid=9)
            self.base_url = "http://127.0.0.1:1234"
            self.baseline = SimpleNamespace(used_mb=100)
            self.peak_used_mb = 200
            self.startup_elapsed_seconds = 1.25

        def start(self) -> None:
            return None

        def chat_completion(self, payload):
            captured.append(payload)
            return {
                "id": "request-fi",
                "choices": [{"message": {"content": '{"title":"Yö"}'}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            }

        def close(self):
            return {"process_exited": True, "gpu_process_released": True}

    monkeypatch.setattr(llama_server, "LlamaServerSession", FakeSession)
    monkeypatch.setenv("VIDEO_GENERATOR_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("VIDEO_GENERATOR_RUN_ROOT", str(tmp_path / "runs" / "fixture"))
    monkeypatch.setenv("VIDEO_GENERATOR_MODEL_PATH", str(model))
    monkeypatch.setenv("VIDEO_GENERATOR_LLAMA_SERVER", str(server))
    monkeypatch.setenv("VIDEO_GENERATOR_LLAMA_ARGUMENTS", "[]")
    worker = LlamaServerWorker(Paths())

    result = worker.dispatch(
        "structured_text.complete",
        {
            "task_id": "outline",
            "instructions": "Kirjoita suomeksi.",
            "input_data": {"topic": "yö"},
            "output_schema": {
                "type": "object",
                "properties": {"title": {"type": "string"}},
                "required": ["title"],
            },
            "max_output_tokens": 100,
            "media_inputs": [],
        },
    )

    assert result["data"] == {"title": "Yö"}
    assert captured[0]["response_format"]["type"] == "json_schema"
    assert "yö" in captured[0]["messages"][1]["content"]
