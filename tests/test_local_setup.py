from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from video_generator import setup
from video_generator.errors import VideoGeneratorError
from video_generator.profiles import BACKEND_DESCRIPTORS, PROFILES
from video_generator.runners import decode_wsl_output
from video_generator.util import sha256_file


def test_decode_wsl_output_accepts_redirected_utf16le() -> None:
    raw = "Ubuntu\r\nDebian\r\n".encode("utf-16-le")

    assert decode_wsl_output(raw).splitlines() == ["Ubuntu", "Debian"]


def test_native_cuda_runner_uses_automatic_torch_backend(
    tmp_path: Path,
    monkeypatch,
) -> None:
    commands: list[list[str]] = []
    definition = setup.LOCAL_DEFINITIONS["local:flux.2-klein-4b"]

    def fake_run(command, *, cwd, environment=None, timeout=7200) -> str:
        values = [str(value) for value in command]
        commands.append(values)
        if values[1:] == ["--version"]:
            return "uv 0.6.9"
        if values[1] == "venv":
            python = setup._native_python(
                tmp_path / ".cache" / "runtimes" / setup.runner_slug(definition.backend_id)
            )
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_bytes(b"")
        return ""

    monkeypatch.setattr(setup, "_find_uv", lambda: "uv")
    monkeypatch.setattr(setup, "_run", fake_run)

    setup._install_native_environment(tmp_path, definition)

    compile_command = next(values for values in commands if values[1:3] == ["pip", "compile"])
    sync_command = next(values for values in commands if values[1:3] == ["pip", "sync"])
    assert compile_command[compile_command.index("--torch-backend") + 1] == "auto"
    assert sync_command[sync_command.index("--torch-backend") + 1] == "auto"


def test_parse_uv_version() -> None:
    assert setup._parse_uv_version("uv 0.6.9 (build metadata)") == (0, 6, 9)


def test_local_profile_uses_native_faster_whisper_and_keeps_parakeet_available() -> None:
    backend_id = "local:faster-whisper-large-v3-turbo"
    definition = setup.LOCAL_DEFINITIONS[backend_id]

    assert PROFILES["local"]["caption_alignment"] == backend_id
    assert definition.kind == "faster-whisper"
    assert definition.platform == "native"
    assert definition.python_version == "3.11"
    assert definition.model_repo == "dropbox-dash/faster-whisper-large-v3-turbo"
    assert definition.model_revision == "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"
    assert definition.requirements_name == "faster-whisper.in"
    assert BACKEND_DESCRIPTORS[backend_id].runner == "native"
    assert setup.LOCAL_DEFINITIONS["local:parakeet-tdt-0.6b-v3"].platform == "wsl"
    assert "local:parakeet-tdt-0.6b-v3" not in setup.selected_backends(
        profile="local", backend_id=None
    )


def test_curated_candidates_pin_the_planned_q4_mtp_artifacts() -> None:
    qwen = setup.CURATED_LLM_CANDIDATES["qwen3.6-27b-q4-mtp"]
    gemma = setup.CURATED_LLM_CANDIDATES["gemma-4-26b-a4b-q4-mtp"]

    assert qwen.quantization == gemma.quantization == "UD-Q4_K_XL"
    assert [artifact.role for artifact in qwen.artifacts] == ["model"]
    assert [artifact.role for artifact in gemma.artifacts] == ["model", "draft-model"]
    assert all(len(candidate.revision) == 40 for candidate in (qwen, gemma))
    assert all(
        len(artifact.sha256) == 64
        for candidate in (qwen, gemma)
        for artifact in candidate.artifacts
    )


def _fixture_candidate() -> setup.CuratedLlmCandidate:
    model_hash = sha256_file(Path(__file__))
    return setup.CuratedLlmCandidate(
        candidate_id="fixture-q4-mtp",
        model_id="fixture/model",
        repository="fixture/model-GGUF",
        revision="1" * 40,
        license_name="Apache-2.0",
        quantization="Q4_K_M",
        mtp="embedded",
        speculative_tokens=2,
        estimated_download_gb=1.0,
        artifacts=(
            setup.CuratedLlmArtifact(
                role="model",
                filename="model.gguf",
                sha256=model_hash,
            ),
        ),
    )


def test_curated_llm_download_pins_source_and_writes_manifest(tmp_path: Path, monkeypatch) -> None:
    candidate = _fixture_candidate()
    artifact_bytes = Path(__file__).read_bytes()
    commands: list[list[str]] = []

    def fake_run(command, *, cwd, environment=None, timeout=7200) -> str:
        values = [str(value) for value in command]
        commands.append(values)
        destination = Path(values[values.index("--local-dir") + 1])
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "model.gguf").write_bytes(artifact_bytes)
        assert "HF_TOKEN" not in environment
        assert "OPENAI_API_KEY" not in environment
        assert "AWS_SECRET_ACCESS_KEY" not in environment
        assert Path(environment["TEMP"]) == destination
        return ""

    monkeypatch.setitem(setup.CURATED_LLM_CANDIDATES, candidate.candidate_id, candidate)
    monkeypatch.setattr(setup, "_find_uv", lambda: "uv")
    monkeypatch.setattr(setup, "_run", fake_run)

    destination = setup.download_curated_llm_candidate(
        project_root=tmp_path,
        candidate_id=candidate.candidate_id,
        environment={
            "HF_TOKEN": "token-not-logged",
            "OPENAI_API_KEY": "cloud-secret",
            "AWS_SECRET_ACCESS_KEY": "other-secret",
        },
    )

    command = commands[0]
    assert command[command.index("--revision") + 1] == candidate.revision
    assert candidate.repository in command
    assert candidate.artifacts[0].filename in command
    manifest = json.loads((destination / "asset-manifest.json").read_text(encoding="utf-8"))
    assert manifest["revision"] == candidate.revision
    assert candidate.revision in manifest["model_card"]
    assert manifest["files"][0]["sha256"] == candidate.artifacts[0].sha256
    assert "token-not-logged" not in json.dumps(manifest)


def test_curated_llm_download_skips_verified_existing_file(tmp_path: Path, monkeypatch) -> None:
    candidate = _fixture_candidate()
    destination = tmp_path / ".cache" / "models" / "llm" / candidate.candidate_id
    destination.mkdir(parents=True)
    (destination / "model.gguf").write_bytes(Path(__file__).read_bytes())
    monkeypatch.setitem(setup.CURATED_LLM_CANDIDATES, candidate.candidate_id, candidate)
    monkeypatch.setattr(
        setup,
        "_run",
        lambda *args, **kwargs: pytest.fail("verified artifacts must not be downloaded again"),
    )

    setup.download_curated_llm_candidate(
        project_root=tmp_path,
        candidate_id=candidate.candidate_id,
        environment={},
    )

    assert (destination / "asset-manifest.json").is_file()


def test_curated_llm_download_rejects_a_hash_mismatch(tmp_path: Path, monkeypatch) -> None:
    base_candidate = _fixture_candidate()
    candidate = replace(
        base_candidate,
        artifacts=(
            *base_candidate.artifacts,
            setup.CuratedLlmArtifact(
                role="draft-model",
                filename="draft.gguf",
                sha256="f" * 64,
            ),
        ),
    )

    def fake_run(command, *, cwd, environment=None, timeout=7200) -> str:
        values = [str(value) for value in command]
        destination = Path(values[values.index("--local-dir") + 1])
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "model.gguf").write_bytes(Path(__file__).read_bytes())
        (destination / "draft.gguf").write_bytes(b"wrong")
        return ""

    monkeypatch.setitem(setup.CURATED_LLM_CANDIDATES, candidate.candidate_id, candidate)
    monkeypatch.setattr(setup, "_find_uv", lambda: "uv")
    monkeypatch.setattr(setup, "_run", fake_run)

    with pytest.raises(VideoGeneratorError, match="SHA-256 mismatch"):
        setup.download_curated_llm_candidate(
            project_root=tmp_path,
            candidate_id=candidate.candidate_id,
            environment={},
        )

    destination = tmp_path / ".cache" / "models" / "llm" / candidate.candidate_id
    assert not (destination / "model.gguf").exists()
    assert not (destination / "draft.gguf").exists()
    assert not (destination / "asset-manifest.json").exists()


@pytest.mark.parametrize(
    ("candidate", "error"),
    [
        pytest.param(
            replace(_fixture_candidate(), candidate_id="../outside"),
            "candidate ID",
            id="candidate-id",
        ),
        pytest.param(
            replace(
                _fixture_candidate(),
                artifacts=(
                    replace(_fixture_candidate().artifacts[0], filename="../model.gguf"),
                ),
            ),
            "filename",
            id="artifact-filename",
        ),
    ],
)
def test_curated_llm_download_rejects_path_traversal(
    tmp_path: Path,
    monkeypatch,
    candidate: setup.CuratedLlmCandidate,
    error: str,
) -> None:
    monkeypatch.setitem(setup.CURATED_LLM_CANDIDATES, candidate.candidate_id, candidate)

    with pytest.raises(VideoGeneratorError, match=error):
        setup.download_curated_llm_candidate(
            project_root=tmp_path,
            candidate_id=candidate.candidate_id,
            environment={},
        )

    assert not (tmp_path / "outside").exists()
