from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from video_generator.util import atomic_write_bytes
from video_generator.workers.higgs_docker import HiggsDockerWorker, compile_higgs_text


def test_higgs_control_compiler_preserves_canonical_text() -> None:
    text = "Tämä teksti kuuluu puhua täsmälleen näin."

    compiled, tokens = compile_higgs_text(text, {"pace": "fast"})

    assert tokens == ["<|prosody:speed_fast|>"]
    assert compiled == tokens[0] + text
    assert compile_higgs_text(text, {"pace": "standard"}) == (text, [])


def test_higgs_container_command_is_local_offline_and_nonprivileged(tmp_path: Path) -> None:
    worker = object.__new__(HiggsDockerWorker)
    worker.container_name = "video-generator-local--higgs-tts-3-4b-fixture"
    worker.image_reference = "lmsysorg/sglang-omni@sha256:" + "2" * 64
    worker.port = 8000
    worker.model_path = tmp_path / "models" / "higgs"
    worker.voice_scratch = tmp_path / "runtimes" / "higgs" / "voice-scratch"
    worker.voice_scratch.mkdir(parents=True)
    worker.paths = SimpleNamespace(run_root=tmp_path / "runs" / "fixture")
    calls: list[list[str]] = []
    worker._run = lambda arguments, **kwargs: calls.append(arguments) or SimpleNamespace(
        returncode=0, stdout="container-id\n", stderr=""
    )

    worker._start_container()

    command = calls[0]
    assert command[:2] == ["run", "--detach"]
    assert ["--network", "none"] == command[command.index("--network") : command.index("--network") + 2]
    assert "--privileged" not in command
    assert "--ipc" not in command
    assert "/var/run/docker.sock" not in " ".join(command)
    assert "no-new-privileges" in command
    assert "/tmp:rw,exec,nosuid,size=8g" in command
    assert not any(value.startswith("/voices:") for value in command)
    assert any("target=/models/higgs-tts-3-4b,readonly" in value for value in command)
    assert any(
        f"source={worker.voice_scratch},target=/voices,readonly" in value
        for value in command
    )


def test_higgs_synthesis_uses_exact_reference_and_returns_valid_wav(tmp_path: Path) -> None:
    private = tmp_path / "private" / "voice"
    run_root = tmp_path / "runs" / "fixture"
    private.mkdir(parents=True)
    run_root.mkdir(parents=True)
    reference = private / "reference.wav"
    reference.write_bytes(b"reference")
    transcript = private / "reference.txt"
    transcript.write_text("Exact reference transcript.", encoding="utf-8")

    class Paths:
        @staticmethod
        def read_private(value: str) -> Path:
            path = (tmp_path / value).resolve()
            path.relative_to(tmp_path / "private")
            return path

        @staticmethod
        def output_run(value: str) -> Path:
            path = (tmp_path / value).resolve()
            path.relative_to(run_root)
            path.parent.mkdir(parents=True, exist_ok=True)
            return path

    paths = Paths()
    paths.project_root = tmp_path
    paths.run_root = run_root
    paths.runtime_root = tmp_path / ".cache" / "runtimes"
    worker = object.__new__(HiggsDockerWorker)
    worker.paths = paths
    worker.container_name = "video-generator-local--higgs-tts-3-4b-fixture"
    worker.port = 8000
    worker.voice_scratch = (
        paths.runtime_root
        / "local--higgs-tts-3-4b"
        / "voice-scratch"
        / worker.container_name
    )
    worker.voice_scratch.mkdir(parents=True)
    requests: list[dict[str, object]] = []

    def fake_run(arguments, *, input_text=None, **kwargs):
        if arguments[0] == "exec":
            request = json.loads(input_text)
            requests.append(request)
            output = run_root / Path(str(request["_output_path"])).relative_to("/run")
            output.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(output), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(24000)
                handle.writeframes(b"\0\0" * 2400)
            return subprocess.CompletedProcess(
                arguments,
                0,
                stdout=json.dumps({"status": 200, "content_type": "audio/wav"}),
                stderr="",
            )
        return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")

    worker._run = fake_run
    result = worker.dispatch(
        "speech.synthesize",
        {
            "scene_id": "scene-001",
            "text": "This is the canonical narration.",
            "output_language": "en",
            "voice": {
                "reference_audio": "private/voice/reference.wav",
                "reference_transcript": "private/voice/reference.txt",
            },
            "delivery": {"pace": "fast"},
            "output_path": "runs/fixture/work/speech.wav",
        },
    )

    assert requests[0]["input"] == "<|prosody:speed_fast|>This is the canonical narration."
    assert requests[0]["references"][0]["text"] == "Exact reference transcript."
    staged = list(worker.voice_scratch.iterdir())
    assert len(staged) == 1
    assert staged[0].read_bytes() == b"reference"
    assert requests[0]["references"][0]["audio_path"] == f"/voices/{staged[0].name}"
    assert result["sample_rate"] == 24000
    assert result["channels"] == 1
    assert result["duration_seconds"] == 0.1


def test_higgs_reference_staging_replaces_only_the_managed_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    worker = object.__new__(HiggsDockerWorker)
    worker.container_name = "video-generator-local--higgs-tts-3-4b-fixture"
    worker.voice_scratch = tmp_path / "voice-scratch"
    worker.voice_scratch.mkdir()
    first = tmp_path / "first.wav"
    second = tmp_path / "second.flac"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    staging_paths: list[Path] = []
    real_atomic_write = atomic_write_bytes

    def record_atomic_write(path: Path, value: bytes) -> None:
        staging_paths.append(path)
        assert path.parent == worker.voice_scratch.parent
        assert path.parent != worker.voice_scratch
        real_atomic_write(path, value)

    monkeypatch.setattr(
        "video_generator.workers.higgs_docker.atomic_write_bytes",
        record_atomic_write,
    )

    first_path = worker._stage_reference(first)
    second_path = worker._stage_reference(second)

    assert first_path != second_path
    entries = list(worker.voice_scratch.iterdir())
    assert len(entries) == 1
    assert entries[0].read_bytes() == b"second"
    assert second_path == f"/voices/{entries[0].name}"
    assert len(staging_paths) == 2


def test_higgs_reference_staging_rejects_non_file(tmp_path: Path) -> None:
    worker = object.__new__(HiggsDockerWorker)
    worker.voice_scratch = tmp_path / "voice-scratch"
    worker.voice_scratch.mkdir()
    reference_directory = tmp_path / "reference.wav"
    reference_directory.mkdir()

    with pytest.raises(ValueError, match="regular file"):
        worker._stage_reference(reference_directory)

    empty_reference = tmp_path / "empty.wav"
    empty_reference.touch()
    with pytest.raises(ValueError, match="is empty"):
        worker._stage_reference(empty_reference)
