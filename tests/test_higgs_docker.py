from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path
from types import SimpleNamespace

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
    assert any("target=/models/higgs-tts-3-4b,readonly" in value for value in command)


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
    worker = object.__new__(HiggsDockerWorker)
    worker.paths = paths
    worker.container_name = "video-generator-local--higgs-tts-3-4b-fixture"
    worker.port = 8000
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
    assert result["sample_rate"] == 24000
    assert result["channels"] == 1
    assert result["duration_seconds"] == 0.1
