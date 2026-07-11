from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from video_generator.workers.main import FasterWhisperWorker


@pytest.mark.parametrize("language", ["en", "fi"])
def test_faster_whisper_worker_uses_pinned_local_cuda_model(
    tmp_path: Path,
    monkeypatch,
    language: str,
) -> None:
    model_path = tmp_path / ".cache" / "models" / "faster-whisper-large-v3-turbo"
    audio_path = tmp_path / "runs" / "fixture" / "scene.wav"
    model_path.mkdir(parents=True)
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"fixture")
    calls: dict[str, object] = {}

    class FakeWhisperModel:
        def __init__(self, path: str, **kwargs) -> None:
            calls["model_path"] = path
            calls["model_options"] = kwargs

        def transcribe(self, path: str, **kwargs):
            calls["audio_path"] = path
            calls["transcribe_options"] = kwargs

            def segments():
                calls["segments_consumed"] = True
                yield SimpleNamespace(
                    text=" Hello maailma.",
                    words=[
                        SimpleNamespace(word=" Hello", start=0.1, end=0.5, probability=0.95),
                        SimpleNamespace(word=" maailma.", start=0.5, end=1.1, probability=0.9),
                    ],
                )

            return segments(), SimpleNamespace()

    faster_whisper = ModuleType("faster_whisper")
    faster_whisper.__version__ = "1.2.1"
    faster_whisper.WhisperModel = FakeWhisperModel
    ctranslate2 = ModuleType("ctranslate2")
    ctranslate2.__version__ = "4.8.1"
    ctranslate2.get_supported_compute_types = lambda device: {"float16", "int8_float16"}
    monkeypatch.setitem(sys.modules, "faster_whisper", faster_whisper)
    monkeypatch.setitem(sys.modules, "ctranslate2", ctranslate2)
    monkeypatch.setenv("VIDEO_GENERATOR_MODEL_PATH", ".cache/models/faster-whisper-large-v3-turbo")

    paths = SimpleNamespace(
        read_model=lambda value: model_path,
        read_run=lambda value: audio_path,
    )
    worker = FasterWhisperWorker(paths)
    result = worker.dispatch(
        "alignment.align",
        {
            "audio_path": "runs/fixture/scene.wav",
            "output_language": language,
            "transcript": "Canonical text must not be injected into recognition.",
        },
    )

    assert calls["model_path"] == str(model_path)
    assert calls["model_options"] == {
        "device": "cuda",
        "compute_type": "float16",
        "local_files_only": True,
    }
    options = calls["transcribe_options"]
    assert options["language"] == language
    assert options["task"] == "transcribe"
    assert options["beam_size"] == 5
    assert options["word_timestamps"] is True
    assert options["condition_on_previous_text"] is False
    assert options["vad_filter"] is False
    assert "initial_prompt" not in options
    assert "hotwords" not in options
    assert calls["segments_consumed"] is True
    assert result["recognized_words"] == [
        {
            "text": "Hello",
            "start_seconds": 0.1,
            "end_seconds": 0.5,
            "confidence": 0.95,
        },
        {
            "text": "maailma.",
            "start_seconds": 0.5,
            "end_seconds": 1.1,
            "confidence": 0.9,
        },
    ]
    assert worker.health()["supported_compute_types"] == ["float16", "int8_float16"]
