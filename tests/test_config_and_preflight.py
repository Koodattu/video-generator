from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from video_generator.config import resolve_config
from video_generator.contracts import Quality
from video_generator.errors import ConfigurationError
from video_generator.preflight import _voice_checks, estimate_cost


def test_custom_style_id_is_supported(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "config.example.toml"
    content = source.read_text(encoding="utf-8").replace(
        'style = "ms_paint_stick"',
        'style = "soft_watercolor"',
    ).replace(
        'style_description = ""',
        'style_description = "Loose paper texture and muted blue-orange washes"',
    )
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\nversion='0.0.0'\n", encoding="utf-8")
    (tmp_path / "config.toml").write_text(content, encoding="utf-8")

    config = resolve_config(tmp_path / "config.toml")

    assert config.style == "soft_watercolor"


def test_elevenlabs_voice_id_falls_back_to_environment(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "config.example.toml"
    content = source.read_text(encoding="utf-8").replace(
        'profile = "local"',
        'profile = "cloud-openai"',
    )
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\nversion='0.0.0'\n", encoding="utf-8")
    (tmp_path / "config.toml").write_text(content, encoding="utf-8")

    config = resolve_config(
        tmp_path / "config.toml",
        environment={"ELEVENLABS_VOICE_ID": "voice-from-environment"},
    )

    assert config.voice.elevenlabs_voice_id == "voice-from-environment"


def test_elevenlabs_voice_id_in_toml_takes_precedence(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "config.example.toml"
    content = source.read_text(encoding="utf-8").replace(
        'profile = "local"',
        'profile = "cloud-openai"',
    ).replace(
        'elevenlabs_voice_id = ""',
        'elevenlabs_voice_id = "voice-from-toml"',
    )
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\nversion='0.0.0'\n", encoding="utf-8")
    (tmp_path / "config.toml").write_text(content, encoding="utf-8")

    config = resolve_config(
        tmp_path / "config.toml",
        environment={"ELEVENLABS_VOICE_ID": "voice-from-environment"},
    )

    assert config.voice.elevenlabs_voice_id == "voice-from-toml"


def test_offline_rejects_active_cloud_override(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "config.example.toml"
    content = source.read_text(encoding="utf-8").replace("offline = false", "offline = true")
    content += '\nresearch = "openai:gpt-5.5"\n'
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\nversion='0.0.0'\n", encoding="utf-8")
    (tmp_path / "config.toml").write_text(content, encoding="utf-8")

    with pytest.raises(ConfigurationError, match="offline Run selects cloud Backends"):
        resolve_config(tmp_path / "config.toml")


def test_cost_estimate_can_start_at_a_rerun_stage(resolved_config) -> None:
    full = estimate_cost(resolved_config)
    remaining = estimate_cost(resolved_config, from_stage="images")

    assert remaining.estimated_usd <= full.estimated_usd
    assert remaining.basis.startswith("planned first attempts from images")


def test_cost_estimate_uses_cloud_generation_size_and_continuity_references(
    resolved_config,
) -> None:
    bindings = dict(resolved_config.task_bindings)
    bindings["image_generate"] = "openai:gpt-image-2"
    config = resolved_config.model_copy(
        update={
            "quality": Quality.FINAL,
            "delivery_width": 4096,
            "delivery_height": 2304,
            "task_bindings": bindings,
        }
    )

    estimate = estimate_cost(config, from_stage="images")
    expected = 0.30 * 2.0 * 1.75 * estimate.scene_count * 2

    assert estimate.line_items["images"] == pytest.approx(expected)


@pytest.mark.skipif(
    shutil.which("ffprobe") is None,
    reason="voice-reference validation requires ffprobe",
)
def test_local_voice_preflight_rejects_invalid_audio(tmp_path: Path, resolved_config) -> None:
    reference = tmp_path / "private" / "voice" / "me.wav"
    reference.parent.mkdir(parents=True)
    reference.write_bytes(b"not a wave file")
    transcript = reference.with_suffix(".txt")
    transcript.write_text("Exact reference transcript.", encoding="utf-8")
    config = resolved_config.model_copy(
        update={
            "project_root": str(tmp_path),
            "voice": resolved_config.voice.model_copy(
                update={
                    "reference_audio": "private/voice/me.wav",
                    "reference_transcript": "private/voice/me.txt",
                }
            ),
        }
    )

    checks = _voice_checks(config, tmp_path)

    media_check = next(item for item in checks if item.name == "voice_reference_audio_media")
    assert media_check.ready is False


def test_local_voice_preflight_rejects_audio_outside_private(
    tmp_path: Path,
    resolved_config,
) -> None:
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"fixture")
    config = resolved_config.model_copy(
        update={
            "project_root": str(tmp_path),
            "voice": resolved_config.voice.model_copy(
                update={"reference_audio": "private/../outside.wav"}
            ),
        }
    )

    checks = _voice_checks(config, tmp_path)

    audio = next(item for item in checks if item.name == "voice_reference_audio")
    assert audio.ready is False
    assert "outside private/" in audio.detail


def test_local_voice_preflight_rejects_transcript_outside_private(
    tmp_path: Path,
    resolved_config,
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("Exact transcript.", encoding="utf-8")
    config = resolved_config.model_copy(
        update={
            "project_root": str(tmp_path),
            "voice": resolved_config.voice.model_copy(
                update={"reference_transcript": "private/../outside.txt"}
            ),
        }
    )

    checks = _voice_checks(config, tmp_path)

    transcript = next(
        item for item in checks if item.name == "voice_reference_transcript"
    )
    assert transcript.ready is False
    assert "outside private/" in transcript.detail


@pytest.mark.parametrize("backend_id", ["local:omnivoice", "local:higgs-tts-3-4b"])
def test_transcript_required_tts_preflight_requires_reference_transcript(
    tmp_path: Path,
    resolved_config,
    backend_id: str,
) -> None:
    bindings = dict(resolved_config.task_bindings)
    bindings["narration_synthesis"] = backend_id
    config = resolved_config.model_copy(
        update={
            "project_root": str(tmp_path),
            "task_bindings": bindings,
            "voice": resolved_config.voice.model_copy(update={"reference_transcript": ""}),
        }
    )

    check = next(
        item for item in _voice_checks(config, tmp_path)
        if item.name == "voice_reference_transcript"
    )

    assert check.ready is False
    assert check.detail == "not configured"
    assert "Set voice.reference_transcript" in (check.action or "")


@pytest.mark.parametrize("backend_id", ["local:omnivoice", "local:higgs-tts-3-4b"])
def test_transcript_required_tts_preflight_rejects_empty_reference_transcript(
    tmp_path: Path,
    resolved_config,
    backend_id: str,
) -> None:
    transcript = tmp_path / "private" / "voice" / "me.txt"
    transcript.parent.mkdir(parents=True)
    transcript.write_text("  \n", encoding="utf-8")
    bindings = dict(resolved_config.task_bindings)
    bindings["narration_synthesis"] = backend_id
    config = resolved_config.model_copy(
        update={
            "project_root": str(tmp_path),
            "task_bindings": bindings,
            "voice": resolved_config.voice.model_copy(
                update={"reference_transcript": "private/voice/me.txt"}
            ),
        }
    )

    check = next(
        item for item in _voice_checks(config, tmp_path)
        if item.name == "voice_reference_transcript"
    )

    assert check.ready is False
    assert check.detail.startswith("empty:")


def test_xvoice_preflight_records_the_explicit_reference_language(
    tmp_path: Path,
    resolved_config,
) -> None:
    bindings = dict(resolved_config.task_bindings)
    bindings["narration_synthesis"] = "local:x-voice"
    config = resolved_config.model_copy(
        update={
            "project_root": str(tmp_path),
            "task_bindings": bindings,
            "voice": resolved_config.voice.model_copy(update={"reference_language": "en"}),
        }
    )

    check = next(
        item for item in _voice_checks(config, tmp_path)
        if item.name == "voice_reference_language"
    )

    assert check.ready is True
    assert check.detail == "reference language: en"


@pytest.mark.parametrize("backend_id", ["local:voxcpm2", "local:moss-tts-v1.5"])
def test_optional_transcript_backends_accept_an_unconfigured_transcript(
    tmp_path: Path,
    resolved_config,
    backend_id: str,
) -> None:
    bindings = dict(resolved_config.task_bindings)
    bindings["narration_synthesis"] = backend_id
    config = resolved_config.model_copy(
        update={
            "project_root": str(tmp_path),
            "task_bindings": bindings,
            "voice": resolved_config.voice.model_copy(update={"reference_transcript": ""}),
        }
    )

    checks = _voice_checks(config, tmp_path)

    assert all(item.name != "voice_reference_transcript" for item in checks)
