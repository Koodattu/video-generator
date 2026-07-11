from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from video_generator.contracts import RenderPlan, RenderScene
from video_generator.media import qc_video, render_video


def test_music_render_uses_ffmpeg_42_compatible_exact_sum_without_double_attenuation(
    tmp_path,
) -> None:
    commands = []

    class Tools:
        ffmpeg = "ffmpeg"

        @staticmethod
        def run(command, **kwargs) -> None:
            del kwargs
            commands.append(command)
            Path(command[-1]).write_bytes(b"fixture")

    plan = RenderPlan(
        scenes=[
            RenderScene(
                scene_id="scene-001",
                image_path="image.png",
                start_seconds=0,
                end_seconds=10,
            )
        ],
        narration_path="narration.wav",
        music_path="music.wav",
        width=1280,
        height=720,
        fps=30,
        duration_seconds=10,
    )

    render_video(
        Tools(),
        plan,
        workspace_root=tmp_path,
        base_path=tmp_path / "base.mp4",
        output_path=tmp_path / "video.mp4",
    )

    graph = commands[0][commands[0].index("-filter_complex") + 1]
    assert "amix=inputs=2:duration=first,volume=2" in graph
    assert "volume=0.16" not in graph
    assert "normalize=" not in graph


@pytest.mark.parametrize(
    ("encoded_duration", "expected_passed"),
    [(120.022, True), (120.040, False)],
)
def test_duration_hard_limit_allows_only_one_frame_of_mux_rounding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    encoded_duration: float,
    expected_passed: bool,
) -> None:
    tools = SimpleNamespace(
        ffmpeg="ffmpeg",
        probe_json=lambda path: {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "pix_fmt": "yuv420p",
                    "width": 1280,
                    "height": 720,
                    "avg_frame_rate": "30/1",
                },
                {"codec_type": "audio", "codec_name": "aac"},
                {"codec_type": "subtitle", "codec_name": "mov_text"},
            ],
            "format": {"duration": str(encoded_duration)},
        },
    )
    monkeypatch.setattr(
        "video_generator.media.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            stderr="mean_volume: -18.0 dB\nmax_volume: -1.0 dB"
        ),
    )

    checks = qc_video(
        tools,
        tmp_path / "video.mp4",
        width=1280,
        height=720,
        fps=30,
        expected_duration=120.0,
        budget=120.0,
        captions_expected=True,
    )

    hard_limit = next(check for check in checks if check.name == "duration_hard_limit")
    assert hard_limit.passed is expected_passed
