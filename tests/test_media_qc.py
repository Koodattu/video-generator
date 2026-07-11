from __future__ import annotations

from types import SimpleNamespace

import pytest

from video_generator.media import qc_video


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
