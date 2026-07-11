from __future__ import annotations

from pathlib import Path

from video_generator.contracts import CaptionTrack, OutputLanguage, WordTiming
from video_generator.media import write_ass
from video_generator.workflow import INTERNAL_REVISION


def test_ass_highlights_only_the_current_word(tmp_path: Path) -> None:
    assert INTERNAL_REVISION == "media-workflow-v2"
    track = CaptionTrack(
        language=OutputLanguage.FINNISH,
        words=[
            WordTiming(text="Tämä", start_seconds=0.0, end_seconds=0.4),
            WordTiming(text="on", start_seconds=0.5, end_seconds=0.7),
            WordTiming(text="testi{}.", start_seconds=0.8, end_seconds=1.2),
        ],
    )
    path = tmp_path / "captions.ass"

    write_ass(track, path, width=1280, height=720)

    content = path.read_text(encoding="utf-8")
    dialogues = [line for line in content.splitlines() if line.startswith("Dialogue:")]
    assert len(dialogues) == 4
    assert "\\k" not in content
    assert "Tämä on testi\\{\\}." in dialogues[0]
    assert "{\\1c&H0046C7FF&}Tämä{\\1c&H00FFFFFF&} on testi\\{\\}." in dialogues[1]
    assert "Tämä {\\1c&H0046C7FF&}on{\\1c&H00FFFFFF&} testi\\{\\}." in dialogues[2]
    assert "Tämä on {\\1c&H0046C7FF&}testi\\{\\}.{\\1c&H00FFFFFF&}" in dialogues[3]
    assert dialogues[1].startswith("Dialogue: 1,0:00:00.00,0:00:00.40")
    assert dialogues[2].startswith("Dialogue: 1,0:00:00.50,0:00:00.70")
