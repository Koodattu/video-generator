from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace

from video_generator.backends.elevenlabs import ElevenLabsSpeechBackend
from video_generator.contracts import OutputLanguage, SpeechRequest, VoiceSettings
from video_generator.net import HttpResponse


class StubHttpClient:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.requests = []

    def request(self, method: str, url: str, **kwargs) -> HttpResponse:
        self.requests.append({"method": method, "url": url, **kwargs})
        return HttpResponse(
            status=200,
            headers={"request-id": "request-tts-1", "character-cost": "12"},
            body=json.dumps(self.payload).encode("utf-8"),
        )


def test_elevenlabs_tts_uses_finnish_voice_and_character_timing(tmp_path: Path) -> None:
    text = "Hei maailma!"
    starts = [index * 0.05 for index in range(len(text))]
    ends = [(index + 1) * 0.05 for index in range(len(text))]
    http = StubHttpClient(
        {
            "audio_base64": base64.b64encode(b"fake-mp3").decode("ascii"),
            "alignment": {
                "characters": list(text),
                "character_start_times_seconds": starts,
                "character_end_times_seconds": ends,
            },
        }
    )
    backend = ElevenLabsSpeechBackend(
        "test-key",
        workspace_root=tmp_path,
        run_root=tmp_path,
        configured_voice_id="voice/id",
        http=http,
    )
    backend.media = SimpleNamespace(
        probe_audio=lambda path: SimpleNamespace(duration_seconds=1.0, sample_rate=44100, channels=1)
    )

    result = backend.synthesize(
        SpeechRequest(
            scene_id="scene-001",
            text=text,
            output_language=OutputLanguage.FINNISH,
            voice=VoiceSettings(
                name="my_voice",
                authorization="self",
                elevenlabs_voice_id="voice/id",
            ),
            output_path="speech.mp3",
        )
    )

    request = http.requests[0]
    assert "/text-to-speech/voice%2Fid/with-timestamps" in request["url"]
    assert request["json_body"]["language_code"] == "fi"
    assert request["json_body"]["model_id"] == "eleven_multilingual_v2"
    assert [word.text for word in result.asset.word_timings] == ["Hei", "maailma!"]
    assert result.asset.provider_request_id == "request-tts-1"
    assert result.usage.input_units == 12
