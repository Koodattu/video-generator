from __future__ import annotations

import difflib
import json
import math
import re
import shutil
import subprocess
import unicodedata
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

from .contracts import (
    CaptionTrack,
    DeliveryFile,
    DeliveryManifest,
    MediaReference,
    NarrationScript,
    NarrationTimeline,
    OutputLanguage,
    QCCheck,
    RenderPlan,
    SpeechAsset,
    TimelineScene,
    WordTiming,
)
from .errors import ErrorKind, MediaError
from .util import atomic_write_text, relative_path, replace_path, sha256_file


@dataclass(frozen=True)
class AudioProbe:
    duration_seconds: float
    sample_rate: int
    channels: int
    codec: str


@dataclass(frozen=True)
class MediaTools:
    ffmpeg: str
    ffprobe: str

    @classmethod
    def discover(cls) -> "MediaTools":
        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        if not ffmpeg or not ffprobe:
            raise MediaError(
                "FFmpeg and ffprobe must be available on PATH",
                kind=ErrorKind.NOT_READY,
                action="Install a current FFmpeg build with libx264, AAC, mov_text, SRT, and libass.",
            )
        return cls(ffmpeg=ffmpeg, ffprobe=ffprobe)

    def run(self, arguments: Sequence[str], *, cwd: Path | None = None, timeout: float = 900) -> subprocess.CompletedProcess[str]:
        try:
            completed = subprocess.run(
                list(arguments),
                cwd=cwd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise MediaError(f"media command failed to start or timed out: {exc}", kind=ErrorKind.INTERNAL) from exc
        if completed.returncode != 0:
            detail = "\n".join(completed.stderr.splitlines()[-30:])
            raise MediaError(
                f"media command failed ({Path(arguments[0]).name} exit {completed.returncode}):\n{detail}",
                kind=ErrorKind.INVALID_OUTPUT,
            )
        return completed

    def probe_json(self, path: Path) -> dict[str, Any]:
        completed = self.run(
            [
                self.ffprobe,
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ],
            timeout=60,
        )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise MediaError(f"ffprobe returned invalid JSON for {path}", kind=ErrorKind.INVALID_OUTPUT) from exc
        if not isinstance(payload, dict):
            raise MediaError(f"ffprobe returned invalid output for {path}", kind=ErrorKind.INVALID_OUTPUT)
        return payload

    def probe_audio(self, path: Path) -> AudioProbe:
        payload = self.probe_json(path)
        stream = next(
            (item for item in payload.get("streams", []) if item.get("codec_type") == "audio"), None
        )
        if not isinstance(stream, dict):
            raise MediaError(f"audio stream is missing: {path}", kind=ErrorKind.INVALID_OUTPUT)
        raw_duration = stream.get("duration") or payload.get("format", {}).get("duration")
        try:
            duration = float(raw_duration)
            sample_rate = int(stream["sample_rate"])
            channels = int(stream["channels"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MediaError(f"audio properties are invalid: {path}", kind=ErrorKind.INVALID_OUTPUT) from exc
        if duration <= 0 or sample_rate <= 0 or channels <= 0:
            raise MediaError(f"audio properties are non-positive: {path}", kind=ErrorKind.INVALID_OUTPUT)
        return AudioProbe(duration, sample_rate, channels, str(stream.get("codec_name") or ""))

    def capability_checks(self, *, animated_captions: bool) -> list[QCCheck]:
        encoders = self.run([self.ffmpeg, "-hide_banner", "-encoders"], timeout=60).stdout
        filters = self.run([self.ffmpeg, "-hide_banner", "-filters"], timeout=60).stdout
        checks = [
            QCCheck(name="ffmpeg", passed=True, detail=self.ffmpeg),
            QCCheck(name="ffprobe", passed=True, detail=self.ffprobe),
            QCCheck(name="h264_encoder", passed="libx264" in encoders, detail="libx264 encoder"),
            QCCheck(name="aac_encoder", passed=re.search(r"\bAAC\b|\baac\b", encoders) is not None, detail="AAC encoder"),
            QCCheck(name="subtitle_filter", passed=(" ass " in filters or " subtitles " in filters), detail="ASS/libass filter"),
        ]
        if not animated_captions:
            checks[-1] = QCCheck(name="subtitle_filter", passed=True, detail="not required")
        return checks


def delivery_ceiling(duration_budget: float, fps: int) -> float:
    return math.floor(duration_budget * fps + 1e-9) / fps


def delivery_duration(timeline_duration: float, fps: int) -> float:
    return math.ceil(timeline_duration * fps - 1e-9) / fps


def normalize_audio(tools: MediaTools, source: Path, destination: Path) -> AudioProbe:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tools.run(
        [
            tools.ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            "-vn",
            "-af",
            "highpass=f=55,loudnorm=I=-18:LRA=7:TP=-1.5",
            "-ar",
            "48000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]
    )
    return tools.probe_audio(destination)


def concatenate_audio(
    tools: MediaTools,
    clips: Sequence[Path],
    durations: Sequence[float],
    pauses: Sequence[float],
    destination: Path,
) -> AudioProbe:
    if not clips or not (len(clips) == len(durations) == len(pauses)):
        raise ValueError("clips, durations, and pauses must be nonempty and have equal lengths")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [tools.ffmpeg, "-y", "-v", "error"]
    for path in clips:
        command.extend(["-i", str(path)])
    filters = []
    labels = []
    for index, (duration, pause) in enumerate(zip(durations, pauses, strict=True)):
        label = f"a{index}"
        chain = f"[{index}:a]atrim=start=0:end={duration:.6f},asetpts=PTS-STARTPTS"
        if pause > 0:
            chain += f",apad,atrim=duration={duration + pause:.6f}"
        filters.append(f"{chain}[{label}]")
        labels.append(f"[{label}]")
    filters.append("".join(labels) + f"concat=n={len(labels)}:v=0:a=1[outa]")
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[outa]",
            "-ar",
            "48000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]
    )
    tools.run(command)
    return tools.probe_audio(destination)


def build_timeline(
    *,
    script: NarrationScript,
    source_assets: Sequence[SpeechAsset],
    normalized_paths: Sequence[Path],
    normalized_probes: Sequence[AudioProbe],
    narration_path: Path,
    narration_probe: AudioProbe,
    workspace_root: Path,
    fps: int,
) -> NarrationTimeline:
    if not (
        len(script.scenes)
        == len(source_assets)
        == len(normalized_paths)
        == len(normalized_probes)
    ):
        raise ValueError("script and audio asset counts differ")
    scenes = []
    cursor = 0.0
    for script_scene, source, normalized_path, probe in zip(
        script.scenes, source_assets, normalized_paths, normalized_probes, strict=True
    ):
        if script_scene.scene_id != source.scene_id:
            raise MediaError("Scene audio order does not match the Narration Script", kind=ErrorKind.INVALID_OUTPUT)
        scale = probe.duration_seconds / source.duration_seconds
        words = [
            WordTiming(
                text=word.text,
                start_seconds=word.start_seconds * scale,
                end_seconds=min(probe.duration_seconds, word.end_seconds * scale),
                confidence=word.confidence,
            )
            for word in source.word_timings
        ]
        speech_end = cursor + probe.duration_seconds
        end = speech_end + script_scene.pause_after_seconds
        scenes.append(
            TimelineScene(
                scene_id=script_scene.scene_id,
                audio=MediaReference(
                    path=relative_path(normalized_path, workspace_root),
                    sha256=sha256_file(normalized_path),
                    mime_type="audio/wav",
                ),
                start_seconds=cursor,
                speech_end_seconds=speech_end,
                end_seconds=end,
                words=words,
            )
        )
        cursor = end
    if abs(narration_probe.duration_seconds - cursor) > 0.03:
        raise MediaError(
            f"concatenated narration duration drifted by {narration_probe.duration_seconds - cursor:+.3f}s",
            kind=ErrorKind.INVALID_OUTPUT,
        )
    cursor = narration_probe.duration_seconds
    if scenes:
        last = scenes[-1]
        final_relative_end = max(0.0, cursor - last.start_seconds)
        final_words = [
            word.model_copy(
                update={
                    "start_seconds": min(word.start_seconds, final_relative_end),
                    "end_seconds": min(word.end_seconds, final_relative_end),
                }
            )
            for word in last.words
        ]
        scenes[-1] = last.model_copy(
            update={
                "speech_end_seconds": min(last.speech_end_seconds, cursor),
                "end_seconds": cursor,
                "words": final_words,
            }
        )
    return NarrationTimeline(
        narration_audio=MediaReference(
            path=relative_path(narration_path, workspace_root),
            sha256=sha256_file(narration_path),
            mime_type="audio/wav",
        ),
        duration_seconds=cursor,
        delivery_duration_seconds=delivery_duration(cursor, fps),
        fps=fps,
        scenes=scenes,
    )


def duration_is_accepted(timeline: NarrationTimeline, budget: float) -> bool:
    return (
        timeline.duration_seconds + 1e-6 >= budget * 0.9
        and timeline.delivery_duration_seconds <= delivery_ceiling(budget, timeline.fps) + 1e-6
    )


def _normalize_token(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in value if unicodedata.category(character)[0] in {"L", "N"})


def reconcile_word_timings(
    canonical_text: str,
    recognized_words: Sequence[WordTiming],
    *,
    scene_duration: float,
    minimum_coverage: float = 0.88,
) -> tuple[list[WordTiming], float]:
    canonical = re.findall(r"\S+", canonical_text, flags=re.UNICODE)
    if not canonical:
        return [], 1.0
    recognized = [word for word in recognized_words if _normalize_token(word.text)]
    matcher = difflib.SequenceMatcher(
        a=[_normalize_token(word) for word in canonical],
        b=[_normalize_token(word.text) for word in recognized],
        autojunk=False,
    )
    mapped: dict[int, WordTiming] = {}
    matched = 0
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            canonical_index = block.a + offset
            recognized_word = recognized[block.b + offset]
            mapped[canonical_index] = recognized_word
            matched += 1
    coverage = matched / len(canonical)
    if coverage + 1e-9 < minimum_coverage:
        raise MediaError(
            f"caption alignment coverage {coverage:.1%} is below the required {minimum_coverage:.1%}",
            kind=ErrorKind.INVALID_OUTPUT,
            action="Inspect the Scene pronunciation/reference audio, then explicitly rerun from captions or narration.",
        )
    output: list[WordTiming | None] = [None] * len(canonical)
    for index, timing in mapped.items():
        output[index] = WordTiming(
            text=canonical[index],
            start_seconds=max(0.0, min(scene_duration, timing.start_seconds)),
            end_seconds=max(0.0, min(scene_duration, timing.end_seconds)),
            confidence=timing.confidence,
        )
    index = 0
    while index < len(output):
        if output[index] is not None:
            index += 1
            continue
        gap_start = index
        while index < len(output) and output[index] is None:
            index += 1
        gap_end = index
        left = output[gap_start - 1].end_seconds if gap_start else 0.0  # type: ignore[union-attr]
        right = output[gap_end].start_seconds if gap_end < len(output) else scene_duration  # type: ignore[union-attr]
        right = max(left, right)
        slot = (right - left) / max(1, gap_end - gap_start)
        for gap_index in range(gap_start, gap_end):
            offset = gap_index - gap_start
            start = left + slot * offset
            end = left + slot * (offset + 0.9)
            output[gap_index] = WordTiming(
                text=canonical[gap_index],
                start_seconds=start,
                end_seconds=max(start, min(right, end)),
                confidence=0.0,
            )
    result = [word for word in output if word is not None]
    previous = 0.0
    for word in result:
        if word.start_seconds < previous:
            word.start_seconds = previous
        if word.end_seconds < word.start_seconds:
            word.end_seconds = word.start_seconds
        previous = word.end_seconds
    return result, coverage


def caption_track_from_timeline(
    timeline: NarrationTimeline,
    script: NarrationScript,
    *,
    scene_words: dict[str, Sequence[WordTiming]] | None = None,
    coverage_by_scene: dict[str, float] | None = None,
    language: OutputLanguage,
) -> CaptionTrack:
    scene_words = scene_words or {}
    coverage_by_scene = coverage_by_scene or {}
    all_words = []
    coverage_values = []
    script_by_id = {scene.scene_id: scene for scene in script.scenes}
    for scene in timeline.scenes:
        words = list(scene_words.get(scene.scene_id, scene.words))
        canonical = re.findall(r"\S+", script_by_id[scene.scene_id].spoken_text, flags=re.UNICODE)
        if [word.text for word in words] != canonical:
            raise MediaError(
                f"caption text for {scene.scene_id} differs from the canonical Narration Script",
                kind=ErrorKind.INVALID_OUTPUT,
            )
        for word in words:
            all_words.append(
                word.model_copy(
                    update={
                        "start_seconds": scene.start_seconds + word.start_seconds,
                        "end_seconds": scene.start_seconds + word.end_seconds,
                    }
                )
            )
        coverage_values.append(coverage_by_scene.get(scene.scene_id, 1.0))
    coverage = min(coverage_values) if coverage_values else 1.0
    return CaptionTrack(language=language, words=all_words, reconciliation_coverage=coverage)


def _caption_cues(words: Sequence[WordTiming]) -> list[list[WordTiming]]:
    cues: list[list[WordTiming]] = []
    current: list[WordTiming] = []
    for word in words:
        if current and (len(current) >= 7 or word.end_seconds - current[0].start_seconds > 3.8):
            cues.append(current)
            current = []
        current.append(word)
        if re.search(r"[.!?…][\"'”’)]*$", word.text) and len(current) >= 3:
            cues.append(current)
            current = []
    if current:
        cues.append(current)
    return cues


def _srt_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


def _ass_timestamp(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6000)
    whole_seconds, centiseconds = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{centiseconds:02d}"


def write_srt(track: CaptionTrack, path: Path) -> None:
    blocks = []
    for index, cue in enumerate(_caption_cues(track.words), start=1):
        blocks.append(
            f"{index}\n{_srt_timestamp(cue[0].start_seconds)} --> {_srt_timestamp(cue[-1].end_seconds)}\n"
            + " ".join(word.text for word in cue)
        )
    atomic_write_text(path, "\n\n".join(blocks) + ("\n" if blocks else ""))


def _ass_escape(value: str) -> str:
    return value.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}").replace("\n", r"\N")


def write_ass(track: CaptionTrack, path: Path, *, width: int, height: int) -> None:
    font_size = round(height * 0.058)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,{font_size},&H00FFFFFF,&H0046C7FF,&H00141414,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,2,80,80,{round(height * 0.07)},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header.rstrip()]
    for cue in _caption_cues(track.words):
        karaoke: list[str] = []
        cursor = cue[0].start_seconds
        for index, word in enumerate(cue):
            gap_cs = round(max(0.0, word.start_seconds - cursor) * 100)
            if gap_cs:
                karaoke.append(r"{\k" + str(gap_cs) + "}")
            duration_cs = max(1, round((word.end_seconds - word.start_seconds) * 100))
            prefix = " " if index else ""
            karaoke.append(prefix + r"{\k" + str(duration_cs) + "}" + _ass_escape(word.text))
            cursor = word.end_seconds
        text = r"{\fad(80,100)}" + "".join(karaoke)
        lines.append(
            f"Dialogue: 0,{_ass_timestamp(cue[0].start_seconds)},{_ass_timestamp(cue[-1].end_seconds)},Default,,0,0,0,,{text}"
        )
    atomic_write_text(path, "\n".join(lines) + "\n")


def normalize_image(tools: MediaTools, source: Path, destination: Path, *, width: int, height: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tools.run(
        [
            tools.ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            "-vf",
            f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},setsar=1",
            "-frames:v",
            "1",
            "-c:v",
            "png",
            str(destination),
        ]
    )


def fit_music(
    tools: MediaTools,
    source: Path,
    destination: Path,
    *,
    duration: float,
    allow_loop: bool,
) -> AudioProbe:
    source_probe = tools.probe_audio(source)
    if source_probe.duration_seconds + 0.02 < duration and not allow_loop:
        raise MediaError(
            f"music is {source_probe.duration_seconds:.2f}s but the Timeline is {duration:.2f}s and looping is disabled",
            kind=ErrorKind.INVALID_OUTPUT,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [tools.ffmpeg, "-y", "-v", "error"]
    if source_probe.duration_seconds + 0.02 < duration:
        command.extend(["-stream_loop", "-1"])
    fade_start = max(0.0, duration - min(2.5, duration * 0.1))
    command.extend(
        [
            "-i",
            str(source),
            "-t",
            f"{duration:.6f}",
            "-af",
            f"loudnorm=I=-30:LRA=5:TP=-4,afade=t=out:st={fade_start:.6f}:d={duration - fade_start:.6f}",
            "-ar",
            "48000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]
    )
    tools.run(command)
    result = tools.probe_audio(destination)
    if abs(result.duration_seconds - duration) > 0.04:
        raise MediaError("fitted music duration does not match the Narration Timeline", kind=ErrorKind.INVALID_OUTPUT)
    return result


def render_video(
    tools: MediaTools,
    plan: RenderPlan,
    *,
    workspace_root: Path,
    base_path: Path,
    output_path: Path,
    burned_output_path: Path | None = None,
) -> list[Path]:
    if not plan.scenes:
        raise MediaError("Render Plan contains no Scenes", kind=ErrorKind.INVALID_OUTPUT)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    frame_durations: list[float] = []
    previous_end_frame = 0
    for index, scene in enumerate(plan.scenes):
        end_frame = (
            round(plan.duration_seconds * plan.fps)
            if index == len(plan.scenes) - 1
            else round(scene.end_seconds * plan.fps)
        )
        frame_count = end_frame - previous_end_frame
        if frame_count <= 0:
            raise MediaError(
                f"Scene {scene.scene_id} has no renderable frames",
                kind=ErrorKind.INVALID_OUTPUT,
            )
        frame_durations.append(frame_count / plan.fps)
        previous_end_frame = end_frame
    command = [tools.ffmpeg, "-y", "-v", "error"]
    for scene, duration in zip(plan.scenes, frame_durations, strict=True):
        command.extend(
            [
                "-loop",
                "1",
                "-framerate",
                str(plan.fps),
                "-t",
                f"{duration:.6f}",
                "-i",
                str(workspace_root / scene.image_path),
            ]
        )
    narration_index = len(plan.scenes)
    command.extend(["-i", str(workspace_root / plan.narration_path)])
    music_index = None
    if plan.music_path:
        music_index = narration_index + 1
        command.extend(["-i", str(workspace_root / plan.music_path)])
    filters = []
    video_labels = []
    for index, (scene, duration) in enumerate(zip(plan.scenes, frame_durations, strict=True)):
        filters.append(
            f"[{index}:v]trim=duration={duration:.6f},setpts=PTS-STARTPTS,"
            f"scale={plan.width}:{plan.height}:force_original_aspect_ratio=increase,"
            f"crop={plan.width}:{plan.height},setsar=1,fps={plan.fps}[v{index}]"
        )
        video_labels.append(f"[v{index}]")
    filters.append("".join(video_labels) + f"concat=n={len(video_labels)}:v=1:a=0[outv]")
    filters.append(
        f"[{narration_index}:a]atrim=duration={plan.duration_seconds:.6f},asetpts=PTS-STARTPTS,"
        f"apad,atrim=duration={plan.duration_seconds:.6f}[narr]"
    )
    if music_index is not None:
        filters.append(
            f"[{music_index}:a]atrim=duration={plan.duration_seconds:.6f},asetpts=PTS-STARTPTS,volume=0.16[music]"
        )
        filters.append("[narr][music]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[outa]")
    else:
        filters.append("[narr]alimiter=limit=0.95[outa]")
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[outv]",
            "-map",
            "[outa]",
            "-t",
            f"{plan.duration_seconds:.6f}",
            "-r",
            str(plan.fps),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "19",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(base_path),
        ]
    )
    tools.run(command, timeout=1800)
    outputs = []
    if plan.caption_srt_path:
        srt = workspace_root / plan.caption_srt_path
        tools.run(
            [
                tools.ffmpeg,
                "-y",
                "-v",
                "error",
                "-i",
                str(base_path),
                "-i",
                str(srt),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
                "-map",
                "1:0",
                "-c:v",
                "copy",
                "-c:a",
                "copy",
                "-c:s",
                "mov_text",
                "-metadata:s:s:0",
                "language=fin" if plan.caption_language is OutputLanguage.FINNISH else "language=eng",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            timeout=900,
        )
        outputs.append(output_path)
    else:
        replace_path(base_path, output_path)
        outputs.append(output_path)
    if burned_output_path and plan.caption_ass_path:
        ass_path = (workspace_root / plan.caption_ass_path).resolve()
        filter_value = "ass=" + ass_path.name.replace("\\", "/").replace(":", r"\:")
        tools.run(
            [
                tools.ffmpeg,
                "-y",
                "-v",
                "error",
                "-i",
                str(output_path),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
                "-vf",
                filter_value,
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "19",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "copy",
                "-movflags",
                "+faststart",
                str(burned_output_path),
            ],
            cwd=ass_path.parent,
            timeout=1800,
        )
        outputs.append(burned_output_path)
    if base_path.exists() and base_path != output_path:
        base_path.unlink()
    return outputs


def _fraction(value: str) -> float:
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return 0.0


def qc_video(
    tools: MediaTools,
    path: Path,
    *,
    width: int,
    height: int,
    fps: int,
    expected_duration: float,
    budget: float,
    captions_expected: bool,
) -> list[QCCheck]:
    payload = tools.probe_json(path)
    streams = payload.get("streams", [])
    video = next((item for item in streams if item.get("codec_type") == "video"), {})
    audio = next((item for item in streams if item.get("codec_type") == "audio"), {})
    subtitles = [item for item in streams if item.get("codec_type") == "subtitle"]
    try:
        duration = float(payload.get("format", {}).get("duration"))
    except (TypeError, ValueError):
        duration = -1.0
    checks = [
        QCCheck(name="video_codec", passed=video.get("codec_name") == "h264", detail=str(video.get("codec_name"))),
        QCCheck(name="pixel_format", passed=video.get("pix_fmt") == "yuv420p", detail=str(video.get("pix_fmt"))),
        QCCheck(
            name="resolution",
            passed=video.get("width") == width and video.get("height") == height,
            detail=f"{video.get('width')}x{video.get('height')}",
        ),
        QCCheck(
            name="frame_rate",
            passed=abs(_fraction(str(video.get("avg_frame_rate") or "0/1")) - fps) < 0.01,
            detail=str(video.get("avg_frame_rate")),
        ),
        QCCheck(name="audio_codec", passed=audio.get("codec_name") == "aac", detail=str(audio.get("codec_name"))),
        QCCheck(
            name="duration_matches_timeline",
            passed=abs(duration - expected_duration) <= 1 / fps + 0.035,
            detail=f"actual={duration:.3f}s expected={expected_duration:.3f}s",
        ),
        QCCheck(
            name="duration_hard_limit",
            passed=0 < duration <= budget + 0.005,
            detail=f"actual={duration:.3f}s budget={budget:.3f}s",
        ),
        QCCheck(
            name="selectable_captions",
            passed=(not captions_expected) or any(item.get("codec_name") == "mov_text" for item in subtitles),
            detail=",".join(str(item.get("codec_name")) for item in subtitles) or "none",
        ),
    ]
    volume = subprocess.run(
        [tools.ffmpeg, "-v", "info", "-i", str(path), "-map", "0:a:0", "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    mean_match = re.search(r"mean_volume:\s*(-?[\d.]+) dB", volume.stderr)
    max_match = re.search(r"max_volume:\s*(-?[\d.]+) dB", volume.stderr)
    mean_volume = float(mean_match.group(1)) if mean_match else -999
    max_volume = float(max_match.group(1)) if max_match else 999
    checks.append(
        QCCheck(
            name="audio_health",
            passed=mean_volume > -55 and max_volume <= 0.1,
            detail=f"mean={mean_volume:.1f}dB max={max_volume:.1f}dB",
        )
    )
    return checks


def delivery_manifest(
    *,
    run_id: str,
    output_files: Iterable[tuple[str, Path, str]],
    workspace_root: Path,
    duration: float,
    checks: Sequence[QCCheck],
    warnings: Sequence[str],
) -> DeliveryManifest:
    outputs = [
        DeliveryFile(
            role=role,
            media=MediaReference(
                path=relative_path(path, workspace_root), sha256=sha256_file(path), mime_type=mime
            ),
        )
        for role, path, mime in output_files
    ]
    return DeliveryManifest(
        run_id=run_id,
        outputs=outputs,
        duration_seconds=duration,
        checks=list(checks),
        warnings=list(warnings),
    )
