from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Protocol

from ..errors import ErrorKind, VideoGeneratorError
from ..util import replace_path


class Worker(Protocol):
    def health(self) -> dict[str, Any]: ...

    def dispatch(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]: ...

    def close(self) -> dict[str, Any]: ...


class Paths:
    def __init__(self) -> None:
        self.project_root = Path(os.environ["VIDEO_GENERATOR_PROJECT_ROOT"]).resolve()
        self.run_root = Path(os.environ["VIDEO_GENERATOR_RUN_ROOT"]).resolve()
        self.model_root = (self.project_root / ".cache" / "models").resolve()
        self.runtime_root = (self.project_root / ".cache" / "runtimes").resolve()
        self.private_root = (self.project_root / "private").resolve()

    def read_run(self, value: str) -> Path:
        return self._inside(value, self.run_root, must_exist=True)

    def read_model(self, value: str) -> Path:
        return self._inside(value, self.model_root, must_exist=True)

    def read_runtime(self, value: str) -> Path:
        return self._inside(value, self.runtime_root, must_exist=True)

    def read_private(self, value: str) -> Path:
        return self._inside(value, self.private_root, must_exist=True)

    def output_run(self, value: str) -> Path:
        path = self._inside(value, self.run_root, must_exist=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _inside(self, value: str, root: Path, *, must_exist: bool) -> Path:
        raw = Path(value)
        path = raw.resolve() if raw.is_absolute() else (self.project_root / raw).resolve()
        path.relative_to(root)
        if must_exist and not path.exists():
            raise FileNotFoundError(path)
        return path


def _reset_cuda_peak(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_cuda_mb(torch: Any) -> float | None:
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    return max(
        float(torch.cuda.max_memory_allocated()),
        float(torch.cuda.max_memory_reserved()),
    ) / (1024 * 1024)


def _positive_prompt_with_exclusions(prompt: str, negative_prompt: str) -> tuple[str, str]:
    exclusions = negative_prompt.strip()
    if not exclusions:
        return prompt, "none"
    return (
        prompt.rstrip()
        + "\n\nHard exclusions: Do not show or imply any of the following: "
        + exclusions,
        "positive_avoid_clause",
    )


def compile_ideogram4_prompt(prompt: str, negative_prompt: str = "") -> str:
    description = prompt.strip()
    lower_description = description.casefold()
    exclusions = negative_prompt.casefold()
    positive_constraints: list[str] = []
    if any(value in exclusions for value in ("text", "label", "logo", "watermark")):
        positive_constraints.append(
            "The finished image is clean, unbranded, and contains no added lettering."
        )
    if "clutter" in exclusions:
        positive_constraints.append("The layout is spacious, orderly, and uncluttered.")
    if any(value in exclusions for value in ("malformed", "anatomy", "extra finger")):
        positive_constraints.append("People, hands, and objects have coherent natural structure.")
    if any(value in exclusions for value in ("photoreal", "3d render", "gradient")):
        positive_constraints.append("The requested illustrative medium remains visually consistent throughout.")
    constraint_clause = " " + " ".join(positive_constraints) if positive_constraints else ""
    if any(value in lower_description for value in ("photograph", "photoreal", "camera", "lens")):
        style_description = {
            "aesthetics": "polished, coherent, faithful to the high-level description",
            "lighting": "use the lighting, shadows, and atmosphere specified in the high-level description",
            "photo": "use the camera, lens, framing, depth of field, and viewpoint specified in the high-level description",
            "medium": "photograph",
        }
    else:
        medium = "3d_render" if "3d" in lower_description else "illustration"
        style_description = {
            "aesthetics": "polished, coherent, uncluttered, faithful to the high-level description",
            "lighting": "use the lighting, shadows, and atmosphere specified in the high-level description",
            "medium": medium,
            "art_style": "use the exact artistic style, line quality, texture, and finish specified in the high-level description",
        }
    caption = {
        "high_level_description": description + constraint_clause,
        "style_description": style_description,
        "compositional_deconstruction": {
            "background": (
                "Render the environment, background, lighting, palette, and framing exactly as "
                "specified in the high-level description."
            ),
            "elements": [
                {
                    "type": "obj",
                    "desc": (
                        "Render every requested visible subject, object, action, relationship, and "
                        "stylistic detail exactly as specified in the high-level description."
                        + constraint_clause
                    ),
                }
            ],
        },
    }
    return json.dumps(caption, separators=(",", ":"), ensure_ascii=False)


def _ideogram4_sampler_preset(steps: int) -> tuple[str, list[float], float, float]:
    presets = {
        12: ("V4_TURBO_12", 1, 0.5, 1.75),
        20: ("V4_DEFAULT_20", 2, 0.0, 1.75),
        48: ("V4_QUALITY_48", 3, 0.0, 1.5),
    }
    try:
        name, polish_steps, mu, std = presets[steps]
    except KeyError as exc:
        supported = ", ".join(str(value) for value in presets)
        raise ValueError(
            f"Ideogram 4 inference_steps must use an official preset: {supported}"
        ) from exc
    guidance_schedule = [7.0] * (steps - polish_steps) + [3.0] * polish_steps
    return name, guidance_schedule, mu, std


def _image_seed(payload: dict[str, Any]) -> int:
    value = payload.get("seed")
    return int(value) if value is not None else int.from_bytes(os.urandom(4), "big")


def _reject_image_references(payload: dict[str, Any], model_name: str) -> None:
    if payload.get("reference_paths"):
        raise ValueError(f"{model_name} is text-to-image only and does not accept reference images")


def _save_diffusers_image(
    worker: Any,
    *,
    payload: dict[str, Any],
    call: dict[str, Any],
    generation_settings: dict[str, Any],
    image_validator: Any | None = None,
) -> dict[str, Any]:
    output = worker.paths.output_run(payload["output_path"])
    _reset_cuda_peak(worker.torch)
    started = time.monotonic()
    with worker.torch.inference_mode():
        result = worker.pipe(**call)
    elapsed = time.monotonic() - started
    generation_peak_vram_mb = _peak_cuda_mb(worker.torch)
    images = getattr(result, "images", None)
    if not images:
        raise VideoGeneratorError(
            "image model returned no image",
            kind=ErrorKind.INVALID_OUTPUT,
        )
    image = images[0]
    if image_validator is not None:
        image_validator(image)
    image.save(output, format="PNG")
    generation_settings.update(
        {
            "load_elapsed_seconds": worker.load_elapsed_seconds,
            "load_peak_vram_mb": worker.load_peak_vram_mb,
            "generation_peak_vram_mb": generation_peak_vram_mb,
        }
    )
    peaks = [
        value
        for value in (worker.load_peak_vram_mb, generation_peak_vram_mb)
        if value is not None
    ]
    return {
        "image_path": payload["output_path"],
        "mime_type": "image/png",
        "generation_settings": generation_settings,
        "usage": {
            "elapsed_seconds": elapsed,
            "peak_vram_mb": max(peaks) if peaks else None,
        },
    }


def _reject_ideogram_safety_placeholder(image: Any) -> None:
    if not all(hasattr(image, name) for name in ("convert", "resize")):
        return
    thumbnail = image.convert("RGB").resize((64, 36))
    get_pixels = getattr(thumbnail, "get_flattened_data", None) or thumbnail.getdata
    pixels = list(get_pixels())
    if not pixels:
        return
    count = len(pixels)
    means = [sum(pixel[channel] for pixel in pixels) / count for channel in range(3)]
    deviations = [
        (
            sum((pixel[channel] - means[channel]) ** 2 for pixel in pixels) / count
        )
        ** 0.5
        for channel in range(3)
    ]
    corners = (pixels[0], pixels[63], pixels[-64], pixels[-1])
    uniform_gray = max(deviations) < 12 and max(means) - min(means) < 4
    neutral_corners = all(
        max(abs(pixel[channel] - means[channel]) for channel in range(3)) < 8
        for pixel in corners
    )
    has_light_center_text = max(sum(pixel) / 3 for pixel in pixels) > 145
    if uniform_gray and neutral_corners and has_light_center_text:
        raise VideoGeneratorError(
            "Ideogram 4 blocked the image with its built-in safety filter",
            kind=ErrorKind.POLICY_REFUSAL,
        )


def _llama_grammar_schema(value: Any) -> Any:
    if isinstance(value, list):
        return [_llama_grammar_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: _llama_grammar_schema(item) for key, item in value.items()}
    for unsupported_keyword in ("maxItems", "maxLength", "minItems", "pattern"):
        result.pop(unsupported_keyword, None)
    if result.get("type") == "number":
        for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
            result.pop(key, None)
    return result


class LlamaServerWorker:
    def __init__(self, paths: Paths) -> None:
        from .llama_server import LlamaServerSession

        model_path = paths.read_model(os.environ["VIDEO_GENERATOR_MODEL_PATH"])
        self.model_path = model_path
        draft_value = os.environ.get("VIDEO_GENERATOR_DRAFT_MODEL_PATH", "")
        draft_model = paths.read_model(draft_value) if draft_value else None
        server = paths.read_runtime(os.environ["VIDEO_GENERATOR_LLAMA_SERVER"])
        arguments = json.loads(os.environ.get("VIDEO_GENERATOR_LLAMA_ARGUMENTS", "[]"))
        if not isinstance(arguments, list) or any(not isinstance(value, str) for value in arguments):
            raise ValueError("VIDEO_GENERATOR_LLAMA_ARGUMENTS must be a JSON string array")
        self.session = LlamaServerSession(
            executable=server,
            model=model_path,
            draft_model=draft_model,
            arguments=arguments,
            startup_timeout_seconds=float(
                os.environ.get("VIDEO_GENERATOR_LLAMA_STARTUP_TIMEOUT", "600")
            ),
            request_timeout_seconds=float(
                os.environ.get("VIDEO_GENERATOR_LLAMA_REQUEST_TIMEOUT", "600")
            ),
        )
        self.session.start()

    def health(self) -> dict[str, Any]:
        process = self.session.process
        return {
            "model_path": str(self.model_path),
            "runtime": "stock-llama-server",
            "server_pid": process.pid if process is not None else None,
            "loopback": self.session.base_url,
            "startup_elapsed_seconds": self.session.startup_elapsed_seconds,
        }

    def dispatch(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        if operation != "structured_text.complete":
            raise ValueError(f"unsupported llama-server operation: {operation}")
        if payload.get("media_inputs"):
            raise ValueError("this local LLM runner is text-only; use an evaluated vision runner")
        output_schema = payload["output_schema"]
        grammar_schema = _llama_grammar_schema(output_schema)
        messages = [
            {
                "role": "system",
                "content": (
                    payload["instructions"]
                    + "\n\nReturn one JSON value matching this exact schema:\n"
                    + json.dumps(output_schema, ensure_ascii=False, separators=(",", ":"))
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload["input_data"], ensure_ascii=False, indent=2),
            },
        ]
        started = time.monotonic()
        temperature = (
            0.1
            if payload["task_id"] == "factual_review"
            else 0.2
            if payload["task_id"] in {"script_draft", "script_revision", "duration_repair"}
            else 0.65
        )
        max_output_tokens = int(payload.get("max_output_tokens", 8000))
        structured_mode = os.environ.get(
            "VIDEO_GENERATOR_LLAMA_STRUCTURED_MODE", "chat_completions"
        )
        if structured_mode == "template_completion":
            response = self.session.structured_completion(
                {
                    "messages": messages,
                    "n_predict": max_output_tokens,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "stream": False,
                    "json_schema": grammar_schema,
                }
            )
            content = response.get("content")
            stop_type = str(response.get("stop_type") or "")
            finish_reason = (
                "stop"
                if response.get("stop") is True and stop_type != "limit"
                else stop_type or "length"
            )
            input_tokens = int(response.get("tokens_evaluated") or 0)
            output_tokens = int(response.get("tokens_predicted") or 0)
        elif structured_mode == "chat_completions":
            response = self.session.chat_completion(
                {
                    "messages": messages,
                    "max_tokens": max_output_tokens,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "stream": False,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": str(payload["task_id"]).replace("-", "_"),
                            "strict": True,
                            "schema": grammar_schema,
                        },
                    },
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            )
            try:
                choice = response["choices"][0]
                content = choice["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError(
                    "llama-server response did not contain assistant content"
                ) from exc
            if isinstance(content, str):
                content = re.sub(r"^.*?</think>\s*", "", content, flags=re.DOTALL)
            finish_reason = str(choice.get("finish_reason") or "")
            usage = response.get("usage") or {}
            input_tokens = int(usage.get("prompt_tokens") or 0)
            output_tokens = int(usage.get("completion_tokens") or 0)
        else:
            raise ValueError(f"unsupported local LLM structured output mode: {structured_mode}")
        if finish_reason != "stop":
            raise ValueError(
                "llama-server response did not finish normally: "
                + finish_reason
            )
        if not isinstance(content, str):
            raise ValueError("llama-server completion content was not text")
        data = json.loads(content)
        baseline = self.session.baseline.used_mb
        peak = self.session.peak_used_mb
        return {
            "data": data,
            "provider_request_id": str(response.get("id") or ""),
            "finish_reason": finish_reason,
            "runtime_revision": os.environ.get("VIDEO_GENERATOR_RUNTIME_REVISION", ""),
            "model_revision": os.environ.get("VIDEO_GENERATOR_MODEL_REVISION", ""),
            "model_id": os.environ.get("VIDEO_GENERATOR_MODEL_ID", ""),
            "profile_id": os.environ.get("VIDEO_GENERATOR_LLM_PROFILE_ID", ""),
            "context_size": int(os.environ.get("VIDEO_GENERATOR_LLAMA_CONTEXT", "0")),
            "speculation": os.environ.get("VIDEO_GENERATOR_LLAMA_SPECULATION", "none"),
            "startup_elapsed_seconds": self.session.startup_elapsed_seconds,
            "server_timings": response.get("timings")
            if isinstance(response.get("timings"), dict)
            else {},
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "elapsed_seconds": time.monotonic() - started,
                "peak_vram_mb": max(0, peak - baseline)
                if peak is not None and baseline is not None
                else None,
            },
        }

    def close(self) -> dict[str, Any]:
        return self.session.close()


class VoxCPMWorker:
    def __init__(self, paths: Paths) -> None:
        from voxcpm import VoxCPM

        model_path = paths.read_model(os.environ["VIDEO_GENERATOR_MODEL_PATH"])
        self.paths = paths
        self.model_path = model_path
        self.model = VoxCPM.from_pretrained(
            str(model_path),
            load_denoiser=False,
            local_files_only=True,
            optimize=os.environ.get("VIDEO_GENERATOR_VOXCPM_OPTIMIZE", "0") == "1",
            device="cuda:0",
        )

    def health(self) -> dict[str, Any]:
        return {"model_path": str(self.model_path), "sample_rate": int(self.model.tts_model.sample_rate)}

    def dispatch(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        if operation != "speech.synthesize":
            raise ValueError(f"unsupported VoxCPM operation: {operation}")
        import soundfile as sf
        import torch

        voice = payload["voice"]
        reference = self.paths.read_private(voice["reference_audio"])
        transcript = ""
        if voice.get("reference_transcript"):
            transcript = self.paths.read_private(voice["reference_transcript"]).read_text(encoding="utf-8").strip()
        output = self.paths.output_run(payload["output_path"])
        seed = int.from_bytes(
            hashlib.sha256(
                (str(payload["scene_id"]) + "\0" + str(payload["text"])).encode("utf-8")
            ).digest()[:4],
            "big",
        )
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        started = time.monotonic()
        wav = self.model.generate(
            text=payload["text"],
            prompt_wav_path=str(reference) if transcript else None,
            prompt_text=transcript or None,
            reference_wav_path=str(reference),
            cfg_value=2.0,
            inference_timesteps=10,
            normalize=False,
            denoise=False,
            retry_badcase=False,
        )
        sample_rate = int(self.model.tts_model.sample_rate)
        sf.write(str(output), wav, sample_rate)
        return {
            "audio_path": payload["output_path"],
            "mime_type": "audio/wav",
            "duration_seconds": len(wav) / sample_rate,
            "sample_rate": sample_rate,
            "channels": 1,
            "timing_precision": "none",
            "word_timings": [],
            "seed": seed,
            "usage": {"elapsed_seconds": time.monotonic() - started},
        }


class OmniVoiceWorker:
    def __init__(self, paths: Paths) -> None:
        import torch
        from omnivoice import OmniVoice

        model_path = paths.read_model(os.environ["VIDEO_GENERATOR_MODEL_PATH"])
        self.paths = paths
        self.model_path = model_path
        self.torch = torch
        _reset_cuda_peak(torch)
        started = time.monotonic()
        self.model = OmniVoice.from_pretrained(
            str(model_path),
            device_map="cuda:0",
            dtype=torch.float16,
            load_asr=False,
        )
        self.voice_clone_prompts: dict[tuple[str, str], Any] = {}
        self.load_elapsed_seconds = time.monotonic() - started
        self.load_peak_vram_mb = _peak_cuda_mb(torch)

    def health(self) -> dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "runtime": "omnivoice",
            "sample_rate": 24000,
            "load_elapsed_seconds": self.load_elapsed_seconds,
            "load_peak_vram_mb": self.load_peak_vram_mb,
        }

    def dispatch(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        if operation != "speech.synthesize":
            raise ValueError(f"unsupported OmniVoice operation: {operation}")
        import soundfile as sf

        voice = payload["voice"]
        if not voice.get("reference_audio") or not voice.get("reference_transcript"):
            raise ValueError(
                "OmniVoice requires both an authorized reference audio file and its transcript"
            )
        reference = self.paths.read_private(voice["reference_audio"])
        transcript = self.paths.read_private(voice["reference_transcript"]).read_text(
            encoding="utf-8"
        ).strip()
        if not transcript:
            raise ValueError("OmniVoice reference transcript is empty")
        language = str(payload["output_language"])
        if language not in {"en", "fi"}:
            raise ValueError(f"unsupported OmniVoice language: {language}")
        output = self.paths.output_run(payload["output_path"])
        seed = int.from_bytes(
            hashlib.sha256(
                (str(payload["scene_id"]) + "\0" + str(payload["text"])).encode("utf-8")
            ).digest()[:4],
            "big",
        )
        self.torch.manual_seed(seed)
        self.torch.cuda.manual_seed_all(seed)
        _reset_cuda_peak(self.torch)
        started = time.monotonic()
        prompt_key = (str(reference), transcript)
        voice_clone_prompt = self.voice_clone_prompts.get(prompt_key)
        if voice_clone_prompt is None:
            voice_clone_prompt = self.model.create_voice_clone_prompt(
                ref_audio=str(reference),
                ref_text=transcript,
            )
            self.voice_clone_prompts[prompt_key] = voice_clone_prompt
        audio = self.model.generate(
            text=payload["text"],
            voice_clone_prompt=voice_clone_prompt,
            language=language,
            speed=1.0,
        )
        elapsed = time.monotonic() - started
        peak_vram_mb = _peak_cuda_mb(self.torch)
        if not audio:
            raise VideoGeneratorError(
                "OmniVoice returned no audio",
                kind=ErrorKind.INVALID_OUTPUT,
            )
        waveform = audio[0]
        if hasattr(waveform, "detach"):
            waveform = waveform.detach().float().cpu().numpy()
        sf.write(str(output), waveform, 24000)
        return {
            "audio_path": payload["output_path"],
            "mime_type": "audio/wav",
            "duration_seconds": len(waveform) / 24000,
            "sample_rate": 24000,
            "channels": 1,
            "timing_precision": "none",
            "word_timings": [],
            "seed": seed,
            "usage": {
                "elapsed_seconds": elapsed,
                "peak_vram_mb": max(
                    value
                    for value in (self.load_peak_vram_mb, peak_vram_mb)
                    if value is not None
                ),
            },
        }


class MossTtsWorker:
    def __init__(self, paths: Paths) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self.dll_directory_handles: list[Any] = []
        self.ffmpeg_dll_directory = ""
        if os.name == "nt":
            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                raise VideoGeneratorError(
                    "MOSS-TTS requires shared FFmpeg on Windows",
                    kind=ErrorKind.NOT_READY,
                )
            self.ffmpeg_dll_directory = str(Path(ffmpeg).resolve().parent)
            self.dll_directory_handles.append(
                os.add_dll_directory(self.ffmpeg_dll_directory)
            )
        model_path = paths.read_model(os.environ["VIDEO_GENERATOR_MODEL_PATH"])
        codec_path = paths.read_model(os.environ["VIDEO_GENERATOR_CODEC_PATH"])
        self.paths = paths
        self.model_path = model_path
        self.codec_path = codec_path
        self.torch = torch
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
        _reset_cuda_peak(torch)
        started = time.monotonic()
        self.processor = AutoProcessor.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            codec_path=str(codec_path),
            codec_weight_dtype="bf16",
            codec_attention_implementation="sdpa",
        )
        self.processor.audio_tokenizer = self.processor.audio_tokenizer.to("cuda")
        self.model = (
            AutoModel.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                local_files_only=True,
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16,
            )
            .to("cuda")
            .eval()
        )
        self.load_elapsed_seconds = time.monotonic() - started
        self.load_peak_vram_mb = _peak_cuda_mb(torch)

    def health(self) -> dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "codec_path": str(self.codec_path),
            "runtime": "transformers-local-remote-code",
            "attention": "sdpa",
            "ffmpeg_dll_directory": self.ffmpeg_dll_directory,
            "sample_rate": 48000,
            "load_elapsed_seconds": self.load_elapsed_seconds,
            "load_peak_vram_mb": self.load_peak_vram_mb,
        }

    def dispatch(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        if operation != "speech.synthesize":
            raise ValueError(f"unsupported MOSS-TTS operation: {operation}")
        import torchaudio

        voice = payload["voice"]
        if not voice.get("reference_audio"):
            raise ValueError("MOSS-TTS requires an authorized reference audio file")
        reference = self.paths.read_private(voice["reference_audio"])
        language_id = str(payload["output_language"])
        try:
            language = {"en": "English", "fi": "Finnish"}[language_id]
        except KeyError as exc:
            raise ValueError(f"unsupported MOSS-TTS language: {language_id}") from exc
        conversation = [
            [
                self.processor.build_user_message(
                    text=payload["text"],
                    reference=[str(reference)],
                    language=language,
                )
            ]
        ]
        batch = self.processor(conversation, mode="generation")
        if hasattr(batch, "to"):
            batch = batch.to("cuda")
        else:
            batch = {
                key: value.to("cuda") if hasattr(value, "to") else value
                for key, value in batch.items()
            }
        output = self.paths.output_run(payload["output_path"])
        seed = int.from_bytes(
            hashlib.sha256(
                (str(payload["scene_id"]) + "\0" + str(payload["text"])).encode("utf-8")
            ).digest()[:4],
            "big",
        )
        self.torch.manual_seed(seed)
        self.torch.cuda.manual_seed_all(seed)
        _reset_cuda_peak(self.torch)
        started = time.monotonic()
        with self.torch.inference_mode():
            outputs = self.model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=4096,
                do_sample=True,
                audio_temperature=1.7,
                audio_top_p=0.8,
                audio_top_k=25,
                audio_repetition_penalty=1.0,
            )
        elapsed = time.monotonic() - started
        peak_vram_mb = _peak_cuda_mb(self.torch)
        decoded = self.processor.decode(outputs)
        message = next((item for item in decoded if item is not None), None)
        if message is None or not getattr(message, "audio_codes_list", None):
            raise VideoGeneratorError(
                "MOSS-TTS returned no audio",
                kind=ErrorKind.INVALID_OUTPUT,
            )
        waveform = message.audio_codes_list[0].detach().cpu()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        sample_rate = 48000
        torchaudio.save(str(output), waveform, sample_rate)
        return {
            "audio_path": payload["output_path"],
            "mime_type": "audio/wav",
            "duration_seconds": int(waveform.shape[-1]) / sample_rate,
            "sample_rate": sample_rate,
            "channels": int(waveform.shape[0]),
            "timing_precision": "none",
            "word_timings": [],
            "seed": seed,
            "usage": {
                "elapsed_seconds": elapsed,
                "peak_vram_mb": max(
                    value
                    for value in (self.load_peak_vram_mb, peak_vram_mb)
                    if value is not None
                ),
            },
        }


class XVoiceWorker:
    def __init__(self, paths: Paths) -> None:
        import torch
        from hydra.utils import get_class
        from omegaconf import OmegaConf
        from x_voice.infer.utils_infer import (
            get_ipa_tokenizer_cache,
            infer_xvoice_process,
            load_model,
            load_vocoder,
            preprocess_ref_audio_text,
        )

        self.paths = paths
        self.torch = torch
        self.preprocess_ref_audio_text = preprocess_ref_audio_text
        self.infer_xvoice_process = infer_xvoice_process
        self.reference_cache: dict[tuple[str, str], tuple[str, str]] = {}
        self.temporary_reference_files: set[Path] = set()
        self.model_path = paths.read_model(os.environ["VIDEO_GENERATOR_MODEL_PATH"])
        self.vocoder_path = paths.read_model(
            os.environ["VIDEO_GENERATOR_XVOICE_VOCODER_PATH"]
        )
        source_path = paths.read_runtime(os.environ["VIDEO_GENERATOR_XVOICE_SOURCE_PATH"])
        config_path = source_path / "src" / "x_voice" / "configs" / "XVoice_Base_Stage1.yaml"
        checkpoint_path = (
            self.model_path / "XVoice_Base_Stage1" / "model_600000.safetensors"
        )
        vocab_path = self.model_path / "XVoice_Base_Stage1" / "vocab.txt"
        for required in (config_path, checkpoint_path, vocab_path):
            if not required.is_file():
                raise FileNotFoundError(required)
        model_config = OmegaConf.load(config_path)
        model_class = get_class(f"x_voice.model.{model_config.model.backbone}")
        model_architecture = OmegaConf.to_container(model_config.model.arch, resolve=True)
        mel_spec_kwargs = OmegaConf.to_container(model_config.model.mel_spec, resolve=True)
        self.tokenizer_name = str(model_config.model.tokenizer)
        self.ipa_tokenizer_getter = get_ipa_tokenizer_cache(
            self.tokenizer_name,
            bool(model_config.model.get("stress", True)),
        )
        _reset_cuda_peak(torch)
        started = time.monotonic()
        self.vocoder = load_vocoder(
            "vocos",
            is_local=True,
            local_path=str(self.vocoder_path),
            device="cuda",
        )
        self.model = load_model(
            model_class,
            model_architecture,
            str(checkpoint_path),
            mel_spec_type="vocos",
            vocab_file=str(vocab_path),
            device="cuda",
            tokenizer=self.tokenizer_name,
            tokenizer_path=model_config.model.get("tokenizer_path"),
            dataset_name=str(model_config.datasets.name),
            mel_spec_kwargs=mel_spec_kwargs,
        )
        self.language_ids = getattr(self.model.transformer, "lang_to_id", {})
        self.load_elapsed_seconds = time.monotonic() - started
        self.load_peak_vram_mb = _peak_cuda_mb(torch)

    def health(self) -> dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "vocoder_path": str(self.vocoder_path),
            "runtime": "x-voice-stage1",
            "runtime_revision": os.environ.get("VIDEO_GENERATOR_RUNTIME_REVISION", ""),
            "sample_rate": 24000,
            "languages": ["en", "fi"],
            "load_elapsed_seconds": self.load_elapsed_seconds,
            "load_peak_vram_mb": self.load_peak_vram_mb,
        }

    def _reference_prompt(self, reference: Path, transcript: str) -> tuple[str, str]:
        key = (str(reference), transcript)
        cached = self.reference_cache.get(key)
        if cached is not None:
            return cached
        audio, text = self.preprocess_ref_audio_text(
            str(reference),
            transcript,
            show_info=lambda _message: None,
        )
        value = (str(audio), str(text))
        self.reference_cache[key] = value
        processed = Path(value[0]).resolve()
        if processed != reference.resolve() and processed.is_relative_to(
            Path(tempfile.gettempdir()).resolve()
        ):
            self.temporary_reference_files.add(processed)
        return value

    def dispatch(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        if operation != "speech.synthesize":
            raise ValueError(f"unsupported X-Voice operation: {operation}")
        import soundfile as sf

        voice = payload["voice"]
        if not voice.get("reference_audio") or not voice.get("reference_transcript"):
            raise ValueError(
                "X-Voice requires both an authorized reference audio file and its exact transcript"
            )
        reference = self.paths.read_private(voice["reference_audio"])
        transcript = self.paths.read_private(voice["reference_transcript"]).read_text(
            encoding="utf-8"
        ).strip()
        if not transcript:
            raise ValueError("X-Voice reference transcript is empty")
        reference_language = str(voice.get("reference_language") or "")
        output_language = str(payload["output_language"])
        if reference_language not in {"en", "fi"}:
            raise ValueError(f"unsupported X-Voice reference language: {reference_language}")
        if output_language not in {"en", "fi"}:
            raise ValueError(f"unsupported X-Voice output language: {output_language}")
        processed_audio, processed_text = self._reference_prompt(reference, transcript)
        output = self.paths.output_run(payload["output_path"])
        seed = int.from_bytes(
            hashlib.sha256(
                (str(payload["scene_id"]) + "\0" + str(payload["text"])).encode("utf-8")
            ).digest()[:4],
            "big",
        )
        self.torch.manual_seed(seed)
        self.torch.cuda.manual_seed_all(seed)
        _reset_cuda_peak(self.torch)
        started = time.monotonic()
        waveform, sample_rate, _spectrogram = self.infer_xvoice_process(
            processed_audio,
            processed_text,
            str(payload["text"]),
            reference_language,
            output_language,
            self.tokenizer_name,
            self.ipa_tokenizer_getter,
            self.model,
            self.vocoder,
            self.language_ids,
            dominant_lang=output_language,
            srp_model=None,
            mel_spec_type_value="vocos",
            target_rms_value=0.1,
            cross_fade_duration_value=0.15,
            nfe_step_value=32,
            cfg_strength_value=2.5,
            layered=True,
            cfg_strength2_value=4.0,
            cfg_schedule_value="square",
            cfg_decay_time_value=0.6,
            sway_sampling_coef_value=-1.0,
            local_speed=1.0,
            fix_duration_value=None,
            sp_type="syllable",
            reverse=False,
            denoise_ref=False,
            loudness_norm=False,
            post_processing=True,
            remove_silence_chunk=False,
            device_name="cuda",
        )
        elapsed = time.monotonic() - started
        peak_vram_mb = _peak_cuda_mb(self.torch)
        if waveform is None or len(waveform) == 0:
            raise VideoGeneratorError(
                "X-Voice returned no audio",
                kind=ErrorKind.INVALID_OUTPUT,
            )
        sf.write(str(output), waveform, int(sample_rate))
        return {
            "audio_path": payload["output_path"],
            "mime_type": "audio/wav",
            "duration_seconds": len(waveform) / int(sample_rate),
            "sample_rate": int(sample_rate),
            "channels": 1,
            "timing_precision": "none",
            "word_timings": [],
            "seed": seed,
            "usage": {
                "elapsed_seconds": elapsed,
                "peak_vram_mb": max(
                    value
                    for value in (self.load_peak_vram_mb, peak_vram_mb)
                    if value is not None
                ),
            },
        }

    def close(self) -> dict[str, Any]:
        removed = 0
        for path in self.temporary_reference_files:
            try:
                path.unlink(missing_ok=True)
                removed += 1
            except OSError:
                pass
        self.temporary_reference_files.clear()
        return {"temporary_reference_files_removed": removed}


class ParakeetWorker:
    def __init__(self, paths: Paths) -> None:
        import nemo.collections.asr as nemo_asr

        model_path = paths.read_model(os.environ["VIDEO_GENERATOR_MODEL_PATH"])
        self.paths = paths
        self.model_path = model_path
        self.model = nemo_asr.models.ASRModel.restore_from(str(model_path)).to("cuda").eval()

    def health(self) -> dict[str, Any]:
        return {"model_path": str(self.model_path), "runtime": "nemo-toolkit"}

    def dispatch(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        if operation != "alignment.align":
            raise ValueError(f"unsupported Parakeet operation: {operation}")
        import torch

        source = self.paths.read_run(payload["audio_path"])
        descriptor, temporary_name = tempfile.mkstemp(prefix="video-generator-align-", suffix=".wav")
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(source),
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    str(temporary),
                ],
                check=True,
            )
            started = time.monotonic()
            with torch.inference_mode():
                hypothesis = self.model.transcribe(
                    audio=[str(temporary)], batch_size=1, timestamps=True, verbose=False
                )[0]
            words = [
                {
                    "text": item["word"],
                    "start_seconds": float(item["start"]),
                    "end_seconds": float(item["end"]),
                    "confidence": None,
                }
                for item in hypothesis.timestamp["word"]
            ]
            return {
                "recognized_words": words,
                "recognized_text": hypothesis.text,
                "usage": {"elapsed_seconds": time.monotonic() - started},
            }
        finally:
            temporary.unlink(missing_ok=True)


class FasterWhisperWorker:
    def __init__(self, paths: Paths) -> None:
        import ctranslate2
        import faster_whisper
        from faster_whisper import WhisperModel

        model_path = paths.read_model(os.environ["VIDEO_GENERATOR_MODEL_PATH"])
        self.paths = paths
        self.model_path = model_path
        self.faster_whisper_version = faster_whisper.__version__
        self.ctranslate2_version = ctranslate2.__version__
        self.device = "cuda"
        self.compute_type = "float16"
        self.supported_compute_types = sorted(ctranslate2.get_supported_compute_types(self.device))
        if self.compute_type not in self.supported_compute_types:
            raise VideoGeneratorError(
                f"CTranslate2 CUDA does not support {self.compute_type} on this GPU",
                kind=ErrorKind.NOT_READY,
            )
        self.model = WhisperModel(
            str(model_path),
            device=self.device,
            compute_type=self.compute_type,
            local_files_only=True,
        )

    def health(self) -> dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "runtime": "faster-whisper",
            "faster_whisper_version": self.faster_whisper_version,
            "ctranslate2_version": self.ctranslate2_version,
            "device": self.device,
            "compute_type": self.compute_type,
            "supported_compute_types": self.supported_compute_types,
        }

    def dispatch(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        if operation != "alignment.align":
            raise ValueError(f"unsupported faster-whisper operation: {operation}")
        source = self.paths.read_run(payload["audio_path"])
        language = str(payload["output_language"])
        if language not in {"en", "fi"}:
            raise ValueError(f"unsupported faster-whisper language: {language}")
        started = time.monotonic()
        segments, _ = self.model.transcribe(
            str(source),
            language=language,
            task="transcribe",
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        recognized_words = []
        recognized_text = []
        for segment in segments:
            recognized_text.append(str(segment.text).strip())
            for word in segment.words or []:
                recognized_words.append(
                    {
                        "text": str(word.word).strip(),
                        "start_seconds": float(word.start),
                        "end_seconds": float(word.end),
                        "confidence": (
                            float(word.probability) if word.probability is not None else None
                        ),
                    }
                )
        return {
            "recognized_words": recognized_words,
            "recognized_text": " ".join(value for value in recognized_text if value),
            "usage": {"elapsed_seconds": time.monotonic() - started},
        }


class FluxWorker:
    def __init__(self, paths: Paths) -> None:
        import torch
        from diffusers import Flux2KleinPipeline

        model_path = paths.read_model(os.environ["VIDEO_GENERATOR_MODEL_PATH"])
        self.paths = paths
        self.model_path = model_path
        self.torch = torch
        _reset_cuda_peak(torch)
        started = time.monotonic()
        self.pipe = Flux2KleinPipeline.from_pretrained(
            str(model_path), torch_dtype=torch.bfloat16, local_files_only=True
        )
        if os.environ.get("VIDEO_GENERATOR_FLUX_CPU_OFFLOAD", "1") == "1":
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        self.load_elapsed_seconds = time.monotonic() - started
        self.load_peak_vram_mb = _peak_cuda_mb(torch)

    def health(self) -> dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "runtime": "diffusers",
            "load_elapsed_seconds": self.load_elapsed_seconds,
            "load_peak_vram_mb": self.load_peak_vram_mb,
        }

    def dispatch(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        if operation != "image.generate":
            raise ValueError(f"unsupported FLUX operation: {operation}")
        seed = _image_seed(payload)
        generator = self.torch.Generator(device="cuda").manual_seed(int(seed))
        settings = payload.get("settings") or {}
        steps = int(settings.get("inference_steps") or 4)
        guidance_value = settings.get("guidance_scale")
        guidance = float(guidance_value if guidance_value is not None else 1.0)
        prompt, negative_prompt_mode = _positive_prompt_with_exclusions(
            payload["prompt"], payload.get("negative_prompt") or ""
        )
        call: dict[str, Any] = {
            "prompt": prompt,
            "height": int(payload["height"]),
            "width": int(payload["width"]),
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "num_images_per_prompt": 1,
            "generator": generator,
            "output_type": "pil",
        }
        references = payload.get("reference_paths") or []
        if references:
            from PIL import Image

            call["image"] = [Image.open(self.paths.read_run(value)).convert("RGB") for value in references]
        return _save_diffusers_image(
            self,
            payload=payload,
            call=call,
            generation_settings={
                "seed": seed,
                "inference_steps": steps,
                "guidance_scale": guidance,
                "cpu_offload": os.environ.get("VIDEO_GENERATOR_FLUX_CPU_OFFLOAD", "1") == "1",
                "negative_prompt_mode": negative_prompt_mode,
            },
        )


class ZImageWorker:
    def __init__(self, paths: Paths) -> None:
        import torch
        from diffusers import ZImagePipeline

        model_path = paths.read_model(os.environ["VIDEO_GENERATOR_MODEL_PATH"])
        self.paths = paths
        self.model_path = model_path
        self.torch = torch
        _reset_cuda_peak(torch)
        started = time.monotonic()
        self.pipe = ZImagePipeline.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            device_map="cuda",
        )
        self.pipe.set_progress_bar_config(disable=True)
        self.load_elapsed_seconds = time.monotonic() - started
        self.load_peak_vram_mb = _peak_cuda_mb(torch)

    def health(self) -> dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "runtime": "diffusers",
            "load_elapsed_seconds": self.load_elapsed_seconds,
            "load_peak_vram_mb": self.load_peak_vram_mb,
        }

    def dispatch(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        if operation != "image.generate":
            raise ValueError(f"unsupported Z-Image operation: {operation}")
        _reject_image_references(payload, "Z-Image Turbo")
        settings = payload.get("settings") or {}
        steps = int(settings.get("inference_steps") or 9)
        guidance_value = settings.get("guidance_scale")
        guidance = float(guidance_value if guidance_value is not None else 0.0)
        seed = _image_seed(payload)
        prompt, negative_prompt_mode = _positive_prompt_with_exclusions(
            payload["prompt"], payload.get("negative_prompt") or ""
        )
        call = {
            "prompt": prompt,
            "height": int(payload["height"]),
            "width": int(payload["width"]),
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "num_images_per_prompt": 1,
            "generator": self.torch.Generator(device="cuda").manual_seed(seed),
            "output_type": "pil",
        }
        return _save_diffusers_image(
            self,
            payload=payload,
            call=call,
            generation_settings={
                "seed": seed,
                "inference_steps": steps,
                "guidance_scale": guidance,
                "cpu_offload": False,
                "negative_prompt_mode": negative_prompt_mode,
            },
        )


class Ideogram4Worker:
    def __init__(self, paths: Paths) -> None:
        import torch
        from diffusers import Ideogram4Pipeline

        model_path = paths.read_model(os.environ["VIDEO_GENERATOR_MODEL_PATH"])
        self.paths = paths
        self.model_path = model_path
        self.torch = torch
        _reset_cuda_peak(torch)
        started = time.monotonic()
        self.pipe = Ideogram4Pipeline.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            device_map="cuda",
        )
        self.pipe.enable_model_cpu_offload()
        self.pipe.set_progress_bar_config(disable=True)
        self.load_elapsed_seconds = time.monotonic() - started
        self.load_peak_vram_mb = _peak_cuda_mb(torch)

    def health(self) -> dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "runtime": "diffusers-nf4",
            "prompt_format": "ideogram4-json",
            "load_elapsed_seconds": self.load_elapsed_seconds,
            "load_peak_vram_mb": self.load_peak_vram_mb,
        }

    def dispatch(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        if operation != "image.generate":
            raise ValueError(f"unsupported Ideogram 4 operation: {operation}")
        _reject_image_references(payload, "Ideogram 4")
        settings = payload.get("settings") or {}
        steps = int(settings.get("inference_steps") or 20)
        preset, guidance_schedule, mu, std = _ideogram4_sampler_preset(steps)
        seed = _image_seed(payload)
        negative_prompt = payload.get("negative_prompt") or ""
        call = {
            "prompt": compile_ideogram4_prompt(payload["prompt"], negative_prompt),
            "height": int(payload["height"]),
            "width": int(payload["width"]),
            "num_inference_steps": steps,
            "guidance_schedule": guidance_schedule,
            "mu": mu,
            "std": std,
            "num_images_per_prompt": 1,
            "generator": self.torch.Generator(device="cuda").manual_seed(seed),
            "output_type": "pil",
        }
        return _save_diffusers_image(
            self,
            payload=payload,
            call=call,
            generation_settings={
                "seed": seed,
                "inference_steps": steps,
                "sampler_preset": preset,
                "guidance_schedule": guidance_schedule,
                "mu": mu,
                "std": std,
                "cpu_offload": True,
                "prompt_format": "ideogram4-json",
                "negative_prompt_mode": (
                    "ideogram_positive_constraints" if negative_prompt.strip() else "none"
                ),
            },
            image_validator=_reject_ideogram_safety_placeholder,
        )


class QwenImageWorker:
    def __init__(self, paths: Paths) -> None:
        import torch
        from diffusers import QwenImagePipeline
        from diffusers.quantizers import PipelineQuantizationConfig

        model_path = paths.read_model(os.environ["VIDEO_GENERATOR_MODEL_PATH"])
        self.paths = paths
        self.model_path = model_path
        self.torch = torch
        quantization_config = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16,
            },
            components_to_quantize=["transformer", "text_encoder"],
        )
        _reset_cuda_peak(torch)
        started = time.monotonic()
        self.pipe = QwenImagePipeline.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            local_files_only=True,
            device_map="cuda",
        )
        self.pipe.enable_model_cpu_offload()
        self.pipe.set_progress_bar_config(disable=True)
        self.load_elapsed_seconds = time.monotonic() - started
        self.load_peak_vram_mb = _peak_cuda_mb(torch)

    def health(self) -> dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "runtime": "diffusers-nf4",
            "load_elapsed_seconds": self.load_elapsed_seconds,
            "load_peak_vram_mb": self.load_peak_vram_mb,
        }

    def dispatch(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        if operation != "image.generate":
            raise ValueError(f"unsupported Qwen-Image operation: {operation}")
        _reject_image_references(payload, "Qwen-Image-2512")
        settings = payload.get("settings") or {}
        steps = int(settings.get("inference_steps") or 35)
        true_cfg_value = settings.get("guidance_scale")
        true_cfg_scale = float(true_cfg_value if true_cfg_value is not None else 4.0)
        seed = _image_seed(payload)
        call = {
            "prompt": payload["prompt"],
            "negative_prompt": payload.get("negative_prompt") or "",
            "height": int(payload["height"]),
            "width": int(payload["width"]),
            "num_inference_steps": steps,
            "true_cfg_scale": true_cfg_scale,
            "num_images_per_prompt": 1,
            "generator": self.torch.Generator(device="cuda").manual_seed(seed),
            "output_type": "pil",
        }
        return _save_diffusers_image(
            self,
            payload=payload,
            call=call,
            generation_settings={
                "seed": seed,
                "inference_steps": steps,
                "true_cfg_scale": true_cfg_scale,
                "cpu_offload": True,
                "quantization": "nf4",
                "negative_prompt_mode": "native",
            },
        )


class AceStepWorker:
    def __init__(self, paths: Paths) -> None:
        from acestep.handler import AceStepHandler

        self.paths = paths
        self.project = Path(os.environ["VIDEO_GENERATOR_ACESTEP_PROJECT_ROOT"]).resolve()
        self.project.relative_to(paths.project_root / ".cache" / "runtimes")
        self.handler = AceStepHandler()
        message, ok = self.handler.initialize_service(
            project_root=str(self.project),
            config_path="acestep-v15-xl-turbo",
            device="cuda",
            use_flash_attention=False,
            compile_model=False,
            offload_to_cpu=False,
            offload_dit_to_cpu=False,
            quantization=None,
        )
        if not ok:
            raise VideoGeneratorError(message, kind=ErrorKind.NOT_READY)

    def health(self) -> dict[str, Any]:
        return {"project_root": str(self.project), "runtime": "ACE-Step-1.5"}

    def dispatch(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        if operation != "music.generate":
            raise ValueError(f"unsupported ACE-Step operation: {operation}")
        from acestep.inference import GenerationConfig, GenerationParams, generate_music

        output = self.paths.output_run(payload["output_path"])
        brief = payload["brief"]
        seed = int.from_bytes(os.urandom(4), "big")
        params = GenerationParams(
            task_type="text2music",
            caption=brief["prompt"],
            lyrics="[Instrumental]",
            instrumental=True,
            duration=float(brief["requested_duration_seconds"]),
            inference_steps=8,
            shift=3.0,
            infer_method="ode",
            seed=seed,
            thinking=False,
            use_cot_metas=False,
            use_cot_caption=False,
            use_cot_language=False,
            use_cot_lyrics=False,
        )
        config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[seed], audio_format="wav")
        started = time.monotonic()
        result = generate_music(self.handler, None, params, config, save_dir=str(output.parent))
        if not result.success or not result.audios:
            raise VideoGeneratorError(
                result.error or result.status_message or "ACE-Step generation failed",
                kind=ErrorKind.INVALID_OUTPUT,
            )
        audio = result.audios[0]
        if not isinstance(audio, dict):
            raise VideoGeneratorError(
                "ACE-Step returned an unsupported audio result shape",
                kind=ErrorKind.INVALID_OUTPUT,
            )
        generated = Path(audio["path"]).resolve()
        generated.relative_to(output.parent.resolve())
        if generated != output:
            replace_path(generated, output)
        sample_rate = int(audio["sample_rate"])
        tensor = audio["tensor"]
        sample_count = int(tensor.shape[-1])
        channels = int(tensor.shape[-2]) if tensor.ndim > 1 else 1
        return {
            "audio_path": payload["output_path"],
            "mime_type": "audio/wav",
            "duration_seconds": sample_count / sample_rate,
            "sample_rate": sample_rate,
            "channels": channels,
            "seed": seed,
            "usage": {"elapsed_seconds": time.monotonic() - started},
        }


def build_worker(kind: str, paths: Paths) -> Worker:
    if kind == "llama-server":
        return LlamaServerWorker(paths)
    if kind == "voxcpm":
        return VoxCPMWorker(paths)
    if kind == "omnivoice":
        return OmniVoiceWorker(paths)
    if kind == "moss-tts":
        return MossTtsWorker(paths)
    if kind == "xvoice":
        return XVoiceWorker(paths)
    if kind == "higgs-docker":
        from .higgs_docker import HiggsDockerWorker

        return HiggsDockerWorker(paths)
    if kind == "parakeet":
        return ParakeetWorker(paths)
    if kind == "faster-whisper":
        return FasterWhisperWorker(paths)
    if kind == "flux":
        return FluxWorker(paths)
    if kind == "z-image":
        return ZImageWorker(paths)
    if kind == "ideogram4":
        return Ideogram4Worker(paths)
    if kind == "qwen-image":
        return QwenImageWorker(paths)
    if kind == "acestep":
        return AceStepWorker(paths)
    raise ValueError(f"unknown worker kind: {kind}")


def error_kind(error: BaseException) -> str:
    if isinstance(error, VideoGeneratorError):
        return error.kind.value
    name = type(error).__name__.lower()
    message = str(error).lower()
    if "outofmemory" in name or "out of memory" in message:
        return "not_ready"
    if "llama-server exited during startup" in message or "did not become healthy" in message:
        return "not_ready"
    if any(marker in message for marker in ("cublas", "cudnn", "cuda driver", "cuda runtime")):
        return "not_ready"
    if isinstance(error, (FileNotFoundError, ImportError, ModuleNotFoundError)):
        return "not_ready"
    if isinstance(error, (ValueError, KeyError, json.JSONDecodeError)):
        return "invalid_output"
    return "internal"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kind",
        required=True,
        choices=[
            "llama-server",
            "voxcpm",
            "omnivoice",
            "moss-tts",
            "xvoice",
            "higgs-docker",
            "parakeet",
            "faster-whisper",
            "flux",
            "z-image",
            "ideogram4",
            "qwen-image",
            "acestep",
        ],
    )
    args = parser.parse_args(argv)
    paths = Paths()
    worker: Worker | None = None
    startup_error: BaseException | None = None
    try:
        with contextlib.redirect_stdout(sys.stderr):
            worker = build_worker(args.kind, paths)
    except BaseException as exc:
        startup_error = exc
    closed = False
    try:
        for line in sys.stdin:
            request_id = ""
            operation = ""
            try:
                envelope = json.loads(line)
                request_id = str(envelope.get("request_id") or "")
                if envelope.get("protocol_version") != 1:
                    raise ValueError("worker protocol version mismatch")
                operation = str(envelope.get("operation") or "")
                with contextlib.redirect_stdout(sys.stderr):
                    if operation == "shutdown":
                        close = getattr(worker, "close", None) if worker is not None else None
                        lifecycle = close() if callable(close) else {}
                        closed = True
                        result = {"stopped": True, "lifecycle": lifecycle}
                    else:
                        if startup_error is not None:
                            raise startup_error
                        if worker is None:
                            raise VideoGeneratorError(
                                "worker did not initialize",
                                kind=ErrorKind.NOT_READY,
                            )
                        result = (
                            worker.health()
                            if operation == "health"
                            else worker.dispatch(operation, envelope.get("payload") or {})
                        )
                response = {
                    "protocol_version": 1,
                    "request_id": request_id,
                    "ok": True,
                    "result": result,
                }
            except BaseException as exc:
                error = {"kind": error_kind(exc), "message": str(exc)}
                if isinstance(exc, VideoGeneratorError):
                    if exc.action:
                        error["action"] = exc.action
                    if exc.details:
                        error["details"] = exc.details
                response = {
                    "protocol_version": 1,
                    "request_id": request_id,
                    "ok": False,
                    "error": error,
                }
            print(json.dumps(response, ensure_ascii=False), flush=True)
            if operation == "shutdown":
                return 0 if response["ok"] else 1
        return 0
    finally:
        if not closed:
            close = getattr(worker, "close", None) if worker is not None else None
            if callable(close):
                with contextlib.suppress(BaseException), contextlib.redirect_stdout(sys.stderr):
                    close()


if __name__ == "__main__":
    raise SystemExit(main())
