from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Protocol

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
            raise ValueError("this Qwen runner is text-only; use an evaluated vision runner")
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
        response = self.session.chat_completion(
            {
                "messages": messages,
                "max_tokens": int(payload.get("max_output_tokens", 8000)),
                "temperature": (
                    0.2
                    if payload["task_id"] in {"script_draft", "script_revision", "duration_repair"}
                    else 0.65
                ),
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
            raise ValueError("llama-server response did not contain assistant content") from exc
        finish_reason = str(choice.get("finish_reason") or "")
        if finish_reason != "stop":
            raise ValueError(
                "llama-server response did not finish normally: "
                + (finish_reason or "missing finish_reason")
            )
        if not isinstance(content, str):
            raise ValueError("llama-server assistant content was not text")
        content = re.sub(r"^.*?</think>\s*", "", content, flags=re.DOTALL)
        data = json.loads(content)
        usage = response.get("usage") or {}
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
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
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
            raise RuntimeError(
                f"CTranslate2 CUDA does not support {self.compute_type} on this GPU"
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
        self.pipe = Flux2KleinPipeline.from_pretrained(
            str(model_path), torch_dtype=torch.bfloat16, local_files_only=True
        )
        if os.environ.get("VIDEO_GENERATOR_FLUX_CPU_OFFLOAD", "1") == "1":
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to("cuda")
        self.pipe.set_progress_bar_config(disable=True)

    def health(self) -> dict[str, Any]:
        return {"model_path": str(self.model_path), "runtime": "diffusers"}

    def dispatch(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        if operation != "image.generate":
            raise ValueError(f"unsupported FLUX operation: {operation}")
        output = self.paths.output_run(payload["output_path"])
        seed = payload.get("seed")
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        generator = self.torch.Generator(device="cuda").manual_seed(int(seed))
        settings = payload.get("settings") or {}
        steps = int(settings.get("inference_steps") or 4)
        guidance = float(settings.get("guidance_scale") or 1.0)
        call: dict[str, Any] = {
            "prompt": payload["prompt"],
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
        started = time.monotonic()
        with self.torch.inference_mode():
            result = self.pipe(**call)
        image = result.images[0]
        image.save(output, format="PNG")
        return {
            "image_path": payload["output_path"],
            "mime_type": "image/png",
            "generation_settings": {
                "seed": seed,
                "inference_steps": steps,
                "guidance_scale": guidance,
                "cpu_offload": os.environ.get("VIDEO_GENERATOR_FLUX_CPU_OFFLOAD", "1") == "1",
            },
            "usage": {"elapsed_seconds": time.monotonic() - started},
        }


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
            raise RuntimeError(message)

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
            raise RuntimeError(result.error or result.status_message or "ACE-Step generation failed")
        audio = result.audios[0]
        if not isinstance(audio, dict):
            raise RuntimeError("ACE-Step returned an unsupported audio result shape")
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
    if kind == "parakeet":
        return ParakeetWorker(paths)
    if kind == "faster-whisper":
        return FasterWhisperWorker(paths)
    if kind == "flux":
        return FluxWorker(paths)
    if kind == "acestep":
        return AceStepWorker(paths)
    raise ValueError(f"unknown worker kind: {kind}")


def error_kind(error: BaseException) -> str:
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
        choices=["llama-server", "voxcpm", "parakeet", "faster-whisper", "flux", "acestep"],
    )
    args = parser.parse_args(argv)
    paths = Paths()
    with contextlib.redirect_stdout(sys.stderr):
        worker = build_worker(args.kind, paths)
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
                        close = getattr(worker, "close", None)
                        lifecycle = close() if callable(close) else {}
                        closed = True
                        result = {"stopped": True, "lifecycle": lifecycle}
                    else:
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
                response = {
                    "protocol_version": 1,
                    "request_id": request_id,
                    "ok": False,
                    "error": {"kind": error_kind(exc), "message": str(exc)},
                }
            print(json.dumps(response, ensure_ascii=False), flush=True)
            if operation == "shutdown":
                return 0 if response["ok"] else 1
        return 0
    finally:
        if not closed:
            close = getattr(worker, "close", None)
            if callable(close):
                with contextlib.suppress(BaseException), contextlib.redirect_stdout(sys.stderr):
                    close()


if __name__ == "__main__":
    raise SystemExit(main())
