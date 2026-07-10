from __future__ import annotations

import argparse
import contextlib
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
        messages = [
            {"role": "system", "content": payload["instructions"]},
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
                "temperature": 0.65,
                "top_p": 0.9,
                "stream": False,
                "response_format": {
                    "type": "json_schema",
                    "schema": payload["output_schema"],
                },
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("llama-server response did not contain assistant content") from exc
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

        voice = payload["voice"]
        reference = self.paths.read_private(voice["reference_audio"])
        transcript = ""
        if voice.get("reference_transcript"):
            transcript = self.paths.read_private(voice["reference_transcript"]).read_text(encoding="utf-8").strip()
        output = self.paths.output_run(payload["output_path"])
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
        choices=["llama-server", "voxcpm", "parakeet", "flux", "acestep"],
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
