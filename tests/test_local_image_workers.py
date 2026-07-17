from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from video_generator.errors import ErrorKind, VideoGeneratorError
from video_generator.workers.main import (
    Ideogram4Worker,
    QwenImageWorker,
    ZImageWorker,
    _ideogram4_sampler_preset,
    _reject_ideogram_safety_placeholder,
    compile_ideogram4_prompt,
)


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def reset_peak_memory_stats() -> None:
        return None

    @staticmethod
    def synchronize() -> None:
        return None

    @staticmethod
    def max_memory_allocated() -> int:
        return 128 * 1024 * 1024

    @staticmethod
    def max_memory_reserved() -> int:
        return 256 * 1024 * 1024


class _FakeGenerator:
    def __init__(self, device: str) -> None:
        self.device = device
        self.seed = None

    def manual_seed(self, seed: int) -> "_FakeGenerator":
        self.seed = seed
        return self


class _FakeImage:
    def save(self, path: Path, *, format: str) -> None:
        assert format == "PNG"
        Path(path).write_bytes(b"png")


def _fake_torch() -> ModuleType:
    torch = ModuleType("torch")
    torch.bfloat16 = "bf16"
    torch.cuda = _FakeCuda()
    torch.Generator = _FakeGenerator
    torch.inference_mode = nullcontext
    return torch


def _paths(tmp_path: Path) -> SimpleNamespace:
    model = tmp_path / ".cache" / "models" / "fixture"
    model.mkdir(parents=True)
    output = tmp_path / "runs" / "fixture" / "image.png"
    output.parent.mkdir(parents=True)
    return SimpleNamespace(
        read_model=lambda value: model,
        output_run=lambda value: output,
    )


def _payload(**updates) -> dict[str, object]:
    value: dict[str, object] = {
        "prompt": "A tiny orange fox carrying an amber lantern through snow.",
        "negative_prompt": "text, logo, watermark",
        "height": 576,
        "width": 1024,
        "seed": 42,
        "reference_paths": [],
        "settings": {},
        "output_path": "runs/fixture/image.png",
    }
    value.update(updates)
    return value


def test_z_image_uses_turbo_settings_and_positive_exclusions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, object] = {}

    class Pipeline:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs):
            calls["load"] = (path, kwargs)
            return cls()

        def set_progress_bar_config(self, **kwargs) -> None:
            calls["progress"] = kwargs

        def __call__(self, **kwargs):
            calls["generate"] = kwargs
            return SimpleNamespace(images=[_FakeImage()])

    diffusers = ModuleType("diffusers")
    diffusers.ZImagePipeline = Pipeline
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setenv("VIDEO_GENERATOR_MODEL_PATH", ".cache/models/fixture")

    worker = ZImageWorker(_paths(tmp_path))
    result = worker.dispatch(
        "image.generate",
        _payload(settings={"inference_steps": 9, "guidance_scale": 0.0}),
    )

    load_options = calls["load"][1]
    assert load_options["local_files_only"] is True
    assert load_options["device_map"] == "cuda"
    generate = calls["generate"]
    assert generate["num_inference_steps"] == 9
    assert generate["guidance_scale"] == 0.0
    assert "negative_prompt" not in generate
    assert "text, logo, watermark" in generate["prompt"]
    assert result["generation_settings"]["negative_prompt_mode"] == "positive_avoid_clause"


def test_ideogram_prompt_is_compact_strict_json() -> None:
    compiled = compile_ideogram4_prompt(
        "A Finnish lakeside cabin in clear hyvää päivää weather.",
        "watermark",
    )
    value = json.loads(compiled)

    assert list(value) == [
        "high_level_description",
        "style_description",
        "compositional_deconstruction",
    ]
    assert list(value["style_description"]) == [
        "aesthetics",
        "lighting",
        "medium",
        "art_style",
    ]
    assert list(value["compositional_deconstruction"]) == ["background", "elements"]
    assert list(value["compositional_deconstruction"]["elements"][0]) == ["type", "desc"]
    assert value["compositional_deconstruction"]["elements"][0]["type"] == "obj"
    assert "hyvää päivää" in compiled
    assert "watermark" not in compiled
    assert "clean, unbranded, and contains no added lettering" in compiled
    assert "\n" not in compiled
    assert compiled == json.dumps(value, separators=(",", ":"), ensure_ascii=False)


@pytest.mark.parametrize(
    ("steps", "preset", "polish_steps", "mu", "std"),
    [
        (12, "V4_TURBO_12", 1, 0.5, 1.75),
        (20, "V4_DEFAULT_20", 2, 0.0, 1.75),
        (48, "V4_QUALITY_48", 3, 0.0, 1.5),
    ],
)
def test_ideogram_worker_uses_official_sampler_presets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: int,
    preset: str,
    polish_steps: int,
    mu: float,
    std: float,
) -> None:
    calls: dict[str, object] = {}

    class Pipeline:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs):
            calls["load"] = kwargs
            return cls()

        def enable_model_cpu_offload(self) -> None:
            calls["offload"] = True

        def set_progress_bar_config(self, **kwargs) -> None:
            return None

        def __call__(self, **kwargs):
            calls["generate"] = kwargs
            return SimpleNamespace(images=[_FakeImage()])

    diffusers = ModuleType("diffusers")
    diffusers.Ideogram4Pipeline = Pipeline
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setenv("VIDEO_GENERATOR_MODEL_PATH", ".cache/models/fixture")

    worker = Ideogram4Worker(_paths(tmp_path))
    result = worker.dispatch(
        "image.generate", _payload(settings={"inference_steps": steps})
    )

    generate = calls["generate"]
    assert calls["offload"] is True
    assert len(generate["guidance_schedule"]) == steps
    assert generate["guidance_schedule"][-polish_steps:] == [3.0] * polish_steps
    assert generate["guidance_schedule"][:-polish_steps] == [7.0] * (
        steps - polish_steps
    )
    assert generate["mu"] == mu
    assert generate["std"] == std
    assert "guidance_scale" not in generate
    assert "negative_prompt" not in generate
    assert json.loads(generate["prompt"])["compositional_deconstruction"]
    assert result["generation_settings"]["sampler_preset"] == preset


def test_ideogram_sampler_rejects_noncanonical_step_count() -> None:
    with pytest.raises(ValueError, match="official preset: 12, 20, 48"):
        _ideogram4_sampler_preset(13)


def test_ideogram_safety_placeholder_is_rejected() -> None:
    class SafetyPlaceholder:
        def convert(self, mode: str):
            assert mode == "RGB"
            return self

        def resize(self, size: tuple[int, int]):
            assert size == (64, 36)
            return self

        def get_flattened_data(self):
            pixels = [(112, 112, 112)] * (64 * 36)
            for index in range(900, 940):
                pixels[index] = (165, 165, 165)
            return pixels

    with pytest.raises(VideoGeneratorError, match="built-in safety filter") as captured:
        _reject_ideogram_safety_placeholder(SafetyPlaceholder())
    assert captured.value.kind is ErrorKind.POLICY_REFUSAL


def test_qwen_image_uses_nf4_offload_and_true_cfg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, object] = {}

    class QuantizationConfig:
        def __init__(self, **kwargs) -> None:
            calls["quantization"] = kwargs

    class Pipeline:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs):
            calls["load"] = kwargs
            return cls()

        def enable_model_cpu_offload(self) -> None:
            calls["offload"] = True

        def set_progress_bar_config(self, **kwargs) -> None:
            return None

        def __call__(self, **kwargs):
            calls["generate"] = kwargs
            return SimpleNamespace(images=[_FakeImage()])

    diffusers = ModuleType("diffusers")
    diffusers.__path__ = []
    diffusers.QwenImagePipeline = Pipeline
    quantizers = ModuleType("diffusers.quantizers")
    quantizers.PipelineQuantizationConfig = QuantizationConfig
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.quantizers", quantizers)
    monkeypatch.setenv("VIDEO_GENERATOR_MODEL_PATH", ".cache/models/fixture")

    worker = QwenImageWorker(_paths(tmp_path))
    result = worker.dispatch(
        "image.generate",
        _payload(settings={"guidance_scale": 4.0}),
    )

    quantization = calls["quantization"]
    assert quantization["quant_backend"] == "bitsandbytes_4bit"
    assert quantization["components_to_quantize"] == ["transformer", "text_encoder"]
    assert quantization["quant_kwargs"]["bnb_4bit_quant_type"] == "nf4"
    assert quantization["quant_kwargs"]["llm_int8_skip_modules"] == [
        "time_text_embed",
        "img_in",
        "txt_in",
        "norm_out",
        "proj_out",
    ]
    assert calls["offload"] is True
    generate = calls["generate"]
    assert generate["num_inference_steps"] == 50
    assert generate["true_cfg_scale"] == 4.0
    assert generate["negative_prompt"] == "text, logo, watermark"
    assert "guidance_scale" not in generate
    assert result["generation_settings"]["nf4_skip_modules"] == [
        "time_text_embed",
        "img_in",
        "txt_in",
        "norm_out",
        "proj_out",
    ]


@pytest.mark.parametrize("worker_type", [ZImageWorker, Ideogram4Worker, QwenImageWorker])
def test_text_to_image_workers_reject_references(worker_type) -> None:
    worker = object.__new__(worker_type)

    with pytest.raises(ValueError, match="text-to-image only"):
        worker.dispatch("image.generate", _payload(reference_paths=["reference.png"]))
