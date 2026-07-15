from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from video_generator.workers.main import MossTtsWorker, OmniVoiceWorker, XVoiceWorker


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

    @staticmethod
    def manual_seed_all(seed: int) -> None:
        return None


def _fake_torch() -> ModuleType:
    torch = ModuleType("torch")
    torch.float16 = "fp16"
    torch.bfloat16 = "bf16"
    torch.cuda = _FakeCuda()
    torch.manual_seed = lambda seed: None
    torch.inference_mode = nullcontext
    torch.backends = SimpleNamespace(
        cuda=SimpleNamespace(enable_cudnn_sdp=lambda enabled: None)
    )
    return torch


def _voice_paths(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    model = tmp_path / ".cache" / "models" / "model"
    codec = tmp_path / ".cache" / "models" / "codec"
    reference = tmp_path / "private" / "voice.wav"
    transcript = tmp_path / "private" / "voice.txt"
    model.mkdir(parents=True)
    codec.mkdir(parents=True)
    reference.parent.mkdir(parents=True)
    reference.write_bytes(b"wav")
    transcript.write_text("This is the authorized reference.", encoding="utf-8")
    return model, codec, reference, transcript


@pytest.mark.parametrize("language", ["en", "fi"])
def test_omnivoice_uses_local_reference_and_language(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    language: str,
) -> None:
    calls: dict[str, object] = {}
    model_path, _, reference, transcript = _voice_paths(tmp_path)
    output = tmp_path / "runs" / "fixture" / "speech.wav"
    output.parent.mkdir(parents=True)

    class Model:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs):
            calls["load"] = (path, kwargs)
            return cls()

        def create_voice_clone_prompt(self, **kwargs):
            calls.setdefault("clone_prompts", []).append(kwargs)
            return "reusable-clone-prompt"

        def generate(self, **kwargs):
            calls.setdefault("generate", []).append(kwargs)
            return [[0.0] * 24000]

    omnivoice = ModuleType("omnivoice")
    omnivoice.OmniVoice = Model
    soundfile = ModuleType("soundfile")

    def write(path: str, waveform, sample_rate: int) -> None:
        calls["write"] = (path, sample_rate, len(waveform))
        Path(path).write_bytes(b"wav")

    soundfile.write = write
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setitem(sys.modules, "omnivoice", omnivoice)
    monkeypatch.setitem(sys.modules, "soundfile", soundfile)
    monkeypatch.setenv("VIDEO_GENERATOR_MODEL_PATH", ".cache/models/model")
    paths = SimpleNamespace(
        read_model=lambda value: model_path,
        read_private=lambda value: transcript if value.endswith(".txt") else reference,
        output_run=lambda value: output,
    )

    worker = OmniVoiceWorker(paths)
    payload = {
        "scene_id": "scene-001",
        "text": "Hello world" if language == "en" else "Hei maailma",
        "output_language": language,
        "output_path": "runs/fixture/speech.wav",
        "voice": {
            "reference_audio": "private/voice.wav",
            "reference_transcript": "private/voice.txt",
        },
    }
    result = worker.dispatch(
        "speech.synthesize",
        payload,
    )
    worker.dispatch("speech.synthesize", {**payload, "scene_id": "scene-002"})

    assert calls["load"] == (
        str(model_path),
        {"device_map": "cuda:0", "dtype": "fp16", "load_asr": False},
    )
    assert calls["clone_prompts"] == [
        {
            "ref_audio": str(reference),
            "ref_text": "This is the authorized reference.",
        }
    ]
    assert len(calls["generate"]) == 2
    generate = calls["generate"][0]
    assert generate["voice_clone_prompt"] == "reusable-clone-prompt"
    assert "ref_audio" not in generate
    assert "ref_text" not in generate
    assert generate["language"] == language
    assert "language_id" not in generate
    assert generate["speed"] == 1.0
    assert result["sample_rate"] == 24000
    assert result["channels"] == 1
    assert result["duration_seconds"] == 1.0


class _FakeWaveform:
    ndim = 2
    shape = (2, 96000)

    def detach(self) -> "_FakeWaveform":
        return self

    def cpu(self) -> "_FakeWaveform":
        return self


@pytest.mark.parametrize(("language_id", "language"), [("en", "English"), ("fi", "Finnish")])
def test_moss_tts_uses_pinned_local_codec_and_sdpa(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    language_id: str,
    language: str,
) -> None:
    calls: dict[str, object] = {}
    model_path, codec_path, reference, _ = _voice_paths(tmp_path)
    output = tmp_path / "runs" / "fixture" / "speech.wav"
    output.parent.mkdir(parents=True)

    class Batch(dict):
        def to(self, device: str):
            calls["batch_device"] = device
            return self

    class AudioTokenizer:
        def to(self, device: str):
            calls["codec_device"] = device
            return self

    class Processor:
        def __init__(self) -> None:
            self.audio_tokenizer = AudioTokenizer()

        @classmethod
        def from_pretrained(cls, path: str, **kwargs):
            calls["processor_load"] = (path, kwargs)
            return cls()

        def build_user_message(self, **kwargs):
            calls["message"] = kwargs
            return "message"

        def __call__(self, conversation, *, mode: str):
            calls["processor_call"] = (conversation, mode)
            return Batch(input_ids="ids", attention_mask="mask")

        def decode(self, outputs):
            calls["decode"] = outputs
            return [SimpleNamespace(audio_codes_list=[_FakeWaveform()])]

    class Model:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs):
            calls["model_load"] = (path, kwargs)
            return cls()

        def to(self, device: str):
            calls["model_device"] = device
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            calls["generate"] = kwargs
            return "outputs"

    transformers = ModuleType("transformers")
    transformers.AutoProcessor = Processor
    transformers.AutoModel = Model
    torchaudio = ModuleType("torchaudio")

    def save(path: str, waveform, sample_rate: int) -> None:
        calls["save"] = (path, waveform, sample_rate)
        Path(path).write_bytes(b"wav")

    torchaudio.save = save
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "torchaudio", torchaudio)
    monkeypatch.setenv("VIDEO_GENERATOR_MODEL_PATH", ".cache/models/model")
    monkeypatch.setenv("VIDEO_GENERATOR_CODEC_PATH", ".cache/models/codec")
    paths = SimpleNamespace(
        read_model=lambda value: codec_path if value.endswith("codec") else model_path,
        read_private=lambda value: reference,
        output_run=lambda value: output,
    )

    worker = MossTtsWorker(paths)
    result = worker.dispatch(
        "speech.synthesize",
        {
            "scene_id": "scene-001",
            "text": "Hello world" if language_id == "en" else "Hei maailma",
            "output_language": language_id,
            "output_path": "runs/fixture/speech.wav",
            "voice": {"reference_audio": "private/voice.wav"},
        },
    )

    processor_options = calls["processor_load"][1]
    assert processor_options["codec_path"] == str(codec_path)
    assert processor_options["trust_remote_code"] is True
    assert "local_files_only" not in processor_options
    assert processor_options["codec_attention_implementation"] == "sdpa"
    model_options = calls["model_load"][1]
    assert model_options["attn_implementation"] == "sdpa"
    assert model_options["local_files_only"] is True
    assert calls["message"] == {
        "text": "Hello world" if language_id == "en" else "Hei maailma",
        "reference": [str(reference)],
        "language": language,
    }
    generate = calls["generate"]
    assert generate["audio_temperature"] == 1.7
    assert generate["audio_top_p"] == 0.8
    assert result["sample_rate"] == 48000
    assert result["channels"] == 2
    assert result["duration_seconds"] == 2.0


def test_xvoice_uses_explicit_reference_and_output_languages_and_caches_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    model_path, vocoder_path, reference, transcript = _voice_paths(tmp_path)
    source = tmp_path / ".cache" / "runtimes" / "local--x-voice" / "source"
    config_path = source / "src" / "x_voice" / "configs" / "XVoice_Base_Stage1.yaml"
    checkpoint = model_path / "XVoice_Base_Stage1" / "model_600000.safetensors"
    vocab = model_path / "XVoice_Base_Stage1" / "vocab.txt"
    config_path.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    config_path.write_text("fixture", encoding="utf-8")
    checkpoint.write_bytes(b"weights")
    vocab.write_text("fixture", encoding="utf-8")
    output = tmp_path / "runs" / "fixture" / "speech.wav"
    output.parent.mkdir(parents=True)

    class Node(dict):
        __getattr__ = dict.__getitem__

    model_config = Node(
        model=Node(
            backbone="DiT",
            tokenizer="ipa_v6",
            arch=Node(dim=1024),
            mel_spec=Node(target_sample_rate=24000),
        ),
        datasets=Node(name="XVoice_Dataset"),
    )

    class OmegaConf:
        @staticmethod
        def load(path):
            calls["config"] = path
            return model_config

        @staticmethod
        def to_container(value, resolve=True):
            return dict(value)

    class Model:
        transformer = SimpleNamespace(lang_to_id={"en": 0, "fi": 1})

    def preprocess(audio, text, show_info):
        calls.setdefault("preprocess", []).append((audio, text))
        return audio, text + ". "

    def infer(*args, **kwargs):
        calls.setdefault("infer", []).append((args, kwargs))
        return [0.0] * 48000, 24000, None

    utils = ModuleType("x_voice.infer.utils_infer")
    utils.get_ipa_tokenizer_cache = lambda tokenizer, stress: "ipa-cache"
    utils.infer_xvoice_process = infer
    utils.load_model = lambda *args, **kwargs: Model()
    utils.load_vocoder = lambda *args, **kwargs: "vocoder"
    utils.preprocess_ref_audio_text = preprocess
    hydra = ModuleType("hydra")
    hydra_utils = ModuleType("hydra.utils")
    hydra_utils.get_class = lambda value: "DiT-class"
    omegaconf = ModuleType("omegaconf")
    omegaconf.OmegaConf = OmegaConf
    soundfile = ModuleType("soundfile")

    def write(path, waveform, sample_rate):
        calls.setdefault("write", []).append((path, len(waveform), sample_rate))
        Path(path).write_bytes(b"wav")

    soundfile.write = write
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setitem(sys.modules, "hydra", hydra)
    monkeypatch.setitem(sys.modules, "hydra.utils", hydra_utils)
    monkeypatch.setitem(sys.modules, "omegaconf", omegaconf)
    monkeypatch.setitem(sys.modules, "x_voice", ModuleType("x_voice"))
    monkeypatch.setitem(sys.modules, "x_voice.infer", ModuleType("x_voice.infer"))
    monkeypatch.setitem(sys.modules, "x_voice.infer.utils_infer", utils)
    monkeypatch.setitem(sys.modules, "soundfile", soundfile)
    monkeypatch.setenv("VIDEO_GENERATOR_MODEL_PATH", ".cache/models/model")
    monkeypatch.setenv("VIDEO_GENERATOR_XVOICE_VOCODER_PATH", ".cache/models/codec")
    monkeypatch.setenv(
        "VIDEO_GENERATOR_XVOICE_SOURCE_PATH",
        ".cache/runtimes/local--x-voice/source",
    )
    monkeypatch.setenv("VIDEO_GENERATOR_RUNTIME_REVISION", "runtime-fixture-v1")
    paths = SimpleNamespace(
        read_model=lambda value: vocoder_path if value.endswith("codec") else model_path,
        read_runtime=lambda value: source,
        read_private=lambda value: transcript if value.endswith(".txt") else reference,
        output_run=lambda value: output,
    )
    worker = XVoiceWorker(paths)
    base_payload = {
        "scene_id": "scene-001",
        "text": "Tämä on X-Voice-testi.",
        "output_language": "fi",
        "output_path": "runs/fixture/speech.wav",
        "voice": {
            "reference_audio": "private/voice.wav",
            "reference_transcript": "private/voice.txt",
            "reference_language": "en",
        },
    }

    result = worker.dispatch("speech.synthesize", base_payload)
    worker.dispatch(
        "speech.synthesize",
        {**base_payload, "scene_id": "scene-002", "output_language": "en"},
    )

    assert calls["preprocess"] == [
        (str(reference), "This is the authorized reference.")
    ]
    assert len(calls["infer"]) == 2
    first_args, first_options = calls["infer"][0]
    assert first_args[3:5] == ("en", "fi")
    assert first_options["dominant_lang"] == "fi"
    assert first_options["nfe_step_value"] == 32
    assert first_options["cfg_strength_value"] == 2.5
    assert first_options["layered"] is True
    assert first_options["sp_type"] == "syllable"
    assert first_options["post_processing"] is True
    health = worker.health()
    assert health["runtime_revision"] == "runtime-fixture-v1"
    assert "source_revision" not in health
    assert result["sample_rate"] == 24000
    assert result["channels"] == 1
    assert result["duration_seconds"] == 2.0
