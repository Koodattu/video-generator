from __future__ import annotations

import io
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from video_generator import setup
from video_generator.errors import VideoGeneratorError
from video_generator.profiles import BACKEND_DESCRIPTORS, PROFILES
from video_generator.runners import RunnerManager, decode_wsl_output
from video_generator.util import sha256_file
from video_generator.workers.prepare import write_asset_manifest


def test_decode_wsl_output_accepts_redirected_utf16le() -> None:
    raw = "Ubuntu\r\nDebian\r\n".encode("utf-16-le")

    assert decode_wsl_output(raw).splitlines() == ["Ubuntu", "Debian"]


def test_ace_setup_removes_only_untracked_generated_build_tree(tmp_path: Path) -> None:
    generated = (
        tmp_path
        / "acestep"
        / "third_parts"
        / "nano-vllm"
        / "build"
        / "lib"
        / "nanovllm"
        / "__init__.py"
    )
    generated.parent.mkdir(parents=True)
    generated.write_text("VALUE = 1\n", encoding="utf-8")

    setup._remove_untracked_ace_build_tree(tmp_path, ["acestep/handler.py"])

    assert not (tmp_path / "acestep" / "third_parts" / "nano-vllm" / "build").exists()


def test_ace_setup_refuses_to_remove_tracked_build_tree(tmp_path: Path) -> None:
    generated = tmp_path / "acestep" / "third_parts" / "nano-vllm" / "build" / "tracked.py"
    generated.parent.mkdir(parents=True)
    generated.write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(VideoGeneratorError, match="Git-tracked"):
        setup._remove_untracked_ace_build_tree(
            tmp_path,
            ["acestep/third_parts/nano-vllm/build/tracked.py"],
        )

    assert generated.is_file()


def test_ace_setup_rejects_build_tree_symlink_before_resolving(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build = tmp_path / "acestep" / "third_parts" / "nano-vllm" / "build"
    build.mkdir(parents=True)
    original_is_symlink = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda path: path == build or original_is_symlink(path),
    )

    with pytest.raises(VideoGeneratorError, match="symbolic link"):
        setup._remove_untracked_ace_build_tree(tmp_path, [])

    assert build.is_dir()


def test_ace_setup_syncs_only_git_tracked_checkpoint_code(tmp_path: Path) -> None:
    source = tmp_path / "acestep" / "models" / "xl_turbo" / "configuration_acestep_v15.py"
    checkpoint = tmp_path / "checkpoints" / "acestep-v15-xl-turbo"
    destination = checkpoint / source.name
    source.parent.mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    source.write_text("PINNED = True\n", encoding="utf-8")
    destination.write_text("PINNED = False\n", encoding="utf-8")

    synced = setup._sync_tracked_ace_checkpoint_code(
        tmp_path,
        checkpoint,
        ["acestep/models/xl_turbo/configuration_acestep_v15.py"],
    )

    assert destination.read_text(encoding="utf-8") == "PINNED = True\n"
    assert synced == [destination.resolve()]


def test_ace_setup_rejects_untracked_checkpoint_sync_source(tmp_path: Path) -> None:
    source = tmp_path / "acestep" / "models" / "xl_turbo" / "configuration_acestep_v15.py"
    checkpoint = tmp_path / "checkpoints" / "acestep-v15-xl-turbo"
    source.parent.mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    source.write_text("INJECTED = True\n", encoding="utf-8")

    with pytest.raises(VideoGeneratorError, match="not Git-tracked"):
        setup._sync_tracked_ace_checkpoint_code(tmp_path, checkpoint, [])


def test_ace_setup_refreshes_nested_manifest_for_synced_code(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    synced = checkpoint / "configuration_acestep_v15.py"
    synced.write_text("PINNED = True\n", encoding="utf-8")
    (checkpoint / "asset-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "revision": "pinned-revision",
                "files": [
                    {
                        "path": synced.name,
                        "size": 1,
                        "sha256": "0" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    setup._refresh_ace_model_asset_manifest(
        checkpoint,
        [synced],
        expected_revision="pinned-revision",
    )

    manifest = json.loads((checkpoint / "asset-manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"][0]["size"] == synced.stat().st_size
    assert manifest["files"][0]["sha256"] == sha256_file(synced)


@pytest.mark.parametrize(
    ("backend_id", "expected_torch_backend"),
    [
        ("local:flux.2-klein-4b", "auto"),
        ("local:omnivoice", "auto"),
        ("local:moss-tts-v1.5", "cu128"),
        ("local:x-voice", "cu128"),
        ("local:z-image-turbo", "auto"),
        ("local:ideogram-4-nf4", "cu130"),
        ("local:qwen-image-2512-nf4", "cu130"),
    ],
)
def test_native_cuda_runner_uses_selected_torch_backend(
    tmp_path: Path,
    monkeypatch,
    backend_id: str,
    expected_torch_backend: str,
) -> None:
    commands: list[list[str]] = []
    definition = setup.LOCAL_DEFINITIONS[backend_id]

    def fake_run(command, *, cwd, environment=None, timeout=7200) -> str:
        values = [str(value) for value in command]
        commands.append(values)
        if values[1:] == ["--version"]:
            return "uv 0.6.9"
        if values[1] == "venv":
            python = setup._native_python(
                tmp_path / ".cache" / "runtimes" / setup.runner_slug(definition.backend_id)
            )
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_bytes(b"")
        return ""

    monkeypatch.setattr(setup, "_find_uv", lambda: "uv")
    monkeypatch.setattr(setup, "_run", fake_run)

    setup._install_native_environment(tmp_path, definition)

    sync_command = next(values for values in commands if values[1:3] == ["pip", "sync"])
    assert not any(values[1:3] == ["pip", "compile"] for values in commands)
    assert sync_command[sync_command.index("--torch-backend") + 1] == expected_torch_backend
    assert "--require-hashes" in sync_command
    runtime_lock = (
        tmp_path
        / ".cache"
        / "runtimes"
        / setup.runner_slug(definition.backend_id)
        / "requirements.lock"
    )
    assert runtime_lock.read_bytes() == setup._requirements_path(
        definition.requirements_name
    ).with_suffix(".lock").read_bytes()


def test_parse_uv_version() -> None:
    assert setup._parse_uv_version("uv 0.6.9 (build metadata)") == (0, 6, 9)


def test_local_profile_uses_native_faster_whisper_and_keeps_parakeet_available() -> None:
    backend_id = "local:faster-whisper-large-v3-turbo"
    definition = setup.LOCAL_DEFINITIONS[backend_id]

    assert PROFILES["local"]["caption_alignment"] == backend_id
    assert definition.kind == "faster-whisper"
    assert definition.platform == "native"
    assert definition.python_version == "3.11"
    assert definition.model_repo == "dropbox-dash/faster-whisper-large-v3-turbo"
    assert definition.model_revision == "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"
    assert definition.requirements_name == "faster-whisper.in"
    assert BACKEND_DESCRIPTORS[backend_id].runner == "native"
    assert setup.LOCAL_DEFINITIONS["local:parakeet-tdt-0.6b-v3"].platform == "wsl"
    assert "local:parakeet-tdt-0.6b-v3" not in setup.selected_backends(
        profile="local", backend_id=None
    )


def test_curated_candidates_pin_the_planned_q4_mtp_artifacts() -> None:
    qwen = setup.CURATED_LLM_CANDIDATES["qwen3.6-27b-q4-mtp"]
    gemma = setup.CURATED_LLM_CANDIDATES["gemma-4-26b-a4b-q4-mtp"]
    eurollm = setup.CURATED_LLM_CANDIDATES["eurollm-22b-instruct-2512-q4"]

    assert qwen.quantization == gemma.quantization == "UD-Q4_K_XL"
    assert [artifact.role for artifact in qwen.artifacts] == ["model"]
    assert [artifact.role for artifact in gemma.artifacts] == ["model", "draft-model"]
    assert eurollm.quantization == "Q4_K_M"
    assert eurollm.mtp == "none"
    assert eurollm.artifacts[0].sha256 == (
        "2a222374c4adacd55b55795e2f9dca42a2f100d5a2d5858442f928c4c8bdf5e7"
    )
    assert all(len(candidate.revision) == 40 for candidate in (qwen, gemma, eurollm))
    assert all(
        len(artifact.sha256) == 64
        for candidate in (qwen, gemma, eurollm)
        for artifact in candidate.artifacts
    )


@pytest.mark.parametrize(
    ("backend_id", "kind", "repository", "revision"),
    [
        (
            "local:omnivoice",
            "omnivoice",
            "k2-fsa/OmniVoice",
            "c5fdb5ccb189668d56333f77ba2629f4cd7535f4",
        ),
        (
            "local:moss-tts-v1.5",
            "moss-tts",
            "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
            "be7766a6735b98bd793f7c79fb720b4d0f5d13b8",
        ),
        (
            "local:x-voice",
            "xvoice",
            "XRXRX/X-Voice",
            "7f24fe778ddf7a47e25d87e5d5153599c1d4d5c2",
        ),
        (
            "local:z-image-turbo",
            "z-image",
            "Tongyi-MAI/Z-Image-Turbo",
            "f332072aa78be7aecdf3ee76d5c247082da564a6",
        ),
        (
            "local:ideogram-4-nf4",
            "ideogram4",
            "ideogram-ai/ideogram-4-nf4-diffusers",
            "1874bc70267ba2c823a7239e1d70dd308c8d64dc",
        ),
        (
            "local:qwen-image-2512-nf4",
            "qwen-image",
            "Qwen/Qwen-Image-2512",
            "25468b98e3276ca6700de15c6628e51b7de54a26",
        ),
    ],
)
def test_native_challenger_definitions_are_pinned(
    backend_id: str,
    kind: str,
    repository: str,
    revision: str,
) -> None:
    definition = setup.LOCAL_DEFINITIONS[backend_id]

    assert definition.kind == kind
    assert definition.platform == "native"
    assert definition.model_repo == repository
    assert definition.model_revision == revision
    assert BACKEND_DESCRIPTORS[backend_id].runner == "native"


def test_xvoice_definition_pins_native_source_vocoder_and_runtime_artifacts() -> None:
    definition = setup.LOCAL_DEFINITIONS["local:x-voice"]
    vocoder = definition.supporting_models[0]

    assert setup.XVOICE_SOURCE_REVISION == "b1a5d25459aecdea5dfce6e892da384400ac32e9"
    assert vocoder.model_repo == "charactr/vocos-mel-24khz"
    assert vocoder.model_revision == "0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21"
    assert definition.allow_patterns == (
        "XVoice_Base_Stage1/model_600000.safetensors",
        "XVoice_Base_Stage1/vocab.txt",
        "README.md",
        "LICENSE*",
    )
    assert len(setup.XVOICE_MICROMAMBA_ARCHIVE_SHA256) == 64
    assert setup.XVOICE_ESPEAK_VERSION == "1.52.0"
    assert len(setup.XVOICE_ESPEAK_EXE_SHA256) == 64
    assert len(setup.XVOICE_ESPEAK_DLL_SHA256) == 64
    assert len(setup.XVOICE_ESPEAK_BUNDLE_SHA256) == 64
    assert len(setup.XVOICE_FASTTEXT_LID_SHA256) == 64
    assert len(setup.XVOICE_CMUDICT_SHA256) == 64
    assert BACKEND_DESCRIPTORS[definition.backend_id].requires_reference_language is True


def test_xvoice_setup_recreates_the_managed_conda_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "project"
    runtime = project_root / ".cache" / "runtimes" / "local--x-voice"
    legacy_prefix = runtime / "conda"
    legacy_prefix.mkdir(parents=True)
    (legacy_prefix / "python.exe").write_bytes(b"old-python")
    legacy_package = legacy_prefix / "Lib" / "site-packages" / "legacy.pth"
    legacy_package.parent.mkdir(parents=True)
    legacy_package.write_text("last-known-good\n", encoding="utf-8")
    legacy_marker = runtime / "requirements.source.sha256"
    legacy_marker.write_text("active-source\n", encoding="utf-8")
    runner_manifest = (
        project_root / ".cache" / "runners" / "local--x-voice" / "runner.json"
    )
    runner_manifest.parent.mkdir(parents=True)
    runner_manifest.write_text(
        json.dumps(
            {
                "backend_id": "local:x-voice",
                "command": [str(legacy_prefix / "python.exe")],
            }
        ),
        encoding="utf-8",
    )
    conda_prefix = runtime / "conda-a"
    stale_package = conda_prefix / "Lib" / "site-packages" / "stale.pth"
    stale_package.parent.mkdir(parents=True)
    stale_package.write_text("interrupted-partial-env\n", encoding="utf-8")

    micromamba = runtime / "tools" / "micromamba.exe"
    micromamba.parent.mkdir(parents=True)
    micromamba.write_bytes(b"micromamba")
    archive = runtime / "tools" / "micromamba.tar.bz2"
    archive.write_bytes(b"archive")
    conda_source = tmp_path / "xvoice-conda-win-64.lock"
    conda_source.write_text("@EXPLICIT\n", encoding="utf-8")
    requirements_source = tmp_path / "xvoice.in"
    requirements_source.write_text("fixture\n", encoding="utf-8")
    requirements_source.with_suffix(".lock").write_text(
        "fixture==1 --hash=sha256:" + "a" * 64 + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        setup,
        "_prepare_xvoice_micromamba",
        lambda *args, **kwargs: (micromamba, archive),
    )
    monkeypatch.setattr(setup, "_find_uv", lambda: "uv.exe")
    monkeypatch.setattr(setup, "runner_setup_source_revision", lambda _: "source-v1")
    monkeypatch.setattr(
        setup,
        "_requirements_path",
        lambda name: conda_source if name == "xvoice-conda-win-64.lock" else requirements_source,
    )
    commands: list[list[str]] = []

    fail_create = True

    def run(command, **kwargs) -> str:
        nonlocal fail_create
        values = [str(value) for value in command]
        commands.append(values)
        if values[0] == str(micromamba):
            assert values[1] == "create"
            assert not conda_prefix.exists()
            conda_prefix.mkdir(parents=True)
            if fail_create:
                (conda_prefix / "partial.txt").write_text("partial\n", encoding="utf-8")
                raise VideoGeneratorError("fixture create failure")
            (conda_prefix / "python.exe").write_bytes(b"new-python")
        if values[:2] == ["uv.exe", "--version"]:
            return "uv 0.11.28"
        if values[:2] == [str(conda_prefix / "python.exe"), "-c"]:
            return "2.1.7"
        return ""

    monkeypatch.setattr(setup, "_run", run)

    with pytest.raises(VideoGeneratorError, match="fixture create failure"):
        setup._prepare_xvoice_environment(
            project_root,
            runtime,
            environment={},
            download=False,
        )

    assert (legacy_prefix / "python.exe").read_bytes() == b"old-python"
    assert legacy_package.is_file()
    assert not conda_prefix.exists()
    assert legacy_marker.read_text(encoding="utf-8") == "active-source\n"
    assert not (runtime / "conda-a.source.sha256").exists()
    fail_create = False
    commands.clear()
    python, _, _, source_marker, _ = setup._prepare_xvoice_environment(
        project_root,
        runtime,
        environment={},
        download=False,
    )

    assert python == conda_prefix / "python.exe"
    assert not stale_package.exists()
    assert legacy_package.is_file()
    assert legacy_marker.read_text(encoding="utf-8") == "active-source\n"
    assert source_marker == runtime / "conda-a.source.sha256"
    assert source_marker.read_text(encoding="utf-8") == "source-v1\n"
    assert any(command[1:3] == ["pip", "install"] for command in commands)
    assert "--require-hashes" in next(
        command for command in commands if command[1:3] == ["pip", "install"]
    )


def test_xvoice_setup_rejects_a_runtime_outside_the_managed_cache(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    runtime = tmp_path / "outside-runtime"
    conda_prefix = runtime / "conda"
    conda_prefix.mkdir(parents=True)
    marker = conda_prefix / "keep.txt"
    marker.write_text("keep\n", encoding="utf-8")

    with pytest.raises(VideoGeneratorError, match="redirected managed runtime path"):
        setup._prepare_xvoice_environment(
            project_root,
            runtime,
            environment={},
            download=False,
        )

    assert marker.is_file()


def test_xvoice_unknown_active_slot_uses_only_an_empty_slot(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    runtime = project_root / ".cache" / "runtimes" / "local--x-voice"
    slot_a = runtime / "conda-a"
    slot_b = runtime / "conda-b"
    slot_a.mkdir(parents=True)
    (slot_a / "python.exe").write_bytes(b"sole-valid-environment")

    active, target = setup._xvoice_environment_target(
        project_root,
        runtime / "conda",
        slot_a,
        slot_b,
    )

    assert active is None
    assert target == slot_b
    (slot_b / "python.exe").parent.mkdir(parents=True)
    (slot_b / "python.exe").write_bytes(b"second-possibly-valid-environment")
    with pytest.raises(VideoGeneratorError, match="safe inactive environment slot"):
        setup._xvoice_environment_target(
            project_root,
            runtime / "conda",
            slot_a,
            slot_b,
        )


def test_xvoice_missing_manifest_slot_preserves_the_other_valid_slot(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    runtime = project_root / ".cache" / "runtimes" / "local--x-voice"
    slot_a = runtime / "conda-a"
    slot_b = runtime / "conda-b"
    slot_b.mkdir(parents=True)
    (slot_b / "python.exe").write_bytes(b"last-valid-environment")
    runner_manifest = (
        project_root / ".cache" / "runners" / "local--x-voice" / "runner.json"
    )
    runner_manifest.parent.mkdir(parents=True)
    runner_manifest.write_text(
        json.dumps(
            {
                "backend_id": "local:x-voice",
                "command": [str(slot_a / "python.exe")],
            }
        ),
        encoding="utf-8",
    )

    active, target = setup._xvoice_environment_target(
        project_root,
        runtime / "conda",
        slot_a,
        slot_b,
    )

    assert active is None
    assert target == slot_a
    assert (slot_b / "python.exe").read_bytes() == b"last-valid-environment"


def test_xvoice_espeak_bundle_digest_is_order_independent() -> None:
    first = {"path": "a", "sha256": "A" * 64}
    second = {"path": "b", "sha256": "b" * 64}

    assert setup._manifest_file_set_sha256({"files": [first, second]}) == (
        setup._manifest_file_set_sha256({"files": [second, first]})
    )


def test_xvoice_static_download_requires_the_pinned_sha256(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"pinned artifact"
    destination = tmp_path / "artifact.bin"
    monkeypatch.setattr(
        setup.urllib.request,
        "urlopen",
        lambda request, timeout: io.BytesIO(payload),
    )

    setup._download_verified_url(
        url="https://example.invalid/artifact.bin",
        destination=destination,
        expected_sha256=setup.hashlib.sha256(payload).hexdigest(),
        download=True,
    )

    assert destination.read_bytes() == payload
    with pytest.raises(VideoGeneratorError, match="SHA-256 mismatch"):
        setup._download_verified_url(
            url="https://example.invalid/artifact.bin",
            destination=destination,
            expected_sha256="0" * 64,
            download=True,
        )


def test_moss_manifest_tracks_primary_and_codec_snapshots(tmp_path: Path) -> None:
    definition = setup.LOCAL_DEFINITIONS["local:moss-tts-v1.5"]
    supporting = definition.supporting_models[0]
    primary = tmp_path / ".cache" / "models" / definition.model_subdir
    codec = tmp_path / ".cache" / "models" / supporting.model_subdir
    runtime = tmp_path / ".cache" / "runtimes" / setup.runner_slug(definition.backend_id)
    for root, repository, revision in (
        (primary, definition.model_repo, definition.model_revision),
        (codec, supporting.model_repo, supporting.model_revision),
    ):
        root.mkdir(parents=True)
        (root / "weights.bin").write_bytes(b"fixture")
        write_asset_manifest(root, repo=repository, revision=revision)
    runtime.mkdir(parents=True)
    lock = runtime / "requirements.lock"
    lock.write_text("fixture\n", encoding="utf-8")
    (runtime / "requirements.source.sha256").write_text("fixture\n", encoding="utf-8")

    manifest_path = setup._write_runner_manifest(
        project_root=tmp_path,
        definition=definition,
        command_python=sys.executable,
        lock_path=lock,
        model_path=primary,
        supporting_model_paths={supporting: codec},
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["model_paths"] == [
        ".cache/models/moss-tts-v1.5",
        ".cache/models/moss-audio-tokenizer-v2",
    ]
    assert manifest["environment"]["VIDEO_GENERATOR_CODEC_PATH"] == (
        ".cache/models/moss-audio-tokenizer-v2"
    )
    assert set(manifest["asset_manifests"]) == {
        ".cache/models/moss-tts-v1.5/asset-manifest.json",
        ".cache/models/moss-audio-tokenizer-v2/asset-manifest.json",
    }
    assert manifest["asset_revisions"] == {
        ".cache/models/moss-tts-v1.5/asset-manifest.json": definition.model_revision,
        ".cache/models/moss-audio-tokenizer-v2/asset-manifest.json": supporting.model_revision,
    }
    manifest["requires_cuda"] = False
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    report = RunnerManager(
        project_root=tmp_path,
        run_root=tmp_path / "runs" / "probe",
    ).probe(definition.backend_id)

    assert report.ready is True
    asset_checks = [item for item in report.items if item.name.startswith("asset_manifest:")]
    assert len(asset_checks) == 2
    assert all(item.ready and item.detail == "verified" for item in asset_checks)


def test_snapshot_download_passes_only_required_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, str] = {}
    destination = tmp_path / ".cache" / "models" / "omnivoice"
    destination.mkdir(parents=True)
    (destination / "stale-injected.py").write_text("raise RuntimeError\n", encoding="utf-8")

    def fake_run(command, *, cwd, environment=None, timeout=7200) -> str:
        captured.update(environment or {})
        values = [str(value) for value in command]
        staging = Path(values[values.index("--destination") + 1])
        assert staging != destination
        (staging / "weights.bin").write_bytes(b"fixture")
        write_asset_manifest(
            staging,
            repo=setup.LOCAL_DEFINITIONS["local:omnivoice"].model_repo,
            revision=setup.LOCAL_DEFINITIONS["local:omnivoice"].model_revision,
        )
        return ""

    monkeypatch.setattr(setup, "_run", fake_run)
    setup._download_snapshot(
        project_root=tmp_path,
        python_command=[sys.executable],
        definition=setup.LOCAL_DEFINITIONS["local:omnivoice"],
        destination=destination,
        environment={
            "HF_TOKEN": "hugging-face-token",
            "OPENAI_API_KEY": "cloud-secret",
            "AWS_SECRET_ACCESS_KEY": "unrelated-secret",
        },
    )

    assert captured["HF_TOKEN"] == "hugging-face-token"
    assert "OPENAI_API_KEY" not in captured
    assert "AWS_SECRET_ACCESS_KEY" not in captured
    assert captured["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert captured["PYTHONPATH"] == str(tmp_path / "src")
    assert (destination / "weights.bin").is_file()
    assert not (destination / "stale-injected.py").exists()


def test_snapshot_download_reuses_a_verified_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    definition = setup.LOCAL_DEFINITIONS["local:omnivoice"]
    destination = tmp_path / ".cache" / "models" / definition.model_subdir
    destination.mkdir(parents=True)
    (destination / "weights.bin").write_bytes(b"fixture")
    write_asset_manifest(
        destination,
        repo=definition.model_repo,
        revision=definition.model_revision,
    )
    monkeypatch.setattr(
        setup,
        "_run",
        lambda *args, **kwargs: pytest.fail("verified snapshots must not be downloaded again"),
    )

    setup._download_snapshot(
        project_root=tmp_path,
        python_command=[sys.executable],
        definition=definition,
        destination=destination,
        environment={},
    )

    assert (destination / "weights.bin").read_bytes() == b"fixture"


def test_snapshot_download_does_not_replace_cache_with_an_invalid_stage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    definition = setup.LOCAL_DEFINITIONS["local:omnivoice"]
    destination = tmp_path / ".cache" / "models" / definition.model_subdir
    destination.mkdir(parents=True)
    stale = destination / "stale.bin"
    stale.write_bytes(b"keep-on-failure")

    def fake_run(command, *, cwd, environment=None, timeout=7200) -> str:
        return ""

    monkeypatch.setattr(setup, "_run", fake_run)

    with pytest.raises(VideoGeneratorError, match="failed verification"):
        setup._download_snapshot(
            project_root=tmp_path,
            python_command=[sys.executable],
            definition=definition,
            destination=destination,
            environment={},
        )

    assert stale.read_bytes() == b"keep-on-failure"


def test_native_install_sanitizes_dependency_tool_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: list[dict[str, str]] = []
    definition = setup.LOCAL_DEFINITIONS["local:omnivoice"]

    def fake_run(command, *, cwd, environment=None, timeout=7200) -> str:
        values = [str(value) for value in command]
        captured.append(dict(environment or {}))
        if values[1:] == ["--version"]:
            return "uv 0.6.9"
        if values[1] == "venv":
            python = setup._native_python(
                tmp_path / ".cache" / "runtimes" / setup.runner_slug(definition.backend_id)
            )
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_bytes(b"")
        return ""

    monkeypatch.setattr(setup, "_find_uv", lambda: "uv")
    monkeypatch.setattr(setup, "_run", fake_run)

    setup._install_native_environment(
        tmp_path,
        definition,
        environment={
            "PATH": "fixture-path",
            "CUDA_PATH": "C:/CUDA",
            "UV_INDEX_URL": "https://packages.example/simple",
            "OPENAI_API_KEY": "cloud-secret",
            "GEMINI_API_KEY": "other-cloud-secret",
            "AWS_SECRET_ACCESS_KEY": "unrelated-secret",
        },
    )

    assert captured
    assert all(value["CUDA_PATH"] == "C:/CUDA" for value in captured)
    assert all(value["UV_INDEX_URL"] == "https://packages.example/simple" for value in captured)
    assert all("OPENAI_API_KEY" not in value for value in captured)
    assert all("GEMINI_API_KEY" not in value for value in captured)
    assert all("AWS_SECRET_ACCESS_KEY" not in value for value in captured)


def _fixture_candidate() -> setup.CuratedLlmCandidate:
    model_hash = sha256_file(Path(__file__))
    return setup.CuratedLlmCandidate(
        candidate_id="fixture-q4-mtp",
        model_id="fixture/model",
        repository="fixture/model-GGUF",
        revision="1" * 40,
        license_name="Apache-2.0",
        quantization="Q4_K_M",
        mtp="embedded",
        speculative_tokens=2,
        estimated_download_gb=1.0,
        artifacts=(
            setup.CuratedLlmArtifact(
                role="model",
                filename="model.gguf",
                sha256=model_hash,
            ),
        ),
    )


def test_curated_llm_download_pins_source_and_writes_manifest(tmp_path: Path, monkeypatch) -> None:
    candidate = _fixture_candidate()
    artifact_bytes = Path(__file__).read_bytes()
    commands: list[list[str]] = []

    def fake_run(command, *, cwd, environment=None, timeout=7200) -> str:
        values = [str(value) for value in command]
        commands.append(values)
        destination = Path(values[values.index("--local-dir") + 1])
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "model.gguf").write_bytes(artifact_bytes)
        assert "HF_TOKEN" not in environment
        assert "OPENAI_API_KEY" not in environment
        assert "AWS_SECRET_ACCESS_KEY" not in environment
        assert Path(environment["TEMP"]) == destination
        return ""

    monkeypatch.setitem(setup.CURATED_LLM_CANDIDATES, candidate.candidate_id, candidate)
    monkeypatch.setattr(setup, "_find_uv", lambda: "uv")
    monkeypatch.setattr(setup, "_run", fake_run)

    destination = setup.download_curated_llm_candidate(
        project_root=tmp_path,
        candidate_id=candidate.candidate_id,
        environment={
            "HF_TOKEN": "token-not-logged",
            "OPENAI_API_KEY": "cloud-secret",
            "AWS_SECRET_ACCESS_KEY": "other-secret",
        },
    )

    command = commands[0]
    assert command[command.index("--revision") + 1] == candidate.revision
    assert candidate.repository in command
    assert candidate.artifacts[0].filename in command
    manifest = json.loads((destination / "asset-manifest.json").read_text(encoding="utf-8"))
    assert manifest["revision"] == candidate.revision
    assert candidate.revision in manifest["model_card"]
    assert manifest["files"][0]["sha256"] == candidate.artifacts[0].sha256
    assert "token-not-logged" not in json.dumps(manifest)


def test_curated_llm_download_skips_verified_existing_file(tmp_path: Path, monkeypatch) -> None:
    candidate = _fixture_candidate()
    destination = tmp_path / ".cache" / "models" / "llm" / candidate.candidate_id
    destination.mkdir(parents=True)
    (destination / "model.gguf").write_bytes(Path(__file__).read_bytes())
    monkeypatch.setitem(setup.CURATED_LLM_CANDIDATES, candidate.candidate_id, candidate)
    monkeypatch.setattr(
        setup,
        "_run",
        lambda *args, **kwargs: pytest.fail("verified artifacts must not be downloaded again"),
    )

    setup.download_curated_llm_candidate(
        project_root=tmp_path,
        candidate_id=candidate.candidate_id,
        environment={},
    )

    assert (destination / "asset-manifest.json").is_file()


def test_curated_llm_download_rejects_a_hash_mismatch(tmp_path: Path, monkeypatch) -> None:
    base_candidate = _fixture_candidate()
    candidate = replace(
        base_candidate,
        artifacts=(
            *base_candidate.artifacts,
            setup.CuratedLlmArtifact(
                role="draft-model",
                filename="draft.gguf",
                sha256="f" * 64,
            ),
        ),
    )

    def fake_run(command, *, cwd, environment=None, timeout=7200) -> str:
        values = [str(value) for value in command]
        destination = Path(values[values.index("--local-dir") + 1])
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "model.gguf").write_bytes(Path(__file__).read_bytes())
        (destination / "draft.gguf").write_bytes(b"wrong")
        return ""

    monkeypatch.setitem(setup.CURATED_LLM_CANDIDATES, candidate.candidate_id, candidate)
    monkeypatch.setattr(setup, "_find_uv", lambda: "uv")
    monkeypatch.setattr(setup, "_run", fake_run)

    with pytest.raises(VideoGeneratorError, match="SHA-256 mismatch"):
        setup.download_curated_llm_candidate(
            project_root=tmp_path,
            candidate_id=candidate.candidate_id,
            environment={},
        )

    destination = tmp_path / ".cache" / "models" / "llm" / candidate.candidate_id
    assert not (destination / "model.gguf").exists()
    assert not (destination / "draft.gguf").exists()
    assert not (destination / "asset-manifest.json").exists()
    assert not list(destination.parent.glob("*.download"))


def test_curated_llm_download_removes_stage_after_command_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    candidate = _fixture_candidate()
    monkeypatch.setitem(setup.CURATED_LLM_CANDIDATES, candidate.candidate_id, candidate)
    monkeypatch.setattr(setup, "_find_uv", lambda: "uv")
    monkeypatch.setattr(
        setup,
        "_run",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("download failed")),
    )

    with pytest.raises(RuntimeError, match="download failed"):
        setup.download_curated_llm_candidate(
            project_root=tmp_path,
            candidate_id=candidate.candidate_id,
            environment={},
        )

    managed_root = tmp_path / ".cache" / "models" / "llm"
    assert not list(managed_root.glob("*.download"))


@pytest.mark.parametrize(
    ("candidate", "error"),
    [
        pytest.param(
            replace(_fixture_candidate(), candidate_id="../outside"),
            "candidate ID",
            id="candidate-id",
        ),
        pytest.param(
            replace(
                _fixture_candidate(),
                artifacts=(
                    replace(_fixture_candidate().artifacts[0], filename="../model.gguf"),
                ),
            ),
            "filename",
            id="artifact-filename",
        ),
    ],
)
def test_curated_llm_download_rejects_path_traversal(
    tmp_path: Path,
    monkeypatch,
    candidate: setup.CuratedLlmCandidate,
    error: str,
) -> None:
    monkeypatch.setitem(setup.CURATED_LLM_CANDIDATES, candidate.candidate_id, candidate)

    with pytest.raises(VideoGeneratorError, match=error):
        setup.download_curated_llm_candidate(
            project_root=tmp_path,
            candidate_id=candidate.candidate_id,
            environment={},
        )

    assert not (tmp_path / "outside").exists()
