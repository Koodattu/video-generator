from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from video_generator.errors import BackendError, ErrorKind
from video_generator.runners import RunnerManager, RunnerSpec, runner_setup_source_revision
from video_generator.util import atomic_write_json, sha256_file


def test_native_runner_protocol_forces_utf8_stdio(tmp_path: Path) -> None:
    manager = RunnerManager(project_root=tmp_path, run_root=tmp_path / "runs" / "fixture")
    spec = SimpleNamespace(
        platform="native",
        command=[sys.executable],
        environment={},
    )

    _, environment = manager._command(spec)

    assert environment["PYTHONUTF8"] == "1"
    assert environment["PYTHONIOENCODING"] == "utf-8"
    assert Path(environment["HF_MODULES_CACHE"]).is_relative_to(
        tmp_path / "runs" / "fixture" / "scratch"
    )
    _, second_environment = manager._command(spec)
    assert second_environment["HF_MODULES_CACHE"] != environment["HF_MODULES_CACHE"]


def test_runner_setup_revision_covers_host_pins_worker_and_lock(monkeypatch) -> None:
    hashed_paths = []

    def record(path: Path) -> str:
        hashed_paths.append(path.name)
        return path.name

    monkeypatch.setattr("video_generator.runners.sha256_file", record)

    revision = runner_setup_source_revision("xvoice")

    assert len(revision) == 64
    assert "setup.py" in hashed_paths
    assert "main.py" in hashed_paths
    assert "xvoice.lock" in hashed_paths
    assert "xvoice-conda-win-64.lock" in hashed_paths


def test_runner_manifest_rejects_stale_host_provenance(tmp_path: Path) -> None:
    manager = RunnerManager(project_root=tmp_path, run_root=tmp_path / "runs" / "fixture")
    spec = RunnerSpec(
        backend_id="local:faster-whisper-large-v3-turbo",
        platform="native",
        command=[sys.executable],
        model_family="faster-whisper",
        model_paths=[".cache/models/model"],
        asset_manifests={".cache/models/model/asset-manifest.json": "hash"},
        runtime_files={".cache/runtimes/runtime/requirements.lock": "hash"},
        runtime_revision="runtime-v1",
        model_revision="0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf",
        setup_source_revision="obsolete",
        license_name="MIT",
    )
    atomic_write_json(
        manager.manifest_path(spec.backend_id),
        spec.model_dump(mode="json"),
    )

    with pytest.raises(BackendError, match="obsolete Setup requirements source"):
        manager.load_spec(spec.backend_id)


def test_runtime_manifest_rejects_untracked_executable_source(tmp_path: Path) -> None:
    source = tmp_path / "runtime.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    manifest = tmp_path / "runtime-source.asset-manifest.json"
    atomic_write_json(
        manifest,
        {
            "schema_version": 1,
            "root": ".",
            "revision": "pinned",
            "exact_file_suffixes": [".py"],
            "exact_exclude_roots": [".venv", "__pycache__"],
            "files": [
                {
                    "path": source.name,
                    "size": source.stat().st_size,
                    "sha256": sha256_file(source),
                }
            ],
        },
    )
    assert RunnerManager._verify_asset_manifest(manifest, expected_revision=None) == "verified"

    (tmp_path / "sitecustomize.py").write_text("raise RuntimeError('unexpected')\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected executable runtime source"):
        RunnerManager._verify_asset_manifest(manifest, expected_revision=None)


def test_stop_current_retains_worker_cleanup_report(tmp_path: Path, monkeypatch) -> None:
    class Handle:
        def close(self) -> None:
            return None

    class Process:
        def __init__(self) -> None:
            self.returncode = None
            self.stdin = Handle()
            self.stdout = Handle()

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

    manager = RunnerManager(project_root=tmp_path, run_root=tmp_path / "runs" / "fixture")
    manager.current = SimpleNamespace(
        spec=SimpleNamespace(backend_id="local:llama-server", model_family="llama-server"),
        process=Process(),
        reader=SimpleNamespace(join=lambda timeout: None),
        stderr_handle=Handle(),
    )
    lifecycle = {
        "process_exited": True,
        "gpu_process_released": True,
        "vram_within_tolerance": True,
        "post_exit": {"observable": True, "used_mb": 512, "process_ids": []},
    }
    monkeypatch.setattr(
        manager,
        "_invoke_current",
        lambda operation, payload, timeout, stop_on_failure: {"lifecycle": lifecycle},
    )

    manager.stop_current()

    assert manager.current is None
    assert manager.last_cleanup["local:llama-server"] == lifecycle
    cleanup_log = tmp_path / "runs" / "fixture" / "logs" / "runner-cleanup-001-local--llama-server.json"
    assert cleanup_log.is_file()


def test_stop_current_records_cleanup_for_generic_gpu_worker(tmp_path: Path, monkeypatch) -> None:
    class Handle:
        def close(self) -> None:
            return None

    class Process:
        def __init__(self) -> None:
            self.pid = 2468
            self.returncode = None
            self.stdin = Handle()
            self.stdout = Handle()

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

    manager = RunnerManager(project_root=tmp_path, run_root=tmp_path / "runs" / "fixture")
    manager.current = SimpleNamespace(
        spec=SimpleNamespace(
            backend_id="local:faster-whisper-large-v3-turbo",
            model_family="faster-whisper",
        ),
        process=Process(),
        gpu_baseline={"observable": True, "used_mb": 512, "process_ids": []},
        reader=SimpleNamespace(join=lambda timeout: None),
        stderr_handle=Handle(),
    )
    monkeypatch.setattr(
        manager,
        "_invoke_current",
        lambda operation, payload, timeout, stop_on_failure: {"lifecycle": {}},
    )
    monkeypatch.setattr(
        "video_generator.workers.llama_server.gpu_snapshot",
        lambda: SimpleNamespace(observable=True, used_mb=512, process_ids=(111,)),
    )

    manager.stop_current()

    lifecycle = manager.last_cleanup["local:faster-whisper-large-v3-turbo"]
    assert lifecycle["process_exited"] is True
    assert lifecycle["gpu_process_released"] is True
    assert lifecycle["worker_pid"] == 2468


def test_live_llama_probe_requires_fresh_cleanup_evidence(tmp_path: Path, monkeypatch) -> None:
    manager = RunnerManager(project_root=tmp_path, run_root=tmp_path / "runs" / "fixture")
    backend_id = "local:llama-server"
    spec = SimpleNamespace(
        backend_id=backend_id,
        platform="native",
        command=[sys.executable],
        requires_cuda=False,
        model_paths=[],
        runtime_files={},
        asset_manifests={},
        startup_timeout_seconds=1,
        model_family="llama-server",
    )
    manager.last_cleanup[backend_id] = {
        "process_exited": True,
        "gpu_process_released": True,
        "vram_within_tolerance": True,
    }
    monkeypatch.setattr(manager, "load_spec", lambda _: spec)
    monkeypatch.setattr(manager, "invoke", lambda *args, **kwargs: {"status": "ok"})
    monkeypatch.setattr(manager, "stop_current", lambda: None)

    report = manager.probe(backend_id, live=True)

    cleanup = next(item for item in report.items if item.name == "live_worker_cleanup")
    assert not cleanup.ready
    assert not report.ready


def test_live_llama_probe_rejects_vram_above_host_baseline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager = RunnerManager(project_root=tmp_path, run_root=tmp_path / "runs" / "fixture")
    backend_id = "local:llama-server"
    spec = SimpleNamespace(
        backend_id=backend_id,
        platform="native",
        command=[sys.executable],
        requires_cuda=False,
        model_paths=[],
        runtime_files={},
        asset_manifests={},
        startup_timeout_seconds=1,
        model_family="llama-server",
    )
    monkeypatch.setattr(manager, "load_spec", lambda _: spec)
    monkeypatch.setattr(manager, "invoke", lambda *args, **kwargs: {"status": "ok"})

    def record_cleanup() -> None:
        manager.last_cleanup[backend_id] = {
            "process_exited": True,
            "gpu_process_released": True,
            "vram_within_tolerance": False,
            "post_exit": {"observable": True, "used_mb": 2048, "process_ids": []},
        }

    monkeypatch.setattr(manager, "stop_current", record_cleanup)

    report = manager.probe(backend_id, live=True)

    cleanup = next(item for item in report.items if item.name == "live_worker_cleanup")
    assert not cleanup.ready
    assert "aggregate Windows VRAM" in (cleanup.action or "")
    assert not report.ready


def test_live_cuda_probe_requires_fresh_cleanup_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager = RunnerManager(project_root=tmp_path, run_root=tmp_path / "runs" / "fixture")
    backend_id = "local:faster-whisper-large-v3-turbo"
    spec = SimpleNamespace(
        backend_id=backend_id,
        platform="native",
        command=[sys.executable],
        requires_cuda=True,
        model_paths=[],
        runtime_files={},
        asset_manifests={},
        startup_timeout_seconds=1,
        model_family="faster-whisper",
    )
    monkeypatch.setattr(manager, "load_spec", lambda _: spec)
    monkeypatch.setattr(manager, "invoke", lambda *args, **kwargs: {"status": "ok"})
    monkeypatch.setattr(manager, "stop_current", lambda: None)

    report = manager.probe(backend_id, live=True)

    cleanup = next(item for item in report.items if item.name == "live_worker_cleanup")
    assert not cleanup.ready
    assert not report.ready


def test_stop_current_records_fresh_cleanup_after_forced_kill(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class Handle:
        def close(self) -> None:
            return None

    class Process:
        pid = 2468

        def __init__(self) -> None:
            self.returncode = None
            self.stdin = Handle()
            self.stdout = Handle()

        def poll(self):
            return self.returncode

    process = Process()
    manager = RunnerManager(project_root=tmp_path, run_root=tmp_path / "runs" / "fixture")
    manager.current = SimpleNamespace(
        spec=SimpleNamespace(
            backend_id="local:faster-whisper-large-v3-turbo",
            model_family="faster-whisper",
            requires_cuda=True,
        ),
        process=process,
        health={},
        gpu_baseline={"observable": True, "used_mb": 512, "process_ids": []},
        reader=SimpleNamespace(join=lambda timeout: None),
        stderr_handle=Handle(),
    )
    monkeypatch.setattr(
        manager,
        "_invoke_current",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            BackendError("shutdown failed", kind=ErrorKind.INTERNAL)
        ),
    )

    def kill(_process) -> None:
        _process.returncode = 1

    monkeypatch.setattr(manager, "_kill_process_tree", kill)
    monkeypatch.setattr(
        "video_generator.workers.llama_server.gpu_snapshot",
        lambda: SimpleNamespace(observable=True, used_mb=512, process_ids=(111,)),
    )

    manager.stop_current()

    cleanup = manager.last_cleanup["local:faster-whisper-large-v3-turbo"]
    assert cleanup["forced"] is True
    assert cleanup["process_exited"] is True
    assert cleanup["gpu_process_released"] is True


@pytest.mark.parametrize(
    ("process_ids", "used_mb", "expected_released", "expected_vram"),
    [
        ((111,), 512, True, True),
        ((8642,), 512, False, True),
        ((), 2048, True, False),
    ],
)
def test_stop_current_replaces_unobservable_cuda_cleanup_with_host_probe(
    tmp_path: Path,
    monkeypatch,
    process_ids: tuple[int, ...],
    used_mb: int,
    expected_released: bool,
    expected_vram: bool,
) -> None:
    class Handle:
        def close(self) -> None:
            return None

    class Process:
        pid = 2468

        def __init__(self) -> None:
            self.returncode = None
            self.stdin = Handle()
            self.stdout = Handle()

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

    manager = RunnerManager(project_root=tmp_path, run_root=tmp_path / "runs" / "fixture")
    manager.current = SimpleNamespace(
        spec=SimpleNamespace(
            backend_id="local:llama-server",
            model_family="llama-server",
            requires_cuda=True,
        ),
        process=Process(),
        health={"server_pid": 8642},
        gpu_baseline={"observable": True, "used_mb": 512, "process_ids": []},
        reader=SimpleNamespace(join=lambda timeout: None),
        stderr_handle=Handle(),
    )
    monkeypatch.setattr(
        manager,
        "_invoke_current",
        lambda *args, **kwargs: {
            "lifecycle": {
                "process_exited": True,
                "gpu_process_released": True,
                "vram_within_tolerance": True,
                "post_exit": {"observable": False, "process_ids": [], "used_mb": None},
            }
        },
    )
    monkeypatch.setattr(
        "video_generator.workers.llama_server.gpu_snapshot",
        lambda: SimpleNamespace(observable=True, used_mb=used_mb, process_ids=process_ids),
    )

    manager.stop_current()

    cleanup = manager.last_cleanup["local:llama-server"]
    assert cleanup["post_exit"]["observable"] is True
    assert cleanup["gpu_process_released"] is expected_released
    assert cleanup["vram_within_tolerance"] is expected_vram
