from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from video_generator.runners import RunnerManager
from video_generator.util import atomic_write_json, sha256_file


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
