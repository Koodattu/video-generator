from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from video_generator.errors import ErrorKind, VideoGeneratorError
from video_generator.runners import RunnerManager
from video_generator.workers import main as worker_main
from video_generator.workers.prepare import write_asset_manifest


def test_worker_error_kind_preserves_typed_error() -> None:
    error = VideoGeneratorError(
        "fixture",
        kind=ErrorKind.POLICY_REFUSAL,
    )

    assert worker_main.error_kind(error) == "policy_refusal"


def test_worker_protocol_reports_startup_failure_as_not_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("VIDEO_GENERATOR_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("VIDEO_GENERATOR_RUN_ROOT", str(tmp_path / "runs" / "fixture"))
    monkeypatch.setattr(
        worker_main,
        "build_worker",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            VideoGeneratorError(
                "runtime prerequisite missing",
                kind=ErrorKind.NOT_READY,
                action="Repair the runtime.",
            )
        ),
    )
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            json.dumps(
                {
                    "protocol_version": 1,
                    "request_id": "health-1",
                    "operation": "health",
                    "payload": {},
                }
            )
            + "\n"
        ),
    )

    assert worker_main.main(["--kind", "omnivoice"]) == 0

    response = json.loads(capsys.readouterr().out)
    assert response["ok"] is False
    assert response["error"] == {
        "kind": "not_ready",
        "message": "runtime prerequisite missing",
        "action": "Repair the runtime.",
    }


def test_manifest_hashes_files_when_destination_is_under_dot_cache(tmp_path: Path) -> None:
    destination = tmp_path / ".cache" / "models" / "fixture"
    destination.mkdir(parents=True)
    (destination / "weights.bin").write_bytes(b"weights")
    nested_cache = destination / ".cache"
    nested_cache.mkdir()
    (nested_cache / "metadata.json").write_text("{}", encoding="utf-8")

    manifest_path = write_asset_manifest(destination, repo="example/model", revision="abc123")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert [item["path"] for item in manifest["files"]] == ["weights.bin"]
    assert manifest["exact_file_suffixes"] == [
        ".py",
        ".pyi",
        ".pyc",
        ".pyd",
        ".dll",
        ".so",
    ]
    assert manifest["exact_exclude_roots"] == [".cache", "__pycache__"]
    assert manifest["allow_patterns"] == []
    assert RunnerManager._verify_asset_manifest(
        manifest_path,
        expected_revision="abc123",
    ) == "verified"

    (destination / "injected.py").write_text("raise RuntimeError\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected executable runtime source"):
        RunnerManager._verify_asset_manifest(
            manifest_path,
            expected_revision="abc123",
        )
