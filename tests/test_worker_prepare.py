from __future__ import annotations

import json
from pathlib import Path

from video_generator.workers.prepare import write_asset_manifest


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
