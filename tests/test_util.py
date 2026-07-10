from __future__ import annotations

import os
from pathlib import Path

from video_generator import util


def test_atomic_write_retries_transient_windows_replace_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / "manifest.json"
    destination.write_bytes(b"old")
    real_replace = os.replace
    attempts = 0

    def flaky_replace(source, target) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError(5, "transient sharing violation", str(target))
        real_replace(source, target)

    monkeypatch.setattr(util.os, "name", "nt")
    monkeypatch.setattr(util.os, "replace", flaky_replace)
    monkeypatch.setattr(util.time, "sleep", lambda _: None)

    util.atomic_write_bytes(destination, b"new")

    assert attempts == 3
    assert destination.read_bytes() == b"new"
