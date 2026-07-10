from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from video_generator.config import resolve_config
from video_generator.contracts import ResolvedRunConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def resolved_config(tmp_path: Path) -> ResolvedRunConfig:
    shutil.copy2(PROJECT_ROOT / "config.example.toml", tmp_path / "config.toml")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\nversion='0.0.0'\n", encoding="utf-8")
    return resolve_config(tmp_path / "config.toml")
