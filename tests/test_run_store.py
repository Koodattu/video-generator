from __future__ import annotations

from pathlib import Path

import pytest

from video_generator.contracts import CreativeBrief
from video_generator.errors import CheckpointError
from video_generator.profiles import PROFILES
from video_generator.prompting import build_frozen_assets
from video_generator.run_store import RunStore
from video_generator.util import read_json, relative_path


def test_parent_linked_fork_rewrites_copied_run_paths(tmp_path: Path, resolved_config) -> None:
    config = resolved_config.model_copy(
        update={
            "profile": "deterministic-test",
            "project_root": str(tmp_path),
            "offline": True,
            "task_bindings": dict(PROFILES["deterministic-test"]),
        }
    )
    brief = CreativeBrief(idea_direction="A tiny winter mystery")
    parent = RunStore.create(
        project_root=tmp_path,
        config=config,
        brief=brief,
        frozen_assets=build_frozen_assets(config),
    )
    metadata = {
        "input_hash": "input",
        "config_hash": "config",
        "backend_id": "deterministic:structured",
        "backend_revision": "1",
        "prompt_version": "prompt",
        "schema_hash": "schema",
    }
    workspace = parent.workspace("research")
    parent.begin_stage("research", attempt=workspace.attempt, **metadata)
    media = workspace.work_dir / "source.txt"
    media.write_text("fixture", encoding="utf-8")
    promoted = parent.promote_stage(
        workspace,
        {"source": {"path": relative_path(media, tmp_path), "sha256": "fixture", "mime_type": "text/plain"}},
    )
    promoted_path = tmp_path / promoted["source"]["path"]
    assert promoted_path.is_file()
    assert "/work/" not in promoted["source"]["path"]

    child = RunStore.fork(
        parent=parent,
        config=config,
        brief=brief,
        frozen_assets=build_frozen_assets(config),
        fork_stage="ideate",
    )

    record = child.stage_record("research")
    assert record is not None
    artifact = read_json(child.root / record.output_paths[0])
    copied_path = artifact["source"]["path"]
    assert child.manifest.run_id in copied_path
    assert parent.manifest.run_id not in copied_path
    assert child.reusable_record("research", **metadata) is not None


def test_aggregate_checkpoint_detects_corrupt_item_media(tmp_path: Path, resolved_config) -> None:
    config = resolved_config.model_copy(
        update={
            "profile": "deterministic-test",
            "project_root": str(tmp_path),
            "offline": True,
            "task_bindings": dict(PROFILES["deterministic-test"]),
        }
    )
    store = RunStore.create(
        project_root=tmp_path,
        config=config,
        brief=CreativeBrief(idea_direction="A tiny winter mystery"),
        frozen_assets=build_frozen_assets(config),
    )
    metadata = {
        "input_hash": "images-input",
        "config_hash": "images-config",
        "backend_id": "deterministic:stick",
        "backend_revision": "1",
        "prompt_version": "",
        "schema_hash": "",
    }
    store.begin_stage("images", attempt=store.next_attempt("images"), **metadata)
    workspace = store.workspace("images", item_id="scene-001")
    media = workspace.work_dir / "image.png"
    media.write_bytes(b"fixture-image")
    promoted = store.promote_item(
        workspace,
        {
            "media": {
                "path": relative_path(media, tmp_path),
                "sha256": "fixture",
                "mime_type": "image/png",
            }
        },
        input_hash="item-input",
        config_hash="item-config",
        backend_id="deterministic:stick",
        backend_revision="1",
    )
    store.complete_fanout_stage("images", {"items": [promoted]})

    (tmp_path / promoted["media"]["path"]).write_bytes(b"corrupt")

    with pytest.raises(CheckpointError, match="missing or corrupt"):
        store.reusable_record("images", **metadata)
