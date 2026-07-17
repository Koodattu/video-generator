from __future__ import annotations

from pathlib import Path

import pytest

from video_generator.contracts import (
    ContentFormat,
    CreativeBrief,
    NarrationPace,
    VideoOrientation,
    VideoStyle,
    VisualShotMode,
)
from video_generator.errors import CheckpointError
from video_generator.profiles import PROFILES
from video_generator.prompting import build_frozen_assets
from video_generator.run_store import RunStore, earliest_config_impact
from video_generator.util import atomic_write_json, hash_value, read_json, relative_path


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


@pytest.mark.parametrize(
    ("updates", "expected_stage"),
    [
        ({"content_format": ContentFormat.EXPLAINER}, "research"),
        ({"narration_pace": NarrationPace.FAST}, "script-draft"),
        ({"narration_delivery": "Measured and reflective."}, "script-draft"),
        ({"video_style": VideoStyle.REMOTION_EXPLAINER}, "research"),
        ({"visual_shot_mode": VisualShotMode.CADENCED}, "captions"),
        ({"shot_target_seconds": 4}, "visual-plan"),
    ],
)
def test_multi_format_config_changes_invalidate_the_earliest_affected_stage(
    resolved_config,
    updates: dict[str, object],
    expected_stage: str,
) -> None:
    changed = resolved_config.model_copy(update=updates)

    assert earliest_config_impact(resolved_config, changed) == expected_stage


def test_claim_inventory_backend_change_invalidates_script_revision(resolved_config) -> None:
    bindings = dict(resolved_config.task_bindings)
    bindings["claim_inventory"] = "deterministic:structured"
    changed = resolved_config.model_copy(update={"task_bindings": bindings})

    assert earliest_config_impact(resolved_config, changed) == "script-revision"


def test_remotion_rhythm_backend_change_invalidates_visual_plan(resolved_config) -> None:
    bindings = dict(resolved_config.task_bindings)
    bindings["remotion_rhythm"] = "deterministic:structured"
    changed = resolved_config.model_copy(update={"task_bindings": bindings})

    assert earliest_config_impact(resolved_config, changed) == "visual-plan"


def test_orientation_change_invalidates_caption_layout_and_visuals(
    resolved_config,
) -> None:
    changed = resolved_config.model_copy(
        update={
            "orientation": VideoOrientation.PORTRAIT,
            "delivery_width": 720,
            "delivery_height": 1280,
        }
    )

    assert earliest_config_impact(resolved_config, changed) == "captions"


def test_switching_to_or_from_higgs_invalidates_script_draft(resolved_config) -> None:
    bindings = dict(resolved_config.task_bindings)
    bindings["narration_synthesis"] = "local:omnivoice"
    changed = resolved_config.model_copy(update={"task_bindings": bindings})

    assert earliest_config_impact(resolved_config, changed) == "script-draft"
    assert earliest_config_impact(changed, resolved_config) == "script-draft"


def test_switching_between_other_tts_backends_still_invalidates_narration(
    resolved_config,
) -> None:
    old_bindings = dict(resolved_config.task_bindings)
    old_bindings["narration_synthesis"] = "local:omnivoice"
    new_bindings = dict(old_bindings)
    new_bindings["narration_synthesis"] = "local:voxcpm2"
    old = resolved_config.model_copy(update={"task_bindings": old_bindings})
    new = resolved_config.model_copy(update={"task_bindings": new_bindings})

    assert earliest_config_impact(old, new) == "narration"


def test_legacy_fiction_run_loads_without_claim_inventory_binding(
    tmp_path: Path,
    resolved_config,
) -> None:
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
    stored_config = read_json(store.config_path)
    del stored_config["task_bindings"]["claim_inventory"]
    atomic_write_json(store.config_path, stored_config)
    manifest = read_json(store.manifest_path)
    manifest["config_hash"] = hash_value(stored_config)
    atomic_write_json(store.manifest_path, manifest)

    reopened = RunStore.open(store.root)

    assert reopened.config.task_bindings["claim_inventory"] == (
        reopened.config.task_bindings["factual_review"]
    )
    assert "claim_inventory" not in read_json(reopened.config_path)["task_bindings"]


def test_legacy_run_loads_without_remotion_task_bindings(
    tmp_path: Path,
    resolved_config,
) -> None:
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
    stored_config = read_json(store.config_path)
    del stored_config["task_bindings"]["remotion_rhythm"]
    del stored_config["task_bindings"]["remotion_direction"]
    del stored_config["task_bindings"]["remotion_asset_select"]
    atomic_write_json(store.config_path, stored_config)
    manifest = read_json(store.manifest_path)
    manifest["config_hash"] = hash_value(stored_config)
    atomic_write_json(store.manifest_path, manifest)

    reopened = RunStore.open(store.root)

    assert reopened.config.task_bindings["remotion_rhythm"] == (
        reopened.config.task_bindings["visual_plan"]
    )
    assert reopened.config.task_bindings["remotion_direction"] == (
        reopened.config.task_bindings["visual_plan"]
    )
    assert reopened.config.task_bindings["remotion_asset_select"] == (
        reopened.config.task_bindings["image_prompt_compile"]
    )
    persisted = read_json(reopened.config_path)["task_bindings"]
    assert "remotion_rhythm" not in persisted
    assert "remotion_direction" not in persisted
    assert "remotion_asset_select" not in persisted
