from __future__ import annotations

import mimetypes
import re
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

from .jobs import RunSupervisor
from ..contracts import PUBLIC_STAGES
from ..errors import CheckpointError
from ..run_store import RunExecutionLock
from ..util import hash_value, read_json


RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def resolve_run_root(project_root: Path, run_id: str) -> Path:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("invalid Run ID")
    runs_root = (project_root / "runs").resolve()
    run_root = (runs_root / run_id).resolve()
    if run_root.parent != runs_root or not (run_root / "manifest.json").is_file():
        raise FileNotFoundError(run_id)
    return run_root


def resolve_artifact_path(run_root: Path, relative_path: str) -> Path:
    if not relative_path or "\x00" in relative_path:
        raise ValueError("invalid artifact path")
    candidate = (run_root / relative_path.replace("\\", "/")).resolve()
    try:
        candidate.relative_to(run_root.resolve())
    except ValueError as exc:
        raise ValueError("artifact path leaves the Run Bundle") from exc
    if not candidate.is_file():
        raise FileNotFoundError(relative_path)
    return candidate


def _read_optional(path: Path) -> dict[str, Any]:
    try:
        value = read_json(path)
    except (OSError, ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _controller_state(
    project_root: Path,
    run_id: str,
    manifest_status: str,
    supervisor: RunSupervisor,
) -> tuple[dict[str, Any] | None, str]:
    controller = supervisor.snapshot(run_id)
    if controller and controller["status"] in {"queued", "running", "stopping"}:
        return controller, str(controller["status"])
    if (
        controller
        and controller["status"] in {"complete", "failed", "stopped"}
        and manifest_status in {"created", "running"}
    ):
        return controller, str(controller["status"])
    if manifest_status != "running":
        return controller, manifest_status
    lock = RunExecutionLock(project_root / "runs" / ".locks" / f"{run_id}.lock")
    try:
        with lock:
            return controller, "interrupted"
    except CheckpointError:
        return controller, "running_external"


def cost_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    calls = manifest.get("cloud_calls")
    calls = calls if isinstance(calls, list) else []
    calculated = 0.0
    calculated_count = 0
    direct_calculated = 0.0
    direct_calculated_count = 0
    inherited_calculated = 0.0
    inherited_calculated_count = 0
    reported = 0.0
    reported_count = 0
    unresolved_maximum = 0.0
    inherited_unresolved = 0.0
    valid_call_count = 0
    inherited_call_count = 0
    rows: dict[tuple[str, str], dict[str, Any]] = {}

    def add_usage(
        *,
        backend_id: str,
        task_id: str,
        estimated: Any,
        actual: Any,
        unresolved: float,
        inherited: bool,
    ) -> None:
        nonlocal calculated, calculated_count, direct_calculated
        nonlocal direct_calculated_count, inherited_calculated
        nonlocal inherited_calculated_count, reported, reported_count
        if isinstance(estimated, (int, float)):
            calculated += float(estimated)
            calculated_count += 1
            if inherited:
                inherited_calculated += float(estimated)
                inherited_calculated_count += 1
            else:
                direct_calculated += float(estimated)
                direct_calculated_count += 1
        if isinstance(actual, (int, float)):
            reported += float(actual)
            reported_count += 1
        key = (backend_id or "unknown", task_id or "unknown")
        row = rows.setdefault(
            key,
            {
                "backend_id": key[0],
                "task_id": key[1],
                "calls": 0,
                "inherited_calls": 0,
                "estimated_usd": 0.0,
                "estimated_calls": 0,
                "actual_usd": 0.0,
                "reported_calls": 0,
                "unresolved_usd": 0.0,
            },
        )
        row["calls"] += 1
        row["inherited_calls"] += int(inherited)
        if isinstance(estimated, (int, float)):
            row["estimated_usd"] += float(estimated)
            row["estimated_calls"] += 1
        if isinstance(actual, (int, float)):
            row["actual_usd"] += float(actual)
            row["reported_calls"] += 1
        row["unresolved_usd"] += unresolved

    if calls:
        for raw in calls:
            if not isinstance(raw, dict):
                continue
            valid_call_count += 1
            status = str(raw.get("status") or "reserved")
            estimated = raw.get("estimated_usd")
            actual = raw.get("actual_usd")
            inherited = bool(raw.get("inherited"))
            inherited_call_count += int(inherited)
            unresolved = (
                float(raw.get("reserved_usd") or 0)
                if status in {"reserved", "unresolved"}
                else 0.0
            )
            if inherited:
                inherited_unresolved += unresolved
            else:
                unresolved_maximum += unresolved
            add_usage(
                backend_id=str(raw.get("backend_id") or "unknown"),
                task_id=str(raw.get("task_id") or "unknown"),
                estimated=estimated,
                actual=actual,
                unresolved=unresolved,
                inherited=inherited,
            )
    else:
        # Stage records are a legacy fallback and are read once; fan-out item files are not summed.
        stages = manifest.get("stages") if isinstance(manifest.get("stages"), dict) else {}
        fork_stage = str(manifest.get("fork_stage") or "")
        fork_index = PUBLIC_STAGES.index(fork_stage) if fork_stage in PUBLIC_STAGES else -1
        for stage_name, stage in stages.items():
            if not isinstance(stage, dict):
                continue
            inherited = fork_index >= 0 and stage_name in PUBLIC_STAGES[:fork_index]
            for usage in stage.get("usage") or []:
                if not isinstance(usage, dict):
                    continue
                add_usage(
                    backend_id=str(usage.get("backend_id") or "unknown"),
                    task_id=str(usage.get("task_id") or "unknown"),
                    estimated=usage.get("estimated_usd"),
                    actual=usage.get("actual_usd"),
                    unresolved=0.0,
                    inherited=inherited,
                )
        reserved = float(manifest.get("reserved_cost_usd") or 0)
        if reserved > 0 and not calculated_count and not reported_count:
            unresolved_maximum = reserved

    return {
        "calculated_list_price_usd": round(calculated, 6) if calculated_count else None,
        "direct_calculated_list_price_usd": (
            round(direct_calculated, 6) if direct_calculated_count else None
        ),
        "inherited_calculated_list_price_usd": (
            round(inherited_calculated, 6) if inherited_calculated_count else None
        ),
        "provider_reported_usd": round(reported, 6) if reported_count else None,
        "unresolved_maximum_usd": round(unresolved_maximum, 6),
        "inherited_unresolved_usd": round(inherited_unresolved, 6),
        "conservative_reserved_usd": round(float(manifest.get("reserved_cost_usd") or 0), 6),
        "call_count": valid_call_count,
        "inherited_call_count": inherited_call_count,
        "priced_call_count": calculated_count,
        "reported_call_count": reported_count,
        "ledger_available": bool(calls),
        "label": (
            "Calculated public list price is shown only where billable usage is known; "
            "provider invoice may differ. Inherited calls were incurred in parent Runs."
        ),
        "breakdown": [
            {
                **row,
                "estimated_usd": (
                    round(float(row["estimated_usd"]), 6)
                    if row["estimated_calls"]
                    else None
                ),
                "actual_usd": (
                    round(float(row["actual_usd"]), 6)
                    if row["reported_calls"]
                    else None
                ),
                "unresolved_usd": round(float(row["unresolved_usd"]), 6),
            }
            for row in sorted(rows.values(), key=lambda value: (value["backend_id"], value["task_id"]))
        ],
    }


def _current_stage(manifest: Mapping[str, Any]) -> str | None:
    stages = manifest.get("stages") if isinstance(manifest.get("stages"), dict) else {}
    for stage in PUBLIC_STAGES:
        record = stages.get(stage)
        if isinstance(record, dict) and record.get("status") == "running":
            return stage
    for stage in PUBLIC_STAGES:
        record = stages.get(stage)
        if not isinstance(record, dict) or record.get("status") != "complete":
            return stage
    return None


def run_summary(
    project_root: Path,
    run_root: Path,
    supervisor: RunSupervisor,
    *,
    manifest: Mapping[str, Any] | None = None,
    brief: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = manifest if manifest is not None else _read_optional(run_root / "manifest.json")
    if not manifest:
        raise ValueError("invalid manifest")
    brief = brief if brief is not None else _read_optional(run_root / "inputs" / "brief.json")
    config = config if config is not None else _read_optional(run_root / "inputs" / "config.resolved.json")
    run_id = str(manifest.get("run_id") or run_root.name)
    controller, effective_status = _controller_state(
        project_root,
        run_id,
        str(manifest.get("status") or "unknown"),
        supervisor,
    )
    stages = manifest.get("stages") if isinstance(manifest.get("stages"), dict) else {}
    complete_count = sum(
        1
        for stage in PUBLIC_STAGES
        if isinstance(stages.get(stage), dict) and stages[stage].get("status") == "complete"
    )
    return {
        "run_id": run_id,
        "manifest_status": manifest.get("status"),
        "status": effective_status,
        "controller": controller,
        "created_at": manifest.get("created_at"),
        "updated_at": manifest.get("updated_at"),
        "parent_run_id": manifest.get("parent_run_id"),
        "profile": config.get("profile"),
        "quality": config.get("quality"),
        "output_language": config.get("output_language"),
        "duration_seconds": config.get("duration_seconds"),
        "idea_direction": brief.get("idea_direction") or "Surprise me",
        "current_stage": _current_stage(manifest),
        "completed_stage_count": complete_count,
        "total_stage_count": len(PUBLIC_STAGES),
        "progress": round(complete_count / len(PUBLIC_STAGES), 4),
        "warnings": manifest.get("warnings") or [],
        "cost": cost_summary(manifest),
    }


def list_runs(project_root: Path, supervisor: RunSupervisor) -> list[dict[str, Any]]:
    runs_root = project_root / "runs"
    if not runs_root.is_dir():
        return []
    result = []
    for run_root in runs_root.iterdir():
        if not run_root.is_dir() or run_root.name.startswith("."):
            continue
        try:
            canonical_root = resolve_run_root(project_root, run_root.name)
            result.append(run_summary(project_root, canonical_root, supervisor))
        except (FileNotFoundError, OSError, ValueError):
            continue
    return sorted(
        result,
        key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""),
        reverse=True,
    )


def _stage_artifact(run_root: Path, manifest: Mapping[str, Any], stage: str) -> dict[str, Any]:
    stages = manifest.get("stages") if isinstance(manifest.get("stages"), dict) else {}
    record = stages.get(stage)
    if not isinstance(record, dict):
        return {}
    paths = [path for path in record.get("output_paths") or [] if isinstance(path, str)]
    paths.sort(key=lambda value: (not value.endswith("aggregate.json"), not value.endswith("artifact.json")))
    for relative in paths:
        if not relative.endswith(("aggregate.json", "artifact.json")):
            continue
        try:
            path = resolve_artifact_path(run_root, relative)
        except (ValueError, FileNotFoundError):
            continue
        value = _read_optional(path)
        if value:
            return value
    return {}


def _by_scene(values: Any, *, nested_scene_key: str = "scene_id") -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not isinstance(values, list):
        return result
    for item in values:
        if not isinstance(item, dict):
            continue
        scene_id = item.get(nested_scene_key)
        if isinstance(scene_id, str):
            result[scene_id] = item
    return result


def _by_visual(values: Any) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not isinstance(values, list):
        return result
    for item in values:
        if not isinstance(item, dict):
            continue
        visual_id = item.get("shot_id") or item.get("scene_id")
        if isinstance(visual_id, str):
            result[visual_id] = item
    return result


def _media_url(project_root: Path, run_root: Path, run_id: str, value: Any) -> str | None:
    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        return None
    candidate = (project_root / value["path"]).resolve()
    try:
        relative = candidate.relative_to(run_root).as_posix()
    except ValueError:
        return None
    return f"/api/runs/{quote(run_id)}/files/{quote(relative, safe='/')}"


def scene_views(
    project_root: Path,
    run_root: Path,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    narration = _stage_artifact(run_root, manifest, "narration")
    visual_plan = _stage_artifact(run_root, manifest, "visual-plan")
    prompt_set = _stage_artifact(run_root, manifest, "image-prompt-compile")
    images = _stage_artifact(run_root, manifest, "images")
    review = _stage_artifact(run_root, manifest, "visual-review")
    run_id = str(manifest.get("run_id") or run_root.name)

    script = narration.get("script") if isinstance(narration.get("script"), dict) else {}
    scripts = _by_scene(script.get("scenes"))
    timeline = narration.get("timeline") if isinstance(narration.get("timeline"), dict) else {}
    timings = _by_scene(timeline.get("scenes"))
    visual_values = visual_plan.get("shots") or visual_plan.get("scenes")
    briefs = _by_visual(visual_values)
    requests = _by_visual(prompt_set.get("requests"))
    final_image_bundle = review.get("images") if isinstance(review.get("images"), dict) else images
    image_items: dict[str, dict[str, Any]] = {}
    for item in final_image_bundle.get("items") or []:
        if not isinstance(item, dict):
            continue
        generated = item.get("generated") if isinstance(item.get("generated"), dict) else {}
        visual_id = generated.get("shot_id") or generated.get("scene_id")
        if isinstance(visual_id, str):
            image_items[visual_id] = item
    remotion_bundle = (
        review.get("assets")
        if isinstance(review.get("assets"), dict)
        else images
    )
    for asset in remotion_bundle.get("assets") or []:
        if not isinstance(asset, dict) or not isinstance(asset.get("shot_id"), str):
            continue
        image_items[asset["shot_id"]] = {
            "normalized_image": asset.get("normalized"),
            "asset": asset,
        }
    report = review.get("report") if isinstance(review.get("report"), dict) else {}
    review_items = _by_visual(report.get("items"))
    visual_ids = list(dict.fromkeys([*briefs, *requests, *image_items])) or list(scripts)
    result = []
    for visual_id in visual_ids:
        image_item = image_items.get(visual_id, {})
        normalized = image_item.get("normalized_image")
        remotion_asset = (
            image_item.get("asset") if isinstance(image_item.get("asset"), dict) else {}
        )
        brief = briefs.get(visual_id, {})
        request = image_item.get("request") or requests.get(visual_id) or {}
        scene_id = str(brief.get("scene_id") or request.get("scene_id") or visual_id)
        shot_id = brief.get("shot_id") or request.get("shot_id")
        timing = timings.get(scene_id)
        if isinstance(shot_id, str) and isinstance(brief, dict):
            timing = {
                "scene_id": scene_id,
                "shot_id": shot_id,
                "start_seconds": brief.get("start_seconds"),
                "end_seconds": brief.get("end_seconds"),
                "audio": timing.get("audio") if isinstance(timing, dict) else None,
            }
        script_item = scripts.get(scene_id)
        if isinstance(shot_id, str) and isinstance(script_item, dict):
            script_item = {
                **script_item,
                "spoken_text": brief.get("narration_excerpt") or script_item.get("spoken_text"),
            }
        result.append(
            {
                "visual_id": visual_id,
                "scene_id": scene_id,
                "shot_id": shot_id,
                "script": script_item,
                "timing": timing,
                "audio_url": _media_url(
                    project_root,
                    run_root,
                    run_id,
                    timing.get("audio") if isinstance(timing, dict) else None,
                ),
                "visual_brief": brief or None,
                "image_request": request or None,
                "image": normalized,
                "image_url": _media_url(project_root, run_root, run_id, normalized),
                "media_kind": remotion_asset.get("media_kind") or "image",
                "asset": remotion_asset or None,
                "review": review_items.get(visual_id),
            }
        )
    return result


def artifact_entries(run_root: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    stages = manifest.get("stages") if isinstance(manifest.get("stages"), dict) else {}
    recorded = {
        relative
        for stage in stages.values()
        if isinstance(stage, dict)
        for relative in (stage.get("output_hashes") or {})
    }
    recorded.update(
        {
            "inputs/config.resolved.json",
            "inputs/brief.json",
            "inputs/frozen-assets/assets.json",
        }
    )
    for item_record_path in (run_root / "stages").glob("*/item-records/*.json"):
        item_record = _read_optional(item_record_path)
        hashes = item_record.get("output_hashes")
        if isinstance(hashes, dict):
            recorded.update(str(relative) for relative in hashes)
    run_id = str(manifest.get("run_id") or run_root.name)
    entries = []
    for path in run_root.rglob("*"):
        if not path.is_file():
            continue
        try:
            resolved = path.resolve()
            relative = resolved.relative_to(run_root).as_posix()
            stat = resolved.stat()
        except (OSError, ValueError):
            continue
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        entries.append(
            {
                "path": relative,
                "name": path.name,
                "category": relative.split("/", 1)[0],
                "size_bytes": stat.st_size,
                "modified_at": stat.st_mtime,
                "mime_type": mime_type,
                "hash_recorded": relative in recorded,
                "url": f"/api/runs/{quote(run_id)}/files/{quote(relative, safe='/')}",
            }
        )
    return sorted(entries, key=lambda item: item["path"])


def run_detail(project_root: Path, run_root: Path, supervisor: RunSupervisor) -> dict[str, Any]:
    manifest = _read_optional(run_root / "manifest.json")
    if not manifest:
        raise ValueError("invalid manifest")
    config = _read_optional(run_root / "inputs" / "config.resolved.json")
    brief = _read_optional(run_root / "inputs" / "brief.json")
    safe_config = {
        "content_mode": "fiction",
        "content_format": "narrative",
        "narration_pace": "standard",
        "video_style": "still_image",
        "remotion_asset_policy": "stock_preferred",
        "remotion_allow_share_alike": False,
        "remotion_require_asset_approval": False,
        "remotion_source_screenshot_hosts": [],
        "visual_shot_mode": "scene_locked",
        **{
            key: value
            for key, value in config.items()
            if key not in {"voice", "project_root"}
        },
    }
    frozen_assets = _read_optional(run_root / "inputs" / "frozen-assets" / "assets.json")
    images = _stage_artifact(run_root, manifest, "images")
    review = _stage_artifact(run_root, manifest, "visual-review")
    reviewed_bundle = review.get("assets") if isinstance(review.get("assets"), dict) else {}
    current_assets = (
        reviewed_bundle.get("assets")
        if isinstance(reviewed_bundle.get("assets"), list)
        else images.get("assets")
        if isinstance(images.get("assets"), list)
        else []
    )
    expected_approvals = {
        (
            asset.get("asset_id"),
            asset.get("shot_id"),
            hash_value(asset),
        )
        for asset in current_assets
        if isinstance(asset, dict)
    }
    approved_records = {
        (
            item.get("asset_id"),
            item.get("shot_id"),
            item.get("record_sha256"),
        )
        for item in frozen_assets.get("remotion_asset_approvals") or []
        if isinstance(item, dict)
    }
    approval_required = bool(safe_config.get("remotion_require_asset_approval"))
    files = artifact_entries(run_root, manifest)
    return {
        "summary": run_summary(
            project_root,
            run_root,
            supervisor,
            manifest=manifest,
            brief=brief,
            config=config,
        ),
        "manifest": manifest,
        "config": safe_config,
        "brief": brief,
        "scenes": scene_views(project_root, run_root, manifest),
        "files": files,
        "outputs": [item for item in files if item["category"] == "outputs"],
        "asset_approval": {
            "required": approval_required,
            "approved": bool(expected_approvals) and expected_approvals.issubset(approved_records),
            "asset_count": len(expected_approvals),
        },
        "cost": cost_summary(manifest),
    }
