from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .config import active_backend_ids, find_project_root, load_brief, load_environment, resolve_config
from .contracts import (
    CreativeBrief,
    OutputLanguage,
    PreflightReport,
    PUBLIC_STAGES,
    Quality,
    ResolvedRunConfig,
    VideoOrientation,
    VideoStyle,
)
from .errors import ConfigurationError, ErrorKind, VideoGeneratorError
from .preflight import run_preflight
from .provenance import build_runtime_snapshot, verify_runtime_snapshot
from .remotion_renderer import setup_remotion_runtime
from .profiles import BACKEND_DESCRIPTORS, PROFILES
from .prompting import build_frozen_assets, canonical_backend_descriptor_payload
from .run_store import RunStore, TASK_STAGE_IMPACT, earliest_config_impact
from .setup import (
    CURATED_LLM_CANDIDATES,
    download_curated_llm_candidate,
    selected_backends,
    setup_backends,
)
from .util import hash_value
from .workflow import WorkflowEngine


EXIT_INTERNAL = 1
EXIT_CONFIG = 2
EXIT_NOT_READY = 3
EXIT_BUDGET = 4
EXIT_INVALID_OUTPUT = 5


def _configure_console_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="replace")
        except (OSError, ValueError):
            continue


def _print_preflight(report: PreflightReport) -> None:
    print(f"Preflight: {'READY' if report.ready else 'NOT READY'}")
    print(f"Profile: {report.profile} | Language: {report.output_language.value}")
    for item in report.checks:
        print(f"  [{'OK' if item.ready else 'FAIL'}] {item.name}: {item.detail}")
        if item.action and not item.ready:
            print(f"         Action: {item.action}")
    for backend in report.backend_reports:
        print(f"  [{'OK' if backend.ready else 'FAIL'}] Backend {backend.backend_id}")
        for item in backend.items:
            print(f"         [{'OK' if item.ready else 'FAIL'}] {item.name}: {item.detail}")
            if item.action and not item.ready:
                print(f"                Action: {item.action}")
    print(
        f"Estimated cloud reservation: ${report.cost.estimated_usd:.2f} remaining, "
        f"${report.cost.projected_total_usd:.2f} projected total "
        f"of ${report.cost.ceiling_usd:.2f} ({report.cost.basis})"
    )
    for warning in report.warnings:
        print(f"  Warning: {warning}")


def _require_ready(report: PreflightReport) -> None:
    if report.ready:
        return
    failed_checks = [item for item in report.checks if not item.ready]
    failed_backends = [item.backend_id for item in report.backend_reports if not item.ready]
    parts = [item.name for item in failed_checks] + [f"Backend {item}" for item in failed_backends]
    actions = [item.action for item in failed_checks if item.action]
    for backend in report.backend_reports:
        actions.extend(item.action for item in backend.items if not item.ready and item.action)
    cost_only = bool(failed_checks) and all(item.name == "cost_ceiling" for item in failed_checks) and not failed_backends
    raise VideoGeneratorError(
        "Preflight failed: " + ", ".join(parts),
        kind=ErrorKind.BUDGET_EXCEEDED if cost_only else ErrorKind.NOT_READY,
        action=" ".join(dict.fromkeys(actions)) or "Resolve the failed Preflight checks and try again.",
    )


def _preflight_exit_code(report: PreflightReport) -> int:
    if report.ready:
        return 0
    failed_checks = [item for item in report.checks if not item.ready]
    if (
        failed_checks
        and all(item.name == "cost_ceiling" for item in failed_checks)
        and all(item.ready for item in report.backend_reports)
    ):
        return EXIT_BUDGET
    return EXIT_NOT_READY


def _resolve_run_dir(value: str, project_root: Path) -> Path:
    supplied = Path(value).expanduser()
    if supplied.exists():
        candidate = supplied.resolve()
    else:
        candidate = (project_root / "runs" / value).resolve()
    runs_root = (project_root / "runs").resolve()
    try:
        candidate.relative_to(runs_root)
    except ValueError as exc:
        raise ConfigurationError(f"Run directory must stay under {runs_root}: {candidate}") from exc
    if not (candidate / "manifest.json").is_file():
        raise ConfigurationError(f"Run Bundle does not exist or is incomplete: {candidate}")
    return candidate


def _execute_locked(store: RunStore, environment: dict[str, str], *, stop_after: str | None) -> int:
    try:
        with WorkflowEngine(store=store, environment=environment, stop_after=stop_after) as engine:
            delivery = engine.run()
    except VideoGeneratorError as exc:
        if store.manifest.status in {"created", "running"}:
            store.set_status("failed", exc)
        raise
    except KeyboardInterrupt:
        store.set_status("stopped")
        raise
    print(f"Run Bundle: {store.root}")
    if delivery is None:
        print(f"Run stopped after {stop_after}.")
    else:
        print(f"Completed in {delivery.duration_seconds:.2f} seconds.")
        for output in delivery.outputs:
            print(f"  Output ({output.role}): {Path(store.config.project_root) / output.media.path}")
    return 0


def _execute(store: RunStore, environment: dict[str, str], *, stop_after: str | None) -> int:
    with store.execution_lock():
        return _execute_locked(store, environment, stop_after=stop_after)


def _fresh_assets(config: ResolvedRunConfig, report: PreflightReport | None = None) -> dict[str, Any]:
    assets = build_frozen_assets(config)
    assets["runtime_snapshot"] = build_runtime_snapshot(config)
    if report is not None:
        assets["creation_preflight"] = report.model_dump(mode="json")
    return assets


def _command_setup(args: argparse.Namespace) -> int:
    config_path = args.config.resolve() if args.config else find_project_root(Path.cwd()) / "config.toml"
    project_root = find_project_root(config_path.parent)
    environment = load_environment(config_path)
    if args.config and not args.profile and not args.backend:
        config = resolve_config(args.config)
        backend_ids = sorted(active_backend_ids(config))
    else:
        try:
            backend_ids = selected_backends(profile=args.profile, backend_id=args.backend)
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc
    results = setup_backends(
        project_root=project_root,
        backend_ids=backend_ids,
        environment=environment,
        download=not args.no_download,
        wsl_distribution=args.wsl_distro,
        llm_profile=args.llm_profile,
    )
    if args.config:
        setup_config = resolve_config(args.config)
        if setup_config.video_style is VideoStyle.REMOTION_EXPLAINER:
            results.extend(
                setup_remotion_runtime(
                    project_root,
                    download=not args.no_download,
                    environment=environment,
                )
            )
    for item in results:
        print(f"[{'OK' if item.ready else 'FAIL'}] {item.name}: {item.detail}")
        if item.action:
            print(f"       Action: {item.action}")
    if not all(item.ready for item in results):
        raise VideoGeneratorError(
            "Setup completed with unprepared Backends",
            kind=ErrorKind.NOT_READY,
            action="Follow the reported Backend actions, then rerun Setup.",
        )
    return 0


def _command_models_list(args: argparse.Namespace) -> int:
    del args
    for candidate in CURATED_LLM_CANDIDATES.values():
        print(
            f"{candidate.candidate_id}: {candidate.model_id}, {candidate.quantization}, "
            f"MTP {candidate.mtp}, approximately {candidate.estimated_download_gb:.1f} GB"
        )
        print(f"  {candidate.repository}@{candidate.revision}")
        for artifact in candidate.artifacts:
            print(f"  {artifact.role}: {artifact.filename}")
    return 0


def _command_models_download(args: argparse.Namespace) -> int:
    project_root = find_project_root(Path.cwd())
    candidate = CURATED_LLM_CANDIDATES[args.candidate]
    destination = project_root / ".cache" / "models" / "llm" / candidate.candidate_id
    print(
        f"Candidate: {candidate.candidate_id} ({candidate.quantization}, MTP {candidate.mtp}, "
        f"approximately {candidate.estimated_download_gb:.1f} GB)"
    )
    print(f"Source: https://huggingface.co/{candidate.repository}/tree/{candidate.revision}")
    print(f"Destination: {destination}")
    if args.dry_run:
        print("Dry run only; no files were downloaded or written.")
        return 0
    environment = load_environment(project_root / "config.toml")
    prepared = download_curated_llm_candidate(
        project_root=project_root,
        candidate_id=candidate.candidate_id,
        environment=environment,
    )
    print(f"Downloaded and SHA-256 verified: {prepared}")
    print("The stock llama.cpp runtime is separate and is not downloaded by this command.")
    return 0


def _command_preflight(args: argparse.Namespace) -> int:
    config = resolve_config(args.config)
    environment = load_environment(args.config)
    report = run_preflight(config=config, environment=environment, live=args.live)
    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        _print_preflight(report)
    return _preflight_exit_code(report)


def _command_dashboard(args: argparse.Namespace) -> int:
    from .dashboard import run_dashboard

    project_root = find_project_root(Path.cwd())
    if not (project_root / "config.toml").is_file():
        raise ConfigurationError(f"dashboard requires config.toml under {project_root}")
    run_dashboard(project_root, port=args.port)
    return 0


def _command_generate(args: argparse.Namespace) -> int:
    overrides: dict[str, Any] = {
        "profile": args.profile,
        "output_language": args.language,
        "duration_seconds": args.duration_seconds,
        "orientation": args.orientation,
    }
    if args.offline is not None:
        overrides["offline"] = args.offline
    if args.video_style is not None:
        overrides["video_style"] = args.video_style
    config = resolve_config(args.config, overrides=overrides)
    brief = load_brief(args.brief)
    environment = load_environment(args.config)
    report = run_preflight(config=config, environment=environment, live=args.live_preflight)
    _print_preflight(report)
    _require_ready(report)
    store = RunStore.create(
        project_root=Path(config.project_root),
        config=config,
        brief=brief,
        frozen_assets=_fresh_assets(config, report),
    )
    return _execute(store, environment, stop_after=args.stop_after)


def _next_incomplete_stage(store: RunStore) -> str | None:
    for stage in PUBLIC_STAGES:
        record = store.stage_record(stage)
        if record is None or record.status != "complete":
            return stage
    return None


def _completed_call_counts(store: RunStore) -> dict[str, int]:
    narration = store.completed_item_ids("narration")
    visual_plan = store.completed_item_ids("visual-plan")
    visual_review = store.completed_item_ids("visual-review")
    music_brief = store.completed_item_ids("music-brief")
    music_brief_stage = store.stage_record("music-brief")
    ledger_counts = {
        task_id: sum(
            1 for call in store.manifest.cloud_calls if call.task_id == task_id
        )
        for task_id in ("remotion_direction", "remotion_asset_select")
    }
    return {
        "search": len(store.completed_item_ids("research")),
        "narration_synthesis": sum(
            1 for item_id in narration if item_id != "duration-repair-script"
        ),
        "duration_repair": int("duration-repair-script" in narration),
        "caption_alignment": len(store.completed_item_ids("captions")),
        "remotion_rhythm": int("rhythm" in visual_plan),
        "image_prompt_compile": len(store.completed_item_ids("image-prompt-compile")),
        "image_generate": len(store.completed_item_ids("images"))
        + sum(1 for item_id in visual_review if item_id.endswith("-regeneration")),
        "visual_review": sum(1 for item_id in visual_review if "-review-" in item_id),
        **ledger_counts,
        "music_brief": int(
            "brief" in music_brief
            or (music_brief_stage is not None and music_brief_stage.status == "complete")
        ),
    }


def _command_resume(args: argparse.Namespace) -> int:
    project_root = find_project_root(Path.cwd())
    run_root = _resolve_run_dir(args.run, project_root)
    initial_store = RunStore.open(run_root)
    with initial_store.execution_lock():
        store = RunStore.open(run_root)
        if store.manifest.status == "complete":
            store.validate_completed_outputs()
            print(f"Run is already complete: {store.root}")
            return 0
        config = store.config
        verify_runtime_snapshot(config, store.frozen_assets)
        environment = load_environment(Path(config.project_root) / "config.toml")
        next_stage = _next_incomplete_stage(store) or PUBLIC_STAGES[-1]
        if args.stop_after and PUBLIC_STAGES.index(args.stop_after) < PUBLIC_STAGES.index(next_stage):
            raise ConfigurationError(
                f"--stop-after {args.stop_after} is before the next incomplete stage {next_stage}"
            )
        report = run_preflight(
            config=config,
            environment=environment,
            live=args.live_preflight,
            run_root=store.root,
            from_stage=next_stage,
            already_reserved_usd=float(store.manifest.reserved_cost_usd),
            completed_calls=_completed_call_counts(store),
        )
        _print_preflight(report)
        _require_ready(report)
        return _execute_locked(store, environment, stop_after=args.stop_after)


def _earliest_frozen_impact(
    old: dict[str, Any], new: dict[str, Any], config: ResolvedRunConfig
) -> str | None:
    stages: list[str] = []
    for section in ("prompts", "schemas"):
        old_values = old.get(section, {}) if isinstance(old.get(section), dict) else {}
        new_values = new.get(section, {}) if isinstance(new.get(section), dict) else {}
        for task_id in set(old_values) | set(new_values):
            if old_values.get(task_id) != new_values.get(task_id) and task_id in TASK_STAGE_IMPACT:
                stages.append(TASK_STAGE_IMPACT[task_id])
    old_targets = old.get("image_targets", {})
    new_targets = new.get("image_targets", {})
    image_backend = config.task_bindings["image_generate"]
    if (
        isinstance(old_targets, dict)
        and isinstance(new_targets, dict)
        and old_targets.get(image_backend) != new_targets.get(image_backend)
    ):
        stages.append("image-prompt-compile")

    old_profile = old.get("profile", {}) if isinstance(old.get("profile"), dict) else {}
    new_profile = new.get("profile", {}) if isinstance(new.get("profile"), dict) else {}
    old_descriptors = old_profile.get("backend_descriptors", {})
    new_descriptors = new_profile.get("backend_descriptors", {})
    if isinstance(old_descriptors, dict) and isinstance(new_descriptors, dict):
        for task_id, backend_id in config.task_bindings.items():
            old_descriptor = old_descriptors.get(backend_id)
            new_descriptor = new_descriptors.get(backend_id)
            if isinstance(old_descriptor, dict):
                old_descriptor = canonical_backend_descriptor_payload(old_descriptor)
            if isinstance(new_descriptor, dict):
                new_descriptor = canonical_backend_descriptor_payload(new_descriptor)
            if old_descriptor != new_descriptor:
                stages.append(TASK_STAGE_IMPACT[task_id])
    old_runtime = old.get("runtime_snapshot", {}) if isinstance(old.get("runtime_snapshot"), dict) else {}
    new_runtime = new.get("runtime_snapshot", {}) if isinstance(new.get("runtime_snapshot"), dict) else {}
    if old_runtime.get("code_hash") != new_runtime.get("code_hash"):
        stages.append("research")
    if old_runtime.get("private_inputs") != new_runtime.get("private_inputs"):
        stages.append("narration")
    if old_runtime.get("ffmpeg") != new_runtime.get("ffmpeg") or old_runtime.get("ffprobe") != new_runtime.get("ffprobe"):
        stages.append("narration")
    if old_runtime.get("remotion") != new_runtime.get("remotion"):
        stages.append("images")
    old_runners = old_runtime.get("runner_manifests", {})
    new_runners = new_runtime.get("runner_manifests", {})
    if isinstance(old_runners, dict) and isinstance(new_runners, dict):
        for task_id, backend_id in config.task_bindings.items():
            if old_runners.get(backend_id) != new_runners.get(backend_id):
                stages.append(TASK_STAGE_IMPACT[task_id])
    return min(stages, key=PUBLIC_STAGES.index) if stages else None


def _earliest_rerun_impact(
    parent: RunStore,
    config: ResolvedRunConfig,
    brief: CreativeBrief,
    assets: dict[str, Any],
) -> str | None:
    stages: list[str] = []
    config_impact = earliest_config_impact(parent.config, config)
    if config_impact:
        stages.append(config_impact)
    if hash_value(parent.brief.model_dump(mode="json")) != hash_value(brief.model_dump(mode="json")):
        stages.append("research")
    frozen_impact = _earliest_frozen_impact(parent.frozen_assets, assets, config)
    if frozen_impact:
        stages.append(frozen_impact)
    return min(stages, key=PUBLIC_STAGES.index) if stages else None


def _command_rerun(args: argparse.Namespace) -> int:
    project_root = find_project_root(Path.cwd())
    parent_root = _resolve_run_dir(args.run, project_root)
    initial_parent = RunStore.open(parent_root)
    if args.stop_after and PUBLIC_STAGES.index(args.stop_after) < PUBLIC_STAGES.index(args.from_stage):
        raise ConfigurationError(
            f"--stop-after {args.stop_after} is before the rerun fork stage {args.from_stage}"
        )
    with initial_parent.execution_lock():
        parent = RunStore.open(parent_root)
        config = resolve_config(args.config) if args.config else parent.config
        if Path(config.project_root).resolve() != Path(parent.config.project_root).resolve():
            raise ConfigurationError(
                "rerun config belongs to a different project root",
                action="Use a config.toml from the parent Run's project.",
            )
        brief = load_brief(args.brief) if args.brief else parent.brief
        assets = _fresh_assets(config)
        impact = _earliest_rerun_impact(parent, config, brief, assets)
        if impact and PUBLIC_STAGES.index(impact) < PUBLIC_STAGES.index(args.from_stage):
            raise ConfigurationError(
                f"the requested changes invalidate work from {impact}, before requested stage {args.from_stage}",
                action=f"Rerun with --from {impact} or restore the parent configuration/assets.",
            )
        environment_path = args.config if args.config else Path(config.project_root) / "config.toml"
        environment = load_environment(environment_path)
        report = run_preflight(
            config=config,
            environment=environment,
            live=args.live_preflight,
            from_stage=args.from_stage,
        )
        affected = PUBLIC_STAGES[PUBLIC_STAGES.index(args.from_stage) :]
        print(f"Parent Run: {parent.root}")
        print(f"Fork stage: {args.from_stage}")
        print("Affected stages: " + ", ".join(affected))
        _print_preflight(report)
        _require_ready(report)
        if args.dry_run:
            print("Dry run only; no child Run Bundle was created.")
            return 0
        assets["creation_preflight"] = report.model_dump(mode="json")
        child = RunStore.fork(
            parent=parent,
            config=config,
            brief=brief,
            frozen_assets=assets,
            fork_stage=args.from_stage,
        )
    return _execute(child, environment, stop_after=args.stop_after)


def _evaluation_brief(language: OutputLanguage, suite: str) -> CreativeBrief:
    if language is OutputLanguage.FINNISH:
        direction = (
            "Lämmin, omaperäinen talvitarina pienestä rohkeasta teosta, oudosta esineestä ja "
            "odottamattomasta ystävyydestä"
        )
        tone = "utelias, lämmin ja kevyesti jännittävä"
        must_include = [
            "yksi visuaalisesti mieleenpainuva esine",
            "selkeä syy-seuraus-loppu",
            "nimet Yrjö ja Åke sekä luku kaksikymmentäseitsemän luontevassa puheessa",
        ]
    else:
        direction = (
            "A warm, original winter story about a small act of courage, a strange object, and an "
            "unexpected friendship"
        )
        tone = "curious, warm, and lightly suspenseful"
        must_include = [
            "one visually memorable object",
            "a clear causal ending",
            "a person's name and a two-digit number in natural speech",
        ]
    return CreativeBrief(
        idea_direction=direction,
        tone=tone,
        themes=["resourcefulness", "unexpected friendship"],
        must_include=must_include,
        avoid=["chosen-one plot", "an announced moral"],
        research_focus=["cold-weather physical details", "winter folklore motifs"],
    )


def _command_evaluate(args: argparse.Namespace) -> int:
    languages = [OutputLanguage(args.language)] if args.language else list(OutputLanguage)
    duration = 30 if args.suite == "smoke" else 90
    quality = Quality.FINAL if args.suite == "quality" else Quality.DRAFT
    run_roots: list[Path] = []
    for language in languages:
        config = resolve_config(
            args.config,
            overrides={
                "profile": args.profile,
                "output_language": language,
                "duration_seconds": duration,
                "quality": quality,
            },
        )
        environment = load_environment(args.config)
        report = run_preflight(config=config, environment=environment, live=args.live_preflight)
        print(f"Evaluation case: {args.suite}/{language.value}")
        _print_preflight(report)
        _require_ready(report)
        brief = _evaluation_brief(language, args.suite)
        store = RunStore.create(
            project_root=Path(config.project_root),
            config=config,
            brief=brief,
            frozen_assets=_fresh_assets(config, report),
        )
        _execute(store, environment, stop_after=None)
        run_roots.append(store.root)
    print("Evaluation Run Bundles:")
    for root in run_roots:
        print(f"  {root}")
    return 0


def _command_runs_prune(args: argparse.Namespace) -> int:
    project_root = find_project_root(Path.cwd())
    runs_root = (project_root / "runs").resolve()
    if not runs_root.exists():
        print("No Run Bundles exist.")
        return 0
    now = datetime.now(timezone.utc)
    stores: dict[str, RunStore] = {}
    for manifest_path in runs_root.glob("*/manifest.json"):
        try:
            store = RunStore.open(manifest_path.parent)
        except Exception as exc:
            print(f"Skipping invalid Run Bundle {manifest_path.parent}: {exc}", file=sys.stderr)
            continue
        stores[store.manifest.run_id] = store
    candidates = {
        run_id
        for run_id, store in stores.items()
        if (now - store.manifest.updated_at).total_seconds() >= args.older_than * 86400
        and (args.include_incomplete or store.manifest.status == "complete")
    }
    protected: set[str] = set()
    frontier = [run_id for run_id in stores if run_id not in candidates]
    while frontier:
        run_id = frontier.pop()
        parent_id = stores[run_id].manifest.parent_run_id
        if parent_id and parent_id in stores and parent_id not in protected:
            protected.add(parent_id)
            frontier.append(parent_id)
    removable = sorted(candidates - protected)
    if not removable:
        print("No Run Bundles match the prune policy.")
        return 0
    print("Run Bundles selected for pruning:")
    pruned = 0
    for run_id in removable:
        print(f"  {stores[run_id].root}")
    if not args.yes:
        print("Dry run only. Add --yes to delete these Run Bundles.")
        return 0
    for run_id in removable:
        root = stores[run_id].root.resolve()
        if root.parent != runs_root:
            raise ConfigurationError(f"refusing to prune path outside the direct runs/ directory: {root}")
        lock = stores[run_id].execution_lock()
        try:
            with lock:
                shutil.rmtree(root)
        except VideoGeneratorError as exc:
            print(f"Skipping active Run {run_id}: {exc.message}", file=sys.stderr)
            continue
        lock.path.unlink(missing_ok=True)
        pruned += 1
    print(f"Pruned {pruned} Run Bundle(s). Model caches and private inputs were untouched.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-generator",
        description="Generate narrated still-image or Remotion videos with per-task local or cloud Backends.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    setup = commands.add_parser("setup", help="prepare selected local runtimes/models and check cloud keys")
    setup_choice = setup.add_mutually_exclusive_group()
    setup_choice.add_argument("--profile", choices=sorted(name for name in PROFILES if name != "deterministic-test"))
    setup_choice.add_argument("--backend", choices=sorted(BACKEND_DESCRIPTORS))
    setup.add_argument("--config", type=Path, help="prepare only Backends active in this resolved config")
    setup.add_argument("--no-download", action="store_true", help="verify/use existing assets without model downloads")
    setup.add_argument("--wsl-distro", default="Ubuntu", help="explicit WSL2 distribution for WSL-only runners")
    setup.add_argument(
        "--llm-profile",
        type=Path,
        help="typed profile for a pinned GGUF, stock llama-server.exe, hashes, context, and MTP mode",
    )
    setup.set_defaults(handler=_command_setup)

    models = commands.add_parser("models", help="inspect or download curated local LLM candidates")
    model_commands = models.add_subparsers(dest="models_command", required=True)
    model_list = model_commands.add_parser("list", help="list pinned local LLM benchmark candidates")
    model_list.set_defaults(handler=_command_models_list)
    model_download = model_commands.add_parser(
        "download",
        help="download one exact candidate into the project model cache and verify its hash",
    )
    model_download.add_argument("candidate", choices=sorted(CURATED_LLM_CANDIDATES))
    model_download.add_argument(
        "--dry-run",
        action="store_true",
        help="show the pinned source, revision, estimated size, and destination without writing files",
    )
    model_download.set_defaults(handler=_command_models_download)

    preflight = commands.add_parser("preflight", help="perform read-only readiness and cost checks")
    preflight.add_argument("--config", type=Path, required=True)
    preflight.add_argument("--live", action="store_true", help="make provider readiness probes; may use network")
    preflight.add_argument("--json", action="store_true", help="emit the typed report as JSON")
    preflight.set_defaults(handler=_command_preflight)

    dashboard = commands.add_parser(
        "dashboard",
        help="open the local FastAPI control plane for Runs, models, costs, and artifacts",
    )
    dashboard.add_argument("--port", type=int, default=8765)
    dashboard.set_defaults(handler=_command_dashboard)

    generate = commands.add_parser("generate", help="run the complete workflow")
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--brief", type=Path, required=True)
    generate.add_argument("--profile", choices=sorted(name for name in PROFILES if name != "deterministic-test"))
    generate.add_argument("--language", choices=[item.value for item in OutputLanguage])
    generate.add_argument("--duration-seconds", type=float)
    generate.add_argument("--orientation", choices=[item.value for item in VideoOrientation])
    generate.add_argument("--video-style", choices=[item.value for item in VideoStyle])
    generate.add_argument("--offline", action=argparse.BooleanOptionalAction, default=None)
    generate.add_argument("--stop-after", choices=PUBLIC_STAGES)
    generate.add_argument("--live-preflight", action="store_true")
    generate.set_defaults(handler=_command_generate)

    resume = commands.add_parser("resume", help="continue a durable Run without repeating valid checkpoints")
    resume.add_argument("run", help="Run ID or path under runs/")
    resume.add_argument("--stop-after", choices=PUBLIC_STAGES)
    resume.add_argument("--live-preflight", action="store_true")
    resume.set_defaults(handler=_command_resume)

    rerun = commands.add_parser("rerun", help="create a parent-linked Run from an explicit stage")
    rerun.add_argument("run", help="parent Run ID or path under runs/")
    rerun.add_argument("--from", dest="from_stage", choices=PUBLIC_STAGES, required=True)
    rerun.add_argument("--config", type=Path)
    rerun.add_argument("--brief", type=Path)
    rerun.add_argument("--stop-after", choices=PUBLIC_STAGES)
    rerun.add_argument("--live-preflight", action="store_true")
    rerun.add_argument("--dry-run", action="store_true", help="show invalidation/readiness/cost without creating a Run")
    rerun.set_defaults(handler=_command_rerun)

    evaluate = commands.add_parser(
        "evaluate", help="run fixed English/Finnish smoke, draft-quality, or quality fixtures"
    )
    evaluate.add_argument("--suite", choices=["smoke", "draft-quality", "quality"], required=True)
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--profile", choices=sorted(name for name in PROFILES if name != "deterministic-test"))
    evaluate.add_argument("--language", choices=[item.value for item in OutputLanguage])
    evaluate.add_argument("--live-preflight", action="store_true")
    evaluate.set_defaults(handler=_command_evaluate)

    runs = commands.add_parser("runs", help="manage durable Run Bundles")
    run_commands = runs.add_subparsers(dest="runs_command", required=True)
    prune = run_commands.add_parser("prune", help="list or remove old Run Bundles")
    prune.add_argument("--older-than", type=int, default=30, metavar="DAYS")
    prune.add_argument("--include-incomplete", action="store_true")
    prune.add_argument("--yes", action="store_true", help="perform deletion; otherwise this is a dry run")
    prune.set_defaults(handler=_command_runs_prune)
    return parser


def _exit_code(error: VideoGeneratorError) -> int:
    if isinstance(error, ConfigurationError) or error.kind is ErrorKind.UNSUPPORTED:
        return EXIT_CONFIG
    if error.kind is ErrorKind.NOT_READY:
        return EXIT_NOT_READY
    if error.kind is ErrorKind.BUDGET_EXCEEDED:
        return EXIT_BUDGET
    if error.kind in {ErrorKind.INVALID_OUTPUT, ErrorKind.POLICY_REFUSAL}:
        return EXIT_INVALID_OUTPUT
    return EXIT_INTERNAL


def main(argv: Sequence[str] | None = None) -> int:
    _configure_console_output()
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if getattr(args, "older_than", 0) < 0:
        parser.error("--older-than must be zero or greater")
    if not 1 <= getattr(args, "port", 8765) <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        return int(args.handler(args))
    except VideoGeneratorError as exc:
        print(f"Error [{exc.kind.value}]: {exc.message}", file=sys.stderr)
        if exc.action:
            print(f"Action: {exc.action}", file=sys.stderr)
        return _exit_code(exc)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error [internal]: {exc}", file=sys.stderr)
        return EXIT_INTERNAL


if __name__ == "__main__":
    raise SystemExit(main())
