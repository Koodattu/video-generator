from __future__ import annotations

from copy import deepcopy

from video_generator.cli import _earliest_frozen_impact, build_parser
from video_generator.prompting import build_frozen_assets


def test_generate_is_end_to_end_by_default() -> None:
    args = build_parser().parse_args(
        ["generate", "--config", "config.toml", "--brief", "brief.toml"]
    )

    assert args.stop_after is None


def test_runs_prune_is_dry_run_by_default() -> None:
    args = build_parser().parse_args(["runs", "prune"])

    assert args.yes is False


def test_evaluate_accepts_long_draft_suite() -> None:
    args = build_parser().parse_args(
        ["evaluate", "--suite", "draft-quality", "--config", "config.toml"]
    )

    assert args.suite == "draft-quality"


def test_setup_accepts_typed_local_llm_profile() -> None:
    args = build_parser().parse_args(
        ["setup", "--profile", "local", "--llm-profile", "local-llm.toml"]
    )

    assert args.llm_profile.name == "local-llm.toml"


def test_models_download_selects_a_curated_candidate() -> None:
    args = build_parser().parse_args(
        ["models", "download", "qwen3.6-27b-q4-mtp", "--dry-run"]
    )

    assert args.candidate == "qwen3.6-27b-q4-mtp"
    assert args.dry_run


def test_backend_descriptor_set_order_does_not_invalidate_rerun(resolved_config) -> None:
    old = build_frozen_assets(resolved_config)
    new = deepcopy(old)
    descriptors = new["profile"]["backend_descriptors"]
    backend_id = next(
        candidate
        for candidate, descriptor in descriptors.items()
        if len(descriptor["languages"]) > 1
    )
    descriptors[backend_id]["languages"].reverse()

    assert _earliest_frozen_impact(old, new, resolved_config) is None


def test_frozen_backend_descriptor_sets_are_sorted(resolved_config) -> None:
    assets = build_frozen_assets(resolved_config)

    for descriptor in assets["profile"]["backend_descriptors"].values():
        assert descriptor["protocols"] == sorted(descriptor["protocols"])
        assert descriptor["languages"] == sorted(descriptor["languages"])
        assert descriptor["allowed_usage_purposes"] == sorted(
            descriptor["allowed_usage_purposes"]
        )
