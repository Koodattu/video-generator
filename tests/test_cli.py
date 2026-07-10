from __future__ import annotations

from video_generator.cli import build_parser


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
