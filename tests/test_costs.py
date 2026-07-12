from __future__ import annotations

from types import SimpleNamespace

import pytest

from video_generator.contracts import CreativeBrief, UsageRecord
from video_generator.costs import PRICING_CATALOG, PRICING_SNAPSHOT, calculate_list_price
from video_generator.executor import TaskExecutor
from video_generator.profiles import PROFILES
from video_generator.prompting import build_frozen_assets
from video_generator.run_store import RunStore


def test_openai_usage_is_priced_from_explicit_billable_units() -> None:
    usage = calculate_list_price(
        UsageRecord(
            task_id="outline",
            backend_id="openai:gpt-5.6-terra",
            input_units=100_000,
            output_units=100_000,
            billable_units={
                "input_tokens": 90_000,
                "cached_input_tokens": 10_000,
                "output_tokens": 100_000,
            },
        )
    )

    assert usage.estimated_usd == 1.7275
    assert usage.cost_status == "estimated"
    assert usage.pricing_snapshot == PRICING_SNAPSHOT
    assert "pricing" in usage.cost_basis


def test_gemini_image_uses_fixed_2k_fallback_when_modality_tokens_are_absent() -> None:
    usage = calculate_list_price(
        UsageRecord(
            task_id="image_generate",
            backend_id="gemini:gemini-3.1-flash-image",
            billable_units={"input_tokens": 1000, "generated_image_2k": 1},
        )
    )

    assert usage.estimated_usd == 0.1015


def test_gemini_image_cached_input_is_unpriced_without_a_public_rate() -> None:
    usage = calculate_list_price(
        UsageRecord(
            task_id="image_generate",
            backend_id="gemini:gemini-3.1-flash-image",
            billable_units={
                "input_tokens": 1000,
                "cached_input_tokens": 100,
                "image_output_tokens": 1680,
            },
        )
    )

    assert usage.estimated_usd is None
    assert usage.cost_status == "unpriced"
    assert "cached_input_tokens" in usage.cost_basis


def test_empty_billable_usage_is_unpriced_instead_of_fake_zero() -> None:
    usage = calculate_list_price(
        UsageRecord(task_id="outline", backend_id="openai:gpt-5.6-terra")
    )

    assert usage.estimated_usd is None
    assert usage.cost_status == "unpriced"


def test_openai_long_context_multipliers_are_applied() -> None:
    usage = calculate_list_price(
        UsageRecord(
            task_id="outline",
            backend_id="openai:gpt-5.6-terra",
            billable_units={"input_tokens": 300_000, "output_tokens": 10_000},
        )
    )

    assert usage.estimated_usd == 1.725
    assert "long-context" in usage.cost_basis


def test_openai_gpt_5_5_long_context_multipliers_are_applied() -> None:
    usage = calculate_list_price(
        UsageRecord(
            task_id="outline",
            backend_id="openai:gpt-5.5",
            billable_units={"input_tokens": 300_000, "output_tokens": 10_000},
        )
    )

    assert usage.estimated_usd == 3.45
    assert "long-context" in usage.cost_basis


def test_explicit_empty_catalog_does_not_fall_back_to_global_prices() -> None:
    usage = calculate_list_price(
        UsageRecord(
            task_id="outline",
            backend_id="openai:gpt-5.4-mini",
            billable_units={"input_tokens": 1000},
        ),
        catalog={},
    )

    assert usage.estimated_usd is None
    assert usage.cost_status == "unpriced"
    assert usage.pricing_snapshot == "unknown"


def test_explicit_frozen_catalog_is_not_affected_by_global_rate_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = {
        **PRICING_CATALOG,
        "backends": {
            **PRICING_CATALOG["backends"],
            "openai:gpt-5.4-mini": {
                **PRICING_CATALOG["backends"]["openai:gpt-5.4-mini"],
                "rates_usd_per_unit": {
                    **PRICING_CATALOG["backends"]["openai:gpt-5.4-mini"]["rates_usd_per_unit"],
                },
            },
        },
    }
    monkeypatch.setitem(
        PRICING_CATALOG["backends"]["openai:gpt-5.4-mini"]["rates_usd_per_unit"],
        "input_tokens",
        "99",
    )
    usage = calculate_list_price(
        UsageRecord(
            task_id="outline",
            backend_id="openai:gpt-5.4-mini",
            billable_units={"input_tokens": 1000},
        ),
        catalog=frozen,
    )

    assert usage.estimated_usd == 0.00075


def test_cloud_call_ledger_settles_durably(tmp_path, resolved_config) -> None:
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

    call_id = store.reserve_cost(0.4, task_id="outline", backend_id="openai:gpt-5.4-mini")
    usage = calculate_list_price(
        UsageRecord(
            task_id="outline",
            backend_id="openai:gpt-5.4-mini",
            call_id=call_id,
            input_units=1000,
            output_units=500,
            billable_units={"input_tokens": 1000, "output_tokens": 500},
            reserved_usd=0.4,
            elapsed_seconds=1.25,
        )
    )
    store.settle_cost(call_id, usage)

    reopened = RunStore.open(store.root)
    call = reopened.manifest.cloud_calls[0]
    assert call.status == "settled"
    assert call.estimated_usd == usage.estimated_usd
    assert call.elapsed_seconds == 1.25
    assert reopened.frozen_assets["profile"]["pricing_catalog"]["snapshot_id"] == PRICING_SNAPSHOT


def test_missing_usage_and_unexpected_failures_remain_unresolved(
    tmp_path, resolved_config
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
    descriptor = SimpleNamespace(reservation_usd=0.4, cloud=True)
    executor = TaskExecutor(
        registry=SimpleNamespace(descriptor=lambda backend_id: descriptor),
        store=store,
        prompts=SimpleNamespace(),
    )

    result = executor._call(
        "outline",
        "openai:gpt-5.4-mini",
        lambda: SimpleNamespace(usage=None),
    )
    assert result.usage.estimated_usd is None
    assert store.manifest.cloud_calls[-1].status == "unresolved"

    def fail() -> None:
        raise RuntimeError("filesystem promotion failed")

    with pytest.raises(RuntimeError, match="filesystem promotion failed"):
        executor._call("outline", "openai:gpt-5.4-mini", fail)

    assert store.manifest.cloud_calls[-1].status == "unresolved"
    assert store.manifest.cloud_calls[-1].error["kind"] == "internal"
