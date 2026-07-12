from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
from typing import Any

from .contracts import UsageRecord


PRICING_SNAPSHOT = "2026-07-12.public-list-prices-v1"


def _entry(
    *,
    rates: dict[str, str],
    source: str,
    effective_date: str,
    basis: str,
    long_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry = {
        "rates_usd_per_unit": rates,
        "source": source,
        "effective_date": effective_date,
        "basis": basis,
    }
    if long_context is not None:
        entry["long_context"] = long_context
    return entry


# Rates are stored as decimal strings so a Run freezes the exact arithmetic inputs used for its
# calculated list-price costs. They are not invoice data: free tiers, subscriptions, discounts,
# regional pricing, taxes, and provider reconciliation can differ.
PRICING_CATALOG: dict[str, Any] = {
    "snapshot_id": PRICING_SNAPSHOT,
    "currency": "USD",
    "label": "calculated public list price",
    "backends": {
        "openai:gpt-5.6-terra": _entry(
            rates={
                "input_tokens": "0.0000025",
                "cached_input_tokens": "0.00000025",
                "cache_write_tokens": "0.000003125",
                "output_tokens": "0.000015",
            },
            source="https://developers.openai.com/api/docs/pricing",
            effective_date="2026-07-12",
            basis="standard token rates",
            long_context={
                "input_threshold_tokens": "272000",
                "rate_multipliers": {
                    "input_tokens": "2",
                    "cached_input_tokens": "2",
                    "cache_write_tokens": "2",
                    "output_tokens": "1.5",
                },
            },
        ),
        "openai:web": _entry(
            rates={
                "input_tokens": "0.0000025",
                "cached_input_tokens": "0.00000025",
                "cache_write_tokens": "0.000003125",
                "output_tokens": "0.000015",
                "search_queries": "0.01",
            },
            source="https://developers.openai.com/api/docs/pricing",
            effective_date="2026-07-12",
            basis="standard token rates plus web-search calls",
            long_context={
                "input_threshold_tokens": "272000",
                "rate_multipliers": {
                    "input_tokens": "2",
                    "cached_input_tokens": "2",
                    "cache_write_tokens": "2",
                    "output_tokens": "1.5",
                },
            },
        ),
        "openai:gpt-5.5": _entry(
            rates={
                "input_tokens": "0.000005",
                "cached_input_tokens": "0.0000005",
                "output_tokens": "0.00003",
            },
            source="https://developers.openai.com/api/docs/pricing",
            effective_date="2026-07-12",
            basis="standard token rates below the long-context threshold",
            long_context={
                "input_threshold_tokens": "272000",
                "rate_multipliers": {
                    "input_tokens": "2",
                    "cached_input_tokens": "2",
                    "output_tokens": "1.5",
                },
            },
        ),
        "openai:gpt-5.4-mini": _entry(
            rates={
                "input_tokens": "0.00000075",
                "cached_input_tokens": "0.000000075",
                "output_tokens": "0.0000045",
            },
            source="https://developers.openai.com/api/docs/pricing",
            effective_date="2026-07-12",
            basis="standard token rates",
        ),
        "openai:gpt-image-2": _entry(
            rates={
                "text_input_tokens": "0.000005",
                "cached_text_input_tokens": "0.00000125",
                "image_input_tokens": "0.000008",
                "cached_image_input_tokens": "0.000002",
                "image_output_tokens": "0.00003",
            },
            source="https://developers.openai.com/api/docs/pricing",
            effective_date="2026-07-12",
            basis="image API modality token rates",
        ),
        "gemini:gemini-3.5-flash": _entry(
            rates={
                "input_tokens": "0.0000015",
                "cached_input_tokens": "0.00000015",
                "output_tokens": "0.000009",
            },
            source="https://ai.google.dev/gemini-api/docs/pricing",
            effective_date="2026-07-12",
            basis="paid-tier token rates; output includes thinking tokens",
        ),
        "gemini:search": _entry(
            rates={
                "input_tokens": "0.0000015",
                "cached_input_tokens": "0.00000015",
                "output_tokens": "0.000009",
                "search_queries": "0.014",
            },
            source="https://ai.google.dev/gemini-api/docs/pricing",
            effective_date="2026-07-12",
            basis="paid-tier token rates plus grounding queries after any free allowance",
        ),
        "gemini:gemini-3.1-flash-image": _entry(
            rates={
                "input_tokens": "0.0000005",
                "output_tokens": "0.000003",
                "image_output_tokens": "0.00006",
                "generated_image_1k": "0.067",
                "generated_image_2k": "0.101",
                "generated_image_4k": "0.151",
            },
            source="https://ai.google.dev/gemini-api/docs/pricing",
            effective_date="2026-07-12",
            basis="paid-tier token rates; fixed image equivalents are fallbacks when modality tokens are absent",
        ),
        "elevenlabs:eleven_multilingual_v2": _entry(
            rates={"tts_characters": "0.0001"},
            source="https://elevenlabs.io/pricing/api",
            effective_date="2026-07-12",
            basis="public API character price",
        ),
        "elevenlabs:music_v2": _entry(
            rates={"music_seconds": "0.0025"},
            source="https://elevenlabs.io/pricing/api",
            effective_date="2026-07-12",
            basis="public API music price per minute converted to seconds",
        ),
        "elevenlabs:forced-alignment": _entry(
            rates={"audio_seconds": "0.00006111111111111111"},
            source="https://elevenlabs.io/pricing/api",
            effective_date="2026-07-12",
            basis="public speech-to-text price per hour converted to seconds",
        ),
        "brave:web": _entry(
            rates={"search_queries": "0.005"},
            source="https://brave.com/search/api/",
            effective_date="2026-07-12",
            basis="public Search plan request price",
        ),
    },
    "not_applicable": {
        "ddgs:duckduckgo": "keyless search backend has no configured per-call API charge",
    },
    "unpriced": {},
}


def frozen_pricing_catalog() -> dict[str, Any]:
    return deepcopy(PRICING_CATALOG)


def _priced_units(usage: UsageRecord) -> dict[str, float]:
    if usage.billable_units:
        return dict(usage.billable_units)
    if usage.backend_id == "openai:gpt-image-2":
        return {
            "text_input_tokens": float(usage.input_units),
            "image_output_tokens": float(usage.output_units),
        }
    return {
        "input_tokens": float(usage.input_units),
        "output_tokens": float(usage.output_units),
    }


def calculate_list_price(
    usage: UsageRecord,
    *,
    catalog: dict[str, Any] | None = None,
) -> UsageRecord:
    """Return Usage enriched with a reproducible calculated list-price estimate."""

    pricing_catalog = PRICING_CATALOG if catalog is None else catalog
    snapshot_id = str(pricing_catalog.get("snapshot_id") or "unknown")
    not_applicable = pricing_catalog.get("not_applicable")
    not_applicable = not_applicable if isinstance(not_applicable, dict) else {}
    if usage.backend_id in not_applicable:
        return usage.model_copy(
            update={
                "estimated_usd": 0.0,
                "cost_status": "not_applicable",
                "pricing_snapshot": snapshot_id,
                "cost_basis": not_applicable[usage.backend_id],
            }
        )
    backends = pricing_catalog.get("backends")
    backends = backends if isinstance(backends, dict) else {}
    entry = backends.get(usage.backend_id)
    if not isinstance(entry, dict):
        unpriced = pricing_catalog.get("unpriced")
        unpriced = unpriced if isinstance(unpriced, dict) else {}
        reason = unpriced.get(
            usage.backend_id, "no frozen public list-price rule for this backend"
        )
        return usage.model_copy(
            update={
                "estimated_usd": None,
                "cost_status": "unpriced",
                "pricing_snapshot": snapshot_id,
                "cost_basis": reason,
            }
        )

    units = _priced_units(usage)
    if usage.backend_id == "gemini:gemini-3.1-flash-image":
        image_tokens = units.get("image_output_tokens", 0)
        if image_tokens > 0:
            units = {key: value for key, value in units.items() if not key.startswith("generated_image_")}
        elif not any(key.startswith("generated_image_") for key in units):
            units["generated_image_2k"] = 1

    rates = entry.get("rates_usd_per_unit")
    rates = rates if isinstance(rates, dict) else {}
    unknown_positive_units = sorted(
        unit_name
        for unit_name, amount in units.items()
        if float(amount) > 0 and unit_name not in rates
    )
    if unknown_positive_units:
        return usage.model_copy(
            update={
                "billable_units": units,
                "estimated_usd": None,
                "cost_status": "unpriced",
                "pricing_snapshot": snapshot_id,
                "cost_basis": (
                    f"{entry.get('basis', 'public list price')}; response included "
                    "unpriced units: " + ", ".join(unknown_positive_units)
                ),
            }
        )

    multipliers: dict[str, Decimal] = {}
    long_context = entry.get("long_context")
    if isinstance(long_context, dict):
        input_total = sum(
            Decimal(str(units.get(name, 0)))
            for name in ("input_tokens", "cached_input_tokens", "cache_write_tokens")
        )
        threshold = Decimal(str(long_context.get("input_threshold_tokens") or 0))
        if input_total > threshold:
            raw_multipliers = long_context.get("rate_multipliers")
            if isinstance(raw_multipliers, dict):
                multipliers = {
                    str(name): Decimal(str(multiplier))
                    for name, multiplier in raw_multipliers.items()
                }

    total = Decimal("0")
    priced_any = False
    for unit_name, amount in units.items():
        rate = rates.get(unit_name)
        if rate is None or float(amount) <= 0:
            continue
        total += (
            Decimal(str(amount))
            * Decimal(str(rate))
            * multipliers.get(unit_name, Decimal("1"))
        )
        priced_any = True
    if not priced_any:
        return usage.model_copy(
            update={
                "billable_units": units,
                "estimated_usd": None,
                "cost_status": "unpriced",
                "pricing_snapshot": snapshot_id,
                "cost_basis": f"{entry['basis']}; response contained no recognized billable units",
            }
        )
    return usage.model_copy(
        update={
            "billable_units": units,
            "estimated_usd": float(total.quantize(Decimal("0.00000001"))),
            "cost_status": "reported" if usage.actual_usd is not None else "estimated",
            "pricing_snapshot": snapshot_id,
            "cost_basis": (
                f"{entry['basis']}"
                + ("; long-context multipliers applied" if multipliers else "")
                + f"; {entry['source']}"
            ),
        }
    )
