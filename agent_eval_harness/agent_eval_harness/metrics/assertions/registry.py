"""Assertion function registry."""
from __future__ import annotations

import json
from collections.abc import Callable

from agent_eval_harness.metrics.types import MetricResult

# spans: list[dict] (one trace's rows from get_spans_for_trace)
AssertionFn = Callable[[list[dict], str, dict], MetricResult]

ASSERTIONS: dict[str, AssertionFn] = {}

# Span types whose output_json holds a component's real (non-tool-call) result.
AGENT_OUTPUT_SPAN_TYPES: tuple[str, ...] = ("agent", "llm_call")


def register(name: str):
    """Decorator: register an assertion function under the given name."""

    def deco(fn: AssertionFn) -> AssertionFn:
        ASSERTIONS[name] = fn
        return fn

    return deco


def _import_all() -> None:
    """Import all assertion modules to populate ASSERTIONS."""
    from agent_eval_harness.metrics.assertions import (  # noqa: F401
        allowed_downstream,
        arg_schema,
        fallback_sentinel,
        field_match,
        llm_call_budget,
        max_items_per_call,
        max_retries,
        metamorphic_relation,
        no_unnecessary_calls,
        referential_integrity,
        retry_on_reject_required,
        schema_valid,
    )


_all_imported = False


def ensure_assertions_imported() -> None:
    """Runs `_import_all` exactly once, immune to ASSERTIONS being partially populated elsewhere."""
    global _all_imported
    if _all_imported:
        return
    _import_all()
    _all_imported = True


def get_assertion(name: str) -> AssertionFn:
    """Retrieve a registered assertion, auto-importing all modules first."""
    ensure_assertions_imported()
    if name not in ASSERTIONS:
        raise KeyError(f"Unknown assertion '{name}'. Registered: {sorted(ASSERTIONS)}")
    return ASSERTIONS[name]


def find_component_output_json(
    spans: list[dict], component_id: str, metric_name: str
) -> tuple[dict | None, MetricResult | None]:
    """First matching span's output_json, or (None, None) if none matched — shared parse logic."""
    for span in spans:
        if span.get("component_id") != component_id:
            continue
        if span.get("span_type") not in AGENT_OUTPUT_SPAN_TYPES:
            continue
        raw = span.get("output_json") or "{}"
        try:
            return json.loads(raw), None
        except json.JSONDecodeError:
            return None, MetricResult(
                metric_name=metric_name,
                metric_class="assertion",
                score=None,
                passed=False,
                details={"reason": "invalid_json_output"},
                component_id=component_id,
            )
    return None, None
