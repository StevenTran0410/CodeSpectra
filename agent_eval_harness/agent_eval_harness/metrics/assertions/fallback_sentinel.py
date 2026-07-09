"""Assertion: fallback_sentinel — output is not the contract's fallback literal
(CS-281 §2). A fallback dict passes schema_valid by construction (it's built to match
the same schema) so this is the only deterministic way to catch a silently-degraded run."""
from __future__ import annotations

import json

from agent_eval_harness.metrics.assertions.registry import register
from agent_eval_harness.metrics.types import MetricResult

_DYNAMIC = "<dynamic>"


def _matches_fallback(data: dict, fallback: dict) -> bool:
    """A field counts as matching if it's byte-equal, OR the harvested fallback marked
    it '<dynamic>' (value varies per-call — e.g. echoes an input id) and the field is
    merely present. Never a false negative from statics being unable to pin an exact value."""
    for key, expected in fallback.items():
        if key not in data:
            return False
        if expected == _DYNAMIC:
            continue
        if data[key] != expected:
            return False
    return True


@register("fallback_sentinel")
def fallback_sentinel(spans: list[dict], component_id: str, params: dict) -> MetricResult:
    fallback = params.get("fallback")
    if not fallback:
        return MetricResult(
            metric_name="assertion.fallback_sentinel",
            metric_class="assertion",
            score=None,
            passed=None,
            details={"note": "no fallback literal supplied in params — nothing to check against"},
            component_id=component_id,
        )

    hits: list[dict] = []
    checked_span_count = 0

    for span in spans:
        if span.get("component_id") != component_id:
            continue
        if span.get("span_type") not in ("agent", "llm_call"):
            continue
        checked_span_count += 1
        raw = span.get("output_json") or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and _matches_fallback(data, fallback):
            hits.append({"span_id": span["id"]})

    passed = checked_span_count > 0 and len(hits) == 0
    return MetricResult(
        metric_name="assertion.fallback_sentinel",
        metric_class="assertion",
        score=None,
        passed=passed,
        details={"fallback_hits": hits, "checked_spans": checked_span_count},
        component_id=component_id,
    )
