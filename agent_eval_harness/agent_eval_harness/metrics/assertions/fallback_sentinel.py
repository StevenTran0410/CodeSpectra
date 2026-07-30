"""Assertion: output is not the contract's fallback literal — the only deterministic way to catch a silently-degraded run, since a fallback dict passes schema_valid by construction."""
from __future__ import annotations

import json

from agent_eval_harness.metrics.assertions.registry import register
from agent_eval_harness.metrics.scoring_helpers import matches_fallback
from agent_eval_harness.metrics.types import MetricResult


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
        if isinstance(data, dict) and matches_fallback(data, fallback):
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
