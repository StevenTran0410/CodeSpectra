"""Assertion: max_retries — verifies a component does not exceed its retry budget.

Counts the number of spans with `component_id == component_id` in one trace.
`passed = count <= limit + 1` (initial attempt + `limit` retries).

params:
    limit (int): maximum number of retries (not counting the first attempt).
                 Default: 1.
"""
from __future__ import annotations

from agent_eval_harness.metrics.assertions.registry import register
from agent_eval_harness.metrics.types import MetricResult


@register("max_retries")
def max_retries(spans: list[dict], component_id: str, params: dict) -> MetricResult:
    limit = int(params.get("limit", params.get("value", 1)))
    component_spans = [s for s in spans if s.get("component_id") == component_id]
    count = len(component_spans)
    max_allowed = limit + 1  # initial attempt + retries
    passed = count <= max_allowed
    return MetricResult(
        metric_name="assertion.max_retries",
        metric_class="assertion",
        score=None,
        passed=passed,
        details={
            "span_count": count,
            "max_allowed": max_allowed,
            "limit": limit,
        },
        component_id=component_id,
    )
