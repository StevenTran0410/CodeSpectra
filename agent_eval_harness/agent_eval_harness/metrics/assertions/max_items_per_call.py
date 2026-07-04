"""Assertion: max_items_per_call — checks planner does not exceed the fanout limit.

Reads `output_json.intents` from every span belonging to `component_id` within
one trace. Each call's intent list must have length <= `params["limit"]`.

Triggered automatically by the sweep runner for every System Map constraint
named `max_items_per_call` (§4.3, source 1).
"""
from __future__ import annotations

import json

from agent_eval_harness.metrics.assertions.registry import register
from agent_eval_harness.metrics.types import MetricResult


@register("max_items_per_call")
def max_items_per_call(spans: list[dict], component_id: str, params: dict) -> MetricResult:
    limit = int(params.get("limit", params.get("value", 2)))
    offending: list[str] = []
    for span in spans:
        if span.get("component_id") != component_id:
            continue
        output = json.loads(span.get("output_json") or "{}")
        intents = output.get("intents")
        if isinstance(intents, list) and len(intents) > limit:
            offending.append(span["id"])
    passed = len(offending) == 0
    return MetricResult(
        metric_name="assertion.max_items_per_call",
        metric_class="assertion",
        score=None,
        passed=passed,
        details={"offending_span_ids": offending, "limit": limit},
        component_id=component_id,
    )
