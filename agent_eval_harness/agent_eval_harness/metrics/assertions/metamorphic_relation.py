"""Assertion: metamorphic_relation — validates that a transformed input preserves an invariant."""
from __future__ import annotations

import json

from agent_eval_harness.datasets.metamorphic_ops import evaluate_invariant
from agent_eval_harness.metrics.assertions.registry import register
from agent_eval_harness.metrics.types import MetricResult


@register("metamorphic_relation")
def metamorphic_relation(spans: list[dict], component_id: str, params: dict) -> MetricResult:
    invariant_op = params.get("invariant_op")
    invariant_params = params.get("invariant_params", {})
    source_expected = params.get("source_expected_output")

    if not invariant_op:
        return MetricResult(
            metric_name="assertion.metamorphic_relation",
            metric_class="assertion",
            score=None,
            passed=None,
            details={"reason": "no_invariant_op"},
            component_id=component_id,
        )

    if source_expected is not None:
        try:
            result = evaluate_invariant(invariant_op, source_expected, invariant_params)
            return MetricResult(
                metric_name="assertion.metamorphic_relation",
                metric_class="assertion",
                score=None,
                passed=result,
                details={"invariant_op": invariant_op, "checked_against": "source_expected_output"},
                component_id=component_id,
            )
        except ValueError as exc:
            return MetricResult(
                metric_name="assertion.metamorphic_relation",
                metric_class="assertion",
                score=None,
                passed=None,
                details={"reason": str(exc)},
                component_id=component_id,
            )

    for span in spans:
        if span.get("component_id") != component_id:
            continue
        if span.get("span_type") not in ("agent", "llm_call"):
            continue
        raw = span.get("output_json") or "{}"
        try:
            output = json.loads(raw)
        except json.JSONDecodeError:
            return MetricResult(
                metric_name="assertion.metamorphic_relation",
                metric_class="assertion",
                score=None,
                passed=False,
                details={"reason": "invalid_json"},
                component_id=component_id,
            )
        try:
            result = evaluate_invariant(invariant_op, output, invariant_params)
            return MetricResult(
                metric_name="assertion.metamorphic_relation",
                metric_class="assertion",
                score=None,
                passed=result,
                details={"invariant_op": invariant_op, "checked_span": span["id"]},
                component_id=component_id,
            )
        except ValueError as exc:
            return MetricResult(
                metric_name="assertion.metamorphic_relation",
                metric_class="assertion",
                score=None,
                passed=None,
                details={"reason": str(exc)},
                component_id=component_id,
            )

    return MetricResult(
        metric_name="assertion.metamorphic_relation",
        metric_class="assertion",
        score=None,
        passed=None,
        details={"reason": "no_result"},
        component_id=component_id,
    )
