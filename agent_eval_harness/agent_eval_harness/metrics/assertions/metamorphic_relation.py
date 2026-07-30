"""Assertion: metamorphic_relation — validates that a transformed input preserves an invariant."""
from __future__ import annotations

import json

from agent_eval_harness.datasets.metamorphic_ops import evaluate_invariant
from agent_eval_harness.metrics.assertions.registry import register
from agent_eval_harness.metrics.types import MetricResult

_METRIC_NAME = "assertion.metamorphic_relation"


def _result(component_id: str, passed: bool | None, details: dict) -> MetricResult:
    return MetricResult(
        metric_name=_METRIC_NAME,
        metric_class="assertion",
        score=None,
        passed=passed,
        details=details,
        component_id=component_id,
    )


def _check_invariant(
    component_id: str, invariant_op: str, invariant_params: dict, output: dict, checked: dict
) -> MetricResult:
    """Evaluates the invariant against output; a malformed op/params degrades to an explicit reason, not a crash."""
    try:
        passed = evaluate_invariant(invariant_op, output, invariant_params)
        return _result(component_id, passed, {"invariant_op": invariant_op, **checked})
    except ValueError as exc:
        return _result(component_id, None, {"reason": str(exc)})


@register("metamorphic_relation")
def metamorphic_relation(spans: list[dict], component_id: str, params: dict) -> MetricResult:
    invariant_op = params.get("invariant_op")
    invariant_params = params.get("invariant_params", {})
    source_expected = params.get("source_expected_output")

    if not invariant_op:
        return _result(component_id, None, {"reason": "no_invariant_op"})

    if source_expected is not None:
        return _check_invariant(
            component_id, invariant_op, invariant_params, source_expected,
            {"checked_against": "source_expected_output"},
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
            return _result(component_id, False, {"reason": "invalid_json"})
        return _check_invariant(
            component_id, invariant_op, invariant_params, output, {"checked_span": span["id"]},
        )

    return _result(component_id, None, {"reason": "no_result"})
