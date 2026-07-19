"""Assertion: field_match — per-field exact|set|tolerance comparison vs gold."""
from __future__ import annotations

from agent_eval_harness.metrics.assertions.registry import find_component_output_json, register
from agent_eval_harness.metrics.types import MetricResult

_COMPARE_EXACT = "exact"
_COMPARE_SET = "set"
_COMPARE_TOLERANCE = "tolerance"


def _compare_exact(actual: object, expected: object) -> bool:
    """Exact equality comparison."""
    return actual == expected


def _compare_set(actual: object, expected: object) -> bool:
    """Compare as sets (unordered lists)."""
    if not isinstance(actual, list) or not isinstance(expected, list):
        return False
    return set(actual) == set(expected)


def _compare_tolerance(actual: object, expected: object, tolerance: float) -> bool:
    """Numeric comparison within tolerance."""
    if not isinstance(actual, (int, float)) or not isinstance(expected, (int, float)):
        return False
    if expected == 0:
        return actual == expected
    return abs((actual - expected) / expected) <= tolerance


def _get_field_value(obj: object, path: str) -> object:
    """Get a value from an object using dot-separated path."""
    if not isinstance(obj, dict):
        return None
    parts = path.split(".")
    current = obj
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


@register("field_match")
def field_match(spans: list[dict], component_id: str, params: dict) -> MetricResult:
    """Compare specified fields between output and expected gold."""
    fields = params.get("fields", {})
    if not fields:
        return MetricResult(
            metric_name="assertion.field_match",
            metric_class="assertion",
            score=None,
            passed=None,
            details={"reason": "no_fields_specified"},
            component_id=component_id,
        )

    expected_gold = params.get("expected_gold", {})
    if not expected_gold:
        return MetricResult(
            metric_name="assertion.field_match",
            metric_class="assertion",
            score=None,
            passed=None,
            details={"reason": "no_expected_gold"},
            component_id=component_id,
        )

    output_data, early_result = find_component_output_json(
        spans, component_id, "assertion.field_match"
    )
    if early_result is not None:
        return early_result

    if output_data is None:
        return MetricResult(
            metric_name="assertion.field_match",
            metric_class="assertion",
            score=None,
            passed=None,
            details={"reason": "no_result"},
            component_id=component_id,
        )

    violations: list[dict] = []
    for field_path, spec in fields.items():
        if not isinstance(spec, dict):
            continue
        comparison_type = spec.get("type", _COMPARE_EXACT)
        expected_val = _get_field_value(expected_gold, field_path)
        actual_val = _get_field_value(output_data, field_path)

        passed = False
        if comparison_type == _COMPARE_EXACT:
            passed = _compare_exact(actual_val, expected_val)
        elif comparison_type == _COMPARE_SET:
            passed = _compare_set(actual_val, expected_val)
        elif comparison_type == _COMPARE_TOLERANCE:
            tolerance = spec.get("tolerance", 0.0)
            passed = _compare_tolerance(actual_val, expected_val, tolerance)

        if not passed:
            violations.append({
                "field": field_path,
                "type": comparison_type,
                "expected": expected_val,
                "actual": actual_val,
            })

    passed = len(violations) == 0
    return MetricResult(
        metric_name="assertion.field_match",
        metric_class="assertion",
        score=None,
        passed=passed,
        details={"violations": violations, "checked_fields": len(fields)},
        component_id=component_id,
    )
