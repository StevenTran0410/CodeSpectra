"""Assertion: arg_schema — validates tool_call span arguments against a JSON Schema.

For each tool_call span belonging to `component_id`, validates its `output_json`
(or `input_json` if `output_json` is absent) against the provided JSON Schema.

params:
    schema (dict): a JSON Schema object (Draft-7 compatible) to validate against.

Dependency: `jsonschema>=4.0.0` (added to base AEH dependencies in pyproject.toml).
"""
from __future__ import annotations

import json

from agent_eval_harness.metrics.assertions.registry import register
from agent_eval_harness.metrics.types import MetricResult


@register("arg_schema")
def arg_schema(spans: list[dict], component_id: str, params: dict) -> MetricResult:
    try:
        import jsonschema
    except ImportError as exc:
        raise ImportError(
            "jsonschema is not installed. Run `pip install jsonschema>=4.0.0`."
        ) from exc

    schema = params.get("schema", {})
    violations: list[dict] = []
    checked_span_count = 0

    for span in spans:
        if span.get("component_id") != component_id:
            continue
        if span.get("span_type") != "tool_call":
            continue
        checked_span_count += 1
        # Use output_json first, fall back to input_json
        raw = span.get("output_json") or span.get("input_json") or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            violations.append({"span_id": span["id"], "error": "invalid JSON"})
            continue
        try:
            jsonschema.validate(instance=data, schema=schema)
        except jsonschema.ValidationError as exc:
            violations.append({"span_id": span["id"], "error": exc.message})

    passed = len(violations) == 0
    return MetricResult(
        metric_name="assertion.arg_schema",
        metric_class="assertion",
        score=None,
        passed=passed,
        details={"violations": violations, "checked_spans": checked_span_count},
        component_id=component_id,
    )
