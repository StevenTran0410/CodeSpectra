"""Assertion: schema_valid — validates an agent's own output span against a JSON Schema
seeded from the contract's harvested `output.json_schema`."""
from __future__ import annotations

import json

from agent_eval_harness.metrics.assertions.registry import register
from agent_eval_harness.metrics.types import MetricResult


@register("schema_valid")
def schema_valid(spans: list[dict], component_id: str, params: dict) -> MetricResult:
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
        if span.get("span_type") not in ("agent", "llm_call"):
            continue
        checked_span_count += 1
        raw = span.get("output_json") or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            violations.append({"span_id": span["id"], "error": "invalid JSON"})
            continue
        try:
            jsonschema.validate(instance=data, schema=schema)
        except jsonschema.ValidationError as exc:
            violations.append({"span_id": span["id"], "error": exc.message})

    passed = checked_span_count > 0 and len(violations) == 0
    return MetricResult(
        metric_name="assertion.schema_valid",
        metric_class="assertion",
        score=None,
        passed=passed,
        details={"violations": violations, "checked_spans": checked_span_count},
        component_id=component_id,
    )
