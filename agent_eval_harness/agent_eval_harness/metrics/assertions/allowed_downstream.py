"""Assertion: allowed_downstream — verifies a component only fans out to allowed targets."""
from __future__ import annotations

from agent_eval_harness.metrics.assertions.registry import register
from agent_eval_harness.metrics.types import MetricResult


@register("allowed_downstream")
def allowed_downstream(spans: list[dict], component_id: str, params: dict) -> MetricResult:
    allowed: set[str] = set(params.get("allowed", []))

    violations: list[dict] = []
    for span in spans:
        if span.get("component_id") != component_id:
            continue
        children = [s for s in spans if s.get("parent_span_id") == span["id"]]
        for child in children:
            child_cid = child.get("component_id")
            if child_cid is None:
                # unmatched — skip, not this assertion's concern
                continue
            if child_cid == component_id:
                # own internal sub-step, not a fan-out — never a violation
                continue
            if child_cid not in allowed:
                violations.append(
                    {"parent_span_id": span["id"], "child_span_id": child["id"],
                     "child_component_id": child_cid}
                )

    passed = len(violations) == 0
    return MetricResult(
        metric_name="assertion.allowed_downstream",
        metric_class="assertion",
        score=None,
        passed=passed,
        details={"violations": violations, "allowed": sorted(allowed)},
        component_id=component_id,
    )
