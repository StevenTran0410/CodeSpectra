"""Assertion: allowed_downstream — verifies a component only fans out to allowed targets.

For each span belonging to `component_id`, finds its child spans (spans whose
`parent_span_id` matches the current span's `id`). Every child's `component_id`
must be in `params["allowed"]` OR equal to `component_id` itself.

params:
    allowed (list[str]): list of component IDs that are permitted downstream.
                         The sweep runner auto-derives this from Component.downstream
                         in the System Map rather than requiring hand-typed config.

Note: spans with `component_id = None` (unmatched) are skipped — unmatched-span
      warnings are the mapping engine's responsibility, not this assertion's.

Note: a child span mapped to the SAME `component_id` as its parent is never a
      violation — that's a component's own internal sub-step (e.g. a manual
      `aeh.llm_call` span nested under that same component's outer wrapper
      span, per test_targets/multi_agent/components.py's PlannerComponent),
      not a fan-out to a different downstream component.
"""
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
        # Find children
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
