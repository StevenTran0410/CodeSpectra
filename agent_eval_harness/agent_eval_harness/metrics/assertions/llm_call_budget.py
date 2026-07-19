"""Assertion: llm_call_budget — LLM span count for component ≤ contract's llm_call_budget."""
from __future__ import annotations

from agent_eval_harness.metrics.assertions.registry import register
from agent_eval_harness.metrics.types import MetricResult


@register("llm_call_budget")
def llm_call_budget(spans: list[dict], component_id: str, params: dict) -> MetricResult:
    """Count LLM spans for a component and check against budget."""
    budget = params.get("llm_call_budget")
    if budget is None:
        return MetricResult(
            metric_name="assertion.llm_call_budget",
            metric_class="assertion",
            score=None,
            passed=None,
            details={"reason": "no_budget_specified"},
            component_id=component_id,
        )

    llm_span_count = 0
    for span in spans:
        if span.get("component_id") != component_id:
            continue
        if span.get("span_type") == "llm_call":
            llm_span_count += 1

    passed = llm_span_count <= budget
    return MetricResult(
        metric_name="assertion.llm_call_budget",
        metric_class="assertion",
        score=None,
        passed=passed,
        details={
            "llm_call_count": llm_span_count,
            "budget": budget,
            "exceeded_by": max(0, llm_span_count - budget),
        },
        component_id=component_id,
    )
