"""Assertion: trajectory_termination — verifies a component with control loops terminates within bounds."""
from __future__ import annotations

from agent_eval_harness.metrics.assertions.registry import register
from agent_eval_harness.metrics.types import MetricResult

DEFAULT_MAX_ITERATIONS = 5


@register("trajectory_termination")
def trajectory_termination(spans: list[dict], component_id: str, params: dict) -> MetricResult:
    """Verify that a component with loop motif or control_motif terminates within iteration bounds.

    Counts the number of spans for this component and ensures it doesn't exceed the maximum
    allowed iterations, indicating proper termination and feedback incorporation.
    """
    max_iterations = int(params.get("max_iterations", DEFAULT_MAX_ITERATIONS))
    component_spans = [s for s in spans if s.get("component_id") == component_id]
    span_count = len(component_spans)
    max_allowed = max_iterations
    passed = span_count <= max_allowed
    return MetricResult(
        metric_name="assertion.trajectory_termination",
        metric_class="assertion",
        score=None,
        passed=passed,
        details={
            "span_count": span_count,
            "max_allowed": max_allowed,
            "max_iterations": max_iterations,
        },
        component_id=component_id,
    )
