"""DeepEval G-Eval and Tool Correctness judges (CS-263 §6.1)."""
from __future__ import annotations

import json

from agent_eval_harness.llm.client import LLMClient
from agent_eval_harness.metrics.types import MetricResult, extract_cost_tokens


async def run_geval(
    metric_name: str,
    rubric_text: str,
    input_text: str,
    actual_output: str,
    llm_client: LLMClient,
    *,
    expected_output: str | None = None,
    component_id: str | None = None,
    trace_id: str | None = None,
    additional_metadata: dict | None = None,
) -> MetricResult:
    """Run a G-Eval metric via DeepEval."""
    try:
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    except ImportError as exc:
        raise ImportError(
            "deepeval is not installed. Install it via `pip install deepeval` "
            "or the [datasets] optional dependency."
        ) from exc

    from agent_eval_harness.llm.deepeval_adapter import make_deepeval_llm_adapter

    adapter = make_deepeval_llm_adapter(llm_client)

    params = [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
    if expected_output is not None:
        params.append(LLMTestCaseParams.EXPECTED_OUTPUT)

    test_case = LLMTestCase(
        input=input_text,
        actual_output=actual_output,
        expected_output=expected_output,
    )

    metric = GEval(
        name=metric_name,
        criteria=rubric_text,
        evaluation_params=params,
        model=adapter,
        verbose_mode=False,
    )

    await metric.a_measure(test_case)

    score: float | None = getattr(metric, "score", None)
    reason: str | None = getattr(metric, "reason", None)

    return MetricResult(
        metric_name=f"llm_judge.geval.{metric_name}",
        metric_class="llm_judge",
        score=score,
        passed=score is not None and score >= 0.5,
        details={
            "rubric": rubric_text,
            "reason": reason,
            **(additional_metadata or {}),
        },
        component_id=component_id,
        trace_id=trace_id,
        evaluator="deepeval.GEval",
        cost_tokens=extract_cost_tokens(adapter),
    )


async def run_tool_correctness(
    spans: list[dict],
    component_id: str,
    expected_tool_names: list[str],
    *,
    trace_id: str | None = None,
) -> MetricResult:
    """Tool Correctness metric — deterministic tool-name comparison."""
    actual_tools: list[str] = []
    for span in spans:
        if span.get("span_type") != "tool_call":
            continue
        details = json.loads(span.get("details_json") or "{}")
        tool_name = details.get("raw_tags", {}).get("aeh.tool.name")
        if tool_name:
            actual_tools.append(tool_name)

    correct = sum(1 for t in actual_tools if t in expected_tool_names)
    total = len(actual_tools) if actual_tools else 1
    score = correct / total
    passed = set(actual_tools) == set(expected_tool_names)

    return MetricResult(
        metric_name="llm_judge.tool_correctness",
        metric_class="llm_judge",
        score=score,
        passed=passed,
        details={
            "actual_tools": actual_tools,
            "expected_tools": expected_tool_names,
            "correct": correct,
        },
        component_id=component_id,
        trace_id=trace_id,
        evaluator="deepeval.ToolCorrectness",
        cost_tokens=0,
    )
