"""Assertion: retry_on_reject_required — verifies the orchestrator retries after rejection.

For each judge span (validator role component) whose `output_json.sufficient == false`,
checks that a LATER worker span exists in the same trace (by execution order) before
the trace's final writer span.

`DEFECT_NO_RETRY` breaks this: with the defect on, there is only one worker span
total — no second one after the rejection — so this assertion fails.

Ordering is by LIST POSITION, not by comparing `started_at` strings: fast
synchronous runs (FakeLLMClient, no real I/O) routinely produce several spans
with an IDENTICAL started_at timestamp (Windows wall-clock resolution can be
coarser than actual execution speed), so `a.started_at > b.started_at` is not a
reliable "happened after" signal even when `a` truly ran after `b`. The `spans`
list itself is already in true execution order (repository.get_spans_for_trace
orders by rowid == insertion order == completion order), so position in that
list is the correct ordering signal to use instead.

params:
    judge_component_id (str): component ID of the validator (default: "judge").
    worker_component_id (str): component ID of the retrieval agent (default: "worker").
    writer_component_id (str): component ID of the writer (default: "writer").
"""
from __future__ import annotations

import json

from agent_eval_harness.metrics.assertions.registry import register
from agent_eval_harness.metrics.types import MetricResult


@register("retry_on_reject_required")
def retry_on_reject_required(spans: list[dict], component_id: str, params: dict) -> MetricResult:
    judge_cid = params.get("judge_component_id", "judge")
    worker_cid = params.get("worker_component_id", "worker")
    writer_cid = params.get("writer_component_id", "writer")

    # spans is assumed to already be in true execution order (see docstring).
    rejection_indices: list[int] = []
    for i, span in enumerate(spans):
        if span.get("component_id") != judge_cid:
            continue
        output = json.loads(span.get("output_json") or "{}")
        if output.get("sufficient") is False:
            rejection_indices.append(i)

    if not rejection_indices:
        # No rejections occurred — assertion is vacuously true
        return MetricResult(
            metric_name="assertion.retry_on_reject_required",
            metric_class="assertion",
            score=None,
            passed=True,
            details={"rejections": 0, "retries_found": 0, "note": "no rejections in trace"},
            component_id=component_id,
        )

    writer_index = next(
        (i for i, s in enumerate(spans) if s.get("component_id") == writer_cid), len(spans)
    )

    missing_retries: list[str] = []
    for ridx in rejection_indices:
        # Look for a worker span strictly between this rejection and the writer.
        retry_found = any(
            spans[j].get("component_id") == worker_cid for j in range(ridx + 1, writer_index)
        )
        if not retry_found:
            missing_retries.append(spans[ridx]["id"])

    passed = len(missing_retries) == 0
    return MetricResult(
        metric_name="assertion.retry_on_reject_required",
        metric_class="assertion",
        score=None,
        passed=passed,
        details={
            "rejections": len(rejection_indices),
            "missing_retries_after_rejection_span_ids": missing_retries,
        },
        component_id=component_id,
    )
