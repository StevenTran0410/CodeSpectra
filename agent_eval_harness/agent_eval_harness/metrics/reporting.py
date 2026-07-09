"""Metric reporting for `aeh report` — NOT the same as agent_eval_harness/reporting.py."""
from __future__ import annotations

import json
from typing import Any

from agent_eval_harness.metrics.types import MetricResult
from agent_eval_harness.store import repository


def format_metric_result(mr: MetricResult) -> str:
    """Format a single MetricResult into a human-readable line."""
    status = "PASS" if mr.passed is True else ("FAIL" if mr.passed is False else "N/A")
    score_str = f"  score={mr.score:.3f}" if mr.score is not None else ""
    tokens_str = f"  tokens={mr.cost_tokens}" if mr.cost_tokens is not None else ""
    return f"  [{status}] {mr.metric_name}{score_str}{tokens_str}"


def format_sweep_results(
    results: list[MetricResult],
    system_map_id: str = "",
) -> str:
    """Format all MetricResult items grouped by component_id."""
    # Group by component_id
    by_component: dict[str, list[MetricResult]] = {}
    for mr in results:
        key = mr.component_id or "(global)"
        by_component.setdefault(key, []).append(mr)

    lines: list[str] = []
    if system_map_id:
        lines.append(f"=== Evaluation Report: {system_map_id} ===")
    else:
        lines.append("=== Evaluation Report ===")

    total_cost = 0
    for cid, mrs in sorted(by_component.items()):
        lines.append(f"\n[{cid}]")
        for mr in mrs:
            lines.append(format_metric_result(mr))
            total_cost += mr.cost_tokens or 0

    lines.append(f"\nevaluator cost: {total_cost:,} judge tokens")
    return "\n".join(lines)


def results_to_json(results: list[MetricResult], run_id: str) -> str:
    """Serialize MetricResults to machine-readable JSON (for CI consumption)."""
    data: dict[str, Any] = {
        "run_id": run_id,
        "total_results": len(results),
        "results": [
            {
                "metric_name": mr.metric_name,
                "metric_class": mr.metric_class,
                "score": mr.score,
                "passed": mr.passed,
                "component_id": mr.component_id,
                "trace_id": mr.trace_id,
                "span_id": mr.span_id,
                "evaluator": mr.evaluator,
                "cost_tokens": mr.cost_tokens,
                "details": mr.details,
            }
            for mr in results
        ],
    }
    return json.dumps(data, indent=2)


async def render_run_report(run_id: str, *, as_json: bool = False) -> str:
    """Fetch evaluations for a completed run and format them for display."""
    rows = await repository.get_evaluations_for_run(run_id)

    if not rows:
        return f"No evaluations found for run {run_id}."

    results = [
        MetricResult(
            metric_name=row["metric_name"],
            metric_class=row["metric_class"],  # type: ignore[arg-type]
            score=row.get("score"),
            passed=bool(row["passed"]) if row.get("passed") is not None else None,
            details=json.loads(row.get("details_json") or "{}"),
            component_id=row.get("component_id"),
            span_id=row.get("span_id"),
            trace_id=row.get("trace_id"),
            evaluator=row.get("evaluator"),
            cost_tokens=row.get("cost_tokens"),
        )
        for row in rows
    ]

    if as_json:
        return results_to_json(results, run_id)
    return format_sweep_results(results)
