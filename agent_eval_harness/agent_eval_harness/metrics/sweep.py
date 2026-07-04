"""Sweep runner — orchestrates `aeh eval` (CS-263 §8).

Flow:
1. Load suite + system_map.
2. For assertion entries: run each relevant dataset case's query through
   execute_run (reuses CS-261 machinery verbatim), then score against spans.
3. For classifier entries: call the resolved component entry_point directly.
4. For llm_judge entries: run traces + score with DeepEval/RAGAS judges.
5. Score every entry, write evaluations rows via repository.insert_evaluation.
6. Modest async fan-out with asyncio.Semaphore(concurrency_limit).
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from agent_eval_harness.llm.client import LLMClient
from agent_eval_harness.mapping.system_map import SystemMap, load_system_map
from agent_eval_harness.metrics.assertions.registry import get_assertion
from agent_eval_harness.metrics.suite import SuiteEntry, load_suite
from agent_eval_harness.metrics.types import MetricResult
from agent_eval_harness.runner import execute_run
from agent_eval_harness.store import repository

logger = logging.getLogger("agent_eval_harness.sweep")


@dataclass
class SweepResult:
    run_id: str
    results: list[MetricResult] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)


async def run_sweep(
    target: str,
    map_path: str,
    suite_path: str,
    llm_client: LLMClient,
    *,
    concurrency: int = 4,
    tier: str = "auto",
) -> SweepResult:
    """Run the full evaluation sweep for a given target + suite.

    Returns a SweepResult containing all MetricResult items and any per-entry
    errors. Does NOT abort the whole sweep if one entry fails — that entry is
    recorded in SweepResult.errors and the sweep continues.
    """
    system_map = load_system_map(map_path)
    suite = load_suite(suite_path)

    run_id = await repository.insert_run(
        target_system_id=system_map.target_system_id,
        eval_plan_id=str(Path(suite_path).name),
    )

    sweep_result = SweepResult(run_id=run_id)
    semaphore = asyncio.Semaphore(concurrency)

    try:
        tasks = [
            _run_entry(entry, system_map, target, map_path, llm_client,
                       run_id, semaphore, tier)
            for entry in suite.entries
        ]
        gathered = await asyncio.gather(*tasks, return_exceptions=True)

        for entry, outcome in zip(suite.entries, gathered):
            if isinstance(outcome, BaseException):
                logger.error("Entry %s failed: %s", entry.id, outcome)
                sweep_result.errors.append({"entry_id": entry.id, "error": str(outcome)})
            elif isinstance(outcome, list):
                sweep_result.results.extend(outcome)
                # Persist each result
                for mr in outcome:
                    await repository.insert_evaluation(
                        run_id=run_id,
                        metric_name=mr.metric_name,
                        metric_class=mr.metric_class,
                        span_id=mr.span_id,
                        trace_id=mr.trace_id,
                        component_id=mr.component_id,
                        score=mr.score,
                        passed=mr.passed,
                        details=mr.details,
                        evaluator=mr.evaluator,
                        cost_tokens=mr.cost_tokens,
                    )
    except Exception:
        await repository.finish_run(run_id, "failed")
        raise

    await repository.finish_run(run_id, "completed")
    return sweep_result


async def _run_entry(
    entry: SuiteEntry,
    system_map: SystemMap,
    target: str,
    map_path: str,
    llm_client: LLMClient,
    run_id: str,
    semaphore: asyncio.Semaphore,
    tier: str,
) -> list[MetricResult]:
    """Dispatch one SuiteEntry to the appropriate scorer."""
    async with semaphore:
        if entry.metric_class == "assertion":
            return await _score_assertion_entry(
                entry, system_map, target, map_path, llm_client, run_id, tier
            )
        elif entry.metric_class == "classifier":
            return await _score_classifier_entry(entry, llm_client, entry.component)
        elif entry.metric_class == "llm_judge":
            return await _score_judge_entry(
                entry, system_map, target, map_path, llm_client, run_id, tier
            )
        else:
            raise ValueError(f"Unknown metric_class: {entry.metric_class!r}")


async def _score_assertion_entry(
    entry: SuiteEntry,
    system_map: SystemMap,
    target: str,
    map_path: str,
    llm_client: LLMClient,
    run_id: str,
    tier: str,
) -> list[MetricResult]:
    """Run traces for all dataset cases, then apply the assertion to each."""
    assertion_fn = get_assertion(entry.metric)

    queries = _get_queries_for_entry(entry, system_map)

    # Auto-derive 'allowed' from System Map if not provided
    params = dict(entry.params)
    if entry.metric == "allowed_downstream" and "allowed" not in params:
        component = system_map.component_by_id(entry.component)
        if component:
            params["allowed"] = component.downstream

    results: list[MetricResult] = []
    for query in queries:
        outcomes = await execute_run(target, map_path, llm_client, [query], tier)
        for outcome in outcomes:
            spans = await repository.get_spans_for_trace(outcome.trace_id)
            result = assertion_fn(spans, entry.component, params)
            result.trace_id = outcome.trace_id
            results.append(result)

    return results or [_empty_assertion_result(entry)]


async def _score_classifier_entry(
    entry: SuiteEntry,
    llm_client: LLMClient,
    component_id: str,
) -> list[MetricResult]:
    from agent_eval_harness.metrics.classifier import score_classifier

    dataset_ref = _resolve_dataset_ref(entry)
    if not dataset_ref:
        raise ValueError(f"Entry '{entry.id}' requires a dataset.ref for classifier scoring")

    # Resolve the component entry_point from entry.params
    entry_point = entry.params.get("entry_point", "")
    if not entry_point:
        raise ValueError(f"Entry '{entry.id}' requires params.entry_point for classifier scoring")

    result = await score_classifier(
        dataset_id=dataset_ref,
        component_entry_point=entry_point,
        llm_client=llm_client,
        component_id=component_id,
        metric_name=entry.metric,
    )
    return [result]


async def _score_judge_entry(
    entry: SuiteEntry,
    system_map: SystemMap,
    target: str,
    map_path: str,
    llm_client: LLMClient,
    run_id: str,
    tier: str,
) -> list[MetricResult]:
    """Run traces and apply LLM-judge scoring."""
    results: list[MetricResult] = []
    queries = _get_queries_for_entry(entry, system_map)

    for query in queries:
        outcomes = await execute_run(target, map_path, llm_client, [query], tier)
        for outcome in outcomes:
            spans = await repository.get_spans_for_trace(outcome.trace_id)
            mr = await _dispatch_judge(entry, spans, query, llm_client, outcome.trace_id)
            if mr:
                results.append(mr)

    return results


async def _dispatch_judge(
    entry: SuiteEntry,
    spans: list[dict],
    query: str,
    llm_client: LLMClient,
    trace_id: str,
) -> MetricResult | None:
    metric = entry.metric

    if metric.startswith("geval."):
        from agent_eval_harness.metrics.judges.deepeval_geval import run_geval

        rubric_text = entry.params.get("rubric_text", "Evaluate the quality of the response.")
        # Extract actual output from the component's span
        actual_output = _extract_span_output(spans, entry.component)
        return await run_geval(
            metric_name=metric[len("geval."):],
            rubric_text=rubric_text,
            input_text=query,
            actual_output=actual_output,
            llm_client=llm_client,
            component_id=entry.component,
            trace_id=trace_id,
        )

    elif metric == "ragas.faithfulness":
        from agent_eval_harness.metrics.judges.ragas_judge import run_ragas_faithfulness

        actual_answer = _extract_span_output(spans, entry.component)
        return await run_ragas_faithfulness(
            spans=spans,
            query=query,
            actual_answer=actual_answer,
            llm_client=llm_client,
            writer_component_id=entry.component,
            component_id=entry.component,
            trace_id=trace_id,
        )

    elif metric == "ragas.answer_relevancy":
        from agent_eval_harness.metrics.judges.ragas_judge import run_ragas_answer_relevancy

        actual_answer = _extract_span_output(spans, entry.component)
        return await run_ragas_answer_relevancy(
            query=query,
            actual_answer=actual_answer,
            llm_client=llm_client,
            component_id=entry.component,
            trace_id=trace_id,
        )

    elif metric == "tool_correctness":
        from agent_eval_harness.metrics.judges.deepeval_geval import run_tool_correctness

        expected_tools = entry.params.get("expected_tools", [])
        return await run_tool_correctness(
            spans=spans,
            component_id=entry.component,
            expected_tool_names=expected_tools,
            trace_id=trace_id,
        )

    logger.warning("Unknown judge metric '%s' for entry '%s' — skipping", metric, entry.id)
    return None


def _get_queries_for_entry(entry: SuiteEntry, system_map: SystemMap) -> list[str]:
    """Resolve queries for a suite entry: from dataset or from entry.params."""
    # Assertion/judge entries that need traces use params.queries or a default
    queries = entry.params.get("queries")
    if queries:
        return list(queries)
    # Default test query
    return ["Test query for evaluation sweep."]


def _resolve_dataset_ref(entry: SuiteEntry) -> str | None:
    if entry.dataset and entry.dataset.ref:
        return entry.dataset.ref
    return None


def _extract_span_output(spans: list[dict], component_id: str) -> str:
    """Extract the final text output from the last span of a component."""
    component_spans = [s for s in spans if s.get("component_id") == component_id]
    if not component_spans:
        return ""
    last = sorted(component_spans, key=lambda s: s.get("started_at", ""))[-1]
    raw = last.get("output_json") or "{}"
    try:
        data = json.loads(raw)
        for key in ("answer", "text", "content", "result", "output"):
            val = data.get(key)
            if isinstance(val, str) and val:
                return val
        return json.dumps(data)
    except json.JSONDecodeError:
        return raw


def _empty_assertion_result(entry: SuiteEntry) -> MetricResult:
    """Placeholder result when no traces could be produced for an assertion entry."""
    return MetricResult(
        metric_name=f"assertion.{entry.metric}",
        metric_class="assertion",
        score=None,
        passed=None,
        details={"note": "no traces produced — no queries resolved for this entry"},
        component_id=entry.component,
    )
