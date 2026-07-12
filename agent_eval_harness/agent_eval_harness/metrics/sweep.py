"""Sweep runner — orchestrates `aeh eval`."""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from agent_eval_harness.llm.client import LLMClient
from agent_eval_harness.llm.embedding_client import EmbeddingClient
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
    active_defects: list[str] | None = None,
    run_id: str | None = None,
    embedding_client: EmbeddingClient | None = None,
    source: Literal["live", "ingested"] = "live",
) -> SweepResult:
    """Run the full evaluation sweep for a given target + suite. `source="ingested"` scores
    an already-ingested Stage 4 run's persisted traces/spans instead of live-executing the
    target — `run_id` must already exist (ingest creates it); assertion gates are scored
    directly against the stored spans, judge/classifier gates aren't supported against
    ingested data yet and come back as an explicit not-yet-supported result."""
    system_map = load_system_map(map_path)
    suite = load_suite(suite_path)

    if source == "ingested" and run_id is None:
        raise ValueError("source='ingested' requires an existing run_id — ingest creates the run row")

    if run_id is None:
        run_id = await repository.insert_run(
            target_system_id=system_map.target_system_id,
            eval_plan_id=str(Path(suite_path).name),
            map_path=map_path,
            active_defects=active_defects,
            target=target,
            suite_path=suite_path,
        )

    sweep_result = SweepResult(run_id=run_id)
    semaphore = asyncio.Semaphore(concurrency)

    try:
        tasks = [
            _run_entry(
                entry,
                system_map,
                target,
                map_path,
                llm_client,
                run_id,
                semaphore,
                tier,
                active_defects=active_defects,
                embedding_client=embedding_client,
                source=source,
            )
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
                        entry_id=entry.id,
                        agent_id=entry.agent_id,
                    )
    except Exception:
        # An ingested run's status already reflects the injected execution itself (set at
        # ingest time) — a scoring failure here is a separate concern and must not overwrite it.
        if source == "live":
            await repository.finish_run(run_id, "failed")
        raise

    if source == "live":
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
    active_defects: list[str] | None = None,
    embedding_client: EmbeddingClient | None = None,
    source: Literal["live", "ingested"] = "live",
) -> list[MetricResult]:
    """Dispatch one SuiteEntry to the appropriate scorer."""
    async with semaphore:
        if source == "ingested":
            if entry.metric_class == "assertion":
                return await _score_assertion_entry_ingested(entry, system_map, run_id)
            return [_not_supported_for_ingested_result(entry)]
        if entry.metric_class == "assertion":
            return await _score_assertion_entry(
                entry,
                system_map,
                target,
                map_path,
                llm_client,
                run_id,
                tier,
                active_defects=active_defects,
            )
        elif entry.metric_class == "classifier":
            return await _score_classifier_entry(entry, llm_client, entry.component)
        elif entry.metric_class == "llm_judge":
            return await _score_judge_entry(
                entry,
                system_map,
                target,
                map_path,
                llm_client,
                run_id,
                tier,
                active_defects=active_defects,
                embedding_client=embedding_client,
            )
        else:
            raise ValueError(f"Unknown metric_class: {entry.metric_class!r}")


def _derive_assertion_params(entry: SuiteEntry, system_map: SystemMap) -> dict:
    params = dict(entry.params)
    if entry.metric == "allowed_downstream" and "allowed" not in params:
        component = system_map.component_by_id(entry.component)
        if component:
            params["allowed"] = component.downstream
    return params


async def _score_assertion_entry(
    entry: SuiteEntry,
    system_map: SystemMap,
    target: str,
    map_path: str,
    llm_client: LLMClient,
    run_id: str,
    tier: str,
    active_defects: list[str] | None = None,
) -> list[MetricResult]:
    """Run traces for all dataset cases, then apply the assertion to each."""
    assertion_fn = get_assertion(entry.metric)
    queries = _get_queries_for_entry(entry, system_map)
    params = _derive_assertion_params(entry, system_map)

    # Batch all queries into one execute_run call rather than one call per query.
    results: list[MetricResult] = []
    outcomes = await execute_run(
        target, map_path, llm_client, queries, tier, active_defects=active_defects, run_id=run_id
    )
    for outcome in outcomes:
        spans = await repository.get_spans_for_trace(outcome.trace_id)
        result = assertion_fn(spans, entry.component, params)
        result.trace_id = outcome.trace_id
        results.append(result)

    return results or [_empty_assertion_result(entry)]


async def _score_assertion_entry_ingested(
    entry: SuiteEntry, system_map: SystemMap, run_id: str
) -> list[MetricResult]:
    """Scores an assertion gate directly against an already-ingested run's persisted traces —
    one result per trace (one trace = one dataset case = one full injected pipeline run),
    mirroring the live path's one-result-per-outcome convention exactly."""
    assertion_fn = get_assertion(entry.metric)
    params = _derive_assertion_params(entry, system_map)

    traces = await repository.get_traces_for_run(run_id)
    if not traces:
        return [_empty_assertion_result(entry)]

    results: list[MetricResult] = []
    for trace in traces:
        spans = await repository.get_spans_for_trace(trace["id"])
        result = assertion_fn(spans, entry.component, params)
        result.trace_id = trace["id"]
        results.append(result)
    return results


def _not_supported_for_ingested_result(entry: SuiteEntry) -> MetricResult:
    return MetricResult(
        metric_name=f"{entry.metric_class}.{entry.metric}",
        metric_class=entry.metric_class,
        score=None,
        passed=None,
        details={"reason": "scoring judge/classifier gates against an ingested run is not supported yet"},
        component_id=entry.component,
    )


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
    active_defects: list[str] | None = None,
    embedding_client: EmbeddingClient | None = None,
) -> list[MetricResult]:
    """Run traces and apply LLM-judge scoring."""
    results: list[MetricResult] = []
    queries = _get_queries_for_entry(entry, system_map)

    # Batch all queries into one execute_run call — see _score_assertion_entry for why.
    outcomes = await execute_run(
        target, map_path, llm_client, queries, tier, active_defects=active_defects, run_id=run_id
    )
    for query, outcome in zip(queries, outcomes):
        spans = await repository.get_spans_for_trace(outcome.trace_id)
        mr = await _dispatch_judge(
            entry, spans, query, llm_client, outcome.trace_id, embedding_client=embedding_client
        )
        if mr:
            results.append(mr)

    return results


async def _dispatch_judge(
    entry: SuiteEntry,
    spans: list[dict],
    query: str,
    llm_client: LLMClient,
    trace_id: str,
    embedding_client: EmbeddingClient | None = None,
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
            embedding_client=embedding_client,
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

    logger.warning(
        "Unknown judge metric '%s' for entry '%s' — unregistered metric reached dispatch",
        metric, entry.id,
    )
    raise ValueError(
        f"Metric '{metric}' is not registered in METRIC_REGISTRY and cannot be dispatched. "
        "Register it or remove it from the plan."
    )


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
