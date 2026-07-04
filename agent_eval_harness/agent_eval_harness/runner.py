"""Orchestrates one CLI `run` invocation end-to-end: load map, build/attach the
right adapter, run query(ies), persist to the store. No metrics/datasets yet —
CS-262/263's job.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass

from agent_eval_harness.instrumentation.base import (
    InstrumentationAdapter,
    PipelineHandle,
    TraceResult,
)
from agent_eval_harness.instrumentation.tier1_haystack import HaystackAdapter
from agent_eval_harness.instrumentation.tier2_boundary import BoundaryWrapperAdapter
from agent_eval_harness.llm.client import LLMClient
from agent_eval_harness.mapping.system_map import SystemMap, load_system_map
from agent_eval_harness.store import repository


def _resolve_build_pipeline(target: str):
    module_path, _, func_name = target.partition(":")
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


def _resolve_linear_order(system_map: SystemMap) -> list[str]:
    """BFS from upstream=[] roots, following downstream edges. Only guaranteed
    correct for linear chains (T1) — CS-261's Tier-2 acceptance only requires
    that shape; arbitrary-topology tier-2 chaining is out of scope.
    """
    roots = [c.id for c in system_map.components if not c.upstream]
    order: list[str] = []
    seen: set[str] = set()
    queue = list(roots)
    while queue:
        component_id = queue.pop(0)
        if component_id in seen:
            continue
        seen.add(component_id)
        order.append(component_id)
        component = system_map.component_by_id(component_id)
        if component:
            for next_id in component.downstream:
                if next_id not in seen:
                    queue.append(next_id)
    return order


def build_adapter(
    target: str,
    map_path: str,
    llm_client: LLMClient,
    tier: str = "auto",
) -> tuple[InstrumentationAdapter, SystemMap]:
    system_map = load_system_map(map_path)

    if tier == "2":
        component_order = _resolve_linear_order(system_map)
        return BoundaryWrapperAdapter(system_map, component_order), system_map

    build_pipeline = _resolve_build_pipeline(target)
    handle: PipelineHandle = build_pipeline(llm_client)
    return HaystackAdapter(handle, system_map), system_map


@dataclass
class RunOutcome:
    run_id: str
    trace_id: str
    trace_result: TraceResult


async def execute_run(
    target: str,
    map_path: str,
    llm_client: LLMClient,
    queries: list[str],
    tier: str = "auto",
) -> list[RunOutcome]:
    adapter, system_map = build_adapter(target, map_path, llm_client, tier)
    run_id = await repository.insert_run(target_system_id=system_map.target_system_id)

    outcomes: list[RunOutcome] = []
    adapter.attach()
    try:
        for query in queries:
            trace_id = await repository.insert_trace(run_id, root_input=query)
            result = await adapter.run(query)
            await repository.insert_spans_bulk(trace_id, result.spans)
            await repository.finalize_trace(
                trace_id, result.final_output, result.total_tokens, result.total_latency_ms
            )
            outcomes.append(RunOutcome(run_id=run_id, trace_id=trace_id, trace_result=result))
    except Exception:
        await repository.finish_run(run_id, "failed")
        raise
    finally:
        adapter.detach()

    await repository.finish_run(run_id, "completed")
    return outcomes
