"""T2 multi_agent: guard -> planner (internally orchestrates worker/judge/writer)."""
from __future__ import annotations

from haystack import AsyncPipeline

from agent_eval_harness.instrumentation.base import PipelineHandle
from agent_eval_harness.llm.client import LLMClient
from test_targets._shared.defects import DefectConfig
from test_targets.multi_agent.components import (
    GuardComponent,
    JudgeComponent,
    PlannerComponent,
    WorkerComponent,
    WriterComponent,
)
from test_targets.multi_agent.tools import case_law_search, decoy_lookup


def build_pipeline(llm_client: LLMClient, defects: DefectConfig | None = None) -> PipelineHandle:
    defects = defects or DefectConfig.from_env()

    worker = WorkerComponent(
        {"case_law_search": case_law_search, "decoy_lookup": decoy_lookup}, defects
    )
    judge = JudgeComponent(llm_client, defects)
    writer = WriterComponent(llm_client, defects)

    pipeline = AsyncPipeline()
    pipeline.add_component("guard", GuardComponent(llm_client, defects))
    pipeline.add_component("planner", PlannerComponent(llm_client, worker, judge, writer, defects))
    pipeline.connect("guard.query", "planner.query")
    pipeline.connect("guard.passed", "planner.passed")

    return PipelineHandle(
        pipeline=pipeline,
        entry_inputs={"guard": "query"},
        exit_node="planner",
    )
