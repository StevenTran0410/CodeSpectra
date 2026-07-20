"""Full stubbed end-to-end test for T1 linear_rag."""
from __future__ import annotations

from pathlib import Path

import pytest

from agent_eval_harness.instrumentation.tier1_haystack import HaystackAdapter
from agent_eval_harness.llm.client import LLMResponse
from agent_eval_harness.llm.fake_client import FakeLLMClient
from agent_eval_harness.mapping.system_map import load_system_map
from test_targets._shared.defects import DefectConfig
from test_targets.linear_rag.pipeline import build_pipeline

_MAP_PATH = Path(__file__).parent.parent / "test_targets" / "linear_rag" / "system_map.yaml"


async def _run(query: str, response: LLMResponse, defects: DefectConfig):
    llm_client = FakeLLMClient(response)
    handle = build_pipeline(llm_client, defects)
    system_map = load_system_map(_MAP_PATH)

    adapter = HaystackAdapter(handle, system_map)
    adapter.attach()
    try:
        return await adapter.run(query), system_map
    finally:
        adapter.detach()


async def test_linear_rag_happy_path() -> None:
    response = LLMResponse(
        content="Employees get 15 days of vacation per year.",
        model="fake-mini",
        prompt_tokens=10,
        completion_tokens=8,
    )
    result, _ = await _run("What is the vacation policy?", response, DefectConfig())

    component_ids = {s.component_id for s in result.spans}
    assert component_ids == {"retriever", "writer"}
    assert result.unmatched == []
    assert "vacation" in result.final_output.lower()
    assert result.total_tokens == 18


async def test_linear_rag_writer_hallucinate_defect() -> None:
    response = LLMResponse(content="Employees get 15 days of vacation per year.", model="fake-mini")
    result, _ = await _run(
        "What is the vacation policy?", response, DefectConfig(writer_hallucinate=True)
    )

    assert "full refund" in result.final_output
