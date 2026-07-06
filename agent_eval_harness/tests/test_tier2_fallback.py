"""Tier-2 boundary-wrapper fallback demo — T1 only (CS-261 §7 acceptance:"""
from __future__ import annotations

from pathlib import Path

import pytest

from agent_eval_harness.instrumentation.tier2_boundary import BoundaryWrapperAdapter
from agent_eval_harness.llm.client import LLMResponse
from agent_eval_harness.llm.fake_client import FakeLLMClient
from agent_eval_harness.mapping.system_map import load_system_map
from test_targets.linear_rag.pipeline import set_default_llm_client

pytestmark = pytest.mark.asyncio

_MAP_PATH = Path(__file__).parent.parent / "test_targets" / "linear_rag" / "system_map.yaml"


async def test_tier2_fallback_coarser_no_nested_spans() -> None:
    set_default_llm_client(
        FakeLLMClient(LLMResponse(content="Vacation is 15 days per year.", model="fake-mini"))
    )
    system_map = load_system_map(_MAP_PATH)
    adapter = BoundaryWrapperAdapter(system_map, ["retriever", "writer"])

    adapter.attach()
    try:
        result = await adapter.run("What is the vacation policy?")
    finally:
        adapter.detach()

    assert len(result.spans) == 2  # coarse: exactly one span per component
    assert {s.component_id for s in result.spans} == {"retriever", "writer"}
    for span in result.spans:
        assert span.tier == "tier2"
        assert span.parent_span_id is None  # no nesting at this granularity
    assert "vacation" in result.final_output.lower()
