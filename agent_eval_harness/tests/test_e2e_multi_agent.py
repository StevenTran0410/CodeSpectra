"""Full stubbed end-to-end tests for T2 multi_agent — one test per defect"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_eval_harness.instrumentation.tier1_haystack import HaystackAdapter
from agent_eval_harness.llm.client import LLMResponse
from agent_eval_harness.llm.fake_client import FakeLLMClient
from agent_eval_harness.mapping.system_map import load_system_map
from test_targets._shared.defects import DefectConfig
from test_targets.multi_agent.pipeline import build_pipeline

pytestmark = pytest.mark.asyncio

_MAP_PATH = Path(__file__).parent.parent / "test_targets" / "multi_agent" / "system_map.yaml"

_QUERY = "Can I get a refund and also change my shipping address?"


async def _run(responses: list[LLMResponse], defects: DefectConfig, query: str = _QUERY):
    llm_client = FakeLLMClient(responses)
    handle = build_pipeline(llm_client, defects)
    system_map = load_system_map(_MAP_PATH)

    adapter = HaystackAdapter(handle, system_map)
    adapter.attach()
    try:
        return await adapter.run(query)
    finally:
        adapter.detach()


def _spans_for(result, component_id: str) -> list:
    return [s for s in result.spans if s.component_id == component_id]


async def test_happy_path_no_defects() -> None:
    responses = [
        LLMResponse(content="not a policy violation", model="fake-nano"),  # guard llm
        LLMResponse(content='["refund request", "shipping address change"]', model="fake-frontier"),
        LLMResponse(content="sufficient context found", model="fake-strong"),  # judge
        LLMResponse(content="Here is your answer.", model="fake-mini"),  # writer
    ]
    result = await _run(responses, DefectConfig())

    assert len(_spans_for(result, "worker")) == 1  # sufficient on first try, no retry
    assert len(_spans_for(result, "judge")) == 1
    assert len(_spans_for(result, "case_law_search_tool")) == 1
    assert len(_spans_for(result, "decoy_tool")) == 1
    # guard's own top-level wrapper span legitimately has no span_match rule of
    assert len(result.unmatched) == 1


async def test_defect_planner_overpack() -> None:
    responses = [
        LLMResponse(content="valid", model="fake-nano"),
        LLMResponse(content='["i1", "i2", "i3", "i4"]', model="fake-frontier"),
        LLMResponse(content="sufficient", model="fake-strong"),
        LLMResponse(content="answer", model="fake-mini"),
    ]
    result = await _run(responses, DefectConfig(planner_overpack=True))

    worker_spans = _spans_for(result, "worker")
    assert len(worker_spans) == 1  # all 4 intents forced into ONE call
    payload = json.loads(worker_spans[0].output_json)
    assert len(payload["intents"]) == 4


async def test_no_defect_planner_respects_fanout_two() -> None:
    """Companion to the overpack test: proves normal behavior IS capped at 2."""
    responses = [
        LLMResponse(content="valid", model="fake-nano"),
        LLMResponse(content='["i1", "i2", "i3", "i4"]', model="fake-frontier"),
        LLMResponse(content="sufficient", model="fake-strong"),
        LLMResponse(content="answer", model="fake-mini"),
    ]
    result = await _run(responses, DefectConfig(planner_overpack=False))

    worker_spans = _spans_for(result, "worker")
    assert len(worker_spans) == 2  # 4 intents batched 2-per-call
    for span in worker_spans:
        payload = json.loads(span.output_json)
        assert len(payload["intents"]) <= 2


async def test_defect_guard_leak() -> None:
    responses = [
        LLMResponse(content="this is a policy violation, reject it", model="fake-nano"),
        LLMResponse(content='["intent"]', model="fake-frontier"),
        LLMResponse(content="sufficient", model="fake-strong"),
        LLMResponse(content="answer", model="fake-mini"),
    ]
    result = await _run(responses, DefectConfig(guard_leak=True))

    guard_llm_spans = _spans_for(result, "guard_llm")
    assert len(guard_llm_spans) == 1
    payload = json.loads(guard_llm_spans[0].output_json)
    assert payload["verdict"] == "pass"  # leaked despite classifier saying reject


async def test_no_defect_guard_rejects_policy_violation() -> None:
    responses = [
        LLMResponse(content="this is a policy violation, reject it", model="fake-nano"),
        LLMResponse(content='["intent"]', model="fake-frontier"),
        LLMResponse(content="sufficient", model="fake-strong"),
        LLMResponse(content="answer", model="fake-mini"),
    ]
    result = await _run(responses, DefectConfig(guard_leak=False))

    guard_llm_spans = _spans_for(result, "guard_llm")
    payload = json.loads(guard_llm_spans[0].output_json)
    assert payload["verdict"] == "reject"


async def test_defect_wrong_tool() -> None:
    responses = [
        LLMResponse(content="valid", model="fake-nano"),
        LLMResponse(content='["intent"]', model="fake-frontier"),
        LLMResponse(content="sufficient", model="fake-strong"),
        LLMResponse(content="answer", model="fake-mini"),
    ]
    result = await _run(responses, DefectConfig(wrong_tool=True))

    tool_spans = [s for s in result.spans if s.span_type == "tool_call"]
    assert tool_spans[0].tags.get("aeh.tool.name") == "decoy_lookup"


async def test_defect_judge_rubber_stamp() -> None:
    responses = [
        LLMResponse(content="valid", model="fake-nano"),
        LLMResponse(content='["intent"]', model="fake-frontier"),
        # no judge response needed — rubber-stamping skips the LLM call entirely
        LLMResponse(content="answer", model="fake-mini"),
    ]
    result = await _run(responses, DefectConfig(judge_rubber_stamp=True))

    judge_spans = _spans_for(result, "judge")
    assert len(judge_spans) == 1  # no retry triggered
    payload = json.loads(judge_spans[0].output_json)
    assert payload["sufficient"] is True
    assert payload.get("rubber_stamped") is True


async def test_defect_no_retry() -> None:
    responses = [
        LLMResponse(content="valid", model="fake-nano"),
        LLMResponse(content='["intent"]', model="fake-frontier"),
        LLMResponse(content="insufficient context, need more", model="fake-strong"),
        LLMResponse(content="answer", model="fake-mini"),
    ]
    result = await _run(responses, DefectConfig(no_retry=True))

    assert len(_spans_for(result, "worker")) == 1  # retry skipped despite rejection


async def test_no_defect_retries_once_on_rejection() -> None:
    """Companion: proves normal behavior DOES retry once when judge rejects."""
    responses = [
        LLMResponse(content="valid", model="fake-nano"),
        LLMResponse(content='["intent"]', model="fake-frontier"),
        LLMResponse(content="insufficient context, need more", model="fake-strong"),
        LLMResponse(content="answer", model="fake-mini"),
    ]
    result = await _run(responses, DefectConfig(no_retry=False))

    assert len(_spans_for(result, "worker")) == 2
    assert len(_spans_for(result, "judge")) == 2


async def test_defect_writer_hallucinate() -> None:
    """Same metric code, two topologies — companion to T1's hallucinate test."""
    responses = [
        LLMResponse(content="valid", model="fake-nano"),
        LLMResponse(content='["intent"]', model="fake-frontier"),
        LLMResponse(content="sufficient", model="fake-strong"),
        LLMResponse(content="Here is the answer.", model="fake-mini"),
    ]
    result = await _run(responses, DefectConfig(writer_hallucinate=True))

    assert "full refund" in result.final_output
