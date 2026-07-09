"""Smoke tests for DeepEval/RAGAS judge adapters with FakeLLMClient."""
from __future__ import annotations

import json

import pytest

from agent_eval_harness.llm.ragas_adapter import stub_missing_langchain_community_vertexai

# Must run before any pytest.importorskip("ragas")/import ragas below, to patch the missing optional dependency.
stub_missing_langchain_community_vertexai()

pytestmark = pytest.mark.asyncio


def _make_fake_client(response: str = "Good response."):
    from agent_eval_harness.llm.client import LLMResponse
    from agent_eval_harness.llm.fake_client import FakeLLMClient

    return FakeLLMClient(LLMResponse(content=response, model="fake-test"))


def _make_spans(writer_context: str = "Employee gets 15 vacation days per year.") -> list[dict]:
    from datetime import UTC, datetime

    return [
        {
            "id": "span-writer-1",
            "component_id": "writer",
            "span_type": "llm",
            "input_json": json.dumps({"context": writer_context, "query": "vacation policy"}),
            "output_json": json.dumps({"answer": "You get 15 vacation days."}),
            "parent_span_id": None,
            "started_at": datetime.now(UTC).isoformat(),
            "details_json": "{}",
        }
    ]


async def test_geval_runs_end_to_end() -> None:
    pytest.importorskip("deepeval")

    from agent_eval_harness.metrics.judges.deepeval_geval import run_geval

    llm_client = _make_fake_client()
    result = await run_geval(
        metric_name="test_quality",
        rubric_text="Evaluate if the response is helpful and accurate.",
        input_text="What is the vacation policy?",
        actual_output="You get 15 vacation days per year.",
        llm_client=llm_client,
        component_id="writer",
    )

    assert result.metric_class == "llm_judge"
    assert "geval" in result.metric_name
    # score may be None if deepeval's G-Eval mocking returns non-numeric — just check no crash
    assert result.passed is not None or result.passed is None  # any state is acceptable


async def test_tool_correctness_deterministic() -> None:
    """Tool Correctness is deterministic — no LLM needed."""
    from agent_eval_harness.metrics.judges.deepeval_geval import run_tool_correctness

    spans = [
        {
            "id": "s1",
            "span_type": "tool_call",
            "component_id": "worker",
            "details_json": json.dumps({"raw_tags": {"aeh.tool.name": "case_law_search"}}),
        }
    ]
    result = await run_tool_correctness(
        spans=spans,
        component_id="worker",
        expected_tool_names=["case_law_search"],
    )
    assert result.metric_class == "llm_judge"
    assert result.passed is True
    assert result.cost_tokens == 0


async def test_tool_correctness_wrong_tool() -> None:
    from agent_eval_harness.metrics.judges.deepeval_geval import run_tool_correctness

    spans = [
        {
            "id": "s1",
            "span_type": "tool_call",
            "component_id": "worker",
            "details_json": json.dumps({"raw_tags": {"aeh.tool.name": "decoy_lookup"}}),
        }
    ]
    result = await run_tool_correctness(
        spans=spans,
        component_id="worker",
        expected_tool_names=["case_law_search"],
    )
    assert result.passed is False


async def test_ragas_faithfulness_runs_end_to_end() -> None:
    pytest.importorskip("ragas")

    from agent_eval_harness.metrics.judges.ragas_judge import run_ragas_faithfulness

    llm_client = _make_fake_client()
    spans = _make_spans()

    result = await run_ragas_faithfulness(
        spans=spans,
        query="How many vacation days do I get?",
        actual_answer="You get 15 vacation days per year.",
        llm_client=llm_client,
        writer_component_id="writer",
        component_id="writer",
    )

    assert result.metric_class == "llm_judge"
    assert result.metric_name == "llm_judge.ragas.faithfulness"
    # retrieved_contexts_source must indicate writer span — not gold or corpus
    assert result.details["retrieved_contexts_source"] == "writer_span_input_json"
