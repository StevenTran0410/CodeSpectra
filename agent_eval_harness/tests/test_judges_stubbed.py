"""Smoke tests for DeepEval/RAGAS judge adapters with FakeLLMClient."""
from __future__ import annotations

import json

import pytest

from agent_eval_harness.llm.ragas_adapter import stub_missing_langchain_community_vertexai

# Must run before any pytest.importorskip("ragas")/import ragas below, to patch the missing optional dependency.
stub_missing_langchain_community_vertexai()


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

    from agent_eval_harness.llm.client import LLMResponse
    from agent_eval_harness.llm.fake_client import FakeLLMClient
    from agent_eval_harness.metrics.judges.deepeval_geval import run_geval

    # GEval.a_measure() makes 2 schema-constrained calls in order: generate steps (schema=Steps),
    # then score against them (schema=ReasonScore) — our adapter has no native
    # a_generate_raw_response, so it falls into the schema-based fallback path.
    steps_response = LLMResponse(
        content=json.dumps({
            "steps": [
                "Check whether the actual output directly answers the input question.",
                "Check whether the actual output is factually consistent with the input.",
            ]
        }),
        model="fake-test",
    )
    # GEval's default score_range is 0-10 (no custom rubric); it normalizes by dividing by the range span, so "8" here becomes 0.8 below.
    score_response = LLMResponse(
        content=json.dumps({
            "score": 8,
            "reason": "The response directly and accurately answers the vacation policy question.",
        }),
        model="fake-test",
    )
    llm_client = FakeLLMClient([steps_response, score_response])

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
    assert result.score == 0.8
    assert result.passed is True
    assert "vacation policy" in result.details["reason"]


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


async def test_ragas_answer_relevancy_no_embedding_client_degrades() -> None:
    pytest.importorskip("ragas")

    from agent_eval_harness.metrics.judges.ragas_judge import run_ragas_answer_relevancy

    llm_client = _make_fake_client()

    result = await run_ragas_answer_relevancy(
        query="How many vacation days do I get?",
        actual_answer="You get 15 vacation days per year.",
        llm_client=llm_client,
        component_id="writer",
        embedding_client=None,
    )

    assert result.metric_class == "llm_judge"
    assert result.metric_name == "llm_judge.ragas.answer_relevancy"
    assert result.score is None
    assert result.passed is False
    assert "no embedding_client configured" in result.details["error"]


async def test_ragas_answer_relevancy_runs_end_to_end_with_embedding_client() -> None:
    pytest.importorskip("ragas")

    from agent_eval_harness.metrics.judges.ragas_judge import run_ragas_answer_relevancy

    # Ragas AnswerRelevancy generates 3 questions (LLM), embeds them + the original (Embeddings),
    # then cosine-similarity scores them — both clients need mocking.
    llm_client = _make_fake_client(
        response=json.dumps({
            "question": "What is the vacation policy?",
            "noncommittal": 0
        })
    )

    class FakeEmbeddingClient:
        async def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    fake_embed = FakeEmbeddingClient()

    result = await run_ragas_answer_relevancy(
        query="How many vacation days do I get?",
        actual_answer="You get 15 vacation days per year.",
        llm_client=llm_client,
        component_id="writer",
        embedding_client=fake_embed,  # type: ignore[arg-type]
    )

    assert result.metric_class == "llm_judge"
    assert result.metric_name == "llm_judge.ragas.answer_relevancy"
    assert result.score is not None
    # Since cosine similarity of identical vectors is 1.0, score should be ~1.0
    assert result.score > 0.0

