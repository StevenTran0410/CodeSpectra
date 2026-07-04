"""Explicit test for the load-bearing correctness property (CS-263 §6.2, §11).

Asserts that the RAGAS faithfulness call's `retrieved_contexts` argument equals
the writer span's `input_json`, NOT the qa_testset gold context NOR the full corpus.

This is literally what CS-260 calls "the point of AEH" (per-component grounding).
"""
from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from agent_eval_harness.llm.ragas_adapter import stub_missing_langchain_community_vertexai

# Must run before any `pytest.importorskip("ragas")`/`import ragas` below — see
# stub_missing_langchain_community_vertexai's own docstring for why.
stub_missing_langchain_community_vertexai()

pytestmark = pytest.mark.asyncio


def _writer_span(context: list[str]) -> dict:
    return {
        "id": "span-writer",
        "component_id": "writer",
        "span_type": "llm",
        "input_json": json.dumps({"context": context, "query": "test"}),
        "output_json": json.dumps({"answer": "The answer is 42."}),
        "parent_span_id": None,
        "started_at": datetime.now(UTC).isoformat(),
        "details_json": "{}",
    }


WRITER_RECEIVED_CONTEXT = ["Only this context was passed to the writer component."]
GOLD_CONTEXT = ["This is the gold standard context from qa_testset — NOT what writer received."]
CORPUS_CONTEXT = ["This is the raw corpus — definitely NOT what writer received."]


async def test_faithfulness_uses_writer_span_input_not_gold() -> None:
    """retrieved_contexts must equal the writer span's input_json.context, not gold."""
    pytest.importorskip("ragas")

    from agent_eval_harness.metrics.judges.ragas_judge import (
        _extract_writer_contexts,
    )

    spans = [_writer_span(WRITER_RECEIVED_CONTEXT)]

    # Verify the extraction function returns writer's context, not gold or corpus
    extracted = _extract_writer_contexts(spans, "writer")
    assert extracted == WRITER_RECEIVED_CONTEXT, (
        f"expected writer's received context {WRITER_RECEIVED_CONTEXT!r}, "
        f"got {extracted!r}"
    )
    assert extracted != GOLD_CONTEXT, "MUST NOT use gold context"
    assert extracted != CORPUS_CONTEXT, "MUST NOT use full corpus"


async def test_faithfulness_different_writer_context_produces_different_extraction() -> None:
    """Changing the writer span's input changes the extracted contexts."""
    pytest.importorskip("ragas")

    from agent_eval_harness.metrics.judges.ragas_judge import _extract_writer_contexts

    context_a = ["Context version A — unique string XYZ123"]
    context_b = ["Context version B — unique string ABC456"]

    spans_a = [_writer_span(context_a)]
    spans_b = [_writer_span(context_b)]

    extracted_a = _extract_writer_contexts(spans_a, "writer")
    extracted_b = _extract_writer_contexts(spans_b, "writer")
    assert extracted_a == context_a
    assert extracted_b == context_b
    assert extracted_a != extracted_b


async def test_faithfulness_empty_when_no_writer_span() -> None:
    """If no writer span exists, returns empty list (graceful degradation)."""
    from agent_eval_harness.metrics.judges.ragas_judge import _extract_writer_contexts

    spans = [
        {
            "id": "span-planner",
            "component_id": "planner",
            "span_type": "llm",
            "input_json": json.dumps({"query": "test"}),
            "output_json": json.dumps({"intents": ["a"]}),
            "parent_span_id": None,
            "started_at": datetime.now(UTC).isoformat(),
            "details_json": "{}",
        }
    ]
    result = _extract_writer_contexts(spans, "writer")
    assert result == []


async def test_faithfulness_result_details_declare_source() -> None:
    """The MetricResult.details must explicitly declare retrieved_contexts_source."""
    pytest.importorskip("ragas")

    from agent_eval_harness.llm.client import LLMResponse
    from agent_eval_harness.llm.fake_client import FakeLLMClient
    from agent_eval_harness.metrics.judges.ragas_judge import run_ragas_faithfulness

    llm_client = FakeLLMClient(LLMResponse(content="faithful answer", model="fake"))
    spans = [_writer_span(WRITER_RECEIVED_CONTEXT)]

    result = await run_ragas_faithfulness(
        spans=spans,
        query="What is the answer?",
        actual_answer="The answer is 42.",
        llm_client=llm_client,
        writer_component_id="writer",
        component_id="writer",
    )

    assert result.details["retrieved_contexts_source"] == "writer_span_input_json"
    assert result.details["writer_component_id"] == "writer"
    assert result.details["context_count"] == len(WRITER_RECEIVED_CONTEXT)
