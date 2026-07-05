"""RAGAS metric judges (CS-263 §6.2).

Metrics:
- context_precision, context_recall: scored against qa_testset golds.
- faithfulness: THE load-bearing correctness property — retrieved_contexts is built
  from the WRITER span's own input_json (extracted from get_spans_for_trace), never
  from the qa_testset gold context or the full corpus. This is the per-component
  grounding principle described in CS-260 as "the point of AEH."
- answer_relevancy: scores the writer's answer relevance to the original query.

Cost tracking: tokens from LangchainLLMAdapter are accumulated into cost_tokens.
"""
from __future__ import annotations

import json

from agent_eval_harness.llm.client import LLMClient
from agent_eval_harness.metrics.types import MetricResult, extract_cost_tokens


def _extract_writer_contexts(spans: list[dict], writer_component_id: str = "writer") -> list[str]:
    """Extract the context list from the writer span's input_json.

    This is the load-bearing correctness property (CS-263 §6.2): the writer's
    RECEIVED contexts are what matter for faithfulness, not the gold or corpus.
    Returns the list of context strings or an empty list if unavailable.
    """
    for span in spans:
        if span.get("component_id") != writer_component_id:
            continue
        raw = span.get("input_json") or "{}"
        try:
            data = json.loads(raw)
            # The writer's input may have "context" as a list or a single string
            ctx = data.get("context")
            if isinstance(ctx, list):
                return [str(c) for c in ctx]
            if isinstance(ctx, str) and ctx:
                return [ctx]
        except (json.JSONDecodeError, AttributeError):
            pass
    return []


async def run_ragas_faithfulness(
    spans: list[dict],
    query: str,
    actual_answer: str,
    llm_client: LLMClient,
    *,
    writer_component_id: str = "writer",
    component_id: str | None = None,
    trace_id: str | None = None,
) -> MetricResult:
    """Faithfulness metric using RAGAS.

    retrieved_contexts is built from the writer span's input_json — never from
    the qa_testset gold context or the full corpus. This is explicitly tested in
    test_writer_faithfulness_uses_received_payload.py.
    """
    try:
        from agent_eval_harness.llm.ragas_adapter import stub_missing_langchain_community_vertexai

        stub_missing_langchain_community_vertexai()

        from ragas import SingleTurnSample
        from ragas.metrics import Faithfulness
    except ImportError as exc:
        raise ImportError(
            "ragas is not installed. Install it via `pip install ragas` "
            "or the [datasets] optional dependency."
        ) from exc

    from agent_eval_harness.llm.ragas_adapter import make_ragas_llm_adapter

    retrieved_contexts = _extract_writer_contexts(spans, writer_component_id)

    llm_adapter = make_ragas_llm_adapter(llm_client)  # Faithfulness needs no embeddings

    sample = SingleTurnSample(
        user_input=query,
        response=actual_answer,
        retrieved_contexts=retrieved_contexts if retrieved_contexts else ["<no context>"],
    )

    metric = Faithfulness(llm=llm_adapter)  # type: ignore[arg-type]
    try:
        score = await metric.single_turn_ascore(sample)
    except Exception:  # noqa: BLE001
        # Fallback for versions that use a different API
        score = None

    return MetricResult(
        metric_name="llm_judge.ragas.faithfulness",
        metric_class="llm_judge",
        score=score,
        passed=score is not None and score >= 0.5,
        details={
            "retrieved_contexts_source": "writer_span_input_json",
            "writer_component_id": writer_component_id,
            "context_count": len(retrieved_contexts),
        },
        component_id=component_id,
        trace_id=trace_id,
        evaluator="ragas.Faithfulness",
        cost_tokens=extract_cost_tokens(llm_adapter),
    )


async def run_ragas_answer_relevancy(
    query: str,
    actual_answer: str,
    llm_client: LLMClient,
    *,
    retrieved_contexts: list[str] | None = None,
    component_id: str | None = None,
    trace_id: str | None = None,
) -> MetricResult:
    """Answer Relevancy metric using RAGAS."""
    try:
        from agent_eval_harness.llm.ragas_adapter import stub_missing_langchain_community_vertexai

        stub_missing_langchain_community_vertexai()

        from ragas import SingleTurnSample
        from ragas.metrics import AnswerRelevancy
    except ImportError as exc:
        raise ImportError(
            "ragas is not installed. Install it via `pip install ragas` "
            "or the [datasets] optional dependency."
        ) from exc

    from agent_eval_harness.llm.ragas_adapter import make_ragas_embeddings, make_ragas_llm_adapter

    llm_adapter = make_ragas_llm_adapter(llm_client)
    embeddings = make_ragas_embeddings()

    sample = SingleTurnSample(
        user_input=query,
        response=actual_answer,
        retrieved_contexts=retrieved_contexts or ["<no context>"],
    )

    metric = AnswerRelevancy(llm=llm_adapter, embeddings=embeddings)  # type: ignore[arg-type]
    try:
        score = await metric.single_turn_ascore(sample)
    except Exception:  # noqa: BLE001
        score = None

    return MetricResult(
        metric_name="llm_judge.ragas.answer_relevancy",
        metric_class="llm_judge",
        score=score,
        passed=score is not None and score >= 0.5,
        details={},
        component_id=component_id,
        trace_id=trace_id,
        evaluator="ragas.AnswerRelevancy",
        cost_tokens=extract_cost_tokens(llm_adapter),
    )


async def run_ragas_context_precision(
    query: str,
    gold_contexts: list[str],
    retrieved_contexts: list[str],
    llm_client: LLMClient,
    *,
    component_id: str | None = None,
    trace_id: str | None = None,
) -> MetricResult:
    """Context Precision metric using RAGAS."""
    try:
        from agent_eval_harness.llm.ragas_adapter import stub_missing_langchain_community_vertexai

        stub_missing_langchain_community_vertexai()

        from ragas import SingleTurnSample
        from ragas.metrics import ContextPrecision
    except ImportError as exc:
        raise ImportError(
            "ragas is not installed. Install it via `pip install ragas` "
            "or the [datasets] optional dependency."
        ) from exc

    from agent_eval_harness.llm.ragas_adapter import make_ragas_llm_adapter

    llm_adapter = make_ragas_llm_adapter(llm_client)

    sample = SingleTurnSample(
        user_input=query,
        retrieved_contexts=retrieved_contexts or ["<no context>"],
        reference_contexts=gold_contexts,
    )

    metric = ContextPrecision(llm=llm_adapter)  # type: ignore[arg-type]
    try:
        score = await metric.single_turn_ascore(sample)
    except Exception:  # noqa: BLE001
        score = None

    return MetricResult(
        metric_name="llm_judge.ragas.context_precision",
        metric_class="llm_judge",
        score=score,
        passed=score is not None and score >= 0.5,
        details={},
        component_id=component_id,
        trace_id=trace_id,
        evaluator="ragas.ContextPrecision",
        cost_tokens=extract_cost_tokens(llm_adapter),
    )
