"""RAGAS metric judges."""
from __future__ import annotations

import json
import logging

from agent_eval_harness.llm.client import LLMClient
from agent_eval_harness.llm.embedding_client import EmbeddingClient
from agent_eval_harness.metrics.scoring_helpers import JUDGE_PASS_THRESHOLD
from agent_eval_harness.metrics.types import MetricResult, extract_cost_tokens

logger = logging.getLogger("agent_eval_harness.metrics.judges.ragas_judge")

# Sentinel context ragas requires when a span genuinely has no retrieved context to grade.
NO_CONTEXT_SENTINEL = "<no context>"


def _extract_writer_contexts(spans: list[dict], writer_component_id: str = "writer") -> list[str]:
    """Extract the context list from the writer span's input_json."""
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
        except (json.JSONDecodeError, AttributeError) as exc:
            logger.debug("could not parse writer context from span input_json: %s", exc)
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
    """Faithfulness metric using RAGAS."""
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
        retrieved_contexts=retrieved_contexts if retrieved_contexts else [NO_CONTEXT_SENTINEL],
    )

    metric = Faithfulness(llm=llm_adapter)  # type: ignore[arg-type]
    try:
        score = await metric.single_turn_ascore(sample)
    except Exception as exc:  # noqa: BLE001 — ragas API varies across versions; degrade to score=None
        logger.warning("ragas.Faithfulness scoring failed, degrading to score=None: %s", exc)
        score = None

    return MetricResult(
        metric_name="llm_judge.ragas.faithfulness",
        metric_class="llm_judge",
        score=score,
        passed=score is not None and score >= JUDGE_PASS_THRESHOLD,
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
    embedding_client: EmbeddingClient | None = None,
) -> MetricResult:
    try:
        from agent_eval_harness.llm.ragas_adapter import stub_missing_langchain_community_vertexai

        stub_missing_langchain_community_vertexai()

        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import AnswerRelevancy
    except ImportError as exc:
        raise ImportError(
            "ragas is not installed. Install it via `pip install ragas` "
            "or the [datasets] optional dependency."
        ) from exc

    from agent_eval_harness.llm.ragas_adapter import make_ragas_embeddings, make_ragas_llm_adapter

    llm_adapter = make_ragas_llm_adapter(llm_client)

    if embedding_client is None:
        return MetricResult(
            metric_name="llm_judge.ragas.answer_relevancy",
            metric_class="llm_judge",
            score=None,
            passed=False,
            details={
                "error": "no embedding_client configured for ragas.answer_relevancy — "
                "this metric needs an embedding provider, not yet wired into "
                "the evaluation-run flow"
            },
            component_id=component_id,
            trace_id=trace_id,
            evaluator="ragas.AnswerRelevancy",
            cost_tokens=0,
        )

    embeddings = make_ragas_embeddings(embedding_client)

    sample = SingleTurnSample(
        user_input=query,
        response=actual_answer,
        retrieved_contexts=retrieved_contexts or [NO_CONTEXT_SENTINEL],
    )

    metric = AnswerRelevancy(llm=llm_adapter, embeddings=embeddings)  # type: ignore[arg-type]
    try:
        score = await metric.single_turn_ascore(sample)
    except Exception as exc:  # noqa: BLE001 — ragas API varies across versions; degrade to score=None
        logger.warning("ragas.AnswerRelevancy scoring failed, degrading to score=None: %s", exc)
        score = None

    return MetricResult(
        metric_name="llm_judge.ragas.answer_relevancy",
        metric_class="llm_judge",
        score=score,
        passed=score is not None and score >= JUDGE_PASS_THRESHOLD,
        details={},
        component_id=component_id,
        trace_id=trace_id,
        evaluator="ragas.AnswerRelevancy",
        cost_tokens=extract_cost_tokens(llm_adapter),
    )
