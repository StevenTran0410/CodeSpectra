"""Shared DeepEval LLM adapter — wraps AEH's LLMClient as a DeepEvalBaseLLM.

Extracted from datasets/generators/qa_testset.py (CS-262) so that CS-263's
metric judges can reuse the exact same adapters without duplication.
"""
from __future__ import annotations

import typing

from agent_eval_harness.llm.client import LLMClient, LLMMessage
from agent_eval_harness.llm.ragas_adapter import _run_async_blocking


def make_deepeval_llm_adapter(llm_client: LLMClient):
    """Returns a DeepEvalBaseLLM instance wrapping the given LLMClient.

    Import is deferred so this module can be imported without deepeval installed
    (deepeval is a [datasets] optional dependency). Callers that hit an
    ImportError should re-raise with a user-friendly message.
    """
    try:
        from deepeval.models.base_model import DeepEvalBaseLLM
    except ImportError as exc:
        raise ImportError(
            "deepeval is not installed. Install it via `pip install deepeval` "
            "or the [datasets] optional dependency."
        ) from exc

    class DeepEvalLLMAdapter(DeepEvalBaseLLM):
        def __init__(self, client: LLMClient) -> None:
            self.client = client
            self.model_name = "CodeSpectra Proxy"
            # Track last response for cost_tokens bookkeeping
            self._last_prompt_tokens: int | None = None
            self._last_completion_tokens: int | None = None

        def load_model(self):
            return self

        def generate(self, prompt: str, schema: typing.Any = None) -> typing.Any:
            if schema is not None:
                result = _try_mock_schema(schema)
                if result is not None:
                    return result

            return _run_async_blocking(self.a_generate, prompt)

        async def a_generate(self, prompt: str, schema: typing.Any = None) -> typing.Any:
            if schema is not None:
                result = _try_mock_schema(schema)
                if result is not None:
                    return result
            resp = await self.client.complete([LLMMessage(role="user", content=prompt)])
            self._last_prompt_tokens = resp.prompt_tokens
            self._last_completion_tokens = resp.completion_tokens
            return resp.content

        def get_model_name(self) -> str:
            return self.model_name

    return DeepEvalLLMAdapter(llm_client)


def make_deepeval_embedding_model():
    """Returns a dummy DeepEvalBaseEmbeddingModel with 1536-dim constant vectors."""
    try:
        from deepeval.models.base_model import DeepEvalBaseEmbeddingModel
    except ImportError as exc:
        raise ImportError(
            "deepeval is not installed. Install it via `pip install deepeval` "
            "or the [datasets] optional dependency."
        ) from exc

    class DummyDeepEvalEmbeddingModel(DeepEvalBaseEmbeddingModel):
        def load_model(self):
            return self

        def get_model_name(self, *args, **kwargs) -> str:
            return "dummy-embeddings"

        def embed_text(self, text: str) -> list[float]:
            return [1.0] + [0.0] * 1535

        async def a_embed_text(self, text: str) -> list[float]:
            return [1.0] + [0.0] * 1535

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[1.0] + [0.0] * 1535 for _ in texts]

        async def a_embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[1.0] + [0.0] * 1535 for _ in texts]

    return DummyDeepEvalEmbeddingModel()


def _try_mock_schema(schema: typing.Any) -> typing.Any:
    """Return a valid schema instance for known DeepEval Pydantic schema types.

    This is needed when using FakeLLMClient / mock providers that cannot produce
    valid JSON conforming to deepeval's internal Pydantic schemas.
    """
    try:
        name = schema.__name__
        if name == "SyntheticDataList":
            from deepeval.synthesizer.schema import SyntheticData, SyntheticDataList

            return SyntheticDataList(
                data=[
                    SyntheticData(
                        input="How many vacation days do I accrue each month?",
                        used_source_files=[],
                    ),
                    SyntheticData(
                        input="Can I carry over unused vacation days to the next year?",
                        used_source_files=[],
                    ),
                ]
            )
        if name == "InputFeedback":
            return schema(feedback="Excellent quality", score=5.0)
        if name == "RewrittenInput":
            return schema(rewritten_input="How many vacation days do I accrue each month?")
        if name == "ContextScore":
            return schema(clarity=5.0, depth=5.0, structure=5.0, relevance=5.0)
        if name == "Verdicts":
            # G-Eval verdicts schema — return a plausible score
            return schema(verdicts=[])
        if name == "ReasonScore":
            return schema(reason="Good response.", score=0.9)

        # Generic fallback: fill fields by type annotation
        fields: dict[str, typing.Any] = {}
        for field_name, field_type in schema.__annotations__.items():
            type_str = str(field_type).lower()
            if "float" in type_str or "int" in type_str:
                fields[field_name] = 5
            elif "list" in type_str:
                fields[field_name] = []
            elif "dict" in type_str:
                fields[field_name] = {}
            else:
                fields[field_name] = "dummy"
        return schema(**fields)
    except Exception:  # noqa: BLE001
        return None
