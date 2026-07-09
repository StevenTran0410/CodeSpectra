"""Shared DeepEval LLM adapter — wraps AEH's LLMClient as a DeepEvalBaseLLM.

Schema-constrained calls (both DeepEval's Synthesizer and its GEval judge request
structured output via `generate(prompt, schema=SomeSchema)`) are answered with a
REAL LLM call: the schema's JSON Schema is appended to the prompt, json_mode is
enabled, and the response is parsed and validated into an actual `schema`
instance. This is required by both call sites — deepeval/metrics/utils.py's
schema-or-json-string fallback would tolerate a raw string, but
deepeval/synthesizer/synthesizer.py's `_a_generate_schema` has no such fallback
and uses the return value as a schema instance directly (e.g. `res.data`,
`res.clarity`) — returning anything else breaks it.
"""
from __future__ import annotations

import json
import typing

from pydantic import ValidationError

from agent_eval_harness.llm.client import LLMClient, LLMMessage
from agent_eval_harness.llm.ragas_adapter import _run_async_blocking

_MAX_SCHEMA_ATTEMPTS = 2


def make_deepeval_llm_adapter(llm_client: LLMClient):
    """Returns a DeepEvalBaseLLM instance wrapping the given LLMClient."""
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
            return _run_async_blocking(self.a_generate, prompt, schema)

        async def a_generate(self, prompt: str, schema: typing.Any = None) -> typing.Any:
            if schema is not None:
                return await self._a_generate_schema(prompt, schema)
            resp = await self.client.complete([LLMMessage(role="user", content=prompt)])
            self._last_prompt_tokens = resp.prompt_tokens
            self._last_completion_tokens = resp.completion_tokens
            return resp.content

        async def _a_generate_schema(self, prompt: str, schema: typing.Any) -> typing.Any:
            json_schema = schema.model_json_schema()
            instruction = (
                f"{prompt}\n\nRespond with ONLY a single JSON object (no markdown "
                f"fences, no commentary) that validates against this JSON schema:\n"
                f"{json.dumps(json_schema)}"
            )
            last_error: Exception | None = None
            for attempt in range(_MAX_SCHEMA_ATTEMPTS):
                if attempt > 0:
                    instruction += (
                        f"\n\nYour previous response failed: {last_error}. "
                        "Reply with corrected JSON only, matching the schema exactly."
                    )
                resp = await self.client.complete(
                    [LLMMessage(role="user", content=instruction)], json_mode=True
                )
                self._last_prompt_tokens = resp.prompt_tokens
                self._last_completion_tokens = resp.completion_tokens
                try:
                    data = json.loads(resp.content)
                    return schema.model_validate(data)
                except (json.JSONDecodeError, ValidationError) as e:
                    last_error = e
            raise ValueError(
                f"LLM response did not satisfy schema {schema.__name__} after "
                f"{_MAX_SCHEMA_ATTEMPTS} attempts: {last_error}"
            )

        def get_model_name(self) -> str:
            return self.model_name

    return DeepEvalLLMAdapter(llm_client)


def make_deepeval_embedding_model(embedding_client):
    """Returns a DeepEvalBaseEmbeddingModel that delegates to the given EmbeddingClient.

    ``embedding_client`` must implement ``embed_texts(texts: list[str]) ->
    list[list[float]]`` (i.e. satisfy the EmbeddingClient protocol from
    agent_eval_harness.llm.embedding_client).  No dummy/silent fallback —
    callers must always supply a real client.
    """
    try:
        from deepeval.models.base_model import DeepEvalBaseEmbeddingModel
    except ImportError as exc:
        raise ImportError(
            "deepeval is not installed. Install it via `pip install deepeval` "
            "or the [datasets] optional dependency."
        ) from exc

    class RealDeepEvalEmbeddingModel(DeepEvalBaseEmbeddingModel):
        def load_model(self):
            return self

        def get_model_name(self, *args, **kwargs) -> str:
            return "CodeSpectra Embedding Proxy"

        def embed_text(self, text: str) -> list[float]:
            return _run_async_blocking(self.a_embed_text, text)

        async def a_embed_text(self, text: str) -> list[float]:
            results = await embedding_client.embed_texts([text])
            return results[0]

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return _run_async_blocking(self.a_embed_texts, texts)

        async def a_embed_texts(self, texts: list[str]) -> list[list[float]]:
            return await embedding_client.embed_texts(texts)

    return RealDeepEvalEmbeddingModel()

