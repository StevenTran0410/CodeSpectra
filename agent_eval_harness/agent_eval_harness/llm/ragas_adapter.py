"""Shared Ragas / Langchain LLM adapter — wraps AEH's LLMClient as a BaseChatModel.

Extracted from datasets/generators/qa_testset.py (CS-262) so that CS-263's
metric judges can reuse the exact same adapters without duplication.
"""
from __future__ import annotations

from agent_eval_harness.llm.client import LLMClient, LLMMessage


def stub_missing_langchain_community_vertexai() -> None:
    """Ragas's internal imports (ragas.llms.base) reference an optional
    langchain_community submodule (chat_models.vertexai) that doesn't exist in
    every installed langchain_community version — pre-register a no-op stub so
    `import ragas` (and anything importing from it) succeeds regardless.

    MUST be called before the FIRST `import ragas` / `from ragas import ...`
    anywhere in the process — including in callers of this module (e.g.
    agent_eval_harness/metrics/judges/ragas_judge.py), since importing THIS
    module alone does not run this function (it must be called explicitly).
    """
    import sys
    import types

    if "langchain_community.chat_models.vertexai" not in sys.modules:
        m = types.ModuleType("langchain_community.chat_models.vertexai")
        m.ChatVertexAI = object  # type: ignore[attr-defined]
        sys.modules["langchain_community.chat_models.vertexai"] = m


def make_ragas_llm_adapter(llm_client: LLMClient):
    """Returns a Langchain BaseChatModel instance wrapping the given LLMClient.

    Import is deferred so this module can be imported without langchain_core installed.
    """
    try:
        stub_missing_langchain_community_vertexai()

        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.messages import BaseMessage, ChatMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
    except ImportError as exc:
        raise ImportError(
            "langchain_core is not installed. Install ragas or the [datasets] optional dependency."
        ) from exc

    class LangchainLLMAdapter(BaseChatModel):
        client: LLMClient
        model_id: str = "codespectra-proxy"

        # Track last response for cost_tokens bookkeeping
        _last_prompt_tokens: int | None = None
        _last_completion_tokens: int | None = None

        def _generate(
            self,
            messages: list[BaseMessage],
            stop=None,
            run_manager=None,
            **kwargs,
        ) -> ChatResult:
            import asyncio

            try:
                return asyncio.run(self._agenerate(messages, stop, run_manager, **kwargs))
            except RuntimeError:
                import nest_asyncio

                nest_asyncio.apply()
                return asyncio.get_event_loop().run_until_complete(
                    self._agenerate(messages, stop, run_manager, **kwargs)
                )

        async def _agenerate(
            self,
            messages: list[BaseMessage],
            stop=None,
            run_manager=None,
            **kwargs,
        ) -> ChatResult:
            formatted = []
            for m in messages:
                role = "user"
                if m.type == "system":
                    role = "system"
                elif m.type == "assistant":
                    role = "assistant"
                formatted.append(LLMMessage(role=role, content=m.content))

            resp = await self.client.complete(formatted)
            object.__setattr__(self, "_last_prompt_tokens", resp.prompt_tokens)
            object.__setattr__(self, "_last_completion_tokens", resp.completion_tokens)
            gen = ChatGeneration(message=ChatMessage(role="assistant", content=resp.content))
            return ChatResult(generations=[gen])

        @property
        def _llm_type(self) -> str:
            return "codespectra-proxy-llm"

    return LangchainLLMAdapter(client=llm_client)


def make_ragas_embeddings():
    """Returns a Langchain Embeddings that returns constant 1536-dim vectors."""
    try:
        from langchain_core.embeddings import Embeddings
    except ImportError as exc:
        raise ImportError(
            "langchain_core is not installed. Install ragas or the [datasets] optional dependency."
        ) from exc

    class DummyEmbeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[1.0] + [0.0] * 1535 for _ in texts]

        def embed_query(self, text: str) -> list[float]:
            return [1.0] + [0.0] * 1535

    return DummyEmbeddings()
