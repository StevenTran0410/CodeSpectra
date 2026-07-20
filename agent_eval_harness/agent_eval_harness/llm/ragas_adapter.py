"""Shared Ragas / Langchain LLM adapter — wraps AEH's LLMClient as a BaseChatModel."""
from __future__ import annotations

from agent_eval_harness.llm.client import LLMClient, LLMMessage


def run_async_blocking(coro_fn, *args, **kwargs):
    """Run an async function from a blocking context."""
    import asyncio

    # Probe for a running loop BEFORE building the coroutine — asyncio.run() first would leave an orphaned, un-awaited coroutine on RuntimeError.
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_fn(*args, **kwargs))

    import nest_asyncio

    nest_asyncio.apply()
    return loop.run_until_complete(coro_fn(*args, **kwargs))


def stub_missing_langchain_community_vertexai() -> None:
    """Ragas's internal imports (ragas.llms.base) reference an optional
    langchain_community.chat_models.vertexai module; stub it out if not installed."""
    import sys
    import types

    if "langchain_community.chat_models.vertexai" not in sys.modules:
        m = types.ModuleType("langchain_community.chat_models.vertexai")
        m.ChatVertexAI = object  # type: ignore[attr-defined]
        sys.modules["langchain_community.chat_models.vertexai"] = m


def make_ragas_llm_adapter(llm_client: LLMClient):
    """Returns a Langchain BaseChatModel instance wrapping the given LLMClient."""
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
            return run_async_blocking(self._agenerate, messages, stop, run_manager, **kwargs)

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


def make_ragas_embeddings(embedding_client):
    """Returns a Langchain Embeddings implementation that delegates to the given EmbeddingClient — no dummy vectors, callers must supply a real client."""
    try:
        from langchain_core.embeddings import Embeddings
    except ImportError as exc:
        raise ImportError(
            "langchain_core is not installed. Install ragas or the [datasets] optional dependency."
        ) from exc

    class RealEmbeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return run_async_blocking(embedding_client.embed_texts, texts)

        def embed_query(self, text: str) -> list[float]:
            results = run_async_blocking(embedding_client.embed_texts, [text])
            return results[0]

    return RealEmbeddings()

