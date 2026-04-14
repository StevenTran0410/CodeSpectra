"""Shared mixin for LLM adapter base classes."""
from __future__ import annotations

from collections.abc import AsyncGenerator

from .types import ChatRequest


class StreamFallbackMixin:
    """Provides a default chat_stream() that wraps the blocking chat() call.

    Adapters that support native token streaming (e.g. OpenAI) should override
    chat_stream() directly. All others get this safe fallback automatically.
    """

    async def chat_stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """Yield the full response as a single chunk — no native streaming required."""
        from .types import ChatResponse  # avoid circular at module level
        response: ChatResponse = await self.chat(request)  # type: ignore[attr-defined]
        content = response.content or ""
        if content:
            yield content
