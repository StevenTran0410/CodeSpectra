"""LLMClient seam — the one interface every LLM-calling component depends on."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable


@dataclass
class LLMMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    token_source: Literal["measured", "estimated"] = "measured"


@runtime_checkable
class LLMClient(Protocol):
    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 512,
        temperature: float | None = 0.2,
        json_mode: bool = False,
        reasoning_effort: str | None = None,
    ) -> LLMResponse: ...


class RateLimitExceeded(Exception):
    """Raised once CodeSpectraProxyClient exhausts retries on a 429 (sustained, not transient)."""
    def __init__(
        self,
        provider_id: str,
        model_id: str | None,
        reasoning_effort: str | None = None,
        thinking_budget: int | None = None,
    ) -> None:
        super().__init__(f"Rate limit exceeded for provider={provider_id} model={model_id}")
        self.provider_id = provider_id
        self.model_id = model_id
        self.reasoning_effort = reasoning_effort
        self.thinking_budget = thinking_budget
