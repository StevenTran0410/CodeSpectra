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
    ) -> LLMResponse: ...
