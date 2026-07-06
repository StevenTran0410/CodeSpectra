"""Deterministic canned LLM client for offline tests."""
from __future__ import annotations

from agent_eval_harness.llm.client import LLMMessage, LLMResponse


class FakeLLMClient:
    def __init__(self, responses: LLMResponse | list[LLMResponse]) -> None:
        self._queue: list[LLMResponse] | None = (
            list(responses) if isinstance(responses, list) else None
        )
        self._single: LLMResponse | None = None if isinstance(responses, list) else responses
        self.calls: list[list[LLMMessage]] = []

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 512,
        temperature: float | None = 0.2,
        json_mode: bool = False,
    ) -> LLMResponse:
        self.calls.append(messages)
        if self._queue:
            return self._queue.pop(0) if len(self._queue) > 1 else self._queue[0]
        assert self._single is not None
        return self._single
