"""Deterministic canned LLM client for offline tests.

Pattern matches backend/tests/conftest.py's own chat_response_sequence helper
(queue.pop(0), repeats last response once exhausted) — no existing "real class
implementing the interface" fake precedent to copy, so this introduces that
convention fresh, as CS-261 §6 requires ("stub-able client seam").
"""
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
