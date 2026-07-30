"""HarnessChatGenerator — a Haystack @component wrapping an injected LLMClient."""
from __future__ import annotations

from haystack import component
from haystack.dataclasses import ChatMessage

from agent_eval_harness.llm.client import LLMClient, LLMMessage


@component
class HarnessChatGenerator:
    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: str = "",
        max_tokens: int = 512,
        temperature: float | None = 0.2,
        json_mode: bool = False,
    ) -> None:
        self._llm_client = llm_client
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._json_mode = json_mode

    __haystack_supports_async__ = True

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage]) -> dict:
        raise NotImplementedError("HarnessChatGenerator is async-only — use run_async().")

    @component.output_types(replies=list[ChatMessage])
    async def run_async(self, messages: list[ChatMessage]) -> dict:
        llm_messages = (
            [LLMMessage(role="system", content=self._system_prompt)]
            if self._system_prompt
            else []
        ) + [LLMMessage(role=m.role.value, content=m.text or "") for m in messages]

        response = await self._llm_client.complete(
            llm_messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            json_mode=self._json_mode,
        )

        reply = ChatMessage.from_assistant(
            response.content,
            meta={
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "total_tokens": (
                        (response.prompt_tokens or 0) + (response.completion_tokens or 0)
                    ),
                },
                "aeh_token_source": response.token_source,
            },
        )
        return {"replies": [reply]}
