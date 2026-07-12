"""Adapts AEH's LLMClient into the ProviderConfigService duck type CodeSpectra's agents call directly, so the injection runner can exercise a real LLM call without AEH ever holding a provider API key itself."""
from __future__ import annotations

from agent_eval_harness.injection.backend_bridge import ensure_backend_importable
from agent_eval_harness.llm.client import LLMClient, LLMMessage

ensure_backend_importable()

from domain.model_connector.types import ChatRequest, ChatResponse  # noqa: E402


class ProxyProviderService:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    async def chat(self, request: ChatRequest) -> ChatResponse:
        messages = [LLMMessage(role=m.role, content=m.content) for m in request.messages]
        resp = await self._llm_client.complete(
            messages,
            max_tokens=request.max_completion_tokens,
            temperature=request.temperature,
            json_mode=request.json_mode,
            reasoning_effort=request.reasoning_effort,
        )
        return ChatResponse(
            provider_id=request.provider_id,
            model_id=request.model_id or resp.model,
            content=resp.content,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
        )
