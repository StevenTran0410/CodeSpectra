"""CodeSpectraProxyClient — the ONLY way AEH ever calls a real LLM.

Never touches a provider API key. Forwards to CodeSpectra's backend, which
dispatches through domain/model_connector using whatever provider the user
already configured there. AEH has no provider/key config of its own.
"""
from __future__ import annotations

import httpx

from agent_eval_harness.llm.client import LLMMessage, LLMResponse


class CodeSpectraProxyClient:
    def __init__(
        self,
        base_url: str,
        bearer_token: str,
        provider_id: str,
        model_id: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = bearer_token
        self._provider_id = provider_id
        self._model_id = model_id
        self._http = http_client or httpx.AsyncClient(timeout=60.0)

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 512,
        temperature: float | None = 0.2,
        json_mode: bool = False,
    ) -> LLMResponse:
        resp = await self._http.post(
            f"{self._base_url}/api/external/llm/complete",
            headers={"Authorization": f"Bearer {self._token}"},
            json={
                "provider_id": self._provider_id,
                "model_id": self._model_id,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
                "max_completion_tokens": max_tokens,
                "temperature": temperature,
                "json_mode": json_mode,
            },
        )
        resp.raise_for_status()
        body = resp.json()
        has_usage = body.get("prompt_tokens") is not None
        return LLMResponse(
            content=body["content"],
            model=body["model_id"],
            prompt_tokens=body.get("prompt_tokens"),
            completion_tokens=body.get("completion_tokens"),
            token_source="measured" if has_usage else "estimated",
        )

    async def aclose(self) -> None:
        await self._http.aclose()
