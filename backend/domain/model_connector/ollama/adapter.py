"""Ollama provider adapter — implements LLM calls against a local Ollama server."""
from typing import AsyncGenerator

import httpx

from domain.model_connector.errors import ProviderError, ProviderErrorCode
from domain.model_connector.types import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ProviderConfig,
)
from shared.logger import logger


class OllamaAdapter:
    """Thin async wrapper over the Ollama /api/chat endpoint."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0),
        )

    async def list_models(self) -> list[str]:
        try:
            res = await self._client.get("/api/tags")
            res.raise_for_status()
            data = res.json()
            return [m["name"] for m in data.get("models", [])]
        except httpx.ConnectError as e:
            raise ProviderError(
                ProviderErrorCode.CONNECTION_REFUSED,
                f"Cannot reach Ollama at {self.config.base_url}. Make sure Ollama is running.",
                provider_id=self.config.id,
                retryable=True,
            ) from e
        except httpx.TimeoutException as e:
            raise ProviderError(
                ProviderErrorCode.TIMEOUT,
                f"Ollama at {self.config.base_url} timed out.",
                provider_id=self.config.id,
                retryable=True,
            ) from e
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                ProviderErrorCode.UNKNOWN,
                f"Ollama returned HTTP {e.response.status_code}.",
                provider_id=self.config.id,
            ) from e
        except Exception as e:
            raise ProviderError(
                ProviderErrorCode.UNKNOWN,
                str(e),
                provider_id=self.config.id,
            ) from e

    async def chat(self, request: ChatRequest) -> ChatResponse:
        payload = {
            "model": self.config.model_id,
            "messages": [m.model_dump() for m in request.messages],
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }
        try:
            logger.debug(f"Ollama chat: model={self.config.model_id}")
            res = await self._client.post("/api/chat", json=payload)
            res.raise_for_status()
            data = res.json()
            msg = data.get("message", {})
            usage = data.get("usage", {})
            return ChatResponse(
                provider_id=self.config.id,
                model_id=self.config.model_id,
                content=msg.get("content", ""),
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
            )
        except httpx.ConnectError as e:
            raise ProviderError(
                ProviderErrorCode.CONNECTION_REFUSED,
                f"Cannot reach Ollama at {self.config.base_url}",
                provider_id=self.config.id,
                retryable=True,
            ) from e
        except Exception as e:
            raise ProviderError(
                ProviderErrorCode.UNKNOWN, str(e), provider_id=self.config.id
            ) from e

    async def test_connection(self) -> tuple[bool, str, str | None]:
        """Returns (ok, message, warning). Warning is non-None when connected but no models pulled."""
        try:
            models = await self.list_models()
            if not models:
                return (
                    True,
                    "Connected — no models pulled yet",
                    "No models found. Run `ollama pull <model>` to download one.",
                )
            return True, f"Connected — {len(models)} model(s) available", None
        except ProviderError as e:
            return False, e.message, None

    async def aclose(self) -> None:
        await self._client.aclose()
