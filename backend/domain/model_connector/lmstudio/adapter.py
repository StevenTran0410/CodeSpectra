"""LM Studio provider adapter — uses the OpenAI-compatible /v1/chat/completions endpoint."""
import httpx

from domain.model_connector.errors import ProviderError, ProviderErrorCode
from domain.model_connector.types import ChatRequest, ChatResponse, ProviderConfig
from shared.logger import logger


class LMStudioAdapter:
    """Wrapper over LM Studio's OpenAI-compatible REST API."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0),
        )

    async def list_models(self) -> list[str]:
        try:
            res = await self._client.get("/v1/models")
            res.raise_for_status()
            data = res.json()
            return [m["id"] for m in data.get("data", [])]
        except httpx.ConnectError as e:
            raise ProviderError(
                ProviderErrorCode.CONNECTION_REFUSED,
                f"Cannot reach LM Studio at {self.config.base_url}: {e}",
                provider_id=self.config.id,
                retryable=True,
            ) from e

    async def chat(self, request: ChatRequest) -> ChatResponse:
        payload = {
            "model": self.config.model_id,
            "messages": [m.model_dump() for m in request.messages],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        try:
            logger.debug(f"LM Studio chat: model={self.config.model_id}")
            res = await self._client.post("/v1/chat/completions", json=payload)
            res.raise_for_status()
            data = res.json()
            choice = data["choices"][0]
            usage = data.get("usage", {})
            return ChatResponse(
                provider_id=self.config.id,
                model_id=self.config.model_id,
                content=choice["message"]["content"],
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
            )
        except httpx.ConnectError as e:
            raise ProviderError(
                ProviderErrorCode.CONNECTION_REFUSED,
                f"Cannot reach LM Studio at {self.config.base_url}",
                provider_id=self.config.id,
                retryable=True,
            ) from e
        except Exception as e:
            raise ProviderError(
                ProviderErrorCode.UNKNOWN, str(e), provider_id=self.config.id
            ) from e

    async def test_connection(self) -> tuple[bool, str]:
        try:
            models = await self.list_models()
            return True, f"Connected — {len(models)} model(s) loaded"
        except ProviderError as e:
            return False, e.message

    async def aclose(self) -> None:
        await self._client.aclose()
