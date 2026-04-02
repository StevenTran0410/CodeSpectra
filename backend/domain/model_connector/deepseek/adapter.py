"""DeepSeek provider adapter — OpenAI-compatible API with DeepSeek base URL."""
import httpx

from domain.model_connector._cloud_base import CloudAdapterBase
from domain.model_connector.errors import ProviderError, ProviderErrorCode
from domain.model_connector.types import ChatRequest, ChatResponse, ProviderConfig
from shared.logger import logger

MODEL_PRESETS = [
    "deepseek-chat",
    "deepseek-reasoner",
]


class DeepSeekAdapter(CloudAdapterBase):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config, base_url="https://api.deepseek.com")

    def _auth(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._require_api_key()}"}

    async def list_models(self) -> list[str]:
        try:
            res = await self._client.get("/v1/models", headers=self._auth())
            res.raise_for_status()
            data = res.json()
            ids = [m["id"] for m in data.get("data", [])]
            return ids or MODEL_PRESETS
        except httpx.ConnectError as e:
            raise self._map_connect_error(e) from e
        except httpx.TimeoutException as e:
            raise self._map_timeout(e) from e
        except httpx.HTTPStatusError as e:
            raise self._map_http_error(e) from e
        except Exception as e:
            raise ProviderError(ProviderErrorCode.UNKNOWN, str(e), provider_id=self.config.id) from e

    async def chat(self, request: ChatRequest) -> ChatResponse:
        payload = {
            "model": self.config.model_id,
            "messages": [m.model_dump() for m in request.messages],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        try:
            logger.debug(f"DeepSeek chat: model={self.config.model_id}")
            res = await self._client.post("/v1/chat/completions", json=payload, headers=self._auth())
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
            raise self._map_connect_error(e) from e
        except httpx.TimeoutException as e:
            raise self._map_timeout(e) from e
        except httpx.HTTPStatusError as e:
            raise self._map_http_error(e) from e
        except Exception as e:
            raise ProviderError(ProviderErrorCode.UNKNOWN, str(e), provider_id=self.config.id) from e

    async def test_connection(self) -> tuple[bool, str, str | None]:
        if not self._api_key:
            return False, "No API key configured", None
        try:
            models = await self.list_models()
            return True, f"Connected — {len(models)} model(s) available", None
        except ProviderError as e:
            return False, e.message, None
