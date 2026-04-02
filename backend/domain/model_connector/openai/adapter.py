"""OpenAI provider adapter."""
import httpx

from domain.model_connector._cloud_base import CloudAdapterBase
from domain.model_connector.errors import ProviderError
from domain.model_connector.types import ChatRequest, ChatResponse, ProviderConfig
from shared.logger import logger

# Chat-capable models returned by /v1/models that we surface to the user
_CHAT_MODEL_PREFIXES = ("gpt-", "o1", "o3", "chatgpt-")

MODEL_PRESETS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "o3-mini",
]


class OpenAIAdapter(CloudAdapterBase):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config, base_url="https://api.openai.com")

    def _auth(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._require_api_key()}"}

    async def list_models(self) -> list[str]:
        try:
            res = await self._client.get("/v1/models", headers=self._auth())
            res.raise_for_status()
            data = res.json()
            models = [
                m["id"]
                for m in data.get("data", [])
                if any(m["id"].startswith(p) for p in _CHAT_MODEL_PREFIXES)
            ]
            return sorted(models) or MODEL_PRESETS
        except httpx.ConnectError as e:
            raise self._map_connect_error(e) from e
        except httpx.TimeoutException as e:
            raise self._map_timeout(e) from e
        except httpx.HTTPStatusError as e:
            raise self._map_http_error(e) from e
        except Exception as e:
            raise ProviderError(self._code_unknown(), str(e), provider_id=self.config.id) from e

    async def chat(self, request: ChatRequest) -> ChatResponse:
        payload = {
            "model": self.config.model_id,
            "messages": [m.model_dump() for m in request.messages],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        try:
            logger.debug(f"OpenAI chat: model={self.config.model_id}")
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
            raise ProviderError(self._code_unknown(), str(e), provider_id=self.config.id) from e

    async def test_connection(self) -> tuple[bool, str, str | None]:
        if not self._api_key:
            return False, "No API key configured", None
        try:
            models = await self.list_models()
            return True, f"Connected — {len(models)} model(s) available", None
        except ProviderError as e:
            return False, e.message, None

    @staticmethod
    def _code_unknown():
        from domain.model_connector.errors import ProviderErrorCode
        return ProviderErrorCode.UNKNOWN
