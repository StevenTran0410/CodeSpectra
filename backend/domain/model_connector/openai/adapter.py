"""OpenAI provider adapter."""
import httpx

from domain.model_connector._cloud_base import CloudAdapterBase
from domain.model_connector.errors import ProviderError
from domain.model_connector.types import ChatRequest, ChatResponse, ProviderConfig
from shared.logger import logger

# Chat-capable models returned by /v1/models that we surface to the user
_CHAT_MODEL_PREFIXES = ("gpt-", "o1", "o3", "chatgpt-")

OPENAI_MODEL_PRESETS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "o3-mini",
]


class OpenAIAdapter(CloudAdapterBase):
    CHAT_MODEL_PREFIXES: tuple[str, ...] | None = _CHAT_MODEL_PREFIXES
    MODEL_PRESETS: list[str] = OPENAI_MODEL_PRESETS

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
                if (
                    self.CHAT_MODEL_PREFIXES is None
                    or any(m["id"].startswith(p) for p in self.CHAT_MODEL_PREFIXES)
                )
            ]
            return sorted(models) or self.MODEL_PRESETS
        except httpx.ConnectError as e:
            raise self._map_connect_error(e) from e
        except httpx.TimeoutException as e:
            raise self._map_timeout(e) from e
        except httpx.HTTPStatusError as e:
            raise self._map_http_error(e) from e
        except Exception as e:
            raise ProviderError(self._code_unknown(), str(e), provider_id=self.config.id) from e

    # Models that reject any explicit temperature value — must omit it from the payload.
    # GPT-5 series and reasoning models (o1/o3/o4) only accept their built-in default.
    _NO_TEMPERATURE_PREFIXES = ("o1", "o3", "o4", "gpt-5")

    def _build_payload(self, request: ChatRequest) -> dict:
        payload: dict = {
            "model": self.config.model_id,
            "messages": [m.model_dump() for m in request.messages],
            "max_completion_tokens": request.max_completion_tokens,
        }
        mid = (self.config.model_id or "").lower()
        model_rejects_temp = any(mid.startswith(p) for p in self._NO_TEMPERATURE_PREFIXES)
        if request.temperature is not None and not model_rejects_temp:
            payload["temperature"] = request.temperature
        # json_object mode is supported by gpt-4o, gpt-4-turbo, gpt-3.5-turbo-1106+
        # but NOT by o1/o3/o4/gpt-5 reasoning models.
        if request.json_mode and not model_rejects_temp:
            payload["response_format"] = {"type": "json_object"}
        return payload

    async def chat(self, request: ChatRequest) -> ChatResponse:
        payload = self._build_payload(request)
        try:
            logger.debug(f"{self.config.kind.value} chat: model={self.config.model_id}")
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
