"""Google Gemini provider adapter — API key as header, /v1beta endpoint format."""
import httpx

from domain.model_connector._cloud_base import CloudAdapterBase
from domain.model_connector.errors import ProviderError, ProviderErrorCode
from domain.model_connector.types import ChatRequest, ChatResponse, ProviderConfig
from shared.logger import logger

MODEL_PRESETS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
]


class GeminiAdapter(CloudAdapterBase):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config, base_url="https://generativelanguage.googleapis.com")

    def _key_param(self) -> dict[str, str]:
        return {"key": self._require_api_key()}

    async def list_models(self) -> list[str]:
        try:
            res = await self._client.get("/v1beta/models", params=self._key_param())
            res.raise_for_status()
            data = res.json()
            models = [
                m["name"].replace("models/", "")
                for m in data.get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", [])
            ]
            return models or MODEL_PRESETS
        except httpx.ConnectError as e:
            raise self._map_connect_error(e) from e
        except httpx.TimeoutException as e:
            raise self._map_timeout(e) from e
        except httpx.HTTPStatusError as e:
            raise self._map_http_error(e) from e
        except Exception as e:
            raise ProviderError(ProviderErrorCode.UNKNOWN, str(e), provider_id=self.config.id) from e

    async def chat(self, request: ChatRequest) -> ChatResponse:
        model = self.config.model_id
        # Build Gemini contents array from messages
        contents = []
        system_parts = []
        for msg in request.messages:
            if msg.role == "system":
                system_parts.append({"text": msg.content})
            else:
                role = "user" if msg.role == "user" else "model"
                contents.append({"role": role, "parts": [{"text": msg.content}]})

        generation_config: dict = {
            "maxOutputTokens": request.max_tokens,
        }
        # Some Gemini models only allow default temperature=1.
        # Keep request temperature only when it is effectively default.
        if abs(float(request.temperature) - 1.0) < 1e-9:
            generation_config["temperature"] = 1.0

        payload: dict = {
            "contents": contents,
            "generationConfig": generation_config,
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}

        url = f"/v1beta/models/{model}:generateContent"
        try:
            logger.debug(f"Gemini chat: model={model}")
            res = await self._client.post(url, json=payload, params=self._key_param())
            res.raise_for_status()
            data = res.json()
            candidates = data.get("candidates", [])
            text = ""
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                text = " ".join(p.get("text", "") for p in parts)
            usage = data.get("usageMetadata", {})
            return ChatResponse(
                provider_id=self.config.id,
                model_id=model,
                content=text,
                prompt_tokens=usage.get("promptTokenCount"),
                completion_tokens=usage.get("candidatesTokenCount"),
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
