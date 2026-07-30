"""Google Gemini provider adapter — API key as header, /v1beta endpoint format."""
import httpx

from domain.model_connector._cloud_base import CloudAdapterBase
from domain.model_connector.errors import ProviderError, ProviderErrorCode
from domain.model_connector.reasoning import ReasoningStyle, classify, thinking_budget_range
from domain.model_connector.types import ChatRequest, ChatResponse, EmbedRequest, EmbedResponse, ProviderConfig, ProviderKind
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
            raw = [m for m in data.get("models", []) if isinstance(m, dict) and "name" in m]
            if not raw:
                return MODEL_PRESETS
            filtered = [
                m["name"].replace("models/", "")
                for m in raw
                if "generateContent" in m.get("supportedGenerationMethods", [])
            ]
            # A non-empty response the capability filter zeroes out entirely means this
            # endpoint isn't shaped like the real Gemini API (e.g. a 3rd-party gateway that
            # doesn't populate supportedGenerationMethods) — surface the raw list instead of
            # silently substituting fake Gemini preset names that don't exist on it.
            return filtered if filtered else [m["name"].replace("models/", "") for m in raw]
        except httpx.ConnectError as e:
            raise self._map_connect_error(e) from e
        except httpx.TimeoutException as e:
            raise self._map_timeout(e) from e
        except httpx.HTTPStatusError as e:
            raise self._map_http_error(e) from e
        except Exception as e:
            raise ProviderError(ProviderErrorCode.UNKNOWN, str(e), provider_id=self.config.id) from e

    async def list_embedding_models(self) -> list[str]:
        """Embedding models the endpoint exposes (supportedGenerationMethods includes embedContent)."""
        try:
            res = await self._client.get("/v1beta/models", params=self._key_param())
            res.raise_for_status()
            raw = [m for m in res.json().get("models", []) if isinstance(m, dict) and "name" in m]
            return [
                m["name"].replace("models/", "")
                for m in raw
                if any("mbedContent" in meth for meth in m.get("supportedGenerationMethods", []))
            ]
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
            "maxOutputTokens": request.max_completion_tokens,
        }
        # Gemini API supports temperature in [0.0, 1.0].
        # Omit when None or caller explicitly wants provider default.
        if request.temperature is not None and 0.0 <= request.temperature <= 1.0:
            generation_config["temperature"] = request.temperature
        if request.json_mode:
            generation_config["responseMimeType"] = "application/json"
        if (
            classify(ProviderKind.GEMINI, model or "") == ReasoningStyle.THINKING_BUDGET
            and request.thinking_budget is not None
        ):
            lo, hi, _can_disable = thinking_budget_range(ProviderKind.GEMINI, model or "")
            budget = request.thinking_budget
            if budget != -1:  # -1 = dynamic thinking, always valid, no clamping
                budget = min(max(budget, lo), hi)
            generation_config["thinkingConfig"] = {"thinkingBudget": budget}

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

    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        model = request.model_id or "gemini-embedding-001"
        # Gemini taskType: RETRIEVAL_DOCUMENT for corpus chunks (default), RETRIEVAL_QUERY
        # for embedding the query side. Using the wrong task_type measurably hurts quality.
        task_type_str = (
            "RETRIEVAL_QUERY" if request.task_type == "retrieval_query" else "RETRIEVAL_DOCUMENT"
        )
        requests_body = [
            {"model": f"models/{model}", "content": {"parts": [{"text": t}]}, "taskType": task_type_str}
            for t in request.texts
        ]
        url = f"/v1beta/models/{model}:batchEmbedContents"
        payload = {"requests": requests_body}
        try:
            logger.debug(f"gemini embed: model={model}, n={len(request.texts)}, task={task_type_str}")
            res = await self._client.post(url, json=payload, params=self._key_param())
            res.raise_for_status()
            data = res.json()
            embeddings = [item["values"] for item in data.get("embeddings", [])]
            dims = len(embeddings[0]) if embeddings else 0
            return EmbedResponse(
                provider_id=self.config.id,
                model_id=model,
                embeddings=embeddings,
                dimensions=dims,
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
