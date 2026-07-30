"""Ollama provider adapter — implements LLM calls against a local Ollama server."""
import httpx

from domain.model_connector._local_base import LocalAdapterBase
from domain.model_connector.errors import ProviderError, ProviderErrorCode
from domain.model_connector.reasoning import ReasoningStyle, classify
from domain.model_connector.types import ChatRequest, ChatResponse, EmbedRequest, EmbedResponse, ProviderKind
from shared.logger import logger

# Local embedding-only models commonly pulled alongside chat models — not usable
# for text generation, so they're excluded from the model picker.
_NON_CHAT_MARKERS = ("embed", "bert")


class OllamaAdapter(LocalAdapterBase):
    """Thin async wrapper over the Ollama /api/chat endpoint."""

    async def list_models(self) -> list[str]:
        try:
            res = await self._client.get("/api/tags")
            res.raise_for_status()
            data = res.json()
            return [
                m["name"]
                for m in data.get("models", [])
                if not any(marker in m["name"].lower() for marker in _NON_CHAT_MARKERS)
            ]
        except httpx.ConnectError as e:
            raise self._map_connect_error(e) from e
        except httpx.TimeoutException as e:
            raise self._map_timeout(e) from e
        except httpx.HTTPStatusError as e:
            raise self._map_http_status(e) from e
        except Exception as e:
            raise self._map_unknown(e) from e

    async def chat(self, request: ChatRequest) -> ChatResponse:
        options: dict = {"num_predict": request.max_completion_tokens}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        payload: dict = {
            "model": self.config.model_id,
            "messages": [m.model_dump() for m in request.messages],
            "stream": False,
            "options": options,
        }
        if request.json_mode:
            payload["format"] = "json"
        if (
            classify(ProviderKind.OLLAMA, self.config.model_id or "") == ReasoningStyle.TOGGLE
            and request.thinking_budget
        ):
            payload["think"] = True
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
            raise self._map_connect_error(e) from e
        except httpx.TimeoutException as e:
            raise self._map_timeout(e) from e
        except httpx.HTTPStatusError as e:
            raise self._map_http_status(e) from e
        except Exception as e:
            raise self._map_unknown(e) from e

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

    async def embed(self, request: EmbedRequest) -> EmbedResponse:  # noqa: ARG002
        raise ProviderError(
            ProviderErrorCode.UNKNOWN,
            "Ollama embeddings are not wired in this version — use OpenAI, Gemini, or Local GPU",
            provider_id=self.config.id,
        )
