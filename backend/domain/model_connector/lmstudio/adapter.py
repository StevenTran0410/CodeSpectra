"""LM Studio provider adapter — uses the OpenAI-compatible /v1/chat/completions endpoint."""
import httpx

from domain.model_connector._local_base import LocalAdapterBase
from domain.model_connector.errors import ProviderError
from domain.model_connector.types import ChatRequest, ChatResponse
from shared.logger import logger

_NO_MODEL_WARNING = (
    "Connected, but no model is loaded in LM Studio. "
    "Load a model in the Chat tab before running analysis."
)


class LMStudioAdapter(LocalAdapterBase):
    """Wrapper over LM Studio's OpenAI-compatible REST API."""

    async def list_models(self) -> list[str]:
        try:
            res = await self._client.get("/v1/models")
            res.raise_for_status()
            data = res.json()
            return [m["id"] for m in data.get("data", [])]
        except httpx.ConnectError as e:
            raise self._map_connect_error(e) from e
        except httpx.TimeoutException as e:
            raise self._map_timeout(e) from e
        except httpx.HTTPStatusError as e:
            raise self._map_http_status(e) from e
        except Exception as e:
            raise self._map_unknown(e) from e

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
            raise self._map_connect_error(e) from e
        except httpx.TimeoutException as e:
            raise ProviderError(
                # Override message — LM Studio times out when model is slow
                self._map_timeout(e).code,
                "LM Studio request timed out — the model may be too slow or context too large.",
                provider_id=self.config.id,
                retryable=True,
            ) from e
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                self._map_http_status(e).code,
                f"LM Studio returned HTTP {e.response.status_code}: {e.response.text[:200]}",
                provider_id=self.config.id,
            ) from e
        except (KeyError, IndexError) as e:
            raise self._map_unknown(e) from e
        except Exception as e:
            raise self._map_unknown(e) from e

    async def test_connection(self) -> tuple[bool, str, str | None]:
        """Returns (ok, message, warning).

        - ok=False: cannot connect at all
        - ok=True, warning=None: connected + models loaded
        - ok=True, warning=str: connected but no model loaded yet
        """
        try:
            models = await self.list_models()
            if not models:
                return True, "Connected — no model loaded yet", _NO_MODEL_WARNING
            return True, f"Connected — {len(models)} model(s) loaded", None
        except ProviderError as e:
            return False, e.message, None
