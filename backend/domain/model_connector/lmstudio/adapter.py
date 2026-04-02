"""LM Studio provider adapter — uses the OpenAI-compatible /v1/chat/completions endpoint."""
import httpx

from domain.model_connector.errors import ProviderError, ProviderErrorCode
from domain.model_connector.types import ChatRequest, ChatResponse, ProviderConfig
from shared.logger import logger

_NO_MODEL_WARNING = (
    "Connected, but no model is loaded in LM Studio. "
    "Load a model in the Chat tab before running analysis."
)


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
                f"Cannot reach LM Studio at {self.config.base_url}. "
                "Make sure LM Studio is open and the Local Server is enabled.",
                provider_id=self.config.id,
                retryable=True,
            ) from e
        except httpx.TimeoutException as e:
            raise ProviderError(
                ProviderErrorCode.TIMEOUT,
                f"LM Studio at {self.config.base_url} timed out.",
                provider_id=self.config.id,
                retryable=True,
            ) from e
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                ProviderErrorCode.UNKNOWN,
                f"LM Studio returned HTTP {e.response.status_code}.",
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
        except httpx.TimeoutException as e:
            raise ProviderError(
                ProviderErrorCode.TIMEOUT,
                "LM Studio request timed out — the model may be too slow or context too large.",
                provider_id=self.config.id,
                retryable=True,
            ) from e
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                ProviderErrorCode.UNKNOWN,
                f"LM Studio returned HTTP {e.response.status_code}: {e.response.text[:200]}",
                provider_id=self.config.id,
            ) from e
        except (KeyError, IndexError) as e:
            raise ProviderError(
                ProviderErrorCode.UNKNOWN,
                f"Unexpected response format from LM Studio: {e}",
                provider_id=self.config.id,
            ) from e
        except Exception as e:
            raise ProviderError(
                ProviderErrorCode.UNKNOWN, str(e), provider_id=self.config.id
            ) from e

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

    async def aclose(self) -> None:
        await self._client.aclose()
