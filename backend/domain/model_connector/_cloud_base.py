"""Shared base for all cloud provider adapters."""
from __future__ import annotations

from collections.abc import AsyncGenerator

import httpx

from shared.logger import logger

from .errors import ProviderError, ProviderErrorCode
from .types import ChatRequest, ProviderConfig


class CloudAdapterBase:
    def __init__(self, config: ProviderConfig, base_url: str | None = None) -> None:
        self.config = config
        self._client = httpx.AsyncClient(
            base_url=base_url or config.base_url,
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0),
        )

    @property
    def _api_key(self) -> str:
        return str(self.config.extra.get("api_key", ""))

    def _require_api_key(self) -> str:
        key = self._api_key
        if not key:
            raise ProviderError(
                ProviderErrorCode.AUTH_FAILED,
                "No API key configured. Add your API key in provider settings.",
                provider_id=self.config.id,
            )
        return key

    def _map_http_error(self, e: httpx.HTTPStatusError) -> ProviderError:
        status = e.response.status_code
        try:
            body = e.response.json()
            msg = (
                body.get("error", {}).get("message")
                or body.get("message")
                or body.get("error")
                or str(body)
            )
        except Exception:
            msg = e.response.text[:300]

        code_map = {
            401: ProviderErrorCode.AUTH_FAILED,
            403: ProviderErrorCode.AUTH_FAILED,
            404: ProviderErrorCode.MODEL_NOT_FOUND,
            429: ProviderErrorCode.RATE_LIMITED,
        }
        code = code_map.get(status, ProviderErrorCode.UNKNOWN)
        # 4xx errors are client-side bugs (bad request, auth, model not found) — log at ERROR
        # so they surface clearly in the terminal, not hidden among INFO/WARNING noise.
        # 429 is excluded because it is expected under load and handled by callers.
        if 400 <= status < 500 and status != 429:
            logger.error(
                "[provider:%s] HTTP %d from %s — %s",
                self.config.id, status, self.config.base_url, msg,
            )
        return ProviderError(code, f"HTTP {status}: {msg}", provider_id=self.config.id)

    def _map_connect_error(self, e: httpx.ConnectError) -> ProviderError:
        return ProviderError(
            ProviderErrorCode.CONNECTION_REFUSED,
            f"Cannot connect to {self.config.base_url}: {e}",
            provider_id=self.config.id,
            retryable=True,
        )

    def _map_timeout(self, e: httpx.TimeoutException) -> ProviderError:
        return ProviderError(
            ProviderErrorCode.TIMEOUT,
            "Request timed out.",
            provider_id=self.config.id,
            retryable=True,
        )

    async def chat_stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """Default fallback: call chat() and yield the full response as a single chunk.

        Subclasses that support native token streaming should override this.
        """
        from .types import ChatResponse  # avoid circular at module level
        response: ChatResponse = await self.chat(request)  # type: ignore[attr-defined]
        content = response.content or ""
        if content:
            yield content

    async def aclose(self) -> None:
        await self._client.aclose()
