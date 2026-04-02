"""Shared base for local LLM adapters (Ollama, LM Studio).

Both adapters talk to a server running on the user's machine over HTTP.
They share identical httpx client setup and identical list_models error mapping.
"""
import httpx

from .errors import ProviderError, ProviderErrorCode
from .types import ProviderConfig


class LocalAdapterBase:
    """Base class for local (on-device) LLM provider adapters."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0),
        )

    # ── Error mapping ────────────────────────────────────────────────────────

    def _map_connect_error(self, e: Exception, *, retryable: bool = True) -> ProviderError:
        return ProviderError(
            ProviderErrorCode.CONNECTION_REFUSED,
            f"Cannot reach {self.config.display_name} at {self.config.base_url}. "
            "Make sure the server is running.",
            provider_id=self.config.id,
            retryable=retryable,
        )

    def _map_timeout(self, e: Exception) -> ProviderError:
        return ProviderError(
            ProviderErrorCode.TIMEOUT,
            f"{self.config.display_name} at {self.config.base_url} timed out.",
            provider_id=self.config.id,
            retryable=True,
        )

    def _map_http_status(self, e: httpx.HTTPStatusError) -> ProviderError:
        return ProviderError(
            ProviderErrorCode.UNKNOWN,
            f"{self.config.display_name} returned HTTP {e.response.status_code}.",
            provider_id=self.config.id,
        )

    def _map_unknown(self, e: Exception) -> ProviderError:
        return ProviderError(
            ProviderErrorCode.UNKNOWN,
            str(e),
            provider_id=self.config.id,
        )

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def aclose(self) -> None:
        await self._client.aclose()
