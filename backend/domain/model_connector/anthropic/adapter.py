"""Anthropic provider adapter — x-api-key header, /v1/messages endpoint."""
import httpx

from domain.model_connector._cloud_base import CloudAdapterBase
from domain.model_connector.errors import ProviderError, ProviderErrorCode
from domain.model_connector.types import ChatRequest, ChatResponse, ProviderConfig
from shared.logger import logger

ANTHROPIC_VERSION = "2023-06-01"

MODEL_PRESETS = [
    "claude-opus-4-5",
    "claude-sonnet-4-5",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-haiku-20240307",
]

# Minimal model for test_connection (cheapest available)
_TEST_MODEL = "claude-3-haiku-20240307"


class AnthropicAdapter(CloudAdapterBase):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config, base_url="https://api.anthropic.com")

    def _auth_headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._require_api_key(),
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    async def list_models(self) -> list[str]:
        """Anthropic has no public list-models endpoint — return known presets."""
        return MODEL_PRESETS

    async def chat(self, request: ChatRequest) -> ChatResponse:
        # Anthropic separates system prompt from user messages
        system_parts = [m.content for m in request.messages if m.role == "system"]
        user_messages = [
            {"role": m.role, "content": m.content}
            for m in request.messages
            if m.role != "system"
        ]
        payload: dict = {
            "model": self.config.model_id,
            "max_completion_tokens": request.max_completion_tokens,
            "messages": user_messages,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)

        try:
            logger.debug(f"Anthropic chat: model={self.config.model_id}")
            res = await self._client.post("/v1/messages", json=payload, headers=self._auth_headers())
            res.raise_for_status()
            data = res.json()
            content_blocks = data.get("content", [])
            text = " ".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")
            usage = data.get("usage", {})
            return ChatResponse(
                provider_id=self.config.id,
                model_id=self.config.model_id,
                content=text,
                prompt_tokens=usage.get("input_tokens"),
                completion_tokens=usage.get("output_tokens"),
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
        """Validate API key with a 1-token call (costs ~$0.000001)."""
        if not self._api_key:
            return False, "No API key configured", None
        payload = {
            "model": _TEST_MODEL,
            "max_completion_tokens": 1,
            "messages": [{"role": "user", "content": "hi"}],
        }
        try:
            res = await self._client.post("/v1/messages", json=payload, headers=self._auth_headers())
            if res.status_code in (200, 400):
                # 200 = success, 400 could be model issue but auth is OK
                return True, "Connected — API key is valid", None
            res.raise_for_status()
            return True, "Connected", None
        except httpx.HTTPStatusError as e:
            err = self._map_http_error(e)
            return False, err.message, None
        except httpx.ConnectError as e:
            return False, self._map_connect_error(e).message, None
        except httpx.TimeoutException as e:
            return False, self._map_timeout(e).message, None
        except Exception as e:
            return False, str(e), None
