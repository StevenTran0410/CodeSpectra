"""Base agent with JSON chat path for typed section schemas (RPA-053)."""

from __future__ import annotations

import re
from typing import Any

from domain.model_connector.service import ProviderConfigService

from ..agent_pipeline import BaseLLMAgent

# Strip control characters that make JSON request bodies invalid (e.g. null bytes
# from binary file content leaking into retrieved chunks).
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize(text: str) -> str:
    """Remove non-printable control characters, keep \\t \\n \\r."""
    return _CTRL_RE.sub("", text)


class BaseTypedAgent(BaseLLMAgent):
    def __init__(self, provider_service: ProviderConfigService) -> None:
        super().__init__(provider_service)

    async def _call(
        self,
        provider_id,
        model_id,
        system_prompt,
        user_prompt,
        max_completion_tokens,
        temperature=0.2,
        json_mode=True,
    ):
        """Override to sanitize prompts before sending to LLM."""
        return await super()._call(
            provider_id,
            model_id,
            _sanitize(system_prompt),
            _sanitize(user_prompt),
            max_completion_tokens,
            temperature,
            json_mode,
        )

    async def _chat_json_typed(
        self,
        provider_id: str,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
        schema_hint: str = "",
        max_completion_tokens: int = 1200,
    ) -> dict[str, Any]:
        return await super()._chat_json(
            provider_id,
            model_id,
            system_prompt,
            user_prompt,
            max_completion_tokens=max_completion_tokens,
            schema_hint=schema_hint,
        )
