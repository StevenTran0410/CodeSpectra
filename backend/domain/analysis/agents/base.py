"""Base agent with JSON chat path for typed section schemas (RPA-053)."""

from __future__ import annotations

import re
from typing import Any

from domain.model_connector.service import ProviderConfigService
from shared.logger import logger

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
        schema_hint: str,
        max_completion_tokens: int = 1200,
    ) -> dict[str, Any]:
        """Like _chat_json but repair prompt uses schema_hint instead of SectionDraft schema."""
        text = await self._call(
            provider_id,
            model_id,
            system_prompt,
            user_prompt,
            max_completion_tokens,
            temperature=0.2,
        )
        try:
            obj = self._try_parse_json(text)
            if isinstance(obj, dict) and obj:
                return obj
        except Exception:
            pass

        logger.warning("Typed LLM agent: attempt 1 bad output (%d chars), retrying", len(text))

        if not text.strip():
            # Model returned empty — retry with explicit JSON instruction, higher budget
            retry_system = (
                system_prompt + "\n\nCRITICAL: You MUST respond with a JSON object. "
                "Start your response with { and end with }. No prose."
            )
            text2 = await self._call(
                provider_id,
                model_id,
                retry_system,
                user_prompt,
                max_completion_tokens * 2,
                temperature=0.1,
            )
        else:
            # Model returned prose — ask it to extract fields from its own output
            text2 = await self._call(
                provider_id,
                model_id,
                system_prompt,
                (
                    "Extract the required JSON fields from your previous response.\n"
                    "Return ONLY valid JSON, no markdown fence, no commentary.\n"
                    f"Required schema:\n{schema_hint}\n\n"
                    f"Previous output:\n{text}"
                ),
                max_completion_tokens,
                temperature=None,
            )
        try:
            obj2 = self._try_parse_json(text2)
            if isinstance(obj2, dict) and obj2:
                return obj2
        except Exception:
            pass

        raise ValueError(f"all_attempts_failed_typed: last_output={text2[:120]!r}")
