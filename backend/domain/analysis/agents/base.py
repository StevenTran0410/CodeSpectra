"""Base agent with JSON chat path for typed section schemas (RPA-053)."""
from __future__ import annotations

from typing import Any

from domain.model_connector.service import ProviderConfigService
from shared.logger import logger

from ..agent_pipeline import BaseLLMAgent


class BaseTypedAgent(BaseLLMAgent):
    def __init__(self, provider_service: ProviderConfigService) -> None:
        super().__init__(provider_service)

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

        logger.warning("Typed LLM agent: attempt 1 bad output, asking model to repair JSON")

        repair_user = (
            "The following is your previous output. Extract the required JSON fields from it.\n"
            "If a field is missing or unclear, use a reasonable default.\n"
            "Return ONLY valid JSON, no markdown fence, no commentary.\n"
            f"Required schema: {schema_hint}\n\n"
            f"Previous output:\n{text}"
        )
        text2 = await self._call(
            provider_id,
            model_id,
            system_prompt,
            repair_user,
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
