"""Synthesis agent (section L) — no retrieval."""

from __future__ import annotations

import json
import time
from typing import Any

from domain.model_connector.service import ProviderConfigService
from shared.logger import logger

from ..agent_pipeline import _normalize_conf
from ..prompts import AGENT_L_SCHEMA_STR, AGENT_L_SYSTEM
from ..schemas import validate_section
from ._section_compressor import compress_audit, compress_section
from .base import BaseTypedAgent

_PROSE_FIELDS = (
    "executive_summary",
    "architecture_narrative",
    "tech_stack_snapshot",
    "developer_quickstart",
    "conventions_digest",
    "risk_highlights",
    "reading_path",
)


def _build_agent_l_input(sections: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for letter in "ABCDEFGHIJ":
        s = sections.get(letter) or {}
        compact[letter] = compress_section(letter, s, char_cap=800)
    compact["K"] = compress_audit(sections.get("K") or {}, char_cap=800)
    return compact


def _agent_l_fallback() -> dict[str, Any]:
    return {
        "executive_summary": "",
        "architecture_narrative": "",
        "tech_stack_snapshot": "",
        "developer_quickstart": "",
        "conventions_digest": "",
        "risk_highlights": "",
        "reading_path": "",
        "confidence": "low",
    }


class SynthesisAgent(BaseTypedAgent):
    def __init__(self, provider_service: ProviderConfigService) -> None:
        super().__init__(provider_service)

    async def run(
        self,
        provider_id: str,
        model_id: str,
        all_sections: dict[str, Any],
    ) -> dict[str, Any]:
        t0 = time.monotonic()
        compact = _build_agent_l_input(all_sections)
        user_prompt = json.dumps(compact, ensure_ascii=False)
        result = await self._chat_json_typed(
            provider_id,
            model_id,
            AGENT_L_SYSTEM,
            user_prompt,
            AGENT_L_SCHEMA_STR,
            max_completion_tokens=4000,
        )
        for field in _PROSE_FIELDS:
            result[field] = str(result.get(field) or "")
        result["confidence"] = _normalize_conf(str(result.get("confidence", "medium")))
        validate_section("L", result)
        ms = int((time.monotonic() - t0) * 1000)
        logger.info("[SynthesisAgent] completed in %dms", ms)
        return result
