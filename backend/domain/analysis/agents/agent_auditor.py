"""Meta-audit agent (section K) — no retrieval."""

from __future__ import annotations

import json
import time
from typing import Any

from domain.model_connector.service import ProviderConfigService
from shared.logger import logger

from ..agent_pipeline import _normalize_conf
from ..prompts import AGENT_K_SCHEMA_STR, AGENT_K_SYSTEM
from ..schemas import validate_section
from .base import BaseTypedAgent


def _build_agent_k_input(sections: dict[str, Any]) -> dict[str, Any]:
    """
    Compress 10 section outputs into a bounded AuditAgent input.

    Per section extracts: confidence, blind_spots[:3], content_preview[:500].
    """
    compressed: dict[str, Any] = {}
    for letter in "ABCDEFGHIJ":
        s = sections.get(letter) or {}
        compressed[letter] = {
            "confidence": s.get("confidence", "medium"),
            "blind_spots": (s.get("blind_spots") or [])[:3],
            "content_preview": str(s.get("content", ""))[:500],
        }
    return compressed


def _agent_k_fallback() -> dict[str, Any]:
    return {
        "overall_confidence": "low",
        "section_scores": {},
        "weakest_sections": [],
        "coverage_percentage": 0.0,
        "notes": "Audit could not be completed.",
        "blind_spots": ["AuditAgent failed"],
    }


class AuditAgent(BaseTypedAgent):
    def __init__(self, provider_service: ProviderConfigService) -> None:
        super().__init__(provider_service)

    async def run(
        self,
        provider_id: str,
        model_id: str,
        all_sections: dict[str, Any],
    ) -> dict[str, Any]:
        t0 = time.monotonic()
        compact = _build_agent_k_input(all_sections)
        user_prompt = json.dumps(compact, ensure_ascii=False)
        result = await self._chat_json_typed(
            provider_id,
            model_id,
            AGENT_K_SYSTEM,
            user_prompt,
            AGENT_K_SCHEMA_STR,
            max_completion_tokens=2000,
        )
        valid_letters = set("ABCDEFGHIJ")
        result["overall_confidence"] = _normalize_conf(
            str(result.get("overall_confidence", "medium"))
        )
        result["section_scores"] = {
            str(k).upper(): _normalize_conf(str(v))
            for k, v in (result.get("section_scores") or {}).items()
            if str(k).upper() in valid_letters
        }
        raw_w = result.get("weakest_sections", [])
        result["weakest_sections"] = [
            str(s).upper()
            for s in (raw_w if isinstance(raw_w, list) else [])
            if str(s).upper() in valid_letters
        ]
        result["coverage_percentage"] = float(result.get("coverage_percentage", 0.0))
        notes = result.get("notes", "")
        result["notes"] = str(notes) if notes is not None else ""
        raw_bs = result.get("blind_spots", [])
        result["blind_spots"] = [
            str(x) for x in (raw_bs if isinstance(raw_bs, list) else []) if x is not None
        ]
        validate_section("K", result)
        ms = int((time.monotonic() - t0) * 1000)
        logger.info("[AuditAgent] completed in %dms", ms)
        return result
