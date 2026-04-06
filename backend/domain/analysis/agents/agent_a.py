"""Project identity agent (section A)."""
from __future__ import annotations

import time
from typing import Any

from domain.model_connector.service import ProviderConfigService
from domain.retrieval.service import RetrievalService
from domain.retrieval.types import RetrievalMode, RetrievalSection, RetrieveRequest
from shared.logger import logger

from ..agent_pipeline import _normalize_conf
from ..prompts import AGENT_A_SCHEMA_STR, AGENT_A_SYSTEM, render_bundle
from ..schemas import validate_section
from .base import BaseTypedAgent


class AgentA(BaseTypedAgent):
    def __init__(
        self,
        provider_service: ProviderConfigService,
        retrieval_service: RetrievalService,
    ) -> None:
        super().__init__(provider_service)
        self._retrieval = retrieval_service

    def _fallback(self, snapshot_id: str, reason: str) -> dict[str, Any]:
        return {
            "repo_name": snapshot_id,
            "domain": "unknown",
            "purpose": "",
            "runtime_type": "unknown",
            "tech_stack": [],
            "business_context": "",
            "confidence": "low",
            "evidence_files": [],
            "blind_spots": [f"Agent failed: {reason}"],
        }

    async def run(self, provider_id: str, model_id: str, snapshot_id: str) -> dict[str, Any]:
        t0 = time.monotonic()
        n_chunks = 0
        try:
            bundle = await self._retrieval.retrieve(
                RetrieveRequest(
                    snapshot_id=snapshot_id,
                    query="README entrypoint package manifest",
                    section=RetrievalSection.IMPORTANT_FILES,
                    mode=RetrievalMode.HYBRID,
                    max_results=16,
                )
            )
            n_chunks = len(bundle.evidences)
            user_prompt = f"snapshot_id={snapshot_id}\n\nEvidence:\n{render_bundle(bundle)}"
            data = await self._chat_json_typed(
                provider_id,
                model_id,
                AGENT_A_SYSTEM,
                user_prompt,
                AGENT_A_SCHEMA_STR,
                max_completion_tokens=900,
            )
            for key in ("repo_name", "domain", "purpose", "runtime_type", "business_context"):
                data[key] = str(data.get(key, "") or "")
            for key in ("tech_stack", "evidence_files", "blind_spots"):
                raw = data.get(key)
                if not isinstance(raw, list):
                    data[key] = []
                else:
                    data[key] = [str(x) for x in raw if x is not None]
            data["confidence"] = _normalize_conf(str(data.get("confidence", "medium")))
            validate_section("A", data)
            ms = int((time.monotonic() - t0) * 1000)
            logger.info("[AgentA] %d chunks retrieved, completed in %dms", n_chunks, ms)
            return data
        except Exception as e:
            logger.warning("[AgentA] failed: %s", e)
            ms = int((time.monotonic() - t0) * 1000)
            logger.info("[AgentA] %d chunks retrieved, completed in %dms", n_chunks, ms)
            return self._fallback(snapshot_id, str(e))
