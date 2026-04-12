"""Domain glossary agent (section I)."""

from __future__ import annotations

import time
from typing import Any

from domain.model_connector.service import ProviderConfigService
from domain.retrieval.service import RetrievalService
from domain.retrieval.types import RetrievalMode, RetrievalSection, RetrieveRequest
from shared.logger import logger

from ..agent_pipeline import _normalize_conf
from ..profiles import NORMAL_PROFILE, AnalysisProfile
from ..prompts import AGENT_I_SCHEMA_STR, AGENT_I_SYSTEM, render_bundle
from ..schemas import validate_section
from .base import BaseTypedAgent


class GlossaryAgent(BaseTypedAgent):
    def __init__(
        self,
        provider_service: ProviderConfigService,
        retrieval_service: RetrievalService,
    ) -> None:
        super().__init__(provider_service)
        self._retrieval = retrieval_service

    def _fallback(self, reason: str) -> dict[str, Any]:
        return {
            "terms": [],
            "confidence": "low",
            "blind_spots": [f"Agent failed: {reason}"],
        }

    async def run(
        self,
        provider_id: str,
        model_id: str,
        snapshot_id: str,
        profile: AnalysisProfile | None = None,
    ) -> dict[str, Any]:
        t0 = time.monotonic()
        n_chunks = 0
        _profile = profile or NORMAL_PROFILE
        self._session_chunk_ids = []
        try:
            bundle = await self._retrieval.retrieve(
                RetrieveRequest(
                    snapshot_id=snapshot_id,
                    query="domain model entity type definition event constant",
                    section=RetrievalSection.GLOSSARY,
                    mode=RetrievalMode.HYBRID,
                    max_results=_profile.retrieval_max_results,
                )
            )
            self._record_bundle(bundle)
            n_chunks = len(bundle.evidences)
            user_prompt = f"snapshot_id={snapshot_id}\n\nEvidence:\n{render_bundle(bundle)}"
            data = await self._chat_json_with_augment(
                retrieval=self._retrieval,
                snapshot_id=snapshot_id,
                provider_id=provider_id,
                model_id=model_id,
                system_prompt=AGENT_I_SYSTEM,
                user_prompt=user_prompt,
                schema_hint=AGENT_I_SCHEMA_STR,
                max_completion_tokens=_profile.tokens_glossary,
                retrieval_section=RetrievalSection.GLOSSARY,
                profile=_profile,
                agent_tag="GlossaryAgent",
            )
            raw_terms = data.get("terms")
            terms: list[dict[str, Any]] = []
            if isinstance(raw_terms, list):
                for t in raw_terms:
                    if not isinstance(t, dict):
                        continue
                    ev = t.get("evidence_files")
                    if not isinstance(ev, list):
                        ev = []
                    terms.append(
                        {
                            "term": str(t.get("term", "") or ""),
                            "definition": str(t.get("definition", "") or ""),
                            "evidence_files": [str(x) for x in ev if x is not None],
                        }
                    )
            data["terms"] = terms
            raw_bs = data.get("blind_spots")
            if not isinstance(raw_bs, list):
                data["blind_spots"] = []
            else:
                data["blind_spots"] = [str(x) for x in raw_bs if x is not None]
            data["confidence"] = _normalize_conf(str(data.get("confidence", "medium")))
            validate_section("I", data)
            data["retrieved_chunk_ids"] = self._pop_chunk_ids()
            ms = int((time.monotonic() - t0) * 1000)
            logger.info("[GlossaryAgent] %d chunks retrieved, completed in %dms", n_chunks, ms)
            return data
        except Exception as e:
            ms = int((time.monotonic() - t0) * 1000)
            logger.warning("[GlossaryAgent] failed in %dms: %s", ms, e)
            return self._fallback(str(e))
