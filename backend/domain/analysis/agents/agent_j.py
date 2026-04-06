"""Risk and complexity agent (section J)."""
from __future__ import annotations

import time
from typing import Any

from domain.model_connector.service import ProviderConfigService
from domain.retrieval.service import RetrievalService
from domain.retrieval.types import RetrievalMode, RetrievalSection, RetrieveRequest
from shared.logger import logger

from ..agent_pipeline import _normalize_conf
from ..prompts import AGENT_J_SCHEMA_STR, AGENT_J_SYSTEM, render_bundle
from ..schemas import validate_section
from ..static_risk import RiskReport
from .base import BaseTypedAgent


class AgentJ(BaseTypedAgent):
    def __init__(
        self,
        provider_service: ProviderConfigService,
        retrieval_service: RetrievalService,
    ) -> None:
        super().__init__(provider_service)
        self._retrieval = retrieval_service

    def _fallback(self, reason: str, static_risk: RiskReport | None) -> dict[str, Any]:
        if static_risk and static_risk.findings:
            findings: list[dict[str, Any]] = []
            for f in static_risk.findings:
                findings.append(
                    {
                        "category": f.category,
                        "severity": f.severity,
                        "title": f.title,
                        "rationale": f.rationale,
                        "evidence": list(f.evidence),
                    }
                )
            ev_files: list[str] = []
            for f in static_risk.findings:
                ev_files.extend(f.evidence[:5])
            return {
                "findings": findings,
                "summary": "Generated from static analysis only (LLM step failed).",
                "confidence": "low",
                "evidence_files": ev_files[:16],
                "blind_spots": [f"Agent failed: {reason}"],
            }
        return {
            "findings": [],
            "summary": f"Agent failed: {reason}",
            "confidence": "low",
            "evidence_files": [],
            "blind_spots": [reason],
        }

    async def run(
        self,
        provider_id: str,
        model_id: str,
        snapshot_id: str,
        static_risk: RiskReport | None = None,
    ) -> dict[str, Any]:
        t0 = time.monotonic()
        n_chunks = 0
        static_ctx = static_risk.as_context_text() if static_risk else "No static risk data."
        try:
            bundle = await self._retrieval.retrieve(
                RetrieveRequest(
                    snapshot_id=snapshot_id,
                    query="risk complexity large file deep nesting TODO FIXME hotspot",
                    section=RetrievalSection.IMPORTANT_FILES,
                    mode=RetrievalMode.HYBRID,
                    max_results=20,
                )
            )
            n_chunks = len(bundle.evidences)
            user_prompt = (
                f"Static risk findings (FACTS — do not contradict):\n{static_ctx}\n\n"
                f"Code evidence:\n{render_bundle(bundle)}"
            )
            data = await self._chat_json_typed(
                provider_id,
                model_id,
                AGENT_J_SYSTEM,
                user_prompt,
                AGENT_J_SCHEMA_STR,
                max_completion_tokens=3000,
            )
            raw_findings = data.get("findings")
            findings: list[dict[str, Any]] = []
            if isinstance(raw_findings, list):
                for f in raw_findings:
                    if not isinstance(f, dict):
                        continue
                    ev = f.get("evidence")
                    if not isinstance(ev, list):
                        ev = []
                    findings.append(
                        {
                            "category": str(f.get("category", "") or "blast_radius"),
                            "severity": str(f.get("severity", "") or "low"),
                            "title": str(f.get("title", "") or ""),
                            "rationale": str(f.get("rationale", "") or ""),
                            "evidence": [str(x) for x in ev if x is not None],
                        }
                    )
            data["findings"] = findings
            data["summary"] = str(data.get("summary", "") or "")
            for key in ("evidence_files", "blind_spots"):
                raw = data.get(key)
                if not isinstance(raw, list):
                    data[key] = []
                else:
                    data[key] = [str(x) for x in raw if x is not None]
            data["confidence"] = _normalize_conf(str(data.get("confidence", "medium")))
            validate_section("J", data)
            ms = int((time.monotonic() - t0) * 1000)
            logger.info("[AgentJ] %d chunks retrieved, completed in %dms", n_chunks, ms)
            return data
        except Exception as e:
            logger.warning("[AgentJ] failed: %s", e)
            ms = int((time.monotonic() - t0) * 1000)
            logger.info("[AgentJ] %d chunks retrieved, completed in %dms", n_chunks, ms)
            return self._fallback(str(e), static_risk)
