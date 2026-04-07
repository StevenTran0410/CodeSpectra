"""Forbidden things / negative conventions agent (section E)."""

from __future__ import annotations

import time
from typing import Any

from domain.model_connector.service import ProviderConfigService
from domain.retrieval.service import RetrievalService
from domain.retrieval.types import RetrievalMode, RetrievalSection, RetrieveRequest
from shared.logger import logger

from ..agent_pipeline import _normalize_conf
from ..prompts import AGENT_E_SCHEMA_STR, AGENT_E_SYSTEM, render_bundle
from ..schemas import validate_section
from ..static_convention import ConventionReport
from ..static_risk import RiskReport
from ._context_builders import (
    build_convention_block,
    build_risk_block,
    extract_d_hint_context,
)
from .base import BaseTypedAgent

_VALID_RULE_STRENGTH = frozenset({"strong", "suspected", "weak"})
_VALID_VIOL_SEV = frozenset({"high", "medium", "low"})

_COMBINED_QUERY = (
    "banned import layer coupling deprecated forbidden boundary violation "
    "anti-pattern code smell inconsistency workaround hack "
    "import from wrong layer cross-boundary violation"
)


class ViolationsAgent(BaseTypedAgent):
    def __init__(
        self,
        provider_service: ProviderConfigService,
        retrieval_service: RetrievalService,
    ) -> None:
        super().__init__(provider_service)
        self._retrieval = retrieval_service

    def _fallback(self, reason: str) -> dict[str, Any]:
        return {
            "rules": [],
            "violations_found": [],
            "confidence": "low",
            "evidence_files": [],
            "blind_spots": [f"Agent failed: {reason}"],
        }

    async def run(
        self,
        provider_id: str,
        model_id: str,
        snapshot_id: str,
        static_convention: ConventionReport | None = None,
        static_risk: RiskReport | None = None,
        agent_d_output: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        t0 = time.monotonic()
        n_chunks = 0
        try:
            prefix_parts: list[str] = []
            cb = build_convention_block(static_convention)
            if cb:
                prefix_parts.append(cb)
            rb = build_risk_block(static_risk, categories=["blast_radius", "anti_pattern"])
            if rb:
                prefix_parts.append(rb)
            dh = extract_d_hint_context(agent_d_output)
            if dh:
                prefix_parts.append(dh)
            prefix = "\n\n".join(prefix_parts) + ("\n\n" if prefix_parts else "")

            bundle = await self._retrieval.retrieve(
                RetrieveRequest(
                    snapshot_id=snapshot_id,
                    query=_COMBINED_QUERY,
                    section=RetrievalSection.CONVENTIONS,
                    mode=RetrievalMode.HYBRID,
                    max_results=30,
                )
            )
            n_chunks = len(bundle.evidences)
            user_prompt = (
                f"{prefix}snapshot_id={snapshot_id}\n\nEvidence:\n{render_bundle(bundle)}"
            )
            data = await self._chat_json_typed(
                provider_id,
                model_id,
                AGENT_E_SYSTEM,
                user_prompt,
                schema_hint=AGENT_E_SCHEMA_STR,
                max_completion_tokens=2000,
            )

            raw_rules = data.get("rules")
            rules_out: list[dict[str, str]] = []
            if isinstance(raw_rules, list):
                for i in raw_rules:
                    if not isinstance(i, dict):
                        continue
                    sev = i.get("strength") or i.get("severity")
                    if sev not in _VALID_RULE_STRENGTH:
                        sev = "weak"
                    rationale = str(i.get("rationale", "") or i.get("inferred_from", "") or "")
                    evf = i.get("evidence_files")
                    if isinstance(evf, list) and evf:
                        extra = ", ".join(str(x) for x in evf[:8] if x is not None)
                        if extra:
                            rationale = (
                                f"{rationale} | files: {extra}"
                                if rationale
                                else f"files: {extra}"
                            )
                    rules_out.append(
                        {
                            "rule": str(i.get("rule", "") or ""),
                            "inferred_from": rationale,
                            "severity": str(sev),
                        }
                    )
            data["rules"] = rules_out

            raw_v = data.get("violations_found")
            if not isinstance(raw_v, list):
                raw_v = data.get("violations")
            vout: list[dict[str, str]] = []
            if isinstance(raw_v, list):
                for i in raw_v:
                    if not isinstance(i, dict):
                        continue
                    sev = str(i.get("severity", "medium") or "medium").lower()
                    if sev not in _VALID_VIOL_SEV:
                        sev = "medium"
                    rule = str(i.get("rule_broken", "") or i.get("rule", "") or "")
                    loc = str(i.get("location", "") or i.get("file", "") or "")
                    desc = str(i.get("description", "") or "")
                    vout.append(
                        {
                            "rule": rule,
                            "file": loc,
                            "description": desc,
                            "severity": sev,
                        }
                    )
            data["violations_found"] = vout
            data.pop("violations", None)

            for key in ("evidence_files", "blind_spots"):
                raw = data.get(key)
                if isinstance(raw, list):
                    data[key] = [str(x) for x in raw if x is not None]
                else:
                    data[key] = []
            data["confidence"] = _normalize_conf(str(data.get("confidence", "medium")))
            validate_section("E", data)
            ms = int((time.monotonic() - t0) * 1000)
            logger.info("[ViolationsAgent] %d chunks retrieved, completed in %dms", n_chunks, ms)
            return data
        except Exception as e:
            ms = int((time.monotonic() - t0) * 1000)
            logger.warning("[ViolationsAgent] failed in %dms: %s", ms, e)
            return self._fallback(str(e))
