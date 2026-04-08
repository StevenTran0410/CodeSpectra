"""Onboarding reading-order agent (section H)."""

from __future__ import annotations

import time
from typing import Any

from domain.model_connector.service import ProviderConfigService
from domain.retrieval.service import RetrievalService
from domain.retrieval.types import RetrievalMode, RetrievalSection, RetrieveRequest
from shared.logger import logger

from ..agent_pipeline import _normalize_conf
from ..prompts import AGENT_H_SCHEMA_STR, AGENT_H_SYSTEM, render_bundle
from ..schemas import validate_section
from .base import BaseTypedAgent

_G_SLOT_KEYS = (
    "entrypoint",
    "backbone",
    "critical_config",
    "highest_centrality",
    "most_dangerous_to_touch",
    "read_first",
)


def _extract_g_hint(important_files_output: dict[str, Any]) -> str:
    entries: list[str] = []
    seen: set[str] = set()

    def push(file: str, reason: str) -> None:
        f = (file or "").strip()
        if not f or f == "unknown":
            return
        r = (reason or "").strip()
        if not r:
            return
        if f in seen:
            return
        seen.add(f)
        entries.append(f"- {f} ({r[:80]})")

    for key in _G_SLOT_KEYS:
        slot = important_files_output.get(key)
        if isinstance(slot, dict):
            push(str(slot.get("file", "")), str(slot.get("reason", "")))
    oi = important_files_output.get("other_important")
    if isinstance(oi, list):
        for item in oi:
            if isinstance(item, dict):
                push(str(item.get("file", "")), str(item.get("reason", "")))
    if not entries:
        return ""
    header = "Key files identified by Important Files Radar:\n"
    return header + "\n".join(entries[:5])


class OnboardingAgent(BaseTypedAgent):
    def __init__(
        self,
        provider_service: ProviderConfigService,
        retrieval_service: RetrievalService,
    ) -> None:
        super().__init__(provider_service)
        self._retrieval = retrieval_service

    def _fallback(self, reason: str) -> dict[str, Any]:
        return {
            "steps": [],
            "total_estimated_minutes": 30,
            "confidence": "low",
            "evidence_files": [],
            "blind_spots": [f"Agent failed: {reason}"],
        }

    async def run(
        self,
        provider_id: str,
        model_id: str,
        snapshot_id: str,
        important_files_output: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        t0 = time.monotonic()
        n_chunks = 0
        try:
            g_hint = (
                _extract_g_hint(important_files_output)
                if isinstance(important_files_output, dict)
                else ""
            )
            bundle = await self._retrieval.retrieve(
                RetrieveRequest(
                    snapshot_id=snapshot_id,
                    query=("README setup guide getting started onboarding contributing"),
                    section=RetrievalSection.IMPORTANT_FILES,
                    mode=RetrievalMode.HYBRID,
                    max_results=30,
                )
            )
            n_chunks = len(bundle.evidences)
            prefix = f"{g_hint}\n\n" if g_hint else ""
            user_prompt = f"{prefix}snapshot_id={snapshot_id}\n\nEvidence:\n{render_bundle(bundle)}"
            data = await self._chat_json_typed(
                provider_id,
                model_id,
                AGENT_H_SYSTEM,
                user_prompt,
                AGENT_H_SCHEMA_STR,
                max_completion_tokens=4000,
            )
            raw_steps = data.get("steps")
            steps: list[dict[str, Any]] = []
            if isinstance(raw_steps, list):
                for item in raw_steps:
                    if not isinstance(item, dict) or "file" not in item:
                        continue
                    try:
                        order = int(item.get("order", 0))
                    except (TypeError, ValueError):
                        order = 0
                    steps.append(
                        {
                            "order": order,
                            "file": str(item.get("file", "") or ""),
                            "goal": str(item.get("goal", "") or ""),
                            "outcome": str(item.get("outcome", "") or ""),
                        }
                    )
            data["steps"] = steps
            try:
                raw_min = data.get("total_estimated_minutes")
                nmin = int(float(raw_min if raw_min is not None else 30))
                if nmin <= 0:
                    nmin = 30
                data["total_estimated_minutes"] = nmin
            except (TypeError, ValueError):
                data["total_estimated_minutes"] = 30
            for key in ("evidence_files", "blind_spots"):
                raw = data.get(key)
                if isinstance(raw, list):
                    data[key] = [str(x) for x in raw if x is not None]
                else:
                    data[key] = []
            data["confidence"] = _normalize_conf(str(data.get("confidence", "medium")))
            validate_section("H", data)
            ms = int((time.monotonic() - t0) * 1000)
            logger.info("[OnboardingAgent] %d chunks retrieved, completed in %dms", n_chunks, ms)
            return data
        except Exception as e:
            ms = int((time.monotonic() - t0) * 1000)
            logger.warning("[OnboardingAgent] failed in %dms: %s", ms, e)
            return self._fallback(str(e))
