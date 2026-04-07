"""Coding conventions agent (section D)."""

from __future__ import annotations

import time
from typing import Any

from domain.model_connector.service import ProviderConfigService
from domain.retrieval.service import RetrievalService
from domain.retrieval.types import RetrievalMode, RetrievalSection, RetrieveRequest
from shared.logger import logger

from ..agent_pipeline import _normalize_conf
from ..prompts import AGENT_D_SCHEMA_STR, AGENT_D_SYSTEM, render_bundle
from ..schemas import validate_section
from ..static_convention import ConventionReport
from ._context_builders import build_convention_block
from .base import BaseTypedAgent

_ASPECT_KEYS = (
    "naming_style",
    "error_handling",
    "async_style",
    "di_style",
    "class_vs_functional",
    "test_style",
)

_COMBINED_QUERY = (
    "async await error handling dependency injection "
    "naming convention class function import style "
    "test pattern fixture mock assert"
)


def _empty_aspect() -> dict[str, list[str] | str]:
    return {"description": "", "evidence_files": []}


def _coerce_aspect(raw: Any) -> dict[str, list[str] | str]:
    if isinstance(raw, dict):
        desc = str(raw.get("description", "") or "")
        ev = raw.get("evidence_files")
        files = [str(x) for x in ev if x is not None] if isinstance(ev, list) else []
        return {"description": desc, "evidence_files": files}
    if raw is None:
        return _empty_aspect()
    return {"description": str(raw), "evidence_files": []}


class AgentD(BaseTypedAgent):
    def __init__(
        self,
        provider_service: ProviderConfigService,
        retrieval_service: RetrievalService,
    ) -> None:
        super().__init__(provider_service)
        self._retrieval = retrieval_service

    def _fallback(self, reason: str) -> dict[str, Any]:
        empty = _empty_aspect()
        return {
            "naming_style": dict(empty),
            "error_handling": dict(empty),
            "async_style": dict(empty),
            "di_style": dict(empty),
            "class_vs_functional": dict(empty),
            "test_style": dict(empty),
            "signals": [],
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
        section_c_output: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        t0 = time.monotonic()
        n_chunks = 0
        try:
            prefix_parts: list[str] = []
            conv = build_convention_block(static_convention)
            if conv:
                prefix_parts.append(conv)
            if section_c_output:
                raw_folders = section_c_output.get("folders")
                if isinstance(raw_folders, list) and raw_folders:
                    lines = ["Folder roles from structure agent:"]
                    for folder in raw_folders[:8]:
                        if isinstance(folder, dict):
                            lines.append(
                                f"  {folder.get('path', '')} → {folder.get('role', '')}"
                            )
                    prefix_parts.append("\n".join(lines))
            prefix = "\n\n".join(prefix_parts) + ("\n\n" if prefix_parts else "")

            bundle = await self._retrieval.retrieve(
                RetrieveRequest(
                    snapshot_id=snapshot_id,
                    query=_COMBINED_QUERY,
                    section=RetrievalSection.CONVENTIONS,
                    mode=RetrievalMode.HYBRID,
                    max_results=20,
                )
            )
            n_chunks = len(bundle.evidences)
            user_prompt = (
                f"{prefix}snapshot_id={snapshot_id}\n\nEvidence:\n{render_bundle(bundle)}"
            )
            data = await self._chat_json_typed(
                provider_id,
                model_id,
                AGENT_D_SYSTEM,
                user_prompt,
                schema_hint=AGENT_D_SCHEMA_STR,
                max_completion_tokens=16000,
            )
            for k in _ASPECT_KEYS:
                data[k] = _coerce_aspect(data.get(k))
            raw_sig = data.get("signals")
            sigs: list[dict[str, str]] = []
            if isinstance(raw_sig, list):
                for i in raw_sig:
                    if not isinstance(i, dict):
                        continue
                    cat = str(i.get("category", "") or "")
                    desc = str(i.get("description", "") or "")
                    pat = str(i.get("pattern", "") or "")
                    if pat and not desc:
                        desc = pat
                    ev_files = i.get("evidence_files")
                    ev_str = ""
                    if isinstance(ev_files, list) and ev_files:
                        ev_str = ", ".join(str(x) for x in ev_files[:8] if x is not None)
                    elif i.get("evidence") is not None:
                        ev_str = str(i.get("evidence", "") or "")
                    sigs.append(
                        {"category": cat, "description": desc, "evidence": ev_str}
                    )
            data["signals"] = sigs
            for key in ("evidence_files", "blind_spots"):
                raw = data.get(key)
                if isinstance(raw, list):
                    data[key] = [str(x) for x in raw if x is not None]
                else:
                    data[key] = []
            data["confidence"] = _normalize_conf(str(data.get("confidence", "medium")))
            validate_section("D", data)
            ms = int((time.monotonic() - t0) * 1000)
            logger.info("[AgentD] %d chunks retrieved, completed in %dms", n_chunks, ms)
            return data
        except Exception as e:
            ms = int((time.monotonic() - t0) * 1000)
            logger.warning("[AgentD] failed in %dms: %s", ms, e)
            return self._fallback(str(e))
