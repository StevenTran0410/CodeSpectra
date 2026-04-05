"""Retrieval Broker Agent: plans and fetches section evidence packs."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from domain.model_connector.service import ProviderConfigService
from domain.model_connector.types import ChatMessage, ChatRequest
from domain.retrieval.service import RetrievalService
from domain.retrieval.types import (
    RetrievalBundle,
    RetrievalEvidence,
    RetrievalMode,
    RetrievalSection,
    RetrieveRequest,
)

from .prompts import BROKER_SYSTEM, build_broker_user_prompt

_DEFAULT_QUERIES: dict[str, list[str]] = {
    "architecture": [
        "system architecture layers entrypoints modules integrations",
        "app startup routing container dependency boundaries",
    ],
    "conventions": [
        "coding conventions naming error handling dependency style",
        "testing style logging validation rule patterns",
    ],
    "feature_map": [
        "feature map functionality modules services data flow",
        "business domain workflows use cases responsibilities",
    ],
    "important_files": [
        "important files entrypoint backbone central files",
        "core service router manager orchestrator files",
    ],
    "risk": [
        "risk complexity hotspot TODO FIXME circular import",
        "technical debt fragile coupling bottleneck unsafe patterns",
    ],
}

_SECTION_SOURCE: dict[str, RetrievalSection] = {
    "architecture": RetrievalSection.ARCHITECTURE,
    "conventions": RetrievalSection.CONVENTIONS,
    "feature_map": RetrievalSection.FEATURE_MAP,
    "important_files": RetrievalSection.IMPORTANT_FILES,
    "risk": RetrievalSection.IMPORTANT_FILES,
}


@dataclass
class RetrievalContext:
    architecture: RetrievalBundle
    conventions: RetrievalBundle
    feature_map: RetrievalBundle
    important: RetrievalBundle
    risk: RetrievalBundle
    queries: dict[str, list[str]]


def _normalize_section_order(order: list[str]) -> list[str]:
    valid = ["architecture", "conventions", "feature_map", "important_files", "risk"]
    seen: set[str] = set()
    out: list[str] = []
    for item in order:
        key = (item or "").strip().lower()
        if key in valid and key not in seen:
            seen.add(key)
            out.append(key)
    for item in valid:
        if item not in seen:
            out.append(item)
    return out


def _safe_json(s: str) -> dict[str, Any]:
    text = (s or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Expected JSON object")
    return data


def _dedupe_evidence(bundles: list[RetrievalBundle], max_results: int) -> list[RetrievalEvidence]:
    picked: dict[str, RetrievalEvidence] = {}
    for b in bundles:
        for ev in b.evidences:
            cur = picked.get(ev.chunk_id)
            if cur is None or ev.score > cur.score:
                picked[ev.chunk_id] = ev
    out = sorted(picked.values(), key=lambda e: (-e.score, e.rel_path, e.chunk_index))
    return out[: max(1, min(max_results, 80))]


class RetrievalBrokerAgent:
    def __init__(self, retrieval: RetrievalService, providers: ProviderConfigService) -> None:
        self._retrieval = retrieval
        self._providers = providers

    async def _plan_queries(
        self,
        snapshot_id: str,
        section_order: list[str],
        provider_id: str,
        model_id: str,
    ) -> dict[str, list[str]]:
        try:
            res = await self._providers.chat(
                ChatRequest(
                    provider_id=provider_id,
                    model_id=model_id,
                    messages=[
                        ChatMessage(role="system", content=BROKER_SYSTEM),
                        ChatMessage(
                            role="user",
                            content=build_broker_user_prompt(snapshot_id, section_order),
                        ),
                    ],
                    max_completion_tokens=650,
                    temperature=0.15,
                )
            )
            raw = _safe_json(res.content)
            out: dict[str, list[str]] = {}
            for key, defaults in _DEFAULT_QUERIES.items():
                arr = raw.get(key)
                if isinstance(arr, list):
                    queries = [
                        q.strip()
                        for q in arr
                        if isinstance(q, str) and q.strip()
                    ][:3]
                    out[key] = queries or defaults
                else:
                    out[key] = defaults
            return out
        except Exception:
            return dict(_DEFAULT_QUERIES)

    async def _retrieve_section(
        self,
        snapshot_id: str,
        section_key: str,
        queries: list[str],
        max_results: int,
        scan_mode: str,
    ) -> RetrievalBundle:
        source_section = _SECTION_SOURCE[section_key]
        mode = RetrievalMode.HYBRID
        if section_key in {"important_files", "risk"} or scan_mode == "quick":
            mode = RetrievalMode.VECTORLESS

        bundles: list[RetrievalBundle] = []
        for q in queries[:3]:
            bundles.append(
                await self._retrieval.retrieve(
                    RetrieveRequest(
                        snapshot_id=snapshot_id,
                        query=q,
                        section=source_section,
                        mode=mode,
                        max_results=max_results,
                    )
                )
            )

        merged = _dedupe_evidence(bundles, max_results=max_results)
        budget = max((b.budget_tokens for b in bundles), default=2200)
        used = sum(e.token_estimate for e in merged)
        return RetrievalBundle(
            snapshot_id=snapshot_id,
            mode=mode,
            section=source_section,
            query=" || ".join(queries[:3]),
            budget_tokens=budget,
            used_tokens=min(used, budget),
            evidences=merged,
        )

    async def collect(
        self,
        snapshot_id: str,
        provider_id: str,
        model_id: str,
        scan_mode: str,
        section_order: list[str],
        max_results: dict[str, int],
    ) -> RetrievalContext:
        normalized_order = _normalize_section_order(section_order)
        query_map = await self._plan_queries(
            snapshot_id=snapshot_id,
            section_order=normalized_order,
            provider_id=provider_id,
            model_id=model_id,
        )

        section_bundles: dict[str, RetrievalBundle] = {}
        for key in normalized_order:
            section_bundles[key] = await self._retrieve_section(
                snapshot_id=snapshot_id,
                section_key=key,
                queries=query_map.get(key, _DEFAULT_QUERIES[key]),
                max_results=max(8, min(int(max_results.get(key, 20)), 40)),
                scan_mode=scan_mode,
            )

        return RetrievalContext(
            architecture=section_bundles["architecture"],
            conventions=section_bundles["conventions"],
            feature_map=section_bundles["feature_map"],
            important=section_bundles["important_files"],
            risk=section_bundles["risk"],
            queries=query_map,
        )
