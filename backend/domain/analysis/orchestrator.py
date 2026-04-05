"""Run Director Agent orchestration for analysis generation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from domain.model_connector.service import ProviderConfigService
from domain.model_connector.types import ChatMessage, ChatRequest
from domain.retrieval.service import RetrievalService

from .agent_pipeline import AnalysisAgentPipeline
from .prompts import DIRECTOR_SYSTEM, build_director_user_prompt
from .retrieval_broker import RetrievalBrokerAgent


@dataclass
class DirectorPlan:
    section_order: list[str]
    max_results: dict[str, int]
    notes: str


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


def _default_plan() -> DirectorPlan:
    return DirectorPlan(
        section_order=["architecture", "important_files", "feature_map", "conventions", "risk"],
        max_results={
            "architecture": 24,
            "conventions": 18,
            "feature_map": 22,
            "important_files": 20,
            "risk": 20,
        },
        notes="default plan",
    )


class RunDirectorAgent:
    def __init__(
        self,
        providers: ProviderConfigService,
        retrieval: RetrievalService,
        pipeline: AnalysisAgentPipeline,
    ) -> None:
        self._providers = providers
        self._pipeline = pipeline
        self._broker = RetrievalBrokerAgent(retrieval, providers)

    async def _plan(
        self,
        provider_id: str,
        model_id: str,
        snapshot_id: str,
        scan_mode: str,
    ) -> DirectorPlan:
        fallback = _default_plan()
        try:
            res = await self._providers.chat(
                ChatRequest(
                    provider_id=provider_id,
                    model_id=model_id,
                    messages=[
                        ChatMessage(role="system", content=DIRECTOR_SYSTEM),
                        ChatMessage(
                            role="user",
                            content=build_director_user_prompt(snapshot_id, scan_mode),
                        ),
                    ],
                    max_completion_tokens=520,
                    temperature=0.1,
                )
            )
            data = _safe_json(res.content)
            section_order_raw = data.get("section_order", fallback.section_order)
            section_order = [
                s.strip().lower()
                for s in section_order_raw
                if isinstance(s, str) and s.strip()
            ]
            section_order = [
                s
                for s in section_order
                if s in {"architecture", "conventions", "feature_map", "important_files", "risk"}
            ] or fallback.section_order

            raw_max = data.get("max_results", {})
            max_results = dict(fallback.max_results)
            if isinstance(raw_max, dict):
                for k, v in raw_max.items():
                    if k in max_results and isinstance(v, (int, float)):
                        max_results[k] = max(8, min(int(v), 40))
            notes = str(data.get("notes", "")).strip() or fallback.notes
            return DirectorPlan(section_order=section_order, max_results=max_results, notes=notes)
        except Exception:
            return fallback

    async def run(
        self,
        provider_id: str,
        model_id: str,
        snapshot_id: str,
        scan_mode: str,
    ) -> dict[str, Any]:
        plan = await self._plan(provider_id, model_id, snapshot_id, scan_mode)
        ctx = await self._broker.collect(
            snapshot_id=snapshot_id,
            provider_id=provider_id,
            model_id=model_id,
            scan_mode=scan_mode,
            section_order=plan.section_order,
            max_results=plan.max_results,
        )
        report = await self._pipeline.run(
            provider_id=provider_id,
            model_id=model_id,
            architecture=ctx.architecture,
            important=ctx.important,
            conventions=ctx.conventions,
            feature_map=ctx.feature_map,
            risk=ctx.risk,
        )
        report["orchestration"] = {
            "director": {
                "section_order": plan.section_order,
                "max_results": plan.max_results,
                "notes": plan.notes,
            },
            "broker_queries": ctx.queries,
        }
        return report
