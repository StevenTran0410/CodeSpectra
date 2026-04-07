"""LLM-powered analysis agent pipeline."""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from domain.model_connector.service import ProviderConfigService
from domain.model_connector.types import ChatMessage, ChatRequest
from domain.retrieval.service import RetrievalService
from domain.structural_graph.types import StructuralGraphSummary
from shared.logger import logger

from .static_risk import RiskReport


def _normalize_conf(v: str) -> str:
    t = (v or "").strip().lower()
    if t in {"high", "medium", "low"}:
        return t
    return "medium"


class BaseLLMAgent:
    def __init__(self, provider_service: ProviderConfigService) -> None:
        self._providers = provider_service

    @staticmethod
    def _try_parse_json(text: str) -> dict[str, Any]:
        raw = (text or "").strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            inner = "\n".join(
                line
                for line in lines
                if not line.strip().startswith("```") and line.strip() != "json"
            ).strip()
            raw = inner if inner else raw.strip("`").lstrip("json").strip()
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            block = m.group(0).strip()
            try:
                obj = json.loads(block)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                try:
                    obj = ast.literal_eval(block)
                    if isinstance(obj, dict):
                        return json.loads(json.dumps(obj))
                except Exception:
                    pass
        raise ValueError("invalid_json_output")

    @staticmethod
    def _is_valid_section_output(data: dict[str, Any]) -> bool:
        content = str(data.get("content", "")).strip()
        if not content:
            return False
        if data.get("confidence") not in {"high", "medium", "low"}:
            return False
        return True

    async def _call(
        self,
        provider_id: str,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
        max_completion_tokens: int,
        temperature: float | None = 0.2,
        json_mode: bool = True,
    ) -> str:
        res = await self._providers.chat(
            ChatRequest(
                provider_id=provider_id,
                model_id=model_id,
                messages=[
                    ChatMessage(role="system", content=system_prompt),
                    ChatMessage(role="user", content=user_prompt),
                ],
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                json_mode=json_mode,
                stream=False,
            )
        )
        return (res.content or "").strip()

    async def _chat_json(
        self,
        provider_id: str,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
        max_completion_tokens: int = 1200,
    ) -> dict[str, Any]:
        text = await self._call(
            provider_id,
            model_id,
            system_prompt,
            user_prompt,
            max_completion_tokens,
            temperature=0.2,
        )
        try:
            obj = self._try_parse_json(text)
            if self._is_valid_section_output(obj):
                return obj
        except Exception:
            pass

        logger.warning("LLM agent: attempt 1 bad output, asking model to extract fields from prose")

        repair_user = (
            "The following is your previous output. Extract the required JSON fields from it.\n"
            "If a field is missing or unclear, use a reasonable default.\n"
            "Return ONLY valid JSON, no markdown fence, no commentary.\n"
            'Required schema: {"content": "string", "confidence": "high|medium|low", '
            '"evidence_files": [], "blind_spots": [], "details": {}}\n\n'
            f"Previous output:\n{text}"
        )
        text2 = await self._call(
            provider_id,
            model_id,
            system_prompt,
            repair_user,
            max_completion_tokens,
            temperature=None,
        )
        try:
            obj = self._try_parse_json(text2)
            if obj.get("content") and obj.get("confidence"):
                return obj
        except Exception:
            pass

        raise ValueError(f"all_attempts_failed: last_output={text[:120]!r}")


class AnalysisAgentPipeline:
    def __init__(
        self,
        provider_service: ProviderConfigService,
        retrieval_service: RetrievalService | None = None,
    ) -> None:
        self._agent_a = None
        self._agent_b = None
        self._agent_c = None
        self._agent_g = None
        self._agent_h = None
        self._agent_i = None
        self._agent_j = None
        if retrieval_service is not None:
            from .agents import AgentA, AgentB, AgentC, AgentG, AgentH, AgentI, AgentJ

            self._agent_a = AgentA(provider_service, retrieval_service)
            self._agent_b = AgentB(provider_service, retrieval_service)
            self._agent_c = AgentC(provider_service, retrieval_service)
            self._agent_g = AgentG(provider_service, retrieval_service)
            self._agent_h = AgentH(provider_service, retrieval_service)
            self._agent_i = AgentI(provider_service, retrieval_service)
            self._agent_j = AgentJ(provider_service, retrieval_service)

    async def run(
        self,
        provider_id: str,
        model_id: str,
        *,
        snapshot_id: str = "",
        repo_name: str = "",
        graph_summary: StructuralGraphSummary | None = None,
        static_risk: RiskReport | None = None,
    ) -> dict[str, Any]:
        sections_v2: dict[str, Any] = {}
        if not (self._agent_a and snapshot_id):
            return {"sections_v2": sections_v2}

        assert self._agent_b is not None
        assert self._agent_c is not None
        assert self._agent_g is not None
        assert self._agent_h is not None
        assert self._agent_i is not None
        assert self._agent_j is not None

        # Wave 1 — baseline agents (no inter-agent dependencies)
        sections_v2["A"] = await self._agent_a.run(provider_id, model_id, snapshot_id, repo_name)
        sections_v2["G"] = await self._agent_g.run(
            provider_id, model_id, snapshot_id, graph_summary
        )
        sections_v2["I"] = await self._agent_i.run(provider_id, model_id, snapshot_id)
        sections_v2["J"] = await self._agent_j.run(provider_id, model_id, snapshot_id, static_risk)

        # Wave 2 — structure agents (B, C independent; H needs G)
        sections_v2["B"] = await self._agent_b.run(
            provider_id, model_id, snapshot_id, graph_summary
        )
        sections_v2["C"] = await self._agent_c.run(provider_id, model_id, snapshot_id)
        sections_v2["H"] = await self._agent_h.run(
            provider_id, model_id, snapshot_id, sections_v2.get("G")
        )

        return {"sections_v2": sections_v2}
