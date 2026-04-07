"""LLM-powered analysis agent pipeline."""

from __future__ import annotations

import ast
import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

from domain.model_connector.service import ProviderConfigService
from domain.model_connector.types import ChatMessage, ChatRequest
from domain.retrieval.service import RetrievalService
from domain.structural_graph.types import StructuralGraphSummary
from shared.logger import logger

from .static_convention import ConventionReport
from .static_risk import RiskReport
from .types import SectionDoneCallback

REPORT_VERSION = 2


def _section_k_pipeline_fallback() -> dict[str, Any]:
    return {
        "overall_confidence": "low",
        "section_scores": {},
        "weakest_sections": [],
        "coverage_percentage": 0.0,
        "notes": "Audit could not be completed.",
        "blind_spots": ["AgentK failed"],
    }


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


async def _run_agent(
    label: str,
    agent: Any,
    coro: Awaitable[dict[str, Any]],
    on_section_done: SectionDoneCallback | None,
    fallback_fn: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Wraps a single agent coroutine for use inside asyncio.gather."""
    t0 = time.monotonic()
    data: dict[str, Any]
    status: str
    error_msg: str | None = None
    try:
        data = await coro
        status = "done"
    except Exception as exc:
        logger.error("[pipeline] agent %s failed: %s", label, exc, exc_info=True)
        data = fallback_fn()
        status = "error"
        error_msg = str(exc)
    duration_ms = int((time.monotonic() - t0) * 1000)
    if on_section_done is not None:
        try:
            await on_section_done(
                label,
                status,
                duration_ms,
                data if status == "done" else None,
                error_msg,
            )
        except Exception:
            logger.warning("[pipeline] on_section_done callback failed for %s", label)
    return data


class AnalysisAgentPipeline:
    def __init__(
        self,
        provider_service: ProviderConfigService,
        retrieval_service: RetrievalService | None = None,
    ) -> None:
        self._agent_a = None
        self._agent_b = None
        self._agent_c = None
        self._agent_d = None
        self._agent_e = None
        self._agent_f = None
        self._agent_g = None
        self._agent_h = None
        self._agent_i = None
        self._agent_j = None
        self._agent_k = None
        if retrieval_service is not None:
            from .agents import (
                AgentA,
                AgentB,
                AgentC,
                AgentD,
                AgentE,
                AgentF,
                AgentG,
                AgentH,
                AgentI,
                AgentJ,
                AgentK,
            )

            self._agent_a = AgentA(provider_service, retrieval_service)
            self._agent_b = AgentB(provider_service, retrieval_service)
            self._agent_c = AgentC(provider_service, retrieval_service)
            self._agent_d = AgentD(provider_service, retrieval_service)
            self._agent_e = AgentE(provider_service, retrieval_service)
            self._agent_f = AgentF(provider_service, retrieval_service)
            self._agent_g = AgentG(provider_service, retrieval_service)
            self._agent_h = AgentH(provider_service, retrieval_service)
            self._agent_i = AgentI(provider_service, retrieval_service)
            self._agent_j = AgentJ(provider_service, retrieval_service)
            self._agent_k = AgentK(provider_service)

    async def run(
        self,
        provider_id: str,
        model_id: str,
        *,
        snapshot_id: str = "",
        repo_name: str = "",
        graph_summary: StructuralGraphSummary | None = None,
        static_risk: RiskReport | None = None,
        static_convention: ConventionReport | None = None,
        on_section_done: SectionDoneCallback | None = None,
    ) -> dict[str, Any]:
        sections: dict[str, Any] = {}
        if not (self._agent_a and snapshot_id):
            return {"version": REPORT_VERSION, "sections": sections}

        assert self._agent_b is not None
        assert self._agent_c is not None
        assert self._agent_d is not None
        assert self._agent_e is not None
        assert self._agent_f is not None
        assert self._agent_g is not None
        assert self._agent_h is not None
        assert self._agent_i is not None
        assert self._agent_j is not None
        assert self._agent_k is not None

        wave_0_results = await asyncio.gather(
            _run_agent(
                "A",
                self._agent_a,
                self._agent_a.run(provider_id, model_id, snapshot_id, repo_name),
                on_section_done,
                lambda: self._agent_a._fallback(snapshot_id, "pipeline", repo_name),
            ),
            _run_agent(
                "B",
                self._agent_b,
                self._agent_b.run(provider_id, model_id, snapshot_id, graph_summary),
                on_section_done,
                lambda: self._agent_b._fallback("pipeline"),
            ),
            _run_agent(
                "C",
                self._agent_c,
                self._agent_c.run(provider_id, model_id, snapshot_id),
                on_section_done,
                lambda: self._agent_c._fallback("pipeline"),
            ),
            _run_agent(
                "D",
                self._agent_d,
                self._agent_d.run(
                    provider_id, model_id, snapshot_id, static_convention, None
                ),
                on_section_done,
                lambda: self._agent_d._fallback("pipeline"),
            ),
            _run_agent(
                "F",
                self._agent_f,
                self._agent_f.run(provider_id, model_id, snapshot_id, graph_summary),
                on_section_done,
                lambda: self._agent_f._fallback("pipeline"),
            ),
            _run_agent(
                "G",
                self._agent_g,
                self._agent_g.run(provider_id, model_id, snapshot_id, graph_summary),
                on_section_done,
                lambda: self._agent_g._fallback("pipeline", []),
            ),
            _run_agent(
                "I",
                self._agent_i,
                self._agent_i.run(provider_id, model_id, snapshot_id),
                on_section_done,
                lambda: self._agent_i._fallback("pipeline"),
            ),
            _run_agent(
                "J",
                self._agent_j,
                self._agent_j.run(provider_id, model_id, snapshot_id, static_risk),
                on_section_done,
                lambda: self._agent_j._fallback("pipeline", static_risk),
            ),
            return_exceptions=True,
        )

        wave_0_letters = ("A", "B", "C", "D", "F", "G", "I", "J")
        fallbacks_w0 = (
            lambda: self._agent_a._fallback(snapshot_id, "pipeline", repo_name),
            lambda: self._agent_b._fallback("pipeline"),
            lambda: self._agent_c._fallback("pipeline"),
            lambda: self._agent_d._fallback("pipeline"),
            lambda: self._agent_f._fallback("pipeline"),
            lambda: self._agent_g._fallback("pipeline", []),
            lambda: self._agent_i._fallback("pipeline"),
            lambda: self._agent_j._fallback("pipeline", static_risk),
        )
        for letter, res, fb in zip(
            wave_0_letters, wave_0_results, fallbacks_w0, strict=True
        ):
            sections[letter] = res if isinstance(res, dict) else fb()

        wave_1_results = await asyncio.gather(
            _run_agent(
                "E",
                self._agent_e,
                self._agent_e.run(
                    provider_id,
                    model_id,
                    snapshot_id,
                    static_convention,
                    static_risk,
                    sections.get("D"),
                ),
                on_section_done,
                lambda: self._agent_e._fallback("pipeline"),
            ),
            _run_agent(
                "H",
                self._agent_h,
                self._agent_h.run(
                    provider_id, model_id, snapshot_id, sections.get("G")
                ),
                on_section_done,
                lambda: self._agent_h._fallback("pipeline"),
            ),
            return_exceptions=True,
        )
        e_res, h_res = wave_1_results
        sections["E"] = (
            e_res if isinstance(e_res, dict) else self._agent_e._fallback("pipeline")
        )
        sections["H"] = (
            h_res if isinstance(h_res, dict) else self._agent_h._fallback("pipeline")
        )

        sections["K"] = await _run_agent(
            "K",
            self._agent_k,
            self._agent_k.run(provider_id, model_id, sections),
            on_section_done,
            _section_k_pipeline_fallback,
        )

        return {"version": REPORT_VERSION, "sections": sections}
