"""LLM-powered analysis agent pipeline."""

from __future__ import annotations

import ast
import json
import os
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

from haystack import AsyncPipeline, component

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


_SECTION_LETTERS = tuple("ABCDEFGHIJK")


def _default_concurrency() -> int:
    raw = os.getenv("ANALYSIS_PIPELINE_CONCURRENCY", "11").strip()
    try:
        return max(1, int(raw))
    except Exception:
        return 11


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


SectionRunner = Callable[[dict[str, Any], dict[str, Any]], Awaitable[dict[str, Any]]]
SectionFallback = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


@component
class _SectionAgentComponent:
    """Haystack async component wrapper around one section agent."""

    __haystack_supports_async__ = True

    def __init__(
        self,
        section: str,
        runner: SectionRunner,
        fallback: SectionFallback,
        on_section_done: SectionDoneCallback | None,
    ) -> None:
        self._section = section
        self._runner = runner
        self._fallback = fallback
        self._on_section_done = on_section_done

    @component.output_types(output=dict)
    def run(
        self,
        ctx: dict[str, Any],
        a_output: dict = None,
        b_output: dict = None,
        c_output: dict = None,
        d_output: dict = None,
        e_output: dict = None,
        f_output: dict = None,
        g_output: dict = None,
        h_output: dict = None,
        i_output: dict = None,
        j_output: dict = None,
    ) -> dict[str, Any]:
        _ = (
            ctx,
            a_output,
            b_output,
            c_output,
            d_output,
            e_output,
            f_output,
            g_output,
            h_output,
            i_output,
            j_output,
        )
        raise NotImplementedError("Use run_async() for analysis components.")

    @component.output_types(output=dict)
    async def run_async(
        self,
        ctx: dict[str, Any],
        a_output: dict = None,
        b_output: dict = None,
        c_output: dict = None,
        d_output: dict = None,
        e_output: dict = None,
        f_output: dict = None,
        g_output: dict = None,
        h_output: dict = None,
        i_output: dict = None,
        j_output: dict = None,
    ) -> dict[str, Any]:
        deps = {
            "a_output": a_output,
            "b_output": b_output,
            "c_output": c_output,
            "d_output": d_output,
            "e_output": e_output,
            "f_output": f_output,
            "g_output": g_output,
            "h_output": h_output,
            "i_output": i_output,
            "j_output": j_output,
        }
        t0 = time.monotonic()
        status = "done"
        error_msg: str | None = None
        try:
            output = await self._runner(ctx, deps)
        except Exception as exc:
            logger.error("[pipeline] agent %s failed: %s", self._section, exc, exc_info=True)
            output = self._fallback(ctx, deps)
            status = "error"
            error_msg = str(exc)
        duration_ms = int((time.monotonic() - t0) * 1000)
        if self._on_section_done is not None:
            try:
                await self._on_section_done(
                    self._section,
                    status,
                    duration_ms,
                    output if status == "done" else None,
                    error_msg,
                )
            except Exception:
                logger.warning(
                    "[pipeline] on_section_done callback failed for %s",
                    self._section,
                )
        return {"output": output}


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
        self._concurrency_limit = _default_concurrency()

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

        ctx = {
            "provider_id": provider_id,
            "model_id": model_id,
            "snapshot_id": snapshot_id,
            "repo_name": repo_name,
            "graph_summary": graph_summary,
            "static_risk": static_risk,
            "static_convention": static_convention,
        }
        pipeline = AsyncPipeline()
        pipeline.add_component(
            "a",
            _SectionAgentComponent(
                "A",
                lambda c, _d: self._agent_a.run(
                    c["provider_id"], c["model_id"], c["snapshot_id"], c["repo_name"]
                ),
                lambda c, _d: self._agent_a._fallback(
                    c["snapshot_id"], "pipeline", c["repo_name"]
                ),
                on_section_done,
            ),
        )
        pipeline.add_component(
            "b",
            _SectionAgentComponent(
                "B",
                lambda c, _d: self._agent_b.run(
                    c["provider_id"], c["model_id"], c["snapshot_id"], c["graph_summary"]
                ),
                lambda _c, _d: self._agent_b._fallback("pipeline"),
                on_section_done,
            ),
        )
        pipeline.add_component(
            "c",
            _SectionAgentComponent(
                "C",
                lambda c, _d: self._agent_c.run(
                    c["provider_id"], c["model_id"], c["snapshot_id"]
                ),
                lambda _c, _d: self._agent_c._fallback("pipeline"),
                on_section_done,
            ),
        )
        pipeline.add_component(
            "d",
            _SectionAgentComponent(
                "D",
                lambda c, _d: self._agent_d.run(
                    c["provider_id"],
                    c["model_id"],
                    c["snapshot_id"],
                    c["static_convention"],
                    None,
                ),
                lambda _c, _d: self._agent_d._fallback("pipeline"),
                on_section_done,
            ),
        )
        pipeline.add_component(
            "e",
            _SectionAgentComponent(
                "E",
                lambda c, d: self._agent_e.run(
                    c["provider_id"],
                    c["model_id"],
                    c["snapshot_id"],
                    c["static_convention"],
                    c["static_risk"],
                    d.get("d_output"),
                ),
                lambda _c, _d: self._agent_e._fallback("pipeline"),
                on_section_done,
            ),
        )
        pipeline.add_component(
            "f",
            _SectionAgentComponent(
                "F",
                lambda c, _d: self._agent_f.run(
                    c["provider_id"], c["model_id"], c["snapshot_id"], c["graph_summary"]
                ),
                lambda _c, _d: self._agent_f._fallback("pipeline"),
                on_section_done,
            ),
        )
        pipeline.add_component(
            "g",
            _SectionAgentComponent(
                "G",
                lambda c, _d: self._agent_g.run(
                    c["provider_id"], c["model_id"], c["snapshot_id"], c["graph_summary"]
                ),
                lambda _c, _d: self._agent_g._fallback("pipeline", []),
                on_section_done,
            ),
        )
        pipeline.add_component(
            "h",
            _SectionAgentComponent(
                "H",
                lambda c, d: self._agent_h.run(
                    c["provider_id"], c["model_id"], c["snapshot_id"], d.get("g_output")
                ),
                lambda _c, _d: self._agent_h._fallback("pipeline"),
                on_section_done,
            ),
        )
        pipeline.add_component(
            "i",
            _SectionAgentComponent(
                "I",
                lambda c, _d: self._agent_i.run(
                    c["provider_id"], c["model_id"], c["snapshot_id"]
                ),
                lambda _c, _d: self._agent_i._fallback("pipeline"),
                on_section_done,
            ),
        )
        pipeline.add_component(
            "j",
            _SectionAgentComponent(
                "J",
                lambda c, _d: self._agent_j.run(
                    c["provider_id"], c["model_id"], c["snapshot_id"], c["static_risk"]
                ),
                lambda c, _d: self._agent_j._fallback("pipeline", c["static_risk"]),
                on_section_done,
            ),
        )
        pipeline.add_component(
            "k",
            _SectionAgentComponent(
                "K",
                lambda c, d: self._agent_k.run(
                    c["provider_id"],
                    c["model_id"],
                    {
                        "A": d.get("a_output"),
                        "B": d.get("b_output"),
                        "C": d.get("c_output"),
                        "D": d.get("d_output"),
                        "E": d.get("e_output"),
                        "F": d.get("f_output"),
                        "G": d.get("g_output"),
                        "H": d.get("h_output"),
                        "I": d.get("i_output"),
                        "J": d.get("j_output"),
                    },
                ),
                lambda _c, _d: _section_k_pipeline_fallback(),
                on_section_done,
            ),
        )
        pipeline.connect("d.output", "e.d_output")
        pipeline.connect("g.output", "h.g_output")
        pipeline.connect("a.output", "k.a_output")
        pipeline.connect("b.output", "k.b_output")
        pipeline.connect("c.output", "k.c_output")
        pipeline.connect("d.output", "k.d_output")
        pipeline.connect("e.output", "k.e_output")
        pipeline.connect("f.output", "k.f_output")
        pipeline.connect("g.output", "k.g_output")
        pipeline.connect("h.output", "k.h_output")
        pipeline.connect("i.output", "k.i_output")
        pipeline.connect("j.output", "k.j_output")
        names = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"}
        data = {name: {"ctx": ctx} for name in names}
        async for partial in pipeline.run_async_generator(
            data=data,
            include_outputs_from=names,
            concurrency_limit=self._concurrency_limit,
        ):
            for name, output_map in partial.items():
                if name not in names or not isinstance(output_map, dict):
                    continue
                output = output_map.get("output")
                if isinstance(output, dict):
                    sections[name.upper()] = output

        # Safety net: every section must exist even if Haystack wiring changes.
        if "A" not in sections:
            sections["A"] = self._agent_a._fallback(snapshot_id, "pipeline", repo_name)
        if "B" not in sections:
            sections["B"] = self._agent_b._fallback("pipeline")
        if "C" not in sections:
            sections["C"] = self._agent_c._fallback("pipeline")
        if "D" not in sections:
            sections["D"] = self._agent_d._fallback("pipeline")
        if "E" not in sections:
            sections["E"] = self._agent_e._fallback("pipeline")
        if "F" not in sections:
            sections["F"] = self._agent_f._fallback("pipeline")
        if "G" not in sections:
            sections["G"] = self._agent_g._fallback("pipeline", [])
        if "H" not in sections:
            sections["H"] = self._agent_h._fallback("pipeline")
        if "I" not in sections:
            sections["I"] = self._agent_i._fallback("pipeline")
        if "J" not in sections:
            sections["J"] = self._agent_j._fallback("pipeline", static_risk)
        if "K" not in sections:
            sections["K"] = _section_k_pipeline_fallback()

        return {"version": REPORT_VERSION, "sections": sections}
