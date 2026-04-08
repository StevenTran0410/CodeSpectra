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

from .agents._context_builders import PipelineMemoryContext, prefetch_pipeline_context
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
        "blind_spots": ["AuditAgent failed"],
    }



_COMPONENT_TO_SECTION: dict[str, str] = {
    "project_identity": "A",
    "architecture": "B",
    "structure": "C",
    "conventions": "D",
    "violations": "E",
    "feature_map": "F",
    "important_files": "G",
    "onboarding": "H",
    "glossary": "I",
    "risk": "J",
    "auditor": "K",
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
        identity_output: dict = None,
        architecture_output: dict = None,
        structure_output: dict = None,
        conventions_output: dict = None,
        violations_output: dict = None,
        feature_map_output: dict = None,
        important_files_output: dict = None,
        onboarding_output: dict = None,
        glossary_output: dict = None,
        risk_output: dict = None,
    ) -> dict[str, Any]:
        _ = (
            ctx,
            identity_output,
            architecture_output,
            structure_output,
            conventions_output,
            violations_output,
            feature_map_output,
            important_files_output,
            onboarding_output,
            glossary_output,
            risk_output,
        )
        raise NotImplementedError("Use run_async() for analysis components.")

    @component.output_types(output=dict)
    async def run_async(
        self,
        ctx: dict[str, Any],
        identity_output: dict = None,
        architecture_output: dict = None,
        structure_output: dict = None,
        conventions_output: dict = None,
        violations_output: dict = None,
        feature_map_output: dict = None,
        important_files_output: dict = None,
        onboarding_output: dict = None,
        glossary_output: dict = None,
        risk_output: dict = None,
    ) -> dict[str, Any]:
        deps = {
            "identity_output": identity_output,
            "architecture_output": architecture_output,
            "structure_output": structure_output,
            "conventions_output": conventions_output,
            "violations_output": violations_output,
            "feature_map_output": feature_map_output,
            "important_files_output": important_files_output,
            "onboarding_output": onboarding_output,
            "glossary_output": glossary_output,
            "risk_output": risk_output,
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
        self._project_identity = None
        self._architecture = None
        self._structure = None
        self._conventions = None
        self._violations = None
        self._feature_map = None
        self._important_files = None
        self._onboarding = None
        self._glossary = None
        self._risk = None
        self._auditor = None
        if retrieval_service is not None:
            from .agents import (
                ArchitectureAgent,
                AuditAgent,
                ConventionsAgent,
                FeatureMapAgent,
                GlossaryAgent,
                ImportantFilesAgent,
                OnboardingAgent,
                ProjectIdentityAgent,
                RiskAgent,
                StructureAgent,
                ViolationsAgent,
            )

            self._project_identity = ProjectIdentityAgent(provider_service, retrieval_service)
            self._architecture = ArchitectureAgent(provider_service, retrieval_service)
            self._structure = StructureAgent(provider_service, retrieval_service)
            self._conventions = ConventionsAgent(provider_service, retrieval_service)
            self._violations = ViolationsAgent(provider_service, retrieval_service)
            self._feature_map = FeatureMapAgent(provider_service, retrieval_service)
            self._important_files = ImportantFilesAgent(provider_service, retrieval_service)
            self._onboarding = OnboardingAgent(provider_service, retrieval_service)
            self._glossary = GlossaryAgent(provider_service, retrieval_service)
            self._risk = RiskAgent(provider_service, retrieval_service)
            self._auditor = AuditAgent(provider_service)
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
        if not (self._project_identity and snapshot_id):
            return {"version": REPORT_VERSION, "sections": sections}

        assert self._architecture is not None
        assert self._structure is not None
        assert self._conventions is not None
        assert self._violations is not None
        assert self._feature_map is not None
        assert self._important_files is not None
        assert self._onboarding is not None
        assert self._glossary is not None
        assert self._risk is not None
        assert self._auditor is not None

        ctx: dict[str, Any] = {
            "provider_id": provider_id,
            "model_id": model_id,
            "snapshot_id": snapshot_id,
            "repo_name": repo_name,
            "graph_summary": graph_summary,
            "static_risk": static_risk,
            "static_convention": static_convention,
        }
        mem_ctx: PipelineMemoryContext | None
        try:
            mem_ctx = await prefetch_pipeline_context(
                self._project_identity._retrieval, snapshot_id
            )
        except Exception as _prefetch_err:
            logger.warning(
                "[pipeline] prefetch_pipeline_context failed, agents will use own retrieval: %s",
                _prefetch_err,
            )
            mem_ctx = None
        ctx["mem_ctx"] = mem_ctx
        if mem_ctx is not None:
            logger.info(
                "[pipeline] PipelineMemoryContext ready: arch_bundle=%d chunks, "
                "folder_tree=%d chars, doc_files=%d chars, manifest_files=%d chars",
                len(mem_ctx.arch_bundle.evidences),
                len(mem_ctx.folder_tree),
                len(mem_ctx.doc_files),
                len(mem_ctx.manifest_files),
            )
        else:
            logger.warning(
                "[pipeline] PipelineMemoryContext unavailable — agents will use own retrieval"
            )
        pipeline = AsyncPipeline()
        pipeline.add_component(
            "project_identity",
            _SectionAgentComponent(
                "A",
                lambda c, _d: self._project_identity.run(
                    c["provider_id"],
                    c["model_id"],
                    c["snapshot_id"],
                    c["repo_name"],
                    mem_ctx=c.get("mem_ctx"),
                ),
                lambda c, _d: self._project_identity._fallback(
                    c["snapshot_id"], "pipeline", c["repo_name"]
                ),
                on_section_done,
            ),
        )
        pipeline.add_component(
            "architecture",
            _SectionAgentComponent(
                "B",
                lambda c, d: self._architecture.run(
                    c["provider_id"],
                    c["model_id"],
                    c["snapshot_id"],
                    c["graph_summary"],
                    arch_bundle=c["mem_ctx"].arch_bundle if c.get("mem_ctx") else None,
                    identity_output=d.get("identity_output"),
                ),
                lambda _c, _d: self._architecture._fallback("pipeline"),
                on_section_done,
            ),
        )
        pipeline.add_component(
            "structure",
            _SectionAgentComponent(
                "C",
                lambda c, d: self._structure.run(
                    c["provider_id"],
                    c["model_id"],
                    c["snapshot_id"],
                    arch_bundle=c["mem_ctx"].arch_bundle if c.get("mem_ctx") else None,
                    folder_tree=c["mem_ctx"].folder_tree if c.get("mem_ctx") else "",
                    identity_output=d.get("identity_output"),
                ),
                lambda _c, _d: self._structure._fallback("pipeline"),
                on_section_done,
            ),
        )
        pipeline.add_component(
            "conventions",
            _SectionAgentComponent(
                "D",
                lambda c, _d: self._conventions.run(
                    c["provider_id"],
                    c["model_id"],
                    c["snapshot_id"],
                    c["static_convention"],
                    None,
                ),
                lambda _c, _d: self._conventions._fallback("pipeline"),
                on_section_done,
            ),
        )
        pipeline.add_component(
            "violations",
            _SectionAgentComponent(
                "E",
                lambda c, d: self._violations.run(
                    c["provider_id"],
                    c["model_id"],
                    c["snapshot_id"],
                    c["static_convention"],
                    c["static_risk"],
                    d.get("conventions_output"),
                ),
                lambda _c, _d: self._violations._fallback("pipeline"),
                on_section_done,
            ),
        )
        pipeline.add_component(
            "feature_map",
            _SectionAgentComponent(
                "F",
                lambda c, d: self._feature_map.run(
                    c["provider_id"],
                    c["model_id"],
                    c["snapshot_id"],
                    c["graph_summary"],
                    identity_output=d.get("identity_output"),
                    architecture_output=d.get("architecture_output"),
                ),
                lambda _c, _d: self._feature_map._fallback("pipeline"),
                on_section_done,
            ),
        )
        pipeline.add_component(
            "important_files",
            _SectionAgentComponent(
                "G",
                lambda c, _d: self._important_files.run(
                    c["provider_id"], c["model_id"], c["snapshot_id"], c["graph_summary"]
                ),
                lambda _c, _d: self._important_files._fallback("pipeline", []),
                on_section_done,
            ),
        )
        pipeline.add_component(
            "onboarding",
            _SectionAgentComponent(
                "H",
                lambda c, d: self._onboarding.run(
                    c["provider_id"],
                    c["model_id"],
                    c["snapshot_id"],
                    d.get("important_files_output"),
                ),
                lambda _c, _d: self._onboarding._fallback("pipeline"),
                on_section_done,
            ),
        )
        pipeline.add_component(
            "glossary",
            _SectionAgentComponent(
                "I",
                lambda c, _d: self._glossary.run(c["provider_id"], c["model_id"], c["snapshot_id"]),
                lambda _c, _d: self._glossary._fallback("pipeline"),
                on_section_done,
            ),
        )
        pipeline.add_component(
            "risk",
            _SectionAgentComponent(
                "J",
                lambda c, _d: self._risk.run(
                    c["provider_id"], c["model_id"], c["snapshot_id"], c["static_risk"]
                ),
                lambda c, _d: self._risk._fallback("pipeline", c["static_risk"]),
                on_section_done,
            ),
        )
        pipeline.add_component(
            "auditor",
            _SectionAgentComponent(
                "K",
                lambda c, d: self._auditor.run(
                    c["provider_id"],
                    c["model_id"],
                    {
                        "A": d.get("identity_output"),
                        "B": d.get("architecture_output"),
                        "C": d.get("structure_output"),
                        "D": d.get("conventions_output"),
                        "E": d.get("violations_output"),
                        "F": d.get("feature_map_output"),
                        "G": d.get("important_files_output"),
                        "H": d.get("onboarding_output"),
                        "I": d.get("glossary_output"),
                        "J": d.get("risk_output"),
                    },
                ),
                lambda _c, _d: _section_k_pipeline_fallback(),
                on_section_done,
            ),
        )
        pipeline.connect("conventions.output", "violations.conventions_output")
        pipeline.connect("important_files.output", "onboarding.important_files_output")
        pipeline.connect("project_identity.output", "auditor.identity_output")
        pipeline.connect("architecture.output", "auditor.architecture_output")
        pipeline.connect("structure.output", "auditor.structure_output")
        pipeline.connect("conventions.output", "auditor.conventions_output")
        pipeline.connect("violations.output", "auditor.violations_output")
        pipeline.connect("feature_map.output", "auditor.feature_map_output")
        pipeline.connect("important_files.output", "auditor.important_files_output")
        pipeline.connect("onboarding.output", "auditor.onboarding_output")
        pipeline.connect("glossary.output", "auditor.glossary_output")
        pipeline.connect("risk.output", "auditor.risk_output")
        pipeline.connect("project_identity.output", "architecture.identity_output")
        pipeline.connect("project_identity.output", "structure.identity_output")
        pipeline.connect("project_identity.output", "feature_map.identity_output")
        pipeline.connect("architecture.output", "feature_map.architecture_output")
        names = {
            "project_identity", "architecture", "structure",
            "conventions", "violations", "feature_map",
            "important_files", "onboarding", "glossary",
            "risk", "auditor",
        }
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
                    section_letter = _COMPONENT_TO_SECTION.get(name)
                    if section_letter and isinstance(output, dict):
                        sections[section_letter] = output

        # Safety net: every section must exist even if Haystack wiring changes.
        if "A" not in sections:
            sections["A"] = self._project_identity._fallback(snapshot_id, "pipeline", repo_name)
        if "B" not in sections:
            sections["B"] = self._architecture._fallback("pipeline")
        if "C" not in sections:
            sections["C"] = self._structure._fallback("pipeline")
        if "D" not in sections:
            sections["D"] = self._conventions._fallback("pipeline")
        if "E" not in sections:
            sections["E"] = self._violations._fallback("pipeline")
        if "F" not in sections:
            sections["F"] = self._feature_map._fallback("pipeline")
        if "G" not in sections:
            sections["G"] = self._important_files._fallback("pipeline", [])
        if "H" not in sections:
            sections["H"] = self._onboarding._fallback("pipeline")
        if "I" not in sections:
            sections["I"] = self._glossary._fallback("pipeline")
        if "J" not in sections:
            sections["J"] = self._risk._fallback("pipeline", static_risk)
        if "K" not in sections:
            sections["K"] = _section_k_pipeline_fallback()

        return {"version": REPORT_VERSION, "sections": sections}
