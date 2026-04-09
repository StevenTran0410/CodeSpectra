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
from .profiles import get_profile
from .static_convention import ConventionReport
from .static_risk import RiskReport
from .types import SectionDoneCallback

REPORT_VERSION = 3

# Maximum number of per-section re-runs in large_codebase_mode quality gate
_LARGE_MODE_MAX_RETRIES: int = 3
# Sections eligible for quality-gate retry (no retrieval agents K/L are excluded as they
# get re-run automatically after the improved base sections)
_RETRYABLE_SECTIONS = frozenset("ABCDEFGHIJ")


def _section_k_pipeline_fallback() -> dict[str, Any]:
    return {
        "overall_confidence": "low",
        "section_scores": {},
        "weakest_sections": [],
        "coverage_percentage": 0.0,
        "notes": "Audit could not be completed.",
        "blind_spots": ["AuditAgent failed"],
    }


def _section_l_pipeline_fallback() -> dict[str, Any]:
    return {
        "executive_summary": "",
        "architecture_narrative": "",
        "tech_stack_snapshot": "",
        "developer_quickstart": "",
        "conventions_digest": "",
        "risk_highlights": "",
        "reading_path": "",
        "confidence": "low",
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
    "synthesizer": "L",
}
_SECTION_LETTERS = tuple("ABCDEFGHIJKL")


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

    def _output_ok(self, obj: dict[str, Any], schema_hint: str) -> bool:
        if schema_hint:
            return isinstance(obj, dict) and bool(obj)
        return self._is_valid_section_output(obj)

    async def _chat_json(
        self,
        provider_id: str,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
        max_completion_tokens: int = 1200,
        schema_hint: str = "",
    ) -> dict[str, Any]:
        agent_name = self.__class__.__name__
        text = await self._call(
            provider_id,
            model_id,
            system_prompt,
            user_prompt,
            max_completion_tokens,
            temperature=0.2,
        )

        reason1 = "attempt1_failed"
        try:
            obj = self._try_parse_json(text)
            if self._output_ok(obj, schema_hint):
                return obj
            reason1 = "validation_failed"
        except Exception as exc:
            reason1 = str(exc) or f"{type(exc).__name__}"

        logger.warning("[%s] output_repair attempt=2 reason=%s", agent_name, reason1)

        if not text.strip():
            retry_system = (
                system_prompt + "\n\nCRITICAL: You MUST respond with a JSON object. "
                "Start your response with { and end with }. No prose."
            )
            text2 = await self._call(
                provider_id,
                model_id,
                retry_system,
                user_prompt,
                max_completion_tokens * 2,
                temperature=0.1,
            )
        else:
            if schema_hint:
                repair_user = (
                    "Your previous output was not valid JSON or did not match the schema.\n"
                    f"Error: {reason1}\n"
                    "Return ONLY valid JSON, no markdown fence, no commentary.\n"
                    f"Required schema:\n{schema_hint}\n\n"
                    f"Previous output:\n{text}"
                )
            else:
                hint_tail = f"\n\nAdditional hint:\n{schema_hint}" if schema_hint else ""
                repair_user = (
                    "Your previous output was not valid JSON.\n"
                    f"Parse/validation error: {reason1}\n"
                    "Return ONLY valid JSON starting with { and ending with }.\n"
                    "Required keys: content (str), confidence (high|medium|low), "
                    "evidence_files (list), blind_spots (list), details (dict).\n"
                    f"{hint_tail}\n\n"
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

        reason2 = "attempt2_failed"
        try:
            obj2 = self._try_parse_json(text2)
            if self._output_ok(obj2, schema_hint):
                return obj2
            if not schema_hint and obj2.get("content") and obj2.get("confidence"):
                return obj2
            reason2 = "validation_failed_after_repair"
        except Exception as exc:
            reason2 = str(exc) or f"{type(exc).__name__}"

        logger.warning("[%s] output_repair attempt=3 reason=%s", agent_name, reason2)

        fallback_prompt = (
            'Return exactly this JSON object and nothing else: '
            '{"content": "", "confidence": "low", "evidence_files": [], '
            '"blind_spots": ["output_repair_failed"], "details": {}}'
        )
        try:
            text3 = await self._call(
                provider_id,
                model_id,
                system_prompt,
                fallback_prompt,
                max_completion_tokens,
                temperature=0.0,
            )
            try:
                obj3 = self._try_parse_json(text3)
                if isinstance(obj3, dict):
                    return obj3
            except Exception:
                pass
        except Exception:
            pass

        return {
            "content": "",
            "confidence": "low",
            "evidence_files": [],
            "blind_spots": ["output_repair_failed"],
            "details": {},
        }


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
        auditor_output: dict = None,
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
            auditor_output,
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
        auditor_output: dict = None,
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
            "auditor_output": auditor_output,
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
        self._synthesizer = None
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
                SynthesisAgent,
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
            self._synthesizer = SynthesisAgent(provider_service)
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
        large_codebase_mode: bool = False,
    ) -> dict[str, Any]:
        profile = get_profile(large_codebase_mode)
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
        assert self._synthesizer is not None

        ctx: dict[str, Any] = {
            "provider_id": provider_id,
            "model_id": model_id,
            "snapshot_id": snapshot_id,
            "repo_name": repo_name,
            "graph_summary": graph_summary,
            "static_risk": static_risk,
            "static_convention": static_convention,
            "profile": profile,
        }
        mem_ctx: PipelineMemoryContext | None
        try:
            mem_ctx = await prefetch_pipeline_context(
                self._project_identity._retrieval, snapshot_id, profile=profile
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
                    profile=c.get("profile"),
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
                    profile=c.get("profile"),
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
                    profile=c.get("profile"),
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
                    profile=c.get("profile"),
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
                    profile=c.get("profile"),
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
                    profile=c.get("profile"),
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
                    c["provider_id"],
                    c["model_id"],
                    c["snapshot_id"],
                    c["graph_summary"],
                    profile=c.get("profile"),
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
                    profile=c.get("profile"),
                ),
                lambda _c, _d: self._onboarding._fallback("pipeline"),
                on_section_done,
            ),
        )
        pipeline.add_component(
            "glossary",
            _SectionAgentComponent(
                "I",
                lambda c, _d: self._glossary.run(
                    c["provider_id"],
                    c["model_id"],
                    c["snapshot_id"],
                    profile=c.get("profile"),
                ),
                lambda _c, _d: self._glossary._fallback("pipeline"),
                on_section_done,
            ),
        )
        pipeline.add_component(
            "risk",
            _SectionAgentComponent(
                "J",
                lambda c, _d: self._risk.run(
                    c["provider_id"],
                    c["model_id"],
                    c["snapshot_id"],
                    c["static_risk"],
                    profile=c.get("profile"),
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
                    profile=c.get("profile"),
                ),
                lambda _c, _d: _section_k_pipeline_fallback(),
                on_section_done,
            ),
        )
        pipeline.add_component(
            "synthesizer",
            _SectionAgentComponent(
                "L",
                lambda c, d: self._synthesizer.run(
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
                        "K": d.get("auditor_output"),
                    },
                    profile=c.get("profile"),
                ),
                lambda _c, _d: _section_l_pipeline_fallback(),
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
        pipeline.connect("auditor.output", "synthesizer.auditor_output")
        pipeline.connect("project_identity.output", "synthesizer.identity_output")
        pipeline.connect("architecture.output", "synthesizer.architecture_output")
        pipeline.connect("structure.output", "synthesizer.structure_output")
        pipeline.connect("conventions.output", "synthesizer.conventions_output")
        pipeline.connect("violations.output", "synthesizer.violations_output")
        pipeline.connect("feature_map.output", "synthesizer.feature_map_output")
        pipeline.connect("important_files.output", "synthesizer.important_files_output")
        pipeline.connect("onboarding.output", "synthesizer.onboarding_output")
        pipeline.connect("glossary.output", "synthesizer.glossary_output")
        pipeline.connect("risk.output", "synthesizer.risk_output")
        names = {
            "project_identity",
            "architecture",
            "structure",
            "conventions",
            "violations",
            "feature_map",
            "important_files",
            "onboarding",
            "glossary",
            "risk",
            "auditor",
            "synthesizer",
        }
        data = {name: {"ctx": ctx} for name in names}
        effective_concurrency = max(1, int(self._concurrency_limit * profile.concurrency_scale))
        if large_codebase_mode:
            logger.info(
                "[pipeline] large_codebase_mode=True profile=%s concurrency=%d->%d",
                profile.mode,
                self._concurrency_limit,
                effective_concurrency,
            )
        async for partial in pipeline.run_async_generator(
            data=data,
            include_outputs_from=names,
            concurrency_limit=effective_concurrency,
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
        if "L" not in sections:
            sections["L"] = _section_l_pipeline_fallback()

        # Phase 3: quality-gate retry loop (large_codebase_mode only).
        # If the auditor (K) or synthesis (L) section indicates low confidence,
        # retry up to _LARGE_MODE_MAX_RETRIES times for the weakest sections.
        if large_codebase_mode:
            sections = await self._quality_gate_retry(
                sections=sections,
                provider_id=provider_id,
                model_id=model_id,
                snapshot_id=snapshot_id,
                repo_name=repo_name,
                graph_summary=graph_summary,
                static_risk=static_risk,
                static_convention=static_convention,
                mem_ctx=mem_ctx,
                profile=profile,
            )

        return {"version": REPORT_VERSION, "sections": sections}

    # ------------------------------------------------------------------
    # Phase 3 — quality-gate retry (large_codebase_mode only)
    # ------------------------------------------------------------------

    def _audit_quality(self, sections: dict[str, Any]) -> list[str]:
        """Return list of section letters with low confidence according to auditor K.

        Falls back to inspecting individual section confidence fields if K is
        absent or incomplete.
        """
        k = sections.get("K") or {}
        weakest: list[str] = []

        # Collect auditor-flagged weak sections
        raw_w = k.get("weakest_sections", [])
        if isinstance(raw_w, list):
            for s in raw_w:
                letter = str(s).upper()
                if letter in _RETRYABLE_SECTIONS:
                    weakest.append(letter)

        # Also add any section that self-reports low confidence
        for letter in _RETRYABLE_SECTIONS:
            sec = sections.get(letter) or {}
            if str(sec.get("confidence", "")).lower() == "low" and letter not in weakest:
                weakest.append(letter)

        return weakest

    async def _retry_section(
        self,
        letter: str,
        *,
        provider_id: str,
        model_id: str,
        snapshot_id: str,
        repo_name: str,
        graph_summary: Any,
        static_risk: Any,
        static_convention: Any,
        mem_ctx: Any,
        profile: Any,
        sections: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Re-run a single section agent and return the new output, or None on failure."""
        arch_bundle = mem_ctx.arch_bundle if mem_ctx else None
        folder_tree = mem_ctx.folder_tree if mem_ctx else ""
        try:
            if letter == "A":
                return await self._project_identity.run(
                    provider_id, model_id, snapshot_id, repo_name,
                    mem_ctx=mem_ctx, profile=profile,
                )
            if letter == "B":
                return await self._architecture.run(
                    provider_id, model_id, snapshot_id, graph_summary,
                    arch_bundle=arch_bundle,
                    identity_output=sections.get("A"),
                    profile=profile,
                )
            if letter == "C":
                return await self._structure.run(
                    provider_id, model_id, snapshot_id,
                    arch_bundle=arch_bundle, folder_tree=folder_tree,
                    identity_output=sections.get("A"),
                    profile=profile,
                )
            if letter == "D":
                return await self._conventions.run(
                    provider_id, model_id, snapshot_id,
                    static_convention, None, profile=profile,
                )
            if letter == "E":
                return await self._violations.run(
                    provider_id, model_id, snapshot_id,
                    static_convention, static_risk,
                    conventions_output=sections.get("D"),
                    profile=profile,
                )
            if letter == "F":
                return await self._feature_map.run(
                    provider_id, model_id, snapshot_id, graph_summary,
                    identity_output=sections.get("A"),
                    architecture_output=sections.get("B"),
                    profile=profile,
                )
            if letter == "G":
                return await self._important_files.run(
                    provider_id, model_id, snapshot_id, graph_summary,
                    profile=profile,
                )
            if letter == "H":
                return await self._onboarding.run(
                    provider_id, model_id, snapshot_id,
                    sections.get("G"), profile=profile,
                )
            if letter == "I":
                return await self._glossary.run(
                    provider_id, model_id, snapshot_id, profile=profile,
                )
            if letter == "J":
                return await self._risk.run(
                    provider_id, model_id, snapshot_id, static_risk, profile=profile,
                )
        except Exception as retry_err:
            logger.warning(
                "[pipeline] quality_gate retry section=%s failed: %s", letter, retry_err
            )
        return None

    async def _quality_gate_retry(
        self,
        sections: dict[str, Any],
        *,
        provider_id: str,
        model_id: str,
        snapshot_id: str,
        repo_name: str,
        graph_summary: Any,
        static_risk: Any,
        static_convention: Any,
        mem_ctx: Any,
        profile: Any,
    ) -> dict[str, Any]:
        """Retry weak sections up to _LARGE_MODE_MAX_RETRIES times.

        After all retries, K and L are re-synthesised so the final report
        reflects the improved base sections.  Gracefully degrades: any
        individual failure is logged and the original section is kept.
        """
        import asyncio as _asyncio

        for attempt in range(1, _LARGE_MODE_MAX_RETRIES + 1):
            weak = self._audit_quality(sections)
            if not weak:
                logger.info("[pipeline] quality_gate pass=%d — all sections acceptable", attempt)
                break
            logger.info(
                "[pipeline] quality_gate pass=%d — retrying sections=%s",
                attempt,
                weak,
            )
            retry_tasks = {
                letter: self._retry_section(
                    letter,
                    provider_id=provider_id,
                    model_id=model_id,
                    snapshot_id=snapshot_id,
                    repo_name=repo_name,
                    graph_summary=graph_summary,
                    static_risk=static_risk,
                    static_convention=static_convention,
                    mem_ctx=mem_ctx,
                    profile=profile,
                    sections=sections,
                )
                for letter in weak
            }
            results = await _asyncio.gather(*retry_tasks.values(), return_exceptions=True)
            improved = False
            for letter, result in zip(retry_tasks.keys(), results):
                if isinstance(result, dict):
                    new_conf = str(result.get("confidence", "low")).lower()
                    old_conf = str((sections.get(letter) or {}).get("confidence", "low")).lower()
                    if new_conf != "low" or old_conf == "low":
                        sections[letter] = result
                        improved = True
                        logger.info(
                            "[pipeline] quality_gate section=%s conf=%s->%s",
                            letter, old_conf, new_conf,
                        )

            if not improved:
                logger.info(
                    "[pipeline] quality_gate pass=%d — no improvement, stopping retries", attempt
                )
                break

        # Re-run K (auditor) and L (synthesizer) with the final section set
        try:
            new_k = await self._auditor.run(
                provider_id, model_id, sections, profile=profile
            )
            sections["K"] = new_k
        except Exception as e:
            logger.warning("[pipeline] quality_gate re-audit failed: %s", e)
        try:
            new_l = await self._synthesizer.run(
                provider_id, model_id, sections, profile=profile
            )
            sections["L"] = new_l
        except Exception as e:
            logger.warning("[pipeline] quality_gate re-synthesize failed: %s", e)

        return sections
