"""LLM-powered analysis agent pipeline."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any

from domain.model_connector.service import ProviderConfigService
from domain.model_connector.types import ChatMessage, ChatRequest
from domain.retrieval.service import RetrievalService
from domain.retrieval.types import RetrievalBundle
from domain.structural_graph.types import StructuralGraphSummary
from shared.logger import logger

from .prompts import (
    AUDITOR_SYSTEM,
    CONVENTION_SYSTEM,
    DOMAIN_RISK_SYSTEM,
    RISK_COMPLEXITY_SYSTEM,
    STRUCTURE_SYSTEM,
    render_bundle,
)
from .static_convention import ConventionReport
from .static_risk import RiskReport

# Sentences indicating the model produced a refusal / meta-comment instead of content.
_META_PATTERNS = re.compile(
    r"^(no input|no text|i cannot|i'm unable|i am unable|"
    r"missing source|please provide|nothing to (rewrite|convert|transform)|"
    r"i don't have|i do not have)",
    re.IGNORECASE,
)


@dataclass
class SectionDraft:
    section: str
    content: str
    confidence: str
    evidence_files: list[str]
    blind_spots: list[str]
    details: dict[str, Any] = field(default_factory=dict)


def _top_files(bundle: RetrievalBundle, n: int = 8) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for e in bundle.evidences:
        if e.rel_path in seen:
            continue
        seen.add(e.rel_path)
        out.append(e.rel_path)
        if len(out) >= n:
            break
    return out


def _normalize_conf(v: str) -> str:
    t = (v or "").strip().lower()
    if t in {"high", "medium", "low"}:
        return t
    return "medium"


def _is_meta_content(text: str) -> bool:
    """True when the model returned a refusal / meta-comment rather than analysis."""
    return bool(_META_PATTERNS.match(text.strip()))


def _heuristic_details_structure(
    architecture: RetrievalBundle, important: RetrievalBundle
) -> dict[str, Any]:
    arch_files = _top_files(architecture, 5)
    imp_files = _top_files(important, 5)
    return {
        "identity_card": ["(could not be generated — re-run analysis)"],
        "architecture_overview": arch_files or ["(no evidence)"],
        "onboarding_digest": [
            {
                "file": f,
                "why": "central file from graph/retrieval",
                "outcome": "understand core structure",
            }
            for f in imp_files[:4]
        ],
    }


def _heuristic_details_domain(
    feature_map: RetrievalBundle, glossary: RetrievalBundle, risk: RetrievalBundle
) -> dict[str, Any]:
    feat_files = _top_files(feature_map, 5)
    risk_files = _top_files(risk, 5)
    return {
        "feature_map": [
            {"feature": f, "core_files": [f], "notes": "retrieved as feature-map evidence"}
            for f in feat_files[:4]
        ],
        "important_files_radar": [{"file": f, "reason": "blast radius"} for f in risk_files[:4]],
        "glossary": [{"term": "(none extracted)", "meaning": "", "evidence": ""}],
        "risk_hotspots": [
            {"area": f, "why": "high-score risk evidence", "files": [f]} for f in risk_files[:3]
        ],
    }


def _fallback_section(section: str, bundles: list[RetrievalBundle], reason: str) -> SectionDraft:
    files: list[str] = []
    for b in bundles:
        for f in _top_files(b, 8):
            if f not in files:
                files.append(f)
    return SectionDraft(
        section=section,
        content=f"Analysis could not be completed. Reason: {reason}",
        confidence="low",
        evidence_files=files[:8],
        blind_spots=[f"Agent failed: {reason}"],
        details={},
    )


class BaseLLMAgent:
    def __init__(self, provider_service: ProviderConfigService) -> None:
        self._providers = provider_service

    @staticmethod
    def _try_parse_json(text: str) -> dict[str, Any]:
        raw = (text or "").strip()
        # Strip markdown fences
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
        # Extract first {...} block
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
        if not content or _is_meta_content(content):
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
        # Attempt 1: normal call at low temperature for deterministic JSON
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

        # Attempt 2: ask model to extract structured fields from its own prose output.
        # Use temperature=None so even strict models (o1/o3/gpt-5) can answer.
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
            if obj.get("content") and not _is_meta_content(str(obj.get("content", ""))):
                return obj
        except Exception:
            pass

        raise ValueError(f"all_attempts_failed: last_output={text[:120]!r}")


class StructureIntelligenceAgent(BaseLLMAgent):
    async def run(
        self,
        provider_id: str,
        model_id: str,
        architecture: RetrievalBundle,
        important: RetrievalBundle,
    ) -> SectionDraft:
        files = _top_files(architecture, 8)
        for f in _top_files(important, 8):
            if f not in files:
                files.append(f)
        prompt = (
            "Task: produce section A/B/C/G/H for onboarding report.\n\n"
            f"Architecture evidence:\n{render_bundle(architecture)}\n\n"
            f"Important-files evidence:\n{render_bundle(important)}\n"
        )
        try:
            data = await self._chat_json(
                provider_id,
                model_id,
                STRUCTURE_SYSTEM,
                prompt,
                max_completion_tokens=1400,
            )
            content = str(data.get("content", "")).strip()
            details = data.get("details")
            if not isinstance(details, dict) or not details:
                details = _heuristic_details_structure(architecture, important)
            return SectionDraft(
                section="A/B/C/G/H",
                content=content or "No content generated.",
                confidence=_normalize_conf(str(data.get("confidence", "medium"))),
                evidence_files=[f for f in data.get("evidence_files", []) if isinstance(f, str)][:8]
                or files[:8],
                blind_spots=[b for b in data.get("blind_spots", []) if isinstance(b, str)][:8],
                details=details,
            )
        except Exception as e:
            draft = _fallback_section("A/B/C/G/H", [architecture, important], str(e))
            draft.details = _heuristic_details_structure(architecture, important)
            return draft


class ConventionIntelligenceAgent(BaseLLMAgent):
    async def run(
        self,
        provider_id: str,
        model_id: str,
        conventions: RetrievalBundle,
        static_findings: ConventionReport | None = None,
    ) -> SectionDraft:
        static_ctx = (
            static_findings.as_context_text() if static_findings else "No static convention data."
        )
        prompt = (
            "Task: produce section D/E for onboarding report.\n\n"
            f"Pre-computed convention signals (ground truth):\n{static_ctx}\n\n"
            f"Code evidence excerpts:\n{render_bundle(conventions)}\n"
        )
        try:
            data = await self._chat_json(
                provider_id,
                model_id,
                CONVENTION_SYSTEM,
                prompt,
                max_completion_tokens=1200,
            )
            content = str(data.get("content", "")).strip()
            details = data.get("details")
            if not isinstance(details, dict):
                details = {}
            # Enrich details with static signals for UI rendering
            if static_findings and static_findings.signals:
                details["convention_signals"] = static_findings.to_dict()["signals"][:12]
            return SectionDraft(
                section="D/E",
                content=content or "No content generated.",
                confidence=_normalize_conf(str(data.get("confidence", "medium"))),
                evidence_files=[f for f in data.get("evidence_files", []) if isinstance(f, str)][:8]
                or _top_files(conventions, 8),
                blind_spots=[b for b in data.get("blind_spots", []) if isinstance(b, str)][:8],
                details=details,
            )
        except Exception as e:
            draft = _fallback_section("D/E", [conventions], str(e))
            if static_findings and static_findings.signals:
                draft.details = {"convention_signals": static_findings.to_dict()["signals"][:12]}
            return draft


class DomainRiskIntelligenceAgent(BaseLLMAgent):
    async def run(
        self,
        provider_id: str,
        model_id: str,
        feature_map: RetrievalBundle,
        glossary: RetrievalBundle,
        risk: RetrievalBundle,
    ) -> SectionDraft:
        prompt = (
            "Task: produce section F/I/J for onboarding report.\n\n"
            f"Feature-map evidence:\n{render_bundle(feature_map)}\n\n"
            f"Glossary evidence:\n{render_bundle(glossary)}\n\n"
            f"Risk evidence:\n{render_bundle(risk)}\n"
        )
        try:
            data = await self._chat_json(
                provider_id,
                model_id,
                DOMAIN_RISK_SYSTEM,
                prompt,
                max_completion_tokens=1400,
            )
            files = _top_files(feature_map, 8)
            for f in _top_files(glossary, 8):
                if f not in files:
                    files.append(f)
            for f in _top_files(risk, 8):
                if f not in files:
                    files.append(f)
            content = str(data.get("content", "")).strip()
            details = data.get("details")
            if not isinstance(details, dict) or not details:
                details = _heuristic_details_domain(feature_map, glossary, risk)
            return SectionDraft(
                section="F/I/J",
                content=content or "No content generated.",
                confidence=_normalize_conf(str(data.get("confidence", "medium"))),
                evidence_files=[f for f in data.get("evidence_files", []) if isinstance(f, str)][:8]
                or files[:8],
                blind_spots=[b for b in data.get("blind_spots", []) if isinstance(b, str)][:8],
                details=details,
            )
        except Exception as e:
            draft = _fallback_section("F/I/J", [feature_map, glossary, risk], str(e))
            draft.details = _heuristic_details_domain(feature_map, glossary, risk)
            return draft


def _heuristic_details_risk(risk_report: RiskReport | None) -> dict[str, Any]:
    if not risk_report or not risk_report.findings:
        return {"risk_findings": [], "summary": "No static risk data available."}
    return risk_report.to_dict() | {"summary": "Static risk analysis — re-run for LLM narrative."}


class RiskComplexityAgent(BaseLLMAgent):
    async def run(
        self,
        provider_id: str,
        model_id: str,
        risk_bundle: RetrievalBundle,
        static_risk: RiskReport | None = None,
    ) -> SectionDraft:
        static_ctx = static_risk.as_context_text() if static_risk else "No static risk data."
        prompt = (
            "Task: produce section J (Risk / Complexity / Unknowns) for onboarding report.\n\n"
            f"Pre-computed static risk findings (ground truth):\n{static_ctx}\n\n"
            f"Supporting code evidence:\n{render_bundle(risk_bundle)}\n"
        )
        try:
            data = await self._chat_json(
                provider_id,
                model_id,
                RISK_COMPLEXITY_SYSTEM,
                prompt,
                max_completion_tokens=1400,
            )
            content = str(data.get("content", "")).strip()
            details = data.get("details")
            if not isinstance(details, dict) or not details:
                details = _heuristic_details_risk(static_risk)
            elif static_risk and static_risk.findings:
                # Always embed the static findings in details for UI rendering
                if "risk_findings" not in details or not details["risk_findings"]:
                    details["risk_findings"] = static_risk.to_dict()["findings"][:10]
            return SectionDraft(
                section="J",
                content=content or "No content generated.",
                confidence=_normalize_conf(str(data.get("confidence", "medium"))),
                evidence_files=[f for f in data.get("evidence_files", []) if isinstance(f, str)][:8]
                or _top_files(risk_bundle, 8),
                blind_spots=[b for b in data.get("blind_spots", []) if isinstance(b, str)][:8],
                details=details,
            )
        except Exception as e:
            draft = _fallback_section("J", [risk_bundle], str(e))
            draft.details = _heuristic_details_risk(static_risk)
            return draft


class EvidenceAuditorComposerAgent(BaseLLMAgent):
    async def run(
        self,
        provider_id: str,
        model_id: str,
        drafts: list[SectionDraft],
    ) -> dict[str, Any]:
        payload = {d.section: {k: v for k, v in d.__dict__.items()} for d in drafts}
        try:
            data = await self._chat_json(
                provider_id=provider_id,
                model_id=model_id,
                system_prompt=AUDITOR_SYSTEM,
                user_prompt=json.dumps(payload, ensure_ascii=True),
                max_completion_tokens=1800,
            )
            raw_sections = data.get("sections", [])
            conf = data.get("confidence_summary", {})
            if isinstance(raw_sections, list) and isinstance(conf, dict):
                details_by_section = {d.section: d.details or {} for d in drafts}
                orig_by_section = {d.section: d for d in drafts}
                normalized: list[dict[str, Any]] = []
                for s in raw_sections:
                    if not isinstance(s, dict):
                        continue
                    sid = str(s.get("section", "")).strip()
                    if not sid:
                        continue
                    content = str(s.get("content", "")).strip()
                    if _is_meta_content(content):
                        orig = orig_by_section.get(sid)
                        if orig:
                            s["content"] = orig.content
                    if not isinstance(s.get("details"), dict) or not s["details"]:
                        s["details"] = details_by_section.get(sid, {})
                    normalized.append(s)
                if normalized:
                    return {
                        "sections": normalized,
                        "confidence_summary": {
                            "high": int(conf.get("high", 0)),
                            "medium": int(conf.get("medium", 0)),
                            "low": int(conf.get("low", 0)),
                        },
                    }
        except Exception:
            pass

        # Deterministic fallback
        return {
            "sections": [
                {
                    "section": s.section,
                    "content": s.content,
                    "confidence": s.confidence,
                    "evidence_files": s.evidence_files,
                    "blind_spots": s.blind_spots,
                    "details": s.details or {},
                }
                for s in drafts
            ],
            "confidence_summary": {
                "high": sum(1 for s in drafts if s.confidence == "high"),
                "medium": sum(1 for s in drafts if s.confidence == "medium"),
                "low": sum(1 for s in drafts if s.confidence == "low"),
            },
        }


class AnalysisAgentPipeline:
    def __init__(
        self,
        provider_service: ProviderConfigService,
        retrieval_service: RetrievalService | None = None,
    ) -> None:
        self._structure = StructureIntelligenceAgent(provider_service)
        self._convention = ConventionIntelligenceAgent(provider_service)
        self._domain_risk = DomainRiskIntelligenceAgent(provider_service)
        self._risk_complexity = RiskComplexityAgent(provider_service)
        self._auditor = EvidenceAuditorComposerAgent(provider_service)
        self._agent_a = None
        self._agent_g = None
        self._agent_i = None
        self._agent_j = None
        self._agent_b = None
        self._agent_c = None
        self._agent_h = None
        if retrieval_service is not None:
            from .agents import AgentA, AgentB, AgentC, AgentG, AgentH, AgentI, AgentJ

            self._agent_a = AgentA(provider_service, retrieval_service)
            self._agent_g = AgentG(provider_service, retrieval_service)
            self._agent_i = AgentI(provider_service, retrieval_service)
            self._agent_j = AgentJ(provider_service, retrieval_service)
            self._agent_b = AgentB(provider_service, retrieval_service)
            self._agent_c = AgentC(provider_service, retrieval_service)
            self._agent_h = AgentH(provider_service, retrieval_service)

    async def run(
        self,
        provider_id: str,
        model_id: str,
        architecture: RetrievalBundle,
        important: RetrievalBundle,
        conventions: RetrievalBundle,
        feature_map: RetrievalBundle,
        glossary: RetrievalBundle,
        risk: RetrievalBundle,
        static_convention: ConventionReport | None = None,
        static_risk: RiskReport | None = None,
        *,
        snapshot_id: str = "",
        repo_name: str = "",
        graph_summary: StructuralGraphSummary | None = None,
    ) -> dict[str, Any]:
        structure = await self._structure.run(provider_id, model_id, architecture, important)
        conv = await self._convention.run(
            provider_id, model_id, conventions, static_findings=static_convention
        )
        dom = await self._domain_risk.run(provider_id, model_id, feature_map, glossary, risk)
        risk_sec = await self._risk_complexity.run(
            provider_id, model_id, risk, static_risk=static_risk
        )
        auditor_result = await self._auditor.run(
            provider_id, model_id, [structure, conv, dom, risk_sec]
        )
        sections_v2: dict[str, Any] = {}
        if self._agent_a and snapshot_id:
            assert self._agent_g is not None
            assert self._agent_i is not None
            assert self._agent_j is not None
            sections_v2["A"] = await self._agent_a.run(
                provider_id, model_id, snapshot_id, repo_name
            )
            sections_v2["G"] = await self._agent_g.run(
                provider_id, model_id, snapshot_id, graph_summary
            )
            sections_v2["I"] = await self._agent_i.run(provider_id, model_id, snapshot_id)
            sections_v2["J"] = await self._agent_j.run(
                provider_id, model_id, snapshot_id, static_risk
            )
            if self._agent_b and snapshot_id:
                sections_v2["B"] = await self._agent_b.run(
                    provider_id, model_id, snapshot_id, graph_summary
                )
                sections_v2["C"] = await self._agent_c.run(provider_id, model_id, snapshot_id)
                sections_v2["H"] = await self._agent_h.run(
                    provider_id, model_id, snapshot_id, sections_v2.get("G")
                )
        return {**auditor_result, "sections_v2": sections_v2}
