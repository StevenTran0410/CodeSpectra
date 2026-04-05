"""LLM-powered analysis agent pipeline."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from domain.model_connector.service import ProviderConfigService
from domain.model_connector.types import ChatMessage, ChatRequest
from domain.retrieval.types import RetrievalBundle

from .prompts import (
    AUDITOR_SYSTEM,
    CONVENTION_SYSTEM,
    DOMAIN_RISK_SYSTEM,
    STRUCTURE_SYSTEM,
    render_bundle,
)


@dataclass
class SectionDraft:
    section: str
    content: str
    confidence: str
    evidence_files: list[str]
    blind_spots: list[str]


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


def _fallback_section(section: str, bundles: list[RetrievalBundle], reason: str) -> SectionDraft:
    files: list[str] = []
    for b in bundles:
        for f in _top_files(b, 8):
            if f not in files:
                files.append(f)
    return SectionDraft(
        section=section,
        content=f"Agent fallback output. Reason: {reason}",
        confidence="low",
        evidence_files=files[:8],
        blind_spots=[reason],
    )


class BaseLLMAgent:
    def __init__(self, provider_service: ProviderConfigService) -> None:
        self._providers = provider_service

    async def _chat_json(
        self,
        provider_id: str,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
        max_completion_tokens: int = 900,
    ) -> dict[str, Any]:
        res = await self._providers.chat(
            ChatRequest(
                provider_id=provider_id,
                model_id=model_id,
                messages=[
                    ChatMessage(role="system", content=system_prompt),
                    ChatMessage(role="user", content=user_prompt),
                ],
                max_completion_tokens=max_completion_tokens,
                temperature=0.2,
                stream=False,
            )
        )
        text = (res.content or "").strip()
        # tolerate fenced JSON from weaker models
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
        return json.loads(text)


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
            "Task: produce section A/B/C/G/H for onboarding.\n\n"
            f"Architecture evidence:\n{render_bundle(architecture)}\n\n"
            f"Important-files evidence:\n{render_bundle(important)}\n"
        )
        try:
            data = await self._chat_json(provider_id, model_id, STRUCTURE_SYSTEM, prompt)
            return SectionDraft(
                section="A/B/C/G/H",
                content=str(data.get("content", "")).strip() or "No content.",
                confidence=_normalize_conf(str(data.get("confidence", "medium"))),
                evidence_files=[
                    f for f in data.get("evidence_files", []) if isinstance(f, str)
                ][:8] or files[:8],
                blind_spots=[
                    b for b in data.get("blind_spots", []) if isinstance(b, str)
                ][:8],
            )
        except Exception as e:
            return _fallback_section("A/B/C/G/H", [architecture, important], str(e))


class ConventionIntelligenceAgent(BaseLLMAgent):
    async def run(
        self,
        provider_id: str,
        model_id: str,
        conventions: RetrievalBundle,
    ) -> SectionDraft:
        prompt = (
            "Task: produce section D/E for onboarding.\n\n"
            f"Conventions evidence:\n{render_bundle(conventions)}\n"
        )
        try:
            data = await self._chat_json(provider_id, model_id, CONVENTION_SYSTEM, prompt)
            return SectionDraft(
                section="D/E",
                content=str(data.get("content", "")).strip() or "No content.",
                confidence=_normalize_conf(str(data.get("confidence", "medium"))),
                evidence_files=[
                    f for f in data.get("evidence_files", []) if isinstance(f, str)
                ][:8] or _top_files(conventions, 8),
                blind_spots=[
                    b for b in data.get("blind_spots", []) if isinstance(b, str)
                ][:8],
            )
        except Exception as e:
            return _fallback_section("D/E", [conventions], str(e))


class DomainRiskIntelligenceAgent(BaseLLMAgent):
    async def run(
        self,
        provider_id: str,
        model_id: str,
        feature_map: RetrievalBundle,
        risk: RetrievalBundle,
    ) -> SectionDraft:
        prompt = (
            "Task: produce section F/I/J for onboarding.\n\n"
            f"Feature-map evidence:\n{render_bundle(feature_map)}\n\n"
            f"Risk evidence:\n{render_bundle(risk)}\n"
        )
        try:
            data = await self._chat_json(provider_id, model_id, DOMAIN_RISK_SYSTEM, prompt)
            files = _top_files(feature_map, 8)
            for f in _top_files(risk, 8):
                if f not in files:
                    files.append(f)
            return SectionDraft(
                section="F/I/J",
                content=str(data.get("content", "")).strip() or "No content.",
                confidence=_normalize_conf(str(data.get("confidence", "medium"))),
                evidence_files=[
                    f for f in data.get("evidence_files", []) if isinstance(f, str)
                ][:8] or files[:8],
                blind_spots=[
                    b for b in data.get("blind_spots", []) if isinstance(b, str)
                ][:8],
            )
        except Exception as e:
            return _fallback_section("F/I/J", [feature_map, risk], str(e))


class EvidenceAuditorComposerAgent(BaseLLMAgent):
    async def run(
        self,
        provider_id: str,
        model_id: str,
        structure: SectionDraft,
        conventions: SectionDraft,
        domain_risk: SectionDraft,
    ) -> dict[str, Any]:
        payload = {
            "structure": structure.__dict__,
            "conventions": conventions.__dict__,
            "domain_risk": domain_risk.__dict__,
        }
        try:
            data = await self._chat_json(
                provider_id=provider_id,
                model_id=model_id,
                system_prompt=AUDITOR_SYSTEM,
                user_prompt=json.dumps(payload, ensure_ascii=True),
                max_completion_tokens=1200,
            )
            sections = data.get("sections", [])
            conf = data.get("confidence_summary", {})
            if isinstance(sections, list) and isinstance(conf, dict):
                return {
                    "sections": sections,
                    "confidence_summary": {
                        "high": int(conf.get("high", 0)),
                        "medium": int(conf.get("medium", 0)),
                        "low": int(conf.get("low", 0)),
                    },
                }
        except Exception:
            pass

        # fallback deterministic composition
        sections = [structure, conventions, domain_risk]
        return {
            "sections": [
                {
                    "section": s.section,
                    "content": s.content,
                    "confidence": s.confidence,
                    "evidence_files": s.evidence_files,
                    "blind_spots": s.blind_spots,
                }
                for s in sections
            ],
            "confidence_summary": {
                "high": sum(1 for s in sections if s.confidence == "high"),
                "medium": sum(1 for s in sections if s.confidence == "medium"),
                "low": sum(1 for s in sections if s.confidence == "low"),
            },
        }


class AnalysisAgentPipeline:
    """Async LLM-powered multi-agent pipeline."""

    def __init__(self, provider_service: ProviderConfigService) -> None:
        self._structure = StructureIntelligenceAgent(provider_service)
        self._convention = ConventionIntelligenceAgent(provider_service)
        self._domain_risk = DomainRiskIntelligenceAgent(provider_service)
        self._auditor = EvidenceAuditorComposerAgent(provider_service)

    async def run(
        self,
        provider_id: str,
        model_id: str,
        architecture: RetrievalBundle,
        important: RetrievalBundle,
        conventions: RetrievalBundle,
        feature_map: RetrievalBundle,
        risk: RetrievalBundle,
    ) -> dict[str, Any]:
        structure = await self._structure.run(provider_id, model_id, architecture, important)
        conv = await self._convention.run(provider_id, model_id, conventions)
        dom = await self._domain_risk.run(provider_id, model_id, feature_map, risk)
        return await self._auditor.run(provider_id, model_id, structure, conv, dom)
