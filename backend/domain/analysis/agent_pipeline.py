"""Haystack-based analysis agent pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from haystack import Pipeline, component

from domain.retrieval.types import RetrievalBundle


@dataclass
class SectionDraft:
    section: str
    content: str
    confidence: str
    evidence_files: list[str]
    blind_spots: list[str]


def _top_files(bundle: RetrievalBundle, n: int = 6) -> list[str]:
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


@component
class StructureIntelligenceAgent:
    @component.output_types(draft=SectionDraft)
    def run(self, architecture: RetrievalBundle, important: RetrievalBundle) -> dict[str, SectionDraft]:
        files = _top_files(architecture, 5) + [f for f in _top_files(important, 5) if f not in _top_files(architecture, 5)]
        confidence = "high" if architecture.evidences else "low"
        return {
            "draft": SectionDraft(
                section="A/B/C/G/H",
                content="Structure/architecture draft synthesized from retrieval evidence.",
                confidence=confidence,
                evidence_files=files[:8],
                blind_spots=[] if files else ["No architecture evidence returned"],
            )
        }


@component
class ConventionIntelligenceAgent:
    @component.output_types(draft=SectionDraft)
    def run(self, conventions: RetrievalBundle) -> dict[str, SectionDraft]:
        files = _top_files(conventions, 8)
        confidence = "medium" if len(files) >= 3 else "low"
        return {
            "draft": SectionDraft(
                section="D/E",
                content="Conventions/negative-rule draft inferred from evidence.",
                confidence=confidence,
                evidence_files=files,
                blind_spots=[] if files else ["Insufficient convention evidence"],
            )
        }


@component
class DomainRiskIntelligenceAgent:
    @component.output_types(draft=SectionDraft)
    def run(self, feature_map: RetrievalBundle, risk: RetrievalBundle) -> dict[str, SectionDraft]:
        files = _top_files(feature_map, 5) + [f for f in _top_files(risk, 5) if f not in _top_files(feature_map, 5)]
        confidence = "medium" if risk.evidences else "low"
        return {
            "draft": SectionDraft(
                section="F/I/J",
                content="Feature/domain/risk draft synthesized from retrieval evidence.",
                confidence=confidence,
                evidence_files=files[:8],
                blind_spots=[] if files else ["No risk/feature evidence available"],
            )
        }


@component
class EvidenceAuditorComposerAgent:
    @component.output_types(report=dict)
    def run(
        self,
        structure: SectionDraft,
        conventions: SectionDraft,
        domain_risk: SectionDraft,
    ) -> dict[str, dict[str, Any]]:
        sections = [structure, conventions, domain_risk]
        report = {
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
        return {"report": report}


class AnalysisAgentPipeline:
    """Thin wrapper so analysis runtime can call a deterministic Haystack graph."""

    def __init__(self) -> None:
        p = Pipeline()
        p.add_component("structure_agent", StructureIntelligenceAgent())
        p.add_component("convention_agent", ConventionIntelligenceAgent())
        p.add_component("domain_risk_agent", DomainRiskIntelligenceAgent())
        p.add_component("auditor_composer", EvidenceAuditorComposerAgent())

        p.connect("structure_agent.draft", "auditor_composer.structure")
        p.connect("convention_agent.draft", "auditor_composer.conventions")
        p.connect("domain_risk_agent.draft", "auditor_composer.domain_risk")
        self._pipeline = p

    def run(
        self,
        architecture: RetrievalBundle,
        important: RetrievalBundle,
        conventions: RetrievalBundle,
        feature_map: RetrievalBundle,
        risk: RetrievalBundle,
    ) -> dict[str, Any]:
        out = self._pipeline.run(
            {
                "structure_agent": {
                    "architecture": architecture,
                    "important": important,
                },
                "convention_agent": {
                    "conventions": conventions,
                },
                "domain_risk_agent": {
                    "feature_map": feature_map,
                    "risk": risk,
                },
            }
        )
        return out["auditor_composer"]["report"]
