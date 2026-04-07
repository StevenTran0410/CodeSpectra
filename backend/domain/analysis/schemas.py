"""Typed section contracts for analysis report (RPA-053)."""
from __future__ import annotations

from typing import Any, TypedDict


class FileRef(TypedDict):
    file: str
    reason: str


class GlossaryTerm(TypedDict):
    term: str
    definition: str
    evidence_files: list[str]


class RiskFindingDict(TypedDict):
    category: str
    severity: str
    title: str
    rationale: str
    evidence: list[str]


class MainServiceItem(TypedDict):
    name: str
    path: str
    role: str


class FolderItem(TypedDict):
    path: str
    role: str
    description: str


class ConventionAspect(TypedDict):
    description: str
    evidence_files: list[str]


class SignalItem(TypedDict):
    category: str
    description: str
    evidence: str


class RuleItem(TypedDict):
    rule: str
    inferred_from: str
    severity: str


class ViolationItem(TypedDict):
    rule: str
    file: str
    description: str
    severity: str


class FeatureItem(TypedDict):
    name: str
    description: str
    entrypoint: str
    key_files: list[str]
    data_path: str
    tests: list[str]
    reading_order: list[str]


class OnboardingStep(TypedDict):
    order: int
    file: str
    goal: str
    outcome: str


class SectionA(TypedDict):
    repo_name: str
    domain: str
    purpose: str
    runtime_type: str
    tech_stack: list[str]
    business_context: str
    confidence: str
    evidence_files: list[str]
    blind_spots: list[str]


class SectionB(TypedDict):
    main_layers: list[str]
    frameworks: list[str]
    entrypoints: list[str]
    main_services: list[MainServiceItem]
    external_integrations: list[str]
    config_sources: list[str]
    database_hints: list[str]
    confidence: str
    evidence_files: list[str]
    blind_spots: list[str]


class SectionC(TypedDict):
    folders: list[FolderItem]
    summary: str
    confidence: str
    evidence_files: list[str]
    blind_spots: list[str]


class SectionD(TypedDict):
    naming_style: ConventionAspect
    error_handling: ConventionAspect
    async_style: ConventionAspect
    di_style: ConventionAspect
    class_vs_functional: ConventionAspect
    test_style: ConventionAspect
    signals: list[SignalItem]
    confidence: str
    evidence_files: list[str]
    blind_spots: list[str]


class SectionE(TypedDict):
    rules: list[RuleItem]
    violations_found: list[ViolationItem]
    confidence: str
    evidence_files: list[str]
    blind_spots: list[str]


class SectionF(TypedDict):
    features: list[FeatureItem]
    confidence: str
    evidence_files: list[str]
    blind_spots: list[str]


class SectionG(TypedDict):
    entrypoint: FileRef
    backbone: FileRef
    critical_config: FileRef
    highest_centrality: FileRef
    most_dangerous_to_touch: FileRef
    read_first: FileRef
    other_important: list[FileRef]
    confidence: str
    evidence_files: list[str]
    blind_spots: list[str]


class SectionH(TypedDict):
    steps: list[OnboardingStep]
    total_estimated_minutes: int
    confidence: str
    evidence_files: list[str]
    blind_spots: list[str]


class SectionI(TypedDict):
    terms: list[GlossaryTerm]
    confidence: str
    blind_spots: list[str]


class SectionJ(TypedDict):
    findings: list[RiskFindingDict]
    summary: str
    confidence: str
    evidence_files: list[str]
    blind_spots: list[str]


class SectionK(TypedDict):
    overall_confidence: str
    section_scores: dict[str, str]
    weakest_sections: list[str]
    coverage_percentage: float
    notes: str
    blind_spots: list[str]


_SECTION_TYPES: dict[str, type] = {
    "A": SectionA,
    "B": SectionB,
    "C": SectionC,
    "D": SectionD,
    "E": SectionE,
    "F": SectionF,
    "G": SectionG,
    "H": SectionH,
    "I": SectionI,
    "J": SectionJ,
    "K": SectionK,
}


def validate_section(section_id: str, data: dict[str, Any]) -> None:
    """Raises ValueError with field name if required keys are missing."""
    t = _SECTION_TYPES.get(section_id)
    if t is None:
        raise ValueError(f"Unknown section_id: {section_id!r}")
    required = set(getattr(t, "__annotations__", {}))
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"Section {section_id} missing fields: {sorted(missing)}")
