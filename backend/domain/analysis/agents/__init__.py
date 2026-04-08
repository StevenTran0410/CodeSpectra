"""Section agents (A–L) — LLM + optional retrieval."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "ArchitectureAgent",
    "AuditAgent",
    "ConventionsAgent",
    "FeatureMapAgent",
    "GlossaryAgent",
    "ImportantFilesAgent",
    "OnboardingAgent",
    "ProjectIdentityAgent",
    "RiskAgent",
    "StructureAgent",
    "SynthesisAgent",
    "ViolationsAgent",
]

_AGENT_EXPORTS: dict[str, str] = {
    "ArchitectureAgent": "agent_architecture",
    "AuditAgent": "agent_auditor",
    "ConventionsAgent": "agent_conventions",
    "FeatureMapAgent": "agent_feature_map",
    "GlossaryAgent": "agent_glossary",
    "ImportantFilesAgent": "agent_important_files",
    "OnboardingAgent": "agent_onboarding",
    "ProjectIdentityAgent": "agent_project_identity",
    "RiskAgent": "agent_risk",
    "StructureAgent": "agent_structure",
    "SynthesisAgent": "agent_synthesis",
    "ViolationsAgent": "agent_violations",
}


def __getattr__(name: str) -> Any:
    mod_name = _AGENT_EXPORTS.get(name)
    if mod_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = importlib.import_module(f"{__name__}.{mod_name}")
    return getattr(mod, name)
