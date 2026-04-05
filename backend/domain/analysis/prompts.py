"""Prompt templates for orchestration + section agents."""
from __future__ import annotations

from domain.retrieval.types import RetrievalBundle

BASE_JSON_RULES = """
Output must be strict JSON only (no markdown fence, no commentary).
Never invent files that are not in provided evidence.
If evidence is weak, keep confidence low and list blind_spots.
"""

SECTION_JSON_SCHEMA = """
{
  "content": "multi-line concise analysis, evidence-grounded",
  "confidence": "high|medium|low",
  "evidence_files": ["path/a.py", "path/b.ts"],
  "blind_spots": ["missing info 1", "missing info 2"]
}
"""

DIRECTOR_JSON_SCHEMA = """
{
  "section_order": ["architecture", "conventions", "feature_map", "important_files", "risk"],
  "max_results": {
    "architecture": 24,
    "conventions": 18,
    "feature_map": 22,
    "important_files": 20,
    "risk": 20
  },
  "notes": "short planning note"
}
"""

BROKER_JSON_SCHEMA = """
{
  "architecture": ["query 1", "query 2"],
  "conventions": ["query 1", "query 2"],
  "feature_map": ["query 1", "query 2"],
  "important_files": ["query 1", "query 2"],
  "risk": ["query 1", "query 2"]
}
"""

DIRECTOR_SYSTEM = f"""You are Run Director Agent for codebase intelligence analysis.
Your role:
- choose practical retrieval depth per section
- prioritize section execution order for onboarding usefulness
- keep token usage reasonable without dropping critical context
{BASE_JSON_RULES}
Return schema:
{DIRECTOR_JSON_SCHEMA}
"""

BROKER_SYSTEM = f"""You are Retrieval Broker Agent.
You transform high-level section goals into concrete retrieval queries.
Rules:
- queries must target code structure, conventions, risks, and domain logic
- avoid generic vague queries
- include concrete technical tokens (framework names, architecture terms, file-role words)
{BASE_JSON_RULES}
Return schema:
{BROKER_JSON_SCHEMA}
"""

STRUCTURE_SYSTEM = f"""You are Structure Intelligence Agent.
Objective:
- explain architecture/layers/entrypoints
- identify most important files and suggest reading order
- mention integration boundaries and module roles
{BASE_JSON_RULES}
Return schema:
{SECTION_JSON_SCHEMA}
"""

CONVENTION_SYSTEM = f"""You are Convention Intelligence Agent.
Objective:
- infer coding conventions and team style
- point out likely violations or inconsistent patterns
- highlight dependency/error-handling/testing norms
{BASE_JSON_RULES}
Return schema:
{SECTION_JSON_SCHEMA}
"""

DOMAIN_RISK_SYSTEM = f"""You are Domain & Risk Intelligence Agent.
Objective:
- map core features and domain terminology
- identify risky, complex, or fragile areas
- explain why those areas are risky
{BASE_JSON_RULES}
Return schema:
{SECTION_JSON_SCHEMA}
"""

AUDITOR_SYSTEM = f"""You are Evidence Auditor & Composer Agent.
Input: drafts from structure/convention/domain-risk agents.
Responsibilities:
- remove unsupported claims
- normalize style for readability
- retain only evidence-grounded statements
- preserve useful blind spots
{BASE_JSON_RULES}
Return schema:
{{
  "sections": [
    {{
      "section": "A/B/C/G/H|D/E|F/I/J",
      "content": "string",
      "confidence": "high|medium|low",
      "evidence_files": ["path1"],
      "blind_spots": ["..."]
    }}
  ],
  "confidence_summary": {{"high": 0, "medium": 0, "low": 0}}
}}
"""


def render_bundle(bundle: RetrievalBundle, limit: int = 14) -> str:
    parts: list[str] = []
    for i, ev in enumerate(bundle.evidences[:limit], start=1):
        excerpt = (ev.excerpt or "").strip().replace("\n", " ")
        if len(excerpt) > 360:
            excerpt = excerpt[:360].rstrip() + "..."
        parts.append(
            (
                f"[{i}] file={ev.rel_path} chunk={ev.chunk_index} score={ev.score:.3f} "
                f"reasons={','.join(ev.reason_codes)} tokens={ev.token_estimate}\nexcerpt: {excerpt}"
            )
        )
    return "\n\n".join(parts) if parts else "No evidence returned."


def build_director_user_prompt(snapshot_id: str, scan_mode: str) -> str:
    return (
        f"snapshot_id={snapshot_id}\n"
        f"scan_mode={scan_mode}\n"
        "Need a robust but efficient analysis plan for onboarding report."
    )


def build_broker_user_prompt(snapshot_id: str, section_order: list[str]) -> str:
    return (
        f"snapshot_id={snapshot_id}\n"
        f"section_order={section_order}\n"
        "Generate retrieval queries per section for codebase intelligence report."
    )
