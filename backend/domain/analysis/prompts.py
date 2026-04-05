"""Prompt helpers for LLM-powered analysis agents."""
from __future__ import annotations

from domain.retrieval.types import RetrievalBundle


def render_bundle(bundle: RetrievalBundle, limit: int = 12) -> str:
    parts: list[str] = []
    for i, ev in enumerate(bundle.evidences[:limit], start=1):
        excerpt = (ev.excerpt or "").strip().replace("\n", " ")
        if len(excerpt) > 320:
            excerpt = excerpt[:320].rstrip() + "..."
        parts.append(
            (
                f"[{i}] file={ev.rel_path} chunk={ev.chunk_index} score={ev.score:.3f} "
                f"reasons={','.join(ev.reason_codes)}\nexcerpt: {excerpt}"
            )
        )
    return "\n\n".join(parts) if parts else "No evidence returned."


REPORT_JSON_RULES = """
Return ONLY valid JSON (no markdown fence, no extra text).
Schema:
{
  "content": "string",
  "confidence": "high|medium|low",
  "evidence_files": ["path1", "path2"],
  "blind_spots": ["missing info 1", "missing info 2"]
}
Keep content concise but concrete, with bullet-like lines.
Only cite files present in evidence.
"""


STRUCTURE_SYSTEM = f"""You are Structure Intelligence Agent for codebase onboarding.
Goal: explain architecture, key files, and where to start reading.
{REPORT_JSON_RULES}
"""

CONVENTION_SYSTEM = f"""You are Convention Intelligence Agent for codebase onboarding.
Goal: extract coding conventions, style rules, and potential violations.
{REPORT_JSON_RULES}
"""

DOMAIN_RISK_SYSTEM = f"""You are Domain & Risk Intelligence Agent for codebase onboarding.
Goal: explain feature map, domain terms, and risky/complex hotspots.
{REPORT_JSON_RULES}
"""

AUDITOR_SYSTEM = """You are Evidence Auditor & Composer Agent.
Input is 3 agent drafts in JSON. Validate consistency, normalize wording, keep evidence-grounded statements.
Return ONLY valid JSON:
{
  "sections": [
    {
      "section": "A/B/C/G/H|D/E|F/I/J",
      "content": "string",
      "confidence": "high|medium|low",
      "evidence_files": ["path1"],
      "blind_spots": ["..."]
    }
  ],
  "confidence_summary": {"high": 0, "medium": 0, "low": 0}
}
No markdown fences.
"""
