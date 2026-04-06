"""Prompt templates for orchestration + section agents."""
from __future__ import annotations

from domain.retrieval.types import RetrievalBundle

# Placed at the END of every system prompt so it's fresh in the model's context window.
_JSON_ENFORCEMENT = """\
RESPONSE FORMAT — MANDATORY:
- Your ENTIRE response must be ONE valid JSON object.
- Start with { and end with }.
- No markdown fences (no ```json), no prose before or after.
- Double-quoted keys and string values only.
- No trailing commas.
- Never invent file paths not present in the evidence."""

SECTION_JSON_SCHEMA = """\
{
  "content": "multi-paragraph analysis grounded in the evidence",
  "confidence": "high|medium|low",
  "evidence_files": ["relative/path/a.py", "relative/path/b.ts"],
  "blind_spots": ["what is missing or uncertain"],
  "details": {}
}"""

DIRECTOR_JSON_SCHEMA = """\
{
  "section_order": ["architecture","conventions","feature_map","important_files","glossary","risk"],
  "max_results": {"architecture":24,"conventions":18,"feature_map":22,"important_files":20,"glossary":16,"risk":20},
  "notes": "one-line planning note"
}"""

BROKER_JSON_SCHEMA = """\
{
  "architecture": ["query 1","query 2"],
  "conventions": ["query 1","query 2"],
  "feature_map": ["query 1","query 2"],
  "important_files": ["query 1","query 2"],
  "glossary": ["query 1","query 2"],
  "risk": ["query 1","query 2"]
}"""

STRUCTURE_DETAILS_SCHEMA = """\
{
  "identity_card": ["what the repo does","primary stack","system role"],
  "architecture_overview": ["entrypoint","main modules","integrations","config","persistence hints"],
  "onboarding_digest": [
    {"file": "path/to/file.py", "why": "reason to read it", "outcome": "what developer learns"}
  ]
}"""

DOMAIN_RISK_DETAILS_SCHEMA = """\
{
  "feature_map": [
    {"feature": "feature name", "core_files": ["path1","path2"], "notes": "what it does"}
  ],
  "important_files_radar": [
    {"file": "path", "reason": "entrypoint|wiring hub|config spine|persistence core|blast radius"}
  ],
  "glossary": [
    {"term": "domain term", "meaning": "short meaning", "evidence": "path/to/file"}
  ],
  "risk_hotspots": [
    {"area": "area name", "why": "why risky", "files": ["path1","path2"]}
  ]
}"""

DIRECTOR_SYSTEM = f"""\
You are Run Director Agent for codebase intelligence analysis.
Decide retrieval depth per section and execution order that maximizes onboarding value.

Required output schema:
{DIRECTOR_JSON_SCHEMA}

{_JSON_ENFORCEMENT}"""

BROKER_SYSTEM = f"""\
You are Retrieval Broker Agent.
Convert high-level section goals into concrete retrieval queries.
Queries must use real technical tokens: framework names, architecture terms, file-role words.
Avoid vague queries like "get all files" or "find important code".

Required output schema:
{BROKER_JSON_SCHEMA}

{_JSON_ENFORCEMENT}"""

STRUCTURE_SYSTEM = f"""\
You are Structure Intelligence Agent.
Analyze the provided code evidence and produce sections A/B/C/G/H:
A) What the repo does and its system role
B) Architecture layers and module boundaries
C) Entrypoints and integration points
G) Most important files and why
H) Recommended reading order for a new developer

Fill "content" with a readable multi-paragraph summary.
Fill "details" with the exact structure below — no other shape accepted:
{STRUCTURE_DETAILS_SCHEMA}

Required output schema:
{SECTION_JSON_SCHEMA}

{_JSON_ENFORCEMENT}"""

CONVENTION_SYSTEM = f"""\
You are Convention Intelligence Agent.
You receive both static analysis findings (pre-computed) and RAG evidence excerpts.
Produce sections D/E:
D) Observed coding conventions — naming, module structure, folder roles, DI style, error handling
E) Hidden rules / anti-patterns — violations, inconsistencies, forbidden imports, suspected undocumented rules

Instructions:
- USE the static signals as ground truth for patterns that already have numeric backing
- USE the code excerpts to add nuance, examples, and catch things static analysis missed
- Produce actionable observations a new developer would immediately find valuable
- Keep "content" as readable paragraphs, not bullet soup
- If a convention is observed in < 3 files, mark it "suspected rule", not "observed rule"
- Do NOT invent conventions not supported by evidence

Required output schema:
{SECTION_JSON_SCHEMA}

{_JSON_ENFORCEMENT}"""

DOMAIN_RISK_SYSTEM = f"""\
You are Domain & Risk Intelligence Agent.
Analyze the provided code evidence and produce sections F/I/J:
F) Core feature map — what major features exist and which files implement them
I) Domain glossary — key domain terms found in the code and their meaning
J) Risk hotspots — files/areas that are complex, fragile, or have high blast radius

Fill "content" with a readable multi-paragraph summary.
Fill "details" with the exact structure below — no other shape accepted:
{DOMAIN_RISK_DETAILS_SCHEMA}

Required output schema:
{SECTION_JSON_SCHEMA}

{_JSON_ENFORCEMENT}"""

RISK_DETAILS_SCHEMA = """\
{
  "risk_findings": [
    {
      "category": "god_object|circular_import|todo_hotspot|test_gap|blast_radius|config_risk",
      "severity": "high|medium|low",
      "title": "short title",
      "rationale": "why this is risky",
      "evidence": ["path/to/file.py"]
    }
  ],
  "summary": "2-3 sentence executive summary of the risk profile"
}"""

RISK_COMPLEXITY_SYSTEM = f"""\
You are Risk & Complexity Intelligence Agent.
You receive pre-computed static risk findings and supporting code evidence.
Produce section J: Risk / Complexity / Unknowns

Instructions:
- The static findings are FACTS — do not contradict them, only add narrative and nuance
- Group findings by severity: high → medium → low
- For each finding, explain WHY it matters to a developer joining the team today
- Distinguish "certain" (e.g. file is 800 lines) from "suspected" (e.g. possible circular import)
- Keep "content" as readable paragraphs a tech lead would nod at
- Cap at ~6 most important findings if there are many — prioritise high severity

Fill "details" with the exact structure below:
{RISK_DETAILS_SCHEMA}

Required output schema:
{SECTION_JSON_SCHEMA}

{_JSON_ENFORCEMENT}"""

AUDITOR_SYSTEM = f"""\
You are Evidence Auditor & Composer Agent.
You receive draft sections from other agents and must:
1. Remove unsupported or invented claims
2. Normalise style for readability
3. Preserve useful blind spots
4. Keep section IDs exactly as received

Required output schema:
{{
  "sections": [
    {{
      "section": "A/B/C/G/H",
      "content": "cleaned content string",
      "confidence": "high|medium|low",
      "evidence_files": ["path1"],
      "blind_spots": ["..."]
    }}
  ],
  "confidence_summary": {{"high": 0, "medium": 0, "low": 0}}
}}

{_JSON_ENFORCEMENT}"""


def render_bundle(bundle: RetrievalBundle, limit: int = 14) -> str:
    parts: list[str] = []
    for i, ev in enumerate(bundle.evidences[:limit], start=1):
        excerpt = (ev.excerpt or "").strip().replace("\n", " ")
        if len(excerpt) > 360:
            excerpt = excerpt[:360].rstrip() + "..."
        parts.append(
            f"[{i}] file={ev.rel_path} chunk={ev.chunk_index} score={ev.score:.3f} "
            f"reasons={','.join(ev.reason_codes)} tokens={ev.token_estimate}\nexcerpt: {excerpt}"
        )
    return "\n\n".join(parts) if parts else "No evidence returned."


def build_director_user_prompt(snapshot_id: str, scan_mode: str) -> str:
    return (
        f"snapshot_id={snapshot_id}\n"
        f"scan_mode={scan_mode}\n"
        "Produce the analysis plan JSON object now."
    )


def build_broker_user_prompt(snapshot_id: str, section_order: list[str]) -> str:
    return (
        f"snapshot_id={snapshot_id}\n"
        f"section_order={section_order}\n"
        "Produce the retrieval queries JSON object now."
    )
