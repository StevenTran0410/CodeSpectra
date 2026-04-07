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
  "max_results": {
    "architecture":24,"conventions":18,"feature_map":22,
    "important_files":20,"glossary":16,"risk":20
  },
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
  "architecture_overview": [
    "entrypoint","main modules","integrations","config","persistence hints"
  ],
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
D) Observed coding conventions — naming, module structure, folder roles, DI style,
   error handling
E) Hidden rules / anti-patterns — violations, inconsistencies, forbidden imports,
   suspected undocumented rules

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


def render_bundle(bundle: RetrievalBundle, limit: int = 28, excerpt_chars: int = 1500) -> str:
    """Render retrieval bundle as LLM context.

    limit: max evidence items to include (default 28 — budgets now 5K-10K tokens).
    excerpt_chars: max chars per chunk sent to LLM. 1500 chars ≈ 375 tokens.
      boundary-expanded chunks may be ~3000 chars (two merged chunks); they are
      allowed through up to 3000 chars to preserve the full function body.
    """
    parts: list[str] = []
    for i, ev in enumerate(bundle.evidences[:limit], start=1):
        excerpt = (ev.excerpt or "").strip()
        # Boundary-expanded chunks get twice the room so the merged function is intact
        cap = excerpt_chars * 2 if "boundary-expanded" in ev.reason_codes else excerpt_chars
        if len(excerpt) > cap:
            excerpt = excerpt[:cap].rstrip() + "..."
        parts.append(
            f"[{i}] file={ev.rel_path} chunk={ev.chunk_index} score={ev.score:.3f} "
            f"reasons={','.join(ev.reason_codes)}\n{excerpt}"
        )
    return "\n\n---\n\n".join(parts) if parts else "No evidence returned."


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


AGENT_A_SCHEMA_STR = """{
  "repo_name": "string",
  "domain": "string",
  "purpose": "string",
  "runtime_type": "web_app|backend_service|monolith|monorepo|library|cli|worker|cron|unknown",
  "tech_stack": ["string"],
  "business_context": "string",
  "confidence": "high|medium|low",
  "evidence_files": ["string"],
  "blind_spots": ["string"]
}"""

AGENT_A_SYSTEM = f"""You are Project Identity Agent (section A only).
Infer repo_name, domain, purpose, runtime_type, and tech_stack from README, package manifests
(pyproject.toml, package.json, Cargo.toml, setup.py), and top-level config when present in the
evidence.
business_context: who uses this system and what outcome they care about (brief).
Output ONLY valid JSON matching this schema (no prose, no fences):

{AGENT_A_SCHEMA_STR}

{_JSON_ENFORCEMENT}"""

AGENT_G_SCHEMA_STR = """{
  "entrypoint": {"file": "string", "reason": "string"},
  "backbone": {"file": "string", "reason": "string"},
  "critical_config": {"file": "string", "reason": "string"},
  "highest_centrality": {"file": "string", "reason": "string"},
  "most_dangerous_to_touch": {"file": "string", "reason": "string"},
  "read_first": {"file": "string", "reason": "string"},
  "other_important": [{"file": "string", "reason": "string"}],
  "confidence": "high|medium|low",
  "evidence_files": ["string"],
  "blind_spots": ["string"]
}"""

AGENT_G_SYSTEM = f"""You are Important Files Radar Agent (section G only).
Fill exactly these six slots — each must name ONE file path that appears in the evidence (or graph
context):
entrypoint — main runtime entry (app main, CLI entry, worker main).
backbone — core wiring/module that ties the system together.
critical_config — single config file or module whose change has the widest blast radius.
highest_centrality — file that is most imported / most central (use graph centrality block when
provided).
most_dangerous_to_touch — highest blast radius if changed without care.
read_first — best first file for a new developer.
other_important — additional high-value files as {{file, reason}} pairs.
Output ONLY valid JSON matching this schema (no prose, no fences):

{AGENT_G_SCHEMA_STR}

{_JSON_ENFORCEMENT}"""

AGENT_I_SCHEMA_STR = """{
  "terms": [{"term": "string", "definition": "string", "evidence_files": ["string"]}],
  "confidence": "high|medium|low",
  "blind_spots": ["string"]
}"""

AGENT_I_SYSTEM = f"""You are Domain Glossary Agent (section I only).
Extract recurring domain and business terms from the evidence (not generic programming words like
"string", "API", "database").
Each term needs a precise definition grounded in how it is used; evidence_files must list relative
paths where it appears.
Output ONLY valid JSON matching this schema (no prose, no fences):

{AGENT_I_SCHEMA_STR}

{_JSON_ENFORCEMENT}"""

_AGENT_J_FINDING_CATEGORIES = (
    "god_object|circular_import|todo_hotspot|test_gap|blast_radius|config_risk|"
    "heavy_branching|generated_mixed"
)

AGENT_J_SCHEMA_STR = f"""{{
  "findings": [
    {{
      "category": "{_AGENT_J_FINDING_CATEGORIES}",
      "severity": "high|medium|low",
      "title": "string",
      "rationale": "string",
      "evidence": ["string"]
    }}
  ],
  "summary": "string",
  "confidence": "high|medium|low",
  "evidence_files": ["string"],
  "blind_spots": ["string"]
}}"""

AGENT_J_SYSTEM = f"""You are Risk & Complexity Agent for section J only.
Static risk findings provided in the user prompt are FACTS — do not contradict them; add narrative
and tie them to code evidence.
Produce findings grouped by severity mentally (high first), at most 8 findings total.
Output ONLY valid JSON matching this schema (no prose, no fences):

{AGENT_J_SCHEMA_STR}

{_JSON_ENFORCEMENT}"""

AGENT_B_SCHEMA_STR = """{
  "main_layers": ["string"],
  "frameworks": ["string"],
  "entrypoints": ["string"],
  "main_services": [{"name": "string", "path": "string", "role": "string"}],
  "external_integrations": ["string"],
  "config_sources": ["string"],
  "database_hints": ["string"],
  "confidence": "high|medium|low",
  "evidence_files": ["string"],
  "blind_spots": ["string"]
}"""

AGENT_B_SYSTEM = f"""You are Architecture Overview Agent (section B only).
Describe how the repo is built (not product/domain): layers, frameworks, startup entrypoints,
main services (name/path/role), external integrations, config sources, database hints.
Paths must come from evidence. Output ONE JSON object for this schema only:

{AGENT_B_SCHEMA_STR}

{_JSON_ENFORCEMENT}"""

AGENT_C_SCHEMA_STR = """{
  "folders": [
    {
      "path": "string",
      "role": "domain|infrastructure|delivery|shared|test|generated|unknown",
      "description": "string"
    }
  ],
  "summary": "string",
  "confidence": "high|medium|low",
  "evidence_files": ["string"],
  "blind_spots": ["string"]
}"""

AGENT_C_SYSTEM = f"""You are Repo Structure Agent (section C only).
Use the file listing and evidence only. Map significant folders to role
(domain|infrastructure|delivery|shared|test|generated|unknown), one concise description per folder,
then a short summary paragraph. Output ONE JSON object:

{AGENT_C_SCHEMA_STR}

{_JSON_ENFORCEMENT}"""

AGENT_H_SCHEMA_STR = """{
  "steps": [
    {
      "order": 1,
      "file": "string",
      "goal": "string",
      "outcome": "string"
    }
  ],
  "total_estimated_minutes": 30,
  "confidence": "high|medium|low",
  "evidence_files": ["string"],
  "blind_spots": ["string"]
}"""

AGENT_H_SYSTEM = f"""You are Onboarding Reading Order Agent (section H only).
Build a numbered reading path (~30 minutes) for a new developer: each step lists file, goal
(what to look for), outcome (what they will understand). Set total_estimated_minutes to your
overall time estimate. Evidence paths only. Output ONE JSON object:

{AGENT_H_SCHEMA_STR}

{_JSON_ENFORCEMENT}"""
