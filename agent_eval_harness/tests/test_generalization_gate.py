"""CS-303 Slice 5 — generalization grep gate (AEH_PHASE2_PLAN.md prime-directive §0 point 2).

Live-scans the inner package `agent_eval_harness/agent_eval_harness/**` (never `tests/` or
`test_targets/`) for CodeSpectra-specific tokens: the 12 agent-ids, `RetrievalService` as a
bare literal, `RunDirectorAgent`, `AnalysisAgentPipeline`, and an A-L section-letter-range
string. A fixed, size-asserted allowlist covers pre-existing, reviewed exceptions (Stage-4
code-injection files that are inherently CodeSpectra-target-specific per the architecture, the
Slice-3c-retained `_KNOWN_SHAPE_KWARG_SETS`, and a handful of generic-English-word false
positives on "violations" as an unrelated details-dict key) — a NEW hit anywhere else fails."""
from __future__ import annotations

import re
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).parent.parent / "agent_eval_harness"

_AGENT_IDS = (
    "glossary", "important_files", "project_identity", "architecture", "structure",
    "conventions", "risk", "violations", "onboarding", "feature_map", "auditor", "synthesizer",
)
_AGENT_ID_ALT = "|".join(re.escape(a) for a in _AGENT_IDS)

# Category -> compiled pattern. agent_id_literal matches a quoted string that IS exactly one of
# the 12 ids, optionally prefixed "lowercase_prefix:" (catches "archetype:agent_id" composite
# dispatch keys like _KNOWN_SHAPE_KWARG_SETS' — an exact-content match, not a substring search,
# so it doesn't sweep up the word inside unrelated prose/docstrings).
_TOKEN_PATTERNS: dict[str, re.Pattern[str]] = {
    "agent_id_literal": re.compile(r'''["'](?:[a-z_]+:)?(''' + _AGENT_ID_ALT + r')["\']'),
    "retrieval_service_literal": re.compile(r"\bRetrievalService\b"),
    "run_director_agent": re.compile(r"\bRunDirectorAgent\b"),
    "analysis_agent_pipeline": re.compile(r"\bAnalysisAgentPipeline\b"),
    "section_letter_range": re.compile(
        r'''["'](ABCDEFGHIJKL|ABCDEFGHIJK|ABCDEFGHIJ|ABCDEFGHI|ABCDEFGH|ABCDEFG|ABCDEF|ABCDE|ABCD|ABC)["']'''
    ),
}


def find_codespectra_tokens(text: str) -> dict[str, list[str]]:
    """Returns {category: [matched substrings]} for every category with >=1 hit in `text`."""
    hits: dict[str, list[str]] = {}
    for category, pattern in _TOKEN_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            hits[category] = matches
    return hits


# Pre-existing, reviewed exceptions — relative to the inner package root. Every entry is either
# (a) a Stage-4 code-injection file that is inherently CodeSpectra-target-specific by
# construction (it imports the TARGET's own domain classes to render runnable code for it —
# out of this ticket's Stage 1-3 scope), (b) the Slice-3c-retained _KNOWN_SHAPE_KWARG_SETS
# (see the PAUSE note beside it in synthetic_agent_io.py), or (c) a generic-English-word false
# positive on "violations" used as an unrelated details-dict key. A NEW file/category pair must
# be added here deliberately — bump the size assertion below in the same diff.
_ALLOWLIST: dict[str, frozenset[str]] = {
    "code_injection/templates/run_eval.py": frozenset({
        "retrieval_service_literal", "run_director_agent", "analysis_agent_pipeline",
    }),
    "code_injection/wiring.py": frozenset({"run_director_agent"}),
    "injection/collaborators.py": frozenset({"retrieval_service_literal"}),
    "discovery/contract_signals.yaml": frozenset({"retrieval_service_literal"}),
    "injection/agent_invokers.py": frozenset({"agent_id_literal"}),
    "datasets/generators/synthetic_agent_io.py": frozenset({"agent_id_literal"}),
    "injection/scoring.py": frozenset({"agent_id_literal"}),
    "metrics/assertions/allowed_downstream.py": frozenset({"agent_id_literal"}),
    "metrics/assertions/arg_schema.py": frozenset({"agent_id_literal"}),
    "metrics/assertions/field_match.py": frozenset({"agent_id_literal"}),
    "metrics/assertions/referential_integrity.py": frozenset({"agent_id_literal"}),
    "metrics/assertions/schema_valid.py": frozenset({"agent_id_literal"}),
}
_ALLOWLIST_SIZE = 12


def test_no_codespectra_tokens_in_production():
    assert len(_ALLOWLIST) == _ALLOWLIST_SIZE, (
        "allowlist grew or shrank without updating _ALLOWLIST_SIZE — review the diff"
    )

    violations: list[str] = []
    for path in sorted(_PACKAGE_ROOT.rglob("*")):
        if not path.is_file() or path.suffix not in (".py", ".yaml", ".yml"):
            continue
        rel = path.relative_to(_PACKAGE_ROOT).as_posix()
        if rel.startswith("tests/") or rel.startswith("test_targets/"):
            continue  # not reachable under _PACKAGE_ROOT today, but keep the guard explicit
        hits = find_codespectra_tokens(path.read_text(encoding="utf-8", errors="ignore"))
        allowed = _ALLOWLIST.get(rel, frozenset())
        for category, matches in hits.items():
            if category not in allowed:
                violations.append(f"{rel}: unexpected {category} hit(s): {matches}")

    assert not violations, "CodeSpectra token(s) found outside the allowlist:\n" + "\n".join(violations)


def test_gate_detects_each_token_category():
    """Mutation proof: plant one token per category and confirm the detector flags it."""
    samples = {
        "agent_id_literal": 'AGENT_DISPATCH = {"glossary": invoke}\n',
        "retrieval_service_literal": "svc = RetrievalService()\n",
        "run_director_agent": "director = RunDirectorAgent()\n",
        "analysis_agent_pipeline": "pipeline = AnalysisAgentPipeline()\n",
        "section_letter_range": 'letters = "ABCDEFGHIJK"\n',
    }
    for category, snippet in samples.items():
        hits = find_codespectra_tokens(snippet)
        assert category in hits, f"detector failed to flag planted {category!r} token: {snippet!r}"

    # negative control: ordinary code with none of the tokens must flag nothing.
    assert find_codespectra_tokens("def run(query: str) -> dict:\n    return {}\n") == {}
