"""Generalization grep gate: live-scans the inner package `agent_eval_harness/agent_eval_harness/**`
(never `tests/` or `test_targets/`) for CodeSpectra-specific tokens — the 12 agent-ids, bare
`RetrievalService`, `RunDirectorAgent`, `AnalysisAgentPipeline`, and an A-L section-letter-range
string. A fixed, size-asserted allowlist covers pre-existing, reviewed exceptions; a NEW hit
anywhere else fails."""
from __future__ import annotations

import re
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).parent.parent / "agent_eval_harness"

_AGENT_IDS = (
    "glossary", "important_files", "project_identity", "architecture", "structure",
    "conventions", "risk", "violations", "onboarding", "feature_map", "auditor", "synthesizer",
)
_AGENT_ID_ALT = "|".join(re.escape(a) for a in _AGENT_IDS)

# agent_id_literal matches a quoted string that IS exactly one of the 12 ids (optionally
# "prefix:"-qualified for composite dispatch keys) — an exact-content match, not a substring
# search, so it doesn't sweep up the word inside unrelated prose/docstrings.
_TOKEN_PATTERNS: dict[str, re.Pattern[str]] = {
    "agent_id_literal": re.compile(r'''["'](?:[a-z_]+:)?(''' + _AGENT_ID_ALT + r')["\']'),
    "retrieval_service_literal": re.compile(r"\bRetrievalService\b"),
    "run_director_agent": re.compile(r"\bRunDirectorAgent\b"),
    "analysis_agent_pipeline": re.compile(r"\bAnalysisAgentPipeline\b"),
    "section_letter_range": re.compile(
        r'''["'](ABCDEFGHIJKL|ABCDEFGHIJK|ABCDEFGHIJ|ABCDEFGHI|ABCDEFGH|ABCDEFG|ABCDEF|ABCDE|ABCD|ABC)["']'''
    ),
    # Convention-shaped hardcode (not a CodeSpectra literal, so blind to the 5 categories above): a
    # kept vocabulary constant, the exact shape _MODEL_CALL_VERBS/_has_model_evidence used to be.
    "lowercase_word_collection": re.compile(
        r'\b[A-Z_][A-Z0-9_]*\s*=\s*(?:frozenset\(\{|\()\s*'
        r'(?:"[a-z][a-z_.-]*"\s*,\s*){1,}(?:"[a-z][a-z_.-]*"\s*,?\s*)?[\)\}]'
    ),
    # A class/instance name's own suffix deciding something structurally, the same shape as the
    # (retained) endswith("Agent") survivor -- flags any NEW one so it gets the same scrutiny.
    "identifier_suffix_gate": re.compile(r'\.(?:endswith|startswith)\(\s*["\'][A-Z][A-Za-z]*["\']\s*\)'),
}


def find_codespectra_tokens(text: str) -> dict[str, list[str]]:
    """Returns {category: [matched substrings]} for every category with >=1 hit in `text`."""
    hits: dict[str, list[str]] = {}
    for category, pattern in _TOKEN_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            hits[category] = matches
    return hits


# Pre-existing, reviewed exceptions, relative to the inner package root: Stage-4 code-injection
# files that are inherently CodeSpectra-target-specific by construction, the retained
# _KNOWN_SHAPE_KWARG_SETS, and a generic-English-word false positive on "violations". A NEW
# file/category pair must be added here deliberately — bump the size assertion below.
_ALLOWLIST: dict[str, frozenset[str]] = {
    "code_injection/templates/run_eval.py": frozenset({
        "retrieval_service_literal", "run_director_agent", "analysis_agent_pipeline",
    }),
    "code_injection/wiring.py": frozenset({"run_director_agent"}),
    "injection/collaborators.py": frozenset({"retrieval_service_literal"}),
    "discovery/contract_signals.yaml": frozenset({"retrieval_service_literal"}),
    "injection/agent_invokers.py": frozenset({"agent_id_literal"}),
    "datasets/generators/synthetic_agent_io.py": frozenset({"agent_id_literal"}),
    "datasets/archetype_vocabulary.py": frozenset({"agent_id_literal"}),
    "injection/scoring.py": frozenset({"agent_id_literal"}),
    "metrics/assertions/allowed_downstream.py": frozenset({"agent_id_literal"}),
    "metrics/assertions/arg_schema.py": frozenset({"agent_id_literal"}),
    "metrics/assertions/field_match.py": frozenset({"agent_id_literal"}),
    "metrics/assertions/referential_integrity.py": frozenset({"agent_id_literal"}),
    "metrics/assertions/schema_valid.py": frozenset({"agent_id_literal"}),
}
_ALLOWLIST_SIZE = 13

# lowercase_word_collection / identifier_suffix_gate survivors: convention-shaped, not a CodeSpectra
# literal, so this is its OWN allowlist -- _ALLOWLIST/_ALLOWLIST_SIZE above stay untouched, zero new
# entries. Every entry here is a pre-existing, reviewed vocabulary constant or suffix check, verified
# this session by running both patterns over agent_eval_harness/agent_eval_harness/**.
_CONVENTION_CATEGORIES = frozenset({"lowercase_word_collection", "identifier_suffix_gate"})
_CONVENTION_ALLOWLIST: dict[str, frozenset[str]] = {
    "datasets/generators/synthetic_agent_io.py": frozenset({"lowercase_word_collection"}),
    "discovery/enrichment.py": frozenset({"lowercase_word_collection"}),
    "discovery/wiring.py": frozenset({"lowercase_word_collection", "identifier_suffix_gate"}),
    "mapping/builder/contract_harvest.py": frozenset({"lowercase_word_collection"}),
    "mapping/builder/roles.py": frozenset({"lowercase_word_collection"}),
    "mapping/builder/system_types.py": frozenset({"lowercase_word_collection"}),
    "mapping/builder/types.py": frozenset({"identifier_suffix_gate"}),
    "metrics/sweep.py": frozenset({"lowercase_word_collection"}),
    "planning/agentic_planner.py": frozenset({"lowercase_word_collection"}),
}
_CONVENTION_ALLOWLIST_SIZE = 9


def test_no_codespectra_tokens_in_production():
    assert len(_ALLOWLIST) == _ALLOWLIST_SIZE, (
        "allowlist grew or shrank without updating _ALLOWLIST_SIZE — review the diff"
    )
    assert len(_CONVENTION_ALLOWLIST) == _CONVENTION_ALLOWLIST_SIZE, (
        "convention allowlist grew or shrank without updating _CONVENTION_ALLOWLIST_SIZE -- review the diff"
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
        allowed_convention = _CONVENTION_ALLOWLIST.get(rel, frozenset())
        for category, matches in hits.items():
            allowlist = allowed_convention if category in _CONVENTION_CATEGORIES else allowed
            if category not in allowlist:
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
        "lowercase_word_collection": '_MY_VERBS = frozenset({"chat", "complete", "generate"})\n',
        "identifier_suffix_gate": 'if cls_name.endswith("Handler"):\n    pass\n',
    }
    for category, snippet in samples.items():
        hits = find_codespectra_tokens(snippet)
        assert category in hits, f"detector failed to flag planted {category!r} token: {snippet!r}"

    # negative control: ordinary code with none of the tokens must flag nothing.
    assert find_codespectra_tokens("def run(query: str) -> dict:\n    return {}\n") == {}


def test_contract_conventions_defaults_are_empty_not_codespectra_shaped() -> None:
    """CS-326 §2.7: the grep-based token gate above is blind to config.py's dataclass field
    defaults (a bare None isn't a quoted literal any _TOKEN_PATTERNS regex matches). Assert
    directly that ContractConventions() ships with no CodeSpectra-shaped default, so an
    unconfigured target gets an explicit "not detected" instead of silently inheriting
    CodeSpectra's own route/pattern/symbol conventions."""
    from agent_eval_harness.config import ContractConventions

    defaults = ContractConventions()
    assert defaults.rerun_section_route is None
    assert defaults.pipeline_fallback_name_pattern is None
    assert defaults.plan_queries_symbol is None


def test_decomposition_gold_has_no_hardcoded_max_items_per_call_literal() -> None:
    """CS-326 §2.7: same blind spot for decomposition_gold.py's constraint-name lookup -- the
    string "max_items_per_call" must not appear as a hardcoded literal in the source; it is only
    ever a caller-supplied value via DecompositionGoldConfig.max_items_constraint_name."""
    decomposition_gold_file = (
        Path(__file__).parent.parent / "agent_eval_harness" / "datasets" / "generators" / "decomposition_gold.py"
    )
    content = decomposition_gold_file.read_text(encoding="utf-8")
    assert '"max_items_per_call"' not in content
    assert "'max_items_per_call'" not in content
