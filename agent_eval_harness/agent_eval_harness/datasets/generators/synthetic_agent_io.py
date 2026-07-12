"""synthetic_agent_io: LLM-synthesized {input, gold} pairs at an agent's real LLM-call boundary, archetype-dispatched. Each case is self-validated against the agent's real output schema; a gold that fails validation past the retry ceiling is dropped, never kept as a best-effort guess."""
from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import json_repair
from pydantic import BaseModel

from agent_eval_harness.datasets.generator_utils import apply_painpoint, strip_markdown_code_block
from agent_eval_harness.datasets.types import DatasetCase
from agent_eval_harness.llm.client import LLMClient, LLMMessage
from agent_eval_harness.store.repository import new_id

logger = logging.getLogger("agent_eval_harness.datasets.generators.synthetic_agent_io")

# Small batches avoid truncating rich archetypes' JSON when count is large.
_MAX_CASES_PER_CALL = 3
_MAX_GENERATION_ROUNDS = 8
_MAX_CONSECUTIVE_DRY_ROUNDS = 2
_GENERATION_MAX_TOKENS = 20000
_CONFIDENCE_ENUM = ("low", "medium", "high")

_FALLBACK_PURPOSE: dict[str, str] = {
    "auditor": (
        "Meta-audits the other upstream sections' own self-reported confidence and blind "
        "spots, scoring per-section confidence, naming the weakest sections, and estimating "
        "overall coverage — without re-deriving any section's underlying content itself."
    ),
    "synthesizer": (
        "Writes the human-facing executive summary and narrative prose sections by "
        "synthesizing all upstream sections (including the auditor's own audit) — every "
        "claim in its output must be traceable to something an upstream section already said."
    ),
    "glossary": (
        "Extracts the repo's domain vocabulary (entities, types, event names, constants) "
        "from retrieved code evidence — each term must be grounded in a real, cited file."
    ),
    "important_files": (
        "Identifies the repo's structurally important files (entrypoint, backbone, "
        "critical config, highest-centrality, riskiest-to-touch, best-to-read-first) from "
        "retrieved code evidence, each with a one-line reason."
    ),
    "project_identity": (
        "Determines the repo's overall identity — domain, purpose, runtime type, tech "
        "stack, business context — from README/manifest/folder-tree/code evidence."
    ),
    "violations": (
        "Infers the repo's unwritten/negative conventions (banned patterns, anti-patterns) "
        "and flags concrete violations of them, grounded in retrieved evidence plus the "
        "upstream coding-conventions agent's own findings."
    ),
    "onboarding": (
        "Produces an ordered, time-estimated reading path for a new engineer onboarding "
        "onto this repo, informed by the upstream important-files agent's key-file picks."
    ),
    "architecture": (
        "Describes the repo's architecture — layers, frameworks, entrypoints, services, "
        "external integrations, config sources — from retrieved code evidence."
    ),
    "structure": (
        "Classifies each top-level folder by architectural role (domain/infrastructure/"
        "delivery/shared/test/generated/unknown) and writes a short structural narrative."
    ),
    "conventions": (
        "Documents the repo's actual coding conventions (naming, error handling, async "
        "style, DI, class-vs-functional, test style) as observed in retrieved evidence."
    ),
    "risk": (
        "Identifies risk/complexity hotspots (large files, deep nesting, TODO/FIXME "
        "clusters, high blast-radius modules) from retrieved evidence."
    ),
    "feature_map": (
        "Maps the repo's user-facing features to their entrypoints, key files, tests, and "
        "reading order, from retrieved evidence plus upstream identity/architecture context."
    ),
}


class SyntheticAgentIOConfig(BaseModel):
    dataset_name: str
    agent_id: str
    archetype: str
    contract: dict[str, Any]
    profile: dict[str, Any] = {}
    count: int = 20
    painpoint: str | None = None


def schema_hash(json_schema: dict[str, Any] | None) -> str:
    canonical = json.dumps(json_schema or {}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def is_dataset_stale(dataset_cases: list[dict[str, Any]], current_json_schema: dict[str, Any] | None) -> bool:
    """True if any case's stored schema_hash label no longer matches the agent's current contract schema. No cases or no labels -> never flagged stale."""
    current_hash = schema_hash(current_json_schema)
    for case in dataset_cases:
        labels_raw = case.get("labels_json")
        if not labels_raw:
            continue
        try:
            labels = json.loads(labels_raw) if isinstance(labels_raw, str) else labels_raw
        except json.JSONDecodeError:
            continue
        stored_hash = labels.get("schema_hash") if isinstance(labels, dict) else None
        if stored_hash and stored_hash != current_hash:
            return True
    return False


def _validate_gold(gold: Any, json_schema: dict[str, Any] | None) -> list[str]:
    """jsonschema.validate against the harvested output schema; empty schema -> treated as valid."""
    if not json_schema:
        return []
    if not isinstance(gold, dict):
        return ["gold is not a JSON object"]
    import jsonschema

    try:
        jsonschema.validate(instance=gold, schema=json_schema)
    except jsonschema.ValidationError as exc:
        return [exc.message]
    return []


def _parse_json_array(content: str) -> list[Any]:
    content = strip_markdown_code_block(content.strip())
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Truncated/malformed completions (e.g. cut off mid-object) are still salvageable —
        # json_repair recovers whichever leading array elements are structurally complete;
        # incomplete ones simply fail _validate_gold downstream like any other bad candidate.
        try:
            parsed = json_repair.loads(content)
        except Exception:
            return []
    if isinstance(parsed, dict):
        # Some models wrap the array in {"cases": [...]}; unwrap the first list value found.
        for value in parsed.values():
            if isinstance(value, list):
                return value
        return []
    return parsed if isinstance(parsed, list) else []


def _upstream_field_spec(field_downstream_consumers: dict[str, list[str]]) -> str:
    lines = []
    for letter in sorted(field_downstream_consumers):
        fields = field_downstream_consumers[letter]
        lines.append(f'  "{letter}": {{ {", ".join(fields)} }}')
    return "\n".join(lines)


def _fan_in_judge_prompt(
    parsed: SyntheticAgentIOConfig, n: int, avoid: list[dict[str, Any]] | None = None
) -> str:
    contract = parsed.contract
    output = contract.get("output") or {}
    json_schema = output.get("json_schema") or {}
    fields_by_letter = contract.get("field_downstream_consumers") or {}
    purpose = (parsed.profile or {}).get("purpose") or _FALLBACK_PURPOSE.get(
        parsed.agent_id, f"Agent '{parsed.agent_id}' consumes all upstream sections."
    )

    prompt = (
        f"You are generating realistic synthetic test cases for evaluating a fan-in judge "
        f"agent in a static-analysis pipeline.\n\n"
        f"AGENT PURPOSE: {purpose}\n\n"
        f"This agent reads a fixed subset of fields from each of several upstream 'section' "
        f"outputs (identified by single letters) and produces one JSON output summarizing/"
        f"synthesizing them. The agent NEVER sees any field beyond what's listed below — do "
        f"not invent additional fields per letter.\n\n"
        f"UPSTREAM FIELDS THE AGENT ACTUALLY READS, PER LETTER:\n{_upstream_field_spec(fields_by_letter)}\n\n"
        f"Any field literally named 'confidence' must be one of: {', '.join(_CONFIDENCE_ENUM)}.\n"
        f"Any field literally named 'blind_spots' must be a short list (0-3) of one-sentence "
        f"strings describing a specific gap in that section's own analysis.\n\n"
        f"THE AGENT'S OWN REQUIRED OUTPUT JSON SCHEMA (this is the 'gold' you must match):\n"
        f"{json.dumps(json_schema, ensure_ascii=False)}\n\n"
        f"Generate exactly {n} DISTINCT cases spanning a realistic range (some upstream "
        f"sections uniformly high-confidence/no blind spots, some with 2-3 sections weak or "
        f"contradictory, at least one borderline/ambiguous case). Each case is an object with "
        f"exactly two keys:\n"
        f'  "input": an object keyed by EVERY letter listed above, each value an object with '
        f"EXACTLY the fields listed for that letter (plausible fictional static-analysis-repo "
        f"content, internally consistent within the case)\n"
        f'  "gold": the single correct output this agent should produce for that "input", '
        f"strictly matching the schema above and GROUNDED ONLY in facts present in \"input\" "
        f"— never inventing a weak section, score, or claim that \"input\" doesn't support.\n\n"
        f"Respond ONLY with a JSON array of {n} such objects. No markdown, no explanation."
    )
    if avoid:
        prompt += (
            f"\n\nThese {len(avoid)} candidate case(s) failed schema validation last attempt — "
            f"generate genuinely different, schema-valid replacements, not near-copies:\n"
            f"{json.dumps(avoid, ensure_ascii=False)[:2000]}"
        )
    return apply_painpoint(prompt, parsed.painpoint)


async def _generate_fan_in_judge(
    parsed: SyntheticAgentIOConfig, llm_client: LLMClient
) -> list[DatasetCase]:
    def build_prompt(n: int, avoid: list[dict[str, Any]] | None) -> str:
        return _fan_in_judge_prompt(parsed, n, avoid)

    return await _generate_validated_cases(
        parsed, llm_client, build_prompt,
        build_case_input=lambda inp: {"shape": "all_sections", "all_sections": inp},
    )


_UPSTREAM_SPECS_BY_AGENT: dict[str, list[tuple[str, str]]] = {
    "violations": [(
        "conventions_output",
        "the D (coding-conventions) agent's own real output shape — must include a "
        "'signals' list of {category, description or pattern} objects",
    )],
    "onboarding": [(
        "important_files_output",
        "the G (important-files) agent's own real output shape — key file slots like "
        "entrypoint/backbone/read_first, each an object with {file, reason}",
    )],
    "conventions": [(
        "structure_output",
        "the C (structure) agent's own real output shape — a 'folders' list of "
        "{path, role, description} objects",
    )],
    "feature_map": [
        ("identity_output", "the A (project-identity) agent's own real output shape — domain/tech_stack/runtime_type"),
        ("architecture_output", "the B (architecture) agent's own real output shape — main_layers/main_services"),
    ],
}

# B/C/F, whose entry method takes a folder_tree kwarg (_archetype_for's mem_ctx-participation signal).
_MEM_CTX_PARTICIPANT_AGENTS: dict[str, list[tuple[str, str]]] = {
    "architecture": [("identity_output", "the A (project-identity) agent's own real output shape — domain/tech_stack/runtime_type")],
    "structure": [("identity_output", "the A (project-identity) agent's own real output shape — domain/tech_stack/runtime_type")],
}

_FOLDER_TREE_SPEC = (
    "folder_tree",
    "an indented directory-listing string for the SAME fictional repo (must include the "
    "directory of every rel_path used in bundle.evidences)",
)


def _rag_writer_prompt(
    parsed: SyntheticAgentIOConfig,
    n: int,
    *,
    upstream_specs: list[tuple[str, str]],
    string_field_specs: list[tuple[str, str]],
    query_planning: bool,
    avoid: list[dict[str, Any]] | None = None,
    extra_instruction: str | None = None,
) -> str:
    contract = parsed.contract
    output = contract.get("output") or {}
    json_schema = output.get("json_schema") or {}
    purpose = (parsed.profile or {}).get("purpose") or _FALLBACK_PURPOSE.get(
        parsed.agent_id, f"Agent '{parsed.agent_id}' analyzes retrieved code evidence."
    )

    parts = [
        "You are generating realistic synthetic test cases for evaluating a "
        "retrieval-augmented code-analysis agent.",
        f"AGENT PURPOSE: {purpose}",
        "This agent reads retrieved code-evidence chunks from a fictional repository you "
        "invent and produces one JSON output about that repository. Every claim in the gold "
        "output must be traceable to something in the evidence/context given — never invent "
        "a fact absent from the synthetic repo you create.",
        'The "input" object for each case must have exactly this shape:\n'
        '  "bundle": {"evidences": [{"rel_path": "<fictional file path>", '
        '"excerpt": "<plausible code/text snippet, 1-4 lines>", "score": <0.0-1.0>}, ...]} '
        "(4-10 evidences, internally consistent with each other, as if drawn from ONE real repo)",
    ]
    if string_field_specs:
        lines = "\n".join(f'  "{name}": {desc}' for name, desc in string_field_specs)
        parts.append(f"  Plus these string context fields:\n{lines}")
    if _FOLDER_TREE_SPEC in string_field_specs:
        parts.append(
            "CRITICAL (faithfulness guard): every bundle.evidences[].rel_path AND every path "
            "mentioned in any other context field MUST also appear in folder_tree — never "
            "reference a file absent from the folder tree you generate."
        )
    if upstream_specs:
        lines = "\n".join(f'  "{name}": {desc}' for name, desc in upstream_specs)
        parts.append(
            "  Plus these upstream agent output field(s) — plausible dicts matching that "
            f"agent's real shape, consistent with the same fictional repo:\n{lines}"
        )
    if query_planning:
        parts.append(
            "This agent also runs an internal query-planning LLM sub-call before retrieval "
            "in the real system — you do not simulate that separately, it has no bearing on "
            "the input/gold shape you produce here."
        )
    if extra_instruction:
        parts.append(extra_instruction)
    parts.append(f"Any field literally named 'confidence' must be one of: {', '.join(_CONFIDENCE_ENUM)}.")
    parts.append(
        "Keep any list-of-object fields (e.g. rules, violations_found, signals) realistic but "
        "short — 2-4 items each, never padded with filler entries."
    )
    parts.append(
        "THE AGENT'S OWN REQUIRED OUTPUT JSON SCHEMA (this is the 'gold' you must match):\n"
        f"{json.dumps(json_schema, ensure_ascii=False)}"
    )
    parts.append(
        f"Generate exactly {n} DISTINCT cases spanning a realistic range of repos/domains "
        '(vary language, size, and maturity). Each case is an object with exactly two keys: '
        '"input" (the shape above) and "gold" (the correct output for that input, strictly '
        "matching the schema, grounded only in the evidence/context given).\n\n"
        f"Respond ONLY with a JSON array of {n} such objects. No markdown, no explanation."
    )
    prompt = "\n\n".join(parts)
    if avoid:
        prompt += (
            f"\n\nThese {len(avoid)} candidate case(s) failed schema validation last attempt — "
            f"generate genuinely different, schema-valid replacements, not near-copies:\n"
            f"{json.dumps(avoid, ensure_ascii=False)[:2000]}"
        )
    return apply_painpoint(prompt, parsed.painpoint)


def _shape_case_input(shape: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return lambda inp: {"shape": shape, **inp}


async def _generate_validated_cases(
    parsed: SyntheticAgentIOConfig,
    llm_client: LLMClient,
    build_prompt: Callable[[int, list[dict[str, Any]] | None], str],
    build_case_input: Callable[[dict[str, Any]], dict[str, Any]],
) -> list[DatasetCase]:
    """Shared generation core for every archetype builder. Requests cases in small batches, stopping once count is reached, the round ceiling hits, or _MAX_CONSECUTIVE_DRY_ROUNDS add nothing."""
    output = parsed.contract.get("output") or {}
    json_schema = output.get("json_schema") or {}
    case_schema_hash = schema_hash(json_schema)

    accepted: list[dict[str, Any]] = []
    rejected_last_round: list[dict[str, Any]] = []
    dry_rounds = 0

    for round_idx in range(_MAX_GENERATION_ROUNDS):
        remaining = parsed.count - len(accepted)
        if remaining <= 0:
            break
        batch_n = min(remaining, _MAX_CASES_PER_CALL)
        resp = await llm_client.complete(
            [LLMMessage(role="user", content=build_prompt(batch_n, rejected_last_round or None))],
            max_tokens=_GENERATION_MAX_TOKENS,
            temperature=0.7,
            json_mode=True,
            reasoning_effort="low",
        )
        candidates = _parse_json_array(resp.content)
        rejected_last_round = []
        newly_accepted = 0
        for candidate in candidates:
            if not isinstance(candidate, dict) or "input" not in candidate or "gold" not in candidate:
                continue
            if not isinstance(candidate["input"], dict):
                continue
            errors = _validate_gold(candidate["gold"], json_schema)
            if errors:
                rejected_last_round.append(candidate)
                continue
            accepted.append(candidate)
            newly_accepted += 1

        if newly_accepted == 0:
            dry_rounds += 1
            if not candidates:
                logger.warning(
                    "synthetic_agent_io[%s/%s]: round %d produced an unparseable/empty "
                    "response (%d chars) — possible truncation at max_tokens=%d",
                    parsed.agent_id, parsed.archetype, round_idx + 1,
                    len(resp.content), _GENERATION_MAX_TOKENS,
                )
            elif rejected_last_round:
                logger.warning(
                    "synthetic_agent_io[%s/%s]: round %d — all %d candidate(s) failed "
                    "schema validation, e.g. %s",
                    parsed.agent_id, parsed.archetype, round_idx + 1,
                    len(rejected_last_round), _validate_gold(rejected_last_round[0].get("gold"), json_schema)[:1],
                )
            if dry_rounds >= _MAX_CONSECUTIVE_DRY_ROUNDS:
                break
        else:
            dry_rounds = 0

    if len(accepted) < parsed.count:
        logger.warning(
            "synthetic_agent_io[%s/%s]: generated %d/%d requested cases",
            parsed.agent_id, parsed.archetype, len(accepted), parsed.count,
        )

    cases: list[DatasetCase] = []
    for candidate in accepted[: parsed.count]:
        cases.append(
            DatasetCase(
                id=new_id(),
                dataset=parsed.dataset_name,
                kind="synthetic_agent_io",
                input=build_case_input(candidate["input"]),
                expected=candidate["gold"],
                labels={
                    "agent_id": parsed.agent_id,
                    "archetype": parsed.archetype,
                    "schema_hash": case_schema_hash,
                },
                provenance="synthetic",
            )
        )
    return cases


async def _generate_rag_single_shot(parsed: SyntheticAgentIOConfig, llm_client: LLMClient) -> list[DatasetCase]:
    """I, G — one fixed-query retrieval call, no upstream, no mem_ctx."""

    def build_prompt(n: int, avoid: list[dict[str, Any]] | None) -> str:
        return _rag_writer_prompt(
            parsed, n, upstream_specs=[], string_field_specs=[], query_planning=False, avoid=avoid,
        )

    return await _generate_validated_cases(
        parsed, llm_client, build_prompt, build_case_input=_shape_case_input("retrieval_only"),
    )


async def _generate_rag_upstream(parsed: SyntheticAgentIOConfig, llm_client: LLMClient) -> list[DatasetCase]:
    """E, H — one fixed-query retrieval call + one upstream agent's raw output dict."""
    specs = _UPSTREAM_SPECS_BY_AGENT.get(parsed.agent_id, [])

    def build_prompt(n: int, avoid: list[dict[str, Any]] | None) -> str:
        return _rag_writer_prompt(
            parsed, n, upstream_specs=specs, string_field_specs=[], query_planning=False, avoid=avoid,
        )

    return await _generate_validated_cases(
        parsed, llm_client, build_prompt, build_case_input=_shape_case_input("retrieval_and_upstream"),
    )


async def _generate_rag_mem_ctx(parsed: SyntheticAgentIOConfig, llm_client: LLMClient) -> list[DatasetCase]:
    """A — 4 mutually-consistent artifacts generated together per case so evidence paths are always a real subset of that case's synthetic folder_tree."""
    string_specs = [
        _FOLDER_TREE_SPEC,
        ("doc_ctx", "a short fictional README/CHANGELOG excerpt for the SAME repo"),
        ("manifest_ctx", "a short fictional manifest file (pyproject.toml/package.json-style) for the SAME repo"),
        ("repo_name", "a plausible repo name string for the SAME repo"),
    ]

    def build_prompt(n: int, avoid: list[dict[str, Any]] | None) -> str:
        return _rag_writer_prompt(
            parsed, n, upstream_specs=[], string_field_specs=string_specs, query_planning=False,
            avoid=avoid,
        )

    return await _generate_validated_cases(
        parsed, llm_client, build_prompt, build_case_input=_shape_case_input("mem_ctx_and_retrieval"),
    )


async def _generate_rag_mem_ctx_participant(parsed: SyntheticAgentIOConfig, llm_client: LLMClient) -> list[DatasetCase]:
    """B, C — an inherited arch_bundle + folder_tree, optionally an upstream identity dict."""
    specs = _MEM_CTX_PARTICIPANT_AGENTS.get(parsed.agent_id, [])

    def build_prompt(n: int, avoid: list[dict[str, Any]] | None) -> str:
        return _rag_writer_prompt(
            parsed, n, upstream_specs=specs, string_field_specs=[_FOLDER_TREE_SPEC],
            query_planning=False, avoid=avoid,
        )

    return await _generate_validated_cases(
        parsed, llm_client, build_prompt, build_case_input=_shape_case_input("mem_ctx_participant"),
    )


async def _generate_rag_query_planning(parsed: SyntheticAgentIOConfig, llm_client: LLMClient) -> list[DatasetCase]:
    """D, J — a query-planning LLM sub-call (not simulated) + retrieve_multi, optionally one upstream agent output dict (D only)."""
    specs = _UPSTREAM_SPECS_BY_AGENT.get(parsed.agent_id, [])

    def build_prompt(n: int, avoid: list[dict[str, Any]] | None) -> str:
        return _rag_writer_prompt(
            parsed, n, upstream_specs=specs, string_field_specs=[], query_planning=True, avoid=avoid,
        )

    return await _generate_validated_cases(
        parsed, llm_client, build_prompt, build_case_input=_shape_case_input("query_planning"),
    )


async def _generate_rag_query_planning_mem_ctx(parsed: SyntheticAgentIOConfig, llm_client: LLMClient) -> list[DatasetCase]:
    """F — query-planning + retrieve_multi + a parallel frontend-screens retrieve, plus folder_tree and two upstream agent output dicts (identity, architecture)."""
    specs = _UPSTREAM_SPECS_BY_AGENT.get(parsed.agent_id, [])

    def build_prompt(n: int, avoid: list[dict[str, Any]] | None) -> str:
        return _rag_writer_prompt(
            parsed, n, upstream_specs=specs, string_field_specs=[_FOLDER_TREE_SPEC],
            query_planning=True, avoid=avoid,
        )

    return await _generate_validated_cases(
        parsed, llm_client, build_prompt, build_case_input=_shape_case_input("query_planning_mem_ctx"),
    )


_ARCHETYPE_BUILDERS: dict[str, Callable[[SyntheticAgentIOConfig, LLMClient], Awaitable[list[DatasetCase]]]] = {
    "fan_in_judge": _generate_fan_in_judge,
    "rag_single_shot": _generate_rag_single_shot,
    "rag_upstream": _generate_rag_upstream,
    "rag_mem_ctx": _generate_rag_mem_ctx,
    "rag_mem_ctx_participant": _generate_rag_mem_ctx_participant,
    "rag_query_planning": _generate_rag_query_planning,
    "rag_query_planning_mem_ctx": _generate_rag_query_planning_mem_ctx,
}


async def generate(
    config: dict, llm_client: LLMClient | None, seed: int | None = None
) -> list[DatasetCase]:
    parsed = SyntheticAgentIOConfig.model_validate(config)
    if llm_client is None:
        raise ValueError("LLM client is required for synthetic_agent_io generation")
    builder = _ARCHETYPE_BUILDERS.get(parsed.archetype)
    if builder is None:
        raise ValueError(
            f"synthetic_agent_io archetype {parsed.archetype!r} not implemented yet "
            f"(CS-289 phasing — only {sorted(_ARCHETYPE_BUILDERS)} exist so far)"
        )
    return await builder(parsed, llm_client)
