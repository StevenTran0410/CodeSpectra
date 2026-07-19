"""synthetic_agent_io: LLM-synthesized {input, gold} pairs at an agent's real LLM-call boundary, archetype-dispatched. Each case is self-validated against the agent's real output schema; a gold that fails validation past the retry ceiling is dropped, never kept as a best-effort guess."""
from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import json_repair
from pydantic import BaseModel

from agent_eval_harness.datasets.generator_utils import (
    apply_painpoint,
    config_kwarg_names_from_case_binding,
    strip_markdown_code_block,
)
from agent_eval_harness.datasets.types import DatasetCase
from agent_eval_harness.llm.client import LLMClient, LLMMessage
from agent_eval_harness.llm.embedding_client import EmbeddingClient
from agent_eval_harness.store.repository import new_id

logger = logging.getLogger("agent_eval_harness.datasets.generators.synthetic_agent_io")

# Small batches avoid truncating rich archetypes' JSON when count is large.
_MAX_CASES_PER_CALL = 3
_MAX_GENERATION_ROUNDS = 8
_MAX_CONSECUTIVE_DRY_ROUNDS = 2
_GENERATION_MAX_TOKENS = 20000
_CONFIDENCE_ENUM = ("low", "medium", "high")
_CONFIDENCE_FIELD_RULE = f"Any field literally named 'confidence' must be one of: {', '.join(_CONFIDENCE_ENUM)}."
_AVOID_LIMIT_CHARS = 2000

_GENERIC_PURPOSE_BY_ARCHETYPE: dict[str, str] = {
    "fan_in_judge": "Synthesizes and evaluates outputs from multiple upstream agents.",
    "rag_single_shot": "Analyzes retrieved code evidence and produces a structured output.",
    "rag_upstream": "Analyzes retrieved evidence combined with an upstream agent's output.",
    "rag_mem_ctx": "Analyzes retrieved evidence enriched with project-level context.",
    "rag_mem_ctx_participant": "Analyzes retrieved evidence with inherited project context.",
    "rag_query_planning": "Plans retrieval queries then analyzes the retrieved evidence.",
    "rag_query_planning_mem_ctx": "Plans queries with project context then analyzes evidence.",
}


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity; returns 0.0 when either vector is zero."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def _extract_text_values(inp: Any) -> str:
    """Joins string leaves at any depth; keys are never embedded because same-batch cases share them."""
    out: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, str):
            out.append(node)
        elif isinstance(node, dict):
            for k, v in node.items():
                if k != "shape":  # constant discriminator, identical across every case
                    walk(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                walk(v)

    walk(inp)
    return " | ".join(out)


def _generic_purpose_for_archetype(archetype: str) -> str:
    """Tier-3 fallback purpose when knowledge and analyst profile have nothing."""
    if archetype in _GENERIC_PURPOSE_BY_ARCHETYPE:
        return _GENERIC_PURPOSE_BY_ARCHETYPE[archetype]
    logger.warning("synthetic_agent_io: no generic purpose for archetype %r — using stub", archetype)
    return f"Evaluates agent behavior for the '{archetype}' archetype."


class SyntheticAgentIOConfig(BaseModel):
    dataset_name: str
    agent_id: str
    archetype: str
    contract: dict[str, Any]
    profile: dict[str, Any] = {}
    failure_modes: list[str] = []  # from AgentKnowledge; drives edge-case generation
    input_contract: list[dict[str, Any]] = []  # D9: AgentKnowledge sidecar ContractArg list (per-kwarg example)
    context_builders: list[dict[str, Any]] = []  # D9: AgentKnowledge sidecar ContextBuilderRef list
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
    """jsonschema.validate against the harvested output schema; empty schema -> treated as valid (no constraints)."""
    if json_schema is None:
        return []
    if isinstance(json_schema, dict) and not json_schema:
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


def _failure_modes_addendum(failure_modes: list[str], *, detailed: bool) -> str:
    """Shared by every prompt builder; `detailed` selects the fuller fan-in/rag-writer wording vs the generic builder's shorter one."""
    header = (
        "\n\nADDITIONALLY, include 1-2 extra cases covering EDGE-CASE or NEGATIVE scenarios "
        "based on the agent's known failure modes"
    )
    header += (
        ". Gold for these cases must reflect correct handling of the failure mode "
        "(graceful degradation or error, not the failure itself):\n"
        if detailed
        else ":\n"
    )
    return header + "\n".join(f"- {fm}" for fm in failure_modes[:5])


def _avoid_addendum(avoid: list[dict[str, Any]], *, detailed: bool) -> str:
    """Shared by every prompt builder; `detailed` selects the fuller fan-in/rag-writer wording vs the generic builder's shorter one."""
    if detailed:
        header = (
            f"\n\nThese {len(avoid)} candidate case(s) failed schema validation last attempt — "
            f"generate genuinely different, schema-valid replacements, not near-copies:\n"
        )
    else:
        header = (
            f"\n\nThese {len(avoid)} candidate(s) failed last attempt — generate different, "
            f"valid replacements:\n"
        )
    return header + json.dumps(avoid, ensure_ascii=False)[:_AVOID_LIMIT_CHARS]


def _fan_in_judge_prompt(
    parsed: SyntheticAgentIOConfig, n: int, avoid: list[dict[str, Any]] | None = None
) -> str:
    contract = parsed.contract
    output = contract.get("output") or {}
    json_schema = output.get("json_schema")
    if json_schema is None:
        json_schema = {}
    fields_by_letter = contract.get("field_downstream_consumers") or {}
    purpose = (parsed.profile or {}).get("purpose") or _generic_purpose_for_archetype(parsed.archetype)

    prompt = (
        f"You are generating realistic synthetic test cases for evaluating a fan-in judge "
        f"agent in a static-analysis pipeline.\n\n"
        f"AGENT PURPOSE: {purpose}\n\n"
        f"This agent reads a fixed subset of fields from each of several upstream 'section' "
        f"outputs (identified by single letters) and produces one JSON output summarizing/"
        f"synthesizing them. The agent NEVER sees any field beyond what's listed below — do "
        f"not invent additional fields per letter.\n\n"
        f"UPSTREAM FIELDS THE AGENT ACTUALLY READS, PER LETTER:\n{_upstream_field_spec(fields_by_letter)}\n\n"
        f"{_CONFIDENCE_FIELD_RULE}\n"
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
    if parsed.failure_modes:
        prompt += _failure_modes_addendum(parsed.failure_modes, detailed=True)
    if avoid:
        prompt += _avoid_addendum(avoid, detailed=True)
    return apply_painpoint(prompt, parsed.painpoint)


async def _generate_fan_in_judge(
    parsed: SyntheticAgentIOConfig, llm_client: LLMClient,
    embedding_client: EmbeddingClient | None = None,
) -> list[DatasetCase]:
    def build_prompt(n: int, avoid: list[dict[str, Any]] | None) -> str:
        return _fan_in_judge_prompt(parsed, n, avoid)

    return await _generate_validated_cases(
        parsed, llm_client, build_prompt,
        build_case_input=lambda inp: {"shape": "all_sections", "all_sections": inp},
        embedding_client=embedding_client,
    )


def _upstream_specs_for(parsed: SyntheticAgentIOConfig) -> list[tuple[str, str]]:
    """Harvested upstream_context_specs only (D3). The CodeSpectra-agent-id legacy override
    dict was removed once a green multi_agent/linear_rag Stage 1→3 run proved the generic/
    harvested path (CS-303 Slice 1/3b) — no agent-id fallback remains."""
    contract_specs = parsed.contract.get("upstream_context_specs") or []
    return [(s["name"], s["description"]) for s in contract_specs if s.get("name")]

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
    purpose = (parsed.profile or {}).get("purpose") or _generic_purpose_for_archetype(parsed.archetype)

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
    parts.append(_CONFIDENCE_FIELD_RULE)
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
    if parsed.failure_modes:
        prompt += _failure_modes_addendum(parsed.failure_modes, detailed=True)
    if avoid:
        prompt += _avoid_addendum(avoid, detailed=True)
    return apply_painpoint(prompt, parsed.painpoint)


def _shape_case_input(shape: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return lambda inp: {"shape": shape, **inp}


async def _generate_validated_cases(
    parsed: SyntheticAgentIOConfig,
    llm_client: LLMClient,
    build_prompt: Callable[[int, list[dict[str, Any]] | None], str],
    build_case_input: Callable[[dict[str, Any]], dict[str, Any]],
    embedding_client: EmbeddingClient | None = None,
) -> list[DatasetCase]:
    """Shared generation core for every archetype builder. Requests cases in small batches, stopping once count is reached, the round ceiling hits, or _MAX_CONSECUTIVE_DRY_ROUNDS add nothing."""
    output = parsed.contract.get("output") or {}
    json_schema = output.get("json_schema") or {}
    case_schema_hash = schema_hash(json_schema)

    accepted: list[dict[str, Any]] = []
    accepted_extra_labels: list[dict[str, Any]] = []  # parallel to accepted
    accepted_embeddings: list[list[float]] = []
    dedup_stash: list[tuple[float, dict[str, Any]]] = []  # (max_sim, candidate) for floor-guard reinstatement
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

            # Embedding dedup — degrades cleanly when client is absent or embedding fails.
            extra_labels: dict[str, Any] = {}
            if embedding_client is not None:
                try:
                    text = _extract_text_values(candidate["input"])
                    [emb] = await embedding_client.embed_texts([text])
                    if accepted_embeddings:
                        max_sim = max(_cosine_sim(emb, acc) for acc in accepted_embeddings)
                        extra_labels["max_sim"] = round(max_sim, 4)
                        if max_sim >= 0.93:
                            # Auto-drop: regenerate instead of shrinking the dataset.
                            dedup_stash.append((max_sim, candidate))
                            rejected_last_round.append(candidate)
                            continue
                        if max_sim >= 0.80:
                            extra_labels["near_duplicate"] = True
                    accepted_embeddings.append(emb)
                except Exception as e:
                    logger.debug(
                        "synthetic_agent_io[%s/%s]: embedding dedup check failed for a "
                        "candidate — accepting without dedup: %s",
                        parsed.agent_id, parsed.archetype, e,
                    )

            accepted.append(candidate)
            accepted_extra_labels.append(extra_labels)
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

    # Floor guard: if dedup auto-drops pushed us below count, reinstate least-similar dropped cases.
    if dedup_stash and len(accepted) < parsed.count:
        dedup_stash.sort(key=lambda t: t[0])  # ascending: lowest sim first (most diverse of the dropped)
        for stash_sim, stash_candidate in dedup_stash:
            if len(accepted) >= parsed.count:
                break
            accepted.append(stash_candidate)
            accepted_extra_labels.append({"max_sim": round(stash_sim, 4), "near_duplicate": True})

    if len(accepted) < parsed.count:
        logger.warning(
            "synthetic_agent_io[%s/%s]: generated %d/%d requested cases",
            parsed.agent_id, parsed.archetype, len(accepted), parsed.count,
        )

    cases: list[DatasetCase] = []
    for i, candidate in enumerate(accepted[: parsed.count]):
        extra = accepted_extra_labels[i] if i < len(accepted_extra_labels) else {}
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
                    **extra,
                },
                provenance="synthetic",
            )
        )
    return cases


async def _generate_rag_single_shot(
    parsed: SyntheticAgentIOConfig, llm_client: LLMClient,
    embedding_client: EmbeddingClient | None = None,
) -> list[DatasetCase]:
    """I, G — one fixed-query retrieval call, no upstream, no mem_ctx."""

    def build_prompt(n: int, avoid: list[dict[str, Any]] | None) -> str:
        return _rag_writer_prompt(
            parsed, n, upstream_specs=[], string_field_specs=[], query_planning=False, avoid=avoid,
        )

    return await _generate_validated_cases(
        parsed, llm_client, build_prompt, build_case_input=_shape_case_input("retrieval_only"),
        embedding_client=embedding_client,
    )


async def _generate_rag_upstream(
    parsed: SyntheticAgentIOConfig, llm_client: LLMClient,
    embedding_client: EmbeddingClient | None = None,
) -> list[DatasetCase]:
    """E, H — one fixed-query retrieval call + one upstream agent's raw output dict."""
    specs = _upstream_specs_for(parsed)

    def build_prompt(n: int, avoid: list[dict[str, Any]] | None) -> str:
        return _rag_writer_prompt(
            parsed, n, upstream_specs=specs, string_field_specs=[], query_planning=False, avoid=avoid,
        )

    return await _generate_validated_cases(
        parsed, llm_client, build_prompt, build_case_input=_shape_case_input("retrieval_and_upstream"),
        embedding_client=embedding_client,
    )


async def _generate_rag_mem_ctx(
    parsed: SyntheticAgentIOConfig, llm_client: LLMClient,
    embedding_client: EmbeddingClient | None = None,
) -> list[DatasetCase]:
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
        embedding_client=embedding_client,
    )


async def _generate_rag_mem_ctx_participant(
    parsed: SyntheticAgentIOConfig, llm_client: LLMClient,
    embedding_client: EmbeddingClient | None = None,
) -> list[DatasetCase]:
    """B, C — an inherited arch_bundle + folder_tree, optionally an upstream identity dict."""
    specs = _upstream_specs_for(parsed)

    def build_prompt(n: int, avoid: list[dict[str, Any]] | None) -> str:
        return _rag_writer_prompt(
            parsed, n, upstream_specs=specs, string_field_specs=[_FOLDER_TREE_SPEC],
            query_planning=False, avoid=avoid,
        )

    return await _generate_validated_cases(
        parsed, llm_client, build_prompt, build_case_input=_shape_case_input("mem_ctx_participant"),
        embedding_client=embedding_client,
    )


async def _generate_rag_query_planning(
    parsed: SyntheticAgentIOConfig, llm_client: LLMClient,
    embedding_client: EmbeddingClient | None = None,
) -> list[DatasetCase]:
    """D, J — a query-planning LLM sub-call (not simulated) + retrieve_multi, optionally one upstream agent output dict (D only)."""
    specs = _upstream_specs_for(parsed)

    def build_prompt(n: int, avoid: list[dict[str, Any]] | None) -> str:
        return _rag_writer_prompt(
            parsed, n, upstream_specs=specs, string_field_specs=[], query_planning=True, avoid=avoid,
        )

    return await _generate_validated_cases(
        parsed, llm_client, build_prompt, build_case_input=_shape_case_input("query_planning"),
        embedding_client=embedding_client,
    )


async def _generate_rag_query_planning_mem_ctx(
    parsed: SyntheticAgentIOConfig, llm_client: LLMClient,
    embedding_client: EmbeddingClient | None = None,
) -> list[DatasetCase]:
    """F — query-planning + retrieve_multi + a parallel frontend-screens retrieve, plus folder_tree and two upstream agent output dicts (identity, architecture)."""
    specs = _upstream_specs_for(parsed)

    def build_prompt(n: int, avoid: list[dict[str, Any]] | None) -> str:
        return _rag_writer_prompt(
            parsed, n, upstream_specs=specs, string_field_specs=[_FOLDER_TREE_SPEC],
            query_planning=True, avoid=avoid,
        )

    return await _generate_validated_cases(
        parsed, llm_client, build_prompt, build_case_input=_shape_case_input("query_planning_mem_ctx"),
        embedding_client=embedding_client,
    )


# CS-303 Slice 3c PAUSE: parity checked (tests/test_stage3_codespectra_parity.py) — for all 10
# shapes below, the generic path's field-descriptor key-set (raw kwarg names minus config)
# does NOT equal what these builders render (e.g. "mem_ctx" -> bundle+folder_tree+doc_ctx+
# manifest_ctx+repo_name is a semantic expansion, not a literal field). Removing this table
# would regress CodeSpectra dataset quality, so it is RETAINED. What generalizes foreign agents
# is the route-hole fix below (`not kwarg_names` -> generic) — a foreign kwarg set essentially
# never equals one of these 10 CodeSpectra-specific shapes, so it already falls through to
# _generate_generic regardless of whether this table exists. Revisit only if
# test_stage3_codespectra_parity.py ever starts asserting equality.
_KNOWN_SHAPE_KWARG_SETS: dict[str, frozenset[str]] = {
    "rag_single_shot:glossary": frozenset({"provider_id", "model_id", "snapshot_id", "profile"}),
    "rag_single_shot:important_files": frozenset({"provider_id", "model_id", "snapshot_id", "graph_summary", "profile"}),
    "rag_mem_ctx:project_identity": frozenset({"provider_id", "model_id", "snapshot_id", "repo_name", "mem_ctx", "profile"}),
    "rag_mem_ctx_participant:architecture": frozenset({"provider_id", "model_id", "snapshot_id", "graph_summary", "arch_bundle", "identity_output", "profile", "folder_tree"}),
    "rag_mem_ctx_participant:structure": frozenset({"provider_id", "model_id", "snapshot_id", "arch_bundle", "folder_tree", "identity_output", "profile"}),
    "rag_query_planning:conventions": frozenset({"provider_id", "model_id", "snapshot_id", "static_convention", "structure_output", "profile"}),
    "rag_query_planning:risk": frozenset({"provider_id", "model_id", "snapshot_id", "static_risk", "profile"}),
    "rag_upstream:violations": frozenset({"provider_id", "model_id", "snapshot_id", "static_convention", "static_risk", "conventions_output", "profile"}),
    "rag_upstream:onboarding": frozenset({"provider_id", "model_id", "snapshot_id", "important_files_output", "profile"}),
    "rag_query_planning_mem_ctx:feature_map": frozenset({"provider_id", "model_id", "snapshot_id", "graph_summary", "identity_output", "architecture_output", "profile", "folder_tree"}),
}

_KNOWN_KWARG_SET_VALUES: frozenset[frozenset[str]] = frozenset(_KNOWN_SHAPE_KWARG_SETS.values())

# Retained with the known-kwarg-set fast-path (see PAUSE note above _KNOWN_SHAPE_KWARG_SETS).
_ARCHETYPE_BUILDERS: dict[
    str,
    Callable[[SyntheticAgentIOConfig, LLMClient, EmbeddingClient | None], Awaitable[list[DatasetCase]]],
] = {
    "fan_in_judge": _generate_fan_in_judge,
    "rag_single_shot": _generate_rag_single_shot,
    "rag_upstream": _generate_rag_upstream,
    "rag_mem_ctx": _generate_rag_mem_ctx,
    "rag_mem_ctx_participant": _generate_rag_mem_ctx_participant,
    "rag_query_planning": _generate_rag_query_planning,
    "rag_query_planning_mem_ctx": _generate_rag_query_planning_mem_ctx,
}

async def _generate_generic(
    parsed: SyntheticAgentIOConfig, llm_client: LLMClient,
    embedding_client: EmbeddingClient | None = None,
) -> list[DatasetCase]:
    """Generic builder: derives the case-input field list mechanically from the harvested
    contract — never from a CodeSpectra-shaped template. Archetype only picks the purpose
    blurb below (D9); it never adds/removes a field."""
    contract = parsed.contract
    output = contract.get("output") or {}
    json_schema: dict | None = output.get("json_schema")
    schema_block = (
        json.dumps(json_schema, ensure_ascii=False)
        if json_schema
        else "(not available — schema_source=none; human review required)"
    )

    invocation = contract.get("invocation") or {}
    kwargs = invocation.get("kwargs") or []
    config_kwarg_names = config_kwarg_names_from_case_binding(invocation.get("case_binding"))
    upstream_context_specs = contract.get("upstream_context_specs") or []
    upstream_by_name = {s["name"]: s for s in upstream_context_specs if s.get("name")}
    builds_kwarg_by_name = {
        cb["builds_kwarg"]: cb for cb in (parsed.context_builders or []) if cb.get("builds_kwarg")
    }
    example_by_kwarg = {
        c["kwarg"]: c["example"] for c in (parsed.input_contract or [])
        if c.get("kwarg") and c.get("example")
    }

    field_descs: dict[str, str] = {}
    for kwarg in kwargs:
        name = kwarg.get("name") or ""
        if not name or name in config_kwarg_names:
            continue  # config-kind kwarg (harvested case_binding) — never part of the case input shape
        if name in upstream_by_name:
            field_descs[name] = upstream_by_name[name].get("description", "upstream agent output")
        elif name in builds_kwarg_by_name:
            cb = builds_kwarg_by_name[name]
            annotation = kwarg.get("annotation") or ""
            type_hint = f" ({annotation})" if annotation else ""
            field_descs[name] = (
                f"a simulated context block built by '{cb.get('name') or 'a context builder'}'"
                f"{type_hint} — plausible content consistent with the rest of this case"
            )
        else:
            annotation = kwarg.get("annotation") or ""
            desc = f"({annotation})" if annotation else "any value"
            example = example_by_kwarg.get(name)
            if example:
                desc += f", e.g. {example}"
            field_descs[name] = desc

    purpose = (parsed.profile or {}).get("purpose") or _generic_purpose_for_archetype(parsed.archetype)
    fields_block = json.dumps(field_descs, indent=2, ensure_ascii=False) if field_descs else "{}"

    def build_prompt(n: int, avoid: list[dict[str, Any]] | None) -> str:
        prompt = (
            f"You are generating synthetic test cases for evaluating an agent.\n\n"
            f"AGENT PURPOSE: {purpose}\n\n"
            f"INPUT SHAPE — each case's 'input' must have exactly these fields:\n{fields_block}\n\n"
            f"OUTPUT SCHEMA (for 'gold'):\n{schema_block}\n\n"
            f"Generate exactly {n} DISTINCT cases. Each case is an object with two keys: "
            f"'input' (matching the shape above) and 'gold' (the expected output, strictly "
            f"matching the schema).\n\n"
            f"Respond ONLY with a JSON array of {n} such objects. No markdown, no explanation."
        )
        if parsed.failure_modes:
            prompt += _failure_modes_addendum(parsed.failure_modes, detailed=False)
        if avoid:
            prompt += _avoid_addendum(avoid, detailed=False)
        return apply_painpoint(prompt, parsed.painpoint)

    return await _generate_validated_cases(
        parsed, llm_client, build_prompt, build_case_input=_shape_case_input("generic"),
        embedding_client=embedding_client,
    )


def _flag_schema_unvalidated(cases: list[DatasetCase], contract: dict[str, Any]) -> None:
    """Stamp needs_human on every case when the agent's output schema wasn't statically
    harvestable — gold was never schema-validated. Applied centrally so all archetype paths
    (fan_in_judge early return, known-kwarg fast-path, generic) are covered, not just generic."""
    note = next(
        (n for n in (contract.get("needs_human") or []) if "no output schema statically harvestable" in n),
        "output schema unavailable — gold was not schema-validated",
    )
    for case in cases:
        if isinstance(case.labels, dict):
            case.labels["schema_source"] = "none"
            case.labels["needs_human"] = note


async def _dispatch_builder(
    parsed: SyntheticAgentIOConfig, llm_client: LLMClient,
    embedding_client: EmbeddingClient | None,
) -> list[DatasetCase]:
    # fan_in_judge is a genuine structural special case (all-sections dict keyed by
    # field_downstream_consumers, not a flat kwarg list) — gated on contract shape via
    # _archetype_for, never on agent-id. See test_dispatch_fan_in_judge_routes_by_shape_not_agent_id.
    if parsed.archetype == "fan_in_judge":
        return await _generate_fan_in_judge(parsed, llm_client, embedding_client)
    # Fast-path for CodeSpectra's known kwarg shapes (CS-303 Slice 3c PAUSE — retained, see note
    # above _KNOWN_SHAPE_KWARG_SETS). An EMPTY kwarg set is NOT a known shape — an agent whose
    # harvest yielded nothing must degrade to the generic path, never a CodeSpectra builder.
    kwarg_names = frozenset(
        k["name"] for k in ((parsed.contract.get("invocation") or {}).get("kwargs") or []) if k.get("name")
    )
    if kwarg_names and kwarg_names in _KNOWN_KWARG_SET_VALUES:
        builder = _ARCHETYPE_BUILDERS.get(parsed.archetype)
        if builder is not None:
            return await builder(parsed, llm_client, embedding_client)
    return await _generate_generic(parsed, llm_client, embedding_client)


async def generate(
    config: dict, llm_client: LLMClient | None, seed: int | None = None,
    embedding_client: EmbeddingClient | None = None,
) -> list[DatasetCase]:
    parsed = SyntheticAgentIOConfig.model_validate(config)
    if llm_client is None:
        raise ValueError("LLM client is required for synthetic_agent_io generation")
    if parsed.archetype == "unimplemented":
        raise ValueError(
            "synthetic_agent_io archetype 'unimplemented' cannot generate cases — "
            "agent has no retrieval signal (check has_retrieval_signal in contract harvest)"
        )
    cases = await _dispatch_builder(parsed, llm_client, embedding_client)
    if not (parsed.contract.get("output") or {}).get("json_schema"):
        _flag_schema_unvalidated(cases, parsed.contract)
    return cases
