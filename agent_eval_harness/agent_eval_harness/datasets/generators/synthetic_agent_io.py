"""synthetic_agent_io: LLM-synthesized {input, gold} pairs at an agent's real LLM-call boundary, archetype-dispatched. Each case is self-validated against the agent's real output schema; a gold that fails validation past the retry ceiling is dropped, never kept as a best-effort guess."""
from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from typing import Any

import json_repair
from pydantic import BaseModel

from agent_eval_harness.datasets.archetype_vocabulary import SYNTHETIC_ID_PLACEHOLDER
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
_GENERATION_MAX_TOKENS = 15000  # trimmed ~1/4 from 20000: less thinking room, faster rounds
_CONFIDENCE_ENUM = ("low", "medium", "high")
_CONFIDENCE_FIELD_RULE = f"Any field literally named 'confidence' must be one of: {', '.join(_CONFIDENCE_ENUM)}."
_AVOID_LIMIT_CHARS = 2000
_MAX_FAILURE_MODES_IN_PROMPT = 5
# Embedding dedup: >= this cosine-sim auto-drops a candidate as a near-exact repeat.
_DEDUP_AUTO_DROP_THRESHOLD = 0.93
# Embedding dedup: >= this cosine-sim (but below the drop threshold) flags near_duplicate=True.
_DEDUP_NEAR_DUPLICATE_THRESHOLD = 0.80

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
    failure_modes: list[str] = []
    input_contract: list[dict[str, Any]] = []
    input_schemas: dict[str, dict[str, Any]] = {}
    virtual_inputs: list[dict[str, Any]] = []
    context_builders: list[dict[str, Any]] = []
    section_output_schemas: dict[str, dict[str, Any]] = {}
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


def _validate_schema_enum_values(gold: Any, schema_enum_values: dict[str, list[str]]) -> list[str]:
    """Validate that field values in gold match harvested schema_enum_values domains."""
    if not schema_enum_values:
        return []
    errors: list[str] = []

    def _check_path(obj: Any, path_parts: list[str], allowed: list[str]) -> None:
        if not path_parts:
            if isinstance(obj, str) and obj not in allowed:
                errors.append(f"value '{obj}' not in allowed enum domain {allowed}")
            return
        curr, *rest = path_parts
        if curr.endswith("[]"):
            key = curr[:-2]
            val = obj.get(key) if isinstance(obj, dict) else None
            if isinstance(val, list):
                for item in val:
                    _check_path(item, rest, allowed)
        elif curr == "*":
            if isinstance(obj, dict):
                for item in obj.values():
                    _check_path(item, rest, allowed)
            elif isinstance(obj, list):
                for item in obj:
                    _check_path(item, rest, allowed)
        elif isinstance(obj, dict) and curr in obj:
            _check_path(obj[curr], rest, allowed)

    for field_path, allowed_vals in schema_enum_values.items():
        parts = field_path.split(".")
        _check_path(gold, parts, allowed_vals)

    return errors


def _coerce_input_object_strings(inp: Any) -> Any:
    """Coerces stringified JSON objects/arrays inside input to python dicts/lists across all archetypes."""
    if isinstance(inp, dict):
        for k, v in list(inp.items()):
            if isinstance(v, str) and v.strip().startswith(("{", "[")):
                try:
                    parsed_val = json.loads(v)
                except Exception:
                    try:
                        parsed_val = json_repair.loads(v)
                    except Exception:
                        parsed_val = None
                if isinstance(parsed_val, (dict, list)):
                    inp[k] = _coerce_input_object_strings(parsed_val)
            elif isinstance(v, (dict, list)):
                inp[k] = _coerce_input_object_strings(v)
    elif isinstance(inp, list):
        return [_coerce_input_object_strings(item) for item in inp]
    return inp


_DEFAULT_CONFIG_KWARGS = frozenset({
    "provider_id", "model_id", "snapshot_id", "profile",
    "session_id", "map_path", "dataset_name", "count",
})


def _validate_input(input_data: Any, parsed: SyntheticAgentIOConfig) -> list[str]:
    """Validates candidate input against harvested contract: required fields present, no foreign fields, matching types."""
    if not isinstance(input_data, dict):
        return ["input is not a JSON object"]

    contract = parsed.contract or {}
    invocation = contract.get("invocation") or {}
    kwargs = invocation.get("kwargs") or []
    cb = invocation.get("case_binding")
    config_names = config_kwarg_names_from_case_binding(cb) if cb else _DEFAULT_CONFIG_KWARGS
    upstream_context_specs = contract.get("upstream_context_specs") or []
    upstream_names = {s["name"] for s in upstream_context_specs if isinstance(s, dict) and s.get("name")}
    virtual_names = {vi["name"] for vi in (parsed.virtual_inputs or []) if isinstance(vi, dict) and vi.get("name")}
    input_schema_names = set(parsed.input_schemas or {})

    valid_kwarg_names = {
        kw.get("name") for kw in kwargs
        if isinstance(kw, dict) and kw.get("name") and kw.get("name") not in config_names
    }
    allowed_keys = (
        valid_kwarg_names
        | upstream_names
        | virtual_names
        | input_schema_names
        | {"shape", "all_sections"}
    )

    errors: list[str] = []

    # 1. Required kwargs present
    for kw in kwargs:
        if not isinstance(kw, dict):
            continue
        kname = kw.get("name")
        if not kname or kname in config_names:
            continue
        if kw.get("required") and kname not in input_data:
            if not kw.get("default_repr"):
                errors.append(f"missing required input field '{kname}'")

    # 2. No foreign fields (when contract specifies expected non-config input fields)
    expected_fields = valid_kwarg_names | upstream_names | virtual_names | input_schema_names
    if expected_fields:
        for k in input_data:
            if k not in allowed_keys:
                errors.append(f"foreign input field '{k}' not in harvested contract")

    # 3. Type matching for resolved object/array schemas
    all_schemas: dict[str, Any] = {}
    for kw in kwargs:
        if isinstance(kw, dict) and kw.get("name") and kw.get("resolved_schema"):
            all_schemas[kw["name"]] = kw["resolved_schema"]
    for s in upstream_context_specs:
        if isinstance(s, dict) and s.get("name") and (s.get("schema") or s.get("resolved_schema")):
            all_schemas[s["name"]] = s.get("schema") or s.get("resolved_schema")
    for fname, fschema in (parsed.input_schemas or {}).items():
        all_schemas[fname] = fschema

    import jsonschema

    for k, val in input_data.items():
        if k in all_schemas and val is not None:
            expected_schema = all_schemas[k]
            if isinstance(expected_schema, dict) and expected_schema:
                stype = expected_schema.get("type")
                if stype == "object" or "properties" in expected_schema:
                    if not isinstance(val, dict):
                        errors.append(f"field '{k}' expected object dict, got {type(val).__name__}")
                    else:
                        try:
                            jsonschema.validate(instance=val, schema=expected_schema)
                        except jsonschema.ValidationError as exc:
                            errors.append(f"field '{k}' schema validation failed: {exc.message}")
                elif stype == "array":
                    if not isinstance(val, list):
                        errors.append(f"field '{k}' expected list array, got {type(val).__name__}")
                    else:
                        try:
                            jsonschema.validate(instance=val, schema=expected_schema)
                        except jsonschema.ValidationError as exc:
                            errors.append(f"field '{k}' schema validation failed: {exc.message}")

    return errors




def _validate_gold(
    gold: Any,
    json_schema: dict[str, Any] | None,
    schema_enum_values: dict[str, list[str]] | None = None,
    cardinality: str | None = None,
    has_streamed_output: bool = False,
) -> list[str]:
    """jsonschema.validate against the harvested output schema + enum domain and cardinality constraints."""
    if cardinality == "object" and not isinstance(gold, dict):
        return ["gold is not a JSON object"]
    if cardinality == "array" and not isinstance(gold, list):
        return ["gold is not a JSON array"]

    if json_schema is None or (isinstance(json_schema, dict) and not json_schema):
        enum_errs = _validate_schema_enum_values(gold, schema_enum_values or {})
        return enum_errs

    if not isinstance(gold, dict):
        return ["gold is not a JSON object"]
    import jsonschema

    try:
        jsonschema.validate(instance=gold, schema=json_schema)
    except jsonschema.ValidationError as exc:
        return [exc.message]

    enum_errs = _validate_schema_enum_values(gold, schema_enum_values or {})
    if enum_errs:
        return enum_errs
    return []



def _parse_json_array(content: str) -> list[Any]:
    content = strip_markdown_code_block(content.strip())
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Truncated/malformed completions are salvageable: json_repair recovers whichever leading array elements are structurally complete.
        try:
            parsed = json_repair.loads(content)
        except Exception as e:
            logger.debug("synthetic_agent_io: json_repair also failed to parse content (%d chars): %s", len(content), e)
            return []
    if isinstance(parsed, dict):
        # A single candidate object (some models drop the array wrapper) — treat as a 1-element batch.
        if "input" in parsed and "gold" in parsed:
            return [parsed]
        # Some models wrap the array in {"cases": [...]}; unwrap the first list value found.
        for value in parsed.values():
            if isinstance(value, list):
                return value
        return []
    return parsed if isinstance(parsed, list) else []


def _summarize_schema(schema: Any, _depth: int = 0) -> Any:
    """Recursively summarize a resolved JSON schema to a nested type shape so prompts constrain complex types at every level."""
    if not isinstance(schema, dict) or _depth > 6:
        return "any"
    if schema.get("type") == "object" or "properties" in schema:
        props = schema.get("properties") or {}
        return {k: _summarize_schema(v, _depth + 1) for k, v in props.items()} or "object"
    if schema.get("type") == "array":
        items = schema.get("items")
        return [_summarize_schema(items, _depth + 1)] if isinstance(items, dict) else "array"
    # Surface enum domains (e.g. plan step "type") so the LLM emits a valid value instead of a
    # plausible synonym like "trace" that the input jsonschema check would then reject.
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return "|".join(str(v) for v in enum)
    return schema.get("type", "any")


def _upstream_field_spec(
    field_downstream_consumers: dict[str, list[str]],
    section_output_schemas: dict[str, dict[str, Any]] | None = None,
) -> str:
    lines = []
    section_schemas = section_output_schemas or {}
    for letter in sorted(field_downstream_consumers):
        fields = field_downstream_consumers[letter]
        sec_schema = section_schemas.get(letter)
        props = sec_schema.get("properties") if isinstance(sec_schema, dict) else {}
        if isinstance(props, dict) and props:
            field_parts = []
            for fname in fields:
                ftype = _summarize_schema(props.get(fname, {})) if fname in props else "any"
                field_parts.append(f'"{fname}": {json.dumps(ftype)}')
            lines.append(f'  "{letter}": {{ {", ".join(field_parts)} }}')
        else:
            lines.append(f'  "{letter}": {{ {", ".join(fields)} }}')
    return "\n".join(lines)


def _failure_modes_addendum(failure_modes: list[str], total_count: int, *, detailed: bool) -> str:
    """Shared by every prompt builder. Caps EDGE/failure cases at ~20% of the dataset (round); every other case must be a substantive success — the old 'include 1-2' let up to 40% come back empty."""
    n_fail = round(total_count * 0.2)
    if n_fail <= 0:
        return "\n\nEVERY case MUST be a normal, SUCCESSFUL case with substantive, NON-EMPTY output — do NOT produce empty/low-confidence/failure cases."
    header = (
        f"\n\nThis dataset has {total_count} cases total. Make EXACTLY {n_fail} of them EDGE-CASE / NEGATIVE "
        f"(failure-mode) cases; every OTHER case MUST be a normal, SUCCESSFUL case with substantive, NON-EMPTY "
        f"output — never produce more than {n_fail} empty/low-confidence case(s). The {n_fail} edge case(s) draw "
        f"on the agent's known failure modes"
    )
    header += (
        ". Gold for these cases must reflect correct handling of the failure mode "
        "(graceful degradation or error, not the failure itself):\n"
        if detailed
        else ":\n"
    )
    return header + "\n".join(f"- {fm}" for fm in failure_modes[:_MAX_FAILURE_MODES_IN_PROMPT])


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


_FAN_IN_JUDGE_SYSTEM = (
    "You are generating realistic synthetic test cases for evaluating a fan-in judge "
    "agent in a static-analysis pipeline.\n\n"
    "This agent reads a fixed subset of fields from each of several upstream 'section' "
    "outputs (identified by single letters) and produces one JSON output summarizing/"
    "synthesizing them. The agent NEVER sees any field beyond the ones provided per "
    "letter — do not invent additional fields per letter.\n\n"
    f"{_CONFIDENCE_FIELD_RULE}\n"
    "Any field literally named 'blind_spots' must be a short list (0-3) of one-sentence "
    "strings describing a specific gap in that section's own analysis.\n\n"
    "Each case is an object with exactly two keys:\n"
    '  "input": an object keyed by EVERY letter provided, each value an object with '
    "EXACTLY the fields listed for that letter (plausible fictional static-analysis-repo "
    "content, internally consistent within the case)\n"
    '  "gold": the single correct output this agent should produce for that "input", '
    "strictly matching the schema provided and GROUNDED ONLY in facts present in \"input\" "
    "— never inventing a weak section, score, or claim that \"input\" doesn't support.\n\n"
    "Respond ONLY with a JSON array of such objects. No markdown, no explanation."
)


def _enum_and_strategy_prompt_block(parsed: SyntheticAgentIOConfig) -> str:
    """Formats schema_enum_values, observability input_kind, has_separable_context, and has_streamed_output for prompt injection."""
    contract = parsed.contract or {}
    output = contract.get("output") or {}
    observability = contract.get("observability") or {}
    blocks: list[str] = []

    input_kind = observability.get("input_kind")
    if input_kind == "query":
        blocks.append("INPUT STRATEGY: Natural language user query framing a technical question or task.")
    elif input_kind == "structured":
        blocks.append("INPUT STRATEGY: Structured data parameters matching exact harvested schemas.")

    if observability.get("has_separable_context"):
        blocks.append("CONTEXT STRUCTURE: Separable context block.")

    enum_vals = output.get("schema_enum_values") or {}
    if enum_vals:
        enum_lines = [f'  "{k}": {json.dumps(v)}' for k, v in enum_vals.items()]
        blocks.append("ALLOWED ENUM DOMAINS FOR OUTPUT FIELDS (gold must use ONLY these values):\n" + "\n".join(enum_lines))

    if output.get("has_streamed_output"):
        blocks.append("STREAMED OUTPUT NOTE: The agent streams text at runtime; 'gold' MUST strictly contain ONLY the scorable JSON metadata output object.")

    return ("\n\n" + "\n\n".join(blocks)) if blocks else ""


def _fan_in_judge_prompt(
    parsed: SyntheticAgentIOConfig, n: int, avoid: list[dict[str, Any]] | None = None
) -> str:
    """Returns the user-message content only; _FAN_IN_JUDGE_SYSTEM carries the agent-invariant static rules."""
    contract = parsed.contract
    output = contract.get("output") or {}
    json_schema = output.get("json_schema")
    if json_schema is None:
        json_schema = {}
    fields_by_letter = contract.get("field_downstream_consumers") or {}
    purpose = (parsed.profile or {}).get("purpose") or _generic_purpose_for_archetype(parsed.archetype)

    agent_invariant = (
        f"AGENT PURPOSE: {purpose}\n\n"
        f"UPSTREAM FIELDS THE AGENT ACTUALLY READS, PER LETTER:\n"
        f"{_upstream_field_spec(fields_by_letter, parsed.section_output_schemas)}\n\n"
        f"THE AGENT'S OWN REQUIRED OUTPUT JSON SCHEMA (this is the 'gold' you must match):\n"
        f"{json.dumps(json_schema, ensure_ascii=False)}"
    )
    if parsed.failure_modes:
        agent_invariant += _failure_modes_addendum(parsed.failure_modes, parsed.count, detailed=True)
    agent_invariant += _enum_and_strategy_prompt_block(parsed)
    agent_invariant = apply_painpoint(agent_invariant, parsed.painpoint)

    round_variant = (
        f"\n\nGenerate exactly {n} DISTINCT cases spanning a realistic range (some upstream "
        f"sections uniformly high-confidence/no blind spots, some with 2-3 sections weak or "
        f"contradictory, at least one borderline/ambiguous case)."
    )
    if avoid:
        round_variant += _avoid_addendum(avoid, detailed=True)
    return agent_invariant + round_variant



async def _generate_fan_in_judge(
    parsed: SyntheticAgentIOConfig, llm_client: LLMClient,
    embedding_client: EmbeddingClient | None = None,
) -> list[DatasetCase]:
    def build_prompt(n: int, avoid: list[dict[str, Any]] | None) -> str:
        return _fan_in_judge_prompt(parsed, n, avoid)

    return await _generate_validated_cases(
        parsed, llm_client, _FAN_IN_JUDGE_SYSTEM, build_prompt,
        build_case_input=lambda inp: {"shape": "all_sections", "all_sections": inp},
        embedding_client=embedding_client,
    )


def _shape_case_input(shape: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return lambda inp: {"shape": shape, **inp}


async def _generate_validated_cases(
    parsed: SyntheticAgentIOConfig,
    llm_client: LLMClient,
    system_prompt: str,
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
    reasoning_effort_override: str | None = None

    for round_idx in range(_MAX_GENERATION_ROUNDS):
        remaining = parsed.count - len(accepted)
        if remaining <= 0:
            break
        batch_n = min(remaining, _MAX_CASES_PER_CALL)
        resp = await llm_client.complete(
            [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=build_prompt(batch_n, rejected_last_round or None)),
            ],
            max_tokens=_GENERATION_MAX_TOKENS,
            temperature=0.7,
            json_mode=True,
            reasoning_effort=reasoning_effort_override,
        )
        candidates = _parse_json_array(resp.content)
        rejected_last_round = []
        newly_accepted = 0
        output_contract = parsed.contract.get("output") or {}
        enum_values = output_contract.get("schema_enum_values") or {}
        cardinality = output_contract.get("cardinality")
        has_streamed = output_contract.get("has_streamed_output", False)

        for candidate in candidates:
            if not isinstance(candidate, dict) or "input" not in candidate or "gold" not in candidate:
                continue
            if not isinstance(candidate["input"], dict):
                continue
            candidate["input"] = _coerce_input_object_strings(candidate["input"])
            # Validate the input as the agent will receive it (post-shaping); fan_in nests raw sections under all_sections, so validating the pre-shape form wrongly flags them foreign/missing.
            input_errors = _validate_input(build_case_input(candidate["input"]), parsed)
            gold_errors = _validate_gold(
                candidate["gold"],
                json_schema,
                schema_enum_values=enum_values,
                cardinality=cardinality,
                has_streamed_output=has_streamed,
            )
            if input_errors or gold_errors:
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
                        if max_sim >= _DEDUP_AUTO_DROP_THRESHOLD:
                            # Auto-drop: regenerate instead of shrinking the dataset.
                            dedup_stash.append((max_sim, candidate))
                            rejected_last_round.append(candidate)
                            continue
                        if max_sim >= _DEDUP_NEAR_DUPLICATE_THRESHOLD:
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
                    "response (%d chars) — possible truncation at max_tokens=%d; "
                    "disabling reasoning for the retry",
                    parsed.agent_id, parsed.archetype, round_idx + 1,
                    len(resp.content), _GENERATION_MAX_TOKENS,
                )
                # Empty/truncated output usually means reasoning ate the token budget — turn it off
                # for the remaining rounds so the retry spends its budget on the JSON itself.
                reasoning_effort_override = "disable"
            elif rejected_last_round:
                logger.warning(
                    "synthetic_agent_io[%s/%s]: round %d — all %d candidate(s) failed "
                    "schema validation, e.g. %s",
                    parsed.agent_id, parsed.archetype, round_idx + 1,
                    len(rejected_last_round),
                    (_validate_input(build_case_input(rejected_last_round[0].get("input", {})), parsed) + _validate_gold(rejected_last_round[0].get("gold"), json_schema))[:1],
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

    def _with_required_ids(case_input: dict[str, Any]) -> dict[str, Any]:
        """Fills required string kwargs an archetype shape drops so the call still type-checks — omitting them made the agent reject its own request before any stub could answer."""
        invocation = parsed.contract.get("invocation") or {}
        config_names = config_kwarg_names_from_case_binding(invocation.get("case_binding"))
        for kwarg in invocation.get("kwargs") or []:
            name = kwarg.get("name") or ""
            annotation = (kwarg.get("annotation") or "str").replace(" ", "")
            if (
                not name
                or name in case_input
                or name in config_names
                or not kwarg.get("required")
                or annotation not in ("str", "str|None")
            ):
                continue
            case_input[name] = SYNTHETIC_ID_PLACEHOLDER
        return case_input

    cases: list[DatasetCase] = []
    for i, candidate in enumerate(accepted[: parsed.count]):
        extra = accepted_extra_labels[i] if i < len(accepted_extra_labels) else {}
        cases.append(
            DatasetCase(
                id=new_id(),
                dataset=parsed.dataset_name,
                kind="synthetic_agent_io",
                input=_with_required_ids(build_case_input(candidate["input"])),
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


_GENERIC_SYSTEM = (
    "You are generating synthetic test cases for evaluating an agent.\n\n"
    "Each case is an object with two keys: 'input' (matching the input shape provided) "
    "and 'gold' (the expected output, strictly matching the output schema provided).\n\n"
    "CRITICAL (faithfulness guard): keep every case internally consistent — every file "
    "path appearing anywhere in the input (evidence rel_paths, paths named in any context "
    "field) MUST also appear in any folder-tree / file-listing field present in that same "
    "input; never reference a file absent from the folder tree you generate. Every claim "
    "in 'gold' must be traceable to the input evidence you created. Any 'blind_spots' or "
    "fallback/degradation notes in 'gold' MUST be grounded in and consistent with 'input' — "
    "never assert absence or failure of a feature/section when present in 'input'.\n\n"
    "Respond ONLY with a JSON array of such objects. No markdown, no explanation."
)


async def _generate_generic(
    parsed: SyntheticAgentIOConfig, llm_client: LLMClient,
    embedding_client: EmbeddingClient | None = None,
) -> list[DatasetCase]:
    """Generic builder: derives the case-input field list mechanically from the harvested contract; archetype only picks the purpose blurb, never adds/removes a field."""
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

    # _summarize_schema is module-level (shared with _upstream_field_spec).

    def _coerce_to_schema(value: Any, schema: Any, _depth: int = 0) -> Any:
        """Deterministically forces a generated value to structurally match its resolved schema at every level (object/array/scalar), guaranteeing faithful structure regardless of LLM drift; callers skip None so an absent Optional field stays absent."""
        if not isinstance(schema, dict) or _depth > 8:
            return value
        stype = schema.get("type")
        if stype == "object" or "properties" in schema:
            if isinstance(value, str) and value.strip():
                try:
                    value = json.loads(value)
                except Exception:
                    try:
                        value = json_repair.loads(value)
                    except Exception:
                        pass
            props = schema.get("properties") or {}
            if not props:
                return value
            src = value if isinstance(value, dict) else {}
            return {k: _coerce_to_schema(src.get(k), sub, _depth + 1) for k, sub in props.items()}
        if stype == "array":
            item_schema = schema.get("items") if isinstance(schema.get("items"), dict) else {}
            if isinstance(value, str) and value.strip():
                try:
                    value = json.loads(value)
                except Exception:
                    try:
                        value = json_repair.loads(value)
                    except Exception:
                        pass
            if isinstance(value, list):
                return [_coerce_to_schema(v, item_schema, _depth + 1) for v in value]
            if isinstance(value, str) and value.strip():
                parts = [p.strip() for p in value.replace(",", "\n").splitlines() if p.strip()]
                return [_coerce_to_schema(p, item_schema, _depth + 1) for p in parts]
            return []
        return value

    field_schemas: dict[str, Any] = {}

    # Collect object schemas from KwargSpec resolved_schema and upstream_context_specs
    for kwarg in kwargs:
        if not isinstance(kwarg, dict):
            continue
        name = kwarg.get("name")
        res_schema = kwarg.get("resolved_schema")
        if name and name not in config_kwarg_names and isinstance(res_schema, dict):
            field_schemas[name] = res_schema
            schema_summary = _summarize_schema(res_schema)
            if isinstance(schema_summary, dict) and schema_summary:
                desc = (
                    f"a JSON object that MUST use EXACTLY the keys shown here at EVERY level — no renaming, "
                    f"no omissions, no extra keys — matching this nested structure (values are the types): "
                    f"{json.dumps(schema_summary)}"
                )
                field_descs[name] = desc

    for spec in upstream_context_specs:
        if not isinstance(spec, dict):
            continue
        name = spec.get("name")
        res_schema = spec.get("schema") or spec.get("resolved_schema")
        if name and isinstance(res_schema, dict):
            field_schemas[name] = res_schema
            schema_summary = _summarize_schema(res_schema)
            if isinstance(schema_summary, dict) and schema_summary:
                desc = (
                    f"a JSON object that MUST use EXACTLY the keys shown here at EVERY level — no renaming, "
                    f"no omissions, no extra keys — matching this nested structure (values are the types): "
                    f"{json.dumps(schema_summary)}"
                )
                if spec.get("description"):
                    desc = spec["description"] + " — " + desc
                field_descs[name] = desc

    # Overwrites the generic kwarg desc with the resolved nested schema so complex typed kwargs generate faithful structures; must not skip kwargs already in field_descs.
    for kwarg_name, kwarg_schema in (parsed.input_schemas or {}).items():
        schema_summary = _summarize_schema(kwarg_schema)
        if not isinstance(schema_summary, dict) or not schema_summary:
            continue  # type resolved to nothing usable — leave the existing desc
        desc = (
            f"a JSON object that MUST use EXACTLY the keys shown here at EVERY level — no renaming, "
            f"no omissions, no extra keys — matching this nested structure (values are the types): "
            f"{json.dumps(schema_summary)}"
        )
        if isinstance(kwarg_schema, dict) and kwarg_schema.get("description"):
            desc = kwarg_schema["description"] + " — " + desc
        field_descs[kwarg_name] = desc
        field_schemas[kwarg_name] = kwarg_schema

    # Process virtual_inputs (harvested retrieval bundles) — same strict-keys treatment.
    for vi in (parsed.virtual_inputs or []):
        name = vi.get("name")
        if not name:
            continue
        if name in field_schemas:
            continue
        fields = vi.get("fields") or []
        schema_summary = {f["name"]: _summarize_schema(f.get("field_schema") or f.get("schema") or {}) for f in fields if f.get("name")}
        if not schema_summary:
            field_descs.setdefault(name, "the agent's internally-retrieved evidence object")
            continue
        example = next((f.get("example") for f in fields if f.get("example")), None)
        desc = (
            f"the agent's internally-retrieved evidence object; it MUST use EXACTLY the keys shown here "
            f"at EVERY level — no renaming, no omissions, no extra keys — matching this nested structure "
            f"(values are the types): {json.dumps(schema_summary)}"
        )
        if example:
            desc += f", e.g. {example}"
        field_descs[name] = desc
        field_schemas[name] = {
            "type": "object",
            "properties": {f["name"]: (f.get("field_schema") or f.get("schema") or {}) for f in fields if f.get("name")},
        }


    purpose = (parsed.profile or {}).get("purpose") or _generic_purpose_for_archetype(parsed.archetype)
    fields_block = json.dumps(field_descs, indent=2, ensure_ascii=False) if field_descs else "{}"

    def build_prompt(n: int, avoid: list[dict[str, Any]] | None) -> str:
        agent_invariant = (
            f"INPUT SHAPE — each case's 'input' must have exactly these fields:\n{fields_block}\n\n"
            f"AGENT PURPOSE: {purpose}\n\n"
            f"OUTPUT SCHEMA (for 'gold'):\n{schema_block}"
        )
        if parsed.failure_modes:
            agent_invariant += _failure_modes_addendum(parsed.failure_modes, parsed.count, detailed=False)
        agent_invariant += _enum_and_strategy_prompt_block(parsed)
        agent_invariant = apply_painpoint(agent_invariant, parsed.painpoint)

        round_variant = f"\n\nGenerate exactly {n} DISTINCT cases."
        if avoid:
            round_variant += _avoid_addendum(avoid, detailed=False)
        return agent_invariant + round_variant

    cases = await _generate_validated_cases(
        parsed, llm_client, _GENERIC_SYSTEM, build_prompt, build_case_input=_shape_case_input("generic"),
        embedding_client=embedding_client,
    )
    # Enforces faithful structure (keys/types/nesting) on every schema-typed field after generation so nested fields match the resolved type even when the LLM drifts; None stays untouched.
    if field_schemas:
        for case in cases:
            if isinstance(case.input, dict):
                for fname, fschema in field_schemas.items():
                    if case.input.get(fname) is not None:
                        case.input[fname] = _coerce_to_schema(case.input[fname], fschema)
    return cases


def _flag_schema_unvalidated(cases: list[DatasetCase], contract: dict[str, Any]) -> None:
    """Stamps needs_human on every case when the agent's output schema wasn't statically harvestable — applied centrally so both archetype paths (fan_in_judge, generic) are covered."""
    note = next(
        (n for n in (contract.get("needs_human") or []) if "no output schema statically harvestable" in n),
        "output schema unavailable — gold was not schema-validated",
    )
    for case in cases:
        if isinstance(case.labels, dict):
            case.labels["schema_source"] = "none"
            case.labels["needs_human"] = note


def _has_single_resolved_object_kwarg(parsed: SyntheticAgentIOConfig) -> bool:
    """True if agent has exactly one non-config invocation kwarg that resolves to an object schema."""
    contract = parsed.contract or {}
    invocation = contract.get("invocation") or {}
    kwargs = invocation.get("kwargs") or []
    cb = invocation.get("case_binding")
    config_names = config_kwarg_names_from_case_binding(cb) if cb else _DEFAULT_CONFIG_KWARGS
    non_config_kwargs = [
        kw for kw in kwargs
        if isinstance(kw, dict) and kw.get("name") and kw.get("name") not in config_names
    ]
    if len(non_config_kwargs) == 1:
        kw = non_config_kwargs[0]
        schema = kw.get("resolved_schema")
        if isinstance(schema, dict) and (schema.get("type") == "object" or "properties" in schema):
            return True
    return False


async def _dispatch_builder(
    parsed: SyntheticAgentIOConfig, llm_client: LLMClient,
    embedding_client: EmbeddingClient | None,
) -> list[DatasetCase]:
    # Only fan_in_judge has a specialized path, unless the agent takes a single resolved object kwarg (e.g. LangGraph state node)
    if parsed.archetype == "fan_in_judge" and not _has_single_resolved_object_kwarg(parsed):
        return await _generate_fan_in_judge(parsed, llm_client, embedding_client)
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
