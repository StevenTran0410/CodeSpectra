"""Derive per-case heuristic flags and per-agent verdicts for AEH evaluation runs.

Heuristics classify low evaluation scores into actionable insights:
1. SOFT_FAIL: Subjective or estimated field exact mismatch where values have matching types or fall within 20% tolerance.
2. EXPECTED_SUSPECT: Gold expected output contains system failure sentinels or is near-empty while actual result is substantive.
3. INPUT_STARVED: Driving input is missing required retrieval content or DR plan_cursor is at/past end or snapshot_id is placeholder.
4. SCHEMA_SUSPECT: Container/nested type mismatch between gold expected and actual result, or gold populates fields that result resets to None.
"""
from __future__ import annotations

import re
from typing import Any, TypedDict


SUBJECTIVE_FIELDS = {"confidence", "overall_confidence"}
ESTIMATE_FIELDS = {"coverage_percentage", "total_estimated_minutes"}

# Narrowed sentinel to system/retrieval failures only — broad phrases like "no evidence" /
# "unable to" are NOT here: they match legitimate "nothing found" golds where the agent, not
# the dataset, is at fault (e.g. a hallucinated glossary against a correctly-empty gold).
FAILURE_SENTINEL_REGEX = re.compile(
    r"(?i)(retrieval (error|fail)|agent failed|output_repair_failed)"
)


class CaseFlags(TypedDict):
    soft_fail_fields: list[str]
    expected_suspect: bool
    input_starved: bool
    schema_suspect_fields: list[str]
    reasons: list[str]


class AgentVerdict(TypedDict):
    verdict: str  # "healthy" | "likely-dataset" | "likely-agent" | "mixed"
    d_count: int
    a_count: int
    low_count: int


def _is_substantive(obj: Any) -> bool:
    """True if object contains non-trivial content (non-empty list/dict, number, bool, or non-empty string)."""
    if obj is None:
        return False
    if isinstance(obj, (int, float, bool)):
        return True
    if isinstance(obj, str):
        return len(obj.strip()) > 0
    if isinstance(obj, (list, tuple, set)):
        return len(obj) > 0 and any(_is_substantive(item) for item in obj)
    if isinstance(obj, dict):
        return len(obj) > 0 and any(_is_substantive(v) for v in obj.values() if v is not None)
    return False


def _contains_failure_sentinel(obj: Any) -> bool:
    """Recursively check if string values match narrowed system failure sentinel patterns."""
    if isinstance(obj, str):
        return bool(FAILURE_SENTINEL_REGEX.search(obj))
    if isinstance(obj, (list, tuple)):
        return any(_contains_failure_sentinel(item) for item in obj)
    if isinstance(obj, dict):
        return any(_contains_failure_sentinel(v) for v in obj.values())
    return False


def _is_near_empty(obj: Any) -> bool:
    """True if object is structurally empty (empty lists/dicts and empty/<=3 char strings)."""
    if obj is None:
        return True
    if isinstance(obj, str):
        return len(obj.strip()) <= 3
    if isinstance(obj, (int, float, bool)):
        return False
    if isinstance(obj, (list, tuple)):
        return len(obj) == 0 or all(_is_near_empty(item) for item in obj)
    if isinstance(obj, dict):
        return len(obj) == 0 or all(_is_near_empty(v) for v in obj.values())
    return False


def _get_type_category(val: Any) -> str:
    """Return container/scalar type category string."""
    if val is None:
        return "none"
    if isinstance(val, bool):
        return "boolean"
    if isinstance(val, (int, float)):
        return "numeric"
    if isinstance(val, str):
        return "string"
    if isinstance(val, dict):
        return "dict"
    if isinstance(val, (list, tuple, set)):
        return "list"
    return "other"


def derive_case_flags(
    input_obj: Any,
    result_obj: Any,
    expected_obj: Any,
    evaluations: list[dict[str, Any]] | None = None,
) -> CaseFlags:
    """Derive heuristic flags for a single evaluation case."""
    soft_fail_fields: list[str] = []
    expected_suspect = False
    input_starved = False
    schema_suspect_fields: list[str] = []
    reasons: list[str] = []

    evals = evaluations or []

    # 1. SOFT_FAIL heuristic
    for ev in evals:
        mname = ev.get("metric_name", "")
        score = ev.get("score")
        if mname.startswith("field_exact.") and (score is None or score < 0.5):
            field_name = mname.replace("field_exact.", "")
            details = ev.get("details") or {}
            actual = details.get("actual")
            expected = details.get("expected")

            if actual is not None and expected is not None:
                act_cat = _get_type_category(actual)
                exp_cat = _get_type_category(expected)

                if field_name in SUBJECTIVE_FIELDS:
                    if act_cat == exp_cat:
                        soft_fail_fields.append(field_name)
                        reasons.append(f"soft_fail on '{field_name}': subjective value diff ({actual!r} vs {expected!r})")
                elif field_name in ESTIMATE_FIELDS:
                    if act_cat == exp_cat:
                        if act_cat == "numeric":
                            try:
                                act_num = float(actual)
                                exp_num = float(expected)
                                diff = abs(act_num - exp_num)
                                denom = max(abs(exp_num), 1.0)
                                if (diff / denom) <= 0.20:
                                    soft_fail_fields.append(field_name)
                                    reasons.append(f"soft_fail on '{field_name}': estimate value within 20% tolerance ({act_num} vs {exp_num})")
                            except (ValueError, TypeError):
                                soft_fail_fields.append(field_name)
                                reasons.append(f"soft_fail on '{field_name}': estimate value diff")
                        else:
                            soft_fail_fields.append(field_name)
                            reasons.append(f"soft_fail on '{field_name}': estimate string diff")

    # 2. EXPECTED_SUSPECT heuristic (narrowed sentinel + substantive result check)
    if expected_obj is not None:
        has_sentinel = _contains_failure_sentinel(expected_obj)
        near_empty = _is_near_empty(expected_obj)
        result_substantive = _is_substantive(result_obj)

        if (has_sentinel or near_empty) and result_substantive:
            expected_suspect = True
            reason_str = "sentinel pattern match" if has_sentinel else "near-empty gold structure"
            reasons.append(f"expected_suspect: gold contains {reason_str} while result is substantive")

    # 3. INPUT_STARVED heuristic (UN-GATED check for DR state nodes)
    if isinstance(input_obj, dict):
        state = input_obj.get("state") if isinstance(input_obj.get("state"), dict) else input_obj
        if isinstance(state, dict):
            plan = state.get("plan")
            cursor = state.get("plan_cursor")

            # Only the unambiguous DR no-op guard: cursor at/past the plan end means the node
            # returns its state unchanged (no work), so an empty write-delta is the dataset's
            # doing, not the agent's. (A placeholder snapshot_id is node-specific — it only
            # starves a node that actually retrieves — so it is NOT used as a generic signal:
            # it fired on every CA agent and on DR _node_plan, which don't retrieve.)
            if isinstance(plan, list) and isinstance(cursor, int):
                if cursor >= len(plan):
                    input_starved = True
                    reasons.append(f"input_starved: DR plan_cursor ({cursor}) >= plan length ({len(plan)})")

    # NOTE: no CA "missing bundle/evidence key" starvation heuristic — most CA agents (risk,
    # violations, auditor, project_identity, …) legitimately take no retrieval bundle, so a
    # missing bundle key is not starvation; firing on it blanketed real agent bugs as "dataset".
    # INPUT_STARVED is reserved for the unambiguous DR no-op signals above.

    # 4. SCHEMA_SUSPECT & RESET-VS-POPULATED heuristic
    if isinstance(expected_obj, dict) and isinstance(result_obj, dict):
        common_keys = set(expected_obj.keys()) & set(result_obj.keys())
        for key in common_keys:
            exp_val = expected_obj[key]
            res_val = result_obj[key]

            if exp_val is None and res_val is None:
                continue

            exp_cat = _get_type_category(exp_val)
            res_cat = _get_type_category(res_val)

            # Check 4a: Top-level category mismatch
            if exp_val is not None and res_val is not None and exp_cat != res_cat:
                schema_suspect_fields.append(key)
                reasons.append(f"schema_suspect on '{key}': type mismatch (gold: {exp_cat} vs result: {res_cat})")
                continue

            # Check 4b: Gold populates field with substantive value, but result resets it to None
            # (Matches DR _node_analyze_step gold-guesses-next-state bug where gold copies forward graph_path_with_conf/current_evidence while real node resets to None)
            if _is_substantive(exp_val) and res_val is None:
                schema_suspect_fields.append(key)
                reasons.append(f"schema_suspect on '{key}': gold populates field but result resets to None")
                continue

            # Check 4c: Nested dict/list value type mismatch (e.g. dict[str, str] vs dict[str, float])
            if exp_cat == "dict" and res_cat == "dict":
                nested_common = set(exp_val.keys()) & set(res_val.keys())
                for nkey in nested_common:
                    ne_val = exp_val[nkey]
                    nr_val = res_val[nkey]
                    if ne_val is not None and nr_val is not None:
                        ne_cat = _get_type_category(ne_val)
                        nr_cat = _get_type_category(nr_val)
                        if ne_cat != nr_cat:
                            schema_suspect_fields.append(f"{key}.{nkey}")
                            reasons.append(f"schema_suspect on '{key}.{nkey}': nested type mismatch (gold: {ne_cat} vs result: {nr_cat})")

            elif exp_cat == "list" and res_cat == "list":
                if len(exp_val) > 0 and len(res_val) > 0:
                    ne_cat = _get_type_category(exp_val[0])
                    nr_cat = _get_type_category(res_val[0])
                    if ne_cat != nr_cat:
                        schema_suspect_fields.append(key)
                        reasons.append(f"schema_suspect on '{key}': list element type mismatch (gold: {ne_cat} vs result: {nr_cat})")

    return {
        "soft_fail_fields": soft_fail_fields,
        "expected_suspect": expected_suspect,
        "input_starved": input_starved,
        "schema_suspect_fields": schema_suspect_fields,
        "reasons": reasons,
    }


def calculate_agent_verdict(
    cases_flags: list[tuple[float | None, CaseFlags]],
    acceptable_threshold: float = 0.70,
) -> AgentVerdict:
    """Roll up per-case flags for low-scoring cases (< acceptable_threshold) into an agent-level verdict."""
    low_cases = [
        flags for score, flags in cases_flags
        if score is not None and score < acceptable_threshold
    ]

    low_count = len(low_cases)
    if low_count == 0:
        return {
            "verdict": "healthy",
            "d_count": 0,
            "a_count": 0,
            "low_count": 0,
        }

    d_count = 0
    a_count = 0

    for flags in low_cases:
        has_dataset_flag = (
            flags.get("expected_suspect", False)
            or flags.get("input_starved", False)
            or len(flags.get("schema_suspect_fields", [])) > 0
        )
        if has_dataset_flag:
            d_count += 1
        else:
            a_count += 1

    d_ratio = d_count / low_count
    a_ratio = a_count / low_count

    if d_ratio >= 0.60:
        verdict = "likely-dataset"
    elif a_ratio >= 0.60:
        verdict = "likely-agent"
    else:
        verdict = "mixed"

    return {
        "verdict": verdict,
        "d_count": d_count,
        "a_count": a_count,
        "low_count": low_count,
    }
