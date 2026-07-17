"""Metamorphic transform + invariant operators: field-name-parameterized, unit-tested,
target-agnostic. Every field name comes from the caller (a metamorphic_relations.yaml entry) —
none is a literal here. Transforms wrap perturbation.py's generic degrade mechanism; invariants
are pure functions over an agent's output dict."""
from __future__ import annotations

from typing import Any

from agent_eval_harness.datasets import perturbation

# ---------------------------------------------------------------------------
# Transforms — mutate a case's input, returning a new dict (caller's is untouched).
# ---------------------------------------------------------------------------

def degrade_section(input_data: dict, target_field: str) -> dict:
    return perturbation.degrade_section(input_data, target_field)


def inject_conflict(input_data: dict, target_fields: list[str], token: str) -> dict:
    return perturbation.inject_conflict(input_data, target_fields, token)


TRANSFORM_OPS = {
    "degrade_section": degrade_section,
    "inject_conflict": inject_conflict,
}


def apply_transform(op: str, input_data: dict, params: dict[str, Any]) -> dict:
    fn = TRANSFORM_OPS.get(op)
    if fn is None:
        raise ValueError(f"Unknown metamorphic transform op: {op!r}")
    return fn(input_data, **params)


# ---------------------------------------------------------------------------
# Invariants — pure predicates over an agent's output dict; True = invariant holds.
# ---------------------------------------------------------------------------

def argmin_k(
    output: dict, scores_field: str, report_field: str, k: int, allow_ties: bool = True
) -> bool:
    """True iff report_field's members are exactly the k lowest-scoring keys of scores_field.
    With allow_ties, any valid tie-break at the k-th boundary score is accepted."""
    scores = output.get(scores_field)
    reported = output.get(report_field)
    if not isinstance(scores, dict) or not isinstance(reported, list) or len(scores) < k:
        return False

    ordered = sorted(scores.items(), key=lambda kv: kv[1])
    kth_score = ordered[k - 1][1]
    reported_set = set(reported)

    if not allow_ties:
        return reported_set == {key for key, _ in ordered[:k]}

    strictly_less = {key for key, score in scores.items() if score < kth_score}
    tied_at_kth = {key for key, score in scores.items() if score == kth_score}
    greater = {key for key, score in scores.items() if score > kth_score}

    if not strictly_less.issubset(reported_set):
        return False
    if reported_set & greater:
        return False
    if len(reported_set) != k:
        return False
    return (reported_set - strictly_less).issubset(tied_at_kth)


def subset_eq(output: dict, sub_field: str, super_field: str) -> bool:
    """True iff output[sub_field] (list) is a subset of output[super_field] (list)."""
    sub = output.get(sub_field)
    sup = output.get(super_field)
    if not isinstance(sub, list) or not isinstance(sup, list):
        return False
    return set(sub).issubset(set(sup))


def contains_injected_token(
    output: dict, output_field: str, token: str, min_count: int = 1
) -> bool:
    """True iff `token` appears at least min_count times in output[output_field]
    (list of strings, or a single string)."""
    val = output.get(output_field)
    if val is None:
        return False
    if isinstance(val, list):
        count = sum(1 for item in val if token in str(item))
    else:
        count = str(val).count(token)
    return count >= min_count


def non_empty(output: dict, target_field: str) -> bool:
    """True iff output[target_field] is present and non-empty."""
    val = output.get(target_field)
    if val is None:
        return False
    if isinstance(val, (list, dict, str)):
        return len(val) > 0
    return True


def field_equals(output: dict, target_field: str, expect: Any) -> bool:
    """True iff output[target_field] == expect."""
    return output.get(target_field) == expect


INVARIANT_OPS = {
    "argmin_k": argmin_k,
    "subset_eq": subset_eq,
    "contains_injected_token": contains_injected_token,
    "non_empty": non_empty,
    "field_equals": field_equals,
}


def evaluate_invariant(op: str, output: dict, params: dict[str, Any]) -> bool:
    fn = INVARIANT_OPS.get(op)
    if fn is None:
        raise ValueError(f"Unknown metamorphic invariant op: {op!r}")
    return fn(output, **params)
