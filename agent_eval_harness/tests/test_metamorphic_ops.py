import pytest

from agent_eval_harness.datasets.metamorphic_ops import (
    apply_transform,
    argmin_k,
    contains_injected_token,
    evaluate_invariant,
    field_equals,
    non_empty,
    subset_eq,
)


# argmin_k — the mandatory behavioral invariant test: it must actually FIRE.

def test_argmin_k_flags_omitted_lower_score():
    """Auditor case: scores D=49,E=58,J=61,G=63; weakest_sections=[D,E,G] wrongly omits J
    (61) whose score is lower than G's (63). The invariant must return False (flagged)."""
    output = {
        "section_scores": {"D": 49, "E": 58, "J": 61, "G": 63},
        "weakest_sections": ["D", "E", "G"],
    }
    holds = argmin_k(
        output, scores_field="section_scores", report_field="weakest_sections", k=3, allow_ties=True
    )
    assert holds is False


def test_argmin_k_allows_valid_tie_at_boundary():
    """95/95 tie for the 3rd (last) weakest slot — a legitimate tie-break must NOT be flagged."""
    output = {
        "section_scores": {"A": 80, "B": 90, "C": 95, "D": 95, "E": 99},
        "weakest_sections": ["A", "B", "C"],  # picks C over D at the tied boundary score
    }
    holds = argmin_k(
        output, scores_field="section_scores", report_field="weakest_sections", k=3, allow_ties=True
    )
    assert holds is True

    # The other valid tie-break choice (D instead of C) must also not be flagged.
    output2 = {**output, "weakest_sections": ["A", "B", "D"]}
    assert argmin_k(
        output2, scores_field="section_scores", report_field="weakest_sections", k=3, allow_ties=True
    ) is True


def test_argmin_k_rejects_boundary_omission_even_with_ties_allowed():
    """A tie is not a license to omit a strictly-lower key outright."""
    output = {
        "section_scores": {"A": 80, "B": 90, "C": 95, "D": 95, "E": 99},
        "weakest_sections": ["C", "D", "E"],  # drops A/B, includes the non-tied E — invalid
    }
    holds = argmin_k(
        output, scores_field="section_scores", report_field="weakest_sections", k=3, allow_ties=True
    )
    assert holds is False


def test_argmin_k_no_ties_requires_exact_match():
    output = {"section_scores": {"A": 1, "B": 2, "C": 3}, "weakest_sections": ["A", "B"]}
    assert argmin_k(
        output, scores_field="section_scores", report_field="weakest_sections", k=2, allow_ties=False
    ) is True
    output_wrong = {**output, "weakest_sections": ["A", "C"]}
    assert argmin_k(
        output_wrong, scores_field="section_scores", report_field="weakest_sections", k=2, allow_ties=False
    ) is False


def test_argmin_k_missing_fields_returns_false():
    assert argmin_k({}, scores_field="scores", report_field="weakest", k=2) is False
    assert argmin_k(
        {"scores": {"a": 1}}, scores_field="scores", report_field="weakest", k=2
    ) is False  # fewer keys than k


# subset_eq

def test_subset_eq_true_and_false():
    assert subset_eq({"a": [1, 2], "b": [1, 2, 3]}, sub_field="a", super_field="b") is True
    assert subset_eq({"a": [1, 4], "b": [1, 2, 3]}, sub_field="a", super_field="b") is False
    assert subset_eq({"a": "not-a-list", "b": [1]}, sub_field="a", super_field="b") is False


# contains_injected_token

def test_contains_injected_token_in_list_field():
    output = {"violations_found": ["contradiction: __mr_conflict__ present", "other"]}
    assert contains_injected_token(
        output, output_field="violations_found", token="__mr_conflict__", min_count=1
    ) is True


def test_contains_injected_token_absent():
    output = {"violations_found": []}
    assert contains_injected_token(
        output, output_field="violations_found", token="__mr_conflict__", min_count=1
    ) is False


def test_contains_injected_token_string_field_min_count():
    output = {"summary": "__mr_conflict__ appears twice: __mr_conflict__"}
    assert contains_injected_token(output, output_field="summary", token="__mr_conflict__", min_count=2) is True
    assert contains_injected_token(output, output_field="summary", token="__mr_conflict__", min_count=3) is False


# non_empty

def test_non_empty_variants():
    assert non_empty({"context": ["x"]}, target_field="context") is True
    assert non_empty({"context": []}, target_field="context") is False
    assert non_empty({"context": None}, target_field="context") is False
    assert non_empty({}, target_field="missing") is False
    assert non_empty({"count": 0}, target_field="count") is True  # scalar zero is not "empty"


# field_equals

def test_field_equals():
    assert field_equals({"sufficient": False}, target_field="sufficient", expect=False) is True
    assert field_equals({"sufficient": True}, target_field="sufficient", expect=False) is False
    assert field_equals({}, target_field="sufficient", expect=False) is False


# apply_transform / evaluate_invariant dispatch (non-CodeSpectra fixture, generic field names)

def test_apply_transform_degrade_section_on_flat_list_field():
    """multi_agent-shaped input: a list-typed top-level field degrades to []."""
    input_data = {"query": "What is the vacation policy?", "context": ["doc a", "doc b"]}
    result = apply_transform("degrade_section", input_data, {"target_field": "context"})
    assert result["context"] == []
    assert input_data["context"] == ["doc a", "doc b"]  # caller dict untouched


def test_apply_transform_inject_conflict():
    input_data = {"purpose": "original", "description": "also original"}
    result = apply_transform(
        "inject_conflict", input_data, {"target_fields": ["purpose", "description"], "token": "__mr_conflict__"}
    )
    assert "__mr_conflict__" in result["purpose"]
    assert "__mr_conflict__" in result["description"]


def test_apply_transform_unknown_op_raises():
    with pytest.raises(ValueError):
        apply_transform("not_a_real_op", {}, {})


def test_evaluate_invariant_dispatch():
    assert evaluate_invariant("field_equals", {"sufficient": False}, {"target_field": "sufficient", "expect": False}) is True


def test_evaluate_invariant_unknown_op_raises():
    with pytest.raises(ValueError):
        evaluate_invariant("not_a_real_op", {}, {})
