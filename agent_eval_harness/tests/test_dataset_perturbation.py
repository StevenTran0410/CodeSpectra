import json
import pathlib

from agent_eval_harness.datasets.perturbation import (
    degrade_section,
    drop_section,
    inject_conflict,
    oversize_section,
)


def test_drop_section():
    all_sections = {"foo": {"repo_name": "test", "content": "hello"}, "bar": {"content": "world"}}
    res = drop_section(all_sections, "foo")
    assert "foo" not in res
    assert "bar" in res
    assert "foo" in all_sections  # Verify caller dict is not mutated
    assert json.loads(json.dumps(res)) == res  # Verify serialization


def test_degrade_section_dict_field_generic():
    """Generic dict-shaped field: every leaf degrades to its own typed empty — no field-name
    special-casing (repo_name/confidence/blind_spots/domain/runtime_type do not exist here)."""
    all_sections = {
        "analysis": {
            "tags": ["a", "b"],
            "nested": {"x": 1},
            "note": "hello",
            "score": 42,
        }
    }
    res = degrade_section(all_sections, "analysis")
    assert res["analysis"]["tags"] == []
    assert res["analysis"]["nested"] == {}
    assert res["analysis"]["note"] == ""
    assert res["analysis"]["score"] is None

    assert all_sections["analysis"]["note"] == "hello"  # caller dict not mutated
    assert json.loads(json.dumps(res)) == res  # serializable


def test_degrade_section_missing_field_is_noop():
    all_sections = {"analysis": {"note": "hello"}}
    res = degrade_section(all_sections, "not_present")
    assert res == all_sections


def test_degrade_section_list_typed_field_yields_empty_list_not_dict():
    """multi_agent's WorkerComponent/JudgeComponent output {"context": list[str]} — a
    list-typed top-level field must degrade to [], never a {} stub (the :23-24 shape bug)."""
    case_like = {"query": "What is the vacation policy?", "context": ["doc one", "doc two"]}
    res = degrade_section(case_like, "context")
    assert res["context"] == []
    assert isinstance(res["context"], list)
    assert case_like["context"] == ["doc one", "doc two"]  # caller dict not mutated


def test_degrade_section_str_typed_field_yields_empty_string():
    case_like = {"query": "hello", "answer": "some text"}
    res = degrade_section(case_like, "answer")
    assert res["answer"] == ""


def test_degrade_section_scalar_typed_field_yields_none():
    case_like = {"query": "hello", "count": 3}
    res = degrade_section(case_like, "count")
    assert res["count"] is None


def test_inject_conflict_flat_string_fields():
    """Token injected directly into flat top-level string fields (no nested dict required)."""
    input_data = {"purpose": "Original", "description": "Also original"}
    res = inject_conflict(input_data, ["purpose", "description"], "__mr_conflict__")
    assert res["purpose"] == "Original __mr_conflict__"
    assert res["description"] == "Also original __mr_conflict__"


def test_inject_conflict_missing_field_gets_token_set_directly():
    input_data = {"purpose": "Original"}
    res = inject_conflict(input_data, ["purpose", "new_field"], "__mr_conflict__")
    assert res["new_field"] == "__mr_conflict__"


def test_oversize_section():
    all_sections = {
        "a": {"repo_name": "test", "purpose": "Short purpose"},
    }
    res = oversize_section(all_sections, "a", target_len=1000)
    assert len(res["a"]["purpose"]) == 1000
    assert "Short purpose" in res["a"]["purpose"]

    assert len(all_sections["a"]["purpose"]) < 100  # Verify caller dict is not mutated
    assert json.loads(json.dumps(res)) == res  # Verify serialization


def test_grep_gate_no_codespectra_literals_in_perturbation() -> None:
    """Nguyen tac so 0 grep gate: zero CodeSpectra report literals remain in perturbation.py —
    the generic isinstance fallback is the whole degrade mechanism."""
    forbidden = [
        "blind_spots",
        "domain",
        "runtime_type",
        "section_scores",
        "weakest_sections",
        "empty_confidence",
    ]
    source_path = (
        pathlib.Path(__file__).parent.parent
        / "agent_eval_harness"
        / "datasets"
        / "perturbation.py"
    )
    content = source_path.read_text(encoding="utf-8")

    found = [literal for literal in forbidden if literal in content]
    assert not found, f"Forbidden CodeSpectra-report literal(s) found in perturbation.py: {found}"

    # No dedicated "confidence" special-case branch — the residual "confidence"/"repo_name"
    # strings that DO remain (inject_conflict's old A/B skip-list is gone; oversize_section's
    # generic skip-list is allowed) must not appear as an `if k == "confidence"` branch.
    assert 'k == "confidence"' not in content
    assert "elif k ==" not in content

    # No single-letter section keys (A-L) used as dict literals/params anywhere.
    import re

    single_letter_keys = re.findall(r'["\']([A-L])["\']\s*[:,)]', content)
    assert not single_letter_keys, f"Single-letter section keys found: {single_letter_keys}"
