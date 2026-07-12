import json

from agent_eval_harness.datasets.perturbation import (
    degrade_section,
    drop_section,
    inject_conflict,
    oversize_section,
)


def test_drop_section():
    all_sections = {"A": {"repo_name": "test", "content": "hello"}, "B": {"content": "world"}}
    res = drop_section(all_sections, "A")
    assert "A" not in res
    assert "B" in res
    assert "A" in all_sections  # Verify caller dict is not mutated
    assert json.loads(json.dumps(res)) == res  # Verify serialization


def test_degrade_section():
    all_sections = {
        "A": {
            "repo_name": "test",
            "confidence": "high",
            "domain": "web",
            "tech_stack": ["python"],
            "purpose": "do something",
            "blind_spots": ["original spot"],
        }
    }
    res = degrade_section(all_sections, "A")
    assert res["A"]["repo_name"] == "test"
    assert res["A"]["confidence"] == "low"
    assert res["A"]["domain"] == "unknown"
    assert res["A"]["tech_stack"] == []
    assert res["A"]["purpose"] == ""
    assert res["A"]["blind_spots"] == ["Agent failed: Degraded"]

    assert all_sections["A"]["confidence"] == "high"  # Verify caller dict is not mutated
    assert json.loads(json.dumps(res)) == res  # Verify serialization


def test_degrade_section_preserves_confidence_when_flag_disabled():
    all_sections = {"A": {"confidence": "high", "purpose": "do something"}}
    res = degrade_section(all_sections, "A", empty_confidence=False)
    assert res["A"]["confidence"] == "high"
    assert res["A"]["purpose"] == ""  # other fields still degrade normally


def test_inject_conflict():
    all_sections = {
        "A": {"repo_name": "test", "purpose": "Original purpose A"},
        "B": {"repo_name": "test", "description": "Original description B"},
    }
    res = inject_conflict(all_sections, "A", "B", "Claim A", "Claim B")
    assert "Claim A" in res["A"]["purpose"]
    assert "Claim B" in res["B"]["description"]

    assert "Claim A" not in all_sections["A"]["purpose"]  # Verify caller dict is not mutated
    assert json.loads(json.dumps(res)) == res  # Verify serialization


def test_oversize_section():
    all_sections = {
        "A": {"repo_name": "test", "purpose": "Short purpose"},
    }
    res = oversize_section(all_sections, "A", target_len=1000)
    assert len(res["A"]["purpose"]) == 1000
    assert "Short purpose" in res["A"]["purpose"]

    assert len(all_sections["A"]["purpose"]) < 100  # Verify caller dict is not mutated
    assert json.loads(json.dumps(res)) == res  # Verify serialization
