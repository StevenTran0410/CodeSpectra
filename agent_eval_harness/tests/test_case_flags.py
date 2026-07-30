from __future__ import annotations

import pytest
from agent_eval_harness.judging.case_flags import (
    calculate_agent_verdict,
    derive_case_flags,
)


def test_soft_fail_heuristic():
    # 1a. Subjective field with matching string types -> SOFT_FAIL
    evals_subjective = [
        {
            "metric_name": "field_exact.confidence",
            "score": 0.0,
            "details": {"actual": "medium", "expected": "high"},
        }
    ]
    flags = derive_case_flags({}, {}, {}, evals_subjective)
    assert "confidence" in flags["soft_fail_fields"]
    assert any("soft_fail on 'confidence'" in r for r in flags["reasons"])

    # 1b. Subjective field with mismatched types -> NOT soft_fail
    evals_subjective_mismatch = [
        {
            "metric_name": "field_exact.confidence",
            "score": 0.0,
            "details": {"actual": 0.8, "expected": "high"},
        }
    ]
    flags_mismatch = derive_case_flags({}, {}, {}, evals_subjective_mismatch)
    assert "confidence" not in flags_mismatch["soft_fail_fields"]

    # 1c. Estimate field within 20% tolerance -> SOFT_FAIL
    evals_estimate = [
        {
            "metric_name": "field_exact.coverage_percentage",
            "score": 0.0,
            "details": {"actual": 85.0, "expected": 100.0},  # 15% <= 20%
        }
    ]
    flags_est = derive_case_flags({}, {}, {}, evals_estimate)
    assert "coverage_percentage" in flags_est["soft_fail_fields"]

    # 1d. Estimate field beyond 20% tolerance -> NOT soft_fail
    evals_estimate_far = [
        {
            "metric_name": "field_exact.coverage_percentage",
            "score": 0.0,
            "details": {"actual": 50.0, "expected": 100.0},  # 50% > 20%
        }
    ]
    flags_far = derive_case_flags({}, {}, {}, evals_estimate_far)
    assert "coverage_percentage" not in flags_far["soft_fail_fields"]


def test_expected_suspect_heuristic():
    substantive_result = {
        "summary": "This is a detailed and substantive analysis of the architectural pipeline containing more than twenty characters of technical explanation.",
        "components": ["analyzer_1", "analyzer_2"],
    }

    # 2a. Gold contains narrowed system failure sentinel (e.g. "retrieval error") -> EXPECTED_SUSPECT
    gold_sentinel = {
        "blind_spots": ["Agent failed: retrieval error during index query"],
    }
    flags = derive_case_flags({}, substantive_result, gold_sentinel, [])
    assert flags["expected_suspect"] is True
    assert any("expected_suspect" in r for r in flags["reasons"])

    # 2b. Gold is near-empty while result is substantive -> EXPECTED_SUSPECT
    gold_empty = {"items": [], "notes": ""}
    flags_empty = derive_case_flags({}, substantive_result, gold_empty, [])
    assert flags_empty["expected_suspect"] is True

    # 2c. Legit non-empty gold -> NOT expected_suspect
    gold_legit = {
        "summary": "Legitimate gold summary that also has substantive detailed content for evaluation.",
        "components": ["c1"],
    }
    flags_legit = derive_case_flags({}, substantive_result, gold_legit, [])
    assert flags_legit["expected_suspect"] is False

    # 2d. Legit empty-term glossary gold ("No explicit glossary terms found") -> NOT expected_suspect (Fix 3)
    gold_glossary_empty = {
        "terms": [],
        "notes": "No explicit glossary terms found in evidence files provided",
    }
    flags_glossary = derive_case_flags({"bundle": {"files": ["a.py"]}}, {"terms": ["API", "SDK"]}, gold_glossary_empty, [])
    assert flags_glossary["expected_suspect"] is False


def test_input_starved_heuristic():
    # 3a. DR state-node input starved: plan_cursor >= len(plan) (UN-GATED check, Fix 1)
    input_dr_starved = {
        "state": {
            "plan": ["step_1", "step_2"],
            "plan_cursor": 2,
        }
    }
    result_state_echo = {
        "question": "What is the architecture?",
        "snapshot_id": "snap-001",
        "plan": ["step_1", "step_2"],
        "plan_cursor": 2,
    }
    flags_dr = derive_case_flags(input_dr_starved, result_state_echo, {"gold": "data"}, [])
    assert flags_dr["input_starved"] is True
    assert any("plan_cursor" in r for r in flags_dr["reasons"])

    # 3b. Placeholder snapshot_id ALONE (no cursor no-op) must NOT flag input_starved — a fake
    # snapshot only starves a node that actually retrieves; firing on it mislabelled CA agents
    # and DR _node_plan (which don't retrieve) as dataset issues.
    input_snap = {
        "state": {
            "snapshot_id": "snap_12345",
        }
    }
    flags_snap = derive_case_flags(input_snap, result_state_echo, {"gold": "data"}, [])
    assert flags_snap["input_starved"] is False

    # 3c. Valid UUID snapshot_id -> NOT input_starved
    input_valid_uuid = {
        "state": {
            "snapshot_id": "12345678-1234-5678-1234-567812345678",
            "plan": ["step_1", "step_2"],
            "plan_cursor": 0,
        }
    }
    flags_uuid = derive_case_flags(input_valid_uuid, result_state_echo, {"gold": "data"}, [])
    assert flags_uuid["input_starved"] is False

    # 3d. A missing bundle/evidence key must NOT flag input_starved — most CA agents legitimately
    # take no retrieval bundle, so this is not starvation (firing on it hid real agent bugs as
    # "dataset"). INPUT_STARVED is reserved for the DR cursor no-op signal.
    input_ca_no_bundle = {"project_id": "p1"}
    substantive_gold = {"evidence": "Substantive evidence text containing more than twenty characters for testing."}
    flags_ca = derive_case_flags(input_ca_no_bundle, {}, substantive_gold, [])
    assert flags_ca["input_starved"] is False


def test_schema_suspect_heuristic():
    # 4a. Container type mismatch (dict vs list) -> SCHEMA_SUSPECT
    gold = {"graph_path_with_conf": {"node": "a", "score": 0.9}}
    result = {"graph_path_with_conf": ["node_a", "node_b"]}
    flags = derive_case_flags({}, result, gold, [])
    assert "graph_path_with_conf" in flags["schema_suspect_fields"]

    # 4b. Gold populates field with substantive value, but result resets it to None -> SCHEMA_SUSPECT (Fix 2)
    gold_stale = {
        "graph_path_with_conf": {"nodes": ["a", "b"]},
        "current_evidence": ["ev1", "ev2"],
    }
    result_reset = {
        "graph_path_with_conf": None,
        "current_evidence": ["ev1"],
    }
    flags_reset = derive_case_flags({}, result_reset, gold_stale, [])
    assert "graph_path_with_conf" in flags_reset["schema_suspect_fields"]

    # 4c. Matching container types -> NOT schema_suspect
    gold_match = {"graph_path_with_conf": ["node_a"]}
    result_match = {"graph_path_with_conf": ["node_b"]}
    flags_match = derive_case_flags({}, result_match, gold_match, [])
    assert "graph_path_with_conf" not in flags_match["schema_suspect_fields"]


def test_real_agent_verdicts_alignment():
    """Verify per-agent verdicts align with real findings from eval_dataset_findings.md:

    - DR _node_retrieve, _node_analyze_step, CA structure -> likely-dataset
    - CA auditor, violations, project_identity -> likely-agent
    - CA glossary -> NOT likely-dataset
    """
    # 1. DR _node_retrieve (plan_cursor >= len(plan) or placeholder snap_id)
    dr_retrieve_cases = [
        (0.20, derive_case_flags(
            {"state": {"plan": ["step1"], "plan_cursor": 1, "snapshot_id": "snap_001"}},
            {"retrieved_chunks": []},
            {"retrieved_chunks": ["chunk_a"]},
            [{"metric_name": "semantic_match", "score": 0.20}]
        )),
        (0.30, derive_case_flags(
            {"state": {"plan": ["step1"], "plan_cursor": 2, "snapshot_id": "snap_002"}},
            {"retrieved_chunks": []},
            {"retrieved_chunks": ["chunk_b"]},
            [{"metric_name": "semantic_match", "score": 0.30}]
        )),
    ]
    v_dr_retrieve = calculate_agent_verdict([(s, f) for s, f in dr_retrieve_cases], 0.70)
    assert v_dr_retrieve["verdict"] == "likely-dataset"

    # 2. DR _node_analyze_step (gold populates graph_path_with_conf that result resets to None)
    dr_analyze_cases = [
        (0.35, derive_case_flags(
            {"state": {"plan": ["step1", "step2"], "plan_cursor": 1, "snapshot_id": "12345678-1234-5678-1234-567812345678"}},
            {"graph_path_with_conf": None},
            {"graph_path_with_conf": {"nodes": ["a"]}},
            [{"metric_name": "semantic_match", "score": 0.35}]
        )),
        (0.40, derive_case_flags(
            {"state": {"plan": ["step1", "step2"], "plan_cursor": 1, "snapshot_id": "12345678-1234-5678-1234-567812345678"}},
            {"graph_path_with_conf": None},
            {"graph_path_with_conf": {"nodes": ["b"]}},
            [{"metric_name": "semantic_match", "score": 0.40}]
        )),
    ]
    v_dr_analyze = calculate_agent_verdict([(s, f) for s, f in dr_analyze_cases], 0.70)
    assert v_dr_analyze["verdict"] == "likely-dataset"

    # 3. CA structure (real signal: gold is a retrieval-failure artifact while the agent succeeded)
    ca_structure_cases = [
        (0.0, derive_case_flags(
            {"arch_bundle": {"evidences": []}, "folder_tree": {"src": {}}},
            {"folders": [{"path": "src", "role": "domain"}], "summary": "A substantive structural analysis derived from the folder tree.", "confidence": "high"},
            {"folders": [], "summary": "", "blind_spots": ["Agent failed: Retrieval error - no architecture evidences found"]},
            [{"metric_name": "semantic_match", "score": 0.0}]
        )),
    ]
    v_ca_structure = calculate_agent_verdict([(s, f) for s, f in ca_structure_cases], 0.70)
    assert v_ca_structure["verdict"] == "likely-dataset"

    # 4. CA auditor (genuine agent issue: section_scores always empty in agent output)
    ca_auditor_cases = [
        (0.45, derive_case_flags(
            {"bundle": {"files": ["main.py"]}},
            {"section_scores": {}, "audit_notes": "flat overview"},
            {"section_scores": {"security": 0.8}, "audit_notes": "detailed section scores"},
            [{"metric_name": "semantic_match", "score": 0.45}]
        )),
    ]
    v_ca_auditor = calculate_agent_verdict([(s, f) for s, f in ca_auditor_cases], 0.70)
    assert v_ca_auditor["verdict"] == "likely-agent"

    # 5. CA violations (genuine agent issue: empty violations_found in agent output)
    ca_violations_cases = [
        (0.30, derive_case_flags(
            {"bundle": {"files": ["auth.py"]}, "static_risk": {"findings": [{"category": "injection"}]}},
            {"violations_found": [], "blind_spots": ["no evidence files provided"]},
            {"violations_found": ["SQL injection vulnerability in auth.py"]},
            [{"metric_name": "semantic_match", "score": 0.30}]
        )),
    ]
    v_ca_violations = calculate_agent_verdict([(s, f) for s, f in ca_violations_cases], 0.70)
    assert v_ca_violations["verdict"] == "likely-agent"

    # 6. CA project_identity (genuine agent issue: architecture label returned instead of runtime)
    ca_identity_cases = [
        (0.30, derive_case_flags(
            {"bundle": {"files": ["main.py"]}},
            {"runtime_type": "backend_service"},
            {"runtime_type": "python"},
            [{"metric_name": "field_exact.runtime_type", "score": 0.0, "details": {"actual": "backend_service", "expected": "python"}}]
        )),
    ]
    v_ca_identity = calculate_agent_verdict([(s, f) for s, f in ca_identity_cases], 0.70)
    assert v_ca_identity["verdict"] == "likely-agent"

    # 7. CA glossary (legit empty gold + agent output -> NOT likely-dataset)
    ca_glossary_cases = [
        (0.50, derive_case_flags(
            {"bundle": {"files": ["a.py"]}},
            {"terms": ["API", "SDK"]},
            {"terms": [], "notes": "No explicit glossary terms found in evidence files"},
            [{"metric_name": "semantic_match", "score": 0.50}]
        )),
    ]
    v_ca_glossary = calculate_agent_verdict([(s, f) for s, f in ca_glossary_cases], 0.70)
    assert v_ca_glossary["verdict"] != "likely-dataset"
