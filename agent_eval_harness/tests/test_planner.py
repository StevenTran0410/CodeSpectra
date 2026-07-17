"""Unit and integration tests for the evaluation planner."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_eval_harness.llm.client import LLMResponse
from agent_eval_harness.llm.fake_client import FakeLLMClient
from agent_eval_harness.mapping.builder.roles import VALID_ROLES
from agent_eval_harness.mapping.system_map import Component, Constraint, SystemMap
from agent_eval_harness.planning.planner import (
    _component_role_rules,
    _resolve_dataset_ref,
    baseline_gates_for_component,
    generate_plan,
    get_component_info,
    role_skip_note,
)

_LINEAR_RAG_MAP = Path(__file__).parent.parent / "test_targets" / "linear_rag" / "system_map.yaml"
_MULTI_AGENT_MAP = Path(__file__).parent.parent / "test_targets" / "multi_agent" / "system_map.yaml"
_T3_RERANKER_MAP = Path(__file__).parent.parent / "test_targets" / "t3_reranker" / "system_map.yaml"

pytestmark = pytest.mark.asyncio


async def test_get_component_info() -> None:
    """Verify docstring and source extraction functions work on valid entry points."""
    info = get_component_info("test_targets.linear_rag.components:RetrieverComponent")
    assert "Pure Python keyword-overlap ranker" in info["docstring"]
    assert "class RetrieverComponent" in info["source_snippet"]


async def test_resolve_dataset_ref_always_returns_fresh_required_block() -> None:
    """Plan (re)generation must never link to an already-existing dataset — every dataset
    requirement always starts unfulfilled so Fulfill Datasets always regenerates for real."""
    ref = _resolve_dataset_ref("qa_testset")
    assert ref is not None
    assert ref.ref is None
    assert ref.required == {"kind": "qa_testset", "min_cases": 20}


async def test_resolve_dataset_ref_guard_classification_min_cases() -> None:
    ref = _resolve_dataset_ref("guard_classification")
    assert ref is not None
    assert ref.required["min_cases"] == 40


async def test_resolve_dataset_ref_empty_kind_returns_none() -> None:
    assert _resolve_dataset_ref("") is None


async def test_generate_plan_linear_rag() -> None:
    """Generate plan for T1 and assert structural match with hand-written suite."""
    llm_client = FakeLLMClient(LLMResponse(content="Rubric text", model="fake"))
    plan = await generate_plan(_LINEAR_RAG_MAP, llm_client)

    # Hand-written suite has writer.faithfulness and writer.answer_relevancy
    suite_entries = {e.id: e for e in plan.entries}
    assert "writer.faithfulness" in suite_entries
    assert "writer.answer_relevancy" in suite_entries

    writer_faith = suite_entries["writer.faithfulness"]
    assert writer_faith.metric == "ragas.faithfulness"
    assert writer_faith.metric_class == "llm_judge"
    assert writer_faith.component == "writer"


async def test_generate_plan_multi_agent() -> None:
    """Generate plan for T2 and assert matching entries."""
    llm_client = FakeLLMClient(LLMResponse(content="Rubric text", model="fake"))
    plan = await generate_plan(_MULTI_AGENT_MAP, llm_client)

    suite_entries = {e.id: e for e in plan.entries}
    assert "guard_rule.classifier" in suite_entries
    assert "guard_llm.classifier" in suite_entries
    assert "planner.max_items_per_call" in suite_entries
    assert "planner.allowed_downstream" in suite_entries
    assert "planner.decomposition_coverage" in suite_entries
    assert "worker.max_retries" in suite_entries
    assert "worker.no_unnecessary_calls" in suite_entries
    assert "worker.retry_on_reject_required" in suite_entries
    assert "judge.classifier" in suite_entries
    assert "writer.faithfulness" in suite_entries
    assert "writer.answer_relevancy" in suite_entries


async def test_baseline_gates_for_component_tool_role_still_returns_constraint_entries() -> None:
    """CS-299: the old early return at planner.py:373 discarded AST-mined constraint gates
    for tool/unknown roles even though constraints are unrelated to role — this is a real
    latent bug, inert only because the CodeSpectra fixture map has zero mined constraints."""
    component = Component(
        id="important_files", role="tool", entry_point="m:ImportantFiles",
        constraints=[Constraint(name="max_retries", value=3, source="test.py:1")],
    )
    system_map = SystemMap(target_system_id="t", components=[component])
    llm_client = FakeLLMClient(LLMResponse(content="{}", model="fake"))

    entries = await baseline_gates_for_component(
        component, {component.id: component}, None, system_map, llm_client
    )

    assert len(entries) == 1
    assert entries[0].metric == "max_retries"
    assert entries[0].id == "important_files.max_retries"
    assert entries[0].provenance == "rule"


async def test_role_skip_note_fires_for_tool_and_unknown_only() -> None:
    tool_comp = Component(id="c1", role="tool", entry_point="m:C1")
    unknown_comp = Component(id="c2", role="unknown", entry_point="m:C2")
    worker_comp = Component(id="c3", role="worker", entry_point="m:C3")

    assert role_skip_note(tool_comp) == (
        "c1: role=tool has no role-derived gate (constraints, if any, still apply)"
    )
    assert role_skip_note(unknown_comp) is not None
    assert role_skip_note(worker_comp) is None


@pytest.mark.parametrize("role", sorted(VALID_ROLES))
async def test_component_role_rules_unchanged_for_every_valid_role(role: str) -> None:
    """CS-300 AC5: role now comes from Stage 2.5 instead of Stage 2, but _component_role_rules
    itself is untouched by this ticket — this test proves it by exercising every role in the
    vocabulary and pinning the exact rule set each one fires, on a minimal single-component map
    (no downstream tools, no validator retry loop, so retrieval_agent fires nothing here)."""
    component = Component(id="c1", role=role, entry_point="m:C1")
    components_by_id = {"c1": component}
    system_map = SystemMap(target_system_id="t", components=[component])

    rules = _component_role_rules(component, components_by_id, None, system_map)
    metrics = {r["metric"] for r in rules}

    if role in ("input_guard.rule", "input_guard.llm"):
        assert metrics == {"classifier.c1_accuracy"}
        assert rules[0]["dataset_kind"] == "guard_classification"
    elif role == "validator":
        assert metrics == {"classifier.c1_accuracy"}
        assert rules[0]["dataset_kind"] == "sufficiency_labeled"
    elif role == "orchestrator":
        assert metrics == {"geval.decomposition_coverage", "allowed_downstream"}
    elif role == "retrieval_agent":
        assert metrics == set()
    elif role == "writer":
        assert metrics == {"ragas.faithfulness", "ragas.answer_relevancy"}
    else:
        assert role in ("tool", "worker", "unknown"), f"unhandled role in VALID_ROLES: {role}"
        assert metrics == set()


async def test_geval_rubric_tailoring() -> None:
    """Verify that rubric text is fetched from LLM response or falls back correctly."""
    # Test case 1: LLM returns custom JSON
    custom_rubric_json = json.dumps({"rubric_text": "Custom Tailored Rubric Wording."})
    llm_client = FakeLLMClient(LLMResponse(content=custom_rubric_json, model="fake"))

    plan = await generate_plan(_MULTI_AGENT_MAP, llm_client)
    decomp_entry = [
        e for e in plan.entries if e.metric == "geval.decomposition_coverage"
    ][0]
    assert decomp_entry.params["rubric_text"] == "Custom Tailored Rubric Wording."

    # Test case 2: LLM returns fallback
    fallback_content = "This is a fallback offline demo answer."
    llm_client_fallback = FakeLLMClient(LLMResponse(content=fallback_content, model="fake"))
    plan_fallback = await generate_plan(_MULTI_AGENT_MAP, llm_client_fallback)
    decomp_entry_fallback = [
        e for e in plan_fallback.entries if e.metric == "geval.decomposition_coverage"
    ][0]
    assert (
        "Evaluate whether the planner's decomposed intents"
        in decomp_entry_fallback.params["rubric_text"]
    )


async def test_generate_plan_t3_generalization() -> None:
    """Verify plan generation on unseen target T3 (retriever -> reranker -> writer)."""
    llm_client = FakeLLMClient(LLMResponse(content="Rubric text", model="fake"))
    plan = await generate_plan(_T3_RERANKER_MAP, llm_client)

    suite_entries = {e.id: e for e in plan.entries}
    # retriever and reranker are retrieval_agents
    assert "writer.faithfulness" in suite_entries
    assert "writer.answer_relevancy" in suite_entries

    # retriever and reranker have no downstream tools in the T3 map, so only writer-level gates are expected.
    assert len(plan.entries) >= 2
