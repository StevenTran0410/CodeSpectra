"""Tests for the metric registry, plan-gate readiness, params completion, feasibility, and rebalancing."""
from __future__ import annotations

import textwrap
import pytest

from agent_eval_harness.mapping.agent_flow import AgentFlow, AgentFlowMap
from agent_eval_harness.mapping.system_map import Component, SystemMap, Constraint
from agent_eval_harness.metrics.registry import validate_metric, get_spec
from agent_eval_harness.metrics.suite import DatasetRef
from agent_eval_harness.planning.agentic_planner import reconcile, EvaluationGate
from agent_eval_harness.planning.contract import EvaluationContract, OutputContract
from agent_eval_harness.planning.validation import validate_plan, PlanValidationReport
from agent_eval_harness.store.database import close_db, init_db


@pytest.fixture(autouse=True)
async def _setup_db(tmp_path, monkeypatch):
    monkeypatch.setenv("AEH_DATA_DIR", str(tmp_path))
    await init_db()
    yield
    await close_db()


def test_metric_registry_validation() -> None:
    """Assert registry functions work correctly and check presence/absence."""
    assert validate_metric("allowed_downstream", "assertion")
    assert validate_metric("ragas.faithfulness", "llm_judge")
    assert validate_metric("geval.accuracy", "llm_judge")
    assert validate_metric("classifier.some_label", "classifier")

    # ragas.context_precision is removed (no handler)
    assert not validate_metric("ragas.context_precision")
    assert not validate_metric("ragas.context_precision", "llm_judge")
    assert not validate_metric("unknown_metric")


def test_decomposition_coverage_is_a_literal_registry_entry_with_precondition() -> None:
    """CS-288: a literal key (not the geval.* prefix fallback) carrying input_kind_is_query."""
    assert validate_metric("geval.decomposition_coverage", "llm_judge")
    spec = get_spec("geval.decomposition_coverage")
    assert spec.meaningless_when == ["input_kind_is_query"]
    assert get_spec("ragas.answer_relevancy").meaningless_when == ["input_kind_is_query"]


@pytest.mark.asyncio
async def test_validate_plan_readiness_reasons(tmp_path) -> None:
    """Validate a plan with multiple defects and verify exact readiness blocked reason codes."""
    plan_content = textwrap.dedent("""
        entries:
          - id: gate.no_queries
            component: c1
            metric: ragas.faithfulness
            metric_class: llm_judge
            rationale: "has no queries or dataset ref"
            provenance: llm_suggested
          - id: gate.dataset_unfulfilled
            component: c1
            metric: classifier.c1_sentiment
            metric_class: classifier
            dataset:
              required: {kind: guard_classification, min_cases: 20}
            params:
              entry_point: "main:cls"
            rationale: "required dataset kind but no ref"
            provenance: llm_suggested
          - id: gate.missing_params
            component: c1
            metric: allowed_downstream
            metric_class: assertion
            rationale: "allowed_downstream requires 'allowed' param"
            provenance: llm_suggested
            params:
              queries: ["q"]
          - id: gate.metric_not_dispatchable
            component: c1
            metric: ragas.context_precision
            metric_class: llm_judge
            rationale: "context_precision has no handler"
            provenance: llm_suggested
            params:
              queries: ["q"]
          - id: gate.invalid_dataset_kind
            component: c1
            metric: ragas.faithfulness
            metric_class: llm_judge
            dataset:
              required: {kind: retrieval_grounded_outputs, min_cases: 20}
            rationale: "unknown dataset kind"
            provenance: llm_suggested
            params:
              queries: ["q"]
    """)
    path = tmp_path / "defects_plan.yaml"
    path.write_text(plan_content, encoding="utf-8")

    report = await validate_plan(path)
    assert isinstance(report, PlanValidationReport)
    assert report.readiness["gate.no_queries"].status == "blocked"
    assert "no_queries" in report.readiness["gate.no_queries"].reasons

    assert report.readiness["gate.dataset_unfulfilled"].status == "blocked"
    assert "dataset_unfulfilled" in report.readiness["gate.dataset_unfulfilled"].reasons

    assert report.readiness["gate.missing_params"].status == "blocked"
    assert "missing_params" in report.readiness["gate.missing_params"].reasons

    assert report.readiness["gate.metric_not_dispatchable"].status == "blocked"
    assert "metric_not_dispatchable" in report.readiness["gate.metric_not_dispatchable"].reasons

    assert report.readiness["gate.invalid_dataset_kind"].status == "blocked"
    assert "invalid_dataset_kind" in report.readiness["gate.invalid_dataset_kind"].reasons


def test_reconcile_params_completion_and_feasibility() -> None:
    """Params completion (allowed_downstream, expected tools, arg_schema conversion) and
    feasibility replacement rules (faithfulness needs separable context, relevancy needs
    query input, tool_correctness needs tools)."""
    # 1. Setup system map
    system_map = SystemMap(
        target_system_id="test_sys",
        components=[
            Component(
                id="agent_comp",
                role="agent",
                model="gpt-4",
                entry_point="main:agent",
                upstream=[],
                downstream=["tool_comp"],
            ),
            Component(
                id="tool_comp",
                role="tool",
                model="gpt-4",
                entry_point="main:tool",
                upstream=["agent_comp"],
                downstream=[],
                constraints=[Constraint(name="span_match", value="ToolComp", source="main.py:10")],
            ),
        ]
    )

    agent_flow_map = AgentFlowMap(
        target_system_id="test_sys",
        agents=[
            AgentFlow(id="agent1", role="orchestrator", label="Agent 1", component_ids=["agent_comp"]),
        ]
    )

    # 2. Setup contracts
    contract = EvaluationContract(agent_id="agent1")
    contract.observability.has_separable_context = False  # static fact
    contract.observability.input_kind = "structured"       # static fact
    contract.observability.has_tools = False              # static fact
    contract.output = OutputContract(
        json_schema={"type": "object", "properties": {"result": {"type": "string"}}},
        schema_source="TestTypedDict"
    )

    contracts = {"agent1": contract}

    # Gates to test: allowed_downstream, tool_correctness, arg_schema, ragas.faithfulness, ragas.answer_relevancy.
    gates = [
        EvaluationGate(
            id="g.allowed", agent_id="agent1", component="agent_comp",
            location="handoff", metric="allowed_downstream", metric_class="assertion",
            toolkit="assertion", provenance="llm_suggested"
        ),
        EvaluationGate(
            id="g.tool", agent_id="agent1", component="agent_comp",
            location="output", metric="tool_correctness", metric_class="llm_judge",
            toolkit="deepeval", provenance="llm_suggested"
        ),
        EvaluationGate(
            id="g.schema", agent_id="agent1", component="agent_comp",
            location="output", metric="arg_schema", metric_class="assertion",
            toolkit="assertion", provenance="llm_suggested"
        ),
        EvaluationGate(
            id="g.faith", agent_id="agent1", component="agent_comp",
            location="output", metric="ragas.faithfulness", metric_class="llm_judge",
            toolkit="ragas", provenance="llm_suggested"
        ),
        EvaluationGate(
            id="g.rel", agent_id="agent1", component="agent_comp",
            location="output", metric="ragas.answer_relevancy", metric_class="llm_judge",
            toolkit="ragas", provenance="llm_suggested"
        ),
    ]

    suite, report = reconcile(
        agent_flow_map=agent_flow_map,
        evidence_by_agent={},
        profiles_by_agent={},
        baseline_by_agent={},
        llm_gates_by_agent={"agent1": gates},
        handoff_gates=[],
        contracts=contracts,
        system_map=system_map,
    )

    rep = report.agents[0]
    rep_gates = {g.metric: g for g in rep.gates}

    # verify allowed derived
    assert "allowed_downstream" in rep_gates
    assert rep_gates["allowed_downstream"].params["allowed"] == ["tool_comp"]

    # verify arg_schema replaced by schema_valid with contract schema
    assert "schema_valid" in rep_gates
    assert rep_gates["schema_valid"].params["schema"] == contract.output.json_schema

    # verify ragas.faithfulness replaced by referential_integrity + grounding_judge_span_prompt
    assert "ragas.faithfulness" not in rep_gates
    assert "referential_integrity" in rep_gates
    assert "grounding_judge_span_prompt" in rep_gates

    # verify ragas.answer_relevancy dropped
    assert "ragas.answer_relevancy" not in rep_gates

    # verify tool_correctness replaced by llm_call_budget
    assert "tool_correctness" not in rep_gates
    assert "llm_call_budget" in rep_gates


def test_reconcile_rebalance_rules() -> None:
    """Rebalance rules: prefer assertion over a duplicate llm_judge on the same property,
    merge near-duplicate geval rubrics, and cap LLM-suggested judges at 3 (baseline exempt)."""
    agent_flow_map = AgentFlowMap(
        target_system_id="test_rebalance",
        agents=[
            AgentFlow(id="agent1", role="orchestrator", label="Agent 1", component_ids=["agent_comp"]),
        ]
    )

    # An assertion + duplicate llm_judge on 'output_schema', 2 near-dup geval gates (should
    # merge), plus 3 more suggested geval gates (exceeding the cap of 3 suggested).
    gates = [
        EvaluationGate(
            id="g.assert", agent_id="agent1", component="agent_comp",
            location="output", metric="schema_valid", metric_class="assertion",
            toolkit="assertion", property="output_schema", provenance="llm_suggested",
            params={"schema": {}}
        ),
        EvaluationGate(
            id="g.judge_dup", agent_id="agent1", component="agent_comp",
            location="output", metric="geval.schema_check", metric_class="llm_judge",
            toolkit="deepeval", property="output_schema", provenance="llm_suggested",
            params={"rubric_text": "Check schema"}
        ),
        EvaluationGate(
            id="g.geval_sim1", agent_id="agent1", component="agent_comp",
            location="output", metric="geval.tone", metric_class="llm_judge",
            toolkit="deepeval", property="tone", provenance="llm_suggested",
            params={"rubric_text": "Verify the response is polite and professional."}
        ),
        EvaluationGate(
            id="g.geval_sim2", agent_id="agent1", component="agent_comp",
            location="output", metric="geval.tone_polite", metric_class="llm_judge",
            toolkit="deepeval", property="tone", provenance="llm_suggested",
            params={"rubric_text": "Verify that the response is polite and professional."}
        ),
        EvaluationGate(
            id="g.geval_extra1", agent_id="agent1", component="agent_comp",
            location="output", metric="geval.clarity", metric_class="llm_judge",
            toolkit="deepeval", property="clarity", provenance="llm_suggested",
            params={"rubric_text": "Verify clear explanation."}
        ),
        EvaluationGate(
            id="g.geval_extra2", agent_id="agent1", component="agent_comp",
            location="output", metric="geval.conciseness", metric_class="llm_judge",
            toolkit="deepeval", property="conciseness", provenance="llm_suggested",
            params={"rubric_text": "Verify conciseness."}
        ),
    ]

    suite, report = reconcile(
        agent_flow_map=agent_flow_map,
        evidence_by_agent={},
        profiles_by_agent={},
        baseline_by_agent={},
        llm_gates_by_agent={"agent1": gates},
        handoff_gates=[],
        contracts={},
    )

    rep = report.agents[0]
    final_gate_ids = {g.id for g in rep.gates}

    # 1. g.judge_dup should be dropped because of schema_valid assertion on same property
    assert "g.judge_dup" not in final_gate_ids
    assert "g.assert" in final_gate_ids

    # 2. g.geval_sim2 should be merged into g.geval_sim1 (0.90+ similarity)
    assert "g.geval_sim2" not in final_gate_ids
    assert "g.geval_sim1" in final_gate_ids

    # 3. Cap: at most 3 suggested LLM judges per agent (MAX_LLM_JUDGE_GATES_PER_AGENT).
    suggested_llm_judges = [g for g in rep.gates if g.metric_class == "llm_judge" and g.provenance != "rule"]
    assert len(suggested_llm_judges) <= 3
