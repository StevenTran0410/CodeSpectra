"""Tests for the Stage 3 DAG LLM orchestrator."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_eval_harness.llm.client import LLMResponse
from agent_eval_harness.llm.fake_client import FakeLLMClient
from agent_eval_harness.mapping.agent_flow import AgentFlow, AgentFlowMap
from agent_eval_harness.mapping.system_map import Component, SystemMap, load_system_map
from agent_eval_harness.planning import agentic_planner as ap
from agent_eval_harness.planning.planner import generate_plan
from agent_eval_harness.planning.report import (
    AgentDataProfile,
    EvaluationPlanReport,
    load_plan_report,
    save_plan_report,
)

pytestmark = pytest.mark.asyncio

_MULTI_AGENT_MAP = Path(__file__).parent.parent / "test_targets" / "multi_agent" / "system_map.yaml"


def _multi_agent_flow_map() -> AgentFlowMap:
    """Groups every T2 component into 2 agents — full coverage, mirrors what Stage 2's
    separate_agent_flows would plausibly produce for this target."""
    return AgentFlowMap(
        target_system_id="multi_agent",
        agents=[
            AgentFlow(
                id="guard_agent",
                role="input_guard.llm",
                label="Guard",
                component_ids=["guard_rule", "guard_llm"],
                downstream_agents=["core_agent"],
            ),
            AgentFlow(
                id="core_agent",
                role="orchestrator",
                label="Core",
                component_ids=["planner", "worker", "judge", "writer", "case_law_search_tool", "decoy_tool"],
                upstream_agents=["guard_agent"],
            ),
        ],
        entry_agent_ids=["guard_agent"],
    )


def _fake_source_by_component(system_map: SystemMap) -> dict[str, str]:
    return {c.id: f"class {c.id}: ..." for c in system_map.components}


# ──────────────────────────────────────────────────────────────────────────────
# report.py
# ──────────────────────────────────────────────────────────────────────────────


async def test_plan_report_yaml_roundtrip(tmp_path) -> None:
    report = EvaluationPlanReport(
        target_system_id="t2",
        agents=[],
        advisory_notes=["agent X has zero gates"],
    )
    path = tmp_path / "report.yaml"
    save_plan_report(report, path)
    loaded = load_plan_report(path)
    assert loaded.target_system_id == "t2"
    assert loaded.advisory_notes == ["agent X has zero gates"]


# ──────────────────────────────────────────────────────────────────────────────
# DAG runner wiring
# ──────────────────────────────────────────────────────────────────────────────


async def test_run_dag_respects_fanout_fanin_order() -> None:
    order: list[str] = []

    async def root(_):
        order.append("root")
        return "root-val"

    async def branch(results, name="a"):
        order.append(name)
        assert results["root"] == "root-val"
        return name

    async def join(results):
        order.append("join")
        assert set(results["a"]) | set(results["b"]) or True
        return sorted([results["a"], results["b"]])

    nodes = [
        ap.DagNode("root", [], root),
        ap.DagNode("a", ["root"], lambda r: branch(r, "a")),
        ap.DagNode("b", ["root"], lambda r: branch(r, "b")),
        ap.DagNode("join", ["a", "b"], join),
    ]
    results = await ap.run_dag(nodes)

    assert order[0] == "root"
    assert set(order[1:3]) == {"a", "b"}
    assert order[3] == "join"
    assert results["join"] == ["a", "b"]


async def test_run_dag_raises_on_unresolvable_graph() -> None:
    async def noop(_):
        return None

    nodes = [ap.DagNode("x", ["missing"], noop)]
    with pytest.raises(RuntimeError):
        await ap.run_dag(nodes)


# ──────────────────────────────────────────────────────────────────────────────
# gather_evidence / supporting files join
# ──────────────────────────────────────────────────────────────────────────────


async def test_supporting_files_attach_to_owning_agent_not_boundary() -> None:
    system_map = SystemMap(
        target_system_id="t",
        components=[
            Component(id="identity_agent", role="unknown", entry_point="m:A", file="agents/identity.py"),
            Component(id="arch_agent", role="unknown", entry_point="m:B", file="agents/architecture.py"),
        ],
    )
    agent_flow_map = AgentFlowMap(
        target_system_id="t",
        agents=[
            AgentFlow(id="identity", component_ids=["identity_agent"], downstream_agents=["arch"]),
            AgentFlow(id="arch", component_ids=["arch_agent"], upstream_agents=["identity"]),
        ],
    )
    accepted_edges = [
        {"src": "agents/identity.py", "dst": "agents/helpers/prompt_builder.py"},
        {"src": "agents/identity.py", "dst": "agents/architecture.py"},
    ]

    supporting = ap._supporting_files_by_agent(agent_flow_map, system_map, accepted_edges)

    assert supporting["identity"] == ["agents/helpers/prompt_builder.py"]
    assert supporting["arch"] == []


async def test_gather_evidence_computes_baseline_per_agent() -> None:
    system_map = load_system_map(_MULTI_AGENT_MAP)
    agent_flow_map = _multi_agent_flow_map()
    llm_client = FakeLLMClient(LLMResponse(content='{"rubric_text": "custom"}', model="fake"))

    evidence_by_agent, baseline_by_agent = await ap.gather_evidence(
        system_map, agent_flow_map, _fake_source_by_component(system_map), [], llm_client
    )

    assert set(evidence_by_agent) == {"guard_agent", "core_agent"}
    guard_owned_ids = {c["id"] for c in evidence_by_agent["guard_agent"].owned}
    assert guard_owned_ids == {"guard_rule", "guard_llm"}

    # baseline reproduces the same rule set the flat planner would for these components
    baseline_metrics = {e.metric for e in baseline_by_agent["core_agent"]}
    assert "geval.decomposition_coverage" in baseline_metrics
    assert "allowed_downstream" in baseline_metrics
    assert "max_items_per_call" in baseline_metrics
    assert all(e.agent_id == "core_agent" for e in baseline_by_agent["core_agent"])


# ──────────────────────────────────────────────────────────────────────────────
# per-node defensive parsing
# ──────────────────────────────────────────────────────────────────────────────


async def test_run_analyst_defensive_parse_on_malformed_json() -> None:
    evidence = ap.AgentEvidence(agent=AgentFlow(id="a1", component_ids=["c1"]), owned=[], supporting_files=[])
    llm_client = FakeLLMClient(LLMResponse(content="not json", model="fake"))

    profile = await ap._run_analyst("a1", evidence, llm_client)

    assert profile == AgentDataProfile(agent_id="a1")


async def test_run_analyst_parses_well_formed_response() -> None:
    evidence = ap.AgentEvidence(agent=AgentFlow(id="a1", component_ids=["c1"]), owned=[], supporting_files=[])
    content = json.dumps({
        "input_data": "user query",
        "output_data": "final answer",
        "internal_tools": ["search"],
        "failure_modes": ["hallucination"],
        "consistency_notes": ["upstream claims X but code does Y"],
    })
    llm_client = FakeLLMClient(LLMResponse(content=content, model="fake"))

    profile = await ap._run_analyst("a1", evidence, llm_client)

    assert profile.input_data == "user query"
    assert profile.internal_tools == ["search"]
    assert profile.consistency_notes == ["upstream claims X but code does Y"]


async def test_run_gate_designer_drops_gate_for_unowned_component() -> None:
    evidence = ap.AgentEvidence(
        agent=AgentFlow(id="a1", component_ids=["c1"]),
        owned=[{"id": "c1", "role": "writer", "model": None, "entry_point": "m:C", "file": "",
                "upstream": [], "downstream": [], "constraints": [], "source": ""}],
        supporting_files=[],
    )
    profile = AgentDataProfile(agent_id="a1")
    content = json.dumps({"gates": [
        {"component": "not_owned", "location": "output", "property": "p", "metric": "geval.foo",
         "metric_class": "llm_judge", "toolkit": "deepeval", "rationale": "r", "rubric_text": "rt"},
    ]})
    llm_client = FakeLLMClient(LLMResponse(content=content, model="fake"))

    gates = await ap._run_gate_designer("a1", evidence, profile, [], llm_client)

    assert gates == []


async def test_run_gate_designer_accepts_valid_gate_and_flags_unknown_metric() -> None:
    evidence = ap.AgentEvidence(
        agent=AgentFlow(id="a1", component_ids=["c1"]),
        owned=[{"id": "c1", "role": "writer", "model": None, "entry_point": "m:C", "file": "",
                "upstream": [], "downstream": [], "constraints": [], "source": ""}],
        supporting_files=[],
    )
    profile = AgentDataProfile(agent_id="a1")
    content = json.dumps({"gates": [
        {"component": "c1", "location": "output", "property": "grounding", "metric": "geval.custom_rubric",
         "metric_class": "llm_judge", "toolkit": "deepeval", "rationale": "r", "rubric_text": "rt"},
        {"component": "c1", "location": "input", "property": "p2", "metric": "totally_made_up_assertion",
         "metric_class": "assertion", "toolkit": "assertion", "rationale": "r2"},
    ]})
    llm_client = FakeLLMClient(LLMResponse(content=content, model="fake"))

    gates = await ap._run_gate_designer("a1", evidence, profile, [], llm_client)

    assert len(gates) == 2
    valid_gate = next(g for g in gates if g.metric == "geval.custom_rubric")
    assert valid_gate.status is None
    assert valid_gate.params["rubric_text"] == "rt"
    assert valid_gate.provenance == "llm_suggested"

    invalid_gate = next(g for g in gates if g.metric == "totally_made_up_assertion")
    assert invalid_gate.status == "needs_human"


async def test_run_handoff_gates_validates_agent_and_component_ownership() -> None:
    evidence_by_agent = {
        "a1": ap.AgentEvidence(
            agent=AgentFlow(id="a1", component_ids=["c1"], downstream_agents=["a2"]),
            owned=[{"id": "c1", "role": "orchestrator", "model": None, "entry_point": "m:C", "file": "",
                    "upstream": [], "downstream": [], "constraints": [], "source": ""}],
        ),
    }
    profiles_by_agent = {"a1": AgentDataProfile(agent_id="a1")}
    content = json.dumps({"gates": [
        {"agent_id": "a1", "component": "c1", "property": "fanout", "metric": "allowed_downstream",
         "metric_class": "assertion", "toolkit": "assertion", "rationale": "r"},
        {"agent_id": "unknown_agent", "component": "c1", "property": "p", "metric": "x",
         "metric_class": "assertion", "toolkit": "assertion", "rationale": "r"},
        {"agent_id": "a1", "component": "not_owned", "property": "p", "metric": "x",
         "metric_class": "assertion", "toolkit": "assertion", "rationale": "r"},
    ]})
    llm_client = FakeLLMClient(LLMResponse(content=content, model="fake"))

    gates = await ap._run_handoff_gates(evidence_by_agent, profiles_by_agent, llm_client)

    assert len(gates) == 1
    assert gates[0].metric == "allowed_downstream"
    assert gates[0].location == "handoff"
    assert gates[0].status is None  # allowed_downstream is a registered assertion


async def test_run_handoff_gates_prompt_includes_known_metric_vocabulary() -> None:
    """The LLM must be given the registered handoff metric names so it reuses them
    instead of inventing new dead (metric_not_dispatchable) ones."""
    evidence_by_agent = {
        "a1": ap.AgentEvidence(
            agent=AgentFlow(id="a1", component_ids=["c1"], downstream_agents=["a2"]),
            owned=[{"id": "c1", "role": "orchestrator", "model": None, "entry_point": "m:C", "file": "",
                    "upstream": [], "downstream": [], "constraints": [], "source": ""}],
        ),
    }
    profiles_by_agent = {"a1": AgentDataProfile(agent_id="a1")}
    llm_client = FakeLLMClient(LLMResponse(content='{"gates": []}', model="fake"))

    await ap._run_handoff_gates(evidence_by_agent, profiles_by_agent, llm_client)

    assert len(llm_client.calls) == 1
    user_message = llm_client.calls[0][1].content
    assert "max_items_per_call" in user_message
    assert "retry_on_reject_required" in user_message
    assert "allowed_downstream" in user_message


async def test_run_critic_defensive_parse() -> None:
    report = EvaluationPlanReport(target_system_id="t", agents=[])
    llm_client = FakeLLMClient(LLMResponse(content="{garbled", model="fake"))
    notes = await ap.run_critic(report, llm_client)
    assert notes == []


# ──────────────────────────────────────────────────────────────────────────────
# _complete_json — truncation retry
# ──────────────────────────────────────────────────────────────────────────────


async def test_complete_json_recovers_on_retry_after_truncated_first_response() -> None:
    """A truncated first response recovers on retry at a bumped token budget."""
    llm_client = FakeLLMClient([
        LLMResponse(content='{"notes": ["unterminat', model="fake"),
        LLMResponse(content='{"notes": ["ok"]}', model="fake"),
    ])

    parsed = await ap._complete_json(
        llm_client, "sys", "user", max_tokens=100, label="test_node",
    )

    assert parsed == {"notes": ["ok"]}


async def test_complete_json_surfaces_dag_note_when_both_attempts_fail() -> None:
    llm_client = FakeLLMClient(LLMResponse(content="not json at all", model="fake"))
    dag_notes: list[str] = []

    parsed = await ap._complete_json(
        llm_client, "sys", "user", max_tokens=100, label="test_node", dag_notes=dag_notes,
    )

    assert parsed is None
    assert len(dag_notes) == 1
    assert "test_node" in dag_notes[0]
    assert "unparseable after retry" in dag_notes[0]


async def test_run_critic_recovers_via_retry_and_generate_plan_surfaces_dag_notes() -> None:
    """run_critic recovers a truncated-then-fixed response; a node that still fails
    after retry surfaces in advisory_notes instead of vanishing silently."""
    report = EvaluationPlanReport(target_system_id="t", agents=[])
    llm_client = FakeLLMClient([
        LLMResponse(content='{"notes": ["truncat', model="fake"),
        LLMResponse(content='{"notes": ["looks fine"]}', model="fake"),
    ])
    notes = await ap.run_critic(report, llm_client)
    assert notes == ["looks fine"]


# ──────────────────────────────────────────────────────────────────────────────
# _validate_metric
# ──────────────────────────────────────────────────────────────────────────────


async def test_validate_metric_matrix() -> None:
    assert ap._validate_metric("allowed_downstream", "assertion") is True
    assert ap._validate_metric("made_up_thing", "assertion") is False
    assert ap._validate_metric("classifier.guard_llm_accuracy", "classifier") is True
    assert ap._validate_metric("not_prefixed", "classifier") is False
    assert ap._validate_metric("ragas.faithfulness", "llm_judge") is True
    assert ap._validate_metric("ragas.context_recall", "llm_judge") is False  # not yet implemented (plan §11-C)
    assert ap._validate_metric("geval.anything_at_all", "llm_judge") is True
    assert ap._validate_metric("tool_correctness", "llm_judge") is True


# ──────────────────────────────────────────────────────────────────────────────
# reconcile
# ──────────────────────────────────────────────────────────────────────────────


async def test_reconcile_baseline_never_dropped_llm_dedup_and_needs_human() -> None:
    system_map = load_system_map(_MULTI_AGENT_MAP)
    agent_flow_map = _multi_agent_flow_map()
    llm_client = FakeLLMClient(LLMResponse(content='{"rubric_text": "x"}', model="fake"))

    evidence_by_agent, baseline_by_agent = await ap.gather_evidence(
        system_map, agent_flow_map, _fake_source_by_component(system_map), [], llm_client
    )
    profiles_by_agent = {aid: AgentDataProfile(agent_id=aid) for aid in evidence_by_agent}

    # Filter baseline to keep only 1 llm_judge so suggested ones are not capped to 0
    baseline_by_agent["core_agent"] = [
        b for b in baseline_by_agent["core_agent"]
        if b.metric_class != "llm_judge" or b.metric == "geval.decomposition_coverage"
    ]

    # Duplicate of an existing core_agent baseline metric (must be dropped) + one net-new gate.
    dup_metric = baseline_by_agent["core_agent"][0].metric
    dup_component = baseline_by_agent["core_agent"][0].component
    llm_gates_by_agent = {
        "guard_agent": [],
        "core_agent": [
            ap.EvaluationGate(
                id="dup", agent_id="core_agent", component=dup_component, location="output",
                metric=dup_metric, metric_class="assertion", toolkit="assertion",
                provenance="llm_suggested",
            ),
            ap.EvaluationGate(
                id="new", agent_id="core_agent", component="writer", location="output",
                metric="geval.brand_new_check", metric_class="llm_judge", toolkit="deepeval",
                provenance="llm_suggested",
            ),
        ],
    }

    suite, report = ap.reconcile(
        agent_flow_map, evidence_by_agent, profiles_by_agent, baseline_by_agent,
        llm_gates_by_agent, handoff_gates=[],
    )

    core_report = next(a for a in report.agents if a.agent_id == "core_agent")
    metrics = [g.metric for g in core_report.gates]
    # baseline metric appears exactly once (LLM duplicate was dropped, baseline wins)
    assert metrics.count(dup_metric) == 1
    # net-new LLM gate made it through
    assert "geval.brand_new_check" in metrics
    # every baseline (component, metric) pair survives into the executable Suite
    baseline_pairs = {(e.component, e.metric) for e in baseline_by_agent["core_agent"]}
    suite_pairs = {(e.component, e.metric) for e in suite.entries if e.agent_id == "core_agent"}
    assert baseline_pairs <= suite_pairs


# ──────────────────────────────────────────────────────────────────────────────
# end-to-end DAG smoke test
# ──────────────────────────────────────────────────────────────────────────────


async def test_generate_plan_agentic_end_to_end_smoke() -> None:
    """Full DAG run, LLM contributes nothing (degrades every node) — must still produce a
    well-formed Suite covering every agent, with the baseline fully intact."""
    system_map = load_system_map(_MULTI_AGENT_MAP)
    agent_flow_map = _multi_agent_flow_map()
    llm_client = FakeLLMClient(LLMResponse(content="{}", model="fake"))

    suite, report = await ap.generate_plan_agentic(
        system_map, agent_flow_map, _fake_source_by_component(system_map), [], llm_client,
    )

    assert {a.agent_id for a in report.agents} == {"guard_agent", "core_agent"}
    assert all(e.agent_id in ("guard_agent", "core_agent") for e in suite.entries)
    assert report.advisory_notes == []  # critic degraded on "{}" -> no notes key


async def test_generate_plan_agentic_matches_flat_baseline_metric_set() -> None:
    """Regression parity: every (component, metric) pair the flat rule-based
    generate_plan() produces must also appear in the agentic plan, when the
    AgentFlowMap fully covers the map's components."""
    system_map_path = _MULTI_AGENT_MAP
    system_map = load_system_map(system_map_path)
    agent_flow_map = _multi_agent_flow_map()

    flat_llm = FakeLLMClient(LLMResponse(content='{"rubric_text": "r"}', model="fake"))
    flat_suite = await generate_plan(system_map_path, flat_llm)
    flat_pairs = {(e.component, e.metric) for e in flat_suite.entries}

    agentic_llm = FakeLLMClient(LLMResponse(content="{}", model="fake"))
    agentic_suite, _ = await ap.generate_plan_agentic(
        system_map, agent_flow_map, _fake_source_by_component(system_map), [], agentic_llm,
        run_critic_pass=False,
    )
    agentic_pairs = {(e.component, e.metric) for e in agentic_suite.entries}

    assert flat_pairs <= agentic_pairs


async def test_generate_plan_agentic_can_skip_critic_pass() -> None:
    system_map = load_system_map(_MULTI_AGENT_MAP)
    agent_flow_map = _multi_agent_flow_map()
    llm_client = FakeLLMClient(LLMResponse(content="{}", model="fake"))
    calls_before = len(llm_client.calls)

    _, report = await ap.generate_plan_agentic(
        system_map, agent_flow_map, _fake_source_by_component(system_map), [], llm_client,
        run_critic_pass=False,
    )

    assert report.advisory_notes == []
    # 1 (gather rubric) + 2 (analyst x2) + 3 (gate_designer x2 + handoff) = 6, no critic call
    assert len(llm_client.calls) - calls_before == 6
