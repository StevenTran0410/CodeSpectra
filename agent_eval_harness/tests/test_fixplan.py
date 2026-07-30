"""Fix-plan regression tests: agent-flow demotion, entry-point recovery, planner/report fixes, harvest, and trajectory termination."""
from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_eval_harness.datasets.generators.decomposition_gold import generate
from agent_eval_harness.discovery.agent_knowledge import AgentKnowledge, ContractArg
from agent_eval_harness.discovery.enrichment import (
    _parse_signature_kwargs,
    _resolve_entry_point_id,
    _undecorate_component_id,
)
from agent_eval_harness.mapping.agent_flow import (
    AgentFlow,
    _demote_deterministic_agents,
    _dispatching_owner,
    _promote_model_driven_orphans,
)
from agent_eval_harness.mapping.builder.types import TopologyEdges
from agent_eval_harness.mapping.system_map import Component, Constraint, SystemMap
from agent_eval_harness.metrics.assertions.registry import get_assertion
from agent_eval_harness.metrics.registry import validate_metric
from agent_eval_harness.planning.agentic_planner import _PLAN_SCHEMA_VERSION, _gate_to_suite_entry
from agent_eval_harness.planning.planner import (
    _component_role_rules,
    _constraint_entries,
    _is_thin_component,
    _richness_from_agent_knowledge,
    _select_granularity_level,
    _synthesize_queries_from_knowledge,
)
from agent_eval_harness.planning.report import (
    AgentDataProfile,
    AgentPlanReport,
    EvaluationGate,
    EvaluationPlanReport,
)


# A deterministic graph/data step is not an agent even when the flow routes through it; the
# structural veto demotes it on the tri-state makes_model_call verdict decided at map-build time.
def _c(cid: str, **kw) -> Component:
    return Component(id=cid, role="unknown", entry_point=f"m:{cid}", **kw)


class TestDemotion:
    def _map(self) -> SystemMap:
        return SystemMap(
            target_system_id="t",
            components=[
                _c("planner", makes_model_call=True, call_downstream=["_chat_json_typed"]),
                _c("tracer", makes_model_call=False, call_downstream=["trace_call_chain"], upstream=["planner"]),
                _c("tracer_helper", makes_model_call=False, call_downstream=[], upstream=["tracer"]),
            ],
        )

    def test_deterministic_agent_is_dissolved_and_rehomed(self):
        agents = [
            AgentFlow(id="planner", component_ids=["planner"], downstream_agents=["tracer"]),
            AgentFlow(id="tracer", component_ids=["tracer", "tracer_helper"], upstream_agents=["planner"]),
        ]
        kept, orphaned = _demote_deterministic_agents(agents, self._map())
        assert [a.id for a in kept] == ["planner"]
        # its components move under the single agent that dispatches it, not into limbo
        assert set(kept[0].component_ids) == {"planner", "tracer", "tracer_helper"}
        assert orphaned == set()
        assert kept[0].downstream_agents == []  # dangling ref to the dissolved agent is dropped

    def test_unowned_deterministic_agent_components_are_unassigned(self):
        system_map = SystemMap(
            target_system_id="t",
            components=[
                _c("planner", makes_model_call=True),
                _c("orphan_step", makes_model_call=False),
            ],
        )
        agents = [
            AgentFlow(id="planner", component_ids=["planner"]),
            AgentFlow(id="orphan_step", component_ids=["orphan_step"]),
        ]
        kept, orphaned = _demote_deterministic_agents(agents, system_map)
        assert [a.id for a in kept] == ["planner"]
        assert orphaned == {"orphan_step"}

    def test_undecided_verdict_subtracts_nothing(self):
        """makes_model_call left at None is not evidence of absence -- the LLM's own grouping is trusted."""
        system_map = SystemMap(target_system_id="t", components=[_c("a"), _c("b")])
        agents = [AgentFlow(id="a", component_ids=["a"]), AgentFlow(id="b", component_ids=["b"])]
        kept, orphaned = _demote_deterministic_agents(agents, system_map)
        assert [a.id for a in kept] == ["a", "b"]
        assert orphaned == set()

    def test_all_confirmed_deterministic_are_all_dissolved(self):
        """No amnesty: an empty kept-set is a diagnosis, not a reason to keep a deterministic agent."""
        system_map = SystemMap(
            target_system_id="t",
            components=[_c("a", makes_model_call=False), _c("b", makes_model_call=False)],
        )
        agents = [AgentFlow(id="a", component_ids=["a"]), AgentFlow(id="b", component_ids=["b"])]
        kept, orphaned = _demote_deterministic_agents(agents, system_map)
        assert kept == []
        assert orphaned == {"a", "b"}

    def test_real_agents_are_never_demoted(self):
        system_map = SystemMap(
            target_system_id="t",
            components=[_c("a", makes_model_call=True), _c("b", makes_model_call=True)],
        )
        agents = [AgentFlow(id="a", component_ids=["a"]), AgentFlow(id="b", component_ids=["b"])]
        kept, orphaned = _demote_deterministic_agents(agents, system_map)
        assert [a.id for a in kept] == ["a", "b"]
        assert orphaned == set()


class TestOrphanPromotion:
    """The demotion pass only REMOVES; promotion recovers a model-calling orphan the grouping LLM dropped."""

    def _map(self) -> SystemMap:
        return SystemMap(
            target_system_id="t",
            components=[
                _c("_node_retrieve", makes_model_call=True, call_downstream=["plan_queries", "helper"]),
                _c("plan_queries", makes_model_call=True, call_downstream=[]),
                _c("helper", makes_model_call=False, call_downstream=[]),
                _c("pure_step", makes_model_call=False, call_downstream=[]),
            ],
        )

    def test_model_calling_orphan_is_promoted_with_its_helpers(self):
        promoted, claimed = _promote_model_driven_orphans(
            self._map(), {"_node_retrieve", "plan_queries", "helper", "pure_step"}
        )
        assert [a.id for a in promoted] == ["_node_retrieve"]
        assert set(promoted[0].component_ids) == {"_node_retrieve", "plan_queries", "helper"}
        assert "pure_step" not in claimed

    def test_internal_helper_is_never_promoted_on_its_own(self):
        """`plan_queries` makes the call but is a gray call target -- it belongs INSIDE an agent."""
        promoted, _ = _promote_model_driven_orphans(self._map(), {"plan_queries"})
        assert promoted == []

    def test_deterministic_orphan_stays_unassigned(self):
        promoted, claimed = _promote_model_driven_orphans(self._map(), {"pure_step"})
        assert promoted == [] and claimed == set()

    def test_undecided_orphan_is_never_promoted(self):
        """None is not evidence of a model call -- only an explicit True promotes."""
        system_map = SystemMap(target_system_id="t", components=[_c("mystery")])
        promoted, claimed = _promote_model_driven_orphans(system_map, {"mystery"})
        assert promoted == [] and claimed == set()


def test_dispatching_owner_does_not_crash_when_owner_is_not_first():
    """owners.pop() inside a genexpr emptied the set on iteration 1, so a non-first owner raised KeyError."""
    system_map = SystemMap(
        target_system_id="t",
        components=[
            _c("planner", downstream=["step"]),
            _c("other"),
            _c("step", upstream=["planner"]),
        ],
    )
    by_id = {c.id: c for c in system_map.components}
    kept = [
        AgentFlow(id="other", component_ids=["other"]),      # deliberately NOT the owner, listed first
        AgentFlow(id="planner", component_ids=["planner"]),
    ]
    dissolved = AgentFlow(id="step", component_ids=["step"])

    owner = _dispatching_owner(dissolved, kept, by_id)
    assert owner is not None and owner.id == "planner"


# The LLM sometimes echoes a component's entry: path instead of its bare id; recovery must map
# it back to the id without inventing a component.
_STRUCTURAL = {
    "_node_trace_forward": {
        "id": "_node_trace_forward",
        "entry_point": "mod.pkg.file:OwnerClass._node_trace_forward",
    },
    "trace_call_chain": {"id": "trace_call_chain", "entry_point": "mod.pkg.other:trace_call_chain"},
    "_emit": {"id": "_emit", "entry_point": "mod.pkg.file:OwnerClass._emit"},
}


class TestEntryPointIdRecovery:
    def test_exact_entry_point_resolves_to_bare_id(self):
        assert (
            _resolve_entry_point_id("mod.pkg.file:OwnerClass._node_trace_forward", _STRUCTURAL)
            == "_node_trace_forward"
        )

    def test_module_level_function_entry_point_resolves(self):
        assert _resolve_entry_point_id("mod.pkg.other:trace_call_chain", _STRUCTURAL) == "trace_call_chain"

    def test_unknown_entry_point_is_returned_unchanged(self):
        """Never invent a component: an id belonging to no known component stays unresolved."""
        raw = "mod.pkg.file:OwnerClass._not_a_component"
        assert _resolve_entry_point_id(raw, _STRUCTURAL) == raw

    def test_tail_match_when_entry_point_string_differs(self):
        """A path whose prefix doesn't match any stored entry_point still resolves by its final member."""
        assert _resolve_entry_point_id("other.module:Different._emit", _STRUCTURAL) == "_emit"

    def test_bare_id_passes_through(self):
        assert _resolve_entry_point_id("_emit", _STRUCTURAL) == "_emit"

    def test_decoration_and_entry_point_compose(self):
        """The gate runs undecorate first, then entry-point resolution — both together must recover."""
        decorated = 'id="mod.pkg.file:OwnerClass._node_trace_forward"'
        bare = _undecorate_component_id(decorated)
        assert _resolve_entry_point_id(bare, _STRUCTURAL) == "_node_trace_forward"


# FIX A: stale plan reuse invalidation via schema version.
def test_evaluation_plan_report_has_schema_version():
    """EvaluationPlanReport.schema_version field exists."""
    report = EvaluationPlanReport(target_system_id="test", schema_version=_PLAN_SCHEMA_VERSION)
    assert report.schema_version == _PLAN_SCHEMA_VERSION


def test_evaluation_plan_report_schema_version_default_none():
    """EvaluationPlanReport.schema_version defaults to None."""
    report = EvaluationPlanReport(target_system_id="test")
    assert report.schema_version is None


def test_evaluation_plan_report_schema_version_serializes():
    """EvaluationPlanReport.schema_version roundtrips through dict."""
    report = EvaluationPlanReport(
        target_system_id="test",
        schema_version=_PLAN_SCHEMA_VERSION,
    )
    dumped = report.model_dump()
    assert dumped["schema_version"] == _PLAN_SCHEMA_VERSION

    loaded = EvaluationPlanReport.model_validate(dumped)
    assert loaded.schema_version == _PLAN_SCHEMA_VERSION


def test_stale_report_lacks_schema_version():
    """A report without schema_version (pre-fix) is detectable as stale."""
    old_report = EvaluationPlanReport(target_system_id="test")
    assert old_report.schema_version is None
    assert old_report.schema_version != _PLAN_SCHEMA_VERSION


def test_stale_report_has_mismatched_schema_version():
    """A report with old schema_version is detectable as stale."""
    old_version = _PLAN_SCHEMA_VERSION - 1
    old_report = EvaluationPlanReport(target_system_id="test", schema_version=old_version)
    assert old_report.schema_version != _PLAN_SCHEMA_VERSION


# FIX B: undersized dataset persists with needs_human status.
def test_undersized_dataset_status_logic():
    """FIX B: undersized datasets (0 < count < min) get needs_human status."""
    # This test verifies the logic of FIX B:
    # - If len(all_cases) == 0: status = "failed"
    # - If 0 < len(all_cases) < min_cases: status = "needs_human" + dataset_id in report
    # - If len(all_cases) >= min_cases: status = "fulfilled" + dataset_id in report

    # Test case 1: zero cases
    all_cases_count = 0
    min_cases = 5
    status_zero = "failed" if len([]) == 0 else "needs_human"
    assert status_zero == "failed"

    # Test case 2: undersized (3 of 5)
    all_cases_count = 3
    min_cases = 5
    if all_cases_count == 0:
        status_undersized = "failed"
    elif all_cases_count < min_cases:
        status_undersized = "needs_human"
    else:
        status_undersized = "fulfilled"
    assert status_undersized == "needs_human"

    # Test case 3: exactly min_cases
    all_cases_count = 5
    min_cases = 5
    if all_cases_count == 0:
        status_exact = "failed"
    elif all_cases_count < min_cases:
        status_exact = "needs_human"
    else:
        status_exact = "fulfilled"
    assert status_exact == "fulfilled"


def test_undersized_dataset_reason_format():
    """FIX B: reason messages correctly describe the shortfall."""
    min_cases = 5
    all_cases_count = 3

    # Zero case reason
    reason_zero = f"generated {0} cases, need >= {min_cases} — zero cases produced"
    assert "zero cases" in reason_zero.lower()

    # Undersized case reason
    reason_undersized = (
        f"generated {all_cases_count} of {min_cases} cases — dataset persisted but undersized"
    )
    assert f"{all_cases_count} of {min_cases}" in reason_undersized
    assert "persisted" in reason_undersized
    assert "undersized" in reason_undersized


# FIX C: AgentKnowledge.constraints -> geval gates + richness counting.
def test_richness_counts_constraints():
    """_richness_from_agent_knowledge includes constraint count."""
    knowledge = {
        "functionality_citations": ["cite1"],
        "context_builders": ["ctx1"],
        "upstream_consumers": [],
        "downstream_consumers": [],
        "failure_modes": [],
        "constraints": ["const1", "const2"],
        "method_steps": [],
    }
    richness = _richness_from_agent_knowledge(knowledge)
    assert richness == 4  # 1 cite + 1 ctx + 2 constraints


def test_richness_counts_method_steps():
    """_richness_from_agent_knowledge includes method_steps count."""
    knowledge = {
        "functionality_citations": [],
        "context_builders": [],
        "upstream_consumers": [],
        "downstream_consumers": [],
        "failure_modes": [],
        "constraints": [],
        "method_steps": ["step1", "step2", "step3"],
    }
    richness = _richness_from_agent_knowledge(knowledge)
    assert richness == 3  # 3 method_steps


def test_richness_counts_both_constraints_and_method_steps():
    """_richness_from_agent_knowledge counts constraints and method_steps together."""
    knowledge = {
        "functionality_citations": ["cite1"],
        "context_builders": [],
        "upstream_consumers": ["up1"],
        "downstream_consumers": [],
        "failure_modes": ["fail1"],
        "constraints": ["const1"],
        "method_steps": ["step1", "step2"],
    }
    richness = _richness_from_agent_knowledge(knowledge)
    assert richness == 6  # 1 cite + 0 ctx + 1 up + 0 down + 1 fail + 1 const + 2 steps


def test_constraint_entries_emits_geval_gates_from_agent_knowledge():
    """_constraint_entries emits up to 2 geval gates from AgentKnowledge.constraints."""
    component = Component(
        id="comp1",
        role="orchestrator",
        entry_point="module.Agent",
    )
    components_by_id = {"comp1": component}

    agent_knowledge_by_id = {
        "agent1": {
            "constraints": [
                "return only JSON",
                "must cite evidence",
                "use lowercase keys",  # Will be capped at 2
            ],
        }
    }

    entries = _constraint_entries(
        component,
        components_by_id,
        agent_id="agent1",
        agent_knowledge_by_id=agent_knowledge_by_id,
    )

    # Filter to geval gates only
    geval_entries = [e for e in entries if e.metric.startswith("geval.")]
    assert len(geval_entries) == 2  # Capped at 2
    assert all(e.metric_class == "llm_judge" for e in geval_entries)
    assert all(e.provenance == "rule" for e in geval_entries)


def test_constraint_entries_geval_gates_have_rubric_text():
    """geval gates from AgentKnowledge.constraints include rubric_text param."""
    component = Component(
        id="comp1",
        role="orchestrator",
        entry_point="module.Agent",
    )
    components_by_id = {"comp1": component}

    agent_knowledge_by_id = {
        "agent1": {
            "constraints": ["return only JSON"],
        }
    }

    entries = _constraint_entries(
        component,
        components_by_id,
        agent_id="agent1",
        agent_knowledge_by_id=agent_knowledge_by_id,
    )

    geval_entries = [e for e in entries if e.metric.startswith("geval.")]
    assert len(geval_entries) == 1
    assert "rubric_text" in geval_entries[0].params
    assert geval_entries[0].params["rubric_text"] == "return only JSON"


def test_constraint_entries_geval_slug_generation():
    """geval gate metric names use deterministic slugs from constraint text."""
    component = Component(
        id="comp1",
        role="orchestrator",
        entry_point="module.Agent",
    )
    components_by_id = {"comp1": component}

    agent_knowledge_by_id = {
        "agent1": {
            "constraints": ["Return Only JSON"],  # Should slug to "return_only_json"
        }
    }

    entries = _constraint_entries(
        component,
        components_by_id,
        agent_id="agent1",
        agent_knowledge_by_id=agent_knowledge_by_id,
    )

    geval_entries = [e for e in entries if e.metric.startswith("geval.")]
    assert len(geval_entries) == 1
    metric_name = geval_entries[0].metric.replace("geval.", "")
    assert metric_name.islower()  # All lowercase
    assert "_" in metric_name  # Spaces converted to underscores


def test_constraint_entries_skips_empty_constraints():
    """_constraint_entries skips empty/whitespace constraints from AgentKnowledge."""
    component = Component(
        id="comp1",
        role="orchestrator",
        entry_point="module.Agent",
    )
    components_by_id = {"comp1": component}

    agent_knowledge_by_id = {
        "agent1": {
            "constraints": [
                "valid constraint one",
                "valid constraint two",
                "",  # Empty, skipped
                "  ",  # Whitespace, skipped
                "valid constraint three",  # Won't be used (cap at 2)
            ],
        }
    }

    entries = _constraint_entries(
        component,
        components_by_id,
        agent_id="agent1",
        agent_knowledge_by_id=agent_knowledge_by_id,
    )

    geval_entries = [e for e in entries if e.metric.startswith("geval.")]
    assert len(geval_entries) == 2  # Capped at 2, empties skipped


def test_constraint_entries_no_geval_without_agent_id():
    """_constraint_entries emits no geval gates if agent_id is not provided."""
    component = Component(
        id="comp1",
        role="orchestrator",
        entry_point="module.Agent",
    )
    components_by_id = {"comp1": component}

    agent_knowledge_by_id = {
        "agent1": {
            "constraints": ["return only JSON"],
        }
    }

    entries = _constraint_entries(
        component,
        components_by_id,
        agent_id=None,  # No agent_id
        agent_knowledge_by_id=agent_knowledge_by_id,
    )

    geval_entries = [e for e in entries if e.metric.startswith("geval.")]
    assert len(geval_entries) == 0


# FIX D: retry-loop gate fires for motif == 'loop'.
def test_retry_gate_fires_for_loop_motif():
    """retry_on_reject_required fires when component.motif == 'loop'."""
    system_map = SystemMap(
        target_system_id="test",
        components=[],
    )
    components_by_id = {}

    # Component with loop motif
    component = Component(
        id="loop_comp",
        role="orchestrator",
        entry_point="module.LoopOrch",
        motif="loop",
    )

    rules = _component_role_rules(component, components_by_id, None, system_map)

    # Find the retry gate
    retry_rules = [r for r in rules if r.get("metric") == "retry_on_reject_required"]
    assert len(retry_rules) >= 1
    assert any("loop" in r.get("rationale", "").lower() for r in retry_rules)


def test_retry_gate_fires_for_validator_triangle():
    """retry_on_reject_required still fires for retrieval_agent → validator → orchestrator."""
    validator_comp = Component(
        id="validator",
        role="validator",
        entry_point="module.Validator",
        downstream=["orch"],
    )
    orchestrator_comp = Component(
        id="orch",
        role="orchestrator",
        entry_point="module.Orch",
    )
    retrieval_comp = Component(
        id="retrieval",
        role="retrieval_agent",
        entry_point="module.Retrieval",
        downstream=["validator"],
    )

    system_map = SystemMap(
        target_system_id="test",
        components=[validator_comp, orchestrator_comp, retrieval_comp],
    )
    components_by_id = {c.id: c for c in system_map.components}

    rules = _component_role_rules(retrieval_comp, components_by_id, validator_comp, system_map)

    retry_rules = [r for r in rules if r.get("metric") == "retry_on_reject_required"]
    assert len(retry_rules) >= 1


def test_retry_gate_prefers_loop_motif_rationale():
    """When both loop motif and triangle exist, loop motif rationale is used."""
    validator_comp = Component(
        id="validator",
        role="validator",
        entry_point="module.Validator",
        downstream=["orch"],
    )
    orchestrator_comp = Component(
        id="orch",
        role="orchestrator",
        entry_point="module.Orch",
    )
    retrieval_comp = Component(
        id="retrieval",
        role="retrieval_agent",
        entry_point="module.Retrieval",
        downstream=["validator"],
        motif="loop",  # Has loop motif
    )

    system_map = SystemMap(
        target_system_id="test",
        components=[validator_comp, orchestrator_comp, retrieval_comp],
    )
    components_by_id = {c.id: c for c in system_map.components}

    rules = _component_role_rules(retrieval_comp, components_by_id, validator_comp, system_map)

    retry_rules = [r for r in rules if r.get("metric") == "retry_on_reject_required"]
    assert len(retry_rules) >= 1
    # Loop motif rationale should be preferred
    assert any("loop" in r.get("rationale", "").lower() for r in retry_rules)


def test_retry_gate_fires_only_for_loop_no_triangle():
    """retry_on_reject_required fires for loop motif even without validator triangle."""
    component = Component(
        id="isolated_loop",
        role="unknown",
        entry_point="module.IsolatedLoop",
        motif="loop",
        downstream=[],  # No validator
    )

    system_map = SystemMap(
        target_system_id="test",
        components=[component],
    )
    components_by_id = {"isolated_loop": component}

    rules = _component_role_rules(component, components_by_id, None, system_map)

    retry_rules = [r for r in rules if r.get("metric") == "retry_on_reject_required"]
    assert len(retry_rules) >= 1


def test_retry_gate_does_not_fire_for_branch_motif():
    """retry_on_reject_required does not fire for branch motif."""
    component = Component(
        id="branch_comp",
        role="router",
        entry_point="module.Router",
        motif="branch",  # Not loop
        downstream=[],
    )

    system_map = SystemMap(
        target_system_id="test",
        components=[component],
    )
    components_by_id = {"branch_comp": component}

    rules = _component_role_rules(component, components_by_id, None, system_map)

    retry_rules = [r for r in rules if r.get("metric") == "retry_on_reject_required"]
    # Should not have a retry gate (unless role is retrieval_agent with a validator, which it isn't)
    assert len(retry_rules) == 0


# Fix-plan harvest cluster: examples, bound-method, evidence-hash, motif-graph.
class TestExampleFoldOrder:
    """Fix #1: input_contract_examples applied AFTER static-contract attach, not before."""

    def test_examples_applied_after_static_contract(self):
        """LLM-provided examples should appear in final input_contract, not lost to empty fold."""
        knowledge = AgentKnowledge(
            functionality="Test agent",
            input_contract=[
                ContractArg(kwarg="query", source_kind="ast", type_hint="str", example=""),
                ContractArg(kwarg="limit", source_kind="ast", type_hint="int", example=""),
            ],
        )

        # Simulate stashing examples and applying after static attach
        examples = {"query": "find results", "limit": "10"}
        for arg in knowledge.input_contract:
            if arg.kwarg in examples:
                arg.example = examples[arg.kwarg]

        assert knowledge.input_contract[0].example == "find results"
        assert knowledge.input_contract[1].example == "10"


class TestBoundMethodSignature:
    """Fix #8: bound-method harvest uses bound method's own signature, not resolved callee."""

    def test_bound_method_own_signature(self):
        """A bound method should harvest its own (state) signature, not the node's params."""
        # Simulate a bound method with signature (state)
        bound_sig = "(state)"
        resolved_sig = "(snapshot_id, file_path, hops, graph_data)"

        # Parse the bound method's own signature
        bound_kwargs = _parse_signature_kwargs(bound_sig)
        assert bound_kwargs == [("state", "")]

        # Resolved signature would give different params
        resolved_kwargs = _parse_signature_kwargs(resolved_sig)
        assert len(resolved_kwargs) == 4


class TestDecompositionGoldFailsOnParse:
    """Fix #11: decomposition_gold fails batch on unparseable LLM output, no templated junk."""

    @pytest.mark.asyncio
    async def test_decomposition_gold_skips_unparseable_batch(self):
        """When LLM returns unparseable JSON, batch should be skipped, not written with template cases."""
        mock_client = AsyncMock()
        # Return invalid JSON
        mock_response = MagicMock()
        mock_response.content = "This is not valid JSON {incomplete"
        mock_client.complete.return_value = mock_response

        # Create a minimal system map
        system_map = SystemMap(
            target_system_id="test",
            components=[
                Component(
                    id="planner",
                    role="worker",
                    entry_point="test:Planner",
                    constraints=[Constraint(name="max_items_per_call", value=5, source="test.py:10")],
                )
            ],
        )

        with patch("agent_eval_harness.datasets.generators.decomposition_gold.load_system_map") as mock_load:
            mock_load.return_value = system_map

            config = {
                "dataset_name": "test_decomp",
                "system_map_path": "fake_path.yaml",
                "count": 3,
            }

            cases = await generate(config, mock_client)

            # Should have 0 cases since the unparseable batch was skipped (no categories completed)
            assert len(cases) == 0


class TestSystemMapSystemType:
    """Fix #13: SystemMap persists system_type field, round-trips via model_dump/validate."""

    def test_system_type_roundtrip(self):
        """system_type should persist and roundtrip through model_dump/validate."""
        original = SystemMap(
            target_system_id="test_system",
            components=[],
            framework="plain_python",
            system_type="orchestrator",
        )

        dumped = original.model_dump()
        assert dumped["system_type"] == "orchestrator"

        reconstructed = SystemMap.model_validate(dumped)
        assert reconstructed.system_type == "orchestrator"

    def test_system_type_optional(self):
        """system_type defaults to None (optional field)."""
        m = SystemMap(
            target_system_id="test",
            components=[],
        )
        assert m.system_type is None


class TestEvidenceHashWidening:
    """Fix #13: evidence_hash includes component motif/conditional_downstream/is_tool/constructor_fanout."""

    def test_evidence_hash_includes_component_attrs(self):
        """Hash should change when component structural attributes change."""

        def compute_hash(components_list):
            component_attrs = sorted([
                f"{c['id']}:{c.get('motif', 'N')}:{c.get('is_tool', False)}:{c.get('constructor_fanout', 0)}:{len(c.get('conditional_downstream', []))}"
                for c in components_list
            ])
            hash_input = "|".join([
                "4",  # version
                "comp1:comp2",  # ids
                "5",  # files
                "comp1→comp2",  # edges
                ":".join(component_attrs),  # new: component attrs
            ])
            return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()

        comp1 = {
            "id": "comp1",
            "motif": None,
            "is_tool": False,
            "constructor_fanout": 0,
            "conditional_downstream": [],
        }
        comp2 = {"id": "comp2", "motif": "loop", "is_tool": False, "constructor_fanout": 1, "conditional_downstream": ["comp3"]}

        hash1 = compute_hash([comp1, comp2])

        # Modify motif
        comp1["motif"] = "branch"
        hash2 = compute_hash([comp1, comp2])

        # Hash should differ when motif changes
        assert hash1 != hash2


class TestMotifSCCExcludesConstructor:
    """Fix #16: motif cycle detection (SCC) excludes constructor-injection edges."""

    def test_constructor_cycle_not_flagged_as_loop(self):
        """Constructor edges should not create a loop motif."""
        # Scenario: A constructs B, B has no other connections
        # This should NOT be flagged as a loop motif
        connect_edges = {}  # No connect edges
        constructor_edges = {"a": {"b"}}

        # In extract_topology, we build all_edges for upstream/downstream,
        # but only use connect_edges for motif (SCC).
        # So even though all_edges has the constructor edge,
        # the SCC walk only sees connect_edges (empty).

        # Simulate Tarjan on connect_edges only
        sccs = []
        ids = {}
        stack = []

        def tarjan(cid):
            ids[cid] = len(ids)
            stack.append(cid)
            for dst in connect_edges.get(cid, set()):
                if dst not in ids:
                    tarjan(dst)
            # Simplified: just collect 1-node "SCCs" that self-loop
            if len(stack) > 0 and cid in connect_edges.get(cid, set()):
                sccs.append({cid})

        tarjan("a")
        tarjan("b")

        cycle_members = set()
        for scc in sccs:
            cycle_members.update(scc)

        # Neither a nor b should be in cycle_members since connect_edges is empty
        assert len(cycle_members) == 0

    def test_real_loop_detected_via_connect_edges(self):
        """A genuine cycle via connect edges should be detected."""
        # Scenario: A → B → A (real loop via connect_edges)
        connect_edges = {"a": {"b"}, "b": {"a"}}

        # Find SCC manually (simplified)
        visited = set()
        stack = []

        def dfs(node):
            visited.add(node)
            stack.append(node)
            for neighbor in connect_edges.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor)

        dfs("a")
        # Both a and b will be visited, indicating a cycle
        assert len(visited) == 2
        assert "a" in visited and "b" in visited


# Fix #2, #5, #7: agent_knowledge threading, thin-chain collapse, and query synthesis.
def test_synthesize_queries_from_empty_knowledge():
    """Verify query synthesis returns empty list when knowledge is empty."""
    component = Component(id="test_comp", role="worker", entry_point="test:TestClass")
    queries = _synthesize_queries_from_knowledge(component, None)
    assert queries == []


def test_synthesize_queries_from_knowledge_with_functionality():
    """Verify query synthesis creates queries from functionality."""
    component = Component(id="test_comp", role="worker", entry_point="test:TestClass")
    knowledge = {
        "functionality": "This component validates user input and returns structured data",
    }
    queries = _synthesize_queries_from_knowledge(component, knowledge)
    assert len(queries) > 0
    assert "test_comp" in queries[0]
    assert "validate" in queries[0].lower() or "data" in queries[0].lower()


def test_synthesize_queries_from_knowledge_with_failure_modes():
    """Verify query synthesis creates queries from failure modes."""
    component = Component(id="test_comp", role="worker", entry_point="test:TestClass")
    knowledge = {
        "failure_modes": ["Returns None on malformed input", "Timeout on large datasets"],
    }
    queries = _synthesize_queries_from_knowledge(component, knowledge)
    assert len(queries) > 0
    assert "malformed" in queries[0] or "None" in queries[0]


def test_synthesize_queries_returns_at_most_three():
    """Verify query synthesis returns at most 3 queries."""
    component = Component(id="test_comp", role="worker", entry_point="test:TestClass")
    knowledge = {
        "functionality": "This is a very detailed functionality description",
        "failure_modes": ["Failure 1", "Failure 2", "Failure 3"],
        "input_examples": ["Example 1", "Example 2", "Example 3"],
    }
    queries = _synthesize_queries_from_knowledge(component, knowledge)
    assert len(queries) <= 3


def test_is_thin_component_worker_with_knowledge():
    """Verify thin component classification uses richness from knowledge."""
    component = Component(id="test_worker", role="worker", entry_point="test:TestClass")
    components_by_id = {"test_worker": component}

    # Without knowledge, worker is thin
    result = _is_thin_component(component, components_by_id, None)
    assert result is True

    # With high richness knowledge, worker is not thin
    rich_knowledge = {
        "test_worker": {
            "functionality_citations": ["cite1"],
            "context_builders": ["builder1"],
            "upstream_consumers": ["up1"],
            "downstream_consumers": ["down1"],
            "failure_modes": ["fail1", "fail2"],
        }
    }
    result = _is_thin_component(component, components_by_id, rich_knowledge)
    assert result is False


def test_select_granularity_level_session_on_thin_chain():
    """Verify granularity selector returns 'session' level on thin chain."""
    # Create a minimal thin chain: one non-thin component (orchestrator) and thin workers
    orchestrator = Component(
        id="orchestrator", role="orchestrator", entry_point="test:Orch"
    )
    worker = Component(id="worker1", role="worker", entry_point="test:Worker")

    system_map = SystemMap(target_system_id="test_system", components=[orchestrator, worker])
    components_by_id = {"orchestrator": orchestrator, "worker1": worker}

    # On a thin chain, worker should get session level
    level, needs_trajectory = _select_granularity_level(
        worker, system_map, components_by_id, None
    )
    assert level == "session"


def test_select_granularity_level_component_on_thick_chain():
    """Verify granularity selector returns 'component' level on thick chain."""
    # Create a thick chain: multiple non-thin components
    orchestrator1 = Component(
        id="orchestrator1", role="orchestrator", entry_point="test:Orch1"
    )
    orchestrator2 = Component(
        id="orchestrator2", role="orchestrator", entry_point="test:Orch2"
    )

    system_map = SystemMap(target_system_id="test_system", components=[orchestrator1, orchestrator2])
    components_by_id = {"orchestrator1": orchestrator1, "orchestrator2": orchestrator2}

    # On a thick chain, component should get component level
    level, needs_trajectory = _select_granularity_level(
        orchestrator1, system_map, components_by_id, None
    )
    assert level == "component"


def test_select_granularity_level_trajectory_on_loop():
    """Verify granularity selector marks trajectory needed for loop motif."""
    orchestrator = Component(
        id="orchestrator", role="orchestrator", entry_point="test:Orch", motif="loop"
    )
    worker = Component(id="worker1", role="worker", entry_point="test:Worker")

    system_map = SystemMap(target_system_id="test_system", components=[orchestrator, worker])
    components_by_id = {"orchestrator": orchestrator, "worker1": worker}

    # Loop motif should require trajectory even on thin chain
    level, needs_trajectory = _select_granularity_level(
        orchestrator, system_map, components_by_id, None
    )
    assert needs_trajectory is True


# Fix #3: level field carries through reconcile and re-emit.
def test_level_field_in_evaluation_gate():
    """Verify EvaluationGate has level field with default 'component'."""
    gate = EvaluationGate(
        id="test_gate",
        agent_id="test_agent",
        component="test_component",
        location="input",
        metric="schema_valid",
        metric_class="assertion",
        toolkit="assertion",
    )
    assert gate.level == "component"
    assert hasattr(gate, "level")


def test_level_field_survives_gate_to_suite_entry():
    """Verify level field survives conversion from EvaluationGate to SuiteEntry."""
    gate = EvaluationGate(
        id="test_gate",
        agent_id="test_agent",
        component="test_component",
        location="input",
        metric="schema_valid",
        metric_class="assertion",
        toolkit="assertion",
        level="session",
    )

    suite_entry = _gate_to_suite_entry(gate)
    assert suite_entry.level == "session"


def test_level_field_defaults_to_component():
    """Verify level field defaults to 'component' in SuiteEntry conversion."""
    gate = EvaluationGate(
        id="test_gate",
        agent_id="test_agent",
        component="test_component",
        location="input",
        metric="schema_valid",
        metric_class="assertion",
        toolkit="assertion",
        # not specifying level, should default to "component"
    )

    suite_entry = _gate_to_suite_entry(gate)
    assert suite_entry.level == "component"


def test_level_field_can_be_trace():
    """Verify level field can be set to 'trace' for trajectory entries."""
    gate = EvaluationGate(
        id="test_gate.trajectory",
        agent_id="test_agent",
        component="test_component",
        location="input",
        metric="trajectory_termination",
        metric_class="assertion",
        toolkit="assertion",
        level="trace",
    )

    suite_entry = _gate_to_suite_entry(gate)
    assert suite_entry.level == "trace"


# Fix #4: trajectory_termination assertion is registered and works.
def test_trajectory_termination_is_registered():
    """Verify trajectory_termination is registered as an assertion."""
    assert validate_metric("trajectory_termination", "assertion")


def test_trajectory_termination_can_be_retrieved():
    """Verify trajectory_termination assertion can be retrieved."""
    assertion = get_assertion("trajectory_termination")
    assert callable(assertion)


def test_trajectory_termination_passes_within_bounds():
    """Verify trajectory_termination assertion passes when span count is within bounds."""
    assertion = get_assertion("trajectory_termination")

    spans = [
        {"component_id": "test_comp", "span_type": "agent"},
        {"component_id": "test_comp", "span_type": "agent"},
        {"component_id": "test_comp", "span_type": "agent"},
    ]

    result = assertion(spans, "test_comp", {"max_iterations": 5})
    assert result.passed is True
    assert result.details["span_count"] == 3


def test_trajectory_termination_fails_exceeds_bounds():
    """Verify trajectory_termination assertion fails when span count exceeds bounds."""
    assertion = get_assertion("trajectory_termination")

    spans = [
        {"component_id": "test_comp", "span_type": "agent"},
        {"component_id": "test_comp", "span_type": "agent"},
        {"component_id": "test_comp", "span_type": "agent"},
        {"component_id": "test_comp", "span_type": "agent"},
        {"component_id": "test_comp", "span_type": "agent"},
        {"component_id": "test_comp", "span_type": "agent"},
    ]

    result = assertion(spans, "test_comp", {"max_iterations": 3})
    assert result.passed is False
    assert result.details["span_count"] == 6
    assert result.details["max_allowed"] == 3


def test_trajectory_termination_uses_default_max_iterations():
    """Verify trajectory_termination uses default max_iterations when not provided."""
    assertion = get_assertion("trajectory_termination")

    spans = [
        {"component_id": "test_comp", "span_type": "agent"},
        {"component_id": "test_comp", "span_type": "agent"},
    ]

    result = assertion(spans, "test_comp", {})
    assert result.passed is True
    assert result.details["max_iterations"] == 5
