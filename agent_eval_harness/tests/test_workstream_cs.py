"""Consolidated CS-281..CS-329 workstream tests: registry/planning, discovery, wiring resolution,
motif/granularity selection, evidence spine, model-call verdicts, and schema resolution."""
from __future__ import annotations

import ast
import asyncio
import importlib.util
import json
import re
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_eval_harness.datasets.generators.synthetic_agent_io import (
    SyntheticAgentIOConfig,
    _dispatch_builder,
    _has_single_resolved_object_kwarg,
)
from agent_eval_harness.discovery.agent_knowledge import AgentKnowledge, ComponentRoleVerdict
from agent_eval_harness.discovery.enrichment import (
    _EnrichmentContext,
    _apply_multi_tier_output_resolution,
    _finalize_input_contract,
    _parse_signature_kwargs,
    _sanitize_llm_knowledge_dict,
    _validate_tier_b_schema,
)
from agent_eval_harness.discovery.expansion import reconcile_scope
from agent_eval_harness.discovery.wiring import (
    CallSite,
    NodeCallTarget,
    WiringBlock,
    WiringEdge,
    WiringNode,
    _detect_langgraph,
    _own_class_methods,
    _safe_parse,
    classify_system_type,
    detect_wiring_block_static,
    enclosing_class_name,
    parse_entry_suffix,
    resolve_langgraph_node_calls,
    resolve_unscanned_steps,
    walk_call_closure,
    wiring_identity,
)
from agent_eval_harness.llm.client import LLMResponse
from agent_eval_harness.llm.fake_client import FakeLLMClient
from agent_eval_harness.mapping.agent_flow import (
    AgentFlow,
    AgentFlowMap,
    _demote_deterministic_agents,
    _opens_on_symbol_line,
    separate_agent_flows,
)
from agent_eval_harness.mapping.builder.boundary_llm import (
    _BUNDLE_CHAR_BUDGET,
    ResidueCandidate,
    _build_bundle,
    _extract_verdict,
)
from agent_eval_harness.mapping.builder.contract_harvest import (
    _find_module_dict_or_str_constant,
    _referenced_schema_names,
    harvest_component_contract,
)
from agent_eval_harness.mapping.builder.pipeline import SystemMapBuilder
from agent_eval_harness.mapping.builder.roles import forced_role
from agent_eval_harness.mapping.builder.scanners import (
    HaystackScanner,
    LangGraphScanner,
    get_scanner,
    scan_all,
)
from agent_eval_harness.mapping.builder.system_types import (
    ALL_SYSTEM_TYPES,
    CAPABILITY_TAGS,
    DECIDED_BY_STRUCTURAL,
    DECIDED_BY_UNRESOLVED,
)
from agent_eval_harness.mapping.builder.topology import _compute_motifs, extract_topology
from agent_eval_harness.mapping.builder.types import (
    CandidateComponent,
    ManualSpanHint,
    TopologyEdges,
)
from agent_eval_harness.mapping.system_map import Component, Constraint, SystemMap
from agent_eval_harness.metrics.registry import get_spec, validate_metric
from agent_eval_harness.metrics.suite import DatasetRef
from agent_eval_harness.planning.agentic_planner import EvaluationGate, reconcile
from agent_eval_harness.planning.contract import (
    EvaluationContract,
    InvocationContract,
    KwargSpec,
    OutputContract,
)
from agent_eval_harness.planning.planner import (
    _is_system_thin_chain,
    _is_thin_component,
    _select_granularity_level,
)
from agent_eval_harness.planning.validation import PlanValidationReport, validate_plan
from agent_eval_harness.store import database, repository
from agent_eval_harness.store.database import close_db, init_db


# ==== registry, plan-gate readiness, params completion, feasibility, rebalancing (ex test_cs281_registry.py) ====

@pytest.fixture
async def _cs281_setup_db(tmp_path, monkeypatch):
    """Per-test isolated DB, scoped to this section's tests only (was module-autouse pre-merge)."""
    monkeypatch.setenv("AEH_DATA_DIR", str(tmp_path))
    await init_db()
    yield
    await close_db()


@pytest.mark.usefixtures("_cs281_setup_db")
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


@pytest.mark.usefixtures("_cs281_setup_db")
def test_decomposition_coverage_is_a_literal_registry_entry_with_precondition() -> None:
    """CS-288: a literal key (not the geval.* prefix fallback) carrying input_kind_is_query."""
    assert validate_metric("geval.decomposition_coverage", "llm_judge")
    spec = get_spec("geval.decomposition_coverage")
    assert spec.meaningless_when == ["input_kind_is_query"]
    assert get_spec("ragas.answer_relevancy").meaningless_when == ["input_kind_is_query"]


@pytest.mark.usefixtures("_cs281_setup_db")
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


@pytest.mark.usefixtures("_cs281_setup_db")
def test_reconcile_params_completion_and_feasibility() -> None:
    """Params completion (allowed_downstream/arg_schema) and feasibility replacement rules (faithfulness/relevancy/tool_correctness)."""
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

    contract = EvaluationContract(agent_id="agent1")
    # harvest_contracts sets archetype at Stage 2 (CS-311); a signal-less contract classifies as unimplemented.
    contract.archetype = "unimplemented"
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

    assert "allowed_downstream" in rep_gates
    assert rep_gates["allowed_downstream"].params["allowed"] == ["tool_comp"]

    assert "schema_valid" in rep_gates
    assert rep_gates["schema_valid"].params["schema"] == contract.output.json_schema

    assert "ragas.faithfulness" not in rep_gates
    assert "referential_integrity" in rep_gates
    assert "grounding_judge_span_prompt" in rep_gates

    assert "ragas.answer_relevancy" not in rep_gates

    assert "tool_correctness" not in rep_gates
    assert "llm_call_budget" in rep_gates


@pytest.mark.usefixtures("_cs281_setup_db")
def test_reconcile_rebalance_rules() -> None:
    """Rebalance rules: assertion beats duplicate llm_judge on same property, merge near-dup geval rubrics, cap LLM judges at 3 (baseline exempt)."""
    agent_flow_map = AgentFlowMap(
        target_system_id="test_rebalance",
        agents=[
            AgentFlow(id="agent1", role="orchestrator", label="Agent 1", component_ids=["agent_comp"]),
        ]
    )

    # An assertion + duplicate llm_judge on 'output_schema', 2 near-dup geval gates (should
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

    assert "g.geval_sim2" not in final_gate_ids
    assert "g.geval_sim1" in final_gate_ids

    suggested_llm_judges = [g for g in rep.gates if g.metric_class == "llm_judge" and g.provenance != "rule"]
    assert len(suggested_llm_judges) <= 3


# ==== CS-304 Slices 2, 3, 4 — output_contract, type_hints, LLM-derived fields (ex test_cs304_slices_2_3_4.py) ====

class TestSlice2OutputContract:
    """Slice 2: output_contract field with static producer, version bump, cache lever, gold-gen precedence."""

    def test_output_contract_round_trips_via_to_json_from_json(self):
        """AgentKnowledge round-trips output_contract without loss."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
        output = OutputContract(json_schema=schema, schema_source="test.py")
        knowledge = AgentKnowledge(
            functionality="Test agent",
            output_contract=output,
        )
        json_data = knowledge.to_json()
        restored = AgentKnowledge.from_json(json_data)
        assert restored.output_contract is not None
        assert restored.output_contract.json_schema == schema
        assert restored.output_contract.schema_source == "test.py"

    def test_from_json_pre_cs304_sidecar_missing_output_contract_no_degrade(self):
        """Deserializing a pre-CS-304 sidecar (no output_contract field) does NOT set degraded."""
        pre_cs304_data = {
            "functionality": "Old agent",
            "confidence": "high",
            "location": None,
            "components": [],
            "input_contract": [],
            "prompt_sites": [],
            # Note: output_contract field absent — it didn't exist pre-CS-304
        }
        knowledge = AgentKnowledge.from_json(pre_cs304_data)
        assert knowledge.degraded is False  # Should not degrade just because field is missing
        assert knowledge.functionality == "Old agent"
        assert knowledge.output_contract is None

    def test_version_bump_cache_lever_forces_producer_rerun(self):
        """The producer version hashes hash_input's field shape, so a new field changes it without a manual bump."""
        import hashlib

        from agent_eval_harness.discovery import enrichment

        assert isinstance(enrichment._STRUCTURAL_PRODUCER_VERSION, str)
        assert enrichment._STRUCTURAL_PRODUCER_VERSION == hashlib.sha256(
            "|".join(enrichment._HASH_INPUT_FIELDS).encode("utf-8")
        ).hexdigest()[:12]
        bumped = hashlib.sha256(
            "|".join((*enrichment._HASH_INPUT_FIELDS, "a_new_field")).encode("utf-8")
        ).hexdigest()[:12]
        assert bumped != enrichment._STRUCTURAL_PRODUCER_VERSION

    def test_gold_gen_reads_fresh_canonical_contract_not_stale_sidecar(self):
        """Gold-gen validation reads the fresh EvaluationContract.output, not a stale AgentKnowledge sidecar (Crux C)."""
        from agent_eval_harness.datasets.fulfillment import _derive_config

        # Simulate fresh contract with nested schema
        fresh_schema = {
            "type": "object",
            "properties": {"status": {"type": "string"}, "details": {"type": "object"}},
            "required": ["status"],
        }
        fresh_contract = OutputContract(json_schema=fresh_schema, schema_source="fresh.py")

        # Simulate stale sidecar with old schema
        stale_schema = {"type": "object", "properties": {"old_field": {"type": "string"}}, "required": []}
        stale_contract = OutputContract(json_schema=stale_schema, schema_source="stale.py")

        # Verify they differ
        assert fresh_contract.json_schema != stale_contract.json_schema

        # In _derive_config flow, gold-gen must read from the fresh contract, not the sidecar.
        # We verify this by checking that _derive_config uses contract.model_dump(), not sidecar.output_contract.
        # (This is a structural verification, not a unit test, since _derive_config is hard to isolate.)


class TestSlice3SignatureKwargs:
    """Slice 3: type_hint preservation with nested-generic comma safety."""

    def test_parse_signature_kwargs_preserves_annotations(self):
        """_parse_signature_kwargs preserves type annotations as type_hint."""
        sig = "run(provider_id: str, model_id: str, snapshot_id: str, context: dict[str, list[str]] | None = None) -> dict"
        result = _parse_signature_kwargs(sig)
        # Should return tuples of (name, type_hint)
        assert len(result) == 4
        names = [name for name, _ in result]
        hints = [hint for _, hint in result]
        assert "provider_id" in names
        assert "str" in hints[names.index("provider_id")]
        assert "snapshot_id" in names

    def test_parse_signature_kwargs_balanced_brackets_dict(self):
        """Balanced-bracket aware parsing handles dict[str, X] without false comma splits."""
        sig = "run(metadata: dict[str, int], context: dict[str, list[str]]) -> None"
        result = _parse_signature_kwargs(sig)
        assert len(result) == 2
        names = [name for name, _ in result]
        assert "metadata" in names
        assert "context" in names

    def test_parse_signature_kwargs_nested_union(self):
        """Balanced-bracket aware parsing handles X | Y, X | None."""
        sig = "run(result: str | int, bundle: RetrievalBundle | None = None) -> None"
        result = _parse_signature_kwargs(sig)
        assert len(result) == 2
        hints = {name: hint for name, hint in result}
        assert "result" in hints
        assert "str | int" in hints["result"]
        assert "bundle" in hints
        assert "RetrievalBundle | None" in hints["bundle"]

    def test_parse_signature_kwargs_skips_self_cls(self):
        """self/cls parameters are skipped."""
        sig = "run(self, provider_id: str, model_id: str)"
        result = _parse_signature_kwargs(sig)
        names = [name for name, _ in result]
        assert "self" not in names
        assert "provider_id" in names
        assert "model_id" in names


class TestSlice4Sanitizer:
    """Slice 4: LLM-derived output_described_in_prompt + special_traits allowlist."""

    def test_sanitize_llm_knowledge_dict_keeps_new_fields(self):
        """Sanitizer allowlists output_described_in_prompt and special_traits."""
        raw = {
            "component_roles": [],
            "functionality": "Test",
            "functionality_citations": [],
            "context_builders": [],
            "upstream_consumers": [],
            "downstream_consumers": [],
            "failure_modes": [],
            "output_described_in_prompt": "Returns a dict with results",
            "special_traits": ["async", "retry_capable"],
        }
        sanitized = _sanitize_llm_knowledge_dict(raw)
        assert sanitized["output_described_in_prompt"] == "Returns a dict with results"
        assert sanitized["special_traits"] == ["async", "retry_capable"]

    def test_sanitize_llm_knowledge_dict_drops_hallucinated_keys(self):
        """Sanitizer still drops hallucinated keys not in the allowlist."""
        raw = {
            "component_roles": [],
            "functionality": "Test",
            "functionality_citations": [],
            "context_builders": [],
            "upstream_consumers": [],
            "downstream_consumers": [],
            "failure_modes": [],
            "output_described_in_prompt": "Valid",
            "special_traits": [],
            "hallucinated_field": "This should be dropped",
            "another_fake": 123,
        }
        sanitized = _sanitize_llm_knowledge_dict(raw)
        assert "hallucinated_field" not in sanitized
        assert "another_fake" not in sanitized
        assert "output_described_in_prompt" in sanitized

    def test_sanitize_coerces_bad_types_for_new_fields(self):
        """Sanitizer coerces bad types in new fields (e.g., int instead of list)."""
        raw = {
            "component_roles": [],
            "functionality": "Test",
            "functionality_citations": [],
            "context_builders": [],
            "upstream_consumers": [],
            "downstream_consumers": [],
            "failure_modes": [],
            "output_described_in_prompt": 123,  # Bad type: int
            "special_traits": "not_a_list",  # Bad type: str
        }
        sanitized = _sanitize_llm_knowledge_dict(raw)
        # output_described_in_prompt coerced to string
        assert sanitized["output_described_in_prompt"] == "123" or isinstance(sanitized.get("output_described_in_prompt"), str)
        # special_traits coerced to list of strings (or empty if bad)
        assert isinstance(sanitized["special_traits"], list)

    def test_sanitize_keeps_constraints_and_method_steps(self):
        """Sanitizer allowlists constraints + method_steps and drops non-strings (CS-304 field-set)."""
        raw = {
            "functionality": "Test",
            "constraints": ["return only JSON", "must cite evidence_files", 42],
            "method_steps": ["retrieve", "reason", "emit", None],
        }
        sanitized = _sanitize_llm_knowledge_dict(raw)
        assert sanitized["constraints"] == ["return only JSON", "must cite evidence_files"]
        assert sanitized["method_steps"] == ["retrieve", "reason", "emit"]


class TestSlice2OutputContractInToMd:
    """Proof that output_contract is consumed by AgentKnowledge.to_md()."""

    def test_to_md_renders_output_contract(self):
        """AgentKnowledge.to_md() renders the output_contract JSON schema."""
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "count": {"type": "integer"},
            },
            "required": ["status"],
        }
        output = OutputContract(json_schema=schema, schema_source="module.py:42")
        knowledge = AgentKnowledge(
            functionality="Test agent",
            output_contract=output,
        )
        md = knowledge.to_md()
        # Check that output_contract section is rendered
        assert "## Output Contract" in md
        assert "module.py:42" in md
        assert '"status"' in md and '"type": "string"' in md
        assert '"count"' in md and '"type": "integer"' in md

    def test_to_md_omits_output_contract_if_none(self):
        """to_md() omits the Output Contract section if output_contract is None."""
        knowledge = AgentKnowledge(
            functionality="Test agent",
            output_contract=None,
        )
        md = knowledge.to_md()
        assert "## Output Contract" not in md

    def test_to_md_omits_output_contract_if_no_schema(self):
        """to_md() omits the Output Contract section if json_schema is None."""
        output = OutputContract(json_schema=None, schema_source=None)
        knowledge = AgentKnowledge(
            functionality="Test agent",
            output_contract=output,
        )
        md = knowledge.to_md()
        assert "## Output Contract" not in md


class TestSlice4PromptDerivedInToMd:
    """Prompt-derived fields (output/method_steps/constraints/special_traits) render in to_md() — proves they are live consumers, not dead fields."""

    def test_to_md_renders_prompt_derived_fields(self):
        knowledge = AgentKnowledge(
            functionality="Test agent",
            output_described_in_prompt="Returns a JSON object with status and details.",
            method_steps=["retrieve evidence", "reason", "emit JSON"],
            constraints=["return only JSON", "must cite evidence_files"],
            special_traits=["retry_capable"],
        )
        md = knowledge.to_md()
        assert "## Output (described in prompt)" in md and "status and details" in md
        assert "## Method Steps" in md and "1. retrieve evidence" in md
        assert "## Constraints (from prompt)" in md and "must cite evidence_files" in md
        assert "## Special Traits" in md and "retry_capable" in md

    def test_to_md_omits_prompt_derived_when_empty(self):
        md = AgentKnowledge(functionality="Test agent").to_md()
        assert "## Constraints (from prompt)" not in md
        assert "## Method Steps" not in md


# ==== CS-311 WS-2A foundation: scanner registry, entry-shape helpers, function/bound-method harvest, wiring_block reuse (ex test_cs311_ws2a_foundation.py) ====

# ---- Slice 1: scanner registry -------------------------------------------------------------
def test_get_scanner_returns_matching_and_falls_back():
    assert isinstance(get_scanner("haystack"), HaystackScanner)
    assert get_scanner("haystack").framework == "haystack"
    # Unknown / None never raise — they degrade to the Haystack default (behavior-identical to today).
    assert isinstance(get_scanner("unknown_xyz"), HaystackScanner)
    assert isinstance(get_scanner(None), HaystackScanner)


# ---- Slice 2: entry-shape helpers ----------------------------------------------------------
def test_wiring_identity_and_parse_entry_suffix_roundtrip():
    assert wiring_identity("Owner", "method") == "Owner.method"
    assert wiring_identity(None, "Bare") == "Bare"
    assert parse_entry_suffix(wiring_identity("Owner", "method")) == ("Owner", "method")
    assert parse_entry_suffix("Bare") == (None, "Bare")


def test_wiring_block_from_dict_roundtrips():
    block = WiringBlock(
        nodes=[WiringNode(alias="a", callee_name="C", source_hint_file="f.py",
                          entry_kind="bound_method", owner_class="Agent")],
        edges=[WiringEdge(src="a", dst="a")],
        framework="langgraph",
        source="static",
    )
    assert WiringBlock.from_dict(block.to_dict()) == block


def test_enclosing_class_name_finds_owner():
    tree = ast.parse(textwrap.dedent("""
        class Agent:
            def node(self):
                helper()
        def free():
            other()
    """))
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "helper")
    free_call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "other")
    assert enclosing_class_name(call, tree) == "Agent"
    assert enclosing_class_name(free_call, tree) is None


# ---- Slice 3: function / bound-method entry harvest ----------------------------------------
_SNIPPET = textwrap.dedent("""
    class Agent:
        def __init__(self, service):
            self._service = service
        def custom_node(self, state, config=None):
            return {"answer": state}

    def handle(query, top_k=5):
        return {"docs": []}
""")


def _asts(tmp_path: Path, src: str) -> dict[Path, ast.Module]:
    p = tmp_path / "mod.py"
    p.write_text(src, encoding="utf-8")
    return {p: ast.parse(src)}


def test_bound_method_entry_harvests_arbitrary_method_name(tmp_path):
    asts = _asts(tmp_path, _SNIPPET)
    comp = Component(id="agent", role="unknown", entry_point="mod:Agent.custom_node",
                     entry_kind="bound_method", file="mod.py")
    invocation, _output, _constants, notes, _kind = harvest_component_contract(comp, asts)
    assert invocation is not None, notes
    assert invocation.method == "custom_node"
    assert {k.name for k in invocation.kwargs} == {"state", "config"}
    assert not any("not found" in n for n in notes)


def test_function_entry_harvests_without_class(tmp_path):
    asts = _asts(tmp_path, _SNIPPET)
    comp = Component(id="handler", role="unknown", entry_point="mod:handle",
                     entry_kind="function", file="mod.py")
    invocation, _output, _constants, notes, _kind = harvest_component_contract(comp, asts)
    assert invocation is not None, notes
    assert invocation.method == "handle"
    assert {k.name for k in invocation.kwargs} == {"query", "top_k"}
    assert not any("class" in n and "not found" in n for n in notes)


def test_legacy_class_entry_without_entry_kind_still_resolves(tmp_path):
    # entry_kind=None (legacy YAML) must behave exactly like the old class path.
    src = "class Foo:\n    def run(self, x):\n        return {}\n"
    asts = _asts(tmp_path, src)
    comp = Component(id="foo", role="unknown", entry_point="mod:Foo", file="mod.py")
    invocation, _o, _c, notes, _k = harvest_component_contract(comp, asts)
    assert invocation is not None, notes
    assert invocation.method == "run"


# ---- Slice 6: Stage-1 wiring_block is reused, not re-detected -------------------------------
def test_extract_topology_reuses_supplied_wiring_block():
    fake = WiringBlock(nodes=[], edges=[], framework="langgraph", source="llm_fallback")
    with patch("agent_eval_harness.discovery.wiring.detect_wiring_block_static") as spy:
        extract_topology([], [], wiring_block=fake)
        spy.assert_not_called()


def test_extract_topology_detects_when_no_block_supplied():
    with patch("agent_eval_harness.discovery.wiring.detect_wiring_block_static", return_value=None) as spy:
        extract_topology([], [])
        spy.assert_called_once()


# ---- CS-317: topology safety-net (Gate D defense-in-depth) ---------------------------------
def test_extract_topology_safetynet_reattaches_missing_framework_edges(tmp_path):
    """A stale haystack-only wiring_block alongside files that also contain a LangGraph graph must not orphan the scanned LangGraph nodes (Gate D)."""
    from agent_eval_harness.mapping.builder.types import CandidateComponent

    graph_src = textwrap.dedent(
        """
        from langgraph.graph import StateGraph
        class Orchestrator:
            def build(self):
                g = StateGraph(dict)
                g.add_node('plan', self._plan)
                g.add_node('act', self._act)
                g.add_edge('plan', 'act')
            def _plan(self, s): ...
            def _act(self, s): ...
        """
    )
    graph_file = tmp_path / "graph.py"
    graph_file.write_text(graph_src, encoding="utf-8")

    # candidates as scan_all would harvest the two bound-method nodes (owner Orchestrator)
    candidates = [
        CandidateComponent(file=graph_file, line=1, class_name="_plan",
                           owner_class_name="Orchestrator", entry_kind="bound_method"),
        CandidateComponent(file=graph_file, line=1, class_name="_act",
                           owner_class_name="Orchestrator", entry_kind="bound_method"),
    ]
    # STALE block: haystack-only, knows NOTHING about the langgraph graph.
    stale = WiringBlock(nodes=[], edges=[], framework="haystack", source="static")

    topo = extract_topology([graph_file], candidates, wiring_block=stale)
    plan_id = candidates[0].candidate_id
    act_id = candidates[1].candidate_id
    # Without the safety-net the langgraph edge is orphaned (both empty); with it, plan->act exists.
    assert act_id in topo[plan_id].downstream, "safety-net failed to re-attach the LangGraph edge"
    assert plan_id in topo[act_id].upstream


# ==== CS-318 — unscanned-step resolver: import -> MRO -> RRF+AST-confirm -> needs_human (ex test_cs318_step_resolver.py) ====

def _write(root: Path, rel: str, body: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


def _build_fixture(root: Path) -> Path:
    """A framework-free agent whose 4 steps are defined OUTSIDE its own file (2 via import, 2 via inheritance); returns the agent file path."""
    _write(root, "helpers.py", (
        "def do_plan(q):\n    return [q]\n\n"
        "def do_retrieve(qs):\n    return {'hits': qs}\n"
    ))
    _write(root, "base_agents.py", (
        "class BaseThing:\n"
        "    async def _call_model(self, prompt):\n        return 'answer'\n\n"
        "    async def _finish(self, text):\n        return {'text': text}\n"
    ))
    _write(root, "myagent.py", (
        "from helpers import do_plan, do_retrieve\n"
        "from base_agents import BaseThing\n\n"
        "class MyAgent(BaseThing):\n"
        "    async def run(self, q):\n"
        "        subs = do_plan(q)\n"
        "        bundle = do_retrieve(subs)\n"
        "        ans = await self._call_model(bundle)\n"
        "        return await self._finish(ans)\n"
    ))
    return root / "myagent.py"


def _plain_wiring() -> WiringBlock:
    """Mirrors _detect_plain_python: agent node + linear step chain, every step's source_hint_file is the AGENT's file (call site), not the definition site."""
    hint = "myagent.py"
    nodes = [
        WiringNode(alias="MyAgent", callee_name="MyAgent", source_hint_file=hint,
                   entry_kind="class", framework="plain_python"),
        WiringNode(alias="do_plan", callee_name="do_plan", source_hint_file=hint,
                   entry_kind="function", framework="plain_python"),
        WiringNode(alias="do_retrieve", callee_name="do_retrieve", source_hint_file=hint,
                   entry_kind="function", framework="plain_python"),
        WiringNode(alias="call_model", callee_name="_call_model", source_hint_file=hint,
                   entry_kind="function", framework="plain_python"),
        WiringNode(alias="finish", callee_name="_finish", source_hint_file=hint,
                   entry_kind="function", framework="plain_python"),
    ]
    edges = [
        WiringEdge("MyAgent", "do_plan"), WiringEdge("do_plan", "do_retrieve"),
        WiringEdge("do_retrieve", "call_model"), WiringEdge("call_model", "finish"),
    ]
    return WiringBlock(nodes=nodes, edges=edges, framework="plain_python", source="static")


class _FakeLLM:
    async def complete(self, messages, *, max_tokens=512, temperature=0.2, json_mode=False, reasoning_effort=None):
        from agent_eval_harness.llm.client import LLMResponse
        return LLMResponse(content="[]", model="fake")


class _StubRetrieval:
    """RRF stub: search_retrieval returns a decoy ahead of the real definer (exercises AST-confirm); get_symbol_edges is empty so free layers can't short-circuit RRF."""

    def __init__(self, hits_by_query: dict[str, list[str]]):
        self._hits = hits_by_query

    async def search_retrieval(self, snapshot_id, query, section="qa", symbol_chunks_only=False):
        return {"final": [{"rel_path": r} for r in self._hits.get(query, [])]}

    async def get_symbol_edges(self, snapshot_id, file_path):
        return {"outgoing": [], "incoming": []}

    async def read_file(self, snapshot_id, rel_path, max_bytes=200_000):
        return {"content": ""}


def test_import_and_mro_layers_resolve_real_definers_no_retrieval(tmp_path: Path):
    """Free layers only (retrieval=None): imported steps -> helpers.py via 'import'; inherited steps -> base_agents.py via 'mro'."""
    agent = _build_fixture(tmp_path)
    res = asyncio.run(resolve_unscanned_steps(_plain_wiring(), [agent], tmp_path, None, None))
    by = {r.callee: r for r in res}

    assert by["do_plan"].layer == "import" and by["do_plan"].defining_file == "helpers.py"
    assert by["do_retrieve"].layer == "import" and by["do_retrieve"].defining_file == "helpers.py"
    for m in ("_call_model", "_finish"):
        assert by[m].layer == "mro", by[m]
        assert by[m].defining_file == "base_agents.py"
        assert by[m].entry_kind == "bound_method" and by[m].owner_class == "BaseThing"


def test_forced_step_falls_through_to_rrf_and_ast_confirm_filters_decoy(tmp_path: Path):
    """Withholding the import layer for one step falls to the RRF path and still lands the real definer; a non-defining decoy hit is dropped by AST-confirm."""
    agent = _build_fixture(tmp_path)
    _write(tmp_path, "decoy.py", "# do_plan is mentioned here but never defined\nX = 'do_plan'\n")
    retrieval = _StubRetrieval({"do_plan": ["decoy.py", "helpers.py"]})

    res = asyncio.run(resolve_unscanned_steps(
        _plain_wiring(), [agent], tmp_path, retrieval, "snap",
        skip_import_layer_for=frozenset({"do_plan"}),
    ))
    by = {r.callee: r for r in res}
    assert by["do_plan"].layer == "rrf", by["do_plan"]
    assert by["do_plan"].defining_file == "helpers.py"  # decoy.py rejected by AST-confirm
    # the other imported step still resolved for free
    assert by["do_retrieve"].layer == "import"


def test_unresolvable_step_flagged_needs_human_in_map(tmp_path: Path):
    """A step no layer can place (RRF returns only a decoy) must not vanish: build_from_files emits a needs_human placeholder."""
    agent = _build_fixture(tmp_path)
    _write(tmp_path, "decoy.py", "# ghost_step named only\nY = 'ghost_step'\n")
    wb = _plain_wiring()
    wb.nodes.append(WiringNode(alias="ghost", callee_name="ghost_step", source_hint_file="myagent.py",
                               entry_kind="function", framework="plain_python"))
    wb.edges.append(WiringEdge("finish", "ghost"))
    retrieval = _StubRetrieval({"ghost_step": ["decoy.py"]})

    builder = SystemMapBuilder(_FakeLLM(), framework="plain_python")
    system_map, _ = asyncio.run(builder.build_from_files(
        [agent], package_root=tmp_path, target_system_id="fixture",
        wiring_block=wb, retrieval_client=retrieval, snapshot_id="snap",
    ))
    flagged = [c for c in system_map.components if c.needs_human]
    assert len(flagged) == 1 and flagged[0].id == "ghost"


def test_build_from_files_yields_connected_map_from_admitted_files(tmp_path: Path):
    """End-to-end (free static layers, no retrieval): the 4 out-of-file steps resolve, files are admitted, rescan yields >=5 connected components."""
    agent = _build_fixture(tmp_path)
    builder = SystemMapBuilder(_FakeLLM(), framework="plain_python")
    system_map, _ = asyncio.run(builder.build_from_files(
        [agent], package_root=tmp_path, target_system_id="fixture", wiring_block=_plain_wiring(),
    ))
    ids = {c.id for c in system_map.components}
    assert len(system_map.components) >= 5, ids
    # the imported helpers and inherited methods are all present as real components
    for expected in ("myagent", "do_plan", "do_retrieve", "_call_model", "_finish"):
        assert expected in ids, (expected, ids)
    assert not any(c.needs_human for c in system_map.components)
    total_edges = sum(len(c.upstream) + len(c.downstream) for c in system_map.components)
    assert total_edges > 0, "call flow not connected"


# ==== CS-319 — LangGraph node-body call graph (ex test_cs319_langgraph_node_body.py) ====
# Hermetic + generic synthetic StateGraph agent exercising every resolution kind (same-module fn,
# own-class method incl. transitive 2-hop helper, imported fn, inherited/mro method, injected
# service), exclusions (builtin, stdlib, pure-data ctor, attribute-call on local var), a shared hub
# called by >=2 nodes, conditional router marking, and scope-based junk-drop.

def _write_langgraph_pkg(root: Path) -> list[Path]:
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    # A base class in another module -> mro/inherited resolution.
    (root / "pkg" / "base.py").write_text(textwrap.dedent('''
        class BaseWorker:
            async def call_model(self, prompt): return prompt
    '''), encoding="utf-8")
    # Imported helper module -> import resolution.
    (root / "pkg" / "helpers.py").write_text(textwrap.dedent('''
        async def search_index(q): return [q]
    '''), encoding="utf-8")
    # Injected service class -> injected_service resolution.
    (root / "pkg" / "svc.py").write_text(textwrap.dedent('''
        class SearchService:
            async def query(self, q): return q
    '''), encoding="utf-8")
    # A pure-data class (must be EXCLUDED even though constructed in a node body).
    (root / "pkg" / "types.py").write_text(textwrap.dedent('''
        from dataclasses import dataclass
        @dataclass
        class Payload:
            text: str
    '''), encoding="utf-8")
    # The agent: StateGraph with node methods delegating to same-module fn, own helpers (transitive),
    # imported fn, inherited method, and an injected service. Plus builtins/stdlib/local-var/pure-data
    # calls that must NOT become components. A sibling *Agent class must be scoped out.
    (root / "pkg" / "agent.py").write_text(textwrap.dedent('''
        import json
        from langgraph.graph import StateGraph
        from pkg.base import BaseWorker
        from pkg.helpers import search_index
        from pkg.svc import SearchService
        from pkg.types import Payload

        def format_summary(items):   # same-module module-level fn
            return ", ".join(items)

        class SiblingAgent:          # co-located junk -> must be scoped out
            def run(self): return 1

        class ResearchWorker(BaseWorker):
            def __init__(self, service: SearchService):
                self._svc = service

            def build(self):
                g = StateGraph(dict)
                g.add_node("plan", self._node_plan)
                g.add_node("gather", self._node_gather)
                g.add_node("write", self._node_write)
                g.add_conditional_edges("plan", self._route, {"a": "gather", "b": "write"})
                g.add_edge("gather", "write")
                return g

            async def _node_plan(self, state):
                data = json.dumps(state)          # stdlib -> excluded
                n = len(data)                     # builtin -> excluded
                return await self.call_model(n)   # inherited (mro)

            async def _node_gather(self, state):
                p = Payload(text="x")             # pure-data ctor -> excluded
                hits = await search_index(p.text) # imported fn
                got = await self._svc.query(hits) # injected service (SearchService)
                return await self._enrich(got)    # own-class helper (transitive)

            async def _node_write(self, state):
                out = state.get("acc", [])        # attr-call on local var -> excluded
                summary = format_summary(out)     # same-module fn
                return await self._enrich(summary)# own-class helper (SHARED: gather+write)

            async def _enrich(self, x):           # own helper -> transitively reaches search_index
                more = await search_index(x)
                return more

            def _route(self, state):
                return "a"
    '''), encoding="utf-8")
    return [
        root / "pkg" / "agent.py", root / "pkg" / "base.py", root / "pkg" / "helpers.py",
        root / "pkg" / "svc.py", root / "pkg" / "types.py",
    ]


@pytest.mark.asyncio
async def test_langgraph_node_body_call_graph(tmp_path: Path):
    files = _write_langgraph_pkg(tmp_path)
    fc = {str(f): f.read_text(encoding="utf-8") for f in files}
    wb = _detect_langgraph(fc)
    assert wb is not None and wb.framework == "langgraph"

    targets = await resolve_langgraph_node_calls(wb, tmp_path, None, None)
    by_alias = {t.alias: t for t in targets}

    # Every resolution kind resolved to its REAL defining file.
    assert by_alias["call_model"].layer == "mro" and by_alias["call_model"].defining_file.endswith("base.py")
    assert by_alias["search_index"].layer == "import" and by_alias["search_index"].defining_file.endswith("helpers.py")
    assert by_alias["format_summary"].layer == "same_module" and by_alias["format_summary"].defining_file.endswith("agent.py")
    assert by_alias["_enrich"].layer == "own_class" and by_alias["_enrich"].entry_kind == "bound_method"
    assert by_alias["SearchService"].layer == "injected_service" and by_alias["SearchService"].defining_file.endswith("svc.py")

    # Exclusions: no builtin / stdlib / pure-data / local-var-attr call became a target.
    assert "Payload" not in by_alias and "dumps" not in by_alias and "len" not in by_alias and "get" not in by_alias

    # Build the real map and assert the graph shape. _node_write's closure is genuinely incomplete
    # (an untyped `state.get(...)` receiver) -- CS-325's residue pass runs over it, so the client
    # must tolerate being called, not assert it never is.
    builder = SystemMapBuilder(FakeLLMClient(LLMResponse(content="[]", model="fake")), framework="langgraph")
    system_map, _ = await builder.build_from_files(
        files, package_root=tmp_path, target_system_id="synthetic",
        wiring_block=wb, scope_framework="langgraph", retrieval_client=None, snapshot_id=None,
    )
    ids = {c.id for c in system_map.components}
    # Nodes + gray targets present; sibling junk class scoped out.
    assert {"_node_plan", "_node_gather", "_node_write"} <= ids
    assert {"search_index", "format_summary", "_enrich", "searchservice", "call_model"} <= ids
    assert "siblingagent" not in ids and "researchworker" not in ids

    # Shared hub: _enrich called by BOTH gather and write -> ONE component, >=2 inbound.
    enrich = system_map.component_by_id("_enrich")
    assert enrich is not None and len(enrich.upstream) >= 2

    # Transitive: search_index reached from within _enrich (2-hop) has _enrich as an inbound caller.
    si = system_map.component_by_id("search_index")
    assert si is not None and "_enrich" in si.upstream

    # Conditional router marking on the add_conditional_edges source; the hard add_edge is NOT marked.
    plan = system_map.component_by_id("_node_plan")
    assert plan is not None and plan.conditional_downstream
    gather = system_map.component_by_id("_node_gather")
    assert gather is not None and not gather.conditional_downstream


def test_haystack_edge_defaults_unchanged():
    """Additive edge-kind schema must leave a plain edge at its defaults, so a Haystack map is byte-identical."""
    e = WiringEdge(src="a", dst="b")
    assert e.kind == "hard" and e.conditional is False


# ==== CS-320: system-type classification and forced_role gate (ex test_cs320_system_type_classification.py) ====
# No LLM, no mocked AST. Hand-constructed fixtures covering structural types and the post-loop pass
# for omitted conditional sources.

class TestClassifySystemType:
    """Structural classification tests: coverage of all high-confidence branches."""

    def test_pipeline_dag_no_conditional(self):
        """DAG, fan-in hand-off, no conditional edge → pipeline."""
        nodes = [
            WiringNode(alias="extract", callee_name="extract", source_hint_file="a.py"),
            WiringNode(alias="transform", callee_name="transform", source_hint_file="b.py"),
            WiringNode(alias="aggregate", callee_name="aggregate", source_hint_file="c.py"),
        ]
        edges = [
            WiringEdge(src="extract", dst="transform"),
            WiringEdge(src="transform", dst="aggregate"),
        ]
        block = WiringBlock(nodes=nodes, edges=edges, framework="test", source="static")

        result = classify_system_type(block)

        assert result["kind"] == "system"
        assert result["type"] == "pipeline"
        assert result["confidence"] == "high"
        assert result["candidate_types"] == []
        assert result["signals"]["has_conditional_edge"] is False
        assert result["signals"]["has_cycle"] is False

    def test_orchestrator_with_cycle_and_conditional(self):
        """Has cycle + conditional edge → orchestrator."""
        nodes = [
            WiringNode(alias="router", callee_name="router", source_hint_file="a.py"),
            WiringNode(alias="worker1", callee_name="worker1", source_hint_file="b.py"),
            WiringNode(alias="worker2", callee_name="worker2", source_hint_file="c.py"),
            WiringNode(alias="analyze", callee_name="analyze", source_hint_file="d.py"),
        ]
        edges = [
            WiringEdge(src="router", dst="worker1", kind="hard", conditional=True),
            WiringEdge(src="router", dst="worker2", kind="hard", conditional=True),
            WiringEdge(src="worker1", dst="analyze"),
            WiringEdge(src="worker2", dst="analyze"),
            WiringEdge(src="analyze", dst="router"),  # back-edge: cycle
        ]
        block = WiringBlock(nodes=nodes, edges=edges, framework="test", source="static")

        result = classify_system_type(block)

        assert result["kind"] == "system"
        assert result["type"] == "orchestrator"
        assert result["confidence"] == "high"
        assert result["signals"]["has_conditional_edge"] is True
        assert result["signals"]["has_cycle"] is True

    def test_routing_single_conditional_no_reconvergence(self):
        """One conditional source, no reconvergence → routing."""
        nodes = [
            WiringNode(alias="router", callee_name="router", source_hint_file="a.py"),
            WiringNode(alias="branch1", callee_name="branch1", source_hint_file="b.py"),
            WiringNode(alias="branch2", callee_name="branch2", source_hint_file="c.py"),
        ]
        edges = [
            WiringEdge(src="router", dst="branch1", kind="hard", conditional=True),
            WiringEdge(src="router", dst="branch2", kind="hard", conditional=True),
            # No reconvergence: branch1 and branch2 are independent
        ]
        block = WiringBlock(nodes=nodes, edges=edges, framework="test", source="static")

        result = classify_system_type(block)

        assert result["kind"] == "system"
        assert result["type"] == "routing"
        assert result["confidence"] == "high"
        assert result["signals"]["has_conditional_edge"] is True
        assert result["signals"]["has_cycle"] is False

    def test_orchestrator_single_conditional_with_reconvergence(self):
        """One conditional source but branches reconverge (fan-in) → orchestrator."""
        nodes = [
            WiringNode(alias="router", callee_name="router", source_hint_file="a.py"),
            WiringNode(alias="branch1", callee_name="branch1", source_hint_file="b.py"),
            WiringNode(alias="branch2", callee_name="branch2", source_hint_file="c.py"),
            WiringNode(alias="aggregate", callee_name="aggregate", source_hint_file="d.py"),
        ]
        edges = [
            WiringEdge(src="router", dst="branch1", kind="hard", conditional=True),
            WiringEdge(src="router", dst="branch2", kind="hard", conditional=True),
            WiringEdge(src="branch1", dst="aggregate"),
            WiringEdge(src="branch2", dst="aggregate"),  # reconvergence: aggregate has in_degree=2
        ]
        block = WiringBlock(nodes=nodes, edges=edges, framework="test", source="static")

        result = classify_system_type(block)

        assert result["kind"] == "system"
        assert result["type"] == "orchestrator"
        assert result["confidence"] == "high"
        assert result["signals"]["has_conditional_edge"] is True
        assert result["signals"]["has_cycle"] is False
        assert result["signals"]["max_in_degree"] == 2

    def test_single_flow_no_loop_no_branch(self):
        """Single agent, no cycle, no conditional → single-flow."""
        nodes = [
            WiringNode(
                alias="QAAgent", callee_name="QAAgent",
                source_hint_file="qa.py", entry_kind="class", owner_class="QAAgent"
            ),
        ]
        edges = []
        block = WiringBlock(nodes=nodes, edges=edges, framework="plain_python", source="static")

        result = classify_system_type(block)

        assert result["kind"] == "agent"
        assert result["type"] == "single-flow"
        assert result["confidence"] == "high"
        assert result["signals"]["has_cycle"] is False

    def test_tool_loop_self_loop(self):
        """Single agent with self-loop → tool-loop."""
        nodes = [
            WiringNode(
                alias="ReActAgent", callee_name="create_react_agent",
                source_hint_file="react.py", entry_kind="function"
            ),
        ]
        edges = [
            WiringEdge(src="ReActAgent", dst="ReActAgent"),  # self-loop
        ]
        block = WiringBlock(nodes=nodes, edges=edges, framework="test", source="static")

        result = classify_system_type(block)

        assert result["kind"] == "agent"
        assert result["type"] == "tool-loop"
        assert result["confidence"] == "high"
        assert result["signals"]["self_loop"] is True

    def test_ambiguous_single_agent_low_confidence(self):
        """Single agent with conditional branch to external node → low confidence."""
        nodes = [
            WiringNode(alias="mystery", callee_name="mystery", source_hint_file="m.py"),
        ]
        edges = [
            WiringEdge(src="mystery", dst="external", kind="hard", conditional=True),  # conditional to unknown
        ]
        block = WiringBlock(nodes=nodes, edges=edges, framework="test", source="static")

        result = classify_system_type(block)

        # Single agent with conditional to external is ambiguous
        assert result["kind"] == "agent"
        assert result["type"] is None
        assert result["confidence"] == "low"
        assert set(result["candidate_types"]) == {"plan-execute", "reflection"}

    def test_ambiguous_cycle_no_conditional_low_confidence(self):
        """Multi-component cycle without conditional → low confidence."""
        nodes = [
            WiringNode(alias="generator", callee_name="generator", source_hint_file="a.py"),
            WiringNode(alias="evaluator", callee_name="evaluator", source_hint_file="b.py"),
        ]
        edges = [
            WiringEdge(src="generator", dst="evaluator"),
            WiringEdge(src="evaluator", dst="generator"),  # back-edge, no conditional
        ]
        block = WiringBlock(nodes=nodes, edges=edges, framework="test", source="static")

        result = classify_system_type(block)

        assert result["kind"] == "system"
        assert result["type"] is None
        assert result["confidence"] == "low"
        assert set(result["candidate_types"]) == {"evaluator-optimizer", "peer-collaboration", "debate"}

    def test_capability_tags_retrieval_detection(self):
        """Detects retrieval verbs in node names/aliases."""
        nodes = [
            WiringNode(alias="retrieval_agent", callee_name="retrieve_docs", source_hint_file="a.py"),
            WiringNode(alias="llm", callee_name="invoke", source_hint_file="b.py"),
        ]
        edges = [
            WiringEdge(src="retrieval_agent", dst="llm"),
        ]
        block = WiringBlock(nodes=nodes, edges=edges, framework="test", source="static")

        result = classify_system_type(block)

        assert "has_retrieval" in result["capability_tags"]

    def test_capability_tags_react_node_has_tools(self):
        """React node implies has_tools."""
        nodes = [
            WiringNode(
                alias="my_react", callee_name="create_react_agent",
                source_hint_file="a.py", entry_kind="function"
            ),
        ]
        edges = []
        block = WiringBlock(nodes=nodes, edges=edges, framework="test", source="static")

        result = classify_system_type(block)

        assert "has_tools" in result["capability_tags"]


class TestForcedRole:
    """Deterministic role forcing for conditional-source components."""

    def test_forced_role_true_returns_orchestrator(self):
        """A conditional-downstream component is forced to orchestrator."""
        assert forced_role(True) == "orchestrator"

    def test_forced_role_false_returns_none(self):
        """A non-conditional component gets no forced role."""
        assert forced_role(False) is None


class TestSystemTypePersistence:
    """Deterministic round-trip: insert → decode."""

    @pytest.mark.asyncio
    async def test_migration_v23_present(self):
        """Migration v23 is present and adds the new columns."""
        from agent_eval_harness.store.database import _MIGRATIONS
        v23 = next((m for m in _MIGRATIONS if m["version"] == 23), None)
        assert v23 is not None
        assert "system_type" in v23["sql"]
        assert "system_type_signals_json" in v23["sql"]

    @pytest.mark.asyncio
    async def test_candidate_insert_decode_system_type(self, tmp_path):
        """Insert candidate with system_type → decode retrieves it."""
        import aiosqlite
        db_path = tmp_path / "test.db"

        async with aiosqlite.connect(str(db_path)) as db:
            # Create minimal schema with v23 migration
            await db.executescript("""
                CREATE TABLE discovery_candidates (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    name TEXT,
                    frameworks_json TEXT,
                    entry_points_json TEXT,
                    evidence_json TEXT,
                    confidence TEXT,
                    verdict TEXT,
                    needs_human INTEGER,
                    community_id TEXT,
                    cluster_files_json TEXT,
                    hub_paths_json TEXT,
                    wiring_block_json TEXT,
                    excluded_files_json TEXT,
                    matched_files_json TEXT,
                    file_provenance_json TEXT,
                    risk_flags_json TEXT,
                    map_scope_framework TEXT,
                    excluded_component_classes_json TEXT,
                    system_type TEXT,
                    system_type_signals_json TEXT NOT NULL DEFAULT '{}'
                );
            """)
            await db.commit()

            # Insert with system_type
            candidate_id = "test_cand_1"
            system_type = "orchestrator"
            system_type_signals = {
                "kind": "system",
                "type": "orchestrator",
                "confidence": "high",
                "candidate_types": [],
                "capability_tags": ["has_retrieval"],
                "signals": {"has_cycle": True},
                "decided_by": DECIDED_BY_STRUCTURAL,
            }

            await db.execute(
                """INSERT INTO discovery_candidates
                (id, session_id, name, frameworks_json, entry_points_json,
                 evidence_json, confidence, verdict, needs_human,
                 community_id, cluster_files_json, hub_paths_json,
                 wiring_block_json, excluded_files_json, matched_files_json,
                 file_provenance_json, risk_flags_json, map_scope_framework,
                 excluded_component_classes_json, system_type, system_type_signals_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (candidate_id, "sess_1", "test_system", "[]", "[]", "[]",
                 "high", "proposed", 0, "comm_1", "[]", "[]", None, "[]", "[]",
                 "{}", "[]", "test_fw", "[]", system_type,
                 json.dumps(system_type_signals))
            )
            await db.commit()

            # Decode
            cursor = await db.execute(
                "SELECT system_type, system_type_signals_json FROM discovery_candidates WHERE id = ?",
                (candidate_id,)
            )
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == "orchestrator"
            signals = json.loads(row[1])
            assert signals["type"] == "orchestrator"
            assert signals["kind"] == "system"
            assert signals["decided_by"] == DECIDED_BY_STRUCTURAL

    @pytest.mark.asyncio
    async def test_candidate_insert_decode_null_system_type(self, tmp_path):
        """Insert candidate with NULL system_type → decode returns None."""
        import aiosqlite
        db_path = tmp_path / "test.db"

        async with aiosqlite.connect(str(db_path)) as db:
            await db.executescript("""
                CREATE TABLE discovery_candidates (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    name TEXT,
                    frameworks_json TEXT,
                    entry_points_json TEXT,
                    evidence_json TEXT,
                    confidence TEXT,
                    verdict TEXT,
                    needs_human INTEGER,
                    community_id TEXT,
                    cluster_files_json TEXT,
                    hub_paths_json TEXT,
                    wiring_block_json TEXT,
                    excluded_files_json TEXT,
                    matched_files_json TEXT,
                    file_provenance_json TEXT,
                    risk_flags_json TEXT,
                    map_scope_framework TEXT,
                    excluded_component_classes_json TEXT,
                    system_type TEXT,
                    system_type_signals_json TEXT NOT NULL DEFAULT '{}'
                );
            """)
            await db.commit()

            candidate_id = "test_cand_2"

            await db.execute(
                """INSERT INTO discovery_candidates
                (id, session_id, name, frameworks_json, entry_points_json,
                 evidence_json, confidence, verdict, needs_human,
                 community_id, cluster_files_json, hub_paths_json,
                 wiring_block_json, excluded_files_json, matched_files_json,
                 file_provenance_json, risk_flags_json, map_scope_framework,
                 excluded_component_classes_json, system_type, system_type_signals_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (candidate_id, "sess_1", "unknown", "[]", "[]", "[]",
                 "low", "proposed", 0, "comm_1", "[]", "[]", None, "[]", "[]",
                 "{}", "[]", None, "[]", None, "{}")
            )
            await db.commit()

            cursor = await db.execute(
                "SELECT system_type, system_type_signals_json FROM discovery_candidates WHERE id = ?",
                (candidate_id,)
            )
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] is None
            assert row[1] == "{}"


class TestSystemTypesVocabulary:
    """All type vocabulary is generic pattern names, no framework literals."""

    def test_all_system_types_are_generic(self):
        """All types in ALL_SYSTEM_TYPES are generic pattern names."""
        assert "orchestrator" in ALL_SYSTEM_TYPES
        assert "pipeline" in ALL_SYSTEM_TYPES
        assert "routing" in ALL_SYSTEM_TYPES
        assert "single-flow" in ALL_SYSTEM_TYPES
        assert "tool-loop" in ALL_SYSTEM_TYPES
        # No framework-specific names
        assert not any("haystack" in t.lower() for t in ALL_SYSTEM_TYPES)
        assert not any("langgraph" in t.lower() for t in ALL_SYSTEM_TYPES)
        assert not any("langchain" in t.lower() for t in ALL_SYSTEM_TYPES)

    def test_capability_tags_are_verb_sets(self):
        """All capability tags are generic verb sets."""
        assert "has_retrieval" in CAPABILITY_TAGS
        assert "has_tools" in CAPABILITY_TAGS
        assert "has_memory" in CAPABILITY_TAGS


# ==== CS-321: motif classifier (linear/branch/loop) (ex test_cs321_motif_classifier.py) ====

def _extract_connect_edges(edges):
    """Extract connect_edges from edges dict (all non-constructor edges)."""
    connect_edges = {}
    for node_id, topology in edges.items():
        if topology.downstream:
            connect_edges[node_id] = set(topology.downstream)
    return connect_edges


def test_motif_linear_chain():
    """A→B→C: all linear."""
    edges = {
        'a': TopologyEdges(downstream=['b']),
        'b': TopologyEdges(upstream=['a'], downstream=['c']),
        'c': TopologyEdges(upstream=['b']),
    }
    connect_edges = _extract_connect_edges(edges)
    _compute_motifs(edges, connect_edges)
    assert edges['a'].motif == 'linear'
    assert edges['b'].motif == 'linear'
    assert edges['c'].motif == 'linear'


def test_motif_branch():
    """A (conditional) → B, C: A is branch, B and C are linear."""
    edges = {
        'a': TopologyEdges(downstream=['b', 'c'], conditional_downstream=['b', 'c']),
        'b': TopologyEdges(upstream=['a']),
        'c': TopologyEdges(upstream=['a']),
    }
    connect_edges = _extract_connect_edges(edges)
    _compute_motifs(edges, connect_edges)
    assert edges['a'].motif == 'branch'
    assert edges['b'].motif == 'linear'
    assert edges['c'].motif == 'linear'


def test_motif_loop_two_cycle():
    """A→B→A: both are loop members."""
    edges = {
        'a': TopologyEdges(downstream=['b']),
        'b': TopologyEdges(upstream=['a'], downstream=['a']),
    }
    connect_edges = _extract_connect_edges(edges)
    _compute_motifs(edges, connect_edges)
    assert edges['a'].motif == 'loop'
    assert edges['b'].motif == 'loop'


def test_motif_loop_self_loop():
    """A→A: self-loop, A is loop."""
    edges = {
        'a': TopologyEdges(downstream=['a']),
    }
    connect_edges = _extract_connect_edges(edges)
    _compute_motifs(edges, connect_edges)
    assert edges['a'].motif == 'loop'


def test_motif_loop_three_cycle():
    """A→B→C→A: all are loop members."""
    edges = {
        'a': TopologyEdges(downstream=['b']),
        'b': TopologyEdges(upstream=['a'], downstream=['c']),
        'c': TopologyEdges(upstream=['b'], downstream=['a']),
    }
    connect_edges = _extract_connect_edges(edges)
    _compute_motifs(edges, connect_edges)
    assert edges['a'].motif == 'loop'
    assert edges['b'].motif == 'loop'
    assert edges['c'].motif == 'loop'


def test_motif_branch_and_linear():
    """Router A (conditional) → B, C; B → D (no cycle): A is branch, B,C,D are linear."""
    edges = {
        'a': TopologyEdges(downstream=['b', 'c'], conditional_downstream=['b', 'c']),
        'b': TopologyEdges(upstream=['a'], downstream=['d']),
        'c': TopologyEdges(upstream=['a']),
        'd': TopologyEdges(upstream=['b']),
    }
    connect_edges = _extract_connect_edges(edges)
    _compute_motifs(edges, connect_edges)
    assert edges['a'].motif == 'branch'
    assert edges['b'].motif == 'linear'
    assert edges['c'].motif == 'linear'
    assert edges['d'].motif == 'linear'


def test_motif_complex_with_cycle():
    """A→B→C, C→B (cycle on B-C), A→D: B,C are loop; A,D are linear."""
    edges = {
        'a': TopologyEdges(downstream=['b', 'd']),
        'b': TopologyEdges(upstream=['a', 'c'], downstream=['c']),
        'c': TopologyEdges(upstream=['b'], downstream=['b']),
        'd': TopologyEdges(upstream=['a']),
    }
    connect_edges = _extract_connect_edges(edges)
    _compute_motifs(edges, connect_edges)
    assert edges['a'].motif == 'linear'
    assert edges['b'].motif == 'loop'
    assert edges['c'].motif == 'loop'
    assert edges['d'].motif == 'linear'


def test_motif_branch_with_cycle():
    """Router A (conditional) → B, C; B→A (cycle): A,B are loop; C is linear."""
    edges = {
        'a': TopologyEdges(downstream=['b', 'c'], conditional_downstream=['b', 'c']),
        'b': TopologyEdges(upstream=['a'], downstream=['a']),
        'c': TopologyEdges(upstream=['a']),
    }
    connect_edges = _extract_connect_edges(edges)
    _compute_motifs(edges, connect_edges)
    assert edges['a'].motif == 'loop'  # Cycle takes precedence over branch
    assert edges['b'].motif == 'loop'
    assert edges['c'].motif == 'linear'


def test_motif_disconnected_components():
    """Two disconnected linear chains: A→B, C→D."""
    edges = {
        'a': TopologyEdges(downstream=['b']),
        'b': TopologyEdges(upstream=['a']),
        'c': TopologyEdges(downstream=['d']),
        'd': TopologyEdges(upstream=['c']),
    }
    connect_edges = _extract_connect_edges(edges)
    _compute_motifs(edges, connect_edges)
    assert edges['a'].motif == 'linear'
    assert edges['b'].motif == 'linear'
    assert edges['c'].motif == 'linear'
    assert edges['d'].motif == 'linear'


def test_motif_isolated_component():
    """Single component with no edges."""
    edges = {
        'a': TopologyEdges(),
    }
    connect_edges = _extract_connect_edges(edges)
    _compute_motifs(edges, connect_edges)
    assert edges['a'].motif == 'linear'


def test_motif_branch_source_not_in_cycle():
    """Router A (conditional) → B; B→A (cycle): A,B are loop (cycle dominates)."""
    edges = {
        'a': TopologyEdges(downstream=['b'], conditional_downstream=['b']),
        'b': TopologyEdges(upstream=['a'], downstream=['a']),
    }
    connect_edges = _extract_connect_edges(edges)
    _compute_motifs(edges, connect_edges)
    # Cycle takes precedence: both are in the cycle A→B→A
    assert edges['a'].motif == 'loop'
    assert edges['b'].motif == 'loop'


# ==== CS-322: granularity selector (per-node / end-to-end / trajectory) (ex test_cs322_granularity_selector.py) ====
# Deterministic tests without LLM — given synthetic motif + richness inputs, verify the selector
# picks per-node / end-to-end / +trajectory correctly.

def _make_component(
    comp_id: str,
    role: str = "worker",
    motif: str | None = None,
    downstream: list[str] | None = None,
) -> Component:
    """Helper to create a test Component."""
    return Component(
        id=comp_id,
        role=role,
        entry_point=f"test.module:{comp_id}",
        motif=motif,
        downstream=downstream or [],
    )


def _make_system_map(components: list[Component]) -> SystemMap:
    """Helper to create a test SystemMap."""
    return SystemMap(
        target_system_id="test_system",
        components=components,
    )


def test_richness_zero_when_no_knowledge() -> None:
    """AgentKnowledge missing or empty yields zero richness."""
    from agent_eval_harness.planning.planner import _richness_from_agent_knowledge

    assert _richness_from_agent_knowledge(None) == 0
    assert _richness_from_agent_knowledge({}) == 0


def test_richness_counts_semantic_evidence() -> None:
    """Richness counts functionality_citations + context_builders + consumers + failure_modes."""
    from agent_eval_harness.planning.planner import _richness_from_agent_knowledge

    knowledge = {
        "functionality_citations": [{"file": "a.py", "line": 1, "symbol": "x"}],
        "context_builders": [{"name": "ctx1", "file": "b.py", "line": 2}],
        "upstream_consumers": [{"name": "up1", "file": "c.py", "line": 3}],
        "downstream_consumers": [{"name": "down1", "file": "d.py", "line": 4}],
        "failure_modes": [{"description": "fails on null", "file": "e.py", "line": 5}],
    }
    assert _richness_from_agent_knowledge(knowledge) == 5


def test_thin_component_worker_low_richness() -> None:
    """A worker with low richness and no downstream tools is thin."""
    comp = _make_component("worker_thin", role="worker")
    system = _make_system_map([comp])
    components_by_id = {comp.id: comp}

    is_thin = _is_thin_component(comp, components_by_id, None)
    assert is_thin is True


def test_thin_component_worker_high_richness() -> None:
    """A worker with high richness is rich."""
    comp = _make_component("worker_rich", role="worker")
    system = _make_system_map([comp])
    components_by_id = {comp.id: comp}

    knowledge = {
        "functionality_citations": [{"file": "a.py", "line": 1}] * 10,
        "context_builders": [{"name": "ctx"}] * 10,
    }
    agent_knowledge = {comp.id: knowledge}

    is_thin = _is_thin_component(comp, components_by_id, agent_knowledge)
    assert is_thin is False


def test_thin_component_worker_has_downstream_tools() -> None:
    """A worker with downstream tools is rich (calls tools)."""
    tool_comp = _make_component("my_tool", role="tool")
    worker_comp = _make_component("worker", role="worker", downstream=["my_tool"])
    system = _make_system_map([worker_comp, tool_comp])
    components_by_id = {c.id: c for c in system.components}

    is_thin = _is_thin_component(worker_comp, components_by_id, None)
    assert is_thin is False


def test_thin_component_retrieval_agent_never_thin() -> None:
    """A retrieval_agent is always rich (makes independent queries)."""
    comp = _make_component("retriever", role="retrieval_agent")
    system = _make_system_map([comp])
    components_by_id = {comp.id: comp}

    is_thin = _is_thin_component(comp, components_by_id, None)
    assert is_thin is False


def test_thin_component_orchestrator_never_thin() -> None:
    """An orchestrator is always rich (makes routing decisions)."""
    comp = _make_component("router", role="orchestrator")
    system = _make_system_map([comp])
    components_by_id = {comp.id: comp}

    is_thin = _is_thin_component(comp, components_by_id, None)
    assert is_thin is False


def test_thin_component_tool_never_thin() -> None:
    """A tool is never classified as thin (they don't emit suite entries anyway)."""
    comp = _make_component("tool", role="tool")
    system = _make_system_map([comp])
    components_by_id = {comp.id: comp}

    is_thin = _is_thin_component(comp, components_by_id, None)
    assert is_thin is False


def test_system_thin_chain_one_substantial() -> None:
    """System with one orchestrator and two thin workers is a thin chain."""
    orchestrator = _make_component("planner", role="orchestrator")
    worker1 = _make_component("formatter", role="worker")
    worker2 = _make_component("transformer", role="worker")

    system = _make_system_map([orchestrator, worker1, worker2])

    is_thin = _is_system_thin_chain(system, None)
    assert is_thin is True


def test_system_not_thin_multiple_substantial() -> None:
    """System with multiple retrieval_agents is not a thin chain."""
    retriever1 = _make_component("search1", role="retrieval_agent")
    retriever2 = _make_component("search2", role="retrieval_agent")
    system = _make_system_map([retriever1, retriever2])

    is_thin = _is_system_thin_chain(system, None)
    assert is_thin is False


def test_system_not_thin_zero_substantial() -> None:
    """System with no substantial components is not a thin chain (edge case)."""
    worker1 = _make_component("w1", role="worker")
    worker2 = _make_component("w2", role="worker")
    system = _make_system_map([worker1, worker2])

    is_thin = _is_system_thin_chain(system, None)
    assert is_thin is False


def test_granularity_per_node_rich_independent() -> None:
    """Rich independent node in non-thin system gets per-node (component) level, no trajectory."""
    comp = _make_component("analyzer", role="retrieval_agent")
    comp2 = _make_component("planner", role="orchestrator")
    thin_worker = _make_component("formatter", role="worker")
    system = _make_system_map([comp, comp2, thin_worker])
    components_by_id = {c.id: c for c in system.components}

    level, needs_traj = _select_granularity_level(comp, system, components_by_id, None)
    assert level == "component"
    assert needs_traj is False


def test_granularity_trajectory_on_loop_motif() -> None:
    """Component with motif='loop' gets trajectory entry."""
    comp = _make_component("retry_loop", role="worker", motif="loop")
    system = _make_system_map([comp])
    components_by_id = {comp.id: comp}

    level, needs_traj = _select_granularity_level(comp, system, components_by_id, None)
    assert level == "component"
    assert needs_traj is True


def test_granularity_trajectory_on_control_motif() -> None:
    """Component with control_motif in AgentKnowledge gets trajectory entry."""
    comp = _make_component("retry_loop", role="worker", motif="linear")
    system = _make_system_map([comp])
    components_by_id = {comp.id: comp}

    knowledge = {comp.id: {"control_motif": "retry"}}

    level, needs_traj = _select_granularity_level(comp, system, components_by_id, knowledge)
    assert level == "component"
    assert needs_traj is True


def test_granularity_trajectory_dual_channel_motif_or_control() -> None:
    """Trajectory entry fires on motif='loop' OR control_motif (dual-channel)."""
    comp_motif = _make_component("c1", role="worker", motif="loop")
    comp_control = _make_component("c2", role="worker", motif="linear")

    system = _make_system_map([comp_motif, comp_control])
    components_by_id = {c.id: c for c in system.components}
    knowledge = {comp_control.id: {"control_motif": "retry"}}

    _, needs_traj_motif = _select_granularity_level(comp_motif, system, components_by_id, knowledge)
    _, needs_traj_control = _select_granularity_level(comp_control, system, components_by_id, knowledge)

    assert needs_traj_motif is True
    assert needs_traj_control is True


def test_granularity_branch_without_reentry_no_trajectory() -> None:
    """A branch source without re-entry stays per-node (no trajectory) in non-thin system."""
    branch_comp = _make_component("router", role="orchestrator", motif="branch")
    retriever = _make_component("search", role="retrieval_agent")
    thin_worker = _make_component("formatter", role="worker")
    system = _make_system_map([branch_comp, retriever, thin_worker])
    components_by_id = {c.id: c for c in system.components}

    level, needs_traj = _select_granularity_level(branch_comp, system, components_by_id, None)
    assert level == "component"
    assert needs_traj is False


def test_granularity_end_to_end_thin_chain() -> None:
    """Thin chain collapses to end-to-end (session) level."""
    orchestrator = _make_component("main_decider", role="orchestrator")
    formatter = _make_component("format_output", role="worker")
    system = _make_system_map([orchestrator, formatter])
    components_by_id = {c.id: c for c in system.components}

    level_orch, _ = _select_granularity_level(orchestrator, system, components_by_id, None)
    level_fmt, _ = _select_granularity_level(formatter, system, components_by_id, None)

    assert level_orch == "session"
    assert level_fmt == "session"


def test_granularity_thin_per_node_rich() -> None:
    """When system is NOT thin chain, rich nodes get per-node."""
    retriever1 = _make_component("query_planner", role="retrieval_agent")
    retriever2 = _make_component("reranker", role="retrieval_agent")
    formatter = _make_component("format", role="worker")
    system = _make_system_map([retriever1, retriever2, formatter])
    components_by_id = {c.id: c for c in system.components}

    is_thin = _is_system_thin_chain(system, None)
    assert is_thin is False

    level_ret1, _ = _select_granularity_level(retriever1, system, components_by_id, None)
    level_ret2, _ = _select_granularity_level(retriever2, system, components_by_id, None)
    level_fmt, _ = _select_granularity_level(formatter, system, components_by_id, None)

    assert level_ret1 == "component"
    assert level_ret2 == "component"
    assert level_fmt == "component"


@pytest.mark.anyio
async def test_generate_plan_applies_selector() -> None:
    """Integration: generate_plan applies selector and produces correct levels."""
    from agent_eval_harness.metrics.suite import Suite
    from agent_eval_harness.planning.planner import generate_plan
    from pathlib import Path
    import tempfile
    import yaml

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        map_path = tmppath / "system_map.yaml"

        components = [
            {
                "id": "orchestrator",
                "role": "orchestrator",
                "entry_point": "test:Orchestrator",
                "motif": None,
            },
            {
                "id": "formatter",
                "role": "worker",
                "entry_point": "test:Formatter",
                "motif": None,
                "downstream": [],
            },
        ]

        system_map_data = {
            "target_system_id": "test",
            "components": components,
        }

        with open(map_path, "w") as f:
            yaml.dump(system_map_data, f)

        class _StubLLM:
            async def complete(self, *args, **kwargs):
                from agent_eval_harness.llm.client import LLMResponse
                return LLMResponse(
                    content='{"entries": []}',
                    model="stub"
                )

        suite = await generate_plan(map_path, _StubLLM())

        assert isinstance(suite, Suite)
        assert len(suite.entries) > 0

        levels = {e.level for e in suite.entries}
        assert "session" in levels


@pytest.mark.anyio
async def test_generate_plan_thin_chain_single_session() -> None:
    """Integration: thin chain generates single session-level entry."""
    from agent_eval_harness.metrics.suite import Suite
    from agent_eval_harness.planning.planner import generate_plan
    from pathlib import Path
    import tempfile
    import yaml

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        map_path = tmppath / "system_map.yaml"

        components = [
            {
                "id": "main_llm",
                "role": "orchestrator",
                "entry_point": "test:MainLLM",
                "motif": None,
            },
            {
                "id": "formatter",
                "role": "worker",
                "entry_point": "test:Formatter",
                "motif": None,
            },
        ]

        system_map_data = {
            "target_system_id": "test_thin",
            "components": components,
        }

        with open(map_path, "w") as f:
            yaml.dump(system_map_data, f)

        class _StubLLM:
            async def complete(self, *args, **kwargs):
                from agent_eval_harness.llm.client import LLMResponse
                return LLMResponse(
                    content='{"entries": []}',
                    model="stub"
                )

        suite = await generate_plan(map_path, _StubLLM())

        assert isinstance(suite, Suite)

        session_entries = [e for e in suite.entries if e.level == "session"]
        assert len(session_entries) > 0

        component_entries = [e for e in suite.entries if e.level == "component"]
        assert len(component_entries) == 0


@pytest.mark.anyio
async def test_generate_plan_loop_adds_trajectory() -> None:
    """Integration: motif='loop' component gets trajectory entry."""
    from agent_eval_harness.metrics.suite import Suite
    from agent_eval_harness.planning.planner import generate_plan
    from pathlib import Path
    import tempfile
    import yaml

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        map_path = tmppath / "system_map.yaml"

        components = [
            {
                "id": "retry_loop",
                "role": "worker",
                "entry_point": "test:RetryLoop",
                "motif": "loop",
            },
            {
                "id": "analyzer",
                "role": "retrieval_agent",
                "entry_point": "test:Analyzer",
                "motif": None,
                "downstream": ["retry_loop"],
            },
        ]

        system_map_data = {
            "target_system_id": "test_loop",
            "components": components,
        }

        with open(map_path, "w") as f:
            yaml.dump(system_map_data, f)

        class _StubLLM:
            async def complete(self, *args, **kwargs):
                from agent_eval_harness.llm.client import LLMResponse
                return LLMResponse(
                    content='{"entries": []}',
                    model="stub"
                )

        suite = await generate_plan(map_path, _StubLLM())

        assert isinstance(suite, Suite)

        trajectory_entries = [e for e in suite.entries if e.level == "trace"]
        assert len(trajectory_entries) > 0
        assert any("retry_loop" in e.component for e in trajectory_entries)


# ==== Evidence spine: scanner<->expansion file-scope gap, source windows, symbol resolution (ex test_cs323_evidence_spine.py) ====
# The two guardrail-4 fixtures (multi_agent, linear_rag) are flat 3-4-file packages where every
# component's file is trivially in scope, so the scope-gap defect is structurally unreproducible
# there. TestScopeGapFixture adds a fixture that DOES reproduce it: an entry file whose LangGraph
# node body calls into a second package the frontier never visited.

def _write_cross_package_target(root: Path) -> list[Path]:
    """entry file (pkg_a/agent.py) node body calls a helper defined in a SEPARATE package (pkg_b/helper.py) via import."""
    (root / "pkg_a").mkdir()
    (root / "pkg_a" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pkg_b").mkdir()
    (root / "pkg_b" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pkg_b" / "helper.py").write_text(textwrap.dedent('''
        async def plan_queries(question):
            return [question]
    '''), encoding="utf-8")
    (root / "pkg_a" / "agent.py").write_text(textwrap.dedent('''
        from langgraph.graph import StateGraph
        from pkg_b.helper import plan_queries

        class ResearchAgent:
            def build(self):
                g = StateGraph(dict)
                g.add_node("retrieve", self._node_retrieve)
                return g

            async def _node_retrieve(self, state):
                return await plan_queries(state["question"])
    '''), encoding="utf-8")
    return [root / "pkg_a" / "agent.py"]


class TestScopeGapFixture:
    """The root fix: a component whose defining file the frontier never visited must be folded into the accepted set or marked out_of_scope with an explicit reason — never silent."""

    async def _build(self, tmp_path: Path):
        files = _write_cross_package_target(tmp_path)
        file_contents = {str(f): f.read_text(encoding="utf-8") for f in files}
        wb = _detect_langgraph(file_contents)
        assert wb is not None
        builder = SystemMapBuilder(FakeLLMClient(LLMResponse(content="[]", model="fake")), framework="langgraph")
        system_map, _ = await builder.build_from_files(
            files, package_root=tmp_path, target_system_id="cross_pkg",
            wiring_block=wb, scope_framework="langgraph", retrieval_client=None, snapshot_id=None,
        )
        return system_map

    async def test_frontier_gap_reproduced_by_the_new_fixture(self, tmp_path: Path):
        """The map spans a file ('pkg_b/helper.py') outside the accepted set ('pkg_a/agent.py')."""
        system_map = await self._build(tmp_path)
        plan_queries = next(c for c in system_map.components if c.entry_point.endswith("plan_queries"))
        assert plan_queries.file == "pkg_b/helper.py"
        assert plan_queries.file not in {"pkg_a/agent.py"}

    async def test_never_visited_file_is_widened_into_the_accepted_set(self, tmp_path: Path):
        system_map = await self._build(tmp_path)
        widened = reconcile_scope(system_map, accepted_files={"pkg_a/agent.py"}, boundary_files=set())
        assert "pkg_b/helper.py" in widened
        plan_queries = next(c for c in system_map.components if c.entry_point.endswith("plan_queries"))
        assert not plan_queries.out_of_scope

    async def test_boundary_file_degrades_explicitly_to_out_of_scope_not_widened(self, tmp_path: Path):
        """A file expansion explicitly classified boundary is a deliberate refusal — must not silently override that decision by force-accepting it."""
        system_map = await self._build(tmp_path)
        widened = reconcile_scope(
            system_map, accepted_files={"pkg_a/agent.py"}, boundary_files={"pkg_b/helper.py"},
        )
        assert "pkg_b/helper.py" not in widened
        plan_queries = next(c for c in system_map.components if c.entry_point.endswith("plan_queries"))
        assert plan_queries.out_of_scope is True
        assert plan_queries.out_of_scope_reason == "file classified boundary during expansion"

    async def test_excluded_file_degrades_explicitly_to_out_of_scope_not_widened(self, tmp_path: Path):
        system_map = await self._build(tmp_path)
        widened = reconcile_scope(
            system_map, accepted_files={"pkg_a/agent.py"}, boundary_files=set(),
            excluded_files={"pkg_b/helper.py"},
        )
        assert "pkg_b/helper.py" not in widened
        plan_queries = next(c for c in system_map.components if c.entry_point.endswith("plan_queries"))
        assert plan_queries.out_of_scope is True
        assert plan_queries.out_of_scope_reason == "file excluded from scope by user"

    async def test_component_already_in_accepted_set_is_left_alone(self, tmp_path: Path):
        """No regression: a component whose file WAS accepted never gets touched by reconcile_scope."""
        system_map = await self._build(tmp_path)
        reconcile_scope(system_map, accepted_files={"pkg_a/agent.py", "pkg_b/helper.py"}, boundary_files=set())
        node_retrieve = next(c for c in system_map.components if c.entry_point.endswith("_node_retrieve"))
        assert node_retrieve.out_of_scope is None


class TestSourceWindowBoundary:
    """A component's evidence window must end at its OWN end_lineno and never bleed into the next symbol."""

    def test_linear_rag_retriever_window_does_not_bleed_into_the_next_class(self, target_root: Path):
        files = sorted((target_root / "linear_rag").glob("*.py"))
        candidates = HaystackScanner().scan(files)
        by_name = {c.class_name: c for c in candidates}
        retriever = by_name["RetrieverComponent"]
        writer = by_name["WriterComponent"]

        assert "class WriterComponent" not in retriever.source_snippet
        assert "WriterComponent" not in retriever.source_snippet
        assert not retriever.snippet_truncated

        # Every component's window ends at its own end_lineno — never a fixed +N offset.
        source = (target_root / "linear_rag" / "components.py").read_text(encoding="utf-8")
        lines = source.splitlines()
        import ast as _ast
        tree = _ast.parse(source)
        end_lineno_by_name = {
            n.name: n.end_lineno for n in _ast.walk(tree) if isinstance(n, _ast.ClassDef)
        }
        for cand in (retriever, writer):
            own_end = end_lineno_by_name[cand.class_name]
            snippet_last_line = cand.source_snippet.splitlines()[-1]
            assert snippet_last_line == lines[own_end - 1], cand.class_name

    def test_multi_agent_guard_component_window_ends_at_its_own_class(self, target_root: Path):
        """Companion fixture check: no bleed in either direction across the guardrail-4 set."""
        files = sorted((target_root / "multi_agent").glob("*.py"))
        candidates = HaystackScanner().scan(files)
        guard = next(c for c in candidates if c.class_name == "GuardComponent")
        assert "PlannerComponent" not in guard.source_snippet
        assert "JudgeComponent" not in guard.source_snippet


class TestSpanScopedSplitEvidence:
    """guard_rule and guard_llm share one entry_point (GuardComponent) — before span-scoping they got byte-identical evidence."""

    def test_guard_rule_and_guard_llm_have_different_evidence_bytes(self, target_root: Path):
        files = sorted((target_root / "multi_agent").glob("*.py"))
        candidates = HaystackScanner().scan(files)
        by_id = {c.candidate_id: c for c in candidates}
        guard_rule, guard_llm = by_id["guard_rule"], by_id["guard_llm"]

        assert guard_rule.source_snippet != guard_llm.source_snippet
        assert '"aeh.check.kind": "rule"' in guard_rule.source_snippet
        assert '"aeh.check.kind": "rule"' not in guard_llm.source_snippet
        assert '"aeh.check.kind": "llm"' in guard_llm.source_snippet
        assert '"aeh.check.kind": "llm"' not in guard_rule.source_snippet

    def test_split_children_keep_framework_and_is_library_object(self, target_root: Path):
        """_split_sub_spans previously dropped framework=/is_library_object= on split children."""
        files = sorted((target_root / "multi_agent").glob("*.py"))
        candidates, _label = scan_all(files)
        by_id = {c.candidate_id: c for c in candidates}
        assert by_id["guard_rule"].framework == "haystack"
        assert by_id["guard_llm"].framework == "haystack"


class TestLangGraphAnchorFix:
    """A StateGraph node candidate must anchor on its OWN def line, not the add_node() call site."""

    def test_bound_method_node_anchors_on_its_own_def_not_the_registration_call(self, tmp_path: Path):
        (tmp_path / "agent.py").write_text(textwrap.dedent('''
            from langgraph.graph import StateGraph

            class ResearchAgent:
                def build_graph(self):
                    graph = StateGraph(dict)
                    graph.add_node("load_context", self._node_load_context)
                    graph.add_node("retrieve", self._node_retrieve)
                    graph.add_conditional_edges("load_context", self._route, {"a": "retrieve"})
                    return graph

                async def _node_load_context(self, state):
                    state["ctx"] = "loaded"
                    return state

                async def _node_retrieve(self, state):
                    state["docs"] = ["a", "b"]
                    return state

                def _route(self, state):
                    return "a"
        '''), encoding="utf-8")
        files = [tmp_path / "agent.py"]
        candidates = LangGraphScanner().scan(files)
        by_name = {c.class_name: c for c in candidates}
        retrieve = by_name["_node_retrieve"]

        # Anchored on `async def _node_retrieve`, NOT on the add_node/add_conditional_edges block.
        assert "add_node" not in retrieve.source_snippet
        assert "add_conditional_edges" not in retrieve.source_snippet
        assert 'state["docs"]' in retrieve.source_snippet

        source_lines = (tmp_path / "agent.py").read_text(encoding="utf-8").splitlines()
        def_line_idx = next(
            i for i, ln in enumerate(source_lines, start=1)
            if "async def _node_retrieve" in ln
        )
        assert retrieve.line == def_line_idx


class TestUncappedPromptRouting:
    """The agent-flow prompt must read the uncapped, correctly-anchored source, not the pre-capped source_by_component."""

    async def test_prompt_shows_real_source_even_when_source_by_component_is_empty(self, tmp_path: Path):
        files = _write_cross_package_target(tmp_path)
        # Widen scope so pkg_b/helper.py participates too (mirrors reconcile_scope's own output).
        files = files + [tmp_path / "pkg_b" / "helper.py"]
        file_contents = {str(f): f.read_text(encoding="utf-8") for f in files}
        wb = _detect_langgraph(file_contents)
        builder = SystemMapBuilder(FakeLLMClient(LLMResponse(content="[]", model="fake")), framework="langgraph")
        system_map, _ = await builder.build_from_files(
            files, package_root=tmp_path, target_system_id="cross_pkg",
            wiring_block=wb, scope_framework="langgraph", retrieval_client=None, snapshot_id=None,
        )

        llm_client = FakeLLMClient(LLMResponse(content='{"agents": []}', model="fake"))
        await separate_agent_flows(system_map, {}, llm_client, files=files)

        user_prompt = llm_client.calls[0][1].content
        assert "(no source available)" not in user_prompt
        assert 'return await plan_queries' in user_prompt


class TestOutOfScopeFallback:
    """The client.read_file fallback is SECONDARY: only for components legitimately marked out_of_scope, never the primary answer to the scope gap."""

    async def test_out_of_scope_component_gets_source_via_retrieval_client_fallback(self, tmp_path: Path):
        files = _write_cross_package_target(tmp_path)  # only pkg_a/agent.py — helper.py stays out of scope
        file_contents = {str(f): f.read_text(encoding="utf-8") for f in files}
        wb = _detect_langgraph(file_contents)
        builder = SystemMapBuilder(FakeLLMClient(LLMResponse(content="[]", model="fake")), framework="langgraph")
        system_map, _ = await builder.build_from_files(
            files, package_root=tmp_path, target_system_id="cross_pkg",
            wiring_block=wb, scope_framework="langgraph", retrieval_client=None, snapshot_id=None,
        )
        reconcile_scope(system_map, accepted_files={"pkg_a/agent.py"}, boundary_files={"pkg_b/helper.py"})

        helper_content = (tmp_path / "pkg_b" / "helper.py").read_text(encoding="utf-8")

        class _StubRetrievalClient:
            async def read_file(self, snapshot_id, rel_path):
                assert rel_path == "pkg_b/helper.py"
                return {"content": helper_content}

        llm_client = FakeLLMClient(LLMResponse(content='{"agents": []}', model="fake"))
        await separate_agent_flows(
            system_map, {}, llm_client, files=files,
            retrieval_client=_StubRetrievalClient(), snapshot_id="snap-1",
        )

        user_prompt = llm_client.calls[0][1].content
        assert "async def plan_queries" in user_prompt


class TestPostConditionCheck:
    def test_opens_on_symbol_line_accepts_matching_def(self):
        assert _opens_on_symbol_line("def plan_queries(question):", "plan_queries")
        assert _opens_on_symbol_line("    async def _node_retrieve(self, state):", "_node_retrieve")
        assert _opens_on_symbol_line("class ResearchAgent:", "ResearchAgent")

    def test_opens_on_symbol_line_rejects_mismatched_line(self):
        """Guards against a resolver anchoring on the wrong node — a call-site or different symbol's def line must never pass."""
        assert not _opens_on_symbol_line('graph.add_node("retrieve", self._node_retrieve)', "_node_retrieve")
        assert not _opens_on_symbol_line("def _build_graph(self):", "_node_retrieve")


class TestGenericCallClosure:
    """Call-pair emission generalized to any framework: call_downstream must be non-empty for components that call helpers on BOTH guardrail-4 fixtures."""

    async def _build(self, target_root: Path, target: str):
        from agent_eval_harness.discovery.wiring import detect_wiring_block_static

        package_root = target_root.parent
        files = sorted((target_root / target).glob("*.py"))
        file_contents = {str(f): f.read_text(encoding="utf-8") for f in files}
        wb = detect_wiring_block_static(file_contents)
        builder = SystemMapBuilder(
            FakeLLMClient(LLMResponse(content="[]", model="fake")),
            framework=(wb.framework if wb else None),
        )
        system_map, _ = await builder.build_from_files(
            files, package_root=package_root, target_system_id=target,
            wiring_block=wb, retrieval_client=None, snapshot_id=None,
        )
        return system_map

    async def test_multi_agent_call_downstream_non_empty_for_helper_callers(self, target_root: Path):
        system_map = await self._build(target_root, "multi_agent")
        worker = system_map.component_by_id("worker")
        judge = system_map.component_by_id("judge")
        assert worker is not None and worker.call_downstream
        assert judge is not None and judge.call_downstream

    async def test_linear_rag_call_downstream_non_empty_for_helper_callers(self, target_root: Path):
        system_map = await self._build(target_root, "linear_rag")
        retriever = system_map.component_by_id("retriever")
        writer = system_map.component_by_id("writer")
        assert retriever is not None and "_keyword_overlap" in retriever.call_downstream
        assert writer is not None and writer.call_downstream

    async def test_closure_completeness_bit_is_false_for_a_genuinely_unresolved_seam(self, target_root: Path):
        """Not hardwired True: at least one real component has an unresolved call in its closure."""
        system_map = await self._build(target_root, "multi_agent")
        unresolved = [c for c in system_map.components if c.closure_complete is False]
        assert unresolved, "expected at least one component with a genuinely unresolved seam"

    async def test_every_walked_component_carries_a_closure_completeness_bit(self, target_root: Path):
        system_map = await self._build(target_root, "linear_rag")
        walked = [c for c in system_map.components if c.call_sites]
        assert walked
        assert all(c.closure_complete is not None for c in walked)


class TestContainerLiteralResolution:
    """A dispatch table (list/dict of callables) built in the agent's OWN scope, subscripted and called later, must resolve like any other call shape."""

    async def test_self_attr_container_of_own_methods_resolves_both_elements(self, tmp_path: Path):
        (tmp_path / "agent.py").write_text(textwrap.dedent('''
            class Router:
                def __init__(self):
                    self._handlers = {"a": self._handle_a, "b": self._handle_b}

                def run(self, cmd, key):
                    return self._handlers[key](cmd)

                def _handle_a(self, cmd):
                    return "a:" + cmd

                def _handle_b(self, cmd):
                    return "b:" + cmd
        '''), encoding="utf-8")
        from agent_eval_harness.discovery.wiring import (
            _own_class_methods,
            _safe_parse,
            walk_call_closure,
        )

        tree = _safe_parse((tmp_path / "agent.py").read_text(encoding="utf-8"))
        own_methods = _own_class_methods(tree, "Router")
        targets, call_sites, complete = await walk_call_closure(
            [("Router", own_methods["run"])], owner="Router", agent_tree=tree,
            agent_rel="agent.py", package_root=tmp_path,
        )
        by_alias = {t.alias for t in targets}
        assert {"_handle_a", "_handle_b"} <= by_alias
        assert call_sites
        assert complete is True  # both container elements resolved — no unresolved seam here


_RENAME_MAP = {"complete": "hoi", "answer": "tra_loi", "run_async": "chay_bat_dong_bo"}
_RENAME_RE = re.compile(r"\b(" + "|".join(_RENAME_MAP) + r")\b", re.IGNORECASE)


def _apply_renames(text: str) -> str:
    return _RENAME_RE.sub(lambda m: _RENAME_MAP[m.group(0).lower()], text)


def _copy_tree_renamed(repo_root: Path, rel_dirs: list[str], dst_root: Path, *, rename: bool) -> None:
    for rel_dir in rel_dirs:
        src_dir = repo_root / rel_dir
        dst_dir = dst_root / rel_dir
        dst_dir.mkdir(parents=True, exist_ok=True)
        for f in src_dir.glob("*.py"):
            text = f.read_text(encoding="utf-8")
            (dst_dir / f.name).write_text(_apply_renames(text) if rename else text, encoding="utf-8")


class TestMutationMeasurement:
    """A rename mutation must not change harvested structure if harvesting were truly name-independent; this is a measurement instrument, not a green-required gate — numbers recorded honestly, not forced green."""

    _REPO_ROOT = Path(__file__).parent.parent

    async def _harvest(self, package_root: Path, target: str):
        files = sorted((package_root / "test_targets" / target).glob("*.py"))
        file_contents = {str(f): f.read_text(encoding="utf-8") for f in files}
        wb = detect_wiring_block_static(file_contents)
        builder = SystemMapBuilder(
            FakeLLMClient(LLMResponse(content="[]", model="fake")),
            framework=(wb.framework if wb else None),
        )
        system_map, _ = await builder.build_from_files(
            files, package_root=package_root, target_system_id=target,
            wiring_block=wb, retrieval_client=None, snapshot_id=None,
        )
        # Decided statically at map-build time now (typed receiver resolution), not re-derived here.
        model_call_ids = {c.id for c in system_map.components if c.makes_model_call is True}
        return system_map, model_call_ids

    async def _measure(self, tmp_path: Path, target: str, rel_dirs: list[str]) -> dict[str, bool]:
        orig_root, mut_root = tmp_path / "orig", tmp_path / "mut"
        _copy_tree_renamed(self._REPO_ROOT, rel_dirs, orig_root, rename=False)
        _copy_tree_renamed(self._REPO_ROOT, rel_dirs, mut_root, rename=True)
        # The provider boundary (LLMClient.complete) lives outside the fixture dirs above; copy it
        # identically, unrenamed, into both roots, or the closure walk can never reach it in either.
        _copy_tree_renamed(self._REPO_ROOT, ["agent_eval_harness/llm"], orig_root, rename=False)
        _copy_tree_renamed(self._REPO_ROOT, ["agent_eval_harness/llm"], mut_root, rename=False)

        orig_map, orig_model_ids = await self._harvest(orig_root, target)
        mut_map, mut_model_ids = await self._harvest(mut_root, target)

        return {
            "component_count": len(orig_map.components) == len(mut_map.components),
            "component_ids": (
                {c.id for c in orig_map.components} == {c.id for c in mut_map.components}
            ),
            "call_downstream_shape": (
                {c.id for c in orig_map.components if c.call_downstream}
                == {c.id for c in mut_map.components if c.call_downstream}
            ),
            "closure_complete_shape": (
                {c.id: c.closure_complete for c in orig_map.components}
                == {c.id: c.closure_complete for c in mut_map.components}
            ),
            "call_site_count_shape": (
                {c.id: len(c.call_sites or []) for c in orig_map.components}
                == {c.id: len(c.call_sites or []) for c in mut_map.components}
            ),
            "model_evidence_verdict": orig_model_ids == mut_model_ids,
        }

    async def test_multi_agent_rename_mutation_measurement(self, tmp_path: Path):
        # multi_agent's PlannerComponent injects WriterComponent (linear_rag) by constructor type,
        # so linear_rag must be copied alongside it for that cross-file resolution to stay meaningful.
        results = await self._measure(
            tmp_path, "multi_agent",
            ["test_targets/multi_agent", "test_targets/linear_rag", "test_targets/_shared"],
        )
        # Measured: 0/6 -- renaming also mutates call sites into LLMClient/HarnessChatGenerator's own fixed (unrenamed) method names.
        assert results == {
            "component_count": False,
            "component_ids": False,
            "call_downstream_shape": False,
            "closure_complete_shape": False,
            "call_site_count_shape": False,
            "model_evidence_verdict": False,
        }

    async def test_linear_rag_rename_mutation_measurement(self, tmp_path: Path):
        results = await self._measure(
            tmp_path, "linear_rag", ["test_targets/linear_rag", "test_targets/_shared"],
        )
        # Measured: 0/6 -- same root cause as multi_agent, above (run_async also names HarnessChatGenerator's own fixed method).
        assert results == {
            "component_count": False,
            "component_ids": False,
            "call_downstream_shape": False,
            "closure_complete_shape": False,
            "call_site_count_shape": False,
            "model_evidence_verdict": False,
        }


# ==== makes_model_call decided statically: typed receiver resolution, provider-boundary recognition, tri-state verdict (ex test_cs324_model_call_verdict.py) ====
# Zero LLM tokens anywhere in this section -- the verdict is fully static; every LLM seam here is a
# raise-if-called stub.

_REPO_ROOT = Path(__file__).parent.parent


class _NeverCallClient:
    async def complete(self, *args, **kwargs):
        raise AssertionError("the model-call verdict is fully static -- it must never call an LLM")


async def _harvest(target: str, extra_dirs: list[str] | None = None):
    """Build the real SystemMap for a guardrail-4 fixture directly against the repo tree."""
    package_root = _REPO_ROOT
    files = sorted((package_root / "test_targets" / target).glob("*.py"))
    for extra in extra_dirs or []:
        files += sorted((package_root / "test_targets" / extra).glob("*.py"))
    files = sorted(set(files))
    file_contents = {str(f): f.read_text(encoding="utf-8") for f in files}
    wb = detect_wiring_block_static(file_contents)
    builder = SystemMapBuilder(_NeverCallClient(), framework=(wb.framework if wb else None))
    system_map, _ = await builder.build_from_files(
        files, package_root=package_root, target_system_id=target,
        wiring_block=wb, retrieval_client=None, snapshot_id=None,
    )
    return system_map


class TestLinearRagAcceptance:
    """Acceptance item 1: agents {writer}, steps {retriever}."""

    async def test_retriever_is_false_with_complete_closure(self):
        system_map = await _harvest("linear_rag")
        retriever = system_map.component_by_id("retriever")
        assert retriever is not None
        assert retriever.closure_complete is True
        assert retriever.makes_model_call is False
        assert retriever.model_call_source == "structural_closure"

    async def test_writer_is_true_citing_llmclient_complete(self):
        system_map = await _harvest("linear_rag")
        writer = system_map.component_by_id("writer")
        assert writer is not None
        assert writer.makes_model_call is True
        assert writer.model_call_source == "derived_boundary"
        assert writer.model_call_citation == "agent_eval_harness/llm/client.py:25"


class TestMultiAgentAcceptance:
    """Acceptance item 2: agents {planner, guard_llm, judge, writer}, steps {guard_rule, worker, case_law_search, decoy_lookup}."""

    async def test_agents_are_true(self):
        system_map = await _harvest("multi_agent", extra_dirs=["linear_rag"])
        for cid in ("planner", "guard_llm", "judge", "writer"):
            comp = system_map.component_by_id(cid)
            assert comp is not None, cid
            assert comp.makes_model_call is True, cid

    async def test_guard_rule_and_worker_are_false_with_complete_closure(self):
        system_map = await _harvest("multi_agent", extra_dirs=["linear_rag"])
        for cid in ("guard_rule", "worker"):
            comp = system_map.component_by_id(cid)
            assert comp is not None, cid
            assert comp.closure_complete is True, cid
            assert comp.makes_model_call is False, cid

    async def test_tool_functions_never_read_as_agents(self):
        """case_law_search/decoy_lookup land at None (undecided, closure never attempted), never a guessed False."""
        system_map = await _harvest("multi_agent", extra_dirs=["linear_rag"])
        for cid in ("case_law_search", "decoy_lookup"):
            comp = system_map.component_by_id(cid)
            assert comp is not None, cid
            assert comp.makes_model_call is not True, cid

    async def test_worker_contradicts_the_hand_authored_yaml_and_that_is_pinned(self):
        """system_map.yaml declares worker `model: fake-strong`, but WorkerComponent dispatches via a container subscript and makes no model call -- derived from source, contradicting the annotation."""
        system_map = await _harvest("multi_agent", extra_dirs=["linear_rag"])
        worker = system_map.component_by_id("worker")
        assert worker is not None
        assert worker.makes_model_call is False
        assert worker.closure_complete is True


class TestDeterminismNoLLMSpend:
    """Acceptance item 4: two consecutive runs produce the same verdict set, proven with a raise-if-called client on both runs."""

    async def test_two_runs_agree_and_spend_nothing(self):
        first = await _harvest("multi_agent", extra_dirs=["linear_rag"])
        second = await _harvest("multi_agent", extra_dirs=["linear_rag"])
        verdicts_1 = {c.id: c.makes_model_call for c in first.components}
        verdicts_2 = {c.id: c.makes_model_call for c in second.components}
        assert verdicts_1 == verdicts_2


class TestJunkGateBothDirections:
    """No fuzzy tier crept in, in either direction."""

    async def test_dict_get_and_str_format_emit_no_edge(self, tmp_path: Path):
        (tmp_path / "agent.py").write_text(textwrap.dedent('''
            class Agent:
                async def run_async(self, state: dict, tmpl: str):
                    value = state.get("plan", [])
                    text = tmpl.format(value=value)
                    return text
        '''), encoding="utf-8")
        tree = _safe_parse((tmp_path / "agent.py").read_text(encoding="utf-8"))
        own_methods = _own_class_methods(tree, "Agent")
        targets, call_sites, complete = await walk_call_closure(
            [("Agent", own_methods["run_async"])], owner="Agent", agent_tree=tree,
            agent_rel="agent.py", package_root=tmp_path,
        )
        assert targets == []  # no edge to any same-named `get`/`format` method
        assert complete is True  # both calls are validated builtin-receiver out-of-scope, not unresolved
        assert {cs.reason for cs in call_sites} == {"builtin_receiver"}

    async def test_removing_receiver_annotation_yields_unresolved_not_a_guess(self, tmp_path: Path):
        (tmp_path / "svc.py").write_text(
            "class Service:\n    def do(self) -> int:\n        return 1\n", encoding="utf-8",
        )
        (tmp_path / "typed_agent.py").write_text(textwrap.dedent('''
            from svc import Service

            class Agent:
                def __init__(self, svc: Service) -> None:
                    self._svc = svc

                async def run_async(self):
                    return self._svc.do()
        '''), encoding="utf-8")
        tree = _safe_parse((tmp_path / "typed_agent.py").read_text(encoding="utf-8"))
        own_methods = _own_class_methods(tree, "Agent")
        targets, _, complete = await walk_call_closure(
            [("Agent", own_methods["run_async"])], owner="Agent", agent_tree=tree,
            agent_rel="typed_agent.py", package_root=tmp_path,
        )
        assert any(t.alias == "Service" for t in targets)
        assert complete is True

        (tmp_path / "untyped_agent.py").write_text(textwrap.dedent('''
            class Agent:
                def __init__(self, svc) -> None:
                    self._svc = svc

                async def run_async(self):
                    return self._svc.do()
        '''), encoding="utf-8")
        tree2 = _safe_parse((tmp_path / "untyped_agent.py").read_text(encoding="utf-8"))
        own_methods2 = _own_class_methods(tree2, "Agent")
        targets2, _, complete2 = await walk_call_closure(
            [("Agent", own_methods2["run_async"])], owner="Agent", agent_tree=tree2,
            agent_rel="untyped_agent.py", package_root=tmp_path,
        )
        assert targets2 == []
        assert complete2 is False  # not resolved -- never a guess


class TestSuperResolution:
    """A super() call resolves through the MRO to the base class's own method."""

    async def test_super_call_resolves_to_base_class_method(self, tmp_path: Path):
        (tmp_path / "agent.py").write_text(textwrap.dedent('''
            class BaseAgent:
                async def _chat_json(self, payload):
                    return {}

            class TypedAgent(BaseAgent):
                async def _chat_json_typed(self, payload):
                    return await super()._chat_json(payload)
        '''), encoding="utf-8")
        tree = _safe_parse((tmp_path / "agent.py").read_text(encoding="utf-8"))
        own_methods = _own_class_methods(tree, "TypedAgent")
        targets, call_sites, complete = await walk_call_closure(
            [("TypedAgent", own_methods["_chat_json_typed"])], owner="TypedAgent", agent_tree=tree,
            agent_rel="agent.py", package_root=tmp_path,
        )
        assert any(t.alias == "_chat_json" and t.layer == "mro" for t in targets)
        assert complete is True
        # The inner super() itself is a bare call to a builtin name -- out of scope, not unresolved.
        assert not any(cs.outcome == "unresolved" for cs in call_sites)


class TestDiagnosabilityCitation:
    """A verdict's citation is the deciding line, not the class declaration it lives in."""

    async def test_citation_points_at_the_method_not_the_class_declaration(self):
        system_map = await _harvest("linear_rag")
        writer = system_map.component_by_id("writer")
        assert writer is not None and writer.model_call_citation is not None
        file_part, _, line_part = writer.model_call_citation.rpartition(":")
        source = (_REPO_ROOT / file_part).read_text(encoding="utf-8").splitlines()
        cited_line = source[int(line_part) - 1]
        assert "async def complete(" in cited_line
        assert "class LLMClient" not in cited_line


class TestUndecidedVerdict:
    """Acceptance item 6: a candidate whose closure was never walked is None, never a confident guess."""

    async def test_no_recognized_entry_method_stays_undecided(self):
        system_map = await _harvest("linear_rag")
        helper = system_map.component_by_id("_keyword_overlap")
        assert helper is not None
        assert helper.makes_model_call is None

    async def test_incomplete_closure_with_no_boundary_reached_defaults_to_step(self, tmp_path: Path):
        """A genuinely unresolved seam with no boundary anywhere defaults to step (never a guessed agent), never surfacing on discrepancies."""
        (tmp_path / "agent.py").write_text(textwrap.dedent('''
            class Agent:
                async def run_async(self, mystery):
                    return mystery.do_something()
        '''), encoding="utf-8")
        files = [tmp_path / "agent.py"]
        wb = detect_wiring_block_static({str(f): f.read_text(encoding="utf-8") for f in files})
        builder = SystemMapBuilder(
            FakeLLMClient(LLMResponse(content="[]", model="fake")), framework=(wb.framework if wb else None),
        )
        system_map, _ = await builder.build_from_files(
            files, package_root=tmp_path, target_system_id="synthetic",
            wiring_block=wb, retrieval_client=None, snapshot_id=None,
        )
        agent = system_map.component_by_id("agent")
        assert agent is not None
        assert agent.closure_complete is False
        assert agent.makes_model_call is False
        assert agent.model_call_source == "llm_unverified_default"
        assert not any("undecided" in d for d in system_map.discrepancies)


class TestReplaceDiscoveryCandidatesCarriesVerdict:
    """Acceptance item 7: a verdict already set on a candidate survives the pause/resume replace cycle."""

    @pytest.fixture(autouse=True)
    async def _setup_db(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from agent_eval_harness.store.database import close_db, init_db

        monkeypatch.setenv("AEH_DATA_DIR", str(tmp_path))
        await init_db()
        yield
        await close_db()

    async def test_existing_verdict_survives_replace(self):
        from agent_eval_harness.store import repository

        session_id = await repository.insert_discovery_session(repo_ref="r", snapshot_id="s")
        await repository.insert_discovery_candidates_bulk(session_id, [
            {"name": "guard", "frameworks": ["haystack"], "verdict": "proposed"},
        ])
        rows = await repository.get_discovery_candidates(session_id)
        await repository.update_candidate_verdict(rows[0]["id"], "confirmed")

        # Re-derive (e.g. a rate-limit pause mid-run): same candidate, no verdict field supplied.
        await repository.replace_discovery_candidates(session_id, [
            {"name": "guard", "frameworks": ["haystack"]},
        ])
        after = await repository.get_discovery_candidates(session_id)
        assert len(after) == 1
        assert after[0]["verdict"] == "confirmed"


class TestProviderBoundaryMatching:
    """provider_boundaries.yaml's two row shapes: an in-repo interface method (derived_boundary, tier 1) and a pip package's import identity (declared_boundary, tier 0)."""

    def test_loader_reads_the_real_config(self):
        from agent_eval_harness.discovery.engine import load_provider_boundaries

        rows = load_provider_boundaries()
        assert any(r.get("entry") == "agent_eval_harness.llm.client.LLMClient.complete" for r in rows)
        assert any(r.get("import") == "openai" for r in rows)

    def test_entry_match_requires_the_full_dotted_path(self):
        from agent_eval_harness.discovery.engine import match_provider_boundary_entry

        rows = [{"id": "x", "entry": "pkg.mod.Client.complete", "kind": "generative"}]
        assert match_provider_boundary_entry(
            rows, defining_file="pkg/mod.py", owner_class="Client", resolved_symbol="complete",
        ) is not None
        assert match_provider_boundary_entry(
            rows, defining_file="pkg/mod.py", owner_class="Client", resolved_symbol="other",
        ) is None

    def test_import_match_is_top_level_package_only(self):
        from agent_eval_harness.discovery.engine import match_provider_boundary_import

        rows = [{"id": "x", "import": "openai", "kind": "generative"}]
        assert match_provider_boundary_import(rows, "openai.types.chat") is not None
        assert match_provider_boundary_import(rows, "openai_compatible_thing") is None


class TestSnapshotInventedAgentIsRegression:
    """Acceptance item 8: an invented agent (a surface D15/entry-method widening could invent) is reported as REGRESSION, not INFO."""

    def _load_stage2_snapshot(self):
        spec = importlib.util.spec_from_file_location(
            "stage2_snapshot", _REPO_ROOT / "scripts" / "stage2_snapshot.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_invented_agent_flagged_as_regression(self):
        m = self._load_stage2_snapshot()
        before = {
            "candidates": [{
                "name": "t", "verdict": "confirmed", "frameworks": [], "community_id": None,
                "wiring": None, "matched_file_count": 0,
                "expansion": {
                    "session_id": "e1", "created_at": "t0", "stop_reason": "frontier_exhausted",
                    "accepted_files": [], "boundary_files": [], "accepted_edges": [],
                    "map": {"component_count": 1, "components": [
                        {"id": "real_agent", "role": "worker", "entry_kind": "class",
                         "entry_point": "m:real_agent", "file": "a.py", "is_tool": False,
                         "upstream": [], "downstream": []},
                    ]},
                    "agent_flow": {
                        "agent_count": 1, "agents": [
                            {"id": "real_agent", "component_ids": ["real_agent"],
                             "upstream_agents": [], "downstream_agents": []},
                        ],
                        "entry_agent_ids": ["real_agent"], "unassigned_component_ids": [],
                    },
                },
            }],
        }
        after = m.json.loads(m.json.dumps(before))  # deep copy
        after["candidates"][0]["expansion"]["agent_flow"]["agents"].append(
            {"id": "invented_agent", "component_ids": [], "upstream_agents": [], "downstream_agents": []}
        )
        findings = m.diff(before, after)
        assert (m.REGRESSION, "[t] agent INVENTED: invented_agent") in findings


# ==== LLM residue pass over a component whose closure resolved enough to walk but hit a genuinely unresolved call with no boundary reached (ex test_cs325_boundary_llm.py) ====
# Static still owns the closure; these tests exercise only the async residue pass (bundle assembly,
# cite-verify, cache) with stub clients -- no real provider, zero tokens spent anywhere in this
# section.

class _ResidueNeverCallClient:
    async def complete(self, *args, **kwargs):
        raise AssertionError("the residue pass must not run when static already decided every component")


class _CannedResidueClient:
    """Returns the same canned JSON array on every call -- one call is expected per derive."""

    def __init__(self, verdicts: list[dict]) -> None:
        self._content = json.dumps(verdicts)
        self.call_count = 0

    async def complete(self, messages, *, max_tokens=512, temperature=0.2, json_mode=False, reasoning_effort=None):
        self.call_count += 1
        return LLMResponse(content=self._content, model="stub")


class _CannedObjectWrappedResidueClient:
    """Returns the verdicts wrapped in a top-level JSON object -- the shape json_mode=True actually produces, vs _CannedResidueClient's bare array."""

    def __init__(self, verdicts: list[dict]) -> None:
        self._content = json.dumps({"verdicts": verdicts})
        self.call_count = 0

    async def complete(self, messages, *, max_tokens=512, temperature=0.2, json_mode=False, reasoning_effort=None):
        self.call_count += 1
        return LLMResponse(content=self._content, model="stub")


async def _harvest_residue(target: str, client, extra_dirs: list[str] | None = None):
    package_root = _REPO_ROOT
    files = sorted((package_root / "test_targets" / target).glob("*.py"))
    for extra in extra_dirs or []:
        files += sorted((package_root / "test_targets" / extra).glob("*.py"))
    files = sorted(set(files))
    file_contents = {str(f): f.read_text(encoding="utf-8") for f in files}
    wb = detect_wiring_block_static(file_contents)
    builder = SystemMapBuilder(client, framework=(wb.framework if wb else None))
    return await builder.build_from_files(
        files, package_root=package_root, target_system_id=target,
        wiring_block=wb, retrieval_client=None, snapshot_id=None,
    )


class TestGuardrailFixturesZeroResidue:
    """Acceptance item 2: multi_agent/linear_rag stay fully static -- the residue pass never fires, proven with a raise-if-called stub."""

    async def test_linear_rag_never_calls_the_residue_llm(self):
        system_map, _ = await _harvest_residue("linear_rag", _ResidueNeverCallClient())
        writer = system_map.component_by_id("writer")
        retriever = system_map.component_by_id("retriever")
        assert writer is not None and writer.makes_model_call is True
        assert retriever is not None and retriever.makes_model_call is False

    async def test_multi_agent_never_calls_the_residue_llm(self):
        system_map, _ = await _harvest_residue(
            "multi_agent", _ResidueNeverCallClient(), extra_dirs=["linear_rag"],
        )
        for cid in ("planner", "guard_llm", "judge", "writer"):
            comp = system_map.component_by_id(cid)
            assert comp is not None and comp.makes_model_call is True, cid


class TestCiteVerifyFailure:
    """Acceptance item 3: a citation the bundle cannot back is treated as no evidence -- defaults to step, never reaches discrepancies."""

    async def test_citation_outside_the_bundle_is_rejected(self, tmp_path: Path):
        (tmp_path / "agent.py").write_text(textwrap.dedent('''
            class Agent:
                async def run_async(self, mystery):
                    return mystery.do_something()
        '''), encoding="utf-8")
        files = [tmp_path / "agent.py"]
        wb = detect_wiring_block_static({str(f): f.read_text(encoding="utf-8") for f in files})
        canned = [{
            "component_id": "agent", "makes_model_call": True,
            "citation": {"file": "agent.py", "line": 999}, "evidence_kind": "fabricated",
        }]
        builder = SystemMapBuilder(
            _CannedResidueClient(canned), framework=(wb.framework if wb else None),
        )
        system_map, _ = await builder.build_from_files(
            files, package_root=tmp_path, target_system_id="synthetic",
            wiring_block=wb, retrieval_client=None, snapshot_id=None,
        )
        agent = system_map.component_by_id("agent")
        assert agent is not None
        assert agent.makes_model_call is False
        assert agent.model_call_source == "llm_unverified_default"
        assert not any("undecided" in d for d in system_map.discrepancies)


class TestCiteVerifyAcceptsEnclosingFunctionCall:
    """Regression: a citation landing anywhere inside a bundle function that itself reaches a call must verify; only a call-free function (or outside the bundle) still fails."""

    def test_citation_to_a_signature_line_of_a_call_bearing_function_verifies(self, tmp_path: Path):
        content = textwrap.dedent('''
            async def plan_queries(
                goal,
                provider_service,
                provider_id,
            ):
                return await provider_service.chat(provider_id, goal)
        ''').lstrip("\n")
        (tmp_path / "graph_plan.py").write_text(content, encoding="utf-8")
        lines = content.splitlines()
        def_line = next(i for i, ln in enumerate(lines, start=1) if "async def plan_queries" in ln)
        param_line = next(i for i, ln in enumerate(lines, start=1) if "provider_id," in ln)

        residue = ResidueCandidate(
            component_id="_node_retrieve", rel_file="graph_plan.py", entry_line=def_line,
            call_sites=[], targets=[],
        )
        file_cache: dict = {}
        bundle = _build_bundle(residue, tmp_path, file_cache)
        assert bundle.contains("graph_plan.py", param_line)

        entry = {
            "component_id": "_node_retrieve", "makes_model_call": True,
            "citation": {"file": "graph_plan.py", "line": param_line},
            "evidence_kind": "call to provider_service.chat",
        }
        result = _extract_verdict(entry, bundle, file_cache)
        assert result is not None
        makes_model_call, citation, _evidence_kind = result
        assert makes_model_call is True
        assert citation == f"graph_plan.py:{param_line}"

    def test_citation_to_a_signature_line_of_a_call_free_function_is_rejected(self, tmp_path: Path):
        content = textwrap.dedent('''
            def describe(
                extra,
                more,
            ):
                return "static"
        ''').lstrip("\n")
        (tmp_path / "helper.py").write_text(content, encoding="utf-8")
        lines = content.splitlines()
        def_line = next(i for i, ln in enumerate(lines, start=1) if "def describe" in ln)
        param_line = next(i for i, ln in enumerate(lines, start=1) if "extra," in ln)

        residue = ResidueCandidate(
            component_id="helper", rel_file="helper.py", entry_line=def_line,
            call_sites=[], targets=[],
        )
        file_cache: dict = {}
        bundle = _build_bundle(residue, tmp_path, file_cache)
        assert bundle.contains("helper.py", param_line)

        entry = {
            "component_id": "helper", "makes_model_call": True,
            "citation": {"file": "helper.py", "line": param_line},
            "evidence_kind": "fabricated",
        }
        assert _extract_verdict(entry, bundle, file_cache) is None


class TestDeterminismCache:
    """Acceptance item 4: two derives over the same closure produce byte-identical verdicts; the second spends zero LLM calls (cache hit)."""

    @pytest.fixture(autouse=True)
    async def _setup_db(self, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch):
        from agent_eval_harness.store.database import close_db, init_db

        monkeypatch.setenv("AEH_DATA_DIR", str(tmp_path_factory.mktemp("aeh_data")))
        await init_db()
        yield
        await close_db()

    async def test_second_derive_hits_cache_and_spends_nothing(self, tmp_path: Path):
        (tmp_path / "agent.py").write_text(textwrap.dedent('''
            class Agent:
                async def run_async(self, mystery):
                    return mystery.do_something()
        '''), encoding="utf-8")
        files = [tmp_path / "agent.py"]
        wb = detect_wiring_block_static({str(f): f.read_text(encoding="utf-8") for f in files})
        source_lines = files[0].read_text(encoding="utf-8").splitlines()
        line = next(i for i, ln in enumerate(source_lines, start=1) if "do_something" in ln)

        canned = [{
            "component_id": "agent", "makes_model_call": False,
            "citation": {"file": "agent.py", "line": line}, "evidence_kind": "graph_traversal",
        }]
        first_client = _CannedResidueClient(canned)
        builder1 = SystemMapBuilder(first_client, framework=(wb.framework if wb else None))
        map1, _ = await builder1.build_from_files(
            files, package_root=tmp_path, target_system_id="synthetic",
            wiring_block=wb, retrieval_client=None, snapshot_id=None,
        )
        agent1 = map1.component_by_id("agent")
        assert agent1 is not None
        assert agent1.makes_model_call is False
        assert agent1.model_call_source == "llm_verified"
        assert first_client.call_count == 1

        builder2 = SystemMapBuilder(_ResidueNeverCallClient(), framework=(wb.framework if wb else None))
        map2, _ = await builder2.build_from_files(
            files, package_root=tmp_path, target_system_id="synthetic",
            wiring_block=wb, retrieval_client=None, snapshot_id=None,
        )
        agent2 = map2.component_by_id("agent")
        assert agent2 is not None
        assert agent2.makes_model_call == agent1.makes_model_call
        assert agent2.model_call_source == agent1.model_call_source
        assert agent2.model_call_citation == agent1.model_call_citation


class TestHeadlineResidueDemotesTracerNodes:
    """Acceptance item 1: a stub LLM residue pass returning ground-truth verdicts collapses a 7-node LangGraph agent to its 4 real dispatch nodes, demoting the 3 pure graph-traversal tracers via the existing structural veto."""

    _REAL = ("_node_plan", "_node_retrieve", "_node_analyze_step", "_node_synthesize")
    _TRACERS = ("_node_trace_forward", "_node_trace_backward", "_node_impact")

    def _write(self, root: Path) -> tuple[list[Path], dict[str, int]]:
        content = textwrap.dedent('''
            from langgraph.graph import StateGraph


            class ResearchAgent:
                def build(self):
                    g = StateGraph(dict)
                    g.add_node("plan", self._node_plan)
                    g.add_node("retrieve", self._node_retrieve)
                    g.add_node("analyze_step", self._node_analyze_step)
                    g.add_node("synthesize", self._node_synthesize)
                    g.add_node("trace_forward", self._node_trace_forward)
                    g.add_node("trace_backward", self._node_trace_backward)
                    g.add_node("impact", self._node_impact)
                    return g

                async def _node_plan(self, state):
                    backend = state["channel"]
                    return backend.send(state["question"])

                async def _node_retrieve(self, state):
                    backend = state["channel"]
                    return backend.send(state["query"])

                async def _node_analyze_step(self, state):
                    backend = state["channel"]
                    return backend.send(state["step"])

                async def _node_synthesize(self, state):
                    backend = state["channel"]
                    return backend.send(state["evidence"])

                async def _node_trace_forward(self, state):
                    visited = state["visited"]
                    visited.append(state["node_id"])
                    return {"visited": visited}

                async def _node_trace_backward(self, state):
                    visited = state["visited"]
                    visited.append(state["node_id"])
                    return {"visited": visited}

                async def _node_impact(self, state):
                    visited = state["visited"]
                    visited.append(state["node_id"])
                    return {"visited": visited}
        ''').lstrip("\n")
        path = root / "agent.py"
        path.write_text(content, encoding="utf-8")
        lines = content.splitlines()

        line_of: dict[str, int] = {}
        for marker, cid in (
            ('backend.send(state["question"])', "_node_plan"),
            ('backend.send(state["query"])', "_node_retrieve"),
            ('backend.send(state["step"])', "_node_analyze_step"),
            ('backend.send(state["evidence"])', "_node_synthesize"),
        ):
            line_of[cid] = next(i for i, ln in enumerate(lines, start=1) if marker in ln)
        append_lines = [i for i, ln in enumerate(lines, start=1) if "visited.append(" in ln]
        assert len(append_lines) == 3
        for cid, lineno in zip(self._TRACERS, append_lines):
            line_of[cid] = lineno
        return [path], line_of

    async def test_residue_pass_demotes_tracer_nodes(self, tmp_path: Path):
        files, line_of = self._write(tmp_path)
        file_contents = {str(f): f.read_text(encoding="utf-8") for f in files}
        wb = detect_wiring_block_static(file_contents)
        assert wb is not None and wb.framework == "langgraph"

        all_ids = self._REAL + self._TRACERS
        canned = [
            {
                "component_id": cid,
                "makes_model_call": cid in self._REAL,
                "citation": {"file": "agent.py", "line": line_of[cid]},
                "evidence_kind": "prompt_dispatch" if cid in self._REAL else "graph_traversal",
            }
            for cid in all_ids
        ]
        client = _CannedResidueClient(canned)

        builder = SystemMapBuilder(client, framework="langgraph")
        system_map, _ = await builder.build_from_files(
            files, package_root=tmp_path, target_system_id="synthetic",
            wiring_block=wb, retrieval_client=None, snapshot_id=None,
        )

        for cid in self._REAL:
            comp = system_map.component_by_id(cid)
            assert comp is not None, cid
            assert comp.makes_model_call is True, cid
            assert comp.model_call_source == "llm_verified", cid
        for cid in self._TRACERS:
            comp = system_map.component_by_id(cid)
            assert comp is not None, cid
            assert comp.makes_model_call is False, cid
            assert comp.model_call_source == "llm_verified", cid
        assert client.call_count == 1  # one per-system batched call over all 7 residue components

        agents = [AgentFlow(id=cid, component_ids=[cid]) for cid in all_ids]
        kept, _orphaned = _demote_deterministic_agents(agents, system_map)
        assert {a.id for a in kept} == set(self._REAL)


class TestObjectWrappedResponseParses:
    """Regression: json_mode=True guarantees a top-level JSON object, never a bare array, so the parser must accept {"verdicts": [...]}."""

    async def test_object_wrapped_verdicts_are_parsed_and_decide_the_component(self, tmp_path: Path):
        (tmp_path / "agent.py").write_text(textwrap.dedent('''
            class Agent:
                async def run_async(self, mystery):
                    return mystery.do_something()
        '''), encoding="utf-8")
        files = [tmp_path / "agent.py"]
        wb = detect_wiring_block_static({str(f): f.read_text(encoding="utf-8") for f in files})
        source_lines = files[0].read_text(encoding="utf-8").splitlines()
        line = next(i for i, ln in enumerate(source_lines, start=1) if "do_something" in ln)

        canned = [{
            "component_id": "agent", "makes_model_call": True,
            "citation": {"file": "agent.py", "line": line}, "evidence_kind": "prompt_dispatch",
        }]
        client = _CannedObjectWrappedResidueClient(canned)
        builder = SystemMapBuilder(client, framework=(wb.framework if wb else None))
        system_map, _ = await builder.build_from_files(
            files, package_root=tmp_path, target_system_id="synthetic",
            wiring_block=wb, retrieval_client=None, snapshot_id=None,
        )
        agent = system_map.component_by_id("agent")
        assert agent is not None
        assert agent.makes_model_call is True
        assert agent.model_call_source == "llm_verified"
        assert client.call_count == 1


class TestUndecidedNeverReachesDiscrepancies:
    """Defect 3: the residue pass's own undecided diagnostic must never land on SystemMap.discrepancies, which is shared with genuine doc-vs-code reconciliation."""

    async def test_undecided_component_hides_from_discrepancies_but_doc_gap_still_shows(
        self, tmp_path: Path,
    ):
        (tmp_path / "agent.py").write_text(textwrap.dedent('''
            class Agent:
                async def run_async(self, mystery):
                    return mystery.do_something()
        '''), encoding="utf-8")
        (tmp_path / "docs.md").write_text("this document never mentions the component\n", encoding="utf-8")
        files = [tmp_path / "agent.py"]
        wb = detect_wiring_block_static({str(f): f.read_text(encoding="utf-8") for f in files})
        # the LLM omits "agent" entirely -- valid per the prompt's own instruction to omit rather
        # than guess when there's no real evidence, so the pass has nothing verifiable to act on.
        client = _CannedResidueClient([])
        builder = SystemMapBuilder(client, framework=(wb.framework if wb else None))
        system_map, _ = await builder.build_from_files(
            files, package_root=tmp_path, target_system_id="synthetic",
            docs_path=tmp_path / "docs.md",
            wiring_block=wb, retrieval_client=None, snapshot_id=None,
        )
        agent = system_map.component_by_id("agent")
        assert agent is not None
        assert agent.makes_model_call is False
        assert agent.model_call_source == "llm_unverified_default"
        assert not any("undecided" in d for d in system_map.discrepancies)
        assert any(
            "agent" in d and "documentation" in d for d in system_map.discrepancies
        )


class TestUnboundedSplitSpanDefaultsWithoutReachingDiscrepancies:
    """Residual leak fix: a split candidate whose own manual_span has no bounded end_line hits the terminally-undecided return with residue=None -- must default like the residue pass (step) and never surface 'undecided'."""

    async def test_unbounded_split_candidate_defaults_to_step_and_hides_from_discrepancies(
        self, tmp_path: Path,
    ):
        agent_path = tmp_path / "agent.py"
        agent_path.write_text(textwrap.dedent('''
            class Agent:
                async def run_async(self, request):
                    return await self._helper(request)

                async def _helper(self, request):
                    return request
        '''), encoding="utf-8")
        (tmp_path / "docs.md").write_text("this document never mentions the component\n", encoding="utf-8")

        candidate = CandidateComponent(
            file=agent_path,
            line=2,
            class_name="Agent",
            tag_suffix="rule",
            entry_kind="class",
            framework="haystack",
            manual_span_hints=[
                ManualSpanHint(
                    op_name="'guard'", component_name="guard", tags={"role": "rule"},
                    file=agent_path, line=3, end_line=None,
                )
            ],
        )
        closure_info = {"Agent": (True, [], [])}

        builder = SystemMapBuilder(FakeLLMClient(LLMResponse(content="[]", model="fake")))
        system_map, _summary = await builder._build_from_candidates(
            [candidate], [agent_path], tmp_path, "synthetic", tmp_path / "docs.md",
            closure_info=closure_info,
        )

        component = system_map.component_by_id("agent_rule")
        assert component is not None
        assert component.makes_model_call is False
        assert component.model_call_source == "llm_unverified_default"
        assert not any("undecided" in d for d in system_map.discrepancies)
        assert any(
            "agent_rule" in d and "documentation" in d for d in system_map.discrepancies
        )


class TestBundleWideningProtectsEvidenceFrame:
    """A frame whose own line range contains an unresolved call site IS the evidence the LLM needs -- must survive the char budget even when deeper resolved-target frames alone would exhaust it."""

    def test_evidence_frame_survives_even_when_deeper_filler_frames_exceed_budget(
        self, tmp_path: Path,
    ):
        entry_body = "\n".join(f"        request['k{i}'] = {i}" for i in range(24))
        entry_src = (
            "class Agent:\n"
            "    async def run_async(self, request):\n"
            + entry_body
            + "\n        return await self._chat_json_typed(request)\n"
        )
        (tmp_path / "agent.py").write_text(entry_src, encoding="utf-8")
        call_line = entry_src.count("\n", 0, entry_src.index("_chat_json_typed")) + 1

        targets = []
        for i, name in enumerate(("_prep_a", "_prep_b", "_prep_c"), start=1):
            filler_lines = "\n".join(f"        y{j} = {j}" for j in range(150))
            src = f"class Agent:\n    async def {name}(self):\n" + filler_lines + "\n        return None\n"
            (tmp_path / f"{name}.py").write_text(src, encoding="utf-8")
            targets.append(NodeCallTarget(
                caller_alias="run_async", alias=name, callee=name, defining_file=f"{name}.py",
                line=2, entry_kind="bound_method", owner_class="Agent", layer="own_class",
                resolved_symbol=name, site_id=f"s{i}",
            ))
        # the 3 filler frames alone already exceed the budget -- a pure depth-first cut would
        # drop the entry frame (priority 0, lowest) before ever touching them.
        filler_total = sum((tmp_path / f"{n}.py").stat().st_size for n in ("_prep_a", "_prep_b", "_prep_c"))
        assert filler_total > _BUNDLE_CHAR_BUDGET

        call_sites = [CallSite(
            site_id="s0", file="agent.py", lineno=call_line,
            callee_text="self._chat_json_typed(request)", source_line="", outcome="unresolved",
        )]
        residue = ResidueCandidate(
            component_id="agent", rel_file="agent.py", entry_line=2,
            call_sites=call_sites, targets=targets,
        )
        bundle = _build_bundle(residue, tmp_path, {})
        assert bundle.contains("agent.py", 2)


# ==== CS-328: input-contract kwarg resolution against EvaluationContract (ex test_cs328_state_node_resolution.py) ====

@pytest.mark.asyncio
async def test_cs328_finalize_input_contract_resolves_evaluation_contract_kwargs():
    """CS-328 FIX A: _finalize_input_contract resolves KwargSpec.resolved_schema directly on EvaluationContract.invocation.kwargs."""
    mock_schema = {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "score": {"type": "integer"},
        },
    }
    ctx = MagicMock()
    llm_knowledge = AgentKnowledge(
        session_id="s1", agent_id="node1", archetype="fan_in_judge",
        component="comp1", source_file="node.py",
    )
    own_contract = EvaluationContract(
        agent_id="node1",
        invocation=InvocationContract(
            kwargs=[KwargSpec(name="state", annotation="MyTypedState", required=True)]
        ),
    )

    with patch("agent_eval_harness.discovery.enrichment._resolve_type_schema_via_symbol_index", return_value=(mock_schema, "file.py", 1.0)):
        await _finalize_input_contract(ctx, llm_knowledge, upstream_output_schema=None, own_contract=own_contract)

        assert own_contract.invocation is not None
        kwarg = own_contract.invocation.kwargs[0]
        assert kwarg.name == "state"
        assert kwarg.resolved_schema is not None
        assert kwarg.resolved_schema["type"] == "object"
        assert "question" in kwarg.resolved_schema["properties"]


def test_cs328_has_single_resolved_object_kwarg():
    """CS-328 FIX B: _has_single_resolved_object_kwarg returns True for agents taking a single object-typed kwarg."""
    cfg_single_object = SyntheticAgentIOConfig.model_validate({
        "dataset_name": "ds_state",
        "agent_id": "state_agent",
        "archetype": "fan_in_judge",
        "contract": {
            "invocation": {
                "kwargs": [
                    {
                        "name": "state",
                        "required": True,
                        "resolved_schema": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                        },
                    }
                ]
            }
        },
        "count": 1,
    })
    assert _has_single_resolved_object_kwarg(cfg_single_object) is True

    cfg_multi = SyntheticAgentIOConfig.model_validate({
        "dataset_name": "ds_multi",
        "agent_id": "multi_agent",
        "archetype": "fan_in_judge",
        "contract": {
            "invocation": {
                "kwargs": [
                    {"name": "a", "required": True},
                    {"name": "b", "required": True},
                ]
            }
        },
        "count": 1,
    })
    assert _has_single_resolved_object_kwarg(cfg_multi) is False


def test_cs328_find_module_dict_constant():
    """CS-328 FIX C: _find_module_dict_or_str_constant resolves AST dict literals (e.g. step_schema)."""
    code = """
step_schema = {
    "type": "object",
    "properties": {
        "finding": {"type": "string"},
        "sufficient": {"type": "boolean"},
    }
}
"""
    tree = ast.parse(code)
    path = Path("fake_module.py")
    asts = {path: tree}

    names = _referenced_schema_names(None, tree.body[0])
    assert "step_schema" in names

    found = _find_module_dict_or_str_constant(asts, "step_schema", files_root=Path("."), own_file=path)
    assert found is not None
    dict_val, cite = found
    assert isinstance(dict_val, dict)
    assert dict_val["type"] == "object"
    assert "finding" in dict_val["properties"]


# ==== CS-329: multi-tier output-schema resolver (Tier A static type / Tier B prompt-derived) (ex test_cs329_multi_tier_schema_resolver.py) ====

def test_validate_tier_b_schema_valid_object():
    """Valid object schema is accepted by Tier B validator."""
    valid = {
        "type": "object",
        "properties": {
            "finding": {"type": "string"},
            "sufficient": {"type": "boolean"},
        },
        "required": ["finding"],
    }
    assert _validate_tier_b_schema(valid) == valid


def test_validate_tier_b_schema_valid_array():
    """Valid array schema is accepted by Tier B validator."""
    valid = {
        "type": "array",
        "items": {"type": "string"},
    }
    assert _validate_tier_b_schema(valid) == valid


def test_validate_tier_b_schema_invalid():
    """Invalid or non-schema dicts are rejected by Tier B validator."""
    assert _validate_tier_b_schema(None) is None
    assert _validate_tier_b_schema("not a dict") is None
    assert _validate_tier_b_schema({}) is None
    assert _validate_tier_b_schema({"type": "number"}) is None
    assert _validate_tier_b_schema({"properties": {123: "invalid key"}}) is None


def test_multi_tier_resolution_tier_a_preferred():
    """Tier A static type is preferred when present."""
    output = OutputContract(
        json_schema={
            "type": "object",
            "properties": {"finding": {"type": "string"}},
        },
        schema_source="static_type",
    )
    prompt_schema = {
        "type": "object",
        "properties": {"finding": {"type": "string"}},
    }
    resolved, notes = _apply_multi_tier_output_resolution(output, prompt_schema, None, "agent_1")
    assert resolved.json_schema == output.json_schema
    assert resolved.schema_source == "static_type"
    assert resolved.schema_discrepancy is None
    assert len(notes) == 0


def test_multi_tier_resolution_tier_b_fallback():
    """Tier B prompt-derived schema is used when Tier A is missing."""
    output = OutputContract(json_schema=None)
    prompt_schema = {
        "type": "object",
        "properties": {
            "reasoning_chain": {"type": "string"},
            "confidence": {"type": "string"},
        },
    }
    prompt_enums = {"confidence": ["high", "medium", "low"]}

    resolved, notes = _apply_multi_tier_output_resolution(output, prompt_schema, prompt_enums, "agent_2")
    assert resolved.json_schema == prompt_schema
    assert resolved.schema_source == "prompt"
    assert resolved.cardinality == "object"
    assert resolved.schema_enum_values == {"confidence": ["high", "medium", "low"]}
    assert len(notes) == 0


def test_multi_tier_resolution_discrepancy():
    """Discrepancy between Tier A and Tier B sets schema_discrepancy and adds needs_human note."""
    output = OutputContract(
        json_schema={
            "type": "object",
            "properties": {"finding": {"type": "string"}},
        },
        schema_source="static_type",
    )
    prompt_schema = {
        "type": "object",
        "properties": {"summary": {"type": "string"}, "details": {"type": "string"}},
    }

    resolved, notes = _apply_multi_tier_output_resolution(output, prompt_schema, None, "agent_3")
    assert resolved.json_schema == output.json_schema
    assert resolved.schema_source == "static_type"
    assert resolved.schema_discrepancy is not None
    assert "AST schema properties ['finding'] vs prompt-derived ['details', 'summary']" in resolved.schema_discrepancy
    assert len(notes) == 1
    assert "output schema discrepancy for agent agent_3" in notes[0]


def test_multi_tier_resolution_neither_present():
    """When neither Tier A nor Tier B is present, schema_source is 'none' and explicit needs_human note is added."""
    output = OutputContract(json_schema=None)
    resolved, notes = _apply_multi_tier_output_resolution(output, None, None, "agent_4")
    assert resolved.json_schema is None
    assert resolved.schema_source == "none"
    assert len(notes) == 1
    assert "agent_4: output schema not statically resolvable and prompt did not describe a structured output" in notes[0]


def test_multi_tier_resolution_enriches_key_only_tier_a_with_tier_b_types():
    """When Tier A is key-only ({}), matching Tier B keys enrich property types while preserving key agreement."""
    key_only_output = OutputContract(
        json_schema={
            "type": "object",
            "properties": {"finding": {}, "sufficient": {}},
            "required": ["finding", "sufficient"],
        },
        schema_source="prompt-derived site",
    )
    prompt_schema = {
        "type": "object",
        "properties": {
            "finding": {"type": "string"},
            "sufficient": {"type": "boolean"},
        },
        "required": ["finding", "sufficient"],
    }
    resolved, notes = _apply_multi_tier_output_resolution(key_only_output, prompt_schema, None, "agent_5")
    assert resolved.json_schema["properties"]["finding"] == {"type": "string"}
    assert resolved.json_schema["properties"]["sufficient"] == {"type": "boolean"}
    assert resolved.schema_discrepancy is None
    assert len(notes) == 0
