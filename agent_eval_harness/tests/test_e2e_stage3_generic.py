"""CS-303 Slice 1 — Stage 1->3 acceptance harness on foreign (non-CodeSpectra) targets.

Proves D9's generic path — driven by the harvested contract, never a CodeSpectra-shaped
template — is what actually produces datasets for `multi_agent` and `linear_rag`. Loads each
target's real system_map.yaml, builds an AgentFlowMap inline (one agent per component; no LLM
call, no I/O beyond parsing), harvests real EvaluationContracts via AST, then drives
synthetic_agent_io.generate() through fulfillment._derive_config() with a FakeLLMClient.

Ground-truth eligible sets were pinned by reading the two system_map.yaml files and the actual
harvested contracts during CS-303 implementation (see the final report for the read-out)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_eval_harness.datasets import fulfillment
from agent_eval_harness.datasets.fulfillment import _derive_config
from agent_eval_harness.mapping.builder.contract_harvest import _archetype_for
from agent_eval_harness.datasets.generators import synthetic_agent_io
from agent_eval_harness.llm.client import LLMResponse
from agent_eval_harness.llm.fake_client import FakeLLMClient
from agent_eval_harness.mapping.agent_flow import AgentFlow, AgentFlowMap
from agent_eval_harness.mapping.builder.contract_harvest import harvest_contracts
from agent_eval_harness.mapping.system_map import load_system_map
from agent_eval_harness.metrics.suite import SuiteEntry
from agent_eval_harness.planning.contract import EvaluationContract

_ROOT = Path(__file__).parent.parent  # agent_eval_harness/ repo root (contains test_targets/)
_TARGETS_DIR = _ROOT / "test_targets"

# Pinned by reading test_targets/<target>/system_map.yaml + the real harvest output.
_ELIGIBLE = {
    "multi_agent": {
        "guard_rule", "guard_llm", "planner", "worker",
        "case_law_search_tool", "decoy_tool", "judge", "writer",
    },
    # CS-311 Slice 3: function-entry harvest now resolves `pipeline:run_retrieve` (a plain function),
    # so `retriever` harvests real kwargs and becomes eligible — this is the dogfood proof of F5.
    "linear_rag": {"writer", "retriever"},
}
# Exclusion reasons for agents NOT in _ELIGIBLE — every one must be "unimplemented" and the
# reason must be "no harvestable input at all", never a leftover "no RetrievalService" guess.
_EXCLUDED_REASONS = {
    "linear_rag": {},
}


def _build_agent_flow_map(system_map) -> AgentFlowMap:
    """One AgentFlow per component — the simplest inline construction named in the CS-303 spec
    (no LLM call, no I/O)."""
    agents = [AgentFlow(id=c.id, component_ids=[c.id]) for c in system_map.components]
    return AgentFlowMap(target_system_id=system_map.target_system_id, agents=agents)


def _harvest(target: str) -> dict[str, EvaluationContract]:
    map_path = _TARGETS_DIR / target / "system_map.yaml"
    system_map = load_system_map(map_path)
    agent_flow_map = _build_agent_flow_map(system_map)
    files = list(_TARGETS_DIR.rglob("*.py"))
    return harvest_contracts(system_map, agent_flow_map, files, _ROOT)


def _plausible_value(annotation: str | None):
    ann = (annotation or "").lower()
    if "list" in ann:
        return ["a plausible value"]
    if "dict" in ann:
        return {}
    if "bool" in ann:
        return True
    if "int" in ann:
        return 1
    return "a plausible value"


def _canned_candidate(contract: EvaluationContract) -> dict:
    """Builds a schema-valid {input, gold} candidate purely from the harvested contract —
    no CodeSpectra knowledge, works identically for any target."""
    kwargs = contract.invocation.kwargs if contract.invocation else []
    case_binding = contract.invocation.case_binding if contract.invocation else {}
    input_: dict = {}
    for kw in kwargs:
        binding = case_binding.get(kw.name, "")
        if binding.startswith("config:"):
            continue
        input_[kw.name] = _plausible_value(kw.annotation)

    schema = contract.output.json_schema if contract.output else None
    gold: dict = {}
    if isinstance(schema, dict):
        for required_field in schema.get("required", []):
            gold[required_field] = []
    return {"input": input_, "gold": gold}


async def _run_agent(agent_id: str, contract: EvaluationContract) -> tuple[str, list, FakeLLMClient]:
    entries = [SuiteEntry(
        id=f"{agent_id}.synthetic_agent_io", component=agent_id, agent_id=agent_id,
        metric="schema_valid", metric_class="assertion",
    )]
    config = await _derive_config(
        "synthetic_agent_io", f"ds_{agent_id}", f"synthetic_agent_io/{agent_id}", entries,
        map_path="unused.yaml", snapshot_id="s1", snapshot_local_path=".",
        provider_id="p", model_id="m", dataset_id_by_group={}, painpoint=None, min_cases=1,
        eval_enabled_by_agent={agent_id: True},
        contract_by_agent={agent_id: contract},
        profile_by_agent={},
        knowledge_by_agent={},
    )
    assert config is not None, f"{agent_id}: _derive_config returned None unexpectedly"
    candidate = _canned_candidate(contract)
    llm_client = FakeLLMClient(LLMResponse(content=json.dumps([candidate]), model="fake"))
    cases = await synthetic_agent_io.generate(config, llm_client)
    return config["archetype"], cases, llm_client


@pytest.mark.parametrize("target", ["multi_agent", "linear_rag"])
async def test_ac1_eligible_set_equality_and_unimplemented_for_right_reason(target, monkeypatch):

    contracts = _harvest(target)
    assert contracts, f"{target}: harvest produced no contracts at all — fixture drifted"

    produced_eligible: set[str] = set()
    produced_unimplemented: set[str] = set()

    for agent_id, contract in contracts.items():
        archetype = _archetype_for(contract)
        if archetype == "unimplemented":
            produced_unimplemented.add(agent_id)
            continue
        produced_eligible.add(agent_id)
        got_archetype, cases, llm_client = await _run_agent(agent_id, contract)
        assert got_archetype == archetype
        assert len(cases) >= 1, f"{target}/{agent_id}: eligible archetype {archetype!r} produced 0 cases"
        assert cases[0].input.get("shape") == "generic", (
            f"{target}/{agent_id}: routed through a CodeSpectra archetype builder "
            f"(shape={cases[0].input.get('shape')!r}) instead of the generic path"
        )
        # contract-shaped: every non-config kwarg name the contract harvested appears as an
        # input field (D9 — field list comes from the contract, not a template).
        kwargs = contract.invocation.kwargs if contract.invocation else []
        case_binding = contract.invocation.case_binding if contract.invocation else {}
        expected_fields = {
            kw.name for kw in kwargs if not case_binding.get(kw.name, "").startswith("config:")
        }
        assert set(cases[0].input) - {"shape"} == expected_fields

    assert produced_eligible == _ELIGIBLE[target], (
        f"{target}: eligible set drifted — got {sorted(produced_eligible)}, "
        f"expected {sorted(_ELIGIBLE[target])}"
    )
    assert produced_unimplemented == set(_EXCLUDED_REASONS.get(target, {})), (
        f"{target}: unimplemented set drifted — got {sorted(produced_unimplemented)}, "
        f"expected {sorted(_EXCLUDED_REASONS.get(target, {}))}"
    )
    for agent_id in produced_unimplemented:
        contract = contracts[agent_id]
        kwarg_names = {k.name for k in contract.invocation.kwargs} if contract.invocation else set()
        assert not kwarg_names, (
            f"{target}/{agent_id}: marked unimplemented but has harvestable kwargs "
            f"{sorted(kwarg_names)} — wrong reason, should have gotten a real archetype"
        )
        assert not contract.has_retrieval_signal

    # CS-310 Slice 4: All archetype builders deleted; foreign agents use the generic path.


def test_ac2_legacy_override_dict_removed():
    """AC2 (post-removal half): _LEGACY_OVERRIDE was deleted once this file's Slice 1 run
    proved the generic/harvested path (CS-303 Slice 3b) — import-absence, not a spy."""
    assert not hasattr(synthetic_agent_io, "_LEGACY_OVERRIDE")
    assert not hasattr(fulfillment, "_LEGACY_OVERRIDE")
