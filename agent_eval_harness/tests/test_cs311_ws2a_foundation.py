"""CS-311 WS-2A foundation: scanner registry, entry-shape helpers, function/bound-method harvest,
and Stage-1->Stage-3 wiring_block reuse. No CodeSpectra literals — all fixtures are inline."""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path
from unittest.mock import patch

from agent_eval_harness.discovery.wiring import (
    WiringBlock,
    WiringEdge,
    WiringNode,
    enclosing_class_name,
    parse_entry_suffix,
    wiring_identity,
)
from agent_eval_harness.mapping.builder.contract_harvest import harvest_component_contract
from agent_eval_harness.mapping.builder.scanners import HaystackScanner, get_scanner
from agent_eval_harness.mapping.builder.topology import extract_topology
from agent_eval_harness.mapping.system_map import Component


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
    """A STALE haystack-only wiring_block (as an old persisted row would be) passed alongside
    source files that ALSO contain a LangGraph graph must NOT orphan the scanned LangGraph nodes:
    the safety-net re-detects and merges the missing framework's edges. Reproduces Gate D."""
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
