"""CS-312 dogfood — ISOLATION PROOF ONLY.

Proves LangGraphScanner + topology + contract harvest work on the REAL
backend/domain/qa/deep_research.py (a 1003-line bound-method LangGraph agent, merged at 628e22b),
not a hand-written fixture. Pure AST — no target import, no DB/LLM.

⚠️ THIS DOES NOT EXERCISE THE APP PIPELINE. It calls LangGraphScanner().scan(...) /
_build_from_candidates(...) DIRECTLY on deep_research.py in isolation. In the running app,
deep_research.py lands inside a haystack-labelled discovery cluster (ui/server.py -> a single
get_scanner('haystack')), so the app still produces ZERO LangGraph components for it. A green run
here is NOT proof the app maps deep_research — that gap is CS-316 (mixed-cluster dispatch)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_eval_harness.discovery.agent_knowledge import AgentKnowledge
from agent_eval_harness.discovery.enrichment import agent_knowledge_dir, enrich_agents
from agent_eval_harness.llm.client import LLMResponse
from agent_eval_harness.llm.fake_client import FakeLLMClient
from agent_eval_harness.mapping.agent_flow import AgentFlow, AgentFlowMap
from agent_eval_harness.mapping.builder.contract_harvest import harvest_contracts
from agent_eval_harness.mapping.builder.pipeline import SystemMapBuilder
from agent_eval_harness.mapping.builder.scanners import LangGraphScanner
from agent_eval_harness.planning.agentic_planner import generate_plan_agentic

_REPO_ROOT = Path(__file__).parent.parent.parent  # tests -> agent_eval_harness -> <repo root>
_DEEP_RESEARCH = _REPO_ROOT / "backend" / "domain" / "qa" / "deep_research.py"


@pytest.mark.skipif(not _DEEP_RESEARCH.exists(), reason="deep_research.py not on this checkout")
async def test_deep_research_scan_topology_and_harvest():
    files = [_DEEP_RESEARCH]
    candidates = LangGraphScanner().scan(files)

    node_names = {c.class_name for c in candidates}
    assert node_names == {
        "_node_load_context", "_node_plan", "_node_trace_forward", "_node_trace_backward",
        "_node_impact", "_node_retrieve", "_node_analyze_step", "_node_synthesize",
    }
    assert all(c.owner_class_name == "DeepResearchAgent" for c in candidates)
    assert all(c.entry_kind == "bound_method" for c in candidates)

    package_root = _REPO_ROOT / "backend"
    builder = SystemMapBuilder(FakeLLMClient(LLMResponse(content="[]", model="fake")))
    system_map, _summary = await builder._build_from_candidates(
        candidates, files, package_root, "deep_research", None
    )
    plan = next(c for c in system_map.components if c.entry_point.endswith("_node_plan"))
    assert len(plan.downstream) >= 4, "the 2 add_conditional_edges (5 destinations each) must appear"

    agent_flow_map = AgentFlowMap(
        target_system_id="deep_research",
        agents=[AgentFlow(id=c.id, component_ids=[c.id]) for c in system_map.components],
    )
    contracts = harvest_contracts(system_map, agent_flow_map, files, package_root)
    assert contracts, "harvest empty on deep_research.py — F5 gate or owner_class resolution broke"
    load_ctx = next(v for k, v in contracts.items() if "load_context" in k)
    assert load_ctx.invocation is not None
    assert load_ctx.invocation.method == "_node_load_context"


@pytest.mark.skipif(not _DEEP_RESEARCH.exists(), reason="deep_research.py not on this checkout")
async def test_deep_research_full_enrich_sidecar_has_distinct_correct_schemas(tmp_path, monkeypatch):
    """CS-326 real-path integration gate: runs the FULL enrich_agents DAG (deterministic
    FakeLLMClient, no paid provider, no snapshot_id -> the DB-backed symbol-index enrichment path is
    inert, exactly like the isolation-proof test above) over the real deep_research.py, persists to
    a temp session dir, and asserts on the RELOADED sidecar's evaluation_contract -- the §2.0
    Stage-2 harvest, which does not require a snapshot/retrieval client -- not suite-count.

    node_plan and node_synthesize are both LLM-calling siblings on the SAME DeepResearchAgent class
    -- this is the exact shape of the historical bug (node_plan wrongly inherited
    _SYNTHESIZE_META_SCHEMA, which belongs to node_synthesize, via a class-wide ast.walk). §2.2's
    prompt-vs-AST cross-check (AgentKnowledge.output_contract) additionally requires a
    symbol-index-capable retrieval client to populate, so it is covered separately, at the AST/prompt
    level, in test_enrichment.py::test_cross_validate_output_schema_on_real_deep_research_nodes."""
    monkeypatch.setenv("AEH_DATA_DIR", str(tmp_path))
    # Defensive re-init: an earlier test file may have closed the shared DB without reopening it
    # (see test_enrichment.py's _ensure_db fixture for the same precedent).
    from agent_eval_harness.store.database import get_db, init_db
    try:
        get_db()
    except RuntimeError:
        await init_db()

    files = [_DEEP_RESEARCH]
    candidates = LangGraphScanner().scan(files)
    package_root = _REPO_ROOT / "backend"
    builder = SystemMapBuilder(FakeLLMClient(LLMResponse(content="[]", model="fake")))
    system_map, _summary = await builder._build_from_candidates(
        candidates, files, package_root, "deep_research", None
    )
    agent_flow_map = AgentFlowMap(
        target_system_id="deep_research",
        agents=[AgentFlow(id=c.id, component_ids=[c.id]) for c in system_map.components],
    )
    accepted_files = [str(_DEEP_RESEARCH.relative_to(package_root)).replace("\\", "/")]

    session_id = "cs326-deep-research"
    llm_client = FakeLLMClient(LLMResponse(content="{}", model="fake"))
    await enrich_agents(
        session_id=session_id,
        agent_flow_map=agent_flow_map,
        system_map=system_map,
        accepted_with_annotations=accepted_files,
        accepted_edges=[],
        client=None,
        llm_client=llm_client,
        snapshot_id="",
        repo_root=package_root,
    )

    sidecar_dir = agent_knowledge_dir(session_id)
    by_node: dict[str, AgentKnowledge] = {}
    for comp in system_map.components:
        json_path = sidecar_dir / f"{comp.id}.json"
        assert json_path.exists(), f"no sidecar written for {comp.id}"
        suffix = comp.entry_point.rsplit("_node_", 1)[-1]
        by_node[suffix] = AgentKnowledge.from_json(json.loads(json_path.read_text(encoding="utf-8")))

    plan, synthesize, analyze_step = by_node["plan"], by_node["synthesize"], by_node["analyze_step"]

    # evaluation_contract harvested at Stage 2 (§2.0) and persisted on every node's sidecar.
    for node in by_node.values():
        assert node.evaluation_contract is not None

    plan_schema = plan.evaluation_contract.output.json_schema if plan.evaluation_contract.output else None
    synth_schema = synthesize.evaluation_contract.output.json_schema if synthesize.evaluation_contract.output else None
    analyze_schema = (
        analyze_step.evaluation_contract.output.json_schema if analyze_step.evaluation_contract.output else None
    )

    # node_synthesize's own AST constant (_SYNTHESIZE_META_SCHEMA) is referenced directly in its own
    # body -> §2.1's entry-scoped resolution finds it correctly. This harvest path (a schema constant
    # parsed as a raw JSON example, contract_harvest.py's _find_module_str_constant path) yields the
    # fields as the schema's own top-level keys, not a {"properties": {...}} envelope.
    assert synth_schema is not None
    assert set(synth_schema.get("properties", {})) == {"reasoning_chain", "confidence", "summary", "unknowns"}
    synth_source = synthesize.evaluation_contract.output.schema_source or ""
    assert "state mutation delta" in synth_source

    # node_plan and node_analyze_step harvest state mutation deltas (Part D).
    # The core §2.1 regression check: NEITHER wrongly inherits node_synthesize's schema or cites its constant.
    assert plan_schema is not None and "plan" in plan_schema.get("properties", {})
    assert analyze_schema is not None and "step_findings" in analyze_schema.get("properties", {})
    assert "reasoning_chain" not in plan_schema.get("properties", {})
    assert "reasoning_chain" not in analyze_schema.get("properties", {})

    # Round-trips through to_json/from_json.
    reloaded = AgentKnowledge.from_json(json.loads(json.dumps(synthesize.to_json())))
    assert reloaded.evaluation_contract is not None
    assert reloaded.evaluation_contract.output.json_schema == synth_schema

    # Plan-time read: with every agent's sidecar populated, generate_plan_agentic must read the
    # persisted contract, never re-harvest.
    import agent_eval_harness.mapping.builder.contract_harvest as ch_module

    def _must_not_reharvest(*_args, **_kwargs):
        raise AssertionError("harvest_contracts must not run when every sidecar already has a contract")

    monkeypatch.setattr(ch_module, "harvest_contracts", _must_not_reharvest)

    knowledge_by_id = {
        comp.id: json.loads((sidecar_dir / f"{comp.id}.json").read_text(encoding="utf-8"))
        for comp in system_map.components
    }
    _, report = await generate_plan_agentic(
        system_map, agent_flow_map, {c.id: "" for c in system_map.components}, [],
        llm_client, run_critic_pass=False, files=None, files_root=None,
        agent_knowledge_by_id=knowledge_by_id,
    )
    assert {a.agent_id for a in report.agents} == {c.id for c in system_map.components}


@pytest.mark.skipif(not _DEEP_RESEARCH.exists(), reason="deep_research.py not on this checkout")
async def test_deep_research_node_plan_symbol_index_resolution_reaches_evaluation_contract(tmp_path, monkeypatch):
    """CS-326 Job 1 real-path proof: the test above documents that §2.2/§2.3a/§2.5's resolution is
    inert without a snapshot/symbol-index client (snapshot_id=""); THIS test supplies a working one
    (a real search_repo_map, resolving real class/method rows extracted from the actual parsed file)
    so the DB-backed branch in _enrich_single_agent actually runs, and asserts the SAME resolution now
    reaches the PERSISTED, RELOADED evaluation_contract -- not just the legacy AgentKnowledge fields.

    node_plan is the ticket's own headline motivating example: §2.2 recovers its AST-silent output
    schema from its own prompt (_PLAN_SYSTEM); §2.3a-positive resolves its `state: DeepResearchState`
    kwarg to the real TypedDict fields (previously `input_schemas = {}`, so generation invented
    fictional fields); §2.5-salience tags `state` with its real provenance."""
    monkeypatch.setenv("AEH_DATA_DIR", str(tmp_path))
    from agent_eval_harness.store.database import get_db, init_db
    try:
        get_db()
    except RuntimeError:
        await init_db()

    import ast

    from tests._stubs import FakeCodeSpectraClient

    files = [_DEEP_RESEARCH]
    candidates = LangGraphScanner().scan(files)
    package_root = _REPO_ROOT / "backend"
    builder = SystemMapBuilder(FakeLLMClient(LLMResponse(content="[]", model="fake")))
    system_map, _summary = await builder._build_from_candidates(
        candidates, files, package_root, "deep_research", None
    )
    agent_flow_map = AgentFlowMap(
        target_system_id="deep_research",
        agents=[AgentFlow(id=c.id, component_ids=[c.id]) for c in system_map.components],
    )
    node_plan_id = next(c for c in system_map.components if c.entry_point.endswith("_node_plan")).id

    # Real class/method symbol rows, extracted from the actual parsed file (not hand-typed) -- what a
    # real DB-backed symbol index would have stored for this exact source.
    tree = ast.parse(_DEEP_RESEARCH.read_text(encoding="utf-8"))
    cls_node = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "DeepResearchAgent")
    entry_node = next(
        n for n in cls_node.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "_node_plan"
    )
    symbol_responses = {
        "deep_research.py": {"symbols": [
            {"kind": "class", "name": "DeepResearchAgent",
             "line_start": cls_node.lineno, "line_end": cls_node.end_lineno},
            {"kind": "method", "name": "_node_plan", "parent_name": "DeepResearchAgent",
             "line_start": entry_node.lineno, "line_end": entry_node.end_lineno,
             "signature": f"async def _node_plan({ast.unparse(entry_node.args)})"},
        ]},
        "DeepResearchState": {"symbols": [
            {"kind": "class", "name": "DeepResearchState", "rel_path": "domain/qa/deep_research.py"},
        ]},
    }

    class _SymbolIndexClient(FakeCodeSpectraClient):
        async def search_repo_map(self, snapshot_id, q, target_kind=None):
            return symbol_responses.get(q, {"symbols": []})

    accepted_files = [str(_DEEP_RESEARCH.relative_to(package_root)).replace("\\", "/")]
    session_id = "cs326-node-plan-symbol-index"
    llm_client = FakeLLMClient(LLMResponse(content="{}", model="fake"))
    await enrich_agents(
        session_id=session_id,
        agent_flow_map=agent_flow_map,
        system_map=system_map,
        accepted_with_annotations=accepted_files,
        accepted_edges=[],
        client=_SymbolIndexClient(),
        llm_client=llm_client,
        snapshot_id="test-snapshot",
        agent_ids=[node_plan_id],
        repo_root=package_root,
    )

    sidecar_dir = agent_knowledge_dir(session_id)
    json_path = sidecar_dir / f"{node_plan_id}.json"
    assert json_path.exists(), "no sidecar written for node_plan"
    reloaded = AgentKnowledge.from_json(json.loads(json_path.read_text(encoding="utf-8")))

    contract = reloaded.evaluation_contract
    assert contract is not None

    # §2.2: node_plan's AST-derived schema is silent (see the sibling test above); with a working
    # symbol index it is recovered from node_plan's own prompt, ON THE PERSISTED CONTRACT'S OUTPUT.
    assert contract.output is not None and contract.output.json_schema is not None
    assert set(contract.output.json_schema.get("properties", {})) == {"debug", "hop_cap", "plan", "plan_cursor"}
    assert contract.output.schema_discrepancy is not None

    # §2.3a-positive + §2.5-salience: the `state` kwarg's real schema and provenance, on the
    # persisted contract's own invocation.kwargs -- not only the legacy AgentKnowledge.input_schemas.
    assert contract.invocation is not None
    state_kwarg = next(k for k in contract.invocation.kwargs if k.name == "state")
    assert state_kwarg.resolved_schema is not None
    assert "question" in state_kwarg.resolved_schema.get("properties", {})
    assert state_kwarg.source_kind == "runtime-state"
