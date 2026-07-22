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

from pathlib import Path

import pytest

from agent_eval_harness.llm.client import LLMResponse
from agent_eval_harness.llm.fake_client import FakeLLMClient
from agent_eval_harness.mapping.agent_flow import AgentFlow, AgentFlowMap
from agent_eval_harness.mapping.builder.contract_harvest import harvest_contracts
from agent_eval_harness.mapping.builder.pipeline import SystemMapBuilder
from agent_eval_harness.mapping.builder.scanners import LangGraphScanner

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
