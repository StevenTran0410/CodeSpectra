"""Consolidated discovery/harvest tests (agent-flow, agent-flow routes, agent-knowledge,
expansion, field-downstream-consumers, prompt-resolver/site-scan, contract-harvest)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import yaml
from httpx import ASGITransport, AsyncClient

from agent_eval_harness.discovery.agent_knowledge import (
    AgentKnowledge,
    Citation,
    ConsumerRef,
    ContextBuilderRef,
    FailureModeRef,
    verify_citations,
)
from agent_eval_harness.discovery.client import CodeSpectraClient
from agent_eval_harness.discovery.expansion import expand_candidate
from agent_eval_harness.discovery.prompt_resolver import (
    build_module_constants,
    resolve_constant,
    resolve_import_site,
)
from agent_eval_harness.discovery.prompt_site_scan import scan_for_prompt_sites
from agent_eval_harness.discovery.wiring import detect_wiring_block_static
from agent_eval_harness.llm.client import LLMMessage, LLMResponse
from agent_eval_harness.llm.fake_client import FakeLLMClient
from agent_eval_harness.llm.proxy_client import CodeSpectraProxyClient
from agent_eval_harness.mapping.agent_flow import (
    AgentFlow,
    AgentFlowMap,
    _build_structural_agents,
    build_source_by_component,
    load_agent_flow_map,
    save_agent_flow_map,
    separate_agent_flows,
)
from agent_eval_harness.mapping.builder.contract_harvest import (
    _parse_files,
    _resolve_class_schema,
    _SchemaResolveCtx,
    harvest_field_downstream_consumers,
)
from agent_eval_harness.mapping.builder.pipeline import SystemMapBuilder
from agent_eval_harness.mapping.system_map import Component, SystemMap, save_system_map
from agent_eval_harness.store import repository
from agent_eval_harness.store.database import close_db, init_db
from agent_eval_harness.ui.server import app
from tests._stubs import FakeCodeSpectraClient as _StubClient

_REPO_ROOT = Path(__file__).parent.parent


# === agent_flow: mapping/agent_flow.py -- the holistic LLM-2 agent-flow separation pass ===


def _map_with(
    *component_ids: str,
    constructs: dict[str, list[str]] | None = None,
    downstream: dict[str, list[str]] | None = None,
    call_downstream: dict[str, list[str]] | None = None,
    makes_model_call: dict[str, bool | None] | None = None,
    entry_kind: dict[str, str] | None = None,
) -> SystemMap:
    """`constructs` marks real constructor ownership (the only thing parent_agent derives from);
    downstream/call_downstream/makes_model_call/entry_kind drive the flow-construction rule."""
    return SystemMap(
        target_system_id="test_system",
        components=[
            Component(
                id=cid, role="unknown", entry_point=f"mod:{cid}", file=f"{cid}.py",
                constructor_downstream=(constructs or {}).get(cid, []),
                downstream=(downstream or {}).get(cid, []),
                call_downstream=(call_downstream or {}).get(cid, []),
                makes_model_call=(makes_model_call or {}).get(cid),
                entry_kind=(entry_kind or {}).get(cid, "class"),
            )
            for cid in component_ids
        ],
    )


async def _harvest_system_map(target: str, extra_dirs: list[str] | None = None) -> SystemMap:
    """Runs the real static scanner+builder over a test_targets fixture -- proves the
    construction rule against genuine scan output, not a hand-authored synthetic map."""
    files = sorted((_REPO_ROOT / "test_targets" / target).glob("*.py"))
    for extra in extra_dirs or []:
        files += sorted((_REPO_ROOT / "test_targets" / extra).glob("*.py"))
    files = sorted(set(files))
    file_contents = {str(f): f.read_text(encoding="utf-8") for f in files}
    wb = detect_wiring_block_static(file_contents)
    builder = SystemMapBuilder(
        FakeLLMClient(LLMResponse(content="[]", model="fake")),
        framework=(wb.framework if wb else None),
    )
    system_map, _ = await builder.build_from_files(
        files, package_root=_REPO_ROOT, target_system_id=target,
        wiring_block=wb, retrieval_client=None, snapshot_id=None,
    )
    return system_map


async def test_separate_agent_flows_structural_agents_labeled_by_matching_llm_group() -> None:
    # orchestrator genuinely constructs validator, so parent_agent is derivable; nothing
    # constructs orchestrator, so it stays a root. rag_tool is orchestrator's own helper.
    system_map = _map_with(
        "orchestrator", "rag_tool", "validator", "writer",
        constructs={"orchestrator": ["validator"]},
        downstream={"orchestrator": ["rag_tool", "validator"], "validator": ["writer"]},
        makes_model_call={
            "orchestrator": True, "rag_tool": False, "validator": True, "writer": True,
        },
        entry_kind={"rag_tool": "function"},
    )

    llm_response = LLMResponse(
        content=json.dumps({
            "agents": [
                {
                    "id": "orchestrator", "label": "Orchestrator", "role": "orchestrator",
                    "summary": "Breaks down intent and routes to RAG.",
                },
                {
                    "id": "validator", "label": "Validator", "role": "validator",
                    "summary": "Validates retrieved context.",
                },
            ],
        }),
        model="fake-test",
    )
    llm_client = FakeLLMClient(llm_response)

    result = await separate_agent_flows(system_map, {}, llm_client)

    assert isinstance(result, AgentFlowMap)
    assert {a.id for a in result.agents} == {"orchestrator", "validator", "writer"}
    assert result.unassigned_component_ids == []

    orchestrator = next(a for a in result.agents if a.id == "orchestrator")
    assert orchestrator.component_ids == ["orchestrator", "rag_tool"]  # helper folded in
    assert orchestrator.downstream_agents == ["validator"]
    assert orchestrator.label == "Orchestrator"
    assert orchestrator.summary == "Breaks down intent and routes to RAG."
    assert orchestrator.parent_agent is None

    # Derived from constructor_downstream, NOT from the LLM (which conflated "feeds me" with
    # "owns me" and invented a hierarchy over a connect()-wired DAG).
    validator = next(a for a in result.agents if a.id == "validator")
    assert validator.upstream_agents == ["orchestrator"]
    assert validator.downstream_agents == ["writer"]
    assert validator.parent_agent == "orchestrator"
    assert validator.label == "Validator"

    # writer is only fed by validator, never constructed by it — a data edge is not ownership —
    # and has no matching LLM group, so it defaults to its own component id as a label.
    writer = next(a for a in result.agents if a.id == "writer")
    assert writer.upstream_agents == ["validator"]
    assert writer.parent_agent is None
    assert writer.label == "writer"

    assert result.entry_agent_ids == ["orchestrator"]  # the only agent with no upstream_agents


async def test_separate_agent_flows_malformed_json_agents_still_structural_no_crash() -> None:
    system_map = _map_with(
        "a", "b", "c",
        downstream={"a": ["b"]},
        makes_model_call={"a": True, "b": False, "c": None},
        entry_kind={"b": "function"},
    )
    llm_client = FakeLLMClient(LLMResponse(content="not json at all {{{", model="fake-test"))

    result = await separate_agent_flows(system_map, {}, llm_client)

    # LLM failure costs labels only — "a" is still the sole agent (makes_model_call is True),
    # with "b" folded in as its helper; "c" is unreached by any agent, so it's excluded.
    assert {a.id for a in result.agents} == {"a"}
    agent_a = result.agents[0]
    assert agent_a.component_ids == ["a", "b"]
    assert agent_a.label == "a"  # no LLM label available -- defaults to its own id
    assert result.unassigned_component_ids == ["c"]
    assert result.entry_agent_ids == ["a"]


async def test_separate_agent_flows_empty_llm_agents_list_still_builds_structural_agents() -> None:
    system_map = _map_with("a", "b", makes_model_call={"a": True, "b": None})
    llm_client = FakeLLMClient(
        LLMResponse(content=json.dumps({"agents": []}), model="fake-test")
    )

    result = await separate_agent_flows(system_map, {}, llm_client)

    assert {a.id for a in result.agents} == {"a"}
    assert result.unassigned_component_ids == ["b"]  # "b" is never reached from "a"


async def test_separate_agent_flows_duplicate_and_non_agent_llm_ids_ignored_for_labels() -> None:
    system_map = _map_with(
        "a", "b",
        makes_model_call={"a": True, "b": False},
        downstream={"a": ["b"]},
        entry_kind={"b": "function"},
    )

    llm_response = LLMResponse(
        content=json.dumps({
            "agents": [
                {"id": "a", "label": "First"},
                {"id": "a", "label": "Second"},  # duplicate id -> ignored, first wins
                {"id": "b", "label": "Not An Agent"},  # b never makes a model call -> no agent
            ],
        }),
        model="fake-test",
    )
    llm_client = FakeLLMClient(llm_response)

    result = await separate_agent_flows(system_map, {}, llm_client)

    assert {a.id for a in result.agents} == {"a"}
    agent_a = result.agents[0]
    assert agent_a.label == "First"
    assert agent_a.component_ids == ["a", "b"]  # b folds in as a's helper regardless of the LLM


async def test_separate_agent_flows_prunes_unreached_but_keeps_folded_and_leaf_components() -> None:
    system_map = _map_with(
        "agent_true", "folded_false", "leaf_service", "orphan_none", "second_agent",
        downstream={"agent_true": ["folded_false", "leaf_service"]},
        makes_model_call={
            "agent_true": True, "folded_false": False, "leaf_service": False,
            "orphan_none": None, "second_agent": True,
        },
        entry_kind={"folded_false": "function", "leaf_service": "class"},
    )

    llm_client = FakeLLMClient(LLMResponse(content=json.dumps({"agents": []}), model="fake-test"))
    result = await separate_agent_flows(system_map, {}, llm_client)

    assert {a.id for a in result.agents} == {"agent_true", "second_agent"}
    agent_true = next(a for a in result.agents if a.id == "agent_true")
    assert agent_true.component_ids == ["agent_true", "folded_false"]
    assert agent_true.boundary_component_ids == ["leaf_service"]
    # never reached from any agent -- excluded from the flow entirely, no hidden-flag needed
    assert result.unassigned_component_ids == ["orphan_none"]
    # every agent id + its folded helpers + its boundary leaves, orphan excluded
    assert result.flow_component_ids == [
        "agent_true", "folded_false", "leaf_service", "second_agent",
    ]


def test_build_structural_agents_downstream_agents_from_topology_not_call_closure() -> None:
    """Regression: downstream_agents must come from Component.downstream (pipeline topology),
    not from walking call_downstream."""
    system_map = _map_with(
        "agent_a", "helper", "agent_b",
        call_downstream={"agent_a": ["helper"]},
        downstream={"agent_a": ["helper", "agent_b"]},
        makes_model_call={"agent_a": True, "helper": False, "agent_b": True},
        entry_kind={"helper": "function"},
    )

    agents, _reached = _build_structural_agents(system_map, {})

    agent_a = next(a for a in agents if a.id == "agent_a")
    assert agent_a.component_ids == ["agent_a", "helper"]  # helper still folds via call_downstream
    assert agent_a.downstream_agents == ["agent_b"]  # edge comes from topology, not call_downstream
    agent_b = next(a for a in agents if a.id == "agent_b")
    assert agent_b.upstream_agents == ["agent_a"]


def test_build_structural_agents_generic_construction_rule() -> None:
    """Deterministic proof of the generic construction rule: a function helper folds
    transitively, a class service is a leaf, a component reachable only via a non-agent is
    excluded, and a downstream agent stays its own top-level node."""
    system_map = _map_with(
        "agent_a", "helper_fn", "deep_helper", "service_x", "service_internal",
        "agent_b", "orphan_alone",
        downstream={
            "agent_a": ["helper_fn", "service_x", "agent_b"],
            "helper_fn": ["deep_helper"],
            "service_x": ["service_internal"],
        },
        makes_model_call={
            "agent_a": True, "helper_fn": False, "deep_helper": None,
            "service_x": False, "service_internal": False,
            "agent_b": True, "orphan_alone": False,
        },
        entry_kind={
            "helper_fn": "function", "deep_helper": "bound_method",
            "service_x": "class", "service_internal": "function",
            "orphan_alone": "function",
        },
    )

    agents, reached = _build_structural_agents(system_map, {})

    assert {a.id for a in agents} == {"agent_a", "agent_b"}
    agent_a = next(a for a in agents if a.id == "agent_a")

    # a function/bound_method helper folds in, transitively through the helper chain
    assert agent_a.component_ids == ["agent_a", "helper_fn", "deep_helper"]

    # a class-kind service is a single boundary leaf -- its own downstream is never walked
    assert agent_a.boundary_component_ids == ["service_x"]
    assert "service_internal" not in agent_a.component_ids
    assert "service_internal" not in agent_a.boundary_component_ids
    assert "service_internal" not in reached

    # another agent found downstream stays its own top-level node, never folded, with an edge
    assert "agent_b" not in agent_a.component_ids
    assert "agent_b" not in agent_a.boundary_component_ids
    assert agent_a.downstream_agents == ["agent_b"]
    agent_b = next(a for a in agents if a.id == "agent_b")
    assert agent_b.upstream_agents == ["agent_a"]

    # reachable only through a non-agent (or not reachable at all) -- excluded
    assert "orphan_alone" not in reached


async def test_multi_agent_fixture_flow_is_clean_no_orphan_top_level_nodes() -> None:
    """Guardrail 4: on the real multi_agent+linear_rag scan, top-level nodes are exactly the
    model-calling agents; no orphan node is top-level."""
    system_map = await _harvest_system_map("multi_agent", extra_dirs=["linear_rag"])
    agents, reached = _build_structural_agents(system_map, {})

    agent_ids = {a.id for a in agents}
    assert agent_ids == {c.id for c in system_map.components if c.makes_model_call is True}

    planner = next(a for a in agents if a.id == "planner")
    assert "worker" in planner.boundary_component_ids
    assert "case_law_search" not in reached
    assert "decoy_lookup" not in reached

    excluded = {c.id for c in system_map.components if c.id not in reached}
    assert excluded & agent_ids == set()
    for cid in excluded:
        assert system_map.component_by_id(cid).makes_model_call is not True


async def test_linear_rag_fixture_flow_is_clean_retriever_never_top_level_orphan() -> None:
    """Guardrail 4: the deterministic retriever sits only upstream of the writer agent, so it's
    pruned rather than left as a top-level orphan."""
    system_map = await _harvest_system_map("linear_rag")
    agents, reached = _build_structural_agents(system_map, {})

    assert {a.id for a in agents} == {"writer"}
    assert "retriever" not in reached
    assert "retriever" not in {a.id for a in agents}


async def test_separate_agent_flows_no_components_short_circuits_without_llm_call() -> None:
    system_map = SystemMap(target_system_id="empty_system", components=[])

    class NeverCallClient:
        async def complete(self, *args, **kwargs):
            raise AssertionError("Should not call LLM when there are no components")

    result = await separate_agent_flows(system_map, {}, NeverCallClient())

    assert result.agents == []
    assert result.unassigned_component_ids == []


def test_build_source_by_component_recovers_snippet_via_rescan(tmp_path: Path) -> None:
    source_file = tmp_path / "widget.py"
    source_file.write_text(
        "from haystack import component\n\n"
        "@component\n"
        "class WidgetComponent:\n"
        "    \"\"\"Does widget things.\"\"\"\n"
        "    def run(self):\n"
        "        return 'ok'\n"
    )
    system_map = SystemMap(
        target_system_id="widget_system",
        components=[
            Component(
                id="widget",
                role="unknown",
                entry_point="widget:WidgetComponent",
                file="widget.py",
            )
        ],
    )

    result = build_source_by_component([source_file], system_map)

    assert "widget" in result
    # source_snippet is the class BODY only, not the "class X:" line itself
    assert "Does widget things" in result["widget"]


def test_build_source_by_component_recovers_langgraph_snippet_in_mixed_map(tmp_path: Path) -> None:
    """CS-316: a langgraph component in a 'haystack+langgraph' map must have its snippet
    re-derived via the union scan, not dropped by the old get_scanner(framework) fallback."""
    source_file = tmp_path / "agent.py"
    source_file.write_text(
        "from langgraph.graph import StateGraph\n\n"
        "class Agent:\n"
        "    def build(self):\n"
        "        graph = StateGraph(dict)\n"
        "        graph.add_node('investigate', self._node_investigate)\n"
        "        return graph\n"
        "    def _node_investigate(self, state):\n"
        "        return 'DISTINCTIVE_BODY_TOKEN'\n"
    )
    system_map = SystemMap(
        target_system_id="mixed_system",
        framework="haystack+langgraph",
        components=[
            Component(
                id="_node_investigate",
                role="unknown",
                entry_point="agent:Agent._node_investigate",
                file="",
            )
        ],
    )

    result = build_source_by_component([source_file], system_map)

    assert result["_node_investigate"], "union scan should re-derive the langgraph snippet"
    assert "DISTINCTIVE_BODY_TOKEN" in result["_node_investigate"]


def test_build_source_by_component_falls_back_gracefully_when_unmatched() -> None:
    """A component the re-scan can't find must degrade to an empty/fallback string, never raise."""
    system_map = SystemMap(
        target_system_id="ghost_system",
        components=[
            Component(
                id="ghost",
                role="unknown",
                entry_point="ghost_module:GhostComponent",
                file="ghost.py",
            )
        ],
    )

    result = build_source_by_component([], system_map)

    assert result == {"ghost": ""}


def test_agent_flow_map_yaml_roundtrip(tmp_path: Path) -> None:
    original = AgentFlowMap(
        target_system_id="roundtrip_system",
        agents=[],
        entry_agent_ids=[],
        unassigned_component_ids=["a", "b"],
    )
    path = tmp_path / "map_agentflows.yaml"

    save_agent_flow_map(original, path)
    loaded = load_agent_flow_map(path)

    assert loaded == original


# === agent_flow_routes: route-level tests for the agent-flow endpoints (POST/GET .../agent-flows) ===

CANNED_GROUPING = json.dumps({
    "agents": [
        {
            "id": "widget",
            "label": "Widget Agent",
            "role": "orchestrator",
            "summary": "Does widget things.",
            "component_ids": ["widget"],
            "upstream_agents": [],
            "downstream_agents": [],
            "parent_agent": None,
        }
    ],
    "entry_agent_ids": ["widget"],
    "unassigned_component_ids": [],
})


# NOTE: these two fixtures were module-wide `autouse=True` in the original standalone file.
# Scoped to explicit `@pytest.mark.usefixtures(...)` on this section's tests only, so they
# don't leak DB re-init / client monkeypatching onto unrelated tests merged into this file.
@pytest.fixture
async def _setup_db(tmp_path, monkeypatch):
    monkeypatch.setenv("AEH_DATA_DIR", str(tmp_path))
    await init_db()
    yield
    await close_db()


@pytest.fixture
def _patch_external_calls(monkeypatch, tmp_path):
    """Both external boundaries (snapshot lookup + LLM-2 call) are faked; everything else runs for real."""

    async def fake_get_snapshot(self, snapshot_id: str) -> dict:
        return {"local_path": str(tmp_path)}

    async def fake_complete(self, messages, *, max_tokens=512, temperature=0.2, json_mode=False, **kwargs):
        return LLMResponse(content=CANNED_GROUPING, model="fake-llm-2")

    monkeypatch.setattr(CodeSpectraClient, "get_snapshot", fake_get_snapshot)
    monkeypatch.setattr(CodeSpectraProxyClient, "complete", fake_complete)


def _write_widget_source(tmp_path: Path) -> None:
    (tmp_path / "widget.py").write_text(
        "from haystack import component\n\n"
        "@component\n"
        "class WidgetComponent:\n"
        "    \"\"\"Does widget things.\"\"\"\n"
        "    def run(self):\n"
        "        return 'ok'\n"
    )


async def _seed_completed_expansion_session(tmp_path: Path, session_id: str = "sess-1") -> str:
    await repository.insert_expansion_session(session_id, "cand-1", "snap-1")

    system_map = SystemMap(
        target_system_id="widget_system",
        components=[
            Component(
                id="widget",
                role="unknown",
                entry_point="widget:WidgetComponent",
                file="widget.py",
                makes_model_call=True,  # agent-hood is structural now, not left to the canned LLM grouping
            )
        ],
    )
    map_path = tmp_path / f"{session_id}.yaml"
    save_system_map(system_map, map_path)

    await repository.finish_expansion_session(
        session_id,
        "completed",
        map_path=str(map_path),
        accepted=["widget.py"],
        boundary=[],
        stop_reason="frontier_exhausted",
    )
    return session_id


async def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.mark.usefixtures("_setup_db", "_patch_external_calls")
async def test_generate_agent_flows_success_persists_and_returns_map(tmp_path: Path) -> None:
    _write_widget_source(tmp_path)
    session_id = await _seed_completed_expansion_session(tmp_path)

    async with await _client() as client:
        resp = await client.post(
            f"/api/discovery/expansion-sessions/{session_id}/agent-flows",
            json={
                "provider_id": "prov-1",
                "model_id": "strong-model",
                "backend_url": "http://fake-backend",
                "backend_token": "tok",
            },
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["target_system_id"] == "widget_system"
    assert [a["id"] for a in body["agents"]] == ["widget"]
    assert body["agents"][0]["component_ids"] == ["widget"]
    assert body["unassigned_component_ids"] == []

    # Persisted: sibling YAML written + DB pointer updated.
    sess = await repository.get_expansion_session(session_id)
    assert sess is not None
    assert sess["agent_flows_path"] is not None
    agent_flows_path = Path(sess["agent_flows_path"])
    assert agent_flows_path.exists()
    assert agent_flows_path.name == f"{session_id}_agentflows.yaml"
    on_disk = yaml.safe_load(agent_flows_path.read_text(encoding="utf-8"))
    assert on_disk["target_system_id"] == "widget_system"


@pytest.mark.usefixtures("_setup_db", "_patch_external_calls")
async def test_get_agent_flows_returns_404_before_generation(tmp_path: Path) -> None:
    session_id = await _seed_completed_expansion_session(tmp_path)

    async with await _client() as client:
        resp = await client.get(f"/api/discovery/expansion-sessions/{session_id}/agent-flows")

    assert resp.status_code == 404


@pytest.mark.usefixtures("_setup_db", "_patch_external_calls")
async def test_get_agent_flows_returns_saved_map_after_generation(tmp_path: Path) -> None:
    _write_widget_source(tmp_path)
    session_id = await _seed_completed_expansion_session(tmp_path)

    async with await _client() as client:
        post_resp = await client.post(
            f"/api/discovery/expansion-sessions/{session_id}/agent-flows",
            json={"provider_id": "prov-1", "backend_url": "http://fake-backend", "backend_token": "tok"},
        )
        assert post_resp.status_code == 200

        get_resp = await client.get(f"/api/discovery/expansion-sessions/{session_id}/agent-flows")

    assert get_resp.status_code == 200
    assert get_resp.json() == post_resp.json()


@pytest.mark.usefixtures("_setup_db", "_patch_external_calls")
async def test_generate_agent_flows_404_for_unknown_session() -> None:
    async with await _client() as client:
        resp = await client.post(
            "/api/discovery/expansion-sessions/does-not-exist/agent-flows",
            json={"provider_id": "prov-1", "backend_url": "http://fake-backend", "backend_token": "tok"},
        )
    assert resp.status_code == 404


@pytest.mark.usefixtures("_setup_db", "_patch_external_calls")
async def test_generate_agent_flows_400_when_expansion_not_completed() -> None:
    session_id = "sess-running"
    await repository.insert_expansion_session(session_id, "cand-1", "snap-1")  # status='running'

    async with await _client() as client:
        resp = await client.post(
            f"/api/discovery/expansion-sessions/{session_id}/agent-flows",
            json={"provider_id": "prov-1", "backend_url": "http://fake-backend", "backend_token": "tok"},
        )
    assert resp.status_code == 400


@pytest.mark.usefixtures("_setup_db", "_patch_external_calls")
async def test_generate_agent_flows_missing_backend_config_returns_400_not_500(
    tmp_path: Path,
) -> None:
    session_id = await _seed_completed_expansion_session(tmp_path)

    async with await _client() as client:
        resp = await client.post(
            f"/api/discovery/expansion-sessions/{session_id}/agent-flows",
            json={"provider_id": "prov-1"},  # no backend_url/backend_token, no .aeh/config.yaml
        )
    assert resp.status_code == 400


@pytest.mark.usefixtures("_setup_db", "_patch_external_calls")
async def test_generate_agent_flows_missing_local_path_returns_400_not_500(
    tmp_path: Path, monkeypatch
) -> None:
    """Regression guard: an inner HTTPException must not get re-wrapped into a 500."""
    session_id = await _seed_completed_expansion_session(tmp_path)

    async def fake_get_snapshot_missing_local_path(self, snapshot_id: str) -> dict:
        return {}  # no "local_path" key

    monkeypatch.setattr(CodeSpectraClient, "get_snapshot", fake_get_snapshot_missing_local_path)

    async with await _client() as client:
        resp = await client.post(
            f"/api/discovery/expansion-sessions/{session_id}/agent-flows",
            json={"provider_id": "prov-1", "backend_url": "http://fake-backend", "backend_token": "tok"},
        )
    assert resp.status_code == 400
    assert "local_path" in resp.json()["detail"]


# === agent_knowledge: tests for agent_knowledge schema hardening ===


@pytest.mark.anyio
async def test_verify_citations_all_five_sources(tmp_path: Path) -> None:
    """Phantom detection across all 5 semantic sources."""
    knowledge = AgentKnowledge(
        functionality="Test functionality",
        functionality_citations=[
            Citation(file="phantom.py", line=10, symbol="phantom_func")
        ],
        context_builders=[
            ContextBuilderRef(name="ctx_builder", file="phantom.py", line=20, builds_kwarg="ctx")
        ],
        upstream_consumers=[
            ConsumerRef(name="upstream_svc", file="phantom.py", line=30)
        ],
        downstream_consumers=[
            ConsumerRef(name="downstream_svc", file="phantom.py", line=40)
        ],
        failure_modes=[
            FailureModeRef(description="Fails when timeout", file="phantom.py", line=50)
        ],
    )

    report = verify_citations(knowledge, tmp_path)

    assert len(report.claims) == 5
    assert all(c.status == "phantom" for c in report.claims), \
        f"Expected all phantom, got: {[c.status for c in report.claims]}"

    assert len(knowledge.needs_human) == 5
    assert all("Phantom citation" in item for item in knowledge.needs_human)


@pytest.mark.anyio
async def test_verify_citations_union_resolves_offline_and_promptsite(tmp_path: Path) -> None:
    """A citation is valid if it resolves to readable context via ANY of three routes; only a citation that resolves to nothing is flagged."""
    from agent_eval_harness.discovery.agent_knowledge import PromptSiteRef

    src = tmp_path / "agent.py"
    src.write_text(
        "import os\n"                       # 1
        "from .prompts import SYS_PROMPT\n"  # 2  (prompt-site line)
        "\n"                                  # 3
        "@component\n"                        # 4  (decorator, one above the class)
        "class MyAgent:\n"                    # 5  (class def)
        "    def run(self):\n"                # 6
        "        return SYS_PROMPT\n",        # 7  (mid-class, class name NOT on this line)
        encoding="utf-8",
    )
    symbols = {"agent.py": [{"name": "MyAgent", "kind": "class", "line_start": 5, "line_end": 7}]}

    knowledge = AgentKnowledge(
        functionality="x",
        prompt_sites=[PromptSiteRef(file="agent.py", line=2, kind="prompt_import", snippet="")],
        functionality_citations=[
            Citation(file="agent.py", line=4, symbol="MyAgent"),   # near-above span -> resolves
            Citation(file="agent.py", line=7, symbol="MyAgent"),   # mid-class span  -> resolves
            Citation(file="agent.py", line=2, symbol="prompt_import"),  # prompt-site -> resolves
            Citation(file="agent.py", line=1, symbol="ghost"),     # nothing there   -> flagged
        ],
    )

    report = verify_citations(knowledge, tmp_path, symbols_by_file=symbols)

    by_line = {c.citation.line: c.status for c in report.claims}
    assert by_line[4] == "verified"
    assert by_line[7] == "verified"
    assert by_line[2] == "verified"
    assert by_line[1] == "unverified"
    assert knowledge.needs_human == ["Unverified citation: agent.py:1:ghost"]


@pytest.mark.anyio
async def test_static_wins_llm_cannot_override() -> None:
    """Static-wins cross-check — LLM cannot override structural fields."""
    from dataclasses import dataclass

    from agent_eval_harness.discovery.enrichment import _enrich_single_agent
    from agent_eval_harness.mapping.agent_flow import AgentFlow, AgentFlowMap
    from agent_eval_harness.store.database import get_db, init_db

    try:
        get_db()
    except RuntimeError:
        await init_db()

    class _StubLLMClient:
        """Stub LLM that returns an attempt to override location."""
        async def complete(self, messages, *, max_tokens=512, temperature=0.2, json_mode=False, reasoning_effort=None):
            from agent_eval_harness.llm.client import LLMResponse
            return LLMResponse(content=json.dumps({
                "functionality": "LLM purpose",
                "location": {
                    "file": "llm_lie.py",
                    "line_start": 0,
                    "line_end": 0,
                    "entry_method": "fake",
                    "entry_line": 0,
                },
            }), model="stub")

    flow_map = AgentFlowMap(
        target_system_id="test_system",
        agents=[AgentFlow(id="test_agent", label="Test Agent", component_ids=[])]
    )
    system_map = SystemMap(target_system_id="test", components=[])
    evidence = {
        'prompt_sites_by_file': {},
        'component_by_agent': {"test_agent": []},
        'edges_by_agent': {"test_agent": []},
        'source_coverage': {"test_agent": 0.0},
    }

    @dataclass
    class _EnrichCtx:
        session_id: str = "test_session"
        snapshot_id: str = ""
        agent_flow_map: AgentFlowMap = None
        system_map: SystemMap = None
        accepted_with_annotations: list = None
        accepted_edges: list = None
        client: object = None
        llm_client: object = None
        depth: str = "normal"
        force_agent_ids: list = None
        semaphore: object = None
        repo_root: object = None
        system_type: str | None = None

        def __post_init__(self):
            if self.agent_flow_map is None:
                self.agent_flow_map = flow_map
            if self.system_map is None:
                self.system_map = system_map
            if self.accepted_with_annotations is None:
                self.accepted_with_annotations = []
            if self.accepted_edges is None:
                self.accepted_edges = []
            if self.force_agent_ids is None:
                self.force_agent_ids = []
            if self.llm_client is None:
                self.llm_client = _StubLLMClient()
            if self.semaphore is None:
                class _Semaphore:
                    async def __aenter__(self): return self
                    async def __aexit__(self, *args): pass
                self.semaphore = _Semaphore()

    depth_cap = {"queries": 3, "llm_calls": 2, "read_file": 2}

    knowledge = await _enrich_single_agent(
        "test_agent",
        evidence,
        _EnrichCtx(),
        depth_cap,
        [],
    )

    assert knowledge.evidence_hash
    assert knowledge.generated_at
    assert knowledge.query_count == 0
    # LLM tried to set location to llm_lie.py — static-wins must reject it
    assert knowledge.location is None or knowledge.location.file != "llm_lie.py"


@pytest.mark.anyio
async def test_migration_v21_agent_knowledge_table_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Migration v21: agent_knowledge table exists after init_db()."""
    monkeypatch.setenv("AEH_DATA_DIR", str(tmp_path))

    from agent_eval_harness.store import database as db_module
    from agent_eval_harness.store.database import get_db
    from agent_eval_harness.store.database import init_db as init_db_for_test

    db_module._db = None

    await init_db_for_test()
    db = get_db()

    async with db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='agent_knowledge'"
    ) as cur:
        row = await cur.fetchone()

    assert row is not None, "agent_knowledge table does not exist"

    async with db.execute("PRAGMA table_info(agent_knowledge)") as cur:
        columns_rows = await cur.fetchall()

    column_names = [r[1] for r in columns_rows]
    expected_columns = [
        "session_id", "agent_id", "md_path", "json_path",
        "evidence_hash", "confidence", "query_count", "generated_at"
    ]
    for expected in expected_columns:
        assert expected in column_names, f"Column {expected} not in agent_knowledge table"

    from agent_eval_harness.store.database import close_db
    await close_db()


def test_from_json_backward_compat_str_consumers() -> None:
    """Backward-compat: from_json() degrades gracefully when legacy list[str] consumers loaded."""
    legacy_data = {
        "upstream_consumers": ["plain_string_consumer"],
        "downstream_consumers": ["another_string"],
        "failure_modes": ["string mode"],
    }

    result = AgentKnowledge.from_json(legacy_data)

    assert isinstance(result, AgentKnowledge)
    assert result.degraded is True
    assert "Validation error" in result.degraded_reason


# === expansion: tests for discovery/expansion.py candidate expansion ===


class _StubLLMClient:
    """Duck-typed stand-in matching the real LLMClient.complete() protocol, not a stale shape."""

    def __init__(self, verdicts: dict[str, str]):
        self._verdicts = verdicts
        self.call_count = 0

    async def complete(self, messages: list[LLMMessage], *, json_mode: bool = False, **_kwargs) -> LLMResponse:
        self.call_count += 1
        prompt = "\n".join(m.content for m in messages)
        import re
        ids = re.findall(r"=== ID: (\S+) ===", prompt)

        verdicts_list = []
        for unique_id in ids:
            if "::" in unique_id:
                path, chunk_id = unique_id.split("::", 1)
            else:
                path = unique_id

            verdict = self._verdicts.get(unique_id) or self._verdicts.get(path) or "boundary"
            verdicts_list.append({
                "id": unique_id,
                "verdict": verdict,
                "reason": "stubbed"
            })

        import json
        content = json.dumps({"verdicts": verdicts_list})
        return LLMResponse(content=content, model="stub")


@pytest.mark.anyio
async def test_expansion_golden_flow() -> None:
    # Golden flow: starts at seed file_a (accept) -> neighbors file_b (expand) -> neighbors file_c (boundary)
    files = {
        "file_a.py": "class Agent: pass",
        "file_b.py": "import file_c",
        "file_c.py": "def utility(): pass"
    }
    neighbors = {
        "file_a.py": ["file_b.py"],
        "file_b.py": ["file_c.py"],
    }
    verdicts = {
        "file_a.py": "expand",
        "file_b.py": "expand",
        "file_c.py": "boundary"
    }

    client = _StubClient(files, neighbors)
    llm_client = _StubLLMClient(verdicts)
    candidate = {
        "cluster_files": ["file_a.py"],
        "hub_paths": []
    }

    res = await expand_candidate("snap-123", candidate, client, llm_client, node_budget=5, hop_cap=3)
    accepted_files = [item["file"] for item in res["accepted"]]
    assert accepted_files == ["file_a.py", "file_b.py"]
    assert res["boundary"] == ["file_c.py"]
    assert res["stop_reason"] == "frontier_exhausted"


@pytest.mark.anyio
async def test_expansion_node_budget() -> None:
    # Low node budget: seed file_a (accept) -> neighbor file_b (accept) with budget of 1
    files = {
        "file_a.py": "class Agent: pass",
        "file_b.py": "import file_a"
    }
    neighbors = {
        "file_a.py": ["file_b.py"]
    }
    verdicts = {
        "file_a.py": "expand",
        "file_b.py": "accept"
    }

    client = _StubClient(files, neighbors)
    llm_client = _StubLLMClient(verdicts)
    candidate = {
        "cluster_files": ["file_a.py"],
        "hub_paths": []
    }

    res = await expand_candidate("snap-123", candidate, client, llm_client, node_budget=1, hop_cap=3)
    accepted_files = [item["file"] for item in res["accepted"]]
    assert accepted_files == ["file_a.py"]
    assert res["stop_reason"] == "node_budget"


@pytest.mark.anyio
async def test_expansion_containment_negative_control() -> None:
    # Negative control: seed file_a is boundary, should not traverse its neighbors
    files = {
        "file_a.py": "class Utility: pass",
        "file_b.py": "class Agent: pass"
    }
    neighbors = {
        "file_a.py": ["file_b.py"]
    }
    verdicts = {
        "file_a.py": "boundary",
        "file_b.py": "accept"
    }

    client = _StubClient(files, neighbors)
    llm_client = _StubLLMClient(verdicts)
    candidate = {
        "cluster_files": ["file_a.py"],
        "hub_paths": []
    }

    res = await expand_candidate("snap-123", candidate, client, llm_client, node_budget=10, hop_cap=3)
    assert res["accepted"] == []
    assert res["boundary"] == ["file_a.py"]
    assert res["stop_reason"] == "frontier_exhausted"
    # Neighbors of boundary should never be called
    assert "file_b.py" not in client.read_calls


@pytest.mark.anyio
async def test_expansion_chunk_level_evidence_and_bypass() -> None:
    files = {
        "file_a.py": "class Agent: pass\ndef run(): pass",
        "file_b.py": "def helper(): pass",
    }
    neighbors = {
        "file_a.py": ["file_b.py"]
    }
    verdicts = {
        "file_b.py": "accept"
    }

    client = _StubClient(files, neighbors)
    llm_client = _StubLLMClient(verdicts)

    candidate = {
        "cluster_files": ["file_a.py"],
        "hub_paths": [],
        "evidence": [
            {
                "file": "file_a.py",
                "chunk_id": "Agent",
                "snippet": "class Agent: pass"
            }
        ],
        "wiring_block": {
            "nodes": [
                {"alias": "agent", "class_name": "Agent", "source_hint_file": "file_a.py"}
            ],
            "edges": []
        }
    }

    res = await expand_candidate("snap-123", candidate, client, llm_client, node_budget=5, hop_cap=3)
    accepted_files = [item["file"] for item in res["accepted"]]
    assert "file_a.py" in accepted_files
    assert "file_b.py" in accepted_files
    assert res["stop_reason"] == "frontier_exhausted"


@pytest.mark.anyio
async def test_expansion_respects_excluded_files() -> None:
    files = {
        "file_a.py": "class Agent: pass",
        "file_b.py": "def helper(): pass",
    }
    neighbors = {
        "file_a.py": ["file_b.py"]
    }
    verdicts = {
        "file_b.py": "accept"
    }

    client = _StubClient(files, neighbors)
    llm_client = _StubLLMClient(verdicts)

    candidate = {
        "cluster_files": ["file_a.py"],
        "hub_paths": [],
        "evidence": [
            {
                "file": "file_a.py",
                "chunk_id": "Agent",
                "snippet": "class Agent: pass"
            }
        ],
        "excluded_files": ["file_a.py"]
    }

    res = await expand_candidate("snap-123", candidate, client, llm_client, node_budget=5, hop_cap=3)
    accepted_files = [item["file"] for item in res["accepted"]]
    assert "file_a.py" not in accepted_files
    assert "file_b.py" not in accepted_files
    assert res["stop_reason"] == "frontier_exhausted"


@pytest.mark.anyio
async def test_expansion_captures_accepted_edges() -> None:
    files = {
        "file_a.py": "class Agent: pass",
        "file_b.py": "def helper(): pass",
        "file_c.py": "def database(): pass",
    }
    # Chain: file_a -> file_b -> file_c
    neighbors = {
        "file_a.py": ["file_b.py"],
        "file_b.py": ["file_c.py"],
    }
    verdicts = {
        "file_a.py": "expand",
        "file_b.py": "expand",
        "file_c.py": "accept",
    }

    client = _StubClient(files, neighbors)
    llm_client = _StubLLMClient(verdicts)

    candidate = {
        "cluster_files": ["file_a.py"],
        "hub_paths": [],
        "evidence": [
            {
                "file": "file_a.py",
                "chunk_id": "Agent",
                "snippet": "class Agent: pass"
            }
        ],
    }

    res = await expand_candidate("snap-123", candidate, client, llm_client, node_budget=5, hop_cap=3)
    accepted_files = [item["file"] for item in res["accepted"]]
    assert accepted_files == ["file_a.py", "file_b.py", "file_c.py"]

    edges = res["accepted_edges"]
    assert len(edges) == 2
    assert {"src": "file_a.py", "dst": "file_b.py"} in edges
    assert {"src": "file_b.py", "dst": "file_c.py"} in edges


@pytest.mark.anyio
async def test_expansion_batch_classification_fewer_calls() -> None:
    # 3 files in one frontier level (seed level)
    files = {
        "file_a.py": "class Agent: pass",
        "file_b.py": "class Tools: pass",
        "file_c.py": "class Router: pass",
    }
    neighbors = {}
    verdicts = {
        "file_a.py::None": "accept",
        "file_b.py::None": "accept",
        "file_c.py::None": "expand",
    }

    client = _StubClient(files, neighbors)
    llm_client = _StubLLMClient(verdicts)

    candidate = {
        "cluster_files": ["file_a.py", "file_b.py", "file_c.py"],
        "hub_paths": [],
        "evidence": [],
    }

    # All three files are in seeds, so they are processed in a single batch level
    res = await expand_candidate("snap-123", candidate, client, llm_client, node_budget=5, hop_cap=3)
    accepted_files = sorted([item["file"] for item in res["accepted"]])
    assert accepted_files == ["file_a.py", "file_b.py", "file_c.py"]
    assert llm_client.call_count == 1


@pytest.mark.anyio
async def test_expansion_batch_classification_missing_degrades_to_boundary() -> None:
    files = {
        "file_a.py": "class Agent: pass",
        "file_b.py": "class Tools: pass",
        "file_c.py": "class Router: pass",
    }
    neighbors = {}

    verdicts = {
        "file_a.py::None": "accept",
        "file_c.py::None": "accept",
    }

    client = _StubClient(files, neighbors)

    class OmissionLLMClient:
        def __init__(self):
            self.call_count = 0

        async def complete(self, messages: list[LLMMessage], *, json_mode: bool = False, **_kwargs) -> LLMResponse:
            self.call_count += 1
            import json
            content = json.dumps({
                "verdicts": [
                    {"id": "file_a.py::None", "verdict": "accept", "reason": "stubbed"},
                    {"id": "file_c.py::None", "verdict": "accept", "reason": "stubbed"},
                ]
            })
            return LLMResponse(content=content, model="stub")

    llm_client = OmissionLLMClient()
    candidate = {
        "cluster_files": ["file_a.py", "file_b.py", "file_c.py"],
        "hub_paths": [],
        "evidence": [],
    }

    res = await expand_candidate("snap-123", candidate, client, llm_client, node_budget=5, hop_cap=3)
    accepted_files = [item["file"] for item in res["accepted"]]
    assert "file_a.py" in accepted_files
    assert "file_c.py" in accepted_files
    assert "file_b.py" not in accepted_files
    assert "file_b.py" in res["boundary"]


@pytest.mark.anyio
async def test_split_candidate_seed_accepted_despite_boundary_verdict() -> None:
    """CS-317: a SPLIT candidate's own hub_paths seed is authoritative membership, accepted even
    when the classifier returns 'boundary'."""
    files = {"qa/agent.py": "class QAAgent:\n    def run(self):\n        return 1\n"}
    client = _StubClient(files, {})
    llm_client = _StubLLMClient({"qa/agent.py": "boundary"})  # classifier would reject it
    candidate = {
        "map_scope_framework": "plain_python",  # marks this a split candidate
        "wiring_block": None,
        "cluster_files": ["qa/agent.py"],
        "hub_paths": ["qa/agent.py"],
    }
    res = await expand_candidate("snap", candidate, client, llm_client, node_budget=5, hop_cap=3)
    accepted_files = [item["file"] for item in res["accepted"]]
    assert "qa/agent.py" in accepted_files


@pytest.mark.anyio
async def test_non_split_candidate_seed_still_classified() -> None:
    """Regression guard: a NON-split candidate's 'boundary' verdict still drops its seed --
    seed-skip must not leak to non-split candidates."""
    files = {"x.py": "class Thing:\n    pass\n"}
    client = _StubClient(files, {})
    llm_client = _StubLLMClient({"x.py": "boundary"})
    candidate = {"cluster_files": ["x.py"], "hub_paths": ["x.py"]}  # no map_scope_framework
    res = await expand_candidate("snap", candidate, client, llm_client, node_budget=5, hop_cap=3)
    accepted_files = [item["file"] for item in res["accepted"]]
    assert "x.py" not in accepted_files


# === field_downstream_consumers: static harvest of fan-in field-per-letter consumer shapes (CS-289 WS-A3) ===
# Fixtures mirror the real shape of backend agent_auditor.py + agent_synthesis.py +
# _section_compressor.py generically, without depending on the real backend source.

COMPRESSOR_SRC = '''
_PREVIEW_KEYS = {
    "A": ["purpose", "domain"],
    "B": ["main_layers"],
    "C": ["summary", "folders"],
}

def compress_section(letter, section, char_cap=500):
    keys = _PREVIEW_KEYS.get(letter, [])
    preview = {}
    for key in keys:
        val = section.get(key)
        if val is not None:
            preview[key] = val
    return str(preview)[:char_cap]


def compress_audit(section_k, char_cap=800):
    subset = {
        "overall_confidence": section_k.get("overall_confidence"),
        "notes": section_k.get("notes"),
    }
    return str(subset)[:char_cap]
'''

AGENT_K_SRC = '''
from .compressor import compress_section

def _build_k_input(sections):
    compressed = {}
    for letter in "ABC":
        s = sections.get(letter) or {}
        compressed[letter] = {
            "confidence": s.get("confidence", "medium"),
            "blind_spots": (s.get("blind_spots") or [])[:3],
            "content_preview": compress_section(letter, s, char_cap=500),
        }
    return compressed


class KAgent:
    async def run(self, provider_id: str, model_id: str, all_sections: dict) -> dict:
        return _build_k_input(all_sections)
'''

AGENT_L_SRC = '''
from .compressor import compress_audit, compress_section

def _build_l_input(sections):
    compact = {}
    for letter in "AB":
        s = sections.get(letter) or {}
        compact[letter] = compress_section(letter, s, char_cap=800)
    compact["K"] = compress_audit(sections.get("K") or {}, char_cap=800)
    return compact


class LAgent:
    async def run(self, provider_id: str, model_id: str, all_sections: dict) -> dict:
        return _build_l_input(all_sections)
'''

TWO_PARAM_AGENT_SRC = '''
class TwoParamAgent:
    async def run(self, provider_id: str, model_id: str, all_sections: dict, extra: dict) -> dict:
        return {}
'''

DYNAMIC_LETTERS_AGENT_SRC = '''
def _letters():
    return "XY"

class DynamicAgent:
    async def run(self, provider_id: str, model_id: str, all_sections: dict) -> dict:
        out = {}
        for letter in _letters():
            s = all_sections.get(letter) or {}
            out[letter] = s.get("summary")
        return out
'''


def _write(tmp_path: Path, rel: str, src: str) -> Path:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(src, encoding="utf-8")
    return p


def _component(agent_id: str, class_name: str, file_rel: str) -> Component:
    return Component(id=agent_id, role="unknown", entry_point=f"{file_rel[:-3].replace('/', '.')}:{class_name}", file=file_rel)


def test_direct_and_cross_file_preview_dict_reads_like_auditor(tmp_path: Path) -> None:
    files = [
        _write(tmp_path, "pkg/compressor.py", COMPRESSOR_SRC),
        _write(tmp_path, "pkg/agent_k.py", AGENT_K_SRC),
    ]
    asts = _parse_files(files)
    system_map = SystemMap(target_system_id="t", components=[_component("K", "KAgent", "pkg/agent_k.py")])
    flow = AgentFlowMap(target_system_id="t", agents=[AgentFlow(id="K", component_ids=["K"])])

    by_agent, notes = harvest_field_downstream_consumers(flow, system_map, asts)

    assert notes == []
    assert by_agent["K"] == {
        "A": ["blind_spots", "confidence", "domain", "purpose"],
        "B": ["blind_spots", "confidence", "main_layers"],
        "C": ["blind_spots", "confidence", "folders", "summary"],
    }


def test_literal_letter_and_second_helper_like_synthesizer(tmp_path: Path) -> None:
    files = [
        _write(tmp_path, "pkg/compressor.py", COMPRESSOR_SRC),
        _write(tmp_path, "pkg/agent_l.py", AGENT_L_SRC),
    ]
    asts = _parse_files(files)
    system_map = SystemMap(target_system_id="t", components=[_component("L", "LAgent", "pkg/agent_l.py")])
    flow = AgentFlowMap(target_system_id="t", agents=[AgentFlow(id="L", component_ids=["L"])])

    by_agent, notes = harvest_field_downstream_consumers(flow, system_map, asts)

    assert notes == []
    assert by_agent["L"] == {
        "A": ["domain", "purpose"],
        "B": ["main_layers"],
        "K": ["notes", "overall_confidence"],
    }


def test_agent_with_two_required_params_is_out_of_scope_silently(tmp_path: Path) -> None:
    files = [_write(tmp_path, "pkg/agent_two.py", TWO_PARAM_AGENT_SRC)]
    asts = _parse_files(files)
    system_map = SystemMap(target_system_id="t", components=[_component("TWO", "TwoParamAgent", "pkg/agent_two.py")])
    flow = AgentFlowMap(target_system_id="t", agents=[AgentFlow(id="TWO", component_ids=["TWO"])])

    by_agent, notes = harvest_field_downstream_consumers(flow, system_map, asts)

    assert by_agent == {}
    assert notes == []  # not fan-in shaped -> silently out of scope, not an error


def test_dynamic_letters_source_yields_no_resolvable_fields_and_a_note(tmp_path: Path) -> None:
    files = [_write(tmp_path, "pkg/agent_dyn.py", DYNAMIC_LETTERS_AGENT_SRC)]
    asts = _parse_files(files)
    system_map = SystemMap(target_system_id="t", components=[_component("DYN", "DynamicAgent", "pkg/agent_dyn.py")])
    flow = AgentFlowMap(target_system_id="t", agents=[AgentFlow(id="DYN", component_ids=["DYN"])])

    by_agent, notes = harvest_field_downstream_consumers(flow, system_map, asts)

    assert by_agent == {}
    assert any("DYN" in n and "no statically-resolvable" in n for n in notes)


# === prompt_resolver: prompt_resolver generalizes -- no module-name assumption, no name pattern (CS-300 AC3) ===


def _consts(source: str) -> dict[str, ast.expr]:
    return build_module_constants(ast.parse(source))


class TestResolveConstant:
    def test_plain_constant_resolves_verbatim(self):
        consts = _consts("X_SYSTEM = 'literal'\n")
        assert resolve_constant(consts["X_SYSTEM"], consts) == "literal"

    def test_joinedstr_is_mandatory_not_optional(self):
        """THE MANDATORY test (Judge spec Section 9 AC3): a Constant-only resolver would
        silently drop ~46% of real prompts here."""
        consts = _consts(
            "ROLE = 'planner'\nWORK = 'plan'\nY_SYSTEM = f'You are {ROLE}. Do {WORK}.'\n"
        )
        value = resolve_constant(consts["Y_SYSTEM"], consts)
        assert value
        assert "You are" in value
        assert "{ROLE}" in value  # placeholder retained, not resolved through the f-string
        assert "{WORK}" in value

    def test_augassign_folds_into_consts(self):
        consts = _consts(
            'WRITER_PROMPT = "Write an answer."\nWRITER_PROMPT += " Cite every source."\n'
        )
        assert resolve_constant(consts["WRITER_PROMPT"], consts) == "Write an answer. Cite every source."

    def test_binop_add_concatenates_both_sides(self):
        consts = _consts("A_PROMPT = 'a' + 'b'\n")
        assert resolve_constant(consts["A_PROMPT"], consts) == "ab"

    def test_binop_add_with_unresolvable_side_is_none(self):
        consts = _consts("A_PROMPT = 'a' + get_suffix()\n")
        assert resolve_constant(consts["A_PROMPT"], consts) is None

    def test_name_resolves_through_another_module_constant(self):
        consts = _consts("_BASE = 'base text'\nB_SYSTEM = _BASE\n")
        assert resolve_constant(consts["B_SYSTEM"], consts) == "base text"

    def test_name_cycle_terminates(self):
        """A→B→A must not infinite-loop — the visited set is required, not decorative."""
        consts = _consts("A_PROMPT = B_PROMPT\nB_PROMPT = A_PROMPT\n")
        assert resolve_constant(consts["A_PROMPT"], consts) is None
        assert resolve_constant(consts["B_PROMPT"], consts) is None

    def test_call_never_guessed(self):
        consts = _consts("X_SYSTEM = build_prompt()\n")
        assert resolve_constant(consts["X_SYSTEM"], consts) is None

    def test_none_node_is_none(self):
        assert resolve_constant(None, {}) is None


class TestResolveImportSite:
    def test_absolute_and_relative_imports_resolve_one_entry_per_alias(self, tmp_path: Path):
        """Nguyên tắc số 0: the module is deliberately named 'agent_texts.py', NOT 'prompts.py'."""
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "agent_texts.py").write_text(
            "PLANNER_SYSTEM = 'You are a planning component. Decompose the query into intents.'\n"
            "_RUBRIC = 'Score 1-5.'\n"
            "JUDGE_SYSTEM = f\"You are a critical judge reviewing another component's output. {_RUBRIC}\"\n",
            encoding="utf-8",
        )
        abs_file = pkg / "agent_abs.py"
        abs_file.write_text("from pkg.agent_texts import PLANNER_SYSTEM, JUDGE_SYSTEM\n", encoding="utf-8")
        rel_file = pkg / "agent_rel.py"
        rel_file.write_text("from .agent_texts import PLANNER_SYSTEM\n", encoding="utf-8")

        abs_tree = ast.parse(abs_file.read_text(encoding="utf-8"))
        abs_import = next(n for n in ast.walk(abs_tree) if isinstance(n, ast.ImportFrom))
        abs_resolved = resolve_import_site(abs_import, abs_file, tmp_path)
        assert len(abs_resolved) == 2  # one entry PER ALIAS
        assert abs_resolved["PLANNER_SYSTEM"] == "You are a planning component. Decompose the query into intents."
        assert "critical judge reviewing another component's output" in (abs_resolved["JUDGE_SYSTEM"] or "")

        rel_tree = ast.parse(rel_file.read_text(encoding="utf-8"))
        rel_import = next(n for n in ast.walk(rel_tree) if isinstance(n, ast.ImportFrom))
        rel_resolved = resolve_import_site(rel_import, rel_file, tmp_path)
        assert rel_resolved["PLANNER_SYSTEM"] == "You are a planning component. Decompose the query into intents."

    def test_aliased_import_keyed_by_local_name(self, tmp_path: Path):
        pkg = tmp_path / "pkg2"
        pkg.mkdir()
        (pkg / "texts.py").write_text("FOO_SYSTEM = 'foo'\n", encoding="utf-8")
        importing = pkg / "agent.py"
        importing.write_text("from pkg2.texts import FOO_SYSTEM as BAR_SYSTEM\n", encoding="utf-8")

        tree = ast.parse(importing.read_text(encoding="utf-8"))
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom))
        resolved = resolve_import_site(node, importing, tmp_path)
        assert resolved == {"BAR_SYSTEM": "foo"}

    def test_unresolvable_third_party_import_degrades_to_none_no_exception(self, tmp_path: Path):
        importing_file = tmp_path / "agent.py"
        importing_file.write_text("from some_third_party_sdk import SYSTEM_PROMPT\n", encoding="utf-8")
        tree = ast.parse(importing_file.read_text(encoding="utf-8"))
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom))
        resolved = resolve_import_site(node, importing_file, tmp_path)
        assert resolved == {"SYSTEM_PROMPT": None}


# === prompt_site_scan: both prompt_site_scan.py snippet bugs must die, or AC3 is theatre (CS-300 B4) ===


class TestModuleAssignmentSnippets:
    """Bug 2: _extract_snippet only unwrapped Assign+Constant; JoinedStr and AugAssign both
    silently returned ''."""

    def test_augassign_snippet_is_resolved_not_empty(self, tmp_path: Path):
        (tmp_path / "agent.py").write_text(
            'WRITER_PROMPT = "Write an answer."\nWRITER_PROMPT += " Cite every source."\n',
            encoding="utf-8",
        )
        sites = scan_for_prompt_sites(tmp_path, ["agent.py"])["agent.py"]
        augassign_site = next(s for s in sites if s.line == 2)
        assert augassign_site.snippet == "Write an answer. Cite every source."

    def test_joinedstr_snippet_is_resolved_not_empty(self, tmp_path: Path):
        (tmp_path / "agent.py").write_text(
            "ROLE = 'planner'\nY_SYSTEM = f'You are {ROLE}.'\n", encoding="utf-8"
        )
        sites = scan_for_prompt_sites(tmp_path, ["agent.py"])["agent.py"]
        site = next(s for s in sites if s.kind == "module_assignment")
        assert "You are" in site.snippet
        assert "{ROLE}" in site.snippet


class TestPromptImportSites:
    """Bug 1: the prompt_import branch hardcoded snippet='' unconditionally, and gated on the
    MODULE name containing 'prompt'/'system' -- forbidden by Nguyên tắc số 0."""

    def test_import_from_a_module_not_named_prompts_still_resolves(self, tmp_path: Path):
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "agent_texts.py").write_text(
            "AGENT_K_SYSTEM = 'You are a critical auditor reviewing the outputs of 10 code "
            "analysis agents (sections A-J).'\n",
            encoding="utf-8",
        )
        (pkg / "agent.py").write_text(
            "from pkg.agent_texts import AGENT_K_SYSTEM\n", encoding="utf-8"
        )

        sites = scan_for_prompt_sites(tmp_path, ["pkg/agent.py"])["pkg/agent.py"]
        import_sites = [s for s in sites if s.kind == "prompt_import"]
        assert len(import_sites) == 1
        assert "critical auditor reviewing the outputs of 10 code analysis agents" in import_sites[0].snippet

    def test_one_site_per_matching_alias_non_matching_alias_ignored(self, tmp_path: Path):
        pkg = tmp_path / "pkg2"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "helpers.py").write_text(
            "PLANNER_SYSTEM = 'planner prompt'\nhelper_function = None\n", encoding="utf-8"
        )
        (pkg / "agent.py").write_text(
            "from pkg2.helpers import PLANNER_SYSTEM, helper_function\n", encoding="utf-8"
        )

        sites = scan_for_prompt_sites(tmp_path, ["pkg2/agent.py"])["pkg2/agent.py"]
        import_sites = [s for s in sites if s.kind == "prompt_import"]
        assert len(import_sites) == 1
        assert import_sites[0].snippet == "planner prompt"

    def test_relative_import_resolves(self, tmp_path: Path):
        pkg = tmp_path / "pkg3"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "agent_texts.py").write_text("JUDGE_SYSTEM = 'judge prompt'\n", encoding="utf-8")
        (pkg / "agent.py").write_text("from .agent_texts import JUDGE_SYSTEM\n", encoding="utf-8")

        sites = scan_for_prompt_sites(tmp_path, ["pkg3/agent.py"])["pkg3/agent.py"]
        import_sites = [s for s in sites if s.kind == "prompt_import"]
        assert len(import_sites) == 1
        assert import_sites[0].snippet == "judge prompt"

    def test_unresolvable_import_degrades_to_empty_snippet_not_exception(self, tmp_path: Path):
        (tmp_path / "agent.py").write_text(
            "from some_third_party_sdk import SYSTEM_PROMPT\n", encoding="utf-8"
        )
        sites = scan_for_prompt_sites(tmp_path, ["agent.py"])["agent.py"]
        import_sites = [s for s in sites if s.kind == "prompt_import"]
        assert len(import_sites) == 1
        assert import_sites[0].snippet == ""


class TestSdkCallCorroboration:
    """A name match alone (_matches_sdk_pattern) must never decide sdk_call by itself -- it
    needs a receiver (an attribute call), never a bare name call."""

    def test_bare_name_call_matching_the_pattern_is_not_an_sdk_call(self, tmp_path: Path):
        (tmp_path / "agent.py").write_text(
            'def create(x):\n    return x\n\ncreate("not a prompt, just a coincidental name match")\n',
            encoding="utf-8",
        )
        sites = scan_for_prompt_sites(tmp_path, ["agent.py"])["agent.py"]
        assert [s for s in sites if s.kind == "sdk_call"] == []

    def test_attribute_call_matching_the_pattern_is_an_sdk_call(self, tmp_path: Path):
        (tmp_path / "agent.py").write_text(
            'client.messages.create("You are a helpful assistant.")\n', encoding="utf-8",
        )
        sites = scan_for_prompt_sites(tmp_path, ["agent.py"])["agent.py"]
        sdk_sites = [s for s in sites if s.kind == "sdk_call"]
        assert len(sdk_sites) == 1
        assert sdk_sites[0].snippet == "You are a helpful assistant."


# === contract_harvest_recursive: recursive schema resolution in contract harvest ===

# Nested TypedDict test
NESTED_TYPEDDICT_SRC = '''
from typing import TypedDict

class Address(TypedDict):
    street: str
    city: str
    zip_code: str

class Person(TypedDict):
    name: str
    age: int
    address: Address
'''

# Pydantic test
PYDANTIC_SRC = '''
from pydantic import BaseModel

class Address(BaseModel):
    street: str
    city: str
    zip_code: str | None = None

class Person(BaseModel):
    name: str
    age: int
    address: Address
'''

# Dataclass test
DATACLASS_SRC = '''
from dataclasses import dataclass

@dataclass
class Address:
    street: str
    city: str
    zip_code: str | None = None

@dataclass
class Person:
    name: str
    age: int
    address: Address
'''

# Cycle test - A references B which references A
CYCLE_SRC = '''
from typing import TypedDict

class A(TypedDict):
    name: str
    b_ref: 'B'

class B(TypedDict):
    name: str
    a_ref: 'A'
'''

# List and dict generic test
GENERICS_SRC = '''
from typing import TypedDict

class Item(TypedDict):
    id: int
    name: str

class Container(TypedDict):
    items: list[Item]
    metadata: dict[str, str]
'''


def _write_fixture(tmp_path: Path, name: str, src: str) -> Path:
    p = tmp_path / f"{name}.py"
    p.write_text(src, encoding="utf-8")
    return p


def test_nested_typeddict(tmp_path: Path):
    """Test resolving nested TypedDict with recursive refs."""
    file_path = _write_fixture(tmp_path, "nested_td", NESTED_TYPEDDICT_SRC)
    asts = _parse_files([file_path])
    ctx = _SchemaResolveCtx(asts=asts, files_root=tmp_path, visited=set(), depth=0, conventions=None)

    schema, source = _resolve_class_schema("Person", ctx)
    assert schema is not None
    assert schema["type"] == "object"
    assert "name" in schema["properties"]
    assert "age" in schema["properties"]
    assert "address" in schema["properties"]
    assert schema["properties"]["address"]["type"] == "object"
    assert "street" in schema["properties"]["address"]["properties"]


def test_pydantic_model(tmp_path: Path):
    """Test resolving Pydantic BaseModel with optional fields."""
    file_path = _write_fixture(tmp_path, "pydantic_model", PYDANTIC_SRC)
    asts = _parse_files([file_path])
    ctx = _SchemaResolveCtx(asts=asts, files_root=tmp_path, visited=set(), depth=0, conventions=None)

    schema, source = _resolve_class_schema("Person", ctx)
    assert schema is not None
    assert schema["type"] == "object"
    assert "name" in schema["required"]
    assert "age" in schema["required"]
    assert "address" in schema["properties"]


def test_dataclass_resolution(tmp_path: Path):
    """Test resolving dataclass with optional fields."""
    file_path = _write_fixture(tmp_path, "dataclass_model", DATACLASS_SRC)
    asts = _parse_files([file_path])
    ctx = _SchemaResolveCtx(asts=asts, files_root=tmp_path, visited=set(), depth=0, conventions=None)

    schema, source = _resolve_class_schema("Person", ctx)
    assert schema is not None
    assert schema["type"] == "object"
    assert "name" in schema["properties"]
    assert "address" in schema["properties"]


def test_cycle_detection(tmp_path: Path):
    """Test that cycles are detected and don't cause infinite loops."""
    file_path = _write_fixture(tmp_path, "cycle_test", CYCLE_SRC)
    asts = _parse_files([file_path])
    ctx = _SchemaResolveCtx(asts=asts, files_root=tmp_path, visited=set(), depth=0, conventions=None)

    schema, source = _resolve_class_schema("A", ctx)
    assert schema is not None
    # The cycle should be handled gracefully — A and B resolve but without infinite recursion
    assert schema["type"] == "object"
    assert "name" in schema["properties"]


def test_list_and_dict_generics(tmp_path: Path):
    """Test resolving list[Class] and dict[str, Class] with nested types."""
    file_path = _write_fixture(tmp_path, "generics_test", GENERICS_SRC)
    asts = _parse_files([file_path])
    ctx = _SchemaResolveCtx(asts=asts, files_root=tmp_path, visited=set(), depth=0, conventions=None)

    schema, source = _resolve_class_schema("Container", ctx)
    assert schema is not None
    assert "items" in schema["properties"]
    assert schema["properties"]["items"]["type"] == "array"
    assert "metadata" in schema["properties"]


def test_depth_overflow(tmp_path: Path):
    """Test that depth overflow is handled gracefully."""
    file_path = _write_fixture(tmp_path, "nested_td", NESTED_TYPEDDICT_SRC)
    asts = _parse_files([file_path])
    ctx = _SchemaResolveCtx(asts=asts, files_root=tmp_path, visited=set(), depth=6, conventions=None)

    result = _resolve_class_schema("Person", ctx)
    # Should return None when depth is exceeded
    assert result is None
