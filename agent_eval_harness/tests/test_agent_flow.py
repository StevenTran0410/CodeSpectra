"""Unit tests for mapping/agent_flow.py — the holistic LLM-2 agent-flow separation pass."""
from __future__ import annotations

import json
from pathlib import Path

from agent_eval_harness.llm.client import LLMResponse
from agent_eval_harness.llm.fake_client import FakeLLMClient
from agent_eval_harness.mapping.agent_flow import (
    AgentFlowMap,
    build_source_by_component,
    load_agent_flow_map,
    save_agent_flow_map,
    separate_agent_flows,
)
from agent_eval_harness.mapping.system_map import Component, SystemMap


def _map_with(*component_ids: str, constructs: dict[str, list[str]] | None = None) -> SystemMap:
    """`constructs` marks real constructor ownership — the only thing parent_agent is derived from."""
    return SystemMap(
        target_system_id="test_system",
        components=[
            Component(
                id=cid, role="unknown", entry_point=f"mod:{cid}", file=f"{cid}.py",
                constructor_downstream=(constructs or {}).get(cid, []),
            )
            for cid in component_ids
        ],
    )


async def test_separate_agent_flows_happy_path_groups_and_covers_every_component() -> None:
    # orchestrator genuinely constructs validator, so parent_agent is derivable; nothing
    # constructs orchestrator, so it stays a root.
    system_map = _map_with(
        "orchestrator", "rag_tool", "validator", "writer",
        constructs={"orchestrator": ["validator"]},
    )

    llm_response = LLMResponse(
        content=json.dumps({
            "agents": [
                {
                    "id": "orchestrator",
                    "label": "Orchestrator",
                    "role": "orchestrator",
                    "summary": "Breaks down intent and routes to RAG.",
                    "component_ids": ["orchestrator", "rag_tool"],
                    "upstream_agents": [],
                    "downstream_agents": ["validator"],
                    "parent_agent": None,
                },
                {
                    "id": "validator",
                    "label": "Validator",
                    "role": "validator",
                    "summary": "Validates retrieved context.",
                    "component_ids": ["validator"],
                    "upstream_agents": ["orchestrator"],
                    "downstream_agents": ["writer"],
                    "parent_agent": "orchestrator",
                },
                {
                    "id": "writer",
                    "label": "Writer",
                    "role": "writer",
                    "summary": "Writes the final answer.",
                    "component_ids": ["writer"],
                    "upstream_agents": ["validator"],
                    "downstream_agents": [],
                    "parent_agent": "validator",
                },
            ],
            "entry_agent_ids": ["orchestrator"],
            "unassigned_component_ids": [],
        }),
        model="fake-test",
    )
    llm_client = FakeLLMClient(llm_response)

    result = await separate_agent_flows(system_map, {}, llm_client)

    assert isinstance(result, AgentFlowMap)
    assert {a.id for a in result.agents} == {"orchestrator", "validator", "writer"}
    assert result.entry_agent_ids == ["orchestrator"]
    assert result.unassigned_component_ids == []

    orchestrator = next(a for a in result.agents if a.id == "orchestrator")
    assert set(orchestrator.component_ids) == {"orchestrator", "rag_tool"}
    assert orchestrator.parent_agent is None

    # Derived from constructor_downstream, NOT from the LLM (which conflated "feeds me" with
    # "owns me" and invented a hierarchy over a connect()-wired DAG).
    validator = next(a for a in result.agents if a.id == "validator")
    assert validator.parent_agent == "orchestrator"

    # writer is only fed by validator, never constructed by it — a data edge is not ownership.
    writer = next(a for a in result.agents if a.id == "writer")
    assert writer.parent_agent is None

    # Every input component id ends up in exactly one agent's component_ids.
    all_claimed = [cid for a in result.agents for cid in a.component_ids]
    assert sorted(all_claimed) == sorted(c.id for c in system_map.components)
    assert len(all_claimed) == len(set(all_claimed))


async def test_separate_agent_flows_malformed_json_everything_unassigned_no_crash() -> None:
    system_map = _map_with("a", "b", "c")
    llm_client = FakeLLMClient(LLMResponse(content="not json at all {{{", model="fake-test"))

    result = await separate_agent_flows(system_map, {}, llm_client)

    assert result.agents == []
    assert sorted(result.unassigned_component_ids) == ["a", "b", "c"]
    assert result.entry_agent_ids == []


async def test_separate_agent_flows_empty_agents_list_everything_unassigned() -> None:
    system_map = _map_with("a", "b")
    llm_client = FakeLLMClient(
        LLMResponse(content=json.dumps({"agents": []}), model="fake-test")
    )

    result = await separate_agent_flows(system_map, {}, llm_client)

    assert result.agents == []
    assert sorted(result.unassigned_component_ids) == ["a", "b"]


async def test_separate_agent_flows_drops_dangling_and_duplicate_and_self_references() -> None:
    system_map = _map_with("a", "b")

    llm_response = LLMResponse(
        content=json.dumps({
            "agents": [
                {
                    "id": "agent_a",
                    "role": "orchestrator",
                    "component_ids": ["a"],
                    "upstream_agents": ["does_not_exist", "agent_a"],
                    "downstream_agents": ["agent_a"],
                    "parent_agent": "agent_a",  # self-reference -> must become None
                },
                {
                    # Duplicate id -> second occurrence must be ignored entirely.
                    "id": "agent_a",
                    "component_ids": ["b"],
                },
                {
                    "id": "agent_b",
                    "role": "not_a_real_role",
                    "component_ids": ["b"],
                    "parent_agent": "nonexistent_parent",
                },
            ],
            "entry_agent_ids": ["agent_a", "nonexistent"],
        }),
        model="fake-test",
    )
    llm_client = FakeLLMClient(llm_response)

    result = await separate_agent_flows(system_map, {}, llm_client)

    assert {a.id for a in result.agents} == {"agent_a", "agent_b"}
    agent_a = next(a for a in result.agents if a.id == "agent_a")
    agent_b = next(a for a in result.agents if a.id == "agent_b")

    # duplicate agent_a entry is ignored outright, leaving "b" free for agent_b to claim
    assert agent_a.component_ids == ["a"]
    assert agent_b.component_ids == ["b"]
    assert agent_b.role == "unknown"  # invalid role forced to unknown

    assert agent_a.upstream_agents == []  # dangling + self-ref both dropped
    assert agent_a.downstream_agents == []  # self-ref dropped
    assert agent_a.parent_agent is None  # self-reference dropped
    assert agent_b.parent_agent is None  # nonexistent parent dropped

    assert result.entry_agent_ids == ["agent_a"]  # dangling entry id dropped
    assert result.unassigned_component_ids == []


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
