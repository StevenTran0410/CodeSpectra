"""LLM-2 pass: groups an already-built SystemMap's components into agents (vs. classify_roles(),
which judges one component at a time with no sibling/topology context)."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from agent_eval_harness.discovery.expansion import extract_symbol_snippet
from agent_eval_harness.llm.client import LLMClient, LLMMessage
from agent_eval_harness.mapping.builder.prompts import AGENT_FLOW_SYSTEM
from agent_eval_harness.mapping.builder.roles import VALID_ROLES
from agent_eval_harness.mapping.builder.scanners import HaystackScanner

from .system_map import Component, SystemMap

logger = logging.getLogger("agent_eval_harness.mapping.agent_flow")


class AgentFlow(BaseModel):
    id: str
    role: str = "unknown"
    label: str = ""
    summary: str = ""
    component_ids: list[str] = Field(default_factory=list)
    upstream_agents: list[str] = Field(default_factory=list)
    downstream_agents: list[str] = Field(default_factory=list)
    parent_agent: str | None = None


class AgentFlowMap(BaseModel):
    target_system_id: str
    agents: list[AgentFlow] = Field(default_factory=list)
    entry_agent_ids: list[str] = Field(default_factory=list)
    unassigned_component_ids: list[str] = Field(default_factory=list)


def load_agent_flow_map(path: str | Path) -> AgentFlowMap:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return AgentFlowMap.model_validate(data)


def save_agent_flow_map(agent_flow_map: AgentFlowMap, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.dump(agent_flow_map.model_dump(), f, default_flow_style=False, sort_keys=False)


def build_source_by_component(files: list[Path], system_map: SystemMap) -> dict[str, str]:
    """Re-derive a source snippet per component — the persisted map has no snippet field."""
    candidates = HaystackScanner().scan(files)
    snippet_by_candidate_id = {c.candidate_id: c.source_snippet for c in candidates}

    result: dict[str, str] = {}
    for component in system_map.components:
        snippet = snippet_by_candidate_id.get(component.id, "")
        if not snippet:
            snippet = _fallback_snippet(component, files)
        result[component.id] = snippet
    return result


def _fallback_snippet(component: Component, files: list[Path]) -> str:
    """Targeted file read for a component the re-scan didn't find. Never raises."""
    if not component.file or not component.entry_point:
        return ""
    match = next((f for f in files if f.as_posix().endswith(component.file)), None)
    if match is None:
        return ""
    try:
        content = match.read_text(encoding="utf-8")
    except OSError:
        return ""
    _, _, class_name = component.entry_point.partition(":")
    if not class_name:
        return ""
    return extract_symbol_snippet(content, class_name)


def _build_user_prompt(system_map: SystemMap, source_by_component: dict[str, str]) -> str:
    lines = [f"Target system: {system_map.target_system_id}", ""]
    for component in system_map.components:
        constraints = (
            ", ".join(f"{c.name}={c.value}" for c in component.constraints) or "none"
        )
        snippet = source_by_component.get(component.id) or "(no source available)"
        lines.append(f"### component: {component.id}")
        lines.append(f"role_hint: {component.role}")
        lines.append(f"file: {component.file}")
        lines.append(f"entry_point: {component.entry_point}")
        lines.append(f"upstream: {component.upstream}")
        lines.append(f"downstream: {component.downstream}")
        lines.append(f"constraints: {constraints}")
        lines.append("source:")
        lines.append(snippet)
        lines.append("")
    return "\n".join(lines)


async def separate_agent_flows(
    system_map: SystemMap,
    source_by_component: dict[str, str],
    llm_client: LLMClient,
) -> AgentFlowMap:
    """One holistic LLM call; a malformed/empty response degrades to everything unassigned
    rather than raising, since components only leave `remaining` once validly claimed."""
    remaining = {c.id for c in system_map.components}
    if not remaining:
        return AgentFlowMap(target_system_id=system_map.target_system_id)

    parsed: dict = {}
    try:
        response = await llm_client.complete(
            [
                LLMMessage(role="system", content=AGENT_FLOW_SYSTEM),
                LLMMessage(
                    role="user",
                    content=_build_user_prompt(system_map, source_by_component),
                ),
            ],
            max_tokens=4096,
            json_mode=True,
        )
        loaded = json.loads(response.content)
        if isinstance(loaded, dict):
            parsed = loaded
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning(f"Agent-flow separation: unparseable LLM response: {e}")

    raw_agents = parsed.get("agents", [])
    if not isinstance(raw_agents, list):
        raw_agents = []

    agents: list[AgentFlow] = []
    seen_agent_ids: set[str] = set()

    for raw in raw_agents:
        if not isinstance(raw, dict):
            continue
        agent_id = raw.get("id")
        if not isinstance(agent_id, str) or not agent_id or agent_id in seen_agent_ids:
            continue

        claimed = raw.get("component_ids", [])
        if not isinstance(claimed, list):
            claimed = []
        owned = [cid for cid in claimed if isinstance(cid, str) and cid in remaining]
        if not owned:
            continue  # nothing left to claim -> not a usable agent grouping

        remaining.difference_update(owned)
        seen_agent_ids.add(agent_id)

        role = raw.get("role")
        if not isinstance(role, str) or role not in VALID_ROLES:
            role = "unknown"

        label = raw.get("label")
        summary = raw.get("summary")
        parent = raw.get("parent_agent")

        agents.append(
            AgentFlow(
                id=agent_id,
                role=role,
                label=label if isinstance(label, str) and label else agent_id,
                summary=summary if isinstance(summary, str) else "",
                component_ids=owned,
                upstream_agents=[
                    a for a in raw.get("upstream_agents", []) if isinstance(a, str)
                ],
                downstream_agents=[
                    a for a in raw.get("downstream_agents", []) if isinstance(a, str)
                ],
                parent_agent=parent if isinstance(parent, str) else None,
            )
        )

    # drop dangling/self agent-id references so the frontend tree never has to guard for them
    for agent in agents:
        agent.upstream_agents = [
            a for a in agent.upstream_agents if a in seen_agent_ids and a != agent.id
        ]
        agent.downstream_agents = [
            a for a in agent.downstream_agents if a in seen_agent_ids and a != agent.id
        ]
        if agent.parent_agent not in seen_agent_ids or agent.parent_agent == agent.id:
            agent.parent_agent = None

    raw_entry_ids = parsed.get("entry_agent_ids", [])
    if not isinstance(raw_entry_ids, list):
        raw_entry_ids = []
    entry_agent_ids = [a for a in raw_entry_ids if isinstance(a, str) and a in seen_agent_ids]
    if not entry_agent_ids:
        entry_agent_ids = [a.id for a in agents if a.parent_agent is None]

    return AgentFlowMap(
        target_system_id=system_map.target_system_id,
        agents=agents,
        entry_agent_ids=entry_agent_ids,
        unassigned_component_ids=sorted(remaining),
    )
