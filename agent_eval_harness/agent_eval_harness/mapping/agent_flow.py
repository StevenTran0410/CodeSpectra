"""LLM-2 pass: groups an already-built SystemMap's components into agents (vs. classify_roles(),
which judges one component at a time with no sibling/topology context)."""
from __future__ import annotations

import ast
import json
import logging
from collections import deque
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from agent_eval_harness.discovery.expansion import extract_symbol_snippet
from agent_eval_harness.llm.client import LLMClient, LLMMessage
from agent_eval_harness.mapping.builder.prompts import AGENT_FLOW_SYSTEM
from agent_eval_harness.mapping.builder.roles import VALID_ROLES
from agent_eval_harness.mapping.builder.scanners import scan_all
from agent_eval_harness.mapping.builder.types import _SNIPPET_CAP_BY_KIND, find_symbol_node

from .system_map import Component, SystemMap

logger = logging.getLogger("agent_eval_harness.mapping.agent_flow")

_AGENT_FLOW_MAX_TOKENS = 4096


class AgentFlow(BaseModel):
    id: str
    role: str = "unknown"
    label: str = ""
    summary: str = ""
    component_ids: list[str] = Field(default_factory=list)
    # Class-kind services reached directly downstream of this agent: single boundary leaves,
    # never expanded past (never folded into component_ids, never a top-level node either).
    boundary_component_ids: list[str] = Field(default_factory=list)
    upstream_agents: list[str] = Field(default_factory=list)
    downstream_agents: list[str] = Field(default_factory=list)
    parent_agent: str | None = None


class AgentFlowMap(BaseModel):
    target_system_id: str
    agents: list[AgentFlow] = Field(default_factory=list)
    entry_agent_ids: list[str] = Field(default_factory=list)
    unassigned_component_ids: list[str] = Field(default_factory=list)
    # Union of every agent's component_ids + boundary_component_ids -- the structural kept-set a
    # consumer (e.g. the Full Graph view) can filter a file-level view down to, without recomputing it.
    flow_component_ids: list[str] = Field(default_factory=list)


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
    # Union scan: a mixed-framework map (e.g. "haystack+langgraph") must re-derive snippets from
    # every scanner, not just the one named in system_map.framework, or non-primary-framework
    # components silently fall back to _fallback_snippet.
    candidates, _framework_label = scan_all(files)
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
    except OSError as exc:
        logger.debug("could not read %s for fallback snippet: %s", match, exc)
        return ""
    _, _, class_name = component.entry_point.partition(":")
    if not class_name:
        return ""
    return extract_symbol_snippet(content, class_name)


def _opens_on_symbol_line(first_line: str, member: str) -> bool:
    """Post-condition: the extracted span must open on the symbol's own def/class line — catches a
    resolver that anchored on the wrong node instead of silently shipping mismatched evidence."""
    stripped = first_line.lstrip()
    return (
        stripped.startswith(f"def {member}")
        or stripped.startswith(f"async def {member}")
        or stripped.startswith(f"class {member}")
    )


def _full_symbol_sources(system_map: SystemMap, files: list[Path]) -> dict[str, str]:
    """Each component's OWN symbol source with no per-METHOD line cap — the old shared snippet
    helper truncated at a prompt-budget limit, which silently hid the completion call in longer
    agent methods. A CLASS-kind symbol (e.g. an injected-service component resolved to its whole
    defining class) still gets the same by-kind cap as the scanner path — an entire heavyweight
    collaborator class is not "an agent method that might hide a completion call past line 30".
    Whole-file scanning is not an option: systems that define every node in one module would mark
    deterministic nodes as model-driven."""
    by_suffix: dict[str, Path] = {}
    for f in files:
        by_suffix[f.as_posix()] = f
    out: dict[str, str] = {}
    parsed: dict[Path, tuple[ast.AST, list[str]] | None] = {}
    for component in system_map.components:
        symbol = (component.entry_point or "").partition(":")[2]
        if not component.file or not symbol:
            continue
        path = next((p for s, p in by_suffix.items() if s.endswith(component.file)), None)
        if path is None:
            continue
        if path not in parsed:
            try:
                text = path.read_text(encoding="utf-8")
                parsed[path] = (ast.parse(text), text.splitlines())
            except (OSError, SyntaxError) as e:
                logger.debug("agent_flow: could not read/parse %s for model evidence: %s", path, e)
                parsed[path] = None
        entry = parsed[path]
        if entry is None:
            continue
        tree, lines = entry
        names = symbol.split(".")
        node = find_symbol_node(tree, names)
        if node is None:
            continue
        end = getattr(node, "end_lineno", None) or node.lineno
        class_cap = _SNIPPET_CAP_BY_KIND.get("ClassDef")
        if isinstance(node, ast.ClassDef) and class_cap:
            end = min(end, node.lineno - 1 + class_cap)
        span = lines[node.lineno - 1:end]
        if not span or not _opens_on_symbol_line(span[0], names[-1]):
            logger.debug(
                "agent_flow: resolved span for %s does not open on its own def/class line — skipping",
                symbol,
            )
            continue
        out[component.id] = "\n".join(span)
    return out


async def _out_of_scope_fallback_sources(
    system_map: SystemMap, retrieval_client: Any, snapshot_id: str, already: dict[str, str],
) -> dict[str, str]:
    """Secondary path for a component marked out_of_scope: fetch its source through the retrieval
    client instead of leaving it with no evidence at all — never the primary answer to the scope
    gap, and never a raw disk read behind expansion's back."""
    out: dict[str, str] = {}
    for component in system_map.components:
        if not component.out_of_scope or already.get(component.id) or not component.file:
            continue
        symbol = (component.entry_point or "").partition(":")[2]
        if not symbol:
            continue
        try:
            resp = await retrieval_client.read_file(snapshot_id, component.file)
        except Exception as exc:
            logger.debug("agent_flow: out-of-scope read_file fallback failed for %s: %s", component.file, exc)
            continue
        content = resp.get("content") if isinstance(resp, dict) else None
        if content:
            out[component.id] = extract_symbol_snippet(content, symbol)
    return out


def _promote_model_driven_orphans(
    system_map: SystemMap, remaining: set[str]
) -> tuple[list[AgentFlow], set[str]]:
    """Recover a model-calling component the LLM left unassigned. The demotion pass only REMOVES, so a
    real agent dropped by the grouping call is lost forever otherwise — observed live on a retrieval
    node whose body only delegates to a helper, so the truncated snippet the LLM reads shows no
    completion call. Internal helpers are excluded by the call-target test: a gray call target of some
    component is that component's helper, never an agent of its own. Only an explicit `True` verdict
    promotes -- `None` (undecided) is not evidence, same precedent as roles.py's admissible_roles."""
    by_id = {c.id: c for c in system_map.components}
    call_targets = {t for c in system_map.components for t in (c.call_downstream or [])}
    promoted: list[AgentFlow] = []
    claimed: set[str] = set()
    for cid in sorted(remaining):
        component = by_id.get(cid)
        if component is None or cid in call_targets or cid in claimed:
            continue
        if component.makes_model_call is not True:
            continue
        owned = [cid] + [
            t for t in (component.call_downstream or [])
            if t in remaining and t not in claimed and t != cid
        ]
        claimed.update(owned)
        promoted.append(AgentFlow(id=cid, label=cid, component_ids=owned))
        logger.info(
            f"agent_flow: promoting {cid!r} to an agent — it makes a model call but the grouping "
            f"call left it unassigned ({len(owned)} component(s))"
        )
    return promoted, claimed


def _dispatching_owner(
    dissolved: AgentFlow, kept: list[AgentFlow], by_id: dict[str, Component]
) -> AgentFlow | None:
    """The kept agent a dissolved step belongs inside: first a component that routes to it via a
    conditional branch (the dispatcher literally chose it), then the only agent feeding it. None when
    ownership is genuinely ambiguous — an unassigned step is honest, a wrong parent is not."""
    heads = [cid for cid in dissolved.component_ids if cid in by_id]
    for signal in ("conditional_downstream", "downstream"):
        owners = {
            a.id for a in kept for own in a.component_ids
            if own in by_id and any(h in (getattr(by_id[own], signal) or []) for h in heads)
        }
        if len(owners) == 1:
            owner_id = next(iter(owners))
            return next((a for a in kept if a.id == owner_id), None)
    feeders = {
        a.id for a in kept for h in heads
        for up in (by_id[h].upstream or []) if up in a.component_ids
    }
    if len(feeders) != 1:
        return None
    feeder_id = next(iter(feeders))
    return next((a for a in kept if a.id == feeder_id), None)


def _demote_deterministic_agents(
    agents: list[AgentFlow], system_map: SystemMap
) -> tuple[list[AgentFlow], set[str]]:
    """Dissolve any "agent" confirmed deterministic (every member's makes_model_call is False -- a
    pure data/graph step is a component of whoever dispatches it, never an agent of its own). `True`
    on any member keeps the group; `None` (undecided) subtracts nothing, same precedent as roles.py's
    "unknown is not evidence of absence" -- a group with no `False` member and no `True` one is left
    exactly as the LLM grouped it. Returns the kept agents and the components that lost their agent.
    An empty kept-set is a logged diagnosis, not a reason to undo every demotion."""
    by_id = {c.id: c for c in system_map.components}
    kept, dissolved = [], []
    for agent in agents:
        verdicts = [by_id[cid].makes_model_call for cid in agent.component_ids if cid in by_id]
        if any(v is True for v in verdicts):
            kept.append(agent)
        elif any(v is False for v in verdicts):
            dissolved.append(agent)
        else:
            kept.append(agent)

    if not kept:
        logger.info("agent_flow: every group was demoted as deterministic -- map has no agents left")

    kept_ids = {a.id for a in kept}
    orphaned: set[str] = set()
    for agent in dissolved:
        logger.info(
            f"agent_flow: demoting {agent.id!r} ({len(agent.component_ids)} component(s)) — "
            "no model call in it or its resolved call targets, so it is a deterministic step, not an agent"
        )
        # Re-home under the agent that DISPATCHES it (a conditional branch source owns its targets),
        # else the single agent feeding it; only a genuinely unowned step is left unassigned.
        owner = _dispatching_owner(agent, kept, by_id)
        if owner is not None:
            owner.component_ids = owner.component_ids + [
                cid for cid in agent.component_ids if cid not in owner.component_ids
            ]
        else:
            orphaned.update(agent.component_ids)

    for agent in kept:
        agent.upstream_agents = [a for a in agent.upstream_agents if a in kept_ids]
        agent.downstream_agents = [a for a in agent.downstream_agents if a in kept_ids]
    return kept, orphaned


def _build_structural_agents(
    system_map: SystemMap, labels_by_id: dict[str, dict[str, Any]],
) -> tuple[list[AgentFlow], set[str]]:
    """The construction rule: a top-level node is any component with makes_model_call is True.
    Expansion from it never follows a non-agent's downstream -- a function/bound_method helper
    folds into component_ids and its own downstream is walked in turn, a class-kind service becomes
    a single boundary leaf with its downstream never walked, and another agent found this way stays
    its own top-level node, never folded. Purely structural (makes_model_call + the call closure) --
    labels_by_id (the grouping LLM's raw output, keyed by its head component id) only supplies
    label/summary/role text, never membership. Agent-to-agent edges are a separate signal: agents are
    wired through pipeline topology (Component.downstream), never by calling each other directly, so
    downstream_agents is read off the agent's own `downstream`, not the call closure above. Returns
    the agents plus every component id reached from some agent, so the caller can prune the rest as
    true orphans."""
    by_id = {c.id: c for c in system_map.components}
    agent_ids = [c.id for c in system_map.components if c.makes_model_call is True]

    agents: dict[str, AgentFlow] = {}
    reached: set[str] = set(agent_ids)

    for agent_id in agent_ids:
        component_ids = [agent_id]
        boundary_component_ids: list[str] = []
        seen = {agent_id}
        queue: deque[str] = deque([agent_id])
        while queue:
            current = by_id.get(queue.popleft())
            if current is None:
                continue
            for target_id in (current.call_downstream or current.downstream):
                target = by_id.get(target_id)
                if target is None or target_id in seen:
                    continue
                seen.add(target_id)
                if target.makes_model_call is True:
                    continue  # another agent found via the call closure stays its own top-level node
                elif target.entry_kind in ("function", "bound_method"):
                    component_ids.append(target_id)
                    queue.append(target_id)
                else:
                    boundary_component_ids.append(target_id)
        reached.update(seen)

        agent_component = by_id[agent_id]
        downstream_agents = sorted({
            target_id for target_id in agent_component.downstream
            if target_id != agent_id
            and by_id.get(target_id) is not None
            and by_id[target_id].makes_model_call is True
        })

        raw = labels_by_id.get(agent_id, {})
        label = raw.get("label")
        summary = raw.get("summary")
        role = raw.get("role")
        agents[agent_id] = AgentFlow(
            id=agent_id,
            role=role if isinstance(role, str) and role in VALID_ROLES else "unknown",
            label=label if isinstance(label, str) and label else agent_id,
            summary=summary if isinstance(summary, str) else "",
            component_ids=component_ids,
            boundary_component_ids=boundary_component_ids,
            downstream_agents=downstream_agents,
        )

    for agent in agents.values():
        for downstream_id in agent.downstream_agents:
            target_agent = agents.get(downstream_id)
            if target_agent is not None and agent.id not in target_agent.upstream_agents:
                target_agent.upstream_agents.append(agent.id)
    for agent in agents.values():
        agent.upstream_agents.sort()

    return list(agents.values()), reached


def _derive_parent_agents(agents: list[AgentFlow], system_map: SystemMap) -> None:
    """parent = "the agent that CONSTRUCTS me", read off constructor_downstream — never asked of
    the LLM, which conflated "who feeds me" with "who owns me" and invented a hierarchy over a
    connect()-wired DAG where nothing constructs anything."""
    agent_of_component = {cid: a.id for a in agents for cid in a.component_ids}
    for agent in agents:
        owners = {
            agent_of_component[c.id]
            for c in system_map.components
            for owned_id in (c.constructor_downstream or [])
            if owned_id in agent.component_ids
            and c.id in agent_of_component
            and agent_of_component[c.id] != agent.id
        }
        agent.parent_agent = owners.pop() if len(owners) == 1 else None


def _build_user_prompt(
    system_map: SystemMap,
    evidence_sources: dict[str, str],
    show_model_call_verdict: bool = False,
) -> str:
    """`evidence_sources` is the uncapped per-component source (was the pre-capped
    `source_by_component`, which silently shipped truncated/misaligned snippets) — the feature-map
    summary moved out of here into the cached system-prompt prefix, see separate_agent_flows."""
    lines = [f"Target system: {system_map.target_system_id}", ""]
    for component in system_map.components:
        constraints = (
            ", ".join(f"{c.name}={c.value}" for c in component.constraints) or "none"
        )
        snippet = evidence_sources.get(component.id) or "(no source available)"
        lines.append(f"### component: {component.id}")
        lines.append(
            f"structural_facts: fan_in={len(component.upstream)}, fan_out={len(component.downstream)}, "
            f"constructor_fanout={component.constructor_fanout}, is_tool={component.is_tool}"
        )
        lines.append(f"file: {component.file}")
        lines.append(f"entry_point: {component.entry_point}")
        lines.append(f"upstream: {component.upstream}")
        lines.append(f"downstream: {component.downstream}")
        lines.append(f"constraints: {constraints}")
        if show_model_call_verdict:
            # Precomputed statically (typed receiver resolution), not re-derived from the excerpt
            # below, which may be truncated or (for a delegating node) show no call at all.
            verdict = component.makes_model_call
            verdict_text = "yes" if verdict is True else "no" if verdict is False else "undecided"
            lines.append(f"makes_model_call (computed, authoritative): {verdict_text}")
        lines.append("source:")
        lines.append(snippet)
        lines.append("")
    return "\n".join(lines)


async def separate_agent_flows(
    system_map: SystemMap,
    source_by_component: dict[str, str],
    llm_client: LLMClient,
    feature_map_hint: str | None = None,
    files: list[Path] | None = None,
    retrieval_client: Any = None,
    snapshot_id: str | None = None,
) -> AgentFlowMap:
    """Node inclusion is structural (makes_model_call + call closure, see _build_structural_agents)
    and never depends on this LLM call; a malformed/empty response only costs label/summary/role
    text, never a dropped or invented agent. `retrieval_client`/`snapshot_id` (both optional) power
    the SECONDARY, out-of-scope-only source fallback — never the primary answer to a
    scanner<->expansion scope gap."""
    if not system_map.components:
        return AgentFlowMap(target_system_id=system_map.target_system_id)

    # Computed before the call so the grouping LLM is TOLD which components make a model call instead
    # of guessing it from a truncated snippet; the verdict itself was already decided statically at
    # map-build time (Component.makes_model_call), the sole authority for agent existence.
    evidence_sources = {**source_by_component, **_full_symbol_sources(system_map, files or [])}
    if retrieval_client is not None and snapshot_id:
        evidence_sources.update(
            await _out_of_scope_fallback_sources(system_map, retrieval_client, snapshot_id, evidence_sources)
        )

    parsed: dict = {}
    # Feature-map summary is project-wide, not per-system — it belongs in the system prompt (a
    # cacheable prefix shared across every system's agent-flow call), never appended after the
    # per-component data, which defeats the provider's prefix cache.
    system_prompt = AGENT_FLOW_SYSTEM
    if feature_map_hint:
        system_prompt += "\n\nProject feature map summary:\n" + feature_map_hint
    user_prompt = _build_user_prompt(system_map, evidence_sources, show_model_call_verdict=True)
    try:
        response = await llm_client.complete(
            [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ],
            max_tokens=_AGENT_FLOW_MAX_TOKENS,
            json_mode=True,
            reasoning_effort="low",  # structured extraction: reasoning burns the completion budget and returns empty content
        )
        if not (response.content or "").strip():
            raise ValueError(
                "LLM returned empty content — reasoning likely consumed the whole completion budget"
            )
        loaded = json.loads(response.content)
        if isinstance(loaded, dict):
            parsed = loaded
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning(
            f"Agent-flow separation failed: {e}. Falling back to ZERO separated agents — "
            "the UI will show one undivided flow instead of per-agent flows."
        )

    raw_agents = parsed.get("agents", [])
    if not isinstance(raw_agents, list):
        raw_agents = []
    # Keyed by the LLM's own "head component id" (per AGENT_FLOW_SYSTEM's schema) so a group's
    # label/summary/role can be matched back to the real, structurally-derived agent of that id;
    # membership/edges in the raw group are never consulted.
    labels_by_id: dict[str, dict[str, Any]] = {}
    for raw in raw_agents:
        if not isinstance(raw, dict):
            continue
        agent_id = raw.get("id")
        if isinstance(agent_id, str) and agent_id and agent_id not in labels_by_id:
            labels_by_id[agent_id] = raw

    agents, reached = _build_structural_agents(system_map, labels_by_id)
    unassigned_component_ids = sorted(c.id for c in system_map.components if c.id not in reached)


    _derive_parent_agents(agents, system_map)
    # A root of the structural call-DAG among agents -- nobody else's flow leads into it -- not
    # "who constructs me" (parent_agent), a different, independent signal.
    entry_agent_ids = [a.id for a in agents if not a.upstream_agents]
    flow_component_ids = sorted({
        cid for agent in agents for cid in (*agent.component_ids, *agent.boundary_component_ids)
    })

    return AgentFlowMap(
        target_system_id=system_map.target_system_id,
        agents=agents,
        entry_agent_ids=entry_agent_ids,
        unassigned_component_ids=unassigned_component_ids,
        flow_component_ids=flow_component_ids,
    )
