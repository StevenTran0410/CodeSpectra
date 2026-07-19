"""Enrichment DAG: gather evidence once, fan out an LLM pass per agent, persist."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent_eval_harness.discovery.agent_knowledge import (
    AgentKnowledge,
    ComponentRef,
    ComponentRoleVerdict,
    ContractArg,
    LocationInfo,
    PromptSiteRef,
    verify_citations,
)
from agent_eval_harness.discovery.prompt_site_scan import scan_for_prompt_sites
from agent_eval_harness.instrumentation._extract import utc_now_iso
from agent_eval_harness.llm.client import LLMClient
from agent_eval_harness.mapping.agent_flow import AgentFlow, AgentFlowMap, save_agent_flow_map
from agent_eval_harness.mapping.builder.contract_harvest import harvest_component_contract
from agent_eval_harness.mapping.builder.prompts import ROLE_TAXONOMY
from agent_eval_harness.mapping.builder.roles import (
    ROLE_CONFIDENCE_THRESHOLD,
    VALID_ROLES,
    admissible_roles,
    structural_facts,
)
from agent_eval_harness.mapping.builder.types import parse_python_source
from agent_eval_harness.mapping.system_map import Component, SystemMap, save_system_map
from agent_eval_harness.planning.agentic_planner import DagNode, complete_json, run_dag
from agent_eval_harness.store import repository

logger = logging.getLogger("agent_eval_harness.discovery.enrichment")

_STRUCTURAL_PRODUCER_VERSION = 2  # Manual bump on output-schema shape change; hash won't self-invalidate

_PROMPT_SITE_CHAR_BUDGET = 2000  # per-site, keeping the HEAD — the role-defining opening survives
_PROMPT_SITE_BLOCK_BUDGET = 20000  # total across all sites for one agent
_MAX_PROMPT_SITES_SHOWN = 15
_MAX_EDGES_SHOWN = 15
_MAX_NEXT_QUERIES = 2
_RETRIEVAL_CONCURRENCY = 4
_QUERY_HIT_LIMIT = 10
_QUERY_EXCERPT_CHAR_LIMIT = 1200
_OWN_FILE_CHUNK_LIMIT = 8
_RELATED_FILE_CHUNK_LIMIT = 3
_MAX_RELATED_FILES = 6
_MAX_SYMBOL_EDGES_FOLLOWED = 8
_ENRICH_MAX_TOKENS = 16000
_ENRICH_REASONING_EFFORT = "medium"


def agent_knowledge_dir(session_id: str) -> Path:
    """Directory for enriched AgentKnowledge JSON sidecars."""
    return Path.home() / "AppData" / "Local" / "codespectra" / "agents" / session_id


_ENTRY_METHOD_NAMES = ("run", "run_async", "__call__")


def _parse_signature_kwargs(signature: str) -> list[tuple[str, str]]:
    """Extract kwarg (name, type_hint) tuples from a method signature, skipping self/cls.

    On any parse ambiguity (e.g., nested generics with commas), return empty
    so the caller can defensively drop input_contract for this agent.
    """
    if not signature or '(' not in signature:
        return []
    try:
        # Extract the parameter list between ( and )
        start = signature.index('(')
        end = signature.rindex(')')
        params_str = signature[start + 1:end]
        if not params_str:
            return []

        # Split by comma at top level (balanced brackets)
        parts = []
        current = ""
        depth = 0
        for ch in params_str:
            if ch in '[{':
                depth += 1
            elif ch in ']}':
                depth -= 1
            elif ch == ',' and depth == 0:
                parts.append(current.strip())
                current = ""
                continue
            current += ch
        if current.strip():
            parts.append(current.strip())

        kwargs = []
        for part in parts:
            if not part:
                continue
            # Extract name (before :) and type_hint (after :)
            if ':' in part:
                name_part, type_part = part.split(':', 1)
                kwarg_name = name_part.strip().split('=')[0].split()[0]
                type_hint = type_part.split('=')[0].strip()
            else:
                kwarg_name = part.split('=')[0].split()[0]
                type_hint = ''
            if kwarg_name and kwarg_name not in ('self', 'cls'):
                kwargs.append((kwarg_name, type_hint))
        return kwargs
    except (ValueError, IndexError):
        # Parse ambiguity — return empty so we drop the field
        return []


def _resolve_repo_root(override_root: Path | None = None) -> Path | None:
    """The TARGET's repo root, or None when unknown — never AEH's own repo, whose files would
    resolve the target's relative paths against a different codebase entirely."""
    if override_root is not None:
        return override_root
    override = os.getenv("AEH_REPO_ROOT")
    return Path(override) if override else None

_ENRICH_SYSTEM = (
    "You are an expert AI software architect analyzing one agent within a discovered "
    "agentic system. Given evidence about the agent (its components, source excerpts, "
    "and detected prompt sites), first decide each component's ROLE, then describe the "
    "agent's functionality, context flow, and failure modes as BEHAVIOURS OF THAT ROLE — "
    "role is your first conclusion, everything else follows from it. Every claim "
    "about where something lives in the source must cite a real file and line number "
    "you can see in the evidence — never invent one; if you cannot find a file/line for "
    "a claim, omit file and line for that item rather than guessing.\n\n"
    "ROLE TAXONOMY — apply this to every component listed in the evidence:\n"
    f"{ROLE_TAXONOMY}\n\n"
    "The evidence has two reliability tiers, marked with '===' headers:\n"
    "- GROUND TRUTH blocks are fetched directly from the database by a known file path — "
    "always the correct file, trust these first for describing what the agent's own code does.\n"
    "- SUPPLEMENTARY blocks come from a keyword search and may include files that only "
    "sound related but aren't — cross-check a supplementary chunk against the ground-truth "
    "blocks before citing it as the agent's own behavior.\n\n"
    "Strict output rules:\n"
    "1. Respond ONLY with the raw JSON object shown in the user prompt's schema — no prose, "
    "no markdown code fences, no fields other than the ones listed there.\n"
    "2. \"functionality\" is always a non-empty string. Never null, never omitted — if the "
    "evidence is too thin to describe the agent confidently, write what is uncertain in "
    "plain words (e.g. \"Insufficient evidence to determine purpose beyond <best guess>\") "
    "rather than returning null or an empty string.\n"
    "3. \"component_roles\" is REQUIRED — one entry per component listed in the evidence, "
    "each role taken from THAT component's own admissible set. Do NOT add fields like "
    "\"degraded\", \"location\", \"components\", \"input_contract\", or \"output_contract\" "
    "— those are computed separately from static analysis, not from you, and including "
    "them will be ignored or cause an error."
)


def _s(v: Any) -> str:
    """Coerce to str; anything not a string (None, a stray int, ...) becomes ''."""
    return v if isinstance(v, str) else ''


def _opt_s(v: Any) -> str | None:
    """Coerce to a non-empty str or None — never pass through a wrong-typed value."""
    return v if isinstance(v, str) and v else None


def _opt_i(v: Any) -> int | None:
    """Coerce to int or None. bool is excluded (bool is a subclass of int in Python)."""
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _f(v: Any) -> float:
    """Coerce to float; anything not numeric becomes 0.0. bool excluded (subclass of int)."""
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0


def _clean_items(items: Any) -> list[dict]:
    return [i for i in items if isinstance(i, dict)] if isinstance(items, list) else []


def _clean_component_role(c: dict) -> dict | None:
    """A component id the LLM didn't name is unusable; an out-of-vocabulary role becomes
    'unknown' at confidence 0.0 here — the STRUCTURAL hard gate (per-component admissible
    set) runs later in _enrich_single_agent, this only guards against a hallucinated word."""
    cid = _s(c.get('id'))
    if not cid:
        return None
    role = _s(c.get('role'))
    if role not in VALID_ROLES:
        return {'id': cid, 'role': 'unknown', 'confidence': 0.0, 'reasoning': _s(c.get('reasoning'))}
    return {'id': cid, 'role': role, 'confidence': _f(c.get('confidence')), 'reasoning': _s(c.get('reasoning'))}


def _richness(k: AgentKnowledge) -> int:
    """Count of concrete claims — a harder proxy to game than raw functionality string length."""
    return (
        len(k.functionality_citations) + len(k.context_builders) +
        len(k.upstream_consumers) + len(k.downstream_consumers) + len(k.failure_modes)
    )


def _sanitize_llm_knowledge_dict(raw: Any) -> dict:
    """Allowlist fields the prompt asks for, coercing bad types — a hallucinated extra key or null would otherwise crash validation."""
    if not isinstance(raw, dict):
        return {}
    return {
        'component_roles': [
            r for r in (
                _clean_component_role(c) for c in _clean_items(raw.get('component_roles'))
            ) if r is not None
        ],
        'functionality': _s(raw.get('functionality')),
        'functionality_citations': [
            {'file': _opt_s(c.get('file')), 'line': _opt_i(c.get('line')), 'symbol': _s(c.get('symbol'))}
            for c in _clean_items(raw.get('functionality_citations'))
        ],
        'context_builders': [
            {
                'name': _s(c.get('name')),
                'file': _opt_s(c.get('file')),
                'line': _opt_i(c.get('line')),
                'builds_kwarg': _s(c.get('builds_kwarg')),
            }
            for c in _clean_items(raw.get('context_builders'))
            if _s(c.get('name'))
        ],
        'upstream_consumers': [
            {'name': _s(c.get('name')), 'file': _opt_s(c.get('file')), 'line': _opt_i(c.get('line'))}
            for c in _clean_items(raw.get('upstream_consumers'))
            if _s(c.get('name'))
        ],
        'downstream_consumers': [
            {'name': _s(c.get('name')), 'file': _opt_s(c.get('file')), 'line': _opt_i(c.get('line'))}
            for c in _clean_items(raw.get('downstream_consumers'))
            if _s(c.get('name'))
        ],
        'failure_modes': [
            {'description': _s(c.get('description')), 'file': _opt_s(c.get('file')), 'line': _opt_i(c.get('line'))}
            for c in _clean_items(raw.get('failure_modes'))
            if _s(c.get('description'))
        ],
        'output_described_in_prompt': _s(raw.get('output_described_in_prompt')),
        'special_traits': [t for t in (raw.get('special_traits') or []) if isinstance(t, str)],
        'constraints': [t for t in (raw.get('constraints') or []) if isinstance(t, str)],
        'method_steps': [t for t in (raw.get('method_steps') or []) if isinstance(t, str)],
    }


def _derive_agent_role(agent: AgentFlow, system_map: SystemMap) -> str:
    """AgentFlow.role is DERIVED in code from its components' gated roles, never asked of the
    LLM: the single non-worker/non-unknown role among the agent's components if exactly one
    exists, else 'worker' — or 'unknown' if every component is still 'unknown'."""
    comp_roles = [
        comp.role for cid in agent.component_ids
        if (comp := system_map.component_by_id(cid)) is not None
    ]
    if not comp_roles or all(r == 'unknown' for r in comp_roles):
        return 'unknown'
    distinct_special = {r for r in comp_roles if r not in ('worker', 'unknown')}
    if len(distinct_special) == 1:
        return next(iter(distinct_special))
    return 'worker'


async def _execute_queries(ctx: _EnrichmentContext, queries: list[str]) -> str:
    """Run RRF queries, format hits into an evidence block. Read `final or fused` —
    final is empty when reranking is disabled. Never raises."""
    if not ctx.snapshot_id:
        logger.warning("Round-2 query requested but no snapshot_id configured for this run")
        return "[No query results — snapshot_id not configured for this enrichment run.]"
    sections = []
    for q in queries:
        try:
            async with ctx.semaphore:
                res = await ctx.client.search_retrieval(ctx.snapshot_id, q, symbol_chunks_only=True)
            hits = res.get("final") or res.get("fused") or []
            lines = [f'Query: "{q}"']
            for h in hits[:_QUERY_HIT_LIMIT]:
                excerpt = (h.get("excerpt") or "").strip()[:_QUERY_EXCERPT_CHAR_LIMIT]
                start, end = h.get("start_line") or 0, h.get("end_line") or 0
                loc = f"{h.get('rel_path', '?')}:{start}-{end}" if end else h.get("rel_path", "?")
                lines.append(f"  - {loc}:\n{excerpt}")
            if not hits:
                lines.append("  (no hits)")
            sections.append("\n".join(lines))
        except Exception as e:
            logger.warning(f"Round-2 query failed for {q!r}: {e}")
            sections.append(f'Query: "{q}"\n  (query failed: {e})')
    return "\n\n".join(sections) if sections else "[No query results.]"


async def _fetch_file_chunks(ctx: _EnrichmentContext, rel_path: str, limit: int) -> list[str]:
    """Direct chunk fetch by known path — no search. Never raises."""
    try:
        async with ctx.semaphore:
            res = await ctx.client.get_file_chunks(ctx.snapshot_id, rel_path, symbol_chunks_only=True)
    except Exception as e:
        logger.warning(f"Direct chunk fetch failed for {rel_path}: {e}")
        return []
    lines = []
    for c in (res.get("chunks") or [])[:limit]:
        lines.append(f"{rel_path}:{c.get('start_line', 0)}-{c.get('end_line', 0)} [{c.get('chunk_type', '?')}]")
        lines.append((c.get("content") or "").strip()[:1500])
    return lines


async def _fetch_direct_evidence(
    ctx: _EnrichmentContext, own_files: set[str], known_edge_files: set[str]
) -> tuple[list[str], list[str]]:
    """Ground-truth evidence for one agent: own file(s) plus related files (known edges + 1-hop symbol-graph calls), all fetched directly by path, zero search."""
    own_lines: list[str] = []
    for f in own_files:
        own_lines.extend(await _fetch_file_chunks(ctx, f, limit=_OWN_FILE_CHUNK_LIMIT))

    related_files: set[str] = set(known_edge_files) - own_files
    for f in own_files:
        try:
            async with ctx.semaphore:
                edges_res = await ctx.client.get_symbol_edges(ctx.snapshot_id, f)
        except Exception as e:
            logger.warning(f"Symbol edge lookup failed for {f}: {e}")
            continue
        outgoing = sorted(
            edges_res.get("outgoing") or [], key=lambda e: e.get("confidence_score", 0), reverse=True
        )
        for edge in outgoing[:_MAX_SYMBOL_EDGES_FOLLOWED]:
            dst = edge.get("dst_symbol", "")
            dst_file = dst.split("::")[0] if "::" in dst else ""
            if dst_file and dst_file not in own_files:
                related_files.add(dst_file)

    related_lines: list[str] = []
    for f in list(related_files)[:_MAX_RELATED_FILES]:
        related_lines.extend(await _fetch_file_chunks(ctx, f, limit=_RELATED_FILE_CHUNK_LIMIT))

    return own_lines, related_lines


@dataclass
class _EnrichmentContext:
    """Shared state for enrichment run."""
    session_id: str
    snapshot_id: str
    agent_flow_map: AgentFlowMap
    system_map: SystemMap
    accepted_with_annotations: list[dict]
    accepted_edges: list[dict]
    client: Any  # retrieval client
    llm_client: LLMClient
    depth: str
    force_agent_ids: list[str]
    semaphore: asyncio.Semaphore
    repo_root: Path | None = None


async def enrich_agents(
    session_id: str,
    agent_flow_map: AgentFlowMap,
    system_map: SystemMap,
    accepted_with_annotations: list[dict],
    accepted_edges: list[dict],
    client: Any,
    llm_client: LLMClient,
    *,
    snapshot_id: str = '',
    depth: str = 'normal',
    agent_ids: list[str] | None = None,
    force_agent_ids: list[str] | None = None,
    map_path: str | Path | None = None,
    agent_flows_path: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> list[AgentKnowledge]:
    """Orchestrate concurrent enrichment of discovered agents.

    Args:
        session_id: Expansion session ID for artifact storage
        agent_flow_map: Discovered agent flows
        system_map: System component map
        accepted_with_annotations: Files with role hints and annotations
        accepted_edges: Discovered file-to-file edges
        client: retrieval client for search queries
        llm_client: LLM client for analysis
        snapshot_id: Snapshot to run retrieval queries against. Omit only in tests that
            never fire a query — a real caller must pass this or queries no-op.
        depth: 'normal' (≤3 queries, ≤2 LLM calls) or 'deep' (≤6 queries, ≤3 rounds)
        agent_ids: If provided, enrich only these agents (subset)
        force_agent_ids: If provided, re-enrich these agents even if cached
        map_path: If provided, gated component roles are written back onto
            `system_map` and persisted here — system_map YAML is the sole authority for role.
        agent_flows_path: If provided, `AgentFlow.role` (derived in code from gated component
            roles, never asked of the LLM) is persisted here.
        repo_root: Root to resolve prompt-site file paths and citations against. None keeps
            the default `_resolve_repo_root()` behaviour byte-identical.

    Returns:
        List of AgentKnowledge objects, one per agent
    """
    resolved_repo_root = Path(repo_root) if repo_root is not None else None
    force_ids = set(force_agent_ids or [])
    target_agents = [
        a for a in agent_flow_map.agents
        if agent_ids is None or a.id in agent_ids
    ]

    if not target_agents:
        return []

    # Per-depth caps
    caps = {
        'normal': {'queries': 3, 'llm_calls': 2},
        'deep': {'queries': 6, 'llm_calls': 3},
    }
    depth_cap = caps.get(depth, caps['normal'])

    # Shared semaphore for retrieval throttling (wraps search_retrieval/query only)
    semaphore = asyncio.Semaphore(_RETRIEVAL_CONCURRENCY)

    ctx = _EnrichmentContext(
        session_id=session_id,
        snapshot_id=snapshot_id,
        agent_flow_map=agent_flow_map,
        system_map=system_map,
        accepted_with_annotations=accepted_with_annotations,
        accepted_edges=accepted_edges,
        client=client,
        llm_client=llm_client,
        depth=depth,
        force_agent_ids=list(force_ids),
        semaphore=semaphore,
        repo_root=resolved_repo_root,
    )

    # Stage 1: Gather shared evidence (prompt sites, component metadata, edges)
    async def _gather(_: dict[str, Any]) -> dict[str, Any]:
        return await _gather_evidence(ctx)

    # Stage 2: Enrich each agent (fan-out, all depend on gather)
    nodes: list[DagNode] = [DagNode("gather", [], _gather)]

    accepted_files = [
        item["file"] if isinstance(item, dict) else item
        for item in accepted_with_annotations
    ]

    for agent in target_agents:
        agent_id = agent.id
        enrich_name = f"enrich:{agent_id}"

        async def _enrich(
            results: dict[str, Any],
            agent_id_: str = agent_id,
        ) -> AgentKnowledge:
            try:
                evidence = results["gather"]
                return await _enrich_single_agent(
                    agent_id_,
                    evidence,
                    ctx,
                    depth_cap,
                    accepted_files,
                )
            except Exception as e:
                logger.exception(f"Error enriching {agent_id_}: {e}")
                return AgentKnowledge(
                    degraded=True,
                    confidence='low',
                    degraded_reason=f"Enrichment failed: {str(e)}"
                )

        nodes.append(DagNode(enrich_name, ["gather"], _enrich))

    # Stage 3: Persist all enriched knowledge
    enrich_names = [f"enrich:{a.id}" for a in target_agents]

    async def _persist(results: dict[str, Any]) -> None:
        appdata_dir = agent_knowledge_dir(session_id)
        appdata_dir.mkdir(parents=True, exist_ok=True)

        role_by_component: dict[str, tuple[str, float]] = {}

        for agent in target_agents:
            knowledge = results[f"enrich:{agent.id}"]
            md_path = appdata_dir / f"{agent.id}.md"
            json_path = appdata_dir / f"{agent.id}.json"

            md_path.write_text(knowledge.to_md(), encoding='utf-8')
            json_path.write_text(json.dumps(knowledge.to_json(), indent=2), encoding='utf-8')
            await repository.upsert_agent_knowledge(
                session_id=session_id,
                agent_id=agent.id,
                md_path=str(md_path),
                json_path=str(json_path),
                evidence_hash=knowledge.evidence_hash,
                confidence=knowledge.confidence,
                query_count=knowledge.query_count,
            )

            # A degraded agent's verdicts are never applied — one agent failing must never
            # blank another agent's good role.
            if knowledge.degraded:
                continue
            for verdict in knowledge.component_roles:
                role_by_component[verdict.id] = (verdict.role, verdict.confidence)

        if not role_by_component:
            return

        for component in ctx.system_map.components:
            if component.id in role_by_component:
                role, confidence = role_by_component[component.id]
                component.role = role
                component.role_confidence = confidence
                component.role_source = 'llm_constrained'

        # AgentFlow.role is DERIVED in code from the just-applied gated component roles,
        # never asked of the LLM.
        if agent_flows_path:
            for flow in ctx.agent_flow_map.agents:
                flow.role = _derive_agent_role(flow, ctx.system_map)
            save_agent_flow_map(ctx.agent_flow_map, agent_flows_path)

        # system_map YAML is the sole authority for role — write it LAST, only after every
        # agent's md/json/DB row (and agent_flows) above has already succeeded.
        if map_path:
            save_system_map(ctx.system_map, map_path)

    nodes.append(DagNode("persist", enrich_names, _persist))

    # DAG validation before run
    dep_names = {n.name for n in nodes}
    for n in nodes:
        missing = [d for d in n.deps if d not in dep_names]
        assert not missing, f"Missing deps {missing} for node {n.name}"

    results = await run_dag(nodes)
    knowledge_list = [
        results[f"enrich:{a.id}"]
        for a in target_agents
    ]

    return knowledge_list


async def _gather_evidence(ctx: _EnrichmentContext) -> dict[str, Any]:
    """Gather shared evidence: prompt sites, component info, edges.

    Returns dict with:
    - prompt_sites_by_file: dict[file, list[PromptSite]]
    - component_by_agent: dict[agent_id, list[dict]]
    - edges_by_agent: dict[agent_id, list[dict]]
    - source_coverage: dict[agent_id, float] (0-1 estimated coverage)
    """
    accepted_files = [
        item["file"] if isinstance(item, dict) else item
        for item in ctx.accepted_with_annotations
    ]

    scan_root = _resolve_repo_root(ctx.repo_root)
    if scan_root is None:
        logger.warning(
            "No target repo root — prompt-site scan skipped, every agent will be enriched without "
            "its real prompt text. Pass repo_root= or set AEH_REPO_ROOT."
        )
    prompt_sites_by_file = (
        scan_for_prompt_sites(scan_root, accepted_files) if scan_root is not None else {}
    )

    # Build per-agent component info
    component_by_agent: dict[str, list[dict]] = {}
    for agent in ctx.agent_flow_map.agents:
        agent_components = []
        for cid in agent.component_ids:
            comp = ctx.system_map.component_by_id(cid)
            if comp:
                agent_components.append({
                    'id': cid,
                    'role': comp.role,
                    'file': comp.file,
                    'entry_point': comp.entry_point,
                    'is_tool': comp.is_tool,
                    'constructor_fanout': comp.constructor_fanout,
                    'fan_in': len(comp.upstream),
                    'fan_out': len(comp.downstream),
                })
        component_by_agent[agent.id] = agent_components

    # Build per-agent edges
    edges_by_agent: dict[str, list[dict]] = {}
    for agent in ctx.agent_flow_map.agents:
        agent_edges = []
        agent_files = {
            c.get('file') for c in component_by_agent.get(agent.id, [])
            if c.get('file')
        }
        for edge in ctx.accepted_edges:
            src, dst = edge.get('src'), edge.get('dst')
            if src in agent_files or dst in agent_files:
                agent_edges.append(edge)
        edges_by_agent[agent.id] = agent_edges

    # Estimate source coverage per agent
    source_coverage: dict[str, float] = {}
    for agent in ctx.agent_flow_map.agents:
        covered_files = len([
            c for c in component_by_agent.get(agent.id, [])
            if c.get('file')
        ])
        total_files = len(agent.component_ids) if agent.component_ids else 1
        source_coverage[agent.id] = min(1.0, covered_files / max(1, total_files))

    return {
        'prompt_sites_by_file': prompt_sites_by_file,
        'component_by_agent': component_by_agent,
        'edges_by_agent': edges_by_agent,
        'source_coverage': source_coverage,
    }


async def _enrich_single_agent(
    agent_id: str,
    evidence: dict[str, Any],
    ctx: _EnrichmentContext,
    depth_cap: dict[str, int],
    accepted_files: list[str],
) -> AgentKnowledge:
    """Enrich a single agent: check cache, verify coverage, run queries/LLM, persist."""

    # 1. Compute evidence hash (hoisted before cache check)
    component_ids = sorted([
        c['id'] for c in evidence['component_by_agent'].get(agent_id, [])
    ])
    edges = sorted([
        (e['src'], e['dst']) for e in evidence['edges_by_agent'].get(agent_id, [])
    ])

    hash_input = '|'.join([
        str(_STRUCTURAL_PRODUCER_VERSION),
        ':'.join(component_ids),
        ':'.join(str(len(accepted_files))),
        ':'.join(f"{s}→{d}" for s, d in edges),
    ])
    evidence_hash = hashlib.sha256(hash_input.encode('utf-8')).hexdigest()

    # 2. Check cache — short-circuit only when hash matches and agent not forced
    cached = await repository.get_agent_knowledge(ctx.session_id, agent_id)
    if (
        cached and
        cached.get('evidence_hash') == evidence_hash and
        agent_id not in ctx.force_agent_ids
    ):
        # Return cached knowledge without LLM calls — read from json_path on disk
        try:
            cached_json_path = Path(cached.get('json_path', ''))
            if cached_json_path.exists():
                cached_data = json.loads(cached_json_path.read_text(encoding='utf-8'))
                return AgentKnowledge.from_json(cached_data)
        except Exception as e:
            logger.warning(f"Failed to deserialize cached knowledge for {agent_id}: {e}")

    # 3. Coverage check — gates the SUPPLEMENTARY pre-query only; the LLM call always runs.
    prompt_sites = evidence['prompt_sites_by_file']
    components = evidence['component_by_agent'].get(agent_id, [])
    component_files = {c['file'] for c in components if c.get('file')}
    prompt_site_files = set(prompt_sites.keys()) & component_files

    coverage_sufficient = (
        len(components) > 0 and
        len(prompt_site_files) > 0 and
        evidence['source_coverage'].get(agent_id, 0) >= 0.8
    )

    # 4. LLM enrichment with round-1 and optional round-2
    query_count = 0
    llm_calls = 0

    agent = None
    for a in ctx.agent_flow_map.agents:
        if a.id == agent_id:
            agent = a
            break

    if not agent:
        return AgentKnowledge(
            degraded=True,
            degraded_reason=f"Agent {agent_id} not found in flow map",
            evidence_hash=evidence_hash,
            generated_at=utc_now_iso(),
        )

    # Evidence the LLM cites from: components (+ admissible-role facts the hard gate below
    # enforces), prompt sites, edges.
    context_lines = [f"Agent: {agent.label}", ""]
    if components:
        context_lines.append(f"Components ({len(components)}):")
        for c in components:
            entry = f" — entry: {c['entry_point']}" if c.get('entry_point') else ""
            facts = structural_facts(c.get('fan_in', 0), c.get('fan_out', 0))
            admissible = sorted(admissible_roles(c.get('is_tool'), c.get('constructor_fanout')))
            context_lines.append(
                f"  - {c['id']} @ {c.get('file', '?')}{entry} — {facts}; admissible roles: {admissible}"
            )
    else:
        context_lines.append("Components: none")

    relevant_sites = [s for f in component_files for s in prompt_sites.get(f, [])]
    if relevant_sites:
        context_lines.append("")
        context_lines.append(f"Prompt/LLM call sites in this agent's files ({len(relevant_sites)}):")
        block_chars = 0
        for site in relevant_sites[:_MAX_PROMPT_SITES_SHOWN]:
            snippet = (site.snippet or '').replace('\n', ' ')
            if len(snippet) > _PROMPT_SITE_CHAR_BUDGET:
                snippet = snippet[:_PROMPT_SITE_CHAR_BUDGET] + " [truncated]"
            line = f"  - {site.file}:{site.line} [{site.kind}] {snippet}"
            if block_chars + len(line) > _PROMPT_SITE_BLOCK_BUDGET:
                context_lines.append("  ... [truncated — remaining prompt sites omitted to stay in budget]")
                break
            context_lines.append(line)
            block_chars += len(line)

    agent_edges = evidence['edges_by_agent'].get(agent_id, [])
    if agent_edges:
        context_lines.append("")
        context_lines.append(f"File-to-file edges touching this agent ({len(agent_edges)}):")
        for edge in agent_edges[:_MAX_EDGES_SHOWN]:
            context_lines.append(f"  - {edge.get('src')} → {edge.get('dst')}")

    # Ground-truth evidence: own file(s) + 1-hop related files (known edges + symbol-graph calls), fetched directly by path — no search.
    if ctx.snapshot_id and component_files:
        known_edge_files = {
            f for e in agent_edges for f in (e.get('src'), e.get('dst')) if f
        }
        own_lines, related_lines = await _fetch_direct_evidence(ctx, component_files, known_edge_files)
        if own_lines:
            context_lines.append("")
            context_lines.append("=== GROUND TRUTH — direct from DB, this agent's own file ===")
            context_lines.extend(own_lines)
        if related_lines:
            context_lines.append("")
            context_lines.append("=== GROUND TRUTH — direct from DB, related files (known edges + symbol calls) ===")
            context_lines.extend(related_lines)

    # Deterministic pre-query: supplementary bare-keyword RRF search, skipped when coverage is already sufficient.
    if not coverage_sufficient and ctx.snapshot_id and query_count < depth_cap['queries']:
        pre_query_block = await _execute_queries(ctx, [agent.label])
        query_count += 1
        context_lines.append("")
        context_lines.append("=== SUPPLEMENTARY — from system query (may include unrelated files; cross-check against ground truth above before citing) ===")
        context_lines.append(pre_query_block)

    context_str = "\n".join(context_lines)

    # Full-schema round-1 prompt with JSON shape examples
    full_schema_prompt = f"""Analyze this agent and return detailed semantic profile in JSON format.

{context_str}

Return JSON with this exact shape (all fields required, empty arrays/null if unknown):
{{
  "component_roles": [
    {{"id": "<id from the Components list above>", "role": "<from THAT component's admissible set>", "confidence": 0.0, "reasoning": "<one sentence>"}}
  ],
  "functionality": "<one sentence describing the agent's core purpose>",
  "functionality_citations": [
    {{"file": "<path/to/file.py>", "line": 42, "symbol": "function_or_class_name"}}
  ],
  "context_builders": [
    {{"name": "<helper_name>", "file": "<path/to/file.py>", "line": 10, "builds_kwarg": "context_arg"}}
  ],
  "upstream_consumers": [
    {{"name": "<consumer_name>", "file": "<path/to/file.py>", "line": 20}}
  ],
  "downstream_consumers": [
    {{"name": "<consumer_name>", "file": "<path/to/file.py>", "line": 30}}
  ],
  "failure_modes": [
    {{"description": "<prose description>", "file": "<path/to/file.py>", "line": 40}}
  ],
  "output_described_in_prompt": "<how the prompt says this agent's output should look — format/shape/fields, 1-2 sentences; empty string if the prompt is silent>",
  "method_steps": ["<one step, in order, of the procedure the prompt tells the agent to follow, e.g. 'retrieve evidence' then 'reason' then 'emit JSON'>"],
  "constraints": ["<one HARD RULE the prompt imposes, e.g. 'return only JSON', 'must cite evidence_files', 'never invent file paths'>"],
  "special_traits": ["<one notable/distinctive behavior the prompt calls out>"],
  "need_more": false,
  "next_queries": []
}}

If you need more information to provide complete answers, set need_more to true and list follow-up queries (max 2).
"""

    llm_knowledge: AgentKnowledge | None = None
    async def _run_llm_rounds(
        prompt: str,
        depth: dict[str, int],
    ) -> AgentKnowledge | None:
        """Run LLM round-1, optionally round-2 based on need_more."""
        nonlocal query_count, llm_calls

        llm_calls += 1
        raw = await complete_json(
            ctx.llm_client, _ENRICH_SYSTEM, prompt,
            max_tokens=_ENRICH_MAX_TOKENS, label=f"enrich[{agent_id}]",
            reasoning_effort=_ENRICH_REASONING_EFFORT,
        )
        if raw is None:
            logger.warning(f"LLM round-1 failed for {agent_id}")
            return None

        # extra='ignore' drops these on model_validate, so grab them from the raw dict first
        need_more = raw.get('need_more', False)
        next_queries = raw.get('next_queries', [])[:_MAX_NEXT_QUERIES]

        round_knowledge = AgentKnowledge.model_validate(_sanitize_llm_knowledge_dict(raw))

        if (
            need_more and
            next_queries and
            llm_calls < depth['llm_calls'] and
            query_count < depth['queries']
        ):
            logger.debug(f"Agent {agent_id}: need_more=true, attempting round-2 with {len(next_queries)} queries")

            executed_queries = min(len(next_queries), depth['queries'] - query_count)
            query_results = await _execute_queries(ctx, next_queries[:executed_queries])
            query_count += executed_queries
            llm_calls += 1

            # Build round-2 prompt with real query results
            round2_prompt = (
                f"Given your previous analysis, here are the follow-up query results you asked for:\n\n"
                f"{query_results}\n\n"
                f"Re-analyze and provide an updated semantic profile using this new evidence. "
                f"Return the SAME JSON shape as before — component_roles, functionality, "
                f"functionality_citations, context_builders, upstream_consumers, "
                f"downstream_consumers, failure_modes, output_described_in_prompt, method_steps, "
                f"constraints, special_traits, need_more, next_queries — with no other "
                f"fields. functionality must still be a non-empty string, never null. If the "
                f"query results above don't actually add anything useful, keep your previous, "
                f"more specific answer rather than replacing it with a vaguer one."
            )

            raw2 = await complete_json(
                ctx.llm_client, _ENRICH_SYSTEM, round2_prompt,
                max_tokens=_ENRICH_MAX_TOKENS, label=f"enrich[{agent_id}]-round2",
                reasoning_effort=_ENRICH_REASONING_EFFORT,
            )
            if raw2 is not None:
                round2_knowledge = AgentKnowledge.model_validate(_sanitize_llm_knowledge_dict(raw2))
                # Richness (concrete claim count), not string length, decides the winner.
                r1, r2 = _richness(round_knowledge), _richness(round2_knowledge)
                if r2 > r1 or (r2 == r1 and len(round2_knowledge.functionality) >= len(round_knowledge.functionality)):
                    round_knowledge = round2_knowledge
                else:
                    logger.debug(
                        f"Agent {agent_id}: round-2 less rich than round-1 ({r2} < {r1} claims), keeping round-1"
                    )

        return round_knowledge

    try:
        llm_knowledge = await _run_llm_rounds(full_schema_prompt, depth_cap)
        if llm_knowledge is None:
            llm_knowledge = AgentKnowledge(degraded=True, degraded_reason="LLM analysis failed")
    except Exception as e:
        logger.exception(f"Enrichment failed for {agent_id}: {e}")
        llm_knowledge = AgentKnowledge(degraded=True, degraded_reason=f"Enrichment exception: {str(e)}")

    # 5. Hard gate — mirrors roles.py's structural subtraction: the prompt lists the
    # admissible set, the code enforces it.
    structural_by_id = {c['id']: c for c in components}
    gated_roles: list[ComponentRoleVerdict] = []
    for verdict in llm_knowledge.component_roles:
        c = structural_by_id.get(verdict.id)
        if c is None:
            continue  # LLM named a component id that isn't this agent's — discard, never invent
        admissible = admissible_roles(c.get('is_tool'), c.get('constructor_fanout'))
        role = verdict.role
        if role not in admissible or verdict.confidence < ROLE_CONFIDENCE_THRESHOLD:
            role = 'unknown'
        gated_roles.append(ComponentRoleVerdict(
            id=verdict.id, role=role, confidence=verdict.confidence, reasoning=verdict.reasoning
        ))
    llm_knowledge.component_roles = gated_roles

    # 6. Static-wins merge — structural fields never come from the LLM
    llm_knowledge.evidence_hash = evidence_hash
    llm_knowledge.query_count = query_count
    llm_knowledge.generated_at = utc_now_iso()

    # code_symbols rows keyed by file, reused for both the structural fill and citation verification.
    symbols_by_file: dict[str, list[dict]] = {}

    # Fill location/components/input_contract from code_symbols if snapshot_id is available
    if ctx.snapshot_id and agent_id in evidence.get('component_by_agent', {}):
        components_list = evidence['component_by_agent'][agent_id]
        if components_list:
            try:
                # Get all unique files for this agent's components
                comp_files = {c['file'] for c in components_list if c.get('file')}
                if comp_files:
                    # Query for each file to get symbol info
                    for comp_file in comp_files:
                        try:
                            file_basename = Path(comp_file).name
                            symbols_response = await ctx.client.search_repo_map(ctx.snapshot_id, q=file_basename)
                            symbols = symbols_response.get('symbols', [])
                            symbols_by_file[comp_file] = symbols

                            # Extract class name from the first component's entry_point
                            if components_list[0].get('entry_point'):
                                _, _, class_name = components_list[0]['entry_point'].partition(':')
                                class_name = class_name.split('.')[0]

                                # Find class row for location
                                class_row = None
                                method_row = None
                                for sym in symbols:
                                    if sym.get('kind') == 'class' and sym.get('name') == class_name:
                                        class_row = sym
                                    elif (sym.get('parent_name') == class_name and
                                          sym.get('kind') == 'method' and
                                          sym.get('name') in _ENTRY_METHOD_NAMES):
                                        if method_row is None or _ENTRY_METHOD_NAMES.index(sym.get('name')) < _ENTRY_METHOD_NAMES.index(method_row.get('name')):
                                            method_row = sym

                                # Fill location from class and method rows
                                if class_row and method_row:
                                    llm_knowledge.location = LocationInfo(
                                        file=comp_file,
                                        line_start=class_row.get('line_start', 0),
                                        line_end=class_row.get('line_end', 0),
                                        entry_method=method_row.get('name', ''),
                                        entry_line=method_row.get('line_start', 0),
                                    )
                                    # Fill input_contract from method signature
                                    sig = method_row.get('signature', '')
                                    kwargs_with_hints = _parse_signature_kwargs(sig)
                                    if kwargs_with_hints is not None and len(kwargs_with_hints) > 0:
                                        llm_knowledge.input_contract = [
                                            ContractArg(kwarg=name, source_kind='signature', type_hint=hint, example='')
                                            for name, hint in kwargs_with_hints
                                        ]

                                    # Fill output_contract from AST harvest
                                    try:
                                        comp_for_harvest = None
                                        for comp in components_list:
                                            if comp.get('file') == comp_file:
                                                comp_for_harvest = comp
                                                break
                                        if comp_for_harvest:
                                            comp_id = comp_for_harvest.get('id', agent_id)
                                            entry_pt = comp_for_harvest.get('entry_point', '')
                                            temp_component = Component(id=comp_id, role='unknown', entry_point=entry_pt, file=comp_file)
                                            comp_path = Path(comp_file)
                                            if comp_path.exists():
                                                parsed = parse_python_source(comp_path)
                                                if parsed:
                                                    asts = {comp_path: parsed[1]}
                                                    _, output, _, _, _ = harvest_component_contract(temp_component, asts, files_root=ctx.repo_root)
                                                    if output:
                                                        llm_knowledge.output_contract = output
                                    except Exception as e:
                                        logger.debug(f"Failed to harvest output_contract for {agent_id}: {e}")

                                # Fill components
                                for comp in components_list:
                                    if class_row and comp.get('file') == comp_file:
                                        llm_knowledge.components.append(ComponentRef(
                                            id=comp['id'],
                                            role=comp.get('role', 'unknown'),
                                            file=comp_file,
                                            line=class_row.get('line_start', 0),
                                        ))
                        except Exception as e:
                            logger.warning(f"Failed to fill structural fields for {agent_id} from {comp_file}: {e}")
            except Exception as e:
                logger.warning(f"Failed to fill structural fields for {agent_id}: {e}")
    else:
        # No snapshot or no components — degrade silently
        if not ctx.snapshot_id:
            logger.warning(f"No snapshot_id for {agent_id} — cannot fill structural fields")

    # Populate prompt_sites from evidence for this agent's components
    if agent_id in evidence.get('component_by_agent', {}):
        comp_files = {
            c['file'] for c in evidence['component_by_agent'][agent_id]
            if c.get('file')
        }
        prompt_sites_by_file = evidence.get('prompt_sites_by_file', {})
        for file, sites in prompt_sites_by_file.items():
            if file in comp_files:
                llm_knowledge.prompt_sites.extend(PromptSiteRef(**asdict(s)) for s in sites)

    # 7. Verify citations — mutates knowledge.needs_human for unverified/phantom claims
    verify_root = _resolve_repo_root(ctx.repo_root)
    if verify_root is None:
        logger.warning(
            "No target repo root — citation verification skipped for %s; the LLM's file:line "
            "claims are recorded UNCHECKED. Pass repo_root= or set AEH_REPO_ROOT.",
            agent_id,
        )
    else:
        # Fetch symbols for any cited file not already pulled during the fill, so span-based
        # citation resolution covers cross-file citations too — not just the agent's own file.
        if ctx.snapshot_id:
            cited_files = {
                c.file for src in (
                    llm_knowledge.functionality_citations, llm_knowledge.context_builders,
                    llm_knowledge.upstream_consumers, llm_knowledge.downstream_consumers,
                    llm_knowledge.failure_modes,
                )
                for c in src if getattr(c, 'file', None)
            }
            for cf in cited_files - symbols_by_file.keys():
                try:
                    resp = await ctx.client.search_repo_map(ctx.snapshot_id, q=Path(cf).name)
                    symbols_by_file[cf] = resp.get('symbols', [])
                except Exception as e:
                    logger.debug("Could not fetch symbols for cited file %s: %s", cf, e)
        vreport = verify_citations(llm_knowledge, verify_root, symbols_by_file=symbols_by_file)
        if vreport.claims:
            logger.debug(
                "Citation verification for %s: %d claims, %d phantom, %d unverified",
                agent_id,
                len(vreport.claims),
                sum(1 for c in vreport.claims if c.status == "phantom"),
                sum(1 for c in vreport.claims if c.status == "unverified"),
            )

    # 8. Confidence from field-fullness + needs_human + query_count
    if llm_knowledge.degraded:
        llm_knowledge.confidence = 'low'
    elif llm_knowledge.needs_human:
        # Hard invariant: needs_human non-empty => never 'high'
        llm_knowledge.confidence = 'medium'
    else:
        # Score field-fullness: count populated among {functionality, functionality_citations,
        # location, components, input_contract, prompt_sites, context_builders, failure_modes}
        field_count = sum([
            bool(llm_knowledge.functionality),
            bool(llm_knowledge.functionality_citations),
            llm_knowledge.location is not None,
            bool(llm_knowledge.components),
            bool(llm_knowledge.input_contract),
            bool(llm_knowledge.prompt_sites),
            bool(llm_knowledge.context_builders),
            bool(llm_knowledge.failure_modes),
        ])
        if field_count >= 5:
            llm_knowledge.confidence = 'high'
        elif field_count >= 2:
            llm_knowledge.confidence = 'medium'
        else:
            llm_knowledge.confidence = 'low'

    return llm_knowledge
