"""Enrichment DAG: gather evidence once, fan out an LLM pass per agent, persist."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_eval_harness.discovery.agent_knowledge import AgentKnowledge, verify_citations
from agent_eval_harness.discovery.prompt_site_scan import scan_for_prompt_sites
from agent_eval_harness.injection.backend_bridge import default_backend_path
from agent_eval_harness.instrumentation._extract import utc_now_iso
from agent_eval_harness.llm.client import LLMClient
from agent_eval_harness.mapping.agent_flow import AgentFlowMap
from agent_eval_harness.mapping.system_map import SystemMap
from agent_eval_harness.planning.agentic_planner import DagNode, complete_json, run_dag
from agent_eval_harness.store import repository

logger = logging.getLogger("agent_eval_harness.discovery.enrichment")


def _resolve_repo_root() -> Path:
    """Repo root containing backend/ — never Path.cwd(), which is agent_eval_harness/ under Electron dev mode."""
    override = os.getenv("AEH_REPO_ROOT")
    return Path(override) if override else default_backend_path().parent

_ENRICH_SYSTEM = (
    "You are an expert AI software architect analyzing one agent within a discovered "
    "agentic system. Given evidence about the agent (its components, source excerpts, "
    "and detected prompt sites), produce a structured JSON semantic profile of what "
    "the agent does, what it consumes and produces, and how it can fail. Every claim "
    "about where something lives in the source must cite a real file and line number "
    "you can see in the evidence — never invent one; if you cannot find a file/line for "
    "a claim, omit file and line for that item rather than guessing.\n\n"
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
    "3. Do NOT add fields like \"confidence\", \"degraded\", \"location\", \"components\", "
    "\"input_contract\", or \"output_contract\" — those are computed separately from static "
    "analysis, not from you, and including them will be ignored or cause an error."
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


def _clean_items(items: Any) -> list[dict]:
    return [i for i in items if isinstance(i, dict)] if isinstance(items, list) else []


def _richness(k: AgentKnowledge) -> int:
    """Count of concrete claims — a harder proxy to game than raw functionality string length."""
    return (
        len(k.functionality_citations) + len(k.context_builders) +
        len(k.upstream_consumers) + len(k.downstream_consumers) + len(k.failure_modes)
    )


def _sanitize_llm_knowledge_dict(raw: Any) -> dict:
    """Allowlist the 6 fields the prompt asks for, coercing bad types — a hallucinated extra key or null would otherwise crash validation."""
    if not isinstance(raw, dict):
        return {}
    return {
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
    }


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
            for h in hits[:10]:
                excerpt = (h.get("excerpt") or "").strip()[:1200]
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
        own_lines.extend(await _fetch_file_chunks(ctx, f, limit=8))

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
        for edge in outgoing[:8]:
            dst = edge.get("dst_symbol", "")
            dst_file = dst.split("::")[0] if "::" in dst else ""
            if dst_file and dst_file not in own_files:
                related_files.add(dst_file)

    related_lines: list[str] = []
    for f in list(related_files)[:6]:
        related_lines.extend(await _fetch_file_chunks(ctx, f, limit=3))

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

    Returns:
        List of AgentKnowledge objects, one per agent
    """
    force_ids = set(force_agent_ids or [])
    target_agents = [
        a for a in agent_flow_map.agents
        if agent_ids is None or a.id in agent_ids
    ]

    if not target_agents:
        return []

    # Per-depth caps
    caps = {
        'normal': {'queries': 3, 'llm_calls': 2, 'read_file': 2},
        'deep': {'queries': 6, 'llm_calls': 3, 'read_file': 4},
    }
    depth_cap = caps.get(depth, caps['normal'])

    # Shared semaphore for retrieval throttling (wraps search_retrieval/query only)
    semaphore = asyncio.Semaphore(4)

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
        appdata_dir = Path.home() / "AppData" / "Local" / "codespectra" / "agents" / session_id
        appdata_dir.mkdir(parents=True, exist_ok=True)

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

    prompt_sites_by_file = scan_for_prompt_sites(_resolve_repo_root(), accepted_files)

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
    _depth_cap: dict[str, int],
    accepted_files: list[str],
) -> AgentKnowledge:
    """Enrich a single agent: check cache, verify coverage, run queries/LLM, persist."""

    # 1. Check cache
    cached = await repository.get_agent_knowledge(ctx.session_id, agent_id)
    if (
        cached and
        cached.get('evidence_hash') and
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

    # 2. Compute evidence hash
    component_ids = sorted([
        c['id'] for c in evidence['component_by_agent'].get(agent_id, [])
    ])
    edges = sorted([
        (e['src'], e['dst']) for e in evidence['edges_by_agent'].get(agent_id, [])
    ])

    hash_input = '|'.join([
        ':'.join(component_ids),
        ':'.join(str(len(accepted_files))),
        ':'.join(f"{s}→{d}" for s, d in edges),
    ])
    evidence_hash = hashlib.sha256(hash_input.encode('utf-8')).hexdigest()

    # 3. Coverage check — gates the SUPPLEMENTARY pre-query only (WS-C step 2); the LLM call always runs.
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

    # Evidence the LLM cites from: components, relevant prompt sites, edges.
    context_lines = [f"Agent: {agent.label}", ""]
    if components:
        context_lines.append(f"Components ({len(components)}):")
        for c in components:
            entry = f" — entry: {c['entry_point']}" if c.get('entry_point') else ""
            context_lines.append(f"  - {c['id']} ({c.get('role', 'unknown')}) @ {c.get('file', '?')}{entry}")
    else:
        context_lines.append("Components: none")

    component_files = {c['file'] for c in components if c.get('file')}
    relevant_sites = [s for f in component_files for s in prompt_sites.get(f, [])]
    if relevant_sites:
        context_lines.append("")
        context_lines.append(f"Prompt/LLM call sites in this agent's files ({len(relevant_sites)}):")
        for site in relevant_sites[:15]:
            snippet = (site.snippet or '').replace('\n', ' ')[:150]
            context_lines.append(f"  - {site.file}:{site.line} [{site.kind}] {snippet}")

    agent_edges = evidence['edges_by_agent'].get(agent_id, [])
    if agent_edges:
        context_lines.append("")
        context_lines.append(f"File-to-file edges touching this agent ({len(agent_edges)}):")
        for edge in agent_edges[:15]:
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
    if not coverage_sufficient and ctx.snapshot_id and query_count < _depth_cap['queries']:
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
            max_tokens=12000, label=f"enrich[{agent_id}]",
            reasoning_effort="medium",
        )
        if raw is None:
            logger.warning(f"LLM round-1 failed for {agent_id}")
            return None

        # extra='ignore' drops these on model_validate, so grab them from the raw dict first
        need_more = raw.get('need_more', False)
        next_queries = raw.get('next_queries', [])[:2]

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
                f"Return the SAME JSON shape as before — functionality, functionality_citations, "
                f"context_builders, upstream_consumers, downstream_consumers, failure_modes, "
                f"need_more, next_queries — with no other fields. functionality must still be a "
                f"non-empty string, never null. If the query results above don't actually add "
                f"anything useful, keep your previous, more specific answer rather than replacing "
                f"it with a vaguer one."
            )

            raw2 = await complete_json(
                ctx.llm_client, _ENRICH_SYSTEM, round2_prompt,
                max_tokens=12000, label=f"enrich[{agent_id}]-round2",
                reasoning_effort="medium",
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
        llm_knowledge = await _run_llm_rounds(full_schema_prompt, _depth_cap)
        if llm_knowledge is None:
            llm_knowledge = AgentKnowledge(degraded=True, degraded_reason="LLM analysis failed")
    except Exception as e:
        logger.exception(f"Enrichment failed for {agent_id}: {e}")
        llm_knowledge = AgentKnowledge(degraded=True, degraded_reason=f"Enrichment exception: {str(e)}")

    # 5. Static-wins merge — structural fields never come from the LLM
    llm_knowledge.evidence_hash = evidence_hash
    llm_knowledge.query_count = query_count
    llm_knowledge.confidence = 'low' if llm_knowledge.degraded else 'high'
    llm_knowledge.generated_at = utc_now_iso()
    llm_knowledge.location = None
    llm_knowledge.components = []

    # Populate prompt_sites from evidence for this agent's components
    if agent_id in evidence.get('component_by_agent', {}):
        comp_files = {
            c['file'] for c in evidence['component_by_agent'][agent_id]
            if c.get('file')
        }
        prompt_sites_by_file = evidence.get('prompt_sites_by_file', {})
        for file, sites in prompt_sites_by_file.items():
            if file in comp_files:
                llm_knowledge.prompt_sites.extend(sites)

    # 6. Verify citations — mutates knowledge.needs_human for unverified/phantom claims
    vreport = verify_citations(llm_knowledge, _resolve_repo_root())
    if vreport.claims:
        logger.debug(
            "Citation verification for %s: %d claims, %d phantom, %d unverified",
            agent_id,
            len(vreport.claims),
            sum(1 for c in vreport.claims if c.status == "phantom"),
            sum(1 for c in vreport.claims if c.status == "unverified"),
        )

    return llm_knowledge
