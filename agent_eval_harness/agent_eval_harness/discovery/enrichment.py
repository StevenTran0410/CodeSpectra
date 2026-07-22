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
    DepRoleVerdict,
    LocationInfo,
    PromptSiteRef,
    VirtualFieldSpec,
    VirtualInputContract,
    verify_citations,
)
from agent_eval_harness.discovery.prompt_site_scan import scan_for_prompt_sites
from agent_eval_harness.instrumentation._extract import utc_now_iso
from agent_eval_harness.llm.client import LLMClient
from agent_eval_harness.mapping.agent_flow import AgentFlow, AgentFlowMap, save_agent_flow_map
from agent_eval_harness.mapping.builder.contract_harvest import (
    _dep_role_for_annotation,
    _find_entry_method,
    _harvest_constructor_dep_bindings,
    _harvest_dep_call_sites,
    _resolve_class_schema,
    _SchemaResolveCtx,
    ENTRY_METHOD_NAMES,
    harvest_component_contract,
)
from agent_eval_harness.discovery.wiring import parse_entry_suffix
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

_STRUCTURAL_PRODUCER_VERSION = 4  # Manual bump on output-schema shape change; hash won't self-invalidate.

_PROMPT_SITE_CHAR_BUDGET = 2000
_PROMPT_SITE_BLOCK_BUDGET = 20000
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
_FILE_CHUNK_CONTENT_CHAR_LIMIT = 1500
_COVERAGE_SUFFICIENT_THRESHOLD = 0.8
_CONFIDENCE_HIGH_MIN_FIELDS = 5
_CONFIDENCE_MEDIUM_MIN_FIELDS = 2

# Framework-agnostic call-classifying verbs (not service names) — generalizes across targets, no service-class dependency.
_RETRIEVAL_VERBS = frozenset({
    "retrieve", "search", "query", "fetch", "lookup", "recall", "similarity_search",
    "get_relevant_documents", "get_documents", "get_context", "get_evidence", "read",
})
_GENERATION_VERBS = frozenset({
    "complete", "completion", "chat", "generate", "predict", "stream", "embed", "ask",
    "acomplete", "achat", "invoke_llm", "create",
})


def _schema_is_evidence_shaped(schema: dict) -> bool:
    """True if a resolved return schema is retrieval-evidence-shaped (a list of structured records), not a scalar/LLM completion — no service name consulted, so it generalizes to any target."""
    if not isinstance(schema, dict):
        return False

    def _is_object(node: object) -> bool:
        return isinstance(node, dict) and (node.get("type") == "object" or "properties" in node)

    if schema.get("type") == "array" and _is_object(schema.get("items")):
        return True
    for prop in (schema.get("properties") or {}).values():
        if isinstance(prop, dict) and prop.get("type") == "array" and _is_object(prop.get("items")):
            return True
    return False


def agent_knowledge_dir(session_id: str) -> Path:
    """Write target for enriched AgentKnowledge JSON sidecars; routes through AEH_DATA_DIR (Roaming) for alignment with aeh.db and plan artifacts."""
    return aeh_data_root() / "agents" / session_id


def _legacy_agent_knowledge_dir(session_id: str) -> Path:
    """Pre-Roaming sidecar location — read-only fallback for sessions enriched before the move."""
    return Path.home() / "AppData" / "Local" / "codespectra" / "agents" / session_id


def resolve_agent_knowledge_dir(session_id: str) -> Path | None:
    """First sidecar dir that actually holds JSON; None so callers degrade to markers."""
    for candidate in (agent_knowledge_dir(session_id), _legacy_agent_knowledge_dir(session_id)):
        if candidate.exists() and any(candidate.glob("*.json")):
            return candidate
    return None


def aeh_data_root() -> Path:
    """Root directory for all AEH artifacts (db, plans, agent sidecars); uses AEH_DATA_DIR env var (set by Electron to userData/Roaming), falling back to cwd for CLI/testing."""
    return Path(os.getenv("AEH_DATA_DIR", "."))


def _parse_signature_kwargs(signature: str) -> list[tuple[str, str]]:
    """Extract kwarg (name, type_hint) tuples from a method signature, skipping self/cls; returns empty on parse ambiguity (e.g. nested generics) so the caller can drop input_contract."""
    if not signature or '(' not in signature:
        return []
    try:
        start = signature.index('(')
        end = signature.rindex(')')
        params_str = signature[start + 1:end]
        if not params_str:
            return []

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
    """Out-of-vocabulary role becomes 'unknown' at confidence 0.0 here — this only guards against a hallucinated word; the real per-component admissible-set gate runs later in _enrich_single_agent."""
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
    """AgentFlow.role is derived from components' gated roles (never asked of the LLM): the single non-worker/non-unknown role if exactly one exists, else 'worker', or 'unknown' if all are unknown."""
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
        lines.append((c.get("content") or "").strip()[:_FILE_CHUNK_CONTENT_CHAR_LIMIT])
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
    """Orchestrate concurrent enrichment of discovered agents, returns one AgentKnowledge per agent. snapshot_id must be passed by real callers (queries no-op without it); system_map YAML is the sole authority for role, written back via map_path."""
    resolved_repo_root = Path(repo_root) if repo_root is not None else None
    force_ids = set(force_agent_ids or [])
    target_agents = [
        a for a in agent_flow_map.agents
        if agent_ids is None or a.id in agent_ids
    ]

    if not target_agents:
        return []

    caps = {
        'normal': {'queries': 3, 'llm_calls': 2},
        'deep': {'queries': 6, 'llm_calls': 3},
    }
    depth_cap = caps.get(depth, caps['normal'])

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

    async def _gather(_: dict[str, Any]) -> dict[str, Any]:
        return await _gather_evidence(ctx)

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

            # A degraded agent's verdicts are never applied — one agent failing must never blank another's good role.
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

        # AgentFlow.role is derived here from the just-applied gated component roles, never asked of the LLM.
        if agent_flows_path:
            for flow in ctx.agent_flow_map.agents:
                flow.role = _derive_agent_role(flow, ctx.system_map)
            save_agent_flow_map(ctx.agent_flow_map, agent_flows_path)

        # system_map YAML is the sole authority for role — write it LAST, after every agent's md/json/DB row (and agent_flows) succeeded.
        if map_path:
            save_system_map(ctx.system_map, map_path)

    nodes.append(DagNode("persist", enrich_names, _persist))

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
    """Gather shared evidence: prompt sites, component info, edges. Returns a dict with
    prompt_sites_by_file, component_by_agent, edges_by_agent, and source_coverage (0-1 per agent)."""
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
                    'entry_kind': comp.entry_kind,
                    'is_tool': comp.is_tool,
                    'constructor_fanout': comp.constructor_fanout,
                    'fan_in': len(comp.upstream),
                    'fan_out': len(comp.downstream),
                })
        component_by_agent[agent.id] = agent_components

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


async def _load_referenced_type_files(
    ctx: _EnrichmentContext, type_name: str, asts: dict, seen: set, depth: int = 0,
) -> None:
    """Transitively parses source files of nested CLASS-typed fields into `asts` so `_resolve_class_schema` can expand cross-file nested types instead of leaving empty placeholders; depth+`seen` bounded, generic to any capitalized annotation."""
    import ast as _ast
    import re as _re
    if depth > 4:
        return
    _prims = {"str", "int", "float", "bool", "bytes", "None", "Any", "dict", "list", "tuple",
              "set", "frozenset", "datetime", "date", "Optional", "Union", "Literal", "Sequence",
              "Mapping", "Iterable", "UUID", "Decimal", "Path", "Enum"}
    cls = None
    for tree in list(asts.values()):
        cls = next((n for n in _ast.walk(tree)
                    if isinstance(n, _ast.ClassDef) and n.name == type_name), None)
        if cls is not None:
            break
    if cls is None:
        return
    referenced: set[str] = set()
    for item in cls.body:
        if isinstance(item, _ast.AnnAssign) and item.annotation is not None:
            for nm in _re.findall(r"[A-Za-z_][A-Za-z0-9_]*", _ast.unparse(item.annotation)):
                if nm and nm[0].isupper() and nm not in _prims:
                    referenced.add(nm)
    for nm in referenced:
        if nm in seen:
            continue
        seen.add(nm)
        already = any(
            any(isinstance(n, _ast.ClassDef) and n.name == nm for n in _ast.walk(t))
            for t in asts.values()
        )
        if not already:
            try:
                resp = await ctx.client.search_repo_map(ctx.snapshot_id, q=nm)
                syms = resp.get("symbols", []) if isinstance(resp, dict) else (resp or [])
                rows = [r for r in syms if r.get("kind") == "class" and r.get("name") == nm]
                rp = rows[0].get("rel_path") if rows else None
                if rp:
                    parsed = parse_python_source((ctx.repo_root / rp) if ctx.repo_root else Path(rp))
                    if parsed is not None:
                        asts[Path(rp)] = parsed[1]
            except Exception:
                continue
        await _load_referenced_type_files(ctx, nm, asts, seen, depth + 1)


async def _resolve_type_schema_via_symbol_index(
    ctx: _EnrichmentContext,
    type_name: str,
) -> tuple[dict, str, float] | None:
    """Core: resolve schema for a type name via symbol-index query. Reusable for both return types and input kwargs."""
    if not ctx.snapshot_id or not type_name:
        return None

    try:
        def _symbols(resp):
            # search_repo_map returns {"symbols": [...]} (a dict), not a bare list.
            return resp.get("symbols", []) if isinstance(resp, dict) else (resp or [])

        type_syms = _symbols(await ctx.client.search_repo_map(ctx.snapshot_id, q=type_name))
        type_rows = [r for r in type_syms if r.get("kind") == "class" and r.get("name") == type_name]
        type_rows = type_rows or type_syms
        rel_path = type_rows[0].get("rel_path") if type_rows else None
        if not rel_path:
            return None

        disk_path = (ctx.repo_root / rel_path) if ctx.repo_root else Path(rel_path)
        parsed = parse_python_source(disk_path)
        if parsed is None:
            return None

        asts = {Path(rel_path): parsed[1]}
        # Transitively load nested class-typed fields' source files so cross-file types resolve too.
        await _load_referenced_type_files(ctx, type_name, asts, {type_name})
        resolve_ctx = _SchemaResolveCtx(
            asts=asts,
            files_root=ctx.repo_root,
            visited=set(),
            depth=0,
            conventions=None,
        )

        schema_result = _resolve_class_schema(type_name, resolve_ctx)
        if schema_result is None:
            return None

        schema, citation = schema_result
        return schema, citation, 0.85

    except Exception as e:
        logger.debug(f"Error resolving schema for {type_name}: {e}")
        return None


async def _resolve_return_schema_via_symbol_index(
    ctx: _EnrichmentContext,
    dep_annotation: str,
    method: str,
) -> tuple[dict, str, float] | None:
    """Resolve return type of a method via symbol-index query."""
    if not ctx.snapshot_id or not dep_annotation or not method:
        return None

    try:
        import re as _re
        import ast as _ast

        def _symbols(resp):
            return resp.get("symbols", []) if isinstance(resp, dict) else (resp or [])

        _wrappers = {"Optional", "Union", "list", "List", "dict", "Dict", "tuple", "Tuple",
                     "set", "Set", "None", "Awaitable", "Coroutine", "Sequence", "Iterable"}
        cls_syms = _symbols(await ctx.client.search_repo_map(ctx.snapshot_id, q=dep_annotation))
        cls_rows = [r for r in cls_syms if r.get("kind") == "class" and r.get("name") == dep_annotation]
        cls_rel = cls_rows[0].get("rel_path") if cls_rows else None
        if not cls_rel:
            return None
        cls_parsed = parse_python_source((ctx.repo_root / cls_rel) if ctx.repo_root else Path(cls_rel))
        if cls_parsed is None:
            return None

        return_type_name = None
        for node in _ast.walk(cls_parsed[1]):
            if isinstance(node, _ast.ClassDef) and node.name == dep_annotation:
                for item in node.body:
                    if (isinstance(item, (_ast.FunctionDef, _ast.AsyncFunctionDef))
                            and item.name == method and item.returns is not None):
                        toks = _re.findall(r"[A-Za-z_][A-Za-z0-9_]*", _ast.unparse(item.returns))
                        picks = [t for t in toks if t not in _wrappers]
                        return_type_name = picks[0] if picks else (toks[0] if toks else None)
                        break
                break

        if not return_type_name:
            return None

        return await _resolve_type_schema_via_symbol_index(ctx, return_type_name)

    except Exception as e:
        logger.debug(f"Error resolving schema for {dep_annotation}.{method}(): {e}")
        return None


async def _harvest_virtual_inputs_static(
    ctx: _EnrichmentContext,
    cls: Any,
    entry: Any,
    own_tree: Any,
    own_file: Path,
    asts: dict[Path, Any],
) -> list[VirtualInputContract]:
    """Static harvest of virtual inputs from constructor deps and call sites."""
    if cls is None or entry is None:
        return []

    bindings = _harvest_constructor_dep_bindings(cls)
    if not bindings:
        return []

    call_sites = _harvest_dep_call_sites(entry, bindings, own_tree, own_file, asts, ctx.repo_root)
    if not call_sites:
        return []

    load_signals = {}
    try:
        import yaml
        signals_file = Path(__file__).parent / "contract_signals.yaml"
        if signals_file.exists():
            with open(signals_file) as f:
                signals = yaml.safe_load(f) or {}
                load_signals = signals.get("dep_role_keywords", {})
    except Exception:
        pass

    contracts: list[VirtualInputContract] = []
    call_sites_by_dep = {}
    for cs in call_sites:
        if cs.dep_attr not in call_sites_by_dep:
            call_sites_by_dep[cs.dep_attr] = []
        call_sites_by_dep[cs.dep_attr].append(cs)

    for binding in bindings:
        if binding.attr not in call_sites_by_dep:
            continue

        calls = call_sites_by_dep[binding.attr]
        methods = list({c.method for c in calls})
        citations = [c.citation for c in calls]

        # Resolve the return schema first — it's both the field source and the primary structural signal for whether this dep is retrieval-evidence.
        schema = {}
        if binding.annotation and methods:
            schema_result = await _resolve_return_schema_via_symbol_index(
                ctx, binding.annotation, methods[0]
            )
            if schema_result:
                schema, _, _ = schema_result

        # Generic virtualization decision (no target-specific service names): evidence-shaped return type, or fallback retrieval-verb/keyword signal; LLM/generation calls are always excluded (evaluated live, never stubbed).
        method_names = {m.lower() for m in methods}
        is_generation = any(v in m for m in method_names for v in _GENERATION_VERBS)
        is_retrieval_verb = any(v in m for m in method_names for v in _RETRIEVAL_VERBS)
        keyword_role = _dep_role_for_annotation(binding.annotation or "", load_signals) if binding.annotation else "unknown"
        if is_generation:
            continue
        if _schema_is_evidence_shaped(schema):
            dep_role = "retrieval"
        elif not schema and (is_retrieval_verb or keyword_role == "retrieval"):
            dep_role = "retrieval"
        else:
            continue

        fields = []
        if schema and "properties" in schema:
            for fname, fschema in schema.get("properties", {}).items():
                fields.append(VirtualFieldSpec(
                    name=fname,
                    schema=fschema,
                    provenance="annotation",
                    confidence=0.85,
                ))

        contracts.append(VirtualInputContract(
            name=f"bundle",
            dep_param=binding.param,
            dep_attr=binding.attr,
            dep_annotation=binding.annotation or "",
            dep_role=dep_role,
            methods_called=methods,
            call_sites=citations,
            fields=fields,
        ))

    return contracts


async def _resolve_input_kwarg_schemas(
    ctx: _EnrichmentContext,
    input_contract: list[ContractArg],
) -> dict[str, dict]:
    """Resolve schemas for input kwargs that have resolvable type annotations."""
    import re as _re

    input_schemas = {}
    for arg in input_contract:
        type_hint = arg.type_hint or ""
        if not type_hint:
            continue

        base_type = _re.search(r"[A-Za-z_][A-Za-z0-9_]*", type_hint)
        if not base_type:
            continue

        type_name = base_type.group(0)

        if type_name in ("str", "int", "float", "bool", "dict", "list", "Any", "None"):
            continue

        schema_result = await _resolve_type_schema_via_symbol_index(ctx, type_name)
        if schema_result:
            schema, _, _ = schema_result
            input_schemas[arg.kwarg] = schema

    return input_schemas


async def _enrich_single_agent(
    agent_id: str,
    evidence: dict[str, Any],
    ctx: _EnrichmentContext,
    depth_cap: dict[str, int],
    accepted_files: list[str],
) -> AgentKnowledge:
    """Enrich a single agent: check cache, verify coverage, run queries/LLM, persist."""

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

    cached = await repository.get_agent_knowledge(ctx.session_id, agent_id)
    if (
        cached and
        cached.get('evidence_hash') == evidence_hash and
        agent_id not in ctx.force_agent_ids
    ):
        try:
            cached_json_path = Path(cached.get('json_path', ''))
            if cached_json_path.exists():
                cached_data = json.loads(cached_json_path.read_text(encoding='utf-8'))
                return AgentKnowledge.from_json(cached_data)
        except Exception as e:
            logger.warning(f"Failed to deserialize cached knowledge for {agent_id}: {e}")

    # Gates the SUPPLEMENTARY pre-query only; the LLM call always runs.
    prompt_sites = evidence['prompt_sites_by_file']
    components = evidence['component_by_agent'].get(agent_id, [])
    component_files = {c['file'] for c in components if c.get('file')}
    prompt_site_files = set(prompt_sites.keys()) & component_files

    coverage_sufficient = (
        len(components) > 0 and
        len(prompt_site_files) > 0 and
        evidence['source_coverage'].get(agent_id, 0) >= _COVERAGE_SUFFICIENT_THRESHOLD
    )

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

    # Static virtual input harvest: populate constructor_dep_roles and virtual_inputs.
    virtual_inputs_list: list[VirtualInputContract] = []
    if components and component_files:
        try:
            from agent_eval_harness.mapping.builder.types import parse_python_source
            asts: dict[Path, Any] = {}
            for comp in components:
                if not comp.get('file'):
                    continue
                comp_file = Path(comp['file'])
                if comp_file not in asts:
                    disk_path = ctx.repo_root / comp_file if ctx.repo_root else comp_file
                    parsed = parse_python_source(disk_path)
                    if parsed is not None:
                        asts[comp_file] = parsed[1]

            for comp in components:
                if not comp.get('file'):
                    continue
                comp_file = Path(comp['file'])
                if comp_file not in asts:
                    continue
                tree = asts[comp_file]

                from agent_eval_harness.mapping.builder.contract_harvest import (
                    _find_class, _find_entry_method, _find_function
                )
                _, _, suffix = comp.get('entry_point', '').partition(':')
                owner, name = parse_entry_suffix(suffix)
                cls = entry = None
                if comp.get('entry_kind') == 'function':
                    entry = _find_function(tree, name)
                else:  # class or bound-method (Owner.method) — resolve the owning class then the named method
                    cls = _find_class(tree, owner or name)
                    if cls is not None:
                        entry = _find_entry_method(cls, name if owner else None)

                if cls is None and entry is None:
                    continue

                virtual_contracts = await _harvest_virtual_inputs_static(
                    ctx, cls, entry, tree, comp_file, asts
                )
                virtual_inputs_list.extend(virtual_contracts)

        except Exception as e:
            logger.debug(f"Virtual input harvest failed for {agent_id}: {e}")

    if not coverage_sufficient and ctx.snapshot_id and query_count < depth_cap['queries']:
        pre_query_block = await _execute_queries(ctx, [agent.label])
        query_count += 1
        context_lines.append("")
        context_lines.append("=== SUPPLEMENTARY — from system query (may include unrelated files; cross-check against ground truth above before citing) ===")
        context_lines.append(pre_query_block)

    context_str = "\n".join(context_lines)

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
  "next_queries": [],
  "virtual_input_complete": true,
  "input_contract_examples": {{"<kwarg_name>": "<example_value>"}},
  "virtual_input_fields": [{{"dep_attr": "<attr_name>", "field": "<field_name>", "file": "<path>", "line": 0, "example": "<value>"}}]
}}

If you need more information to provide complete answers, set need_more to true and list follow-up queries (max 2).
If evidence or other internally-retrieved data reaches this agent (virtual inputs), virtual_input_complete should only be false if:
  - The base static analysis found no schema for the virtual input, AND
  - The agent's code references fields that are not yet known.
In that case, list those missing fields in virtual_input_fields with (file, line, example).
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
        virtual_input_complete = raw.get('virtual_input_complete', True)
        input_contract_examples = raw.get('input_contract_examples', {})
        virtual_input_fields = raw.get('virtual_input_fields', [])

        round_knowledge = AgentKnowledge.model_validate(_sanitize_llm_knowledge_dict(raw))

        # Fill example values and merge LLM-tier fields onto static-tier deps.
        if input_contract_examples and isinstance(input_contract_examples, dict):
            for arg in round_knowledge.input_contract:
                if arg.kwarg in input_contract_examples:
                    arg.example = input_contract_examples[arg.kwarg]

        if virtual_input_fields and isinstance(virtual_input_fields, list):
            for llm_field in virtual_input_fields:
                if not isinstance(llm_field, dict):
                    continue
                dep_attr = llm_field.get('dep_attr', '')
                field_name = llm_field.get('field', '')

                for vi in round_knowledge.virtual_inputs:
                    if vi.dep_attr == dep_attr:
                        if not any(f.name == field_name for f in vi.fields):
                            # LLM-sourced fields stay low-confidence; verify_citations (called later) checks them.
                            vi.fields.append(VirtualFieldSpec(
                                name=field_name,
                                schema={},
                                provenance="prompt",
                                example=llm_field.get('example'),
                                confidence=0.3,
                                needs_human=True,
                            ))
                        break

        # Extend-gate ORs into the existing need_more flag (reuses the cap, no separate counter).
        if not virtual_input_complete:
            need_more = True

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

                # Apply the same merge logic to round2.
                input_contract_examples_r2 = raw2.get('input_contract_examples', {})
                virtual_input_fields_r2 = raw2.get('virtual_input_fields', [])

                if input_contract_examples_r2 and isinstance(input_contract_examples_r2, dict):
                    for arg in round2_knowledge.input_contract:
                        if arg.kwarg in input_contract_examples_r2:
                            arg.example = input_contract_examples_r2[arg.kwarg]

                if virtual_input_fields_r2 and isinstance(virtual_input_fields_r2, list):
                    for llm_field in virtual_input_fields_r2:
                        if not isinstance(llm_field, dict):
                            continue
                        dep_attr = llm_field.get('dep_attr', '')
                        field_name = llm_field.get('field', '')

                        for vi in round2_knowledge.virtual_inputs:
                            if vi.dep_attr == dep_attr:
                                if not any(f.name == field_name for f in vi.fields):
                                    # Note: LLM fields stay low-confidence; verify_citations is called after merge
                                    vi.fields.append(VirtualFieldSpec(
                                        name=field_name,
                                        schema={},
                                        provenance="prompt",
                                        example=llm_field.get('example'),
                                        confidence=0.3,
                                        needs_human=True,
                                    ))
                                break

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

    # Mirrors roles.py's structural subtraction: the prompt lists the admissible set, the code enforces it.
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

    # Structural fields never come from the LLM.
    llm_knowledge.evidence_hash = evidence_hash
    llm_knowledge.query_count = query_count
    llm_knowledge.generated_at = utc_now_iso()
    llm_knowledge.virtual_inputs = virtual_inputs_list

    seen_deps = set()
    for vi in virtual_inputs_list:
        if vi.dep_attr not in seen_deps:
            llm_knowledge.constructor_dep_roles.append(DepRoleVerdict(
                dep=vi.dep_attr,
                role=vi.dep_role,
                confidence=0.85,
                reasoning=f"Harvested from call sites: {', '.join(vi.methods_called)}"
            ))
            seen_deps.add(vi.dep_attr)

    # code_symbols rows keyed by file, reused for both the structural fill and citation verification.
    symbols_by_file: dict[str, list[dict]] = {}

    if ctx.snapshot_id and agent_id in evidence.get('component_by_agent', {}):
        components_list = evidence['component_by_agent'][agent_id]
        if components_list:
            try:
                comp_files = {c['file'] for c in components_list if c.get('file')}
                if comp_files:
                    for comp_file in comp_files:
                        try:
                            file_basename = Path(comp_file).name
                            symbols_response = await ctx.client.search_repo_map(ctx.snapshot_id, q=file_basename)
                            symbols = symbols_response.get('symbols', [])
                            symbols_by_file[comp_file] = symbols

                            # Resolve THIS comp_file's own component (not components_list[0]) so multi-file agents don't cross-wire.
                            comp_for_harvest = next(
                                (c for c in components_list if c.get('file') == comp_file), None
                            )
                            if comp_for_harvest and comp_for_harvest.get('entry_point'):
                                _, _, suffix = comp_for_harvest['entry_point'].partition(':')
                                owner, name = parse_entry_suffix(suffix)
                                entry_kind = comp_for_harvest.get('entry_kind')

                                class_row = method_row = function_row = None
                                if entry_kind == 'function':
                                    function_row = next(
                                        (s for s in symbols if s.get('kind') == 'function' and s.get('name') == name),
                                        None,
                                    )
                                else:
                                    target_class = owner or name
                                    explicit_method = name if owner else None
                                    class_row = next(
                                        (s for s in symbols if s.get('kind') == 'class' and s.get('name') == target_class),
                                        None,
                                    )
                                    if explicit_method:
                                        method_row = next(
                                            (s for s in symbols if s.get('parent_name') == target_class
                                             and s.get('kind') == 'method' and s.get('name') == explicit_method),
                                            None,
                                        )
                                    else:
                                        for sym in symbols:
                                            if (sym.get('parent_name') == target_class and sym.get('kind') == 'method'
                                                    and sym.get('name') in ENTRY_METHOD_NAMES):
                                                if method_row is None or ENTRY_METHOD_NAMES.index(sym.get('name')) < ENTRY_METHOD_NAMES.index(method_row.get('name')):
                                                    method_row = sym

                                if (class_row and method_row) or function_row:
                                    if function_row:
                                        # Collapse class-span + method-span onto the function's own span (Decision #10).
                                        llm_knowledge.location = LocationInfo(
                                            file=comp_file,
                                            line_start=function_row.get('line_start', 0),
                                            line_end=function_row.get('line_end', 0),
                                            entry_method=function_row.get('name', ''),
                                            entry_line=function_row.get('line_start', 0),
                                        )
                                    else:
                                        llm_knowledge.location = LocationInfo(
                                            file=comp_file,
                                            line_start=class_row.get('line_start', 0),
                                            line_end=class_row.get('line_end', 0),
                                            entry_method=method_row.get('name', ''),
                                            entry_line=method_row.get('line_start', 0),
                                        )

                                    # Harvest kwargs from the real source AST (not the symbol-index signature, which drops type hints) so nested field schemas can resolve.
                                    invocation_contract = None
                                    try:
                                        comp_id = comp_for_harvest.get('id', agent_id)
                                        entry_pt = comp_for_harvest.get('entry_point', '')
                                        temp_component = Component(id=comp_id, role='unknown', entry_point=entry_pt, file=comp_file, entry_kind=entry_kind)
                                        comp_path = (ctx.repo_root / comp_file) if ctx.repo_root else Path(comp_file)
                                        if comp_path.exists():
                                            parsed = parse_python_source(comp_path)
                                            if parsed:
                                                asts = {comp_path: parsed[1]}
                                                invocation_contract, output, _, _, _ = harvest_component_contract(temp_component, asts, files_root=ctx.repo_root)
                                                if output:
                                                    llm_knowledge.output_contract = output
                                    except Exception as e:
                                        logger.debug(f"Failed to harvest contracts for {agent_id}: {e}")

                                    # Build input_contract from AST-harvested kwargs (real type hints) if available
                                    if invocation_contract and invocation_contract.kwargs:
                                        llm_knowledge.input_contract = [
                                            ContractArg(kwarg=k.name, source_kind='ast', type_hint=k.annotation or '', example='')
                                            for k in invocation_contract.kwargs
                                        ]
                                        llm_knowledge.input_schemas = await _resolve_input_kwarg_schemas(ctx, llm_knowledge.input_contract)
                                    elif method_row is not None:
                                        # Fallback to truncated signature parsing (backward compat; class/bound-method only).
                                        sig = method_row.get('signature', '')
                                        kwargs_with_hints = _parse_signature_kwargs(sig)
                                        if kwargs_with_hints:
                                            llm_knowledge.input_contract = [
                                                ContractArg(kwarg=name, source_kind='signature', type_hint=hint, example='')
                                                for name, hint in kwargs_with_hints
                                            ]
                                            llm_knowledge.input_schemas = await _resolve_input_kwarg_schemas(ctx, llm_knowledge.input_contract)

                                anchor_row = class_row or function_row
                                if anchor_row:
                                    for comp in components_list:
                                        if comp.get('file') == comp_file:
                                            llm_knowledge.components.append(ComponentRef(
                                                id=comp['id'],
                                                role=comp.get('role', 'unknown'),
                                                file=comp_file,
                                                line=anchor_row.get('line_start', 0),
                                            ))
                        except Exception as e:
                            logger.warning(f"Failed to fill structural fields for {agent_id} from {comp_file}: {e}")
            except Exception as e:
                logger.warning(f"Failed to fill structural fields for {agent_id}: {e}")
    else:
        if not ctx.snapshot_id:
            logger.warning(f"No snapshot_id for {agent_id} — cannot fill structural fields")

    if agent_id in evidence.get('component_by_agent', {}):
        comp_files = {
            c['file'] for c in evidence['component_by_agent'][agent_id]
            if c.get('file')
        }
        prompt_sites_by_file = evidence.get('prompt_sites_by_file', {})
        for file, sites in prompt_sites_by_file.items():
            if file in comp_files:
                llm_knowledge.prompt_sites.extend(PromptSiteRef(**asdict(s)) for s in sites)

    # Mutates knowledge.needs_human for unverified/phantom claims.
    verify_root = _resolve_repo_root(ctx.repo_root)
    if verify_root is None:
        logger.warning(
            "No target repo root — citation verification skipped for %s; the LLM's file:line "
            "claims are recorded UNCHECKED. Pass repo_root= or set AEH_REPO_ROOT.",
            agent_id,
        )
    else:
        # Fetch symbols for any cited file not already pulled, so span-based citation resolution covers cross-file citations too.
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

    if llm_knowledge.degraded:
        llm_knowledge.confidence = 'low'
    elif llm_knowledge.needs_human:
        # Hard invariant: needs_human non-empty => never 'high'
        llm_knowledge.confidence = 'medium'
    else:
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
        if field_count >= _CONFIDENCE_HIGH_MIN_FIELDS:
            llm_knowledge.confidence = 'high'
        elif field_count >= _CONFIDENCE_MEDIUM_MIN_FIELDS:
            llm_knowledge.confidence = 'medium'
        else:
            llm_knowledge.confidence = 'low'

    return llm_knowledge
