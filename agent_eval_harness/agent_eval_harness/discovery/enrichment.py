"""Enrichment DAG: gather evidence once, fan out an LLM pass per agent, persist."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent_eval_harness.config import ContractConventions
from agent_eval_harness.datasets.archetype_vocabulary import EVIDENCE_CASE_KEY
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
from agent_eval_harness.llm.client import LLMClient, LLMMessage
from agent_eval_harness.mapping.agent_flow import AgentFlow, AgentFlowMap, save_agent_flow_map
from agent_eval_harness.mapping.builder.contract_harvest import (
    _CONFIG_PARAMS,
    _dep_role_for_annotation,
    _entry_schema_scope,
    _find_entry_method,
    _harvest_constructor_dep_bindings,
    _harvest_dep_call_sites,
    _harvest_kwargs,
    _resolve_class_schema,
    _resolve_entry,
    _SchemaResolveCtx,
    harvest_component_contract,
    harvest_contracts,
)
from agent_eval_harness.discovery.wiring import parse_entry_suffix
from agent_eval_harness.mapping.builder.prompts import ROLE_TAXONOMY
from agent_eval_harness.mapping.builder.roles import (
    ROLE_CONFIDENCE_THRESHOLD,
    VALID_ROLES,
    admissible_roles,
    forced_role,
    structural_facts,
)
from agent_eval_harness.mapping.builder.types import parse_python_source
from agent_eval_harness.mapping.system_map import Component, SystemMap, save_system_map
from agent_eval_harness.planning.agentic_planner import (
    DagNode,
    _apply_virtual_bindings,
    complete_json,
    complete_json_with_raw,
    run_dag,
)
from agent_eval_harness.planning.contract import EvaluationContract, KwargSpec, OutputContract
from agent_eval_harness.store import repository

logger = logging.getLogger("agent_eval_harness.discovery.enrichment")

# Derived from the hash_input field shape below, not a manually bumped integer -- adding, removing, or
# renaming a field in _enrich_single_agent's hash_input automatically invalidates every cached entry.
_HASH_INPUT_FIELDS = (
    "component_ids", "accepted_file_count", "edges",
    "component_attrs:id,motif,is_tool,constructor_fanout,conditional_downstream_count,call_downstream_count",
    "eval_contract:v1",  # bump this token whenever a contract_harvest/enrichment fix changes harvested output for unchanged structure -- busts every persisted sidecar exactly once.
)
_STRUCTURAL_PRODUCER_VERSION = hashlib.sha256("|".join(_HASH_INPUT_FIELDS).encode("utf-8")).hexdigest()[:12]

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
_FAN_IN_SINK_PERCENTILE = 0.9  # a component at/above this percentile of the MAP's own fan-in distribution is a sink
_MAX_SYMBOL_EDGES_FOLLOWED = 8
_ENRICH_MAX_TOKENS = 16000
_ENRICH_REASONING_EFFORT = "high"  # role classification benefits from reasoning; on DeepSeek's effort_toggle, only "high"/"max" actually enable thinking ("medium" left it OFF)
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
_OPAQUE_CONTAINER_TYPES = frozenset({"dict", "list"})
_PRIMITIVE_TYPES = frozenset({"str", "int", "float", "bool", "Any", "None"})
# ContractArg.source_kind vocabulary (stays a permissive str -- see agent_knowledge.ContractArg).
# "internally-retrieved" is reserved for VirtualInputContract-adjacent evidence; a regular kwarg is
# never internally-retrieved by definition, so this classifier never emits it.
_PROVENANCE_CONFIG_STATIC = "config-static"
_PROVENANCE_RUNTIME_STATE = "runtime-state"
_PROVENANCE_UPSTREAM_OUTPUT = "upstream-agent-output"
# recognized_entry_methods needs an AST tree; this fallback only ever sees search-index symbol rows.
_SEARCH_INDEX_ENTRY_METHOD_NAMES = ("run", "run_async", "__call__")



_ID_DECORATION_RE = re.compile(r"\s+[@(]")


def _undecorate_component_id(raw: str) -> str:
    """Strip the prompt's display decoration off a component id the model echoed verbatim
    (`id="x"`, `x @ path/to.py`, `x (file: ...)`) so the verdict still matches a real component."""
    s = raw.strip()
    quoted = re.fullmatch(r'id\s*=\s*"([^"]+)"', s) or re.fullmatch(r'"([^"]+)"', s)
    if quoted:
        s = quoted.group(1)
    return _ID_DECORATION_RE.split(s, maxsplit=1)[0].strip().strip('"')


def _resolve_entry_point_id(raw: str, structural_by_id: dict[str, dict]) -> str:
    """Map an entry-point-shaped id the model echoed (`module.path:Owner.member`) back to the real
    component id — the prompt lists both `id` and `entry:`, and the model sometimes copies the latter,
    which used to cost every component of that agent its role."""
    s = raw.strip()
    for cid, comp in structural_by_id.items():
        if s == (comp.get('entry_point') or ''):
            return cid
    tail = s.rsplit(':', 1)[-1].rsplit('.', 1)[-1].strip()
    return tail if tail in structural_by_id else raw


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
    "each role taken from THAT component's own admissible set. Each verdict's \"id\" MUST be the "
    "component's bare id — the FIRST token on its evidence line, before the ' @ ' — never the "
    "longer dotted/colon-separated path shown after 'entry:'; an id in any other form is discarded "
    "and that component loses its role.\n"
    "Decide the role from the component's PRIMARY purpose in this system, read from its own code — "
    "never from a library or framework name, and never from behaviour that every component happens "
    "to share. 'worker' is the FALLBACK: first check whether a more specific role fits and pick "
    "'worker' only when none does — but do NOT force a component into a specialized role either; if "
    "its defining job is simply turning inputs into an output, 'worker' is the correct answer, not a "
    "stretch. Each role below has a generic, framework-neutral example; match on the underlying "
    "behaviour, not on the example's wording.\n"
    "- retrieval_agent: a component that DECIDES what to fetch at runtime — it plans, derives, "
    "rewrites, or iterates on its own queries/lookups, or manages a set of fetch/tool calls — so the "
    "retrieval strategy itself is part of its work and worth evaluating. This holds even when the "
    "component also produces its own output. The discriminator is WHERE THE QUERY COMES FROM: a "
    "component that fetches with a FIXED, predetermined query (or does not retrieve at all) and "
    "simply transforms the result is a worker, not a retrieval_agent — its retrieval carries no "
    "decision. Example: a component that, given a goal, generates several targeted queries, runs "
    "them, and gathers the results — as opposed to one that always issues the same hardcoded query.\n"
    "- worker: turns its inputs into a structured output, typically via one model call or a "
    "computation — the general-purpose producer. Fetching with a fixed/hardcoded query (even several "
    "of them) is still worker. Example: a component that reads some context and emits a summary, a "
    "classification, or a structured record.\n"
    "- validator: judges ANOTHER component's output — scores it, checks it, or flags problems — "
    "rather than merely validating its own output's shape. Example: a component that reviews "
    "another component's answer and assigns it a quality score.\n"
    "- writer: produces the system's final, outward-facing artifact, with nothing downstream "
    "consuming it. Example: the terminal component that composes the final answer returned to the "
    "caller.\n"
    "- orchestrator: decides WHICH other components run and in what order. Example: a coordinator "
    "that routes control to different sub-components based on the current state.\n"
    "- input_guard.rule / input_guard.llm: screens, validates, or rejects the input before real "
    "work begins — by fixed rules, or by a model, respectively. Example: a component that rejects an "
    "out-of-scope or malformed request up front.\n"
    "- tool: a discrete callable capability an agent invokes on demand, not a stage in the main "
    "flow. Example: a function the agent can call to compute a value or look up a single fact.\n"
    "- unknown: use ONLY when the evidence genuinely does not distinguish between roles.\n\n"
    "CONTROL FLOW:\n"
    "Identify any component whose OUTPUT re-enters the system as a control decision — "
    "i.e. it drives a repeat/retry loop, gates further work, or selects which component runs next. "
    "You MUST set the \"control_motif\" field for such a component: "
    "\"loop\" if its output drives a repeat/retry/iteration loop (the component runs again, or "
    "re-runs other components, until a condition is met); "
    "\"branch\" if its output selects which component runs next (dynamic dispatch/routing); "
    "null for an ordinary component whose output only flows forward to the next fixed step. "
    "A component IS a loop/branch controller even when the loop/dispatch is written in plain code "
    "in a driver/orchestrator that reads this component's output — judge by what the output DOES, "
    "not by whether this component's own body contains the loop keyword. "
    "This is about SYSTEM control flow BETWEEN components: a component that merely refines or iterates "
    "INTERNALLY (its own multi-round retrieval/refinement) is NOT a controller — use null for it. "
    "When control_motif is loop or branch, also describe its termination condition and the "
    "feedback channel (what output it reads to decide). Use generic control-flow terms only.\n\n"
    "Do NOT add fields like "
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
        'output_schema_from_prompt': raw.get('output_schema_from_prompt') if isinstance(raw.get('output_schema_from_prompt'), dict) else None,
        'output_enum_values_from_prompt': raw.get('output_enum_values_from_prompt') if isinstance(raw.get('output_enum_values_from_prompt'), dict) else {},
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
    ctx: _EnrichmentContext, own_files: set[str], known_edge_files: set[str],
    boost_control_flow: bool = False,
) -> tuple[list[str], list[str]]:
    """Ground-truth evidence for one agent: own file(s) plus related files (known edges + 1-hop symbol-graph calls), all fetched directly by path, zero search.
    When boost_control_flow is set, raise the own- AND related-file chunk caps so a control loop — which may live in a *driver* file that reads this component's output, not in the component's own body — surfaces into evidence."""
    own_chunk_limit = 16 if boost_control_flow else _OWN_FILE_CHUNK_LIMIT
    related_chunk_limit = 6 if boost_control_flow else _RELATED_FILE_CHUNK_LIMIT
    own_lines: list[str] = []
    for f in own_files:
        own_lines.extend(await _fetch_file_chunks(ctx, f, limit=own_chunk_limit))

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
    for f in sorted(related_files)[:_MAX_RELATED_FILES]:
        related_lines.extend(await _fetch_file_chunks(ctx, f, limit=related_chunk_limit))

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
    system_type: str | None = None


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
    system_type: str | None = None,
    conventions: ContractConventions | None = None,
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
        system_type=system_type,
    )

    async def _gather(_: dict[str, Any]) -> dict[str, Any]:
        return await _gather_evidence(ctx)

    nodes: list[DagNode] = [DagNode("gather", [], _gather)]

    accepted_files = [
        item["file"] if isinstance(item, dict) else item
        for item in accepted_with_annotations
    ]

    # Cross-agent EvaluationContract harvest: pure AST, no LLM, cheap -- runs once here (not per-agent,
    # not at plan-time) so Stage 3 can read it from the sidecar instead of re-harvesting.
    harvest_files = [
        (resolved_repo_root / f) if resolved_repo_root else Path(f) for f in accepted_files
    ]
    contracts = harvest_contracts(system_map, agent_flow_map, harvest_files, resolved_repo_root, conventions)

    for agent in target_agents:
        agent_id = agent.id
        enrich_name = f"enrich:{agent_id}"

        async def _enrich(
            results: dict[str, Any],
            agent_id_: str = agent_id,
        ) -> AgentKnowledge:
            try:
                evidence = results["gather"]
                knowledge = await _enrich_single_agent(
                    agent_id_,
                    evidence,
                    ctx,
                    depth_cap,
                    accepted_files,
                    contracts,
                )
                contract = contracts.get(agent_id_)
                if contract:
                    _apply_virtual_bindings(contract, knowledge.virtual_inputs)
                    knowledge.evaluation_contract = contract
                return knowledge
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


def _fan_in_sink_threshold(system_map: SystemMap) -> float | None:
    """A fixed fan-in constant fired on 3 infra services and missed a real terminal writer in an
    earlier iteration (see roles.py's structural_facts demotion) — degree alone needs the map's own
    distribution, not a cross-target number. None when the map has too few components or no real
    concentration (every fan-in <= 1) to say anything meaningful."""
    fan_ins = sorted(len(c.upstream) for c in system_map.components)
    if len(fan_ins) < 2:
        return None
    idx = max(0, math.ceil(_FAN_IN_SINK_PERCENTILE * len(fan_ins)) - 1)
    threshold = fan_ins[idx]
    return float(threshold) if threshold > 1 else None


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
                    'conditional_downstream': list(comp.conditional_downstream),
                    'call_downstream': list(comp.call_downstream),
                    'motif': comp.motif,
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
        'fan_in_sink_threshold': _fan_in_sink_threshold(ctx.system_map),
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


async def _resolve_kwarg_evidence_schema(ctx: _EnrichmentContext, annotation: str) -> dict:
    """Resolve a signature kwarg's own annotation to a schema for the evidence-shape check --
    unwraps a single list[X]/List[X] layer before resolving X via _resolve_type_schema_via_symbol_index,
    since a bare 'list' base type never carries field shape on its own."""
    import re as _re

    list_match = _re.match(r"^\s*(?:list|List)\[\s*([A-Za-z_][A-Za-z0-9_]*)\s*\]\s*$", annotation or "")
    if list_match:
        item_name = list_match.group(1)
        if item_name in _PRIMITIVE_TYPES:
            return {}
        item_schema_result = await _resolve_type_schema_via_symbol_index(ctx, item_name)
        if not item_schema_result:
            return {}
        item_schema, _, _ = item_schema_result
        return {"type": "array", "items": item_schema}

    base_match = _re.search(r"[A-Za-z_][A-Za-z0-9_]*", annotation or "")
    if not base_match or base_match.group(0) in _PRIMITIVE_TYPES or base_match.group(0) in _OPAQUE_CONTAINER_TYPES:
        return {}
    schema_result = await _resolve_type_schema_via_symbol_index(ctx, base_match.group(0))
    return schema_result[0] if schema_result else {}


def _evidence_item_schema(schema: dict) -> dict | None:
    """The per-record schema for an evidence-shaped kwarg schema, when directly array-shaped."""
    if isinstance(schema, dict) and schema.get("type") == "array" and isinstance(schema.get("items"), dict):
        return schema["items"]
    return None


async def _harvest_virtual_inputs_static(
    ctx: _EnrichmentContext,
    cls: Any,
    entry: Any,
    own_tree: Any,
    own_file: Path,
    asts: dict[Path, Any],
    entry_kwargs: list[KwargSpec] | None = None,
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

    # The entry's own signature evidence param (if any) grounds the WEAK verb/keyword-only
    # classification below and, when the dep's own return schema didn't resolve, supplies the field
    # shape and the real name -- so a virtual input is never named off a bare literal.
    evidence_kwarg_name: str | None = None
    evidence_kwarg_schema: dict = {}
    for kw in (entry_kwargs or []):
        kw_schema = await _resolve_kwarg_evidence_schema(ctx, kw.annotation or "")
        if _schema_is_evidence_shaped(kw_schema):
            evidence_kwarg_name = kw.name
            evidence_kwarg_schema = kw_schema
            break

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
        elif not schema and (is_retrieval_verb or keyword_role == "retrieval") and evidence_kwarg_name is not None:
            # A verb/keyword match alone has no structural backing -- trusted only when a real
            # signature-bound evidence param corroborates it, never fabricated off the name alone.
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
        elif evidence_kwarg_name is not None:
            item_schema = _evidence_item_schema(evidence_kwarg_schema)
            if item_schema and "properties" in item_schema:
                for fname, fschema in item_schema.get("properties", {}).items():
                    fields.append(VirtualFieldSpec(
                        name=fname,
                        schema=fschema,
                        provenance="annotation",
                        confidence=0.85,
                    ))

        contracts.append(VirtualInputContract(
            name=evidence_kwarg_name or EVIDENCE_CASE_KEY,
            dep_param=binding.param,
            dep_attr=binding.attr,
            dep_annotation=binding.annotation or "",
            dep_role=dep_role,
            methods_called=methods,
            call_sites=citations,
            fields=fields,
        ))

    return contracts


def _base_type_name(type_hint: str) -> str | None:
    import re as _re
    match = _re.search(r"[A-Za-z_][A-Za-z0-9_]*", type_hint)
    return match.group(0) if match else None


async def _resolve_input_kwarg_schemas(
    ctx: _EnrichmentContext,
    input_contract: list[ContractArg],
    upstream_output_schema: dict | None = None,
) -> tuple[dict[str, dict], str | None, list[str]]:
    """Resolve schemas for input kwargs that have resolvable type annotations. Returns
    (schemas, upstream_resolved_kwarg, needs_human_notes): a bare dict/list kwarg carries no
    field-level shape of its own, so it resolves via the sole upstream agent's own output schema
    when unambiguous (upstream_resolved_kwarg names which one), else it is flagged rather than
    silently dropped."""
    opaque_kwargs = [
        arg.kwarg for arg in input_contract
        if _base_type_name(arg.type_hint or "") in _OPAQUE_CONTAINER_TYPES
    ]

    input_schemas: dict[str, dict] = {}
    notes: list[str] = []
    upstream_resolved_kwarg: str | None = None
    for arg in input_contract:
        type_name = _base_type_name(arg.type_hint or "")
        if type_name is None or type_name in _PRIMITIVE_TYPES:
            continue

        if type_name in _OPAQUE_CONTAINER_TYPES:
            if upstream_output_schema is not None and len(opaque_kwargs) == 1:
                input_schemas[arg.kwarg] = upstream_output_schema
                upstream_resolved_kwarg = arg.kwarg
            else:
                notes.append(f"{arg.kwarg}: bare '{type_name}' type hint, unresolvable — needs_human")
            continue

        schema_result = await _resolve_type_schema_via_symbol_index(ctx, type_name)
        if schema_result:
            schema, _, _ = schema_result
            input_schemas[arg.kwarg] = schema

    return input_schemas, upstream_resolved_kwarg, notes


async def _finalize_input_contract(
    ctx: _EnrichmentContext,
    llm_knowledge: AgentKnowledge,
    upstream_output_schema: dict | None,
    own_contract: EvaluationContract | None = None,
) -> None:
    """Resolves input_schemas for llm_knowledge.input_contract in-place and tags each kwarg's
    source_kind by provenance (config-static / upstream-agent-output / runtime-state). When
    own_contract is given, mirrors both the resolved schema and the provenance tag onto the matching
    KwargSpec in own_contract.invocation.kwargs (by name) so the SAME resolution reaches the
    persisted EvaluationContract, not only the legacy AgentKnowledge fields."""
    schemas, upstream_resolved_kwarg, notes = await _resolve_input_kwarg_schemas(
        ctx, llm_knowledge.input_contract, upstream_output_schema
    )
    llm_knowledge.input_schemas = schemas
    llm_knowledge.needs_human.extend(notes)
    contract_kwargs_by_name = (
        {k.name: k for k in own_contract.invocation.kwargs}
        if own_contract is not None and own_contract.invocation is not None
        else {}
    )
    for arg in llm_knowledge.input_contract:
        if arg.kwarg in _CONFIG_PARAMS:
            arg.source_kind = _PROVENANCE_CONFIG_STATIC
        elif arg.kwarg == upstream_resolved_kwarg:
            arg.source_kind = _PROVENANCE_UPSTREAM_OUTPUT
        else:
            arg.source_kind = _PROVENANCE_RUNTIME_STATE
        contract_kwarg = contract_kwargs_by_name.get(arg.kwarg)
        if contract_kwarg is not None:
            contract_kwarg.source_kind = arg.source_kind
            # Don't clobber a schema the harvester already resolved (e.g. a state-node's read-subset);
            # only fill when harvest left it unresolved.
            if arg.kwarg in schemas and contract_kwarg.resolved_schema is None:
                contract_kwarg.resolved_schema = schemas[arg.kwarg]

    if own_contract is not None and own_contract.invocation is not None:
        for kwarg in own_contract.invocation.kwargs:
            if kwarg.resolved_schema is None and kwarg.annotation:
                type_name = _base_type_name(kwarg.annotation)
                if type_name is not None and type_name not in _PRIMITIVE_TYPES and type_name not in _OPAQUE_CONTAINER_TYPES:
                    schema_result = await _resolve_type_schema_via_symbol_index(ctx, type_name)
                    if schema_result:
                        schema, _, _ = schema_result
                        kwarg.resolved_schema = schema

    if own_contract is not None and notes:
        own_contract.needs_human.extend(notes)


def _entry_scoped_prompt_sites(cls: Any, entry: Any, tree: Any, file_sites: list[Any]) -> list[Any]:
    """Filters one file's prompt sites down to the ones the entry method's own closure actually
    references -- the same self.<helper>() closure _entry_schema_scope uses for schema constants --
    so a sibling entry method's prompt constant on a multi-entry class (e.g. LangGraph nodes sharing
    one class) never cross-attributes. A module_assignment site is scoped by whether its assigned
    NAME is referenced in the closure (its definition line is elsewhere); any other kind is scoped by
    whether its own recorded line falls inside the closure (it records a use site, not a definition)."""
    if entry is None or not file_sites:
        return []
    import ast as _ast

    scope_nodes = _entry_schema_scope(cls, entry) if cls is not None else [entry]
    referenced_names: set[str] = set()
    for scope_node in scope_nodes:
        for node in _ast.walk(scope_node):
            if isinstance(node, _ast.Name):
                referenced_names.add(node.id)
            elif isinstance(node, _ast.Attribute):
                referenced_names.add(node.attr)

    def _assigned_name_at(line: int) -> str | None:
        for node in _ast.walk(tree):
            if isinstance(node, (_ast.Assign, _ast.AugAssign)) and node.lineno == line:
                target = node.targets[0] if isinstance(node, _ast.Assign) else node.target
                if isinstance(target, _ast.Name):
                    return target.id
                if isinstance(target, _ast.Attribute):
                    return target.attr
        return None

    def _within_closure(line: int) -> bool:
        return any(
            scope_node.lineno <= line <= (getattr(scope_node, "end_lineno", None) or scope_node.lineno)
            for scope_node in scope_nodes
        )

    scoped = []
    for site in file_sites:
        if site.kind == "module_assignment":
            name = _assigned_name_at(site.line)
            if name is not None and name not in referenced_names:
                continue
        elif not _within_closure(site.line):
            continue
        scoped.append(site)
    return scoped


def _extract_prompt_json_schema(prompt_sites: list[PromptSiteRef]) -> tuple[set[str], str] | None:
    """First balanced-brace {...} JSON object embedded in a prompt site's snippet, preferring an
    untruncated module_assignment-kind snippet over the 120-char-capped sdk_call-kind (which can cut
    a JSON block mid-token). Returns (top-level key set, "file:line") or None when nothing parses."""
    ordered = sorted(prompt_sites, key=lambda s: 0 if s.kind == "module_assignment" else 1)
    for site in ordered:
        start = site.snippet.find("{")
        if start == -1:
            continue
        depth = 0
        end = None
        for i, ch in enumerate(site.snippet[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            continue
        try:
            parsed = json.loads(site.snippet[start:end])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and parsed:
            return set(parsed.keys()), f"{site.file}:{site.line}"
    return None


def _schema_top_level_keys(json_schema: dict) -> set[str]:
    """The field-name key set for a harvested json_schema, regardless of which of the two shapes
    harvest_component_contract produces: a JSON-Schema envelope ({"type":"object","properties":
    {...}}) from the dict-literal/class-resolver paths, or a raw example dict -- the fields ARE the
    top-level keys -- from the schema-constant-parsed-as-a-JSON-string path."""
    properties = json_schema.get("properties")
    if isinstance(properties, dict):
        return set(properties.keys())
    return set(json_schema.keys())


def _cross_validate_output_schema(
    output: OutputContract | None, candidate_sites: list[Any]
) -> tuple[OutputContract | None, str | None]:
    """Cross-checks an AST-derived output schema against a JSON block embedded in an entry-scoped
    prompt site (see _entry_scoped_prompt_sites -- candidate_sites must already be scoped to this
    entry, never the whole file). AST silence and a genuine key-set mismatch both count as
    disagreement; on disagreement the prompt-derived key-set wins (schema_source stays truthful,
    pointing at the prompt site) and a schema_discrepancy + needs_human note record why. No
    parseable prompt JSON, or an agreeing key-set, returns `output` untouched.
    Takes/returns a bare OutputContract (not an AgentKnowledge) so the SAME validation applies
    unchanged to both the legacy AgentKnowledge.output_contract and the persisted
    EvaluationContract.output -- one shared helper, two callers, never duplicated logic."""
    prompt_schema = _extract_prompt_json_schema(candidate_sites)
    if prompt_schema is None:
        return output, None
    prompt_keys, prompt_cite = prompt_schema

    ast_keys = (
        _schema_top_level_keys(output.json_schema)
        if output and output.json_schema else None
    )
    if ast_keys is not None and ast_keys == prompt_keys:
        return output, None

    is_authoritative = bool(output and (output.validated_in_target or (output.schema_source and (".py" in output.schema_source or "state mutation delta" in output.schema_source))))
    if is_authoritative:
        output.schema_discrepancy = (
            f"AST schema properties {sorted(ast_keys) if ast_keys is not None else '(none)'} "
            f"vs prompt-derived {sorted(prompt_keys)} at {prompt_cite}"
        )
        note = f"output schema discrepancy: AST vs prompt-derived key-set disagree ({prompt_cite})"
        return output, note

    if output is None:
        output = OutputContract()
    output.json_schema = {
        "type": "object",
        "properties": {k: {} for k in sorted(prompt_keys)},
        "required": sorted(prompt_keys),
    }
    output.schema_source = f"{prompt_cite} (prompt-derived)"
    output.cardinality = "object"  # a prompt-embedded JSON block is always parsed as a flat object
    output.schema_discrepancy = (
        f"AST schema properties {sorted(ast_keys) if ast_keys is not None else '(none)'} "
        f"vs prompt-derived {sorted(prompt_keys)} at {prompt_cite}"
    )
    note = f"output schema discrepancy: AST vs prompt-derived key-set disagree ({prompt_cite})"
    return output, note


def _validate_tier_b_schema(schema: Any) -> dict[str, Any] | None:
    """Validates an LLM-emitted JSON Schema dict (Tier B). Must be a dict with valid structure (type object/array or properties/items)."""
    if not isinstance(schema, dict) or not schema:
        return None
    stype = schema.get("type")
    properties = schema.get("properties")
    items = schema.get("items")
    if stype not in ("object", "array") and not isinstance(properties, dict) and not isinstance(items, dict):
        return None
    if isinstance(properties, dict):
        for k in properties:
            if not isinstance(k, str):
                return None
    return schema


def _apply_multi_tier_output_resolution(
    output: OutputContract | None,
    prompt_schema: dict[str, Any] | None,
    prompt_enums: dict[str, list[str]] | None,
    agent_id: str,
) -> tuple[OutputContract, list[str]]:
    """Multi-tier output schema resolution policy:
    1. Tier A (static type / AST constant): if present, prefer static type (schema_source = "static_type" / "static_constant").
       If Tier B is also present, cross-check for discrepancy; if key-set matches and Tier A is key-only ({}), enrich with Tier B types.
    2. Tier B (prompt-derived structured schema): if Tier A is missing but Tier B valid, use Tier B (schema_source = "prompt").
    3. Neither present: json_schema = None, schema_source = "none", explicit needs_human note.
    """
    notes: list[str] = []
    if output is None:
        output = OutputContract()

    valid_prompt_schema = _validate_tier_b_schema(prompt_schema)
    valid_prompt_enums = (
        {k: v for k, v in prompt_enums.items() if isinstance(k, str) and isinstance(v, list)}
        if isinstance(prompt_enums, dict) else {}
    )

    tier_a_schema = output.json_schema

    if tier_a_schema is not None:
        is_flat_literal = isinstance(tier_a_schema, dict) and "properties" not in tier_a_schema and "type" not in tier_a_schema
        if is_flat_literal and valid_prompt_schema is not None:
            ast_keys = _schema_top_level_keys(tier_a_schema)
            prompt_keys = _schema_top_level_keys(valid_prompt_schema)
            if not ast_keys or ast_keys == prompt_keys:
                output.json_schema = valid_prompt_schema
                output.schema_source = "prompt"
                output.cardinality = "array" if valid_prompt_schema.get("type") == "array" else "object"
                if valid_prompt_enums:
                    output.schema_enum_values.update(valid_prompt_enums)
                return output, notes

        output.schema_source = output.schema_source or "static_type"
        is_authoritative_typeddict = bool(output.validated_in_target or (output.schema_source and ".py" in output.schema_source))
        if valid_prompt_schema is not None and not is_authoritative_typeddict:
            ast_keys = _schema_top_level_keys(tier_a_schema)
            prompt_keys = _schema_top_level_keys(valid_prompt_schema)
            if ast_keys and prompt_keys and ast_keys != prompt_keys:
                output.schema_discrepancy = (
                    f"AST schema properties {sorted(ast_keys)} vs prompt-derived {sorted(prompt_keys)}"
                )
                notes.append(f"output schema discrepancy for agent {agent_id}: static type vs prompt description")
            elif ast_keys and prompt_keys and ast_keys == prompt_keys:
                a_props = tier_a_schema.get("properties")
                b_props = valid_prompt_schema.get("properties")
                if isinstance(a_props, dict) and isinstance(b_props, dict):
                    if all(v == {} for v in a_props.values()):
                        output.json_schema = dict(tier_a_schema)
                        output.json_schema["properties"] = b_props
        if valid_prompt_enums:
            for k, v in valid_prompt_enums.items():
                if k not in output.schema_enum_values:
                    output.schema_enum_values[k] = v
        return output, notes

    if valid_prompt_schema is not None:
        output.json_schema = valid_prompt_schema
        output.schema_source = "prompt"
        output.cardinality = "array" if valid_prompt_schema.get("type") == "array" else "object"
        if valid_prompt_enums:
            output.schema_enum_values.update(valid_prompt_enums)
        return output, notes

    output.json_schema = None
    output.schema_source = "none"
    notes.append(f"{agent_id}: output schema not statically resolvable and prompt did not describe a structured output")
    return output, notes


async def _enrich_single_agent(
    agent_id: str,
    evidence: dict[str, Any],
    ctx: _EnrichmentContext,
    depth_cap: dict[str, int],
    accepted_files: list[str],
    contracts: dict[str, EvaluationContract] | None = None,
) -> AgentKnowledge:
    """Enrich a single agent: check cache, verify coverage, run queries/LLM, persist."""

    components_list = evidence['component_by_agent'].get(agent_id, [])
    component_ids = sorted([c['id'] for c in components_list])
    edges = sorted([
        (e['src'], e['dst']) for e in evidence['edges_by_agent'].get(agent_id, [])
    ])

    # Include component structural attributes in hash so re-topology invalidates cache.
    component_attrs = sorted([
        f"{c['id']}:{c.get('motif', 'N')}:{c.get('is_tool', False)}:{c.get('constructor_fanout', 0)}:"
        f"{len(c.get('conditional_downstream', []))}:{len(c.get('call_downstream', []))}"
        for c in components_list
    ])

    hash_input = '|'.join([
        str(_STRUCTURAL_PRODUCER_VERSION),
        ':'.join(component_ids),
        str(len(accepted_files)),
        ':'.join(f"{s}→{d}" for s, d in edges),
        ':'.join(component_attrs),
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
    # Structural veto for the LLM's control_motif claim: a component is only a plausible loop/branch
    # CONTROLLER if it is a structural loop/branch region, a conditional-edge source, or a high-fan-in
    # sink (an aggregator whose output drives a driver-side gate). Structure filters; the LLM only proposes.
    fan_in_threshold = evidence.get('fan_in_sink_threshold')
    is_plausible_controller = any(
        c.get('motif') in ('loop', 'branch')
        or c.get('conditional_downstream')
        or (fan_in_threshold is not None and c.get('fan_in', 0) >= fan_in_threshold)
        for c in components
    )
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

    # This agent's own just-harvested EvaluationContract (same dict harvest_contracts() built once,
    # up in enrich_agents; the wrapper later attaches it to knowledge.evaluation_contract) --
    # mutated in place below so the output/input-kwarg resolution reaches the persisted contract,
    # not only the legacy AgentKnowledge fields.
    own_contract: EvaluationContract | None = contracts.get(agent_id) if contracts else None

    # The sole upstream agent's own resolved output schema -- used to resolve a bare dict/list kwarg
    # (e.g. worker_output: dict) that carries another agent's output but has no shape of its own.
    # More than one upstream (or none resolved) is ambiguous -- left to needs_human, never guessed.
    upstream_output_schema: dict | None = None
    if contracts:
        upstream_schemas = [
            contracts[u].output.json_schema
            for u in agent.upstream_agents
            if u in contracts and contracts[u].output and contracts[u].output.json_schema
        ]
        if len(upstream_schemas) == 1:
            upstream_output_schema = upstream_schemas[0]

    # Per-SYSTEM context leads, per-agent detail follows: every agent of one system then shares a
    # longer identical prefix, which is what the provider's prefix cache can reuse.
    context_lines: list[str] = []
    # CS-320: Inject system_type context if available
    if ctx.system_type:
        if ctx.system_type == "orchestrator":
            context_lines.append("System type: 'orchestrator' — router node present → treat conditional-edge sources as orchestrator")
        elif ctx.system_type == "pipeline":
            context_lines.append("System type: 'pipeline' — no central orchestrator expected, do not force one")
        elif ctx.system_type in ("single-flow", "tool-loop", "plan-execute", "reflection"):
            context_lines.append(f"System type: '{ctx.system_type}' — single agent, no multi-component orchestration")
        else:
            context_lines.append(f"System type: '{ctx.system_type}'")
        context_lines.append("")
    context_lines += [f"Agent: {agent.label}", ""]
    if components:
        context_lines.append(f"Components ({len(components)}):")
        for c in components:
            entry = f" — entry: {c['entry_point']}" if c.get('entry_point') else ""
            facts = structural_facts(c.get('fan_in', 0), c.get('fan_out', 0))
            admissible = sorted(admissible_roles(c.get('is_tool'), c.get('constructor_fanout')))
            structural = [facts]
            if c.get('conditional_downstream') or c.get('call_downstream') or c.get('motif'):
                edges_parts = []
                if c.get('conditional_downstream'):
                    edges_parts.append(f"conditional→{','.join(c['conditional_downstream'])}")
                if c.get('call_downstream'):
                    edges_parts.append(f"call→{','.join(c['call_downstream'])}")
                if c.get('motif'):
                    edges_parts.append(f"motif:{c['motif']}")
                structural.append('; '.join(edges_parts))
            structural_str = '; '.join(structural)
            context_lines.append(
                f"  - {c['id']} @ {c.get('file', '?')}{entry} — {structural_str}; admissible roles: {admissible}"
            )
            if c.get('motif') == 'loop':
                context_lines.append("    ↳ Loop member: harvest its termination/exit condition and the feedback signal it consumes each iteration")
            elif c.get('motif') == 'branch':
                context_lines.append("    ↳ Branch source: harvest the policy by which it selects among downstream targets")
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
        # Boost control-flow evidence when any component looks control-relevant: a structural loop/branch
        # region, a conditional-edge source, or a high-fan-in sink (an aggregator whose output typically
        # drives a driver-side retry/gate loop that is invisible to the wiring graph).
        boost_control_flow = any(
            c.get('motif') in ('loop', 'branch')
            or c.get('conditional_downstream')
            or (fan_in_threshold is not None and c.get('fan_in', 0) >= fan_in_threshold)
            for c in components
        )
        own_lines, related_lines = await _fetch_direct_evidence(ctx, component_files, known_edge_files, boost_control_flow=boost_control_flow)
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

                entry_kwargs = _harvest_kwargs(entry)[0] if entry is not None else []
                virtual_contracts = await _harvest_virtual_inputs_static(
                    ctx, cls, entry, tree, comp_file, asts, entry_kwargs
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
  "output_schema_from_prompt": <a valid JSON Schema dict matching the output shape described by the prompt, e.g. {{"type": "object", "properties": {{"field1": {{"type": "string"}}}}}}; null if the prompt is silent or describes no structured format>,
  "output_enum_values_from_prompt": <a dict mapping field path to list of allowed enum string values if prompt specifies allowed values, e.g. {{"confidence": ["high", "medium", "low"]}}; empty dict {{}} if none>,
  "method_steps": ["<one step, in order, of the procedure the prompt tells the agent to follow, e.g. 'retrieve evidence' then 'reason' then 'emit JSON'>"],
  "constraints": ["<one HARD RULE the prompt imposes, e.g. 'return only JSON', 'must cite evidence_files', 'never invent file paths'>"],
  "special_traits": ["<one notable/distinctive behavior the prompt calls out>"],
  "control_motif": null,
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
    stashed_input_examples: dict[str, Any] = {}
    async def _run_llm_rounds(
        prompt: str,
        depth: dict[str, int],
    ) -> AgentKnowledge | None:
        """Run LLM round-1, optionally round-2 based on need_more."""
        nonlocal query_count, llm_calls, stashed_input_examples

        llm_calls += 1
        raw, raw_text = await complete_json_with_raw(
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
        control_motif = raw.get('control_motif')

        round_knowledge = AgentKnowledge.model_validate(_sanitize_llm_knowledge_dict(raw))
        if control_motif and isinstance(control_motif, str) and control_motif in ('loop', 'branch') and is_plausible_controller:
            round_knowledge.control_motif = control_motif

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

            # Round 2 replays round 1 verbatim as history: without it the model was told to "keep your
            # previous answer" while never being shown that answer OR the original evidence. Replaying
            # it also means the whole round-1 prefix is a provider cache hit instead of new input.
            round2_history = [
                LLMMessage(role="user", content=prompt),
                LLMMessage(role="assistant", content=raw_text),
            ]
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
                history=round2_history,
            )
            if raw2 is not None:
                round2_knowledge = AgentKnowledge.model_validate(_sanitize_llm_knowledge_dict(raw2))
                control_motif_r2 = raw2.get('control_motif')
                if control_motif_r2 and isinstance(control_motif_r2, str) and control_motif_r2 in ('loop', 'branch') and is_plausible_controller:
                    round2_knowledge.control_motif = control_motif_r2

                # Stash examples and fields for later merge (after static-contract attach).
                input_contract_examples_r2 = raw2.get('input_contract_examples', {})
                virtual_input_fields_r2 = raw2.get('virtual_input_fields', [])

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
                    stashed_input_examples = input_contract_examples_r2
                else:
                    logger.debug(
                        f"Agent {agent_id}: round-2 less rich than round-1 ({r2} < {r1} claims), keeping round-1"
                    )
                    stashed_input_examples = input_contract_examples
            else:
                stashed_input_examples = input_contract_examples

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
            # Tolerate a decorated id (`<id> @ <file>`, `<id> (file: ...)`, `id="<id>"`) before
            # discarding: a formatting slip silently cost the component its role, surfacing later
            # as a stray 'unknown' that looks like the model failed to classify it.
            bare = _undecorate_component_id(verdict.id)
            c = structural_by_id.get(bare)
            if c is None:
                bare = _resolve_entry_point_id(bare, structural_by_id)
                c = structural_by_id.get(bare)
            if c is not None:
                logger.info(
                    f"Agent {agent_id}: recovered decorated component id {verdict.id!r} -> {bare!r}"
                )
                verdict = verdict.model_copy(update={"id": bare})
        if c is None:
            # LLM named a component id that isn't this agent's — discard, never invent.
            logger.warning(
                f"Agent {agent_id}: discarding role for unknown component id {verdict.id!r} "
                f"(known ids: {sorted(structural_by_id)})"
            )
            continue
        # CS-320: forced_role applies before admissible check
        forced = forced_role(bool(c.get('conditional_downstream')))
        if forced:
            role = forced
        else:
            admissible = admissible_roles(c.get('is_tool'), c.get('constructor_fanout'))
            role = verdict.role
            if role not in admissible or verdict.confidence < ROLE_CONFIDENCE_THRESHOLD:
                role = 'unknown'
        gated_roles.append(ComponentRoleVerdict(
            id=verdict.id, role=role, confidence=verdict.confidence, reasoning=verdict.reasoning
        ))
    # CS-320: Post-loop pass for conditional-source components the LLM omitted
    llm_emitted_ids = {v.id for v in llm_knowledge.component_roles}
    for cid, c in structural_by_id.items():
        if cid not in llm_emitted_ids and c.get('conditional_downstream'):
            gated_roles.append(ComponentRoleVerdict(
                id=cid, role='orchestrator', confidence=1.0,
                reasoning='structural: conditional-edge source'
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
                    head_id = agent_id if any(c.get('id') == agent_id for c in components_list) else (components_list[0].get('id') if components_list else agent_id)
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
                            if comp_for_harvest is None or comp_for_harvest.get('id') != head_id:
                                continue  # only the agent's head/entry component may set location/input_contract/output_contract

                            if comp_for_harvest.get('entry_point'):
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
                                                    and sym.get('name') in _SEARCH_INDEX_ENTRY_METHOD_NAMES):
                                                if method_row is None or _SEARCH_INDEX_ENTRY_METHOD_NAMES.index(sym.get('name')) < _SEARCH_INDEX_ENTRY_METHOD_NAMES.index(method_row.get('name')):
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
                                                    _, entry_cls, entry_fn = _resolve_entry(temp_component, asts)
                                                    scoped_sites = _entry_scoped_prompt_sites(
                                                        entry_cls, entry_fn, asts[comp_path],
                                                        prompt_sites.get(comp_file, []),
                                                    )
                                                    ak_output, ak_note = _cross_validate_output_schema(
                                                        llm_knowledge.output_contract, scoped_sites,
                                                    )
                                                    llm_knowledge.output_contract = ak_output
                                                    if ak_note:
                                                        llm_knowledge.needs_human.append(ak_note)
                                                    # Same recovery, on the persisted contract's own output.
                                                    if own_contract is not None:
                                                        ec_output, ec_note = _cross_validate_output_schema(
                                                            own_contract.output, scoped_sites,
                                                        )
                                                        own_contract.output = ec_output
                                                        if ec_note:
                                                            own_contract.needs_human.append(ec_note)
                                    except Exception as e:
                                        logger.debug(f"Failed to harvest contracts for {agent_id}: {e}")

                                    # Build input_contract from AST-harvested kwargs (real type hints) if available.
                                    # For bound_method, use the method's own signature, not the resolved callee'params.
                                    if entry_kind == "bound_method" and method_row is not None:
                                        sig = method_row.get('signature', '')
                                        kwargs_with_hints = _parse_signature_kwargs(sig)
                                        if kwargs_with_hints:
                                            llm_knowledge.input_contract = [
                                                ContractArg(kwarg=name, type_hint=hint, example='')
                                                for name, hint in kwargs_with_hints
                                            ]
                                            await _finalize_input_contract(
                                                ctx, llm_knowledge, upstream_output_schema, own_contract,
                                            )
                                    elif invocation_contract and invocation_contract.kwargs:
                                        llm_knowledge.input_contract = [
                                            ContractArg(kwarg=k.name, type_hint=k.annotation or '', example='')
                                            for k in invocation_contract.kwargs
                                        ]
                                        await _finalize_input_contract(
                                            ctx, llm_knowledge, upstream_output_schema, own_contract,
                                        )
                                    elif method_row is not None:
                                        # Fallback to truncated signature parsing (backward compat; class/bound-method only).
                                        sig = method_row.get('signature', '')
                                        kwargs_with_hints = _parse_signature_kwargs(sig)
                                        if kwargs_with_hints:
                                            llm_knowledge.input_contract = [
                                                ContractArg(kwarg=name, type_hint=hint, example='')
                                                for name, hint in kwargs_with_hints
                                            ]
                                            await _finalize_input_contract(
                                                ctx, llm_knowledge, upstream_output_schema, own_contract,
                                            )

                                    # Apply LLM examples after static-contract attach.
                                    if stashed_input_examples and isinstance(stashed_input_examples, dict):
                                        for arg in llm_knowledge.input_contract:
                                            if arg.kwarg in stashed_input_examples:
                                                arg.example = stashed_input_examples[arg.kwarg]

                                    # Static fallback: if arg.example is still empty, check own_contract invocation kwargs for default_repr
                                    if own_contract and own_contract.invocation and own_contract.invocation.kwargs:
                                        default_repr_by_kwarg = {
                                            k.name: k.default_repr for k in own_contract.invocation.kwargs
                                            if k.name and k.default_repr
                                        }
                                        for arg in llm_knowledge.input_contract:
                                            if not arg.example and arg.kwarg in default_repr_by_kwarg:
                                                arg.example = default_repr_by_kwarg[arg.kwarg]

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

    # Multi-tier output schema resolution policy
    p_schema = getattr(llm_knowledge, "output_schema_from_prompt", None)
    p_enums = getattr(llm_knowledge, "output_enum_values_from_prompt", None)

    resolved_out, out_notes = _apply_multi_tier_output_resolution(
        llm_knowledge.output_contract, p_schema, p_enums, agent_id
    )
    llm_knowledge.output_contract = resolved_out
    llm_knowledge.needs_human.extend(out_notes)

    if own_contract is not None:
        resolved_ec_out, ec_notes = _apply_multi_tier_output_resolution(
            own_contract.output, p_schema, p_enums, agent_id
        )
        own_contract.output = resolved_ec_out
        own_contract.needs_human.extend(ec_notes)

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
