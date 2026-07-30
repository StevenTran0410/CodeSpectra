"""CS-317 acceptance harness — REAL Stage-1->2 pipeline, in-process, NO Electron GUI, NO HTTP.

Reproduces the FULL discovery->consolidation->expansion->map-build->topology path on the actual
CodeSpectra snapshot, so the multi-system PARTITION fix is proven on the real app path (not isolation):
the qa/analysis community must split into THREE separate single-framework candidates — Haystack (Code
Analysis), LangGraph (Deep Research), plain-python (Ask mode) — each expanding + building into its OWN
map. NO candidate may carry a blended 'haystack+langgraph' framework.

Design (mirrors run_stage3_smoke.py's offline philosophy):
  * client -> InProcessClient: a CodeSpectraClient-shaped adapter that delegates to the backend's
             REAL domain services (RetrievalService/StructuralGraphService/RepoMapService/
             ManifestService/SyncEngineService) in-process — the exact services the /api/external
             routes call — returning model_dump() dicts, the same shape the HTTP proxy would yield.
             No FastAPI server, no bearer token, no Electron.
  * LLM    -> the FSOFT provider row already in codespectra.db (GPT-5.4-mini), via the backend
             ProviderConfigService (FsoftLLMClient). The API key stays in the DB; never printed.
  * repo   -> snapshot.local_path (on disk) for the map-build file reads.

Run from repo root inside the agent_eval_harness uv env:
    uv run python agent_eval_harness/scripts/run_stage12_mixed_acceptance.py
    uv run python agent_eval_harness/scripts/run_stage12_mixed_acceptance.py --snapshot <id> --effort medium

Exit code 0 iff ALL acceptance assertions hold (three pure maps: Haystack >=12; LangGraph 8
bound_method with StateGraph edges; plain-python QAAgent — and no blended framework anywhere).
NOTE: this file lives under scripts/ (NOT agent_eval_harness/agent_eval_harness/**), so referencing
concrete target symbols (deep_research/DeepResearchAgent/QAAgent) here does NOT violate nguyen-tac-so-0.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sqlite3
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]  # scripts -> agent_eval_harness -> <repo>
_APPDATA = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
_DATA_DIR = os.path.join(_APPDATA, "codespectra")
os.environ.setdefault("AEH_DATA_DIR", _DATA_DIR)
os.environ.setdefault("CODESPECTRA_DATA_DIR", _DATA_DIR)
_CSDB = os.path.join(_DATA_DIR, "codespectra.db")

for p in (str(_REPO / "backend"), str(_REPO / "agent_eval_harness")):
    if p not in sys.path:
        sys.path.insert(0, p)

DEFAULT_SNAPSHOT = "cb66719d-7cfc-4254-bc68-4c2fa4299ff6"
FSOFT_PROVIDER_ID = "a4b16186-c964-4338-b96c-53af776b5fdc"
FSOFT_MODEL_ID = "GPT-5.4-mini"
DEFAULT_EFFORT = "medium"

# Target markers used ONLY to classify/report the real map (allowed in scripts/, not production).
_LANGGRAPH_FILE_MARK = "deep_research.py"
_LANGGRAPH_OWNER = "DeepResearchAgent"
_PLAIN_MARK = "QAAgent"
_HAYSTACK_DIR_MARK = "/analysis/"


def _resolve_fsoft_provider(explicit_id: str | None) -> tuple[str, str]:
    con = sqlite3.connect(_CSDB)
    con.row_factory = sqlite3.Row
    try:
        rows = list(con.execute("SELECT id, model_id, display_name FROM provider_configs"))
    finally:
        con.close()
    by_id = {r["id"]: r for r in rows}
    if explicit_id and explicit_id in by_id:
        r = by_id[explicit_id]
        return r["id"], r["model_id"] or FSOFT_MODEL_ID
    for r in rows:
        if "fsoft" in (r["display_name"] or "").lower():
            return r["id"], r["model_id"] or FSOFT_MODEL_ID
    if FSOFT_PROVIDER_ID in by_id:
        r = by_id[FSOFT_PROVIDER_ID]
        return r["id"], r["model_id"] or FSOFT_MODEL_ID
    return FSOFT_PROVIDER_ID, FSOFT_MODEL_ID


class FsoftLLMClient:
    """Satisfies the AEH LLMClient Protocol via the backend ProviderConfigService (loads the FSOFT
    key from codespectra.db internally; never touches the raw key)."""

    def __init__(self, provider_id: str, model_id: str, default_effort: str = DEFAULT_EFFORT) -> None:
        from domain.model_connector.service import ProviderConfigService
        self._svc = ProviderConfigService()
        self._pid, self._mid, self._effort = provider_id, model_id, default_effort

    async def complete(self, messages, *, max_tokens=1024, temperature=0.2, json_mode=False, reasoning_effort=None):
        from agent_eval_harness.llm.client import LLMResponse
        from domain.model_connector.types import ChatMessage, ChatRequest
        effort = reasoning_effort if reasoning_effort is not None else self._effort
        resp = await self._svc.chat(ChatRequest(
            provider_id=self._pid, model_id=self._mid,
            messages=[ChatMessage(role=m.role, content=m.content) for m in messages],
            max_completion_tokens=max_tokens, temperature=temperature,
            reasoning_effort=effort, json_mode=json_mode, stream=False,
        ))
        return LLMResponse(
            content=resp.content, model=resp.model_id,
            prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens,
            token_source="measured" if resp.prompt_tokens is not None else "estimated",
        )

    async def aclose(self) -> None:
        pass


class InProcessClient:
    """CodeSpectraClient-shaped adapter over the backend's real domain services (in-process)."""

    def __init__(self) -> None:
        from domain.retrieval.service import RetrievalService
        from domain.structural_graph.service import StructuralGraphService
        from domain.repo_map.service import RepoMapService
        from domain.manifest.service import ManifestService
        from domain.sync_engine.service import SyncEngineService
        self._retrieval = RetrievalService()
        self._graph = StructuralGraphService()
        self._repo_map = RepoMapService()
        self._manifest = ManifestService()
        self._sync = SyncEngineService()

    async def search_retrieval(self, snapshot_id, query, section="qa", symbol_chunks_only=False) -> dict:
        from domain.retrieval.types import RrfFusionRequest
        req = RrfFusionRequest(snapshot_id=snapshot_id, query=query, section=section,
                               symbol_chunks_only=symbol_chunks_only)
        return (await self._retrieval.retrieve_rrf_fusion(req)).model_dump(mode="json")

    async def get_file_chunks(self, snapshot_id, rel_path, symbol_chunks_only=False) -> dict:
        return (await self._retrieval.chunks_for_file(snapshot_id, rel_path, symbol_chunks_only)).model_dump(mode="json")

    async def get_communities(self, snapshot_id) -> dict:
        return (await self._graph.list_communities(snapshot_id)).model_dump(mode="json")

    async def get_symbol_edges(self, snapshot_id, file_path) -> dict:
        return (await self._graph.symbol_edges_for_file(snapshot_id, file_path)).model_dump(mode="json")

    async def search_repo_map(self, snapshot_id, q, limit=120) -> dict:
        return (await self._repo_map.search(snapshot_id, q, limit)).model_dump(mode="json")

    async def read_file(self, snapshot_id, rel_path, max_bytes=200_000) -> dict:
        return (await self._manifest.read_file(snapshot_id, rel_path, max_bytes)).model_dump(mode="json")

    async def get_snapshot(self, snapshot_id) -> dict:
        return (await self._sync.get_snapshot(snapshot_id)).model_dump(mode="json")

    async def aclose(self) -> None:
        pass


def _classify_component(comp) -> str:
    f = (comp.file or "").replace("\\", "/")
    ep = comp.entry_point or ""
    if _LANGGRAPH_FILE_MARK in f or _LANGGRAPH_OWNER in ep:
        return "langgraph"
    if _PLAIN_MARK in ep or f.endswith("/qa/agent.py"):
        return "plain_python"
    if _HAYSTACK_DIR_MARK in f:
        return "haystack"
    return "other"


def _files_of(c: dict) -> set[str]:
    return (set(c.get("cluster_files", [])) | set(c.get("matched_files", []) or [])
            | set(c.get("hub_paths", [])))


def _wb(c: dict) -> dict:
    return c.get("wiring_block") or {}


def _wb_framework(c: dict) -> str | None:
    return _wb(c).get("framework")


def _wb_source_files(c: dict) -> set[str]:
    return {(n.get("source_hint_file") or "").replace("\\", "/") for n in _wb(c).get("nodes", [])}


# CS-318: force exactly ONE real qa step past the import layer so it falls through to the LIVE RRF
# path (search_retrieval + AST-confirm against the real symbol index) and resolves to its true
# definer. Retrieval-only — costs NO chat-provider tokens, so it respects the FSOFT-429 / Gemini caps.
# A concrete symbol name is allowed here because this file lives under scripts/, not the inner package.
_FORCE_RRF_STEP = "plan_queries"


async def _expand_and_build(cand: dict, client, llm, local_path, args, *, resolver_skip=frozenset()):
    """Run the REAL Stage-2 expansion + map build + topology for ONE candidate, returning its own
    single-framework SystemMap and the accepted file list."""
    from agent_eval_harness.discovery.expansion import expand_candidate
    from agent_eval_harness.discovery.wiring import WiringBlock
    from agent_eval_harness.mapping.builder.pipeline import SystemMapBuilder

    res = await expand_candidate(args.snapshot, cand, client, llm,
                                 node_budget=args.node_budget, hop_cap=args.hop_cap)
    accepted = res.get("accepted", [])
    accepted_files = [(p["file"] if isinstance(p, dict) else p) for p in accepted]
    abs_files = [local_path / f for f in accepted_files]
    wb = cand.get("wiring_block")
    # Declared framework: the wiring_block's for a wired system, else the candidate's own framework
    # (a plain-python candidate has no wiring_block) — authoritative for the map's single-framework label.
    declared_framework = (wb or {}).get("framework") or (cand.get("frameworks") or [None])[0]
    builder = SystemMapBuilder(llm, framework=declared_framework)
    system_map, _summary = await builder.build_from_files(
        abs_files, package_root=local_path, target_system_id=cand["name"],
        wiring_block=WiringBlock.from_dict(wb) if wb else None,
        scope_framework=cand.get("map_scope_framework"),
        exclude_component_classes=set(cand.get("excluded_component_classes") or []),
        retrieval_client=client, snapshot_id=args.snapshot,
        resolver_skip_import_for=resolver_skip,
    )
    return system_map, accepted_files


async def _probe_resolver_layers(cand: dict, client, local_path, args, *, resolver_skip=frozenset()):
    """Directly probe the CS-318 resolver on the plain candidate's OWN files (no expansion needed):
    returns {callee: StepResolution} so the acceptance can assert WHICH layer placed each step and
    that RRF landed the correct definer — proving the free layers did the work, not RRF masking a
    regression."""
    from agent_eval_harness.discovery.wiring import WiringBlock, resolve_unscanned_steps
    wb = cand.get("wiring_block")
    if not wb:
        return {}
    seed = [local_path / f for f in (cand.get("hub_paths") or cand.get("matched_files") or [])]
    resolutions = await resolve_unscanned_steps(
        WiringBlock.from_dict(wb), seed, local_path, client, args.snapshot,
        skip_import_layer_for=resolver_skip,
    )
    return {r.callee: r for r in resolutions}


async def run(args) -> int:
    from agent_eval_harness.store.database import init_db as aeh_init_db, close_db as aeh_close_db
    from infrastructure.db.database import init_db as backend_init_db, close_db as backend_close_db

    await backend_init_db()
    await aeh_init_db()
    llm = None
    failures: list[str] = []
    try:
        provider_id, model_id = _resolve_fsoft_provider(args.provider_id)
        llm = FsoftLLMClient(provider_id, model_id, default_effort=args.effort)
        from agent_eval_harness.llm.client import LLMMessage
        smoke = await llm.complete([LLMMessage(role="user", content="Reply with exactly: OK")], max_tokens=512)
        print(f"[llm smoke] {model_id} (effort={args.effort}) -> {smoke.content!r}")

        client = InProcessClient()
        snap = await client.get_snapshot(args.snapshot)
        local_path = Path(snap["local_path"])
        print(f"[snapshot] {args.snapshot}  local_path={local_path}")
        if not local_path.exists():
            print(f"[FATAL] snapshot local_path missing on disk: {local_path}")
            return 2

        # ---- REAL Stage 1: discovery + consolidation ----
        from agent_eval_harness.discovery.engine import discover_agentic_systems
        print("[stage 1] discover_agentic_systems (fingerprint -> cluster -> synth -> PARTITION into "
              "per-system candidates -> consolidate)...")
        candidates = await discover_agentic_systems(args.snapshot, "CodeSpectra", client, llm)
        print(f"[stage 1] {len(candidates)} candidate(s)")
        for c in candidates:
            print(f"   - name={c.get('name')!r} system_id={c.get('system_id')} "
                  f"community_id={c.get('community_id')} frameworks={c.get('frameworks')} "
                  f"wb_framework={_wb_framework(c)!r} n_files={len(_files_of(c))}")

        # A split candidate must NEVER carry a blended framework — the whole point of the revision.
        blended = [c for c in candidates if "+" in (_wb_framework(c) or "")]
        for c in blended:
            failures.append(f"BLENDED framework on candidate name={c.get('name')!r} "
                            f"wb_framework={_wb_framework(c)!r} — partition did not split it")

        # Identify the THREE distinct systems of the qa/analysis community by content.
        hay = next((c for c in candidates
                    if _wb_framework(c) == "haystack"
                    and any(_HAYSTACK_DIR_MARK in f for f in _files_of(c))), None)
        lg = next((c for c in candidates
                   if _wb_framework(c) == "langgraph"
                   and any(_LANGGRAPH_FILE_MARK in f for f in _wb_source_files(c))), None)
        plain = next((c for c in candidates
                      if (c.get("frameworks") or []) == ["plain_python"]
                      and any(f.replace("\\", "/").endswith("/qa/agent.py") for f in _files_of(c))), None)

        if hay is None:
            failures.append("MISSING candidate: pure Haystack (framework='haystack') for /analysis/")
        if lg is None:
            failures.append("MISSING candidate: pure LangGraph (framework='langgraph') owning deep_research")
        if plain is None:
            failures.append(f"MISSING candidate: plain_python ({_PLAIN_MARK}, qa/agent.py, wiring_block None)")

        if hay is None or lg is None or plain is None:
            print("\n=== RESULT ===")
            for f in failures:
                print(f"  FAIL: {f}")
            print("ACCEPTANCE: FAILED (did not partition into three clean systems)")
            return 3

        # LangGraph candidate must be PURE with real StateGraph edges + deep_research matched.
        lg_wb = _wb(lg)
        if lg_wb.get("framework") != "langgraph":
            failures.append(f"LangGraph candidate framework impure: {lg_wb.get('framework')!r}")
        if not lg_wb.get("edges"):
            failures.append("LangGraph candidate wiring_block has ZERO edges (StateGraph orphaned pre-build)")
        if not any(_LANGGRAPH_FILE_MARK in f for f in (lg.get("matched_files") or [])):
            failures.append("deep_research file absent from LangGraph candidate matched_files")
        # CS-318: the plain candidate is now a split wired system (plain_python wiring_block, steps
        # composed from imported helpers + inherited base methods) — that block is what the resolver
        # consumes. If it carries a block at all it must be pure plain_python.
        plain_wb_fw = (plain.get("wiring_block") or {}).get("framework")
        if plain_wb_fw and "plain_python" not in plain_wb_fw:
            failures.append(f"plain_python candidate wiring_block impure: {plain_wb_fw!r}")

        # ---- CS-318 resolver-layer probe: prove WHICH layer placed each step (free import/mro did the
        # real work; exactly one forced step exercises the LIVE RRF path). Retrieval-only, no chat tokens.
        plain_layers = await _probe_resolver_layers(
            plain, client, local_path, args, resolver_skip=frozenset({_FORCE_RRF_STEP})
        )
        print("\n[stage 2 resolver] plain-python step resolution layers:")
        for callee, r in sorted(plain_layers.items()):
            print(f"    {callee!r:22} layer={r.layer:10} file={r.defining_file or '(unresolved)'}")

        # ---- REAL Stage 2 + map build, PER candidate -> its OWN single-framework map ----
        print("\n[stage 2 + build] expanding + building each candidate into its own map...")
        maps: dict[str, object] = {}
        for label, cand in (("haystack", hay), ("langgraph", lg), ("plain_python", plain)):
            skip = frozenset({_FORCE_RRF_STEP}) if label == "plain_python" else frozenset()
            system_map, accepted_files = await _expand_and_build(
                cand, client, llm, local_path, args, resolver_skip=skip
            )
            maps[label] = system_map
            comps = system_map.components
            total_edges = sum(len(c.upstream) + len(c.downstream) for c in comps)
            print(f"  [{label}] name={cand['name']!r} accepted_files={len(accepted_files)} "
                  f"components={len(comps)} map.framework={system_map.framework!r} topology_edges={total_edges}")
            if "+" in (system_map.framework or ""):
                failures.append(f"{label} map has BLENDED framework {system_map.framework!r}")

        # ---- Per-map COMPONENT-SET scoping assertions (no sibling bleed) ----
        import re as _re

        def _print_components(label: str, comps) -> None:
            print(f"  [{label}] component set ({len(comps)}):")
            for c in comps:
                print(f"    id={c.id!r} entry_kind={c.entry_kind} framework={getattr(c, 'framework', None)!r} "
                      f"file={(c.file or '')}")

        hay_map = maps["haystack"]
        lg_map = maps["langgraph"]
        plain_map = maps["plain_python"]
        _print_components("haystack", hay_map.components)
        _print_components("langgraph", lg_map.components)
        _print_components("plain_python", plain_map.components)

        # A component looks like a standalone agent class if its id ends in 'agent' (BaseLLMAgent,
        # DeepResearchAgent, QAAgent, _SectionAgent -> id 'basellmagent'/'deepresearchagent'/...).
        _AGENT_CLASS = _re.compile(r"agent$", _re.IGNORECASE)

        # (a) Haystack: only its @component agents, >= 12.
        if len(hay_map.components) < 12:
            failures.append(f"Haystack map components {len(hay_map.components)} < 12")

        # (b) LangGraph (CS-319): the 8 StateGraph node methods MUST all be present, and on top of that
        # the intra-node "gray" call graph must be expanded — the nodes' call targets (imported helpers,
        # inherited base methods, same-module fns, own-class helpers, injected services) become real
        # components. The 4 co-located junk classes (BaseLLMAgent / _SectionAgent / QAAgent /
        # DeepResearchAgent) must NOT appear (scope_framework persistence). Symbol names below are the
        # acceptance ORACLE for THIS target, matched against real components — not spec.
        lg_ids = {(c.id or "").lower() for c in lg_map.components}
        lg_edges = sum(len(c.upstream) + len(c.downstream) for c in lg_map.components)
        # b.1 — the 8 StateGraph nodes survive.
        expected_nodes = {
            "_node_load_context", "_node_plan", "_node_trace_forward", "_node_trace_backward",
            "_node_impact", "_node_retrieve", "_node_analyze_step", "_node_synthesize",
        }
        missing_nodes = expected_nodes - lg_ids
        if missing_nodes:
            failures.append(f"LangGraph map missing StateGraph node(s): {sorted(missing_nodes)}")
        # b.2 — the gray call targets (>=1 per resolution kind) are present at their REAL defining files.
        expected_gray = {
            "trace_call_chain": "graph_queries.py", "get_impact_cone": "graph_queries.py",
            "_retrieve_evidence": "deep_research.py", "_accumulate_evidence": "deep_research.py",
            "_validate_plan_step": "deep_research.py", "_load_graph_ctx": "deep_research.py",
            "render_bundle_subset": "deep_research.py", "plan_queries": "_graph_plan.py",
            "retrieve_multi": "_graph_plan.py", "retrievalservice": "service.py",
            "_chat_json_typed": "base.py", "_call_stream": "base.py",
        }
        by_id = {(c.id or "").lower(): c for c in lg_map.components}
        for gid, want_file in expected_gray.items():
            comp = by_id.get(gid)
            if comp is None:
                failures.append(f"LangGraph gray target missing: {gid!r} (expected from {want_file})")
            elif not (comp.file or "").replace("\\", "/").endswith(want_file):
                failures.append(f"LangGraph gray target {gid!r} points at {comp.file!r}, expected {want_file}")
        # b.3 — shared hub: _retrieve_evidence is ONE component with >=2 inbound call edges.
        rev = by_id.get("_retrieve_evidence")
        if rev is not None and len(rev.upstream) < 2:
            failures.append(f"_retrieve_evidence not a shared hub: inbound={rev.upstream} (expected >=2)")
        # b.4 — conditional router marking on the two add_conditional_edges sources.
        for router in ("_node_plan", "_node_analyze_step"):
            comp = by_id.get(router)
            if comp is not None and not getattr(comp, "conditional_downstream", None):
                failures.append(f"{router} outgoing edges not marked conditional (router branch missing)")
        # b.5 — the 4 co-located junk classes are gone.
        junk = [c.id for c in lg_map.components
                if c.entry_kind == "class" and _AGENT_CLASS.search(c.id or "")]
        if junk:
            failures.append(f"LangGraph map bleeds co-located agent classes (scope not applied): {junk}")
        if lg_edges == 0:
            failures.append("LangGraph map ORPHANED — zero topology edges")

        # (c) plain_python: contains QAAgent, and NO DeepResearchAgent (sibling-wiring-claimed) bleed.
        def _is_qaagent(c) -> bool:
            return (_PLAIN_MARK in (c.entry_point or "")) or (c.file or "").replace("\\", "/").endswith("/qa/agent.py")
        has_qaagent = any(_is_qaagent(c) for c in plain_map.components)
        if not has_qaagent:
            failures.append(f"plain_python map missing {_PLAIN_MARK} component")
        if any(_LANGGRAPH_OWNER.lower() == (c.id or "").lower() for c in plain_map.components):
            failures.append(f"plain_python map bleeds sibling-owned {_LANGGRAPH_OWNER}")

        # (c.1) CS-318 core: >=5 components (up from 1), connected call flow, and every wiring step
        # resolved to a REAL component — none silently dropped, none a needs_human placeholder.
        plain_edges = sum(len(c.upstream) + len(c.downstream) for c in plain_map.components)
        needs_human = [c for c in plain_map.components if getattr(c, "needs_human", None)]
        if len(plain_map.components) < 5:
            failures.append(f"plain_python map has {len(plain_map.components)} components, expected >=5 (steps dropped)")
        if plain_edges == 0:
            failures.append("plain_python map has ZERO topology edges — call flow not connected")
        if needs_human:
            failures.append(f"plain_python steps unresolved by every layer (needs_human): {[c.id for c in needs_human]}")

        # (c.2) Resolver guard against RRF noise: the 4 out-of-file steps must resolve to their TRUE
        # definers (_graph_plan.py via import, base.py via inheritance) — no junk file admitted; and the
        # ONE forced step must have gone down the LIVE RRF path and still landed the correct definer.
        def _rel(r) -> str:
            return (r.defining_file or "").replace("\\", "/")
        forced = plain_layers.get(_FORCE_RRF_STEP)
        if forced is None:
            failures.append(f"forced RRF step {_FORCE_RRF_STEP!r} not present among resolver steps")
        else:
            if forced.layer != "rrf":
                failures.append(f"forced step {_FORCE_RRF_STEP!r} resolved via {forced.layer!r}, expected 'rrf' (live-RRF path not exercised)")
            if not _rel(forced).endswith("_graph_plan.py"):
                failures.append(f"RRF resolved {_FORCE_RRF_STEP!r} to {_rel(forced)!r}, expected the real _graph_plan.py definer")
        # Every resolved step's file must be a genuine definer (no RRF junk leaked as a component).
        for callee, r in plain_layers.items():
            if r.defining_file and not (_rel(r).endswith("_graph_plan.py") or _rel(r).endswith("base.py")):
                failures.append(f"resolver admitted a non-definer file for {callee!r}: {_rel(r)!r} (possible RRF noise)")
        # The free layers must have done the real work on the other steps (not RRF masking a regression).
        free_layers = {r.layer for c, r in plain_layers.items() if c != _FORCE_RRF_STEP and r.defining_file}
        if not free_layers or not (free_layers <= {"import", "mro"}):
            failures.append(f"non-forced steps did not all resolve via free import/mro layers: {free_layers}")

        # (d) QAAgent must appear in NONE of the other two maps.
        for other_label, other_map in (("haystack", hay_map), ("langgraph", lg_map)):
            if any(_is_qaagent(c) for c in other_map.components):
                failures.append(f"QAAgent bled into the {other_label} map")

        # (e) CS-318: LangGraph completes Stage 2 end-to-end — a non-empty agent flow on top of the map.
        try:
            from agent_eval_harness.mapping.agent_flow import build_source_by_component, separate_agent_flows
            lg_abs = [local_path / f for f in (lg.get("matched_files") or [])]
            lg_src = build_source_by_component(lg_abs, lg_map)
            lg_flow = await separate_agent_flows(lg_map, lg_src, llm, files=lg_abs)
            n_agents = len(lg_flow.agents)
            print(f"\n[stage 2 agent-flow] LangGraph agent flow: agents={n_agents} "
                  f"entry={len(lg_flow.entry_agent_ids)} unassigned={len(lg_flow.unassigned_component_ids)}")
            if n_agents == 0:
                failures.append("LangGraph agent flow is EMPTY — Stage 2 did not complete end-to-end")
        except Exception as e:
            failures.append(f"LangGraph agent-flow generation raised: {e}")

        print("\n=== ACCEPTANCE SUMMARY (THREE scoped single-framework systems) ===")
        print(f"  (1) Haystack     : name={hay['name']!r}  components={len(hay_map.components)}  "
              f"framework={hay_map.framework!r}")
        print(f"  (2) LangGraph    : name={lg['name']!r}  components={len(lg_map.components)}  "
              f"bound_method={len(lg_boundmethod)}  sibling_agent_classes={len(lg_agent_classes)}  "
              f"topology_edges={lg_edges}  framework={lg_map.framework!r}")
        print(f"  (3) plain_python : name={plain['name']!r}  components={len(plain_map.components)}  "
              f"QAAgent_present={has_qaagent}  framework={plain_map.framework!r}")

        print("\n=== RESULT ===")
        if failures:
            for f in failures:
                print(f"  FAIL: {f}")
            print("ACCEPTANCE: FAILED")
            return 1
        print("  THREE clean single-framework systems: Haystack, LangGraph (deep_research, "
              "non-orphaned), plain_python (QAAgent). No blended framework anywhere.")
        print("ACCEPTANCE: PASSED")
        return 0
    finally:
        if llm is not None:
            await llm.aclose()
        await aeh_close_db()
        await backend_close_db()


def main() -> None:
    ap = argparse.ArgumentParser(description="CS-317 real Stage-1->2 mixed-framework acceptance harness.")
    ap.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    ap.add_argument("--effort", default=DEFAULT_EFFORT, choices=["low", "medium", "high"])
    ap.add_argument("--provider-id", default=None)
    ap.add_argument("--node-budget", type=int, default=100)
    ap.add_argument("--hop-cap", type=int, default=3)
    args = ap.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
