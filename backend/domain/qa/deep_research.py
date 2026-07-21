"""Deep Research Agent — iterative codebase investigation via LangGraph + retrieval."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

_ProgressCb = Callable[[dict], Awaitable[None]]

from infrastructure.db.database import get_db
from domain.model_connector.service import ProviderConfigService
from domain.retrieval.service import RetrievalService
from domain.retrieval.types import RetrievalEvidence, RetrievalMode, RetrievalSection, RetrieveRequest
from domain.analysis.agents.base import BaseTypedAgent
from domain.analysis.agents._graph_plan import plan_queries, retrieve_multi
from shared.logger import logger
from pydantic import BaseModel

from .graph_queries import (
    TraceStep,
    trace_call_chain,
    get_impact_cone,
)


@dataclass
class _GraphCtx:
    """Preloaded graph metadata for a research session (centrality + community)."""

    centrality: dict[str, int] = field(default_factory=dict)  # rel_path → score
    top_central: list[str] = field(default_factory=list)       # top-N files by score
    community_of: dict[str, int] = field(default_factory=dict) # rel_path → community_id
    community_hubs: dict[int, list[str]] = field(default_factory=dict)  # cid → hub files


class DeepResearchState(TypedDict, total=False):
    """LangGraph state for deep research investigation."""

    question: str
    snapshot_id: str
    provider_id: str
    model_id: str
    max_hops: int
    hop_cap: int
    gctx: dict
    plan: list[dict]
    plan_cursor: int
    step_findings: list[dict[str, Any]]
    visited_files: list[str]
    accumulated_chunk_ids: list[str]
    all_evidences: list[RetrievalEvidence]
    debug: dict[str, Any]
    step_type: str
    target: str
    description: str
    new_files: list[str]
    graph_path: list[str] | None
    graph_path_with_conf: list[str] | None
    tentative_files: list[str]
    tentative_confidence: dict[str, float]
    current_evidence: list[RetrievalEvidence]
    sufficient: bool
    summary: str
    reasoning_chain: list[dict]
    confidence: str
    unknowns: list[str]


async def _load_graph_ctx(snapshot_id: str) -> _GraphCtx:
    """Load centrality + community data once at research start. Silently returns
    empty context if data is not yet built for this snapshot."""
    ctx = _GraphCtx()
    db = get_db()
    try:
        async with db.execute(
            "SELECT top_central_files FROM structural_graph_summaries WHERE snapshot_id=?",
            (snapshot_id,),
        ) as cur:
            row = await cur.fetchone()
        if row:
            for f in json.loads(row["top_central_files"] or "[]"):
                ctx.centrality[f["rel_path"]] = int(f.get("score", 0))
            ctx.top_central = sorted(ctx.centrality, key=lambda p: -ctx.centrality[p])[:20]
    except Exception as exc:
        logger.debug("[DeepResearch._load_graph_ctx] centrality unavailable: %s", exc)

    try:
        async with db.execute(
            "SELECT node_path, community_id FROM graph_community_members WHERE snapshot_id=?",
            (snapshot_id,),
        ) as cur:
            rows = await cur.fetchall()
        ctx.community_of = {r["node_path"]: r["community_id"] for r in rows}

        async with db.execute(
            "SELECT community_id, hub_paths FROM graph_community_summaries WHERE snapshot_id=?",
            (snapshot_id,),
        ) as cur:
            rows = await cur.fetchall()
        ctx.community_hubs = {
            r["community_id"]: json.loads(r["hub_paths"] or "[]") for r in rows
        }
    except Exception as exc:
        logger.debug("[DeepResearch._load_graph_ctx] community unavailable: %s", exc)

    return ctx


def _symbol_graph_path(trace: list[TraceStep], limit: int = 6) -> list[str]:
    """Build a symbol-level graph path string from a trace.
    Prefers 'file::symbol' format when symbols are available."""
    path: list[str] = []
    for step in trace[:limit]:
        if step.symbols_involved:
            sym = step.symbols_involved[0]
            path.append(sym)
        else:
            path.append(step.file)
    return path


def _path_confidence_breakdown(trace: list[TraceStep], limit: int = 6) -> tuple[float, list[float]]:
    """Extract final path confidence and per-hop confidence scores from trace.

    Returns:
        (final_path_confidence, hop_confidences) where hop_confidences is the
        confidence_score for each hop in the path (excluding seed step 0).
    """
    if not trace:
        return 1.0, []
    final_conf = trace[-1].path_confidence if trace else 1.0
    hop_confs = []
    for step in trace[1:limit]:
        max_conf = max((h.confidence_score for h in step.hops), default=1.0)
        hop_confs.append(max_conf)
    return final_conf, hop_confs


def _community_context_block(files: list[str], ctx: _GraphCtx) -> str:
    """Produce a compact module-context note for a list of files."""
    if not ctx.community_of:
        return ""
    seen_cids: set[int] = set()
    lines: list[str] = []
    for f in files:
        cid = ctx.community_of.get(f)
        if cid is not None and cid not in seen_cids:
            seen_cids.add(cid)
            hubs = ctx.community_hubs.get(cid, [])[:3]
            if hubs:
                lines.append(f"  Module {cid} key files: {', '.join(hubs)}")
    return ("MODULE CONTEXT:\n" + "\n".join(lines) + "\n\n") if lines else ""


class ResearchCitation(BaseModel):
    """Citation reference in research results."""

    file: str
    line_start: int | None = None
    line_end: int | None = None
    snippet: str = ""


class ResearchStepResult(BaseModel):
    """Single investigation step result."""

    step_number: int
    description: str
    files_involved: list[str]
    finding: str
    graph_path: list[str] | None = None
    tentative_files: list[str] = []
    tentative_confidence: dict[str, float] = {}


class DeepResearchResult(BaseModel):
    """Final deep research response."""

    summary: str
    reasoning_chain: list[ResearchStepResult]
    files_explored: list[str]
    confidence: str  # high | medium | low
    unknowns: list[str]
    elapsed_ms: int
    research_debug: dict | None = None


class DeepResearchRequest(BaseModel):
    """Deep research request parameters."""

    snapshot_id: str
    report_id: str | None = None
    question: str
    provider_id: str
    model_id: str
    max_hops: int = 5
    include_debug: bool = False


_PLAN_SYSTEM = """\
You are a senior engineer planning a codebase investigation.
Given a question, produce an investigation plan with 2-5 concrete steps.

Each step must have:
- "type": one of "trace_forward" | "trace_backward" | "impact" | "retrieve"
- "target": file path (for trace/impact) or search query (for retrieve)
- "description": what this step investigates

Output JSON: {"steps": [{"type": "...", "target": "...", "description": "..."}]}
Rules:
- trace_forward: follow what a file/symbol calls downstream
- trace_backward: find who calls into a file/symbol
- impact: find all files that depend on a target file
- retrieve: keyword/semantic search for evidence

Start with a "retrieve" step to find seed files, then follow with trace steps.
Output ONLY the JSON object. No prose.
"""

_STEP_SYSTEM = """\
You are analyzing one step of a codebase investigation.
You receive:
- The original question
- The investigation step being analyzed
- Code evidence from relevant files
- Findings from previous steps

Analyze the evidence and report what you found for this step.
Output JSON: {
  "finding": "string — what was discovered in this step",
  "key_files": ["list of the 1-3 most important files in this evidence"],
  "graph_path": ["ordered file chain if a dependency path was found, else null"],
  "sufficient": true/false — whether we have enough to answer the question
}
Output ONLY the JSON object.
"""

_SYNTHESIZE_ANSWER_SYSTEM = """\
You are synthesizing a codebase investigation into a final answer.
You receive the original question, all step findings, AND the actual code evidence
collected across every investigation step.

Use the code evidence to produce a precise, code-grounded answer — cite specific
functions, file paths, and actual logic from the evidence. When citing a line number,
copy it from the `lines=start-end` range printed on the evidence chunk — never guess
or estimate one from the excerpt text. Do NOT invent details that are not in the
evidence; state them as unknowns.

FORMATTING — MANDATORY:
- Write in rich Markdown (use ### headings, **bold**, bullet lists, ```code blocks```).
- Include actual code snippets where relevant.
- Use --- horizontal rules to separate major sections.
- Do NOT write a flat wall of text.

Respond with ONLY the markdown answer. No JSON, no preamble.
"""

_SYNTHESIZE_META_SYSTEM = """\
You are extracting structured metadata from a completed codebase investigation.
Given the question, step findings, and the synthesized answer, output ONLY a JSON object:
{
  "reasoning_chain": [
    {"step_number": 1, "description": "string", "files_involved": ["string"],
     "finding": "string", "graph_path": ["string"]}
  ],
  "confidence": "high|medium|low",
  "unknowns": ["string"]
}
Output ONLY the JSON object. No markdown, no prose.
"""

_SYNTHESIZE_META_SCHEMA = (
    '{"reasoning_chain": [{"step_number": 1, "description": "string", '
    '"files_involved": ["string"], "finding": "string", "graph_path": ["string"], '
    '"tentative_files": ["string"], "tentative_confidence": {"file": 0.5}}], '
    '"confidence": "high|medium|low", "unknowns": ["string"]}'
)

_RESEARCH_RETRIEVAL_BUDGET = 8_000


class DeepResearchAgent(BaseTypedAgent):
    """Agent that performs multi-step codebase research via LangGraph + retrieval."""

    def __init__(
        self,
        provider_service: ProviderConfigService,
        retrieval_service: RetrievalService,
    ) -> None:
        super().__init__(provider_service)
        self._retrieval = retrieval_service
        self._current_progress_cb: _ProgressCb | None = None

    async def research(
        self,
        question: str,
        snapshot_id: str,
        provider_id: str,
        model_id: str,
        max_hops: int = 5,
        include_debug: bool = False,
        progress_cb: _ProgressCb | None = None,
    ) -> dict[str, Any]:
        """Execute a deep research investigation via LangGraph."""
        t0 = time.monotonic()
        self._current_progress_cb = progress_cb

        graph = self._build_graph()
        data_dir = Path(os.getenv("CODESPECTRA_DATA_DIR", "."))
        data_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = str(data_dir / "deep_research_checkpoints.db")

        question_hash = hashlib.sha1(question.encode()).hexdigest()[:12]
        thread_id = f"{snapshot_id}:{question_hash}"

        debug: dict[str, Any] = {} if include_debug else {}
        if include_debug:
            debug["thread_id"] = thread_id

        initial_state: DeepResearchState = {
            "question": question,
            "snapshot_id": snapshot_id,
            "provider_id": provider_id,
            "model_id": model_id,
            "max_hops": max_hops,
            "hop_cap": max_hops,
            "gctx": {},
            "plan": [],
            "plan_cursor": 0,
            "step_findings": [],
            "visited_files": [],
            "accumulated_chunk_ids": [],
            "all_evidences": [],
            "debug": debug,
        }

        async with AsyncSqliteSaver.from_conn_string(checkpoint_path) as saver:
            compiled = graph.compile(checkpointer=saver)
            config = {
                "configurable": {
                    "thread_id": thread_id,
                }
            }

            async for _ in compiled.astream(initial_state, config=config, stream_mode="values"):
                pass

            final = await compiled.aget_state(config)
            state = final.values

        self._current_progress_cb = None

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        result: dict[str, Any] = {
            "summary": state.get("summary", ""),
            "reasoning_chain": state.get("reasoning_chain") or [],
            "files_explored": sorted(set(state.get("visited_files", []))),
            "confidence": str(state.get("confidence", "medium")).lower(),
            "unknowns": state.get("unknowns") or [],
            "elapsed_ms": elapsed_ms,
            "research_debug": debug if include_debug else None,
        }
        if result["confidence"] not in ("high", "medium", "low"):
            result["confidence"] = "medium"

        if include_debug:
            result["research_debug"]["thread_id"] = thread_id

        logger.info(
            "[DeepResearchAgent] done in %dms, steps=%d, files_explored=%d, confidence=%s",
            elapsed_ms,
            len(state.get("step_findings", [])),
            len(state.get("visited_files", [])),
            result["confidence"],
        )
        return result

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph StateGraph for deep research."""
        graph = StateGraph(DeepResearchState)

        graph.add_edge(START, "load_context")
        graph.add_edge("load_context", "plan")

        graph.add_node("load_context", self._node_load_context)
        graph.add_node("plan", self._node_plan)
        graph.add_node("trace_forward", self._node_trace_forward)
        graph.add_node("trace_backward", self._node_trace_backward)
        graph.add_node("impact", self._node_impact)
        graph.add_node("retrieve", self._node_retrieve)
        graph.add_node("analyze_step", self._node_analyze_step)
        graph.add_node("synthesize", self._node_synthesize)

        graph.add_conditional_edges(
            "plan",
            self._route_step_fn,
            {
                "trace_forward": "trace_forward",
                "trace_backward": "trace_backward",
                "impact": "impact",
                "retrieve": "retrieve",
                "synthesize": "synthesize",
            }
        )

        graph.add_conditional_edges(
            "analyze_step",
            self._route_step_fn,
            {
                "trace_forward": "trace_forward",
                "trace_backward": "trace_backward",
                "impact": "impact",
                "retrieve": "retrieve",
                "synthesize": "synthesize",
            }
        )

        graph.add_edge("trace_forward", "analyze_step")
        graph.add_edge("trace_backward", "analyze_step")
        graph.add_edge("impact", "analyze_step")
        graph.add_edge("retrieve", "analyze_step")
        graph.add_edge("synthesize", END)

        return graph

    def _route_step_fn(self, state: DeepResearchState) -> str:
        """Conditional routing function: plan -> execute or synthesize."""
        if state.get("plan_cursor", 0) >= len(state.get("plan", [])) or state.get("sufficient", False):
            return "synthesize"
        plan = state.get("plan", [])
        if plan:
            step_type = plan[state["plan_cursor"]].get("type", "retrieve")
            return step_type
        return "synthesize"

    async def _node_load_context(self, state: DeepResearchState) -> DeepResearchState:
        """Load centrality + community context."""
        snapshot_id = state["snapshot_id"]
        gctx = await _load_graph_ctx(snapshot_id)
        logger.debug(
            "[DeepResearch] graph ctx: %d central files, %d community nodes",
            len(gctx.top_central), len(gctx.community_of),
        )
        state["gctx"] = {
            "centrality": gctx.centrality,
            "top_central": gctx.top_central,
            "community_of": gctx.community_of,
            "community_hubs": gctx.community_hubs,
        }
        return state

    async def _node_plan(self, state: DeepResearchState) -> DeepResearchState:
        """Plan the investigation."""
        await self._emit(self._current_progress_cb, "planning", "Planning investigation...")

        gctx_dict = state.get("gctx", {})
        gctx = _GraphCtx(
            centrality=gctx_dict.get("centrality", {}),
            top_central=gctx_dict.get("top_central", []),
            community_of=gctx_dict.get("community_of", {}),
            community_hubs=gctx_dict.get("community_hubs", {}),
        )

        plan = await self._plan(
            state["question"],
            state["provider_id"],
            state["model_id"],
            gctx,
            state["snapshot_id"],
        )

        hop_cap = state["max_hops"]
        if len(plan) >= 4 and any(s.get("type") in ("trace_forward", "trace_backward") for s in plan):
            hop_cap = min(3, state["max_hops"])
            logger.debug("[DeepResearch] plan has %d steps with trace ops; capping hops at %d", len(plan), hop_cap)

        if state["debug"]:
            state["debug"]["plan"] = plan

        state["plan"] = plan
        state["hop_cap"] = hop_cap
        state["plan_cursor"] = 0
        return state

    async def _node_trace_forward(self, state: DeepResearchState) -> DeepResearchState:
        """Execute trace_forward step."""
        return await self._execute_trace_step(state, "forward")

    async def _node_trace_backward(self, state: DeepResearchState) -> DeepResearchState:
        """Execute trace_backward step."""
        return await self._execute_trace_step(state, "backward")

    async def _execute_trace_step(self, state: DeepResearchState, direction: str) -> DeepResearchState:
        """Execute a trace_forward or trace_backward step."""
        plan = state.get("plan", [])
        cursor = state.get("plan_cursor", 0)

        if cursor >= len(plan):
            return state

        plan_step = plan[cursor]
        target = plan_step.get("target", state["question"])
        description = plan_step.get("description", "")
        visited_files_set = set(state.get("visited_files", []))

        trace = await trace_call_chain(
            state["snapshot_id"], target, direction,
            max_hops=min(state["hop_cap"], 4), high_confidence_only=True,
        )
        new_files = [s.file for s in trace if s.file not in visited_files_set]

        tentative_files: list[str] = []
        tentative_confidence: dict[str, float] = {}
        graph_path_with_conf: list[str] | None = None

        if len(new_files) < 2:
            trace_low = await trace_call_chain(
                state["snapshot_id"], target, direction,
                max_hops=min(state["hop_cap"], 4), high_confidence_only=False,
            )
            high_conf_files = {s.file for s in trace}
            extra = [
                s.file for s in trace_low
                if s.file not in visited_files_set and s.file not in high_conf_files
            ]
            if extra:
                logger.debug("[DeepResearch] confidence fallback added %d files", len(extra))
                new_files.extend(extra[:5])
                tentative_files = extra[:5]
                for file in tentative_files:
                    for step in trace_low:
                        if step.file == file:
                            max_conf = max((h.confidence_score for h in step.hops), default=1.0)
                            tentative_confidence[file] = max_conf
                            break
            trace = trace_low

        if len(trace) > 1:
            graph_path = _symbol_graph_path(trace, limit=6)
            path_conf, hop_confs = _path_confidence_breakdown(trace, limit=6)
            hop_str = "→".join(f"{c:.2f}" for c in hop_confs) if hop_confs else ""
            confidence_note = f" (confidence: {path_conf:.2f}" + (f", hops: {hop_str}" if hop_str else "") + ")"
            graph_path_with_conf = [confidence_note] + graph_path
            state["graph_path"] = graph_path
            state["graph_path_with_conf"] = graph_path_with_conf

        gctx_dict = state.get("gctx", {})
        centrality = gctx_dict.get("centrality", {})
        if centrality and new_files:
            new_files = sorted(new_files, key=lambda f: -centrality.get(f, 0))

        state["new_files"] = new_files
        state["tentative_files"] = tentative_files
        state["tentative_confidence"] = tentative_confidence

        step_num = len(state.get("step_findings", [])) + 1
        await self._emit(self._current_progress_cb, "retrieving", description or f"Step {step_num}: {plan_step.get('type')}", step=step_num)

        evidence = await self._retrieve_evidence(
            state["snapshot_id"],
            description or state["question"],
            forced_files=new_files[:8],
        )
        if not evidence:
            logger.warning(
                "[DeepResearch.step%d] trace returned 0 chunks; falling back to retrieve with target='%s'",
                step_num, target
            )
            evidence = await self._retrieve_evidence(
                state["snapshot_id"],
                target,
                forced_files=[],
            )

        state = self._accumulate_evidence(state, evidence, new_files, description or f"Step {step_num}: {plan_step.get('type')}")
        return state

    async def _node_impact(self, state: DeepResearchState) -> DeepResearchState:
        """Execute impact step."""
        plan = state.get("plan", [])
        cursor = state.get("plan_cursor", 0)

        if cursor >= len(plan):
            return state

        plan_step = plan[cursor]
        target = plan_step.get("target", state["question"])
        description = plan_step.get("description", "")
        visited_files_set = set(state.get("visited_files", []))

        impact = await get_impact_cone(state["snapshot_id"], target, hops=3)
        new_files = [f for f in impact.impacted_files if f not in visited_files_set]

        gctx_dict = state.get("gctx", {})
        centrality = gctx_dict.get("centrality", {})
        if centrality and new_files:
            new_files = sorted(new_files, key=lambda f: -centrality.get(f, 0))

        state["new_files"] = new_files
        state["tentative_files"] = []
        state["tentative_confidence"] = {}

        step_num = len(state.get("step_findings", [])) + 1
        await self._emit(self._current_progress_cb, "retrieving", description or f"Step {step_num}: impact", step=step_num)

        evidence = await self._retrieve_evidence(
            state["snapshot_id"],
            description or state["question"],
            forced_files=new_files[:8],
        )

        state = self._accumulate_evidence(state, evidence, new_files, description or f"Step {step_num}: impact")
        return state

    async def _node_retrieve(self, state: DeepResearchState) -> DeepResearchState:
        """Execute retrieve step."""
        plan = state.get("plan", [])
        cursor = state.get("plan_cursor", 0)

        if cursor >= len(plan):
            return state

        plan_step = plan[cursor]
        target = plan_step.get("target", state["question"])
        description = plan_step.get("description", "")

        state["new_files"] = []
        state["tentative_files"] = []
        state["tentative_confidence"] = {}

        step_num = len(state.get("step_findings", [])) + 1
        await self._emit(self._current_progress_cb, "retrieving", description or f"Step {step_num}: retrieve", step=step_num)

        sub_queries = await plan_queries(
            goal=description or target,
            provider_service=self._providers,
            provider_id=state["provider_id"],
            model_id=state["model_id"],
            fallback=[target],
        )
        bundle = await retrieve_multi(
            self._retrieval,
            state["snapshot_id"],
            sub_queries,
            RetrievalSection.QA,
            RetrievalMode.HYBRID,
            max_results_each=10,
        )
        evidence = bundle.evidences
        if bundle.quality:
            logger.info(
                "[DeepResearch.step%d] retrieval_quality flags=%s label=%s",
                step_num, bundle.quality.flags, bundle.quality.quality_label
            )

        state = self._accumulate_evidence(state, evidence, [], description or f"Step {step_num}: retrieve")
        return state

    async def _node_analyze_step(self, state: DeepResearchState) -> DeepResearchState:
        """Analyze the current step and decide if sufficient."""
        plan = state.get("plan", [])
        cursor = state.get("plan_cursor", 0)

        if cursor >= len(plan):
            return state

        plan_step = plan[cursor]
        step_type = plan_step.get("type", "retrieve")
        target = plan_step.get("target", state["question"])
        description = plan_step.get("description", "")
        new_files = state.get("new_files", [])

        step_num = len(state.get("step_findings", [])) + 1
        await self._emit(self._current_progress_cb, "thinking", f"Analyzing step {step_num}...", step=step_num)

        step_findings = state.get("step_findings", [])
        all_evidences = state.get("all_evidences", [])
        current_evidence = state.get("current_evidence", [])

        prior_context = "\n".join(f"Step {i+1}: {f['finding']}" for i, f in enumerate(step_findings))
        user_prompt = (
            f"ORIGINAL QUESTION: {state['question']}\n\n"
            f"CURRENT STEP: {description}\n"
            f"STEP TYPE: {step_type}, TARGET: {target}\n\n"
        )
        if prior_context:
            user_prompt += f"PRIOR FINDINGS:\n{prior_context}\n\n"

        gctx_dict = state.get("gctx", {})
        gctx = _GraphCtx(
            centrality=gctx_dict.get("centrality", {}),
            top_central=gctx_dict.get("top_central", []),
            community_of=gctx_dict.get("community_of", {}),
            community_hubs=gctx_dict.get("community_hubs", {}),
        )
        all_step_files = list({ev.rel_path for ev in current_evidence} | set(new_files))
        community_block = _community_context_block(all_step_files, gctx)
        if community_block:
            user_prompt += community_block

        prior_evidences = [ev for ev in all_evidences if ev.chunk_id not in {e.chunk_id for e in current_evidence}]
        prior_evidence_block = (
            render_bundle_subset(prior_evidences, limit=8, excerpt_chars=500)
            if prior_evidences else ""
        )
        if prior_evidence_block:
            user_prompt += f"PREVIOUSLY SEEN CODE (condensed):\n{prior_evidence_block}\n\n"

        graph_path = state.get("graph_path")
        if graph_path:
            if len(graph_path) > 1 and isinstance(graph_path[0], str) and graph_path[0].startswith(" (confidence:"):
                user_prompt += f"CALL GRAPH PATH (symbol-level){graph_path[0]}: {' → '.join(graph_path[1:])}\n\n"
            else:
                user_prompt += f"CALL GRAPH PATH (symbol-level): {' → '.join(graph_path)}\n\n"

        current_evidence_block = render_bundle_subset(current_evidence, limit=12, excerpt_chars=1200)
        user_prompt += f"CURRENT STEP CODE EVIDENCE:\n{current_evidence_block}"

        step_schema = '{"finding": "string", "key_files": ["string"], "graph_path": ["string"], "sufficient": true}'
        step_result = await self._chat_json_typed(
            state["provider_id"],
            state["model_id"],
            _STEP_SYSTEM,
            user_prompt,
            step_schema,
            max_completion_tokens=800,
        )

        stored_graph_path = state.get("graph_path_with_conf") or state.get("graph_path")
        state["step_findings"].append(
            {
                "step_number": len(state.get("step_findings", [])) + 1,
                "description": description,
                "files_involved": step_result.get("key_files") or new_files[:5],
                "finding": str(step_result.get("finding") or ""),
                "graph_path": step_result.get("graph_path") or stored_graph_path,
                "tentative_files": state.get("tentative_files", []),
                "tentative_confidence": state.get("tentative_confidence", {}),
            }
        )

        state["sufficient"] = bool(step_result.get("sufficient", False))
        state["plan_cursor"] += 1
        state["graph_path"] = None
        state["graph_path_with_conf"] = None
        state["current_evidence"] = []

        return state

    async def _node_synthesize(self, state: DeepResearchState) -> DeepResearchState:
        """Synthesize all findings into final answer."""
        await self._emit(self._current_progress_cb, "synthesizing", "Synthesizing all findings...")

        step_findings = state.get("step_findings", [])
        all_evidences = state.get("all_evidences", [])

        accumulated_evidence_block = render_bundle_subset(all_evidences, limit=25, excerpt_chars=700)
        findings_text = "\n".join(
            f"Step {f['step_number']}: {f['description']}\n"
            f"  Finding: {f['finding']}\n"
            f"  Files: {', '.join(f['files_involved'][:3])}"
            + (f"\n  [Note: {len(f.get('tentative_files', []))} tentative (low-confidence fallback)]"
               if f.get('tentative_files') else "")
            for f in step_findings
        )
        synthesis_base = (
            f"QUESTION: {state['question']}\n\n"
            f"STEP FINDINGS:\n{findings_text}"
            f"\n\nACCUMULATED CODE EVIDENCE (all investigation steps):\n{accumulated_evidence_block}"
        )

        async def _on_token(tok: str) -> None:
            if self._current_progress_cb:
                await self._current_progress_cb({"type": "token", "text": tok})

        summary_text = await self._call_stream(
            state["provider_id"],
            state["model_id"],
            _SYNTHESIZE_ANSWER_SYSTEM,
            synthesis_base,
            max_completion_tokens=3000,
            on_token=_on_token,
        )

        meta_user = (
            f"QUESTION: {state['question']}\n\n"
            f"STEP FINDINGS:\n{findings_text}\n\n"
            f"SYNTHESIZED ANSWER (abbreviated):\n{summary_text[:2000]}"
        )
        synthesis_meta = await self._chat_json_typed(
            state["provider_id"],
            state["model_id"],
            _SYNTHESIZE_META_SYSTEM,
            meta_user,
            _SYNTHESIZE_META_SCHEMA,
            max_completion_tokens=800,
        )

        state["summary"] = summary_text
        state["reasoning_chain"] = synthesis_meta.get("reasoning_chain") or []
        state["confidence"] = str(synthesis_meta.get("confidence") or "medium").lower()
        state["unknowns"] = synthesis_meta.get("unknowns") or []

        return state

    def _accumulate_evidence(
        self,
        state: DeepResearchState,
        evidence: list[RetrievalEvidence],
        new_files: list[str],
        description: str,
    ) -> DeepResearchState:
        """Accumulate evidence and track visited files."""
        del new_files, description
        visited = set(state.get("visited_files", []))
        visited.update(ev.rel_path for ev in evidence)
        state["visited_files"] = sorted(visited)

        accumulated_chunk_ids = set(state.get("accumulated_chunk_ids", []))
        all_evidences = list(state.get("all_evidences", []))

        for ev in evidence:
            if ev.chunk_id not in accumulated_chunk_ids:
                all_evidences.append(ev)
                accumulated_chunk_ids.add(ev.chunk_id)

        state["accumulated_chunk_ids"] = sorted(accumulated_chunk_ids)
        state["all_evidences"] = all_evidences
        state["current_evidence"] = evidence

        return state

    async def _emit(
        self,
        progress_cb: _ProgressCb | None,
        phase: str,
        detail: str = "",
        step: int | None = None,
    ) -> None:
        """Emit a progress event."""
        if progress_cb:
            ev: dict = {"type": "status", "phase": phase, "detail": detail}
            if step is not None:
                ev["step"] = step
            await progress_cb(ev)

    async def _validate_plan_step(
        self,
        step: dict,
        snapshot_id: str,
    ) -> dict:
        """Validate that a plan step's target exists in snapshot. Adds warning if not found."""
        target = step.get("target", "")
        step_type = step.get("type", "retrieve")
        if step_type in ("trace_forward", "trace_backward") and target:
            db = get_db()
            async with db.execute(
                "SELECT 1 FROM manifest_files WHERE snapshot_id=? AND rel_path LIKE ? LIMIT 1",
                (snapshot_id, f"%{target}%"),
            ) as cur:
                row = await cur.fetchone()
            if not row:
                async with db.execute(
                    "SELECT 1 FROM code_symbols WHERE snapshot_id=? AND name LIKE ? LIMIT 1",
                    (snapshot_id, f"%{target}%"),
                ) as cur:
                    row2 = await cur.fetchone()
                if not row2:
                    step["_warning"] = f"target '{target}' not found in snapshot — step may produce empty results"
        return step

    async def _plan(
        self,
        question: str,
        provider_id: str,
        model_id: str,
        gctx: _GraphCtx | None = None,
        snapshot_id: str | None = None,
    ) -> list[dict]:
        """Ask LLM to produce an investigation plan. Falls back to single retrieve step."""
        plan_schema = '{"steps": [{"type": "retrieve|trace_forward|trace_backward|impact", "target": "string", "description": "string"}]}'
        plan_user = f"Question: {question}"
        if gctx and gctx.top_central:
            plan_user += (
                f"\n\nTop architectural files in this codebase (high centrality): "
                f"{', '.join(gctx.top_central[:12])}"
            )
        try:
            result = await self._chat_json_typed(
                provider_id,
                model_id,
                _PLAN_SYSTEM,
                plan_user,
                plan_schema,
                max_completion_tokens=500,
            )
            steps = result.get("steps") or []
            if isinstance(steps, list) and steps:
                parsed_steps = [s for s in steps if isinstance(s, dict) and s.get("type") and s.get("target")][:5]
                if snapshot_id:
                    validated_steps = []
                    for step in parsed_steps:
                        validated_step = await self._validate_plan_step(step, snapshot_id)
                        if "_warning" in validated_step:
                            logger.warning("[DeepResearchAgent._plan] %s", validated_step["_warning"])
                        validated_steps.append(validated_step)
                    return validated_steps
                return parsed_steps
        except Exception as exc:
            logger.warning("[DeepResearchAgent._plan] failed: %s", exc)
        return [{"type": "retrieve", "target": question, "description": "Retrieve initial evidence"}]

    async def _retrieve_evidence(
        self,
        snapshot_id: str,
        query: str,
        forced_files: list[str] | None = None,
    ) -> list[RetrievalEvidence]:
        """Retrieve code evidence for a query, optionally injecting specific files."""
        evidences: list[RetrievalEvidence] = []

        try:
            bundle = await self._retrieval.retrieve(
                RetrieveRequest(
                    snapshot_id=snapshot_id,
                    query=query,
                    section=RetrievalSection.QA,
                    mode=RetrievalMode.HYBRID,
                    max_results=15,
                )
            )
            evidences.extend(bundle.evidences)
        except Exception as exc:
            logger.warning("[DeepResearchAgent._retrieve_evidence] retrieval failed: %s", exc)

        if forced_files:
            existing_paths = {ev.rel_path for ev in evidences}
            db = get_db()
            used = sum(ev.token_estimate for ev in evidences)
            for path in forced_files:
                if path in existing_paths or used >= _RESEARCH_RETRIEVAL_BUDGET:
                    continue
                async with db.execute(
                    """SELECT id, rel_path, chunk_index, content, token_estimate, start_line, end_line
                       FROM retrieval_chunks
                       WHERE snapshot_id=? AND rel_path=?
                       ORDER BY chunk_index ASC LIMIT 3""",
                    (snapshot_id, path),
                ) as cur:
                    rows = await cur.fetchall()
                for row in rows:
                    if not row["content"]:
                        continue
                    tok = int(row["token_estimate"] or max(1, len(row["content"]) // 4))
                    if used + tok > _RESEARCH_RETRIEVAL_BUDGET:
                        break
                    evidences.append(
                        RetrievalEvidence(
                            chunk_id=row["id"],
                            rel_path=row["rel_path"],
                            chunk_index=row["chunk_index"],
                            reason_codes=["deep-research-graph-inject"],
                            score=0.5,
                            token_estimate=tok,
                            excerpt=row["content"],
                            start_line=row["start_line"] or 0,
                            end_line=row["end_line"] or 0,
                        )
                    )
                    used += tok
                    existing_paths.add(path)

        return evidences


def render_bundle_subset(evidences: list[RetrievalEvidence], limit: int, excerpt_chars: int) -> str:
    """Render a list of RetrievalEvidence to a text block for LLM context.
    Mirrors render_bundle() from domain.analysis.prompts but takes a list directly."""
    from domain.retrieval.types import RetrievalBundle
    from domain.analysis.prompts import render_bundle

    if not evidences:
        return "(no evidence)"
    bundle = RetrievalBundle(
        snapshot_id="",
        mode=RetrievalMode.HYBRID,
        section=RetrievalSection.QA,
        query="deep-research",
        budget_tokens=0,
        used_tokens=0,
        evidences=evidences[:limit],
    )
    return render_bundle(bundle, limit=limit, excerpt_chars=excerpt_chars)
