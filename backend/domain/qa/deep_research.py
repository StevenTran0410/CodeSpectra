"""Deep Research Agent — iterative codebase investigation via graph + retrieval."""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

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
    ImpactResult,
    get_callees_of,
    get_callers_of,
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
            # Use the most specific symbol (strip file prefix for display)
            sym = step.symbols_involved[0]
            path.append(sym)
        else:
            path.append(step.file)
    return path


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

# Step 1 of synthesis: stream a rich markdown answer
_SYNTHESIZE_ANSWER_SYSTEM = """\
You are synthesizing a codebase investigation into a final answer.
You receive the original question, all step findings, AND the actual code evidence
collected across every investigation step.

Use the code evidence to produce a precise, code-grounded answer — cite specific
functions, file paths, line numbers, and actual logic from the evidence.
Do NOT invent details that are not in the evidence; state them as unknowns.

FORMATTING — MANDATORY:
- Write in rich Markdown (use ### headings, **bold**, bullet lists, ```code blocks```).
- Include actual code snippets where relevant.
- Use --- horizontal rules to separate major sections.
- Do NOT write a flat wall of text.

Respond with ONLY the markdown answer. No JSON, no preamble.
"""

# Step 2 of synthesis: compact metadata JSON
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
    '"files_involved": ["string"], "finding": "string", "graph_path": ["string"]}], '
    '"confidence": "high|medium|low", "unknowns": ["string"]}'
)

_RESEARCH_RETRIEVAL_BUDGET = 8_000


class DeepResearchAgent(BaseTypedAgent):
    """Agent that performs multi-step codebase research via call chains and retrieval."""

    def __init__(
        self,
        provider_service: ProviderConfigService,
        retrieval_service: RetrievalService,
    ) -> None:
        super().__init__(provider_service)
        self._retrieval = retrieval_service

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
        """Execute a deep research investigation."""
        t0 = time.monotonic()
        debug: dict[str, Any] = {} if include_debug else {}

        async def _emit(phase: str, detail: str = "", step: int | None = None) -> None:
            if progress_cb:
                ev: dict = {"type": "status", "phase": phase, "detail": detail}
                if step is not None:
                    ev["step"] = step
                await progress_cb(ev)

        # ── Enhancement: load centrality + community context upfront ──────────
        gctx = await _load_graph_ctx(snapshot_id)
        logger.debug(
            "[DeepResearch] graph ctx: %d central files, %d community nodes",
            len(gctx.top_central), len(gctx.community_of),
        )

        # Step 1: Plan the investigation (hint planner with top central files)
        await _emit("planning", "Planning investigation...")
        plan = await self._plan(question, provider_id, model_id, gctx, snapshot_id)
        if include_debug:
            debug["plan"] = plan

        hop_cap = max_hops
        if len(plan) >= 4 and any(s.get("type") in ("trace_forward", "trace_backward") for s in plan):
            hop_cap = min(3, max_hops)
            logger.debug("[DeepResearch] plan has %d steps with trace ops; capping hops at %d", len(plan), hop_cap)

        # Step 2: Execute each plan step
        step_findings: list[dict[str, Any]] = []
        visited_files: set[str] = set()
        all_evidences: list[RetrievalEvidence] = []
        accumulated_chunk_ids: set[str] = set()

        for plan_step in plan:
            step_type = plan_step.get("type", "retrieve")
            target = plan_step.get("target", question)
            description = plan_step.get("description", "")

            # ── Graph operations ───────────────────────────────────────────────
            new_files: list[str] = []
            graph_path: list[str] | None = None

            if step_type in ("trace_forward", "trace_backward"):
                direction = "forward" if step_type == "trace_forward" else "backward"
                trace = await trace_call_chain(
                    snapshot_id, target, direction,
                    max_hops=min(hop_cap, 4), high_confidence_only=True,
                )
                new_files = [s.file for s in trace if s.file not in visited_files]

                # Enhancement 5: confidence fallback — if trace found very few files,
                # retry with low-confidence edges and mark them tentative
                if len(new_files) < 2:
                    trace_low = await trace_call_chain(
                        snapshot_id, target, direction,
                        max_hops=min(hop_cap, 4), high_confidence_only=False,
                    )
                    high_conf_files = {s.file for s in trace}
                    extra = [
                        s.file for s in trace_low
                        if s.file not in visited_files and s.file not in high_conf_files
                    ]
                    if extra:
                        logger.debug("[DeepResearch] confidence fallback added %d files", len(extra))
                        new_files.extend(extra[:5])
                    trace = trace_low  # use richer trace for symbol path

                # Enhancement 3: symbol-level graph path
                if len(trace) > 1:
                    graph_path = _symbol_graph_path(trace, limit=6)

            elif step_type == "impact":
                impact = await get_impact_cone(snapshot_id, target, hops=3)
                new_files = [f for f in impact.impacted_files if f not in visited_files]

            else:  # "retrieve"
                new_files = []

            # Enhancement 2: centrality-biased injection — sort forced_files by score
            if gctx.centrality and new_files:
                new_files = sorted(new_files, key=lambda f: -gctx.centrality.get(f, 0))

            # ── Retrieve evidence ──────────────────────────────────────────────
            step_num = len(step_findings) + 1
            await _emit("retrieving", description or f"Step {step_num}: {step_type}", step=step_num)
            if step_type == "retrieve":
                # Enhancement 1: multi-query decomposition for retrieve steps
                sub_queries = await plan_queries(
                    goal=description or target,
                    provider_service=self._providers,
                    provider_id=provider_id,
                    model_id=model_id,
                    fallback=[target],
                )
                bundle = await retrieve_multi(
                    self._retrieval,
                    snapshot_id,
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
            else:
                retrieval_query = description or question
                evidence = await self._retrieve_evidence(
                    snapshot_id,
                    retrieval_query,
                    forced_files=new_files[:8],
                )
                if not evidence and step_type in ("trace_forward", "trace_backward"):
                    logger.warning(
                        "[DeepResearch.step%d] trace returned 0 chunks; falling back to retrieve with target='%s'",
                        step_num, target
                    )
                    evidence = await self._retrieve_evidence(
                        snapshot_id,
                        target,
                        forced_files=[],
                    )

            visited_files.update(ev.rel_path for ev in evidence)

            # Accumulate evidence across steps (deduplicated by chunk_id)
            for ev in evidence:
                if ev.chunk_id not in accumulated_chunk_ids:
                    all_evidences.append(ev)
                    accumulated_chunk_ids.add(ev.chunk_id)

            # Current step evidence block (full detail)
            current_evidence_block = render_bundle_subset(evidence, limit=12, excerpt_chars=1200)

            # Prior accumulated evidence (condensed — gives each step LLM running context)
            prior_evidences = [ev for ev in all_evidences if ev.chunk_id not in {e.chunk_id for e in evidence}]
            prior_evidence_block = (
                render_bundle_subset(prior_evidences, limit=8, excerpt_chars=500)
                if prior_evidences else ""
            )

            # Enhancement 4: community context block
            all_step_files = list({ev.rel_path for ev in evidence} | set(new_files))
            community_block = _community_context_block(all_step_files, gctx)

            # ── LLM analyzes this step ─────────────────────────────────────────
            await _emit("thinking", f"Analyzing step {step_num}...", step=step_num)
            prior_context = "\n".join(f"Step {i+1}: {f['finding']}" for i, f in enumerate(step_findings))
            user_prompt = (
                f"ORIGINAL QUESTION: {question}\n\n"
                f"CURRENT STEP: {description}\n"
                f"STEP TYPE: {step_type}, TARGET: {target}\n\n"
            )
            if prior_context:
                user_prompt += f"PRIOR FINDINGS:\n{prior_context}\n\n"
            if community_block:
                user_prompt += community_block
            if prior_evidence_block:
                user_prompt += f"PREVIOUSLY SEEN CODE (condensed):\n{prior_evidence_block}\n\n"
            if graph_path:
                user_prompt += f"CALL GRAPH PATH (symbol-level): {' → '.join(graph_path)}\n\n"
            user_prompt += f"CURRENT STEP CODE EVIDENCE:\n{current_evidence_block}"

            step_schema = '{"finding": "string", "key_files": ["string"], "graph_path": ["string"], "sufficient": true}'
            step_result = await self._chat_json_typed(
                provider_id,
                model_id,
                _STEP_SYSTEM,
                user_prompt,
                step_schema,
                max_completion_tokens=800,
            )

            step_findings.append(
                {
                    "step_number": len(step_findings) + 1,
                    "description": description,
                    "files_involved": step_result.get("key_files") or new_files[:5],
                    "finding": str(step_result.get("finding") or ""),
                    "graph_path": step_result.get("graph_path") or graph_path,
                }
            )

            if step_result.get("sufficient"):
                break

        # Step 3: Synthesize — two-step: stream markdown first, then get metadata
        await _emit("synthesizing", "Synthesizing all findings...")
        accumulated_evidence_block = render_bundle_subset(all_evidences, limit=25, excerpt_chars=700)

        findings_text = "\n".join(
            f"Step {f['step_number']}: {f['description']}\n"
            f"  Finding: {f['finding']}\n"
            f"  Files: {', '.join(f['files_involved'][:3])}"
            for f in step_findings
        )
        synthesis_base = (
            f"QUESTION: {question}\n\n"
            f"STEP FINDINGS:\n{findings_text}"
            f"\n\nACCUMULATED CODE EVIDENCE (all investigation steps):\n{accumulated_evidence_block}"
        )

        # 3a. Stream the markdown answer
        async def _on_token(tok: str) -> None:
            if progress_cb:
                await progress_cb({"type": "token", "text": tok})

        summary_text = await self._call_stream(
            provider_id,
            model_id,
            _SYNTHESIZE_ANSWER_SYSTEM,
            synthesis_base,
            max_completion_tokens=3000,
            on_token=_on_token,
        )

        # 3b. Fast metadata call
        meta_user = (
            f"QUESTION: {question}\n\n"
            f"STEP FINDINGS:\n{findings_text}\n\n"
            f"SYNTHESIZED ANSWER (abbreviated):\n{summary_text[:2000]}"
        )
        synthesis_meta = await self._chat_json_typed(
            provider_id,
            model_id,
            _SYNTHESIZE_META_SYSTEM,
            meta_user,
            _SYNTHESIZE_META_SCHEMA,
            max_completion_tokens=800,
        )

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        result: dict[str, Any] = {
            "summary": summary_text,
            "reasoning_chain": synthesis_meta.get("reasoning_chain") or [],
            "files_explored": sorted(visited_files),
            "confidence": str(synthesis_meta.get("confidence") or "medium").lower(),
            "unknowns": synthesis_meta.get("unknowns") or [],
            "elapsed_ms": elapsed_ms,
            "research_debug": debug if include_debug else None,
        }
        if result["confidence"] not in ("high", "medium", "low"):
            result["confidence"] = "medium"

        logger.info(
            "[DeepResearchAgent] done in %dms, steps=%d, files_explored=%d, confidence=%s",
            elapsed_ms,
            len(step_findings),
            len(visited_files),
            result["confidence"],
        )
        return result

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
        # Hint planner with top architectural files so it can suggest better trace targets
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

        # Standard retrieval
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

        # Force-inject graph-discovered files not already in evidence
        if forced_files:
            existing_paths = {ev.rel_path for ev in evidences}
            db = get_db()
            used = sum(ev.token_estimate for ev in evidences)
            for path in forced_files:
                if path in existing_paths or used >= _RESEARCH_RETRIEVAL_BUDGET:
                    continue
                async with db.execute(
                    """SELECT id, rel_path, chunk_index, content, token_estimate
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
    # Build a minimal bundle and use the existing renderer
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
