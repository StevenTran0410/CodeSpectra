"""Deep Research Agent — iterative codebase investigation via graph + retrieval."""

from __future__ import annotations

import json
import time
from typing import Any

from infrastructure.db.database import get_db
from domain.model_connector.service import ProviderConfigService
from domain.retrieval.service import RetrievalService
from domain.retrieval.types import RetrievalEvidence, RetrievalMode, RetrievalSection, RetrieveRequest
from domain.analysis.agents.base import BaseTypedAgent
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

_SYNTHESIZE_SYSTEM = """\
You are synthesizing a codebase investigation into a final answer.
You receive the original question and all step findings.
Produce a comprehensive answer with a reasoning chain.

FORMATTING — MANDATORY:
- Write "summary" in rich Markdown (use ### headings, **bold**, bullet lists, code blocks).
- Keep "reasoning_chain" concise — each step gets 1-2 sentences.
- List all actual unknowns — do not fabricate certainty.

Output JSON: {
  "summary": "rich markdown answer",
  "reasoning_chain": [{"step_number": 1, "description": "...", "files_involved": [...], "finding": "...", "graph_path": [...]}],
  "confidence": "high|medium|low",
  "unknowns": ["list of what could not be determined"]
}
Output ONLY the JSON object.
"""

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
    ) -> dict[str, Any]:
        """Execute a deep research investigation."""
        t0 = time.monotonic()
        debug: dict[str, Any] = {} if include_debug else {}

        # Step 1: Plan the investigation
        plan = await self._plan(question, provider_id, model_id)
        if include_debug:
            debug["plan"] = plan

        # Step 2: Execute each plan step
        step_findings: list[dict[str, Any]] = []
        visited_files: set[str] = set()
        all_evidence_blocks: list[str] = []

        for plan_step in plan:
            step_type = plan_step.get("type", "retrieve")
            target = plan_step.get("target", question)
            description = plan_step.get("description", "")

            # Graph operations
            new_files: list[str] = []
            graph_path: list[str] | None = None

            if step_type == "trace_forward":
                trace = await trace_call_chain(snapshot_id, target, "forward", max_hops=min(max_hops, 4))
                new_files = [s.file for s in trace if s.file not in visited_files]
                if len(trace) > 1:
                    graph_path = [s.file for s in trace[:6]]

            elif step_type == "trace_backward":
                trace = await trace_call_chain(snapshot_id, target, "backward", max_hops=min(max_hops, 4))
                new_files = [s.file for s in trace if s.file not in visited_files]
                if len(trace) > 1:
                    graph_path = [s.file for s in trace[:6]]

            elif step_type == "impact":
                impact = await get_impact_cone(snapshot_id, target, hops=3)
                new_files = [f for f in impact.impacted_files if f not in visited_files]

            else:  # "retrieve"
                new_files = []  # retrieval handles its own file discovery

            # Retrieve evidence for discovered files + the query itself
            evidence = await self._retrieve_evidence(
                snapshot_id,
                target if step_type == "retrieve" else (question + " " + " ".join(new_files[:3])),
                forced_files=new_files[:8],
            )
            visited_files.update(ev.rel_path for ev in evidence)

            evidence_block = render_bundle_subset(evidence, limit=12, excerpt_chars=1200)
            all_evidence_blocks.append(evidence_block)

            # LLM analyzes this step
            prior_context = "\n".join(f"Step {i+1}: {f['finding']}" for i, f in enumerate(step_findings))
            user_prompt = (
                f"ORIGINAL QUESTION: {question}\n\n"
                f"CURRENT STEP: {description}\n"
                f"STEP TYPE: {step_type}, TARGET: {target}\n\n"
            )
            if prior_context:
                user_prompt += f"PRIOR FINDINGS:\n{prior_context}\n\n"
            if graph_path:
                user_prompt += f"GRAPH PATH DISCOVERED: {' → '.join(graph_path)}\n\n"
            user_prompt += f"CODE EVIDENCE:\n{evidence_block}"

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

        # Step 3: Synthesize
        synthesis_user = (
            f"QUESTION: {question}\n\n"
            f"STEP FINDINGS:\n"
            + "\n".join(
                f"Step {f['step_number']}: {f['description']}\n  Finding: {f['finding']}\n  Files: {', '.join(f['files_involved'][:3])}"
                for f in step_findings
            )
        )
        synth_schema = (
            '{"summary": "string", "reasoning_chain": [{"step_number": 0, "description": "string", '
            '"files_involved": ["string"], "finding": "string", "graph_path": ["string"]}], '
            '"confidence": "high|medium|low", "unknowns": ["string"]}'
        )
        synthesis = await self._chat_json_typed(
            provider_id,
            model_id,
            _SYNTHESIZE_SYSTEM,
            synthesis_user,
            synth_schema,
            max_completion_tokens=2000,
        )

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        result: dict[str, Any] = {
            "summary": str(synthesis.get("summary") or ""),
            "reasoning_chain": synthesis.get("reasoning_chain") or [],
            "files_explored": sorted(visited_files),
            "confidence": str(synthesis.get("confidence") or "medium").lower(),
            "unknowns": synthesis.get("unknowns") or [],
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

    async def _plan(self, question: str, provider_id: str, model_id: str) -> list[dict]:
        """Ask LLM to produce an investigation plan. Falls back to single retrieve step."""
        plan_schema = '{"steps": [{"type": "retrieve|trace_forward|trace_backward|impact", "target": "string", "description": "string"}]}'
        try:
            result = await self._chat_json_typed(
                provider_id,
                model_id,
                _PLAN_SYSTEM,
                f"Question: {question}",
                plan_schema,
                max_completion_tokens=500,
            )
            steps = result.get("steps") or []
            if isinstance(steps, list) and steps:
                return [s for s in steps if isinstance(s, dict) and s.get("type") and s.get("target")][:5]
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
