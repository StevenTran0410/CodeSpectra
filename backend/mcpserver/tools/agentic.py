"""ask_codebase and deep_research — MCP Sampling over existing retrieval."""
from __future__ import annotations

import time

from mcp.server.fastmcp import Context, FastMCP
from mcp import types

from shared.logger import logger
from ..project_index import get_snapshot_id


# ── Sampling helper ──────────────────────────────────────────────────────────

async def _sample(ctx: Context, prompt: str, max_tokens: int = 1000) -> str:
    """Send a prompt to the host LLM (Claude) via MCP Sampling."""
    result = await ctx.session.create_message(
        messages=[types.SamplingMessage(
            role="user",
            content=types.TextContent(type="text", text=prompt),
        )],
        max_tokens=max_tokens,
    )
    return result.content.text


def _format_chunks(evidences) -> str:
    parts = []
    for e in evidences:
        parts.append(f"--- {e.rel_path} (score: {e.score:.2f}) ---\n{e.excerpt[:800]}")
    return "\n\n".join(parts)


def _dedup(evidences, max_items: int = 20):
    seen = set()
    result = []
    for e in sorted(evidences, key=lambda x: x.score, reverse=True):
        key = (e.rel_path, getattr(e, "chunk_index", 0))
        if key not in seen:
            seen.add(key)
            result.append(e)
        if len(result) >= max_items:
            break
    return result


# ── Tools ────────────────────────────────────────────────────────────────────

def register(mcp: FastMCP) -> None:

    @mcp.tool()
    async def ask_codebase(
        project_path: str,
        question: str,
        ctx: Context = None,
    ) -> dict:
        """Answer a question about the codebase using multi-round retrieval
        + MCP Sampling. Claude reasons over retrieved context — no API key needed."""
        sid = get_snapshot_id(project_path)

        from domain.retrieval.service import RetrievalService
        from domain.retrieval.types import RetrieveRequest, RetrievalSection

        svc = RetrievalService()

        # Round 1: initial retrieval
        r1 = await svc.retrieve(
            RetrieveRequest(
                snapshot_id=sid,
                query=question,
                max_results=10,
                section=RetrievalSection.QA,
            )
        )
        if not r1.evidences:
            return {
                "answer": "No relevant context found for this question.",
                "citations": [],
                "confidence": "low",
                "files_explored": [],
            }

        context_r1 = _format_chunks(r1.evidences)

        # Sampling: ask Claude what to explore deeper
        try:
            followup = await _sample(ctx, (
                f"Context from codebase retrieval:\n{context_r1}\n\n"
                f"Question: {question}\n\n"
                "What specific aspect needs deeper exploration? "
                "Reply with ONE focused search query (max 20 words)."
            ), max_tokens=150)
        except Exception as e:
            logger.warning("MCP Sampling failed for followup: %s", e)
            followup = question

        # Round 2: follow-up retrieval
        r2 = await svc.retrieve(
            RetrieveRequest(
                snapshot_id=sid,
                query=followup,
                max_results=10,
                section=RetrievalSection.QA,
            )
        )

        # Merge + dedup
        all_ev = _dedup(r1.evidences + r2.evidences, max_items=20)
        context_all = _format_chunks(all_ev)

        # Sampling: synthesize answer
        try:
            answer = await _sample(ctx, (
                f"Complete codebase context:\n{context_all}\n\n"
                f"Question: {question}\n\n"
                "Provide a detailed answer with [file:line] citations. Use Markdown."
            ), max_tokens=2000)
        except Exception as e:
            logger.warning("MCP Sampling failed for synthesis: %s", e)
            answer = "Unable to synthesize answer via MCP sampling."

        return {
            "answer": answer,
            "citations": [
                {"file": e.rel_path, "excerpt": e.excerpt[:200]}
                for e in all_ev[:10]
            ],
            "confidence": "high" if len(all_ev) > 5 else "medium",
            "files_explored": list({e.rel_path for e in all_ev}),
        }

    @mcp.tool()
    async def deep_research(
        project_path: str,
        question: str,
        max_hops: int = 5,
        ctx: Context = None,
    ) -> dict:
        """Multi-hop iterative research using graph traversal + MCP Sampling.
        Higher quality than ask_codebase for complex architectural questions."""
        sid = get_snapshot_id(project_path)

        from domain.retrieval.service import RetrievalService
        from domain.retrieval.types import RetrieveRequest, RetrievalSection
        from domain.structural_graph.service import StructuralGraphService

        retrieval_svc = RetrievalService()
        graph_svc = StructuralGraphService()
        start = time.time()

        reasoning_chain = []
        all_evidence = []
        explored_files: set[str] = set()
        current_query = question

        for hop in range(min(max_hops, 8)):
            # Retrieve
            result = await retrieval_svc.retrieve(
                RetrieveRequest(
                    snapshot_id=sid,
                    query=current_query,
                    max_results=8,
                    section=RetrievalSection.QA,
                )
            )
            if not result.evidences:
                break

            hop_files = {e.rel_path for e in result.evidences}
            all_evidence.extend(result.evidences)
            explored_files.update(hop_files)

            # Graph expansion from top hit
            top_file = result.evidences[0].rel_path
            try:
                neighbors = await graph_svc.neighbors(sid, top_file, hops=1, limit=20)
                for n in neighbors.nodes:
                    explored_files.add(n if isinstance(n, str) else str(n))
            except Exception:
                pass  # graph may not cover all files

            reasoning_chain.append({
                "hop": hop + 1,
                "query": current_query,
                "files_found": sorted(hop_files),
                "graph_expanded_from": top_file,
            })

            if hop >= max_hops - 1:
                break

            # Sampling: decide next query
            context_so_far = _format_chunks(_dedup(all_evidence, 15))
            try:
                next_query = await _sample(ctx, (
                    f"Research question: {question}\n\n"
                    f"Context gathered (hop {hop + 1}):\n{context_so_far}\n\n"
                    "What specific aspect should be explored next? "
                    "Reply ONE focused query (max 20 words). "
                    "If fully answered, reply: DONE"
                ), max_tokens=150)
            except Exception as e:
                logger.warning("MCP Sampling failed at hop %d: %s", hop + 1, e)
                break

            if next_query.strip().upper() == "DONE":
                break
            current_query = next_query

        # Final synthesis
        final_ev = _dedup(all_evidence, 25)
        final_ctx = _format_chunks(final_ev)
        try:
            summary = await _sample(ctx, (
                f"Research question: {question}\n\n"
                f"Complete gathered context:\n{final_ctx}\n\n"
                "Provide a comprehensive answer:\n"
                "1. Summary\n2. Key findings with [file:line] citations\n"
                "3. What remains unknown\nUse Markdown."
            ), max_tokens=3000)
        except Exception as e:
            logger.warning("MCP Sampling failed for final synthesis: %s", e)
            summary = "Unable to synthesize final answer via MCP sampling."

        return {
            "summary": summary,
            "reasoning_chain": reasoning_chain,
            "files_explored": sorted(explored_files),
            "confidence": "high" if len(final_ev) > 10 else "medium",
            "elapsed_ms": int((time.time() - start) * 1000),
        }
