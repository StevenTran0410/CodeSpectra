"""RRF Multi-Signal Fusion Query System (CS-252).

Combines BM25 lexical signal with graph-confidence signal via reciprocal rank fusion (RRF).
Produces debug bundles for analysis and comparison.
"""
from __future__ import annotations

from haystack.utils.misc import _reciprocal_rank_fusion
from haystack import Document

from infrastructure.db.database import get_db

from .types import SignalRankEntry, FusedRankEntry, RrfFusionBundle, RetrievalSection
from .bm25_scorer import BM25Scorer
from .two_stage_retrieval import (
    _load_graph_context,
    _stage1_score_rows,
    _query_terms,
    _CHUNK_FULL_COLS,
)


# CAP = 2.0: anti-gaming constant derived from "~2-3 high-confidence 0.85-0.95 edges".
# Prevents many low-confidence (0.15-0.3) ambiguous edges from outranking 2-3 genuinely
# high-confidence ones via edge-count gaming. See capped-SUM formula in docstring.
_CAPPED_SUM_CAP = 2.0


async def _load_confidence_weighted_edges(snapshot_id: str) -> dict[str, list[float]]:
    """Load confidence scores for symbol graph edges, keyed by destination file.

    Returns dict[dst_file] -> list of confidence_score values from edges where src_file
    is in the seed set (determined by caller). Confidence-lossy _load_graph_context()
    (from two_stage_retrieval.py:274-286) only stores binary pre-filter (dst_file set),
    never the confidence value. This function intentionally re-queries to preserve the
    full confidence signal.
    """
    db = get_db()
    edges_by_dst: dict[str, list[float]] = {}

    query = "SELECT DISTINCT src_symbol, dst_symbol, confidence_score FROM symbol_graph_edges WHERE snapshot_id=?"
    async with db.execute(query, (snapshot_id,)) as cur:
        rows = await cur.fetchall()

    for row in rows:
        src_symbol = row["src_symbol"] or ""
        dst_symbol = row["dst_symbol"] or ""
        confidence = float(row["confidence_score"] or 0.0)

        # Extract file names using :: split convention (matching two_stage_retrieval.py:283-284)
        src_file = src_symbol.split("::")[0] if "::" in src_symbol else src_symbol
        dst_file = dst_symbol.split("::")[0] if "::" in dst_symbol else dst_symbol

        if src_file and dst_file and src_file != dst_file and confidence > 0.0:
            edges_by_dst.setdefault(dst_file, []).append(confidence)

    return edges_by_dst


def build_graph_confidence_rank_list(
    all_rows: list[dict],
    ctx,
    stage1_files: set[str],
    confidence_edges: dict[str, list[float]],
    top_k: int = 100,
) -> list[SignalRankEntry]:
    """Build rank list from graph-confidence signal with capped-SUM scoring.

    Formula: rank_score(file) = min(sum(confidence_list_for_file), CAP) * centrality_boost
    - CAP = 2.0: anti-gaming constant. Derived from "~2-3 high-confidence 0.85-0.95 edges",
      prevents many low-confidence (0.15-0.3) ambiguous edges from outranking 2-3 genuinely
      high-confidence ones via edge-count gaming.
    - centrality_boost = 1.5 if file in ctx.central_files else 1.0

    Args:
        all_rows: Full chunk rows from retrieval_chunks
        ctx: _GraphContext with central_files, file_symbol_refs
        stage1_files: Set of files that ranked in BM25 stage 1
        confidence_edges: dict[dst_file] -> list[confidence_score] from edges in seed files
        top_k: Max entries to return

    Returns:
        List of SignalRankEntry sorted descending by rank_score, with rank=1..N
    """
    file_to_chunks: dict[str, list[dict]] = {}
    for row in all_rows:
        rel_path = row["rel_path"]
        file_to_chunks.setdefault(rel_path, []).append(row)

    # Compute graph-confidence scores per file
    scored_files: list[tuple[float, str, list[dict]]] = []
    for rel_path, chunks in file_to_chunks.items():
        confidence_list = confidence_edges.get(rel_path, [])
        confidence_sum = min(sum(confidence_list), _CAPPED_SUM_CAP)

        # Apply centrality boost
        centrality_boost = 1.5 if rel_path in ctx.central_files else 1.0
        rank_score = confidence_sum * centrality_boost

        if rank_score > 0.0 or rel_path in stage1_files:
            scored_files.append((rank_score, rel_path, chunks))

    # Sort descending and take top K files
    scored_files.sort(key=lambda x: (-x[0], x[1]))
    top_files = scored_files[:top_k]

    # Convert to SignalRankEntry: one per top file, using best chunk
    entries: list[SignalRankEntry] = []
    for rank, (score, rel_path, chunks) in enumerate(top_files, start=1):
        best_chunk = max(chunks, key=lambda c: int(c.get("chunk_index", 0)))
        excerpt = best_chunk.get("content", "")[:200] if best_chunk.get("content") else ""
        entries.append(SignalRankEntry(
            chunk_id=best_chunk["id"],
            rel_path=rel_path,
            rank=rank,
            raw_score=score,
            signal_name="graph_confidence",
            excerpt=excerpt,
        ))

    return entries


def build_bm25_rank_list(
    candidates,  # list[StageCandidate] output from _stage1_score_rows
    top_k: int = 100,
) -> list[SignalRankEntry]:
    """Convert BM25 stage 1 candidates to SignalRankEntry list.

    Args:
        candidates: Output from _stage1_score_rows (already sorted descending by BM25)
        top_k: Max entries to return

    Returns:
        List of SignalRankEntry with rank=1..N
    """
    entries: list[SignalRankEntry] = []
    for rank, candidate in enumerate(candidates[:top_k], start=1):
        entries.append(SignalRankEntry(
            chunk_id=candidate.chunk_id,
            rel_path=candidate.rel_path,
            rank=rank,
            raw_score=candidate.bm25_score,
            signal_name="bm25",
        ))
    return entries


def fuse_signal_lists(
    signal_lists: list[list[SignalRankEntry]],
    weights: list[float] | None = None,
) -> list[FusedRankEntry]:
    """Fuse multiple ranked signal lists via Reciprocal Rank Fusion (RRF).

    Uses the real haystack._reciprocal_rank_fusion function with k=61 (hardcoded inside).
    Internally constructs minimal Document objects (never escaped as public types);
    unwraps results into FusedRankEntry with per_signal_ranks dict.

    Args:
        signal_lists: List of ranked entry lists per signal (e.g., [bm25_entries, graph_entries])
        weights: Optional list of weights per signal; defaults to equal weight

    Returns:
        List of FusedRankEntry sorted by fused_score descending
    """
    if not signal_lists or all(not lst for lst in signal_lists):
        return []

    # Build Document lists in rank order for each signal
    # Document.id must match chunk_id; content must be non-empty (for validation)
    document_lists = []
    signal_names: list[str] = []
    for signal_entries in signal_lists:
        if not signal_entries:
            continue
        signal_name = signal_entries[0].signal_name
        signal_names.append(signal_name)
        doc_list = [
            Document(
                id=entry.chunk_id,
                content=entry.excerpt or entry.chunk_id,  # non-empty required
                score=None,
            )
            for entry in signal_entries
        ]
        document_lists.append(doc_list)

    if not document_lists:
        return []

    # Call the real haystack RRF function (k=61 hardcoded inside)
    fused_docs = _reciprocal_rank_fusion(document_lists, weights=weights)

    # Unwrap and build per_signal_ranks mapping
    fused_entries: list[FusedRankEntry] = []
    for fused_doc in fused_docs:
        chunk_id = fused_doc.id
        fused_score = float(fused_doc.score or 0.0)

        # Build per_signal_ranks: signal_name -> rank from that signal's list
        per_signal_ranks: dict[str, int] = {}
        for signal_name, signal_entries in zip(signal_names, signal_lists):
            for rank, entry in enumerate(signal_entries, start=1):
                if entry.chunk_id == chunk_id:
                    per_signal_ranks[signal_name] = rank
                    break

        # Find the excerpt from the original entries (use first signal that has this chunk)
        excerpt = ""
        for signal_entries in signal_lists:
            for entry in signal_entries:
                if entry.chunk_id == chunk_id:
                    excerpt = entry.excerpt
                    break
            if excerpt:
                break

        # Extract rel_path from any signal's entry
        rel_path = ""
        for signal_entries in signal_lists:
            for entry in signal_entries:
                if entry.chunk_id == chunk_id:
                    rel_path = entry.rel_path
                    break
            if rel_path:
                break

        fused_entries.append(FusedRankEntry(
            chunk_id=chunk_id,
            rel_path=rel_path,
            fused_score=fused_score,
            per_signal_ranks=per_signal_ranks,
            excerpt=excerpt,
        ))

    return fused_entries


async def retrieve_rrf_fusion(
    snapshot_id: str,
    query: str,
    section: RetrievalSection,
    budget: int | None = None,  # Unused in this debug path, kept for API consistency
    min_confidence: float | None = None,
) -> RrfFusionBundle:
    """Run RRF multi-signal fusion retrieval (CS-252 debug path).

    Loads identical all_rows/ctx/symbol_index as retrieve_two_stage, builds both BM25
    and graph-confidence signal rank lists, fuses via RRF, returns raw unbounded results.
    Does NOT route through _rank_and_budget()/_apply_diversity_filter() — this is a
    debug/comparison path, not a production synthesis input.

    Args:
        snapshot_id: Snapshot to retrieve from
        query: Search query
        section: Retrieval section
        budget: Token budget (unused in this debug path)
        min_confidence: Optional confidence threshold for graph edges

    Returns:
        RrfFusionBundle with bm25_signal, graph_signal, fused lists
    """
    db = get_db()
    terms = _query_terms(query)
    if not terms:
        raise ValueError("Query must contain searchable terms")

    # Load graph context and confidence-weighted edges (same as two_stage_retrieval)
    ctx = await _load_graph_context(snapshot_id, min_confidence)
    confidence_edges = await _load_confidence_weighted_edges(snapshot_id)

    # Load all chunks
    async with db.execute(
        f"SELECT {_CHUNK_FULL_COLS} FROM retrieval_chunks WHERE snapshot_id=?",
        (snapshot_id,),
    ) as cur:
        all_rows = await cur.fetchall()

    if not all_rows:
        raise ValueError("Retrieval index not built for this snapshot")

    # Load BM25 scorer
    async with db.execute(
        "SELECT avgdl, idf_json, k1, b FROM retrieval_bm25_stats WHERE snapshot_id=?",
        (snapshot_id,),
    ) as cur:
        bm25_row = await cur.fetchone()
    scorer = BM25Scorer.from_stats_row(bm25_row)

    # Stage 1: BM25 scoring
    stage1_candidates = _stage1_score_rows(all_rows, scorer, terms, top_k=100)
    stage1_files = {c.rel_path for c in stage1_candidates}

    bm25_signal = build_bm25_rank_list(stage1_candidates, top_k=100)

    # Build graph-confidence signal
    graph_signal = build_graph_confidence_rank_list(
        all_rows,
        ctx,
        stage1_files,
        confidence_edges,
        top_k=100,
    )

    # Fuse signals via RRF
    fused = fuse_signal_lists([bm25_signal, graph_signal])

    return RrfFusionBundle(
        snapshot_id=snapshot_id,
        query=query,
        section=section,
        bm25_signal=bm25_signal,
        graph_signal=graph_signal,
        fused=fused,
    )
