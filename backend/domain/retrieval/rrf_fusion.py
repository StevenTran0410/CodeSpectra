"""RRF Multi-Signal Fusion Query System (CS-252).

Combines BM25 lexical signal with graph-confidence signal via reciprocal rank fusion (RRF).
Produces debug bundles for analysis and comparison.
"""

from __future__ import annotations

from haystack import Document
from haystack.utils.misc import _reciprocal_rank_fusion

from infrastructure.db.database import get_db

from .bm25_scorer import BM25Scorer
from .cross_encoder_rerank import (
    detect_gpu,
    is_gpu_reranker_enabled,
    recommended_rerank_batch_size,
    rerank_fused_entries,
    release_gpu_cache,
)
from .two_stage_retrieval import (
    _CATEGORY_HINT_BONUS,
    _CHUNK_FULL_COLS,
    _MODULE_PROXIMITY_BONUS,
    _SECTION_CATEGORY_HINTS,
    _load_graph_context,
    _query_terms,
    _stage1_score_rows,
    load_symbol_index,
)
from .types import FusedRankEntry, RetrievalSection, RrfFusionBundle, SignalRankEntry

# CAP = 2.0: anti-gaming constant derived from "~2-3 high-confidence 0.85-0.95 edges".
# Prevents many low-confidence (0.15-0.3) ambiguous edges from outranking 2-3 genuinely
# high-confidence ones via edge-count gaming. See capped-SUM formula in docstring.
_CAPPED_SUM_CAP = 2.0



def _symbol_overlap_fallback(
    chunks: list[dict], symbol_index: dict[str, list[tuple[str, int, int]]] | None
) -> dict | None:
    """Prefer chunk whose line range overlaps a known symbol definition.

    Args:
        chunks: List of chunk dicts (must have rel_path, start_line, end_line keys)
        symbol_index: dict[term_lower, list[(rel_path, start_line, end_line)]]
            from load_symbol_index

    Returns:
        First chunk whose [start_line, end_line] overlaps a symbol for that file, or None
    """
    if not symbol_index:
        return None

    # Flatten all symbol ranges by file
    symbols_by_file: dict[str, list[tuple[int, int]]] = {}
    for sym_ranges in symbol_index.values():
        for rel_path, sym_start, sym_end in sym_ranges:
            symbols_by_file.setdefault(rel_path, []).append((sym_start, sym_end))

    # Check each chunk for overlap
    for chunk in chunks:
        chunk_rel_path: str | None = chunk.get("rel_path")
        if not isinstance(chunk_rel_path, str):
            continue
        chunk_start = chunk.get("start_line", 0)
        chunk_end = chunk.get("end_line", 0)

        if chunk_start == 0 and chunk_end == 0:
            continue

        for sym_start, sym_end in symbols_by_file.get(chunk_rel_path, []):
            # Check if symbol range overlaps chunk range
            if sym_start >= chunk_start and sym_end <= chunk_end:
                return chunk

    return None


def _group_chunks_by_file(all_rows: list[dict]) -> dict[str, list[dict]]:
    """Group chunk rows by rel_path.

    Shared by the 3 signal builders that need full per-file chunk lists
    (graph_confidence, module_proximity, category_hint). build_bm25_rank_list
    groups differently (by max BM25 score per file, not full chunk lists),
    so it doesn't use this helper.
    """
    file_to_chunks: dict[str, list[dict]] = {}
    for row in all_rows:
        rel_path = row["rel_path"]
        file_to_chunks.setdefault(rel_path, []).append(row)
    return file_to_chunks


def _best_chunk_per_file(
    all_rows: list[dict],
    stage1_candidates: list,
    symbol_index: dict[str, list[tuple[str, int, int]]] | None,
) -> dict[str, dict]:
    """Compute best chunk per file using BM25 scores when available.

    Fallback chain:
    1. Highest BM25-scored chunk for that file (if any passed stage1)
    2. Chunk overlapping a known symbol definition (if symbol_index available)
    3. First chunk (chunk_index == 0 semantically, or list order)

    Args:
        all_rows: Full chunk rows
        stage1_candidates: Ranked candidates from BM25 stage 1
        symbol_index: Symbol definition ranges, or None

    Returns:
        dict[rel_path -> best_row_dict]
    """
    # Build BM25 score lookup: chunk_id -> bm25_score
    chunk_bm25: dict[str, float] = {c.chunk_id: c.bm25_score for c in stage1_candidates}

    # Group all_rows by rel_path
    file_to_chunks: dict[str, list[dict]] = {}
    for row in all_rows:
        rel_path = row.get("rel_path")
        if rel_path:
            file_to_chunks.setdefault(rel_path, []).append(row)

    # Pick best chunk per file
    best_by_file: dict[str, dict] = {}
    for rel_path, chunks in file_to_chunks.items():
        # Try BM25 first
        scored = [(chunk_bm25[c["id"]], c) for c in chunks if c["id"] in chunk_bm25]
        if scored:
            best = max(scored, key=lambda t: t[0])[1]
        else:
            # No BM25: try symbol overlap, else first chunk
            best = _symbol_overlap_fallback(chunks, symbol_index) or chunks[0]

        best_by_file[rel_path] = best

    return best_by_file


async def _load_confidence_weighted_edges(
    snapshot_id: str, seed_files: set[str]
) -> dict[str, list[float]]:
    """Load confidence scores for symbol graph edges from seed files, keyed by destination file.

    Returns dict[dst_file] -> list of confidence_score values from edges where src_file
    is in seed_files. Confidence-lossy _load_graph_context() (from
    two_stage_retrieval.py:274-286) only stores binary pre-filter (dst_file set), never the
    confidence value. This function intentionally re-queries to preserve the full confidence
    signal.

    The seed-file filter is pushed into the SQL WHERE clause (substr-based prefix match,
    same collision-safe convention as CS-249's copy_unchanged_symbol_edges -- "foo.py" never
    matches "foo2.py::..."), chunked at 200 paths per round-trip, instead of fetching every
    edge in the snapshot and filtering in Python (CS-255).
    """
    if not seed_files:
        return {}

    db = get_db()
    edges_by_dst: dict[str, list[float]] = {}
    seed_list = sorted(seed_files)
    chunk_size = 200  # 2 bind params per path + 1 fixed param, well under SQLite's variable limit

    for i in range(0, len(seed_list), chunk_size):
        chunk = seed_list[i : i + chunk_size]
        conditions = []
        params: list[str] = [snapshot_id]
        for path in chunk:
            conditions.append("substr(src_symbol, 1, length(?) + 2) = ? || '::'")
            params.extend([path, path])
        condition_str = " OR ".join(f"({c})" for c in conditions)

        query = (
            "SELECT DISTINCT src_symbol, dst_symbol, confidence_score "
            f"FROM symbol_graph_edges WHERE snapshot_id=? AND ({condition_str})"
        )
        async with db.execute(query, tuple(params)) as cur:
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
    best_chunk_by_file: dict[str, dict],
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
        best_chunk_by_file: dict[rel_path] -> best_row_dict (precomputed from _best_chunk_per_file)
        top_k: Max entries to return

    Returns:
        List of SignalRankEntry sorted descending by rank_score, with rank=1..N
    """
    file_to_chunks = _group_chunks_by_file(all_rows)

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

    # Convert to SignalRankEntry: one per top file, using best chunk from precomputed map
    entries: list[SignalRankEntry] = []
    for rank, (score, rel_path, chunks) in enumerate(top_files, start=1):
        best_chunk = best_chunk_by_file.get(rel_path) or max(
            chunks, key=lambda c: int(c.get("chunk_index", 0))
        )
        excerpt = best_chunk.get("content", "")[:200] if best_chunk.get("content") else ""
        entries.append(
            SignalRankEntry(
                chunk_id=best_chunk["id"],
                rel_path=rel_path,
                rank=rank,
                raw_score=score,
                signal_name="graph_confidence",
                excerpt=excerpt,
                token_estimate=int(best_chunk.get("token_estimate") or 0),
            )
        )

    return entries


def build_bm25_rank_list(
    candidates,  # list[StageCandidate] output from _stage1_score_rows
    best_chunk_by_file: dict[str, dict],
    top_k: int = 100,
) -> list[SignalRankEntry]:
    """Convert BM25 stage 1 candidates to SignalRankEntry list, collapsed to one per file.

    Args:
        candidates: Output from _stage1_score_rows (already sorted descending by BM25)
        best_chunk_by_file: dict[rel_path] -> best_row_dict (precomputed from _best_chunk_per_file)
        top_k: Max entries to return

    Returns:
        List of SignalRankEntry with rank=1..N, collapsed to one per file using best_chunk_by_file
    """
    # Group candidates by rel_path and keep highest BM25 score
    file_to_score: dict[str, float] = {}
    for candidate in candidates:
        rel_path = candidate.rel_path
        if rel_path not in file_to_score or candidate.bm25_score > file_to_score[rel_path]:
            file_to_score[rel_path] = candidate.bm25_score

    # Sort files by score descending
    sorted_files = sorted(file_to_score.items(), key=lambda x: -x[1])

    # Build entries using best_chunk_by_file for chunk_id
    entries: list[SignalRankEntry] = []
    for rank, (rel_path, bm25_score) in enumerate(sorted_files[:top_k], start=1):
        best_chunk = best_chunk_by_file.get(rel_path)
        if best_chunk:
            excerpt = best_chunk.get("content", "")[:200] if best_chunk.get("content") else ""
            entries.append(
                SignalRankEntry(
                    chunk_id=best_chunk["id"],
                    rel_path=rel_path,
                    rank=rank,
                    raw_score=bm25_score,
                    signal_name="bm25",
                    excerpt=excerpt,
                    token_estimate=int(best_chunk.get("token_estimate") or 0),
                )
            )
    return entries


def build_module_proximity_rank_list(
    all_rows: list[dict],
    ctx,
    seed_community_ids: set[int],
    best_chunk_by_file: dict[str, dict],
    top_k: int = 100,
) -> list[SignalRankEntry]:
    """Build rank list from module-proximity signal (community proximity to seed files).

    Ported from two_stage_retrieval.py:490 mod_bonus logic. Applies _MODULE_PROXIMITY_BONUS (1.3)
    to files in seed communities.

    Args:
        all_rows: Full chunk rows from retrieval_chunks
        ctx: _GraphContext with file_community mapping
        seed_community_ids: Set of community IDs found in top-20 BM25 candidates
        best_chunk_by_file: dict[rel_path] -> best_row_dict (precomputed from _best_chunk_per_file)
        top_k: Max entries to return

    Returns:
        List of SignalRankEntry with signal_name='module_proximity', sorted descending
    """
    file_to_chunks = _group_chunks_by_file(all_rows)

    # Score files: _MODULE_PROXIMITY_BONUS if community in seed set, else skip
    scored_files: list[tuple[float, str]] = []
    for rel_path in file_to_chunks.keys():
        cid = ctx.file_community.get(rel_path)
        if cid is not None and cid in seed_community_ids:
            scored_files.append((_MODULE_PROXIMITY_BONUS, rel_path))

    # Sort descending, take top K
    scored_files.sort(key=lambda x: (-x[0], x[1]))
    top_files = scored_files[:top_k]

    # Convert to SignalRankEntry
    entries: list[SignalRankEntry] = []
    for rank, (score, rel_path) in enumerate(top_files, start=1):
        best_chunk = best_chunk_by_file.get(rel_path)
        if best_chunk:
            excerpt = best_chunk.get("content", "")[:200] if best_chunk.get("content") else ""
            entries.append(
                SignalRankEntry(
                    chunk_id=best_chunk["id"],
                    rel_path=rel_path,
                    rank=rank,
                    raw_score=score,
                    signal_name="module_proximity",
                    excerpt=excerpt,
                    token_estimate=int(best_chunk.get("token_estimate") or 0),
                )
            )

    return entries


def build_category_hint_rank_list(
    all_rows: list[dict],
    section: RetrievalSection,
    best_chunk_by_file: dict[str, dict],
    top_k: int = 100,
) -> list[SignalRankEntry]:
    """Build rank list from category-hint signal (chunk category matches section hints).

    Ported from two_stage_retrieval.py:485-486 category-hint bonus logic. Applies
    _CATEGORY_HINT_BONUS (1.4) to files whose chunks match the section's hint category set.

    Args:
        all_rows: Full chunk rows from retrieval_chunks
        section: RetrievalSection enum to look up hint categories
        best_chunk_by_file: dict[rel_path] -> best_row_dict (precomputed from _best_chunk_per_file)
        top_k: Max entries to return

    Returns:
        List of SignalRankEntry with signal_name='category_hint', sorted descending
    """
    category_hints = _SECTION_CATEGORY_HINTS.get(section, set())
    if not category_hints:
        return []

    file_to_chunks = _group_chunks_by_file(all_rows)

    # Score files: _CATEGORY_HINT_BONUS if any chunk's category in hints, else skip
    scored_files: list[tuple[float, str]] = []
    for rel_path, chunks in file_to_chunks.items():
        if any(c.get("category") in category_hints for c in chunks):
            scored_files.append((_CATEGORY_HINT_BONUS, rel_path))

    # Sort descending, take top K
    scored_files.sort(key=lambda x: (-x[0], x[1]))
    top_files = scored_files[:top_k]

    # Convert to SignalRankEntry
    entries: list[SignalRankEntry] = []
    for rank, (score, rel_path) in enumerate(top_files, start=1):
        best_chunk = best_chunk_by_file.get(rel_path)
        if best_chunk:
            excerpt = best_chunk.get("content", "")[:200] if best_chunk.get("content") else ""
            entries.append(
                SignalRankEntry(
                    chunk_id=best_chunk["id"],
                    rel_path=rel_path,
                    rank=rank,
                    raw_score=score,
                    signal_name="category_hint",
                    excerpt=excerpt,
                    token_estimate=int(best_chunk.get("token_estimate") or 0),
                )
            )

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

    # Pre-build chunk_id -> {ranks, excerpt, rel_path} in one O(total entries) pass,
    # instead of re-scanning every signal's entries per fused doc (was O(signals * M^2),
    # CS-255). Preserves exact original semantics: per signal, the FIRST occurrence of
    # a chunk_id (lowest/best rank) wins; for excerpt/rel_path, the first signal list (in
    # order) yielding a truthy value wins.
    chunk_meta: dict[str, dict] = {}
    for signal_name, signal_entries in zip(signal_names, signal_lists):
        for rank, entry in enumerate(signal_entries, start=1):
            meta = chunk_meta.setdefault(
                entry.chunk_id, {"ranks": {}, "excerpt": "", "rel_path": "", "token_estimate": 0}
            )
            if signal_name not in meta["ranks"]:
                meta["ranks"][signal_name] = rank
            if not meta["excerpt"] and entry.excerpt:
                meta["excerpt"] = entry.excerpt
            if not meta["rel_path"] and entry.rel_path:
                meta["rel_path"] = entry.rel_path
            if not meta["token_estimate"] and entry.token_estimate:
                meta["token_estimate"] = entry.token_estimate

    fused_entries: list[FusedRankEntry] = []
    for fused_doc in fused_docs:
        chunk_id = fused_doc.id
        fused_score = float(fused_doc.score or 0.0)
        meta = chunk_meta.get(
            chunk_id, {"ranks": {}, "excerpt": "", "rel_path": "", "token_estimate": 0}
        )

        fused_entries.append(
            FusedRankEntry(
                chunk_id=chunk_id,
                rel_path=meta["rel_path"],
                fused_score=fused_score,
                per_signal_ranks=meta["ranks"],
                excerpt=meta["excerpt"],
                token_estimate=meta["token_estimate"],
            )
        )

    # _reciprocal_rank_fusion() returns documents_map.values() in insertion
    # order (≈ first signal list's order), NOT sorted by the fused score it
    # just computed. Without this sort, "fused" results are just the first
    # signal's order with extra metadata attached — no actual re-ranking.
    fused_entries.sort(key=lambda e: e.fused_score, reverse=True)
    return fused_entries


# Rerank coverage target: at least 100 candidates when available, but never more
# than 3/4 of the total fused pool (rerank is meant to refine a top slice, not
# replace RRF fusion's own judgment over the long tail).
_RERANK_COVERAGE_MIN = 100
_RERANK_COVERAGE_MAX_FRACTION = 0.75


def _rerank_coverage_target(total_fused: int) -> int:
    return min(total_fused, max(_RERANK_COVERAGE_MIN, round(total_fused * _RERANK_COVERAGE_MAX_FRACTION)))


def _rerank_in_batches(
    query: str,
    fused: list[FusedRankEntry],
    chunk_content_by_id: dict[str, str],
    batch_size: int,
) -> tuple[list, str]:
    """Rerank `fused` in sequential batches of `batch_size` (the model joins all
    passages in one call into a single shared sequence, so a single oversized call
    risks CUDA OOM — see cross_encoder_rerank.recommended_rerank_batch_size).

    Scores from different batches are merge-sorted together at the end. This is an
    approximation, not an exact equivalent of reranking the whole pool in one call —
    an empirical check found cross-batch score deltas for the same passage were small
    relative to the relevant/irrelevant score gap, but the model's listwise design
    does let documents in the same call influence each other's embeddings.
    """
    target = _rerank_coverage_target(len(fused))
    pool = fused[:target]

    all_reranked: list = []
    last_status = "ok"
    for start in range(0, len(pool), batch_size):
        batch = pool[start : start + batch_size]
        reranked_batch, status = rerank_fused_entries(
            query, batch, chunk_content_by_id, rank_offset=start
        )
        last_status = status
        if status != "ok":
            break
        all_reranked.extend(reranked_batch)
        release_gpu_cache()  # return this batch's freed VRAM before the next one starts

    if not all_reranked:
        return [], last_status

    all_reranked.sort(key=lambda e: e.rerank_score, reverse=True)
    return all_reranked, "ok"


async def retrieve_rrf_fusion(
    snapshot_id: str,
    query: str,
    section: RetrievalSection,
    budget: int | None = None,  # Unused in this debug path, kept for API consistency
    min_confidence: float | None = None,
) -> RrfFusionBundle:
    """Run RRF multi-signal fusion retrieval (CS-252/CS-253 debug path).

    Loads identical all_rows/ctx/symbol_index as retrieve_two_stage, builds all 4 signal
    rank lists (BM25, graph-confidence, module-proximity, category-hint), fuses via RRF,
    returns raw unbounded results. Does NOT route through
    _rank_and_budget()/_apply_diversity_filter() — this is a debug/comparison path, not a
    production synthesis input.

    Args:
        snapshot_id: Snapshot to retrieve from
        query: Search query
        section: Retrieval section
        budget: Token budget (unused in this debug path)
        min_confidence: Optional confidence threshold for graph edges

    Returns:
        RrfFusionBundle with bm25_signal, graph_signal, module_signal, category_signal, fused lists
    """
    db = get_db()
    terms = _query_terms(query)
    if not terms:
        raise ValueError("Query must contain searchable terms")

    # Load graph context and symbol index (same as two_stage_retrieval)
    ctx = await _load_graph_context(snapshot_id, min_confidence)
    symbol_index = await load_symbol_index(snapshot_id)

    # Load all chunks
    async with db.execute(
        f"SELECT {_CHUNK_FULL_COLS} FROM retrieval_chunks WHERE snapshot_id=?",
        (snapshot_id,),
    ) as cur:
        all_rows = list(await cur.fetchall())

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

    # Confidence-weighted edges, filtered to seed files in SQL (CS-255) instead of
    # fetching every edge in the snapshot and filtering in Python.
    confidence_edges = await _load_confidence_weighted_edges(snapshot_id, stage1_files)

    # Compute seed_community_ids from top-20 stage1 candidates
    # (mirrors two_stage_retrieval.py:647-651)
    seed_community_ids: set[int] = set()
    for c in stage1_candidates[:20]:
        cid = ctx.file_community.get(c.rel_path)
        if cid is not None:
            seed_community_ids.add(cid)

    # Convert all_rows to dicts once (reuse everywhere)
    all_rows_dicts = [dict(r) for r in all_rows]

    # Build chunk_content_by_id for reranking (CS-254)
    chunk_content_by_id = {r["id"]: r.get("content", "") for r in all_rows_dicts}

    # Precompute best_chunk_per_file once, thread into all 4 builders
    best_chunk_by_file = _best_chunk_per_file(all_rows_dicts, stage1_candidates, symbol_index)

    # Build all 4 signal lists
    bm25_signal = build_bm25_rank_list(stage1_candidates, best_chunk_by_file, top_k=100)

    graph_signal = build_graph_confidence_rank_list(
        all_rows_dicts,
        ctx,
        stage1_files,
        confidence_edges,
        best_chunk_by_file,
        top_k=100,
    )

    module_signal = build_module_proximity_rank_list(
        all_rows_dicts,
        ctx,
        seed_community_ids,
        best_chunk_by_file,
        top_k=100,
    )

    category_signal = build_category_hint_rank_list(
        all_rows_dicts,
        section,
        best_chunk_by_file,
        top_k=100,
    )

    # Fuse all 4 signals via RRF
    fused = fuse_signal_lists([bm25_signal, graph_signal, module_signal, category_signal])

    # Call cross-encoder reranking (CS-254) — gated by the global GPU Reranker
    # toggle (Settings). Off by default; user must explicitly opt in. Runs in
    # VRAM-sized batches (see recommended_rerank_batch_size) since the model joins
    # ALL passages in one call into a single shared sequence, not pairwise.
    if await is_gpu_reranker_enabled():
        _, vram_gb = detect_gpu()
        batch_size = recommended_rerank_batch_size(vram_gb)
        reranked, reranker_status = _rerank_in_batches(
            query, fused, chunk_content_by_id, batch_size
        )
    else:
        reranked, reranker_status = [], "disabled"

    return RrfFusionBundle(
        snapshot_id=snapshot_id,
        query=query,
        section=section,
        bm25_signal=bm25_signal,
        graph_signal=graph_signal,
        module_signal=module_signal,
        category_signal=category_signal,
        fused=fused,
        reranked=reranked,
        reranker_status=reranker_status,
    )
