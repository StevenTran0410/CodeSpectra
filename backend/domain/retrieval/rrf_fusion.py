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
from .quality import compute_retrieval_quality
from .types import (
    FusedRankEntry,
    RankedChunk,
    RetrievalBundle,
    RetrievalEvidence,
    RetrievalMode,
    RetrievalSection,
    RrfFusionBundle,
    SignalRankEntry,
)
from domain.qa.graph_queries import get_callees_of, get_callers_of

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


def _resolve_symbol_for_chunk(
    chunk_id: str,
    all_rows_dicts: list[dict],
    symbol_index: dict[str, list[tuple[str, int, int]]] | None,
) -> str | None:
    """Resolve a chunk to its enclosing symbol name via overlap matching.

    Given a chunk (identified by chunk_id), find its [start_line, end_line] in all_rows_dicts,
    then look up which symbol's definition range (from symbol_index) overlaps that chunk.

    Args:
        chunk_id: The chunk's ID to resolve
        all_rows_dicts: All chunk rows as dicts (must have id, rel_path, start_line, end_line)
        symbol_index: dict[name_lower, list[(rel_path, start_line, end_line)]] from load_symbol_index

    Returns:
        The symbol name (as it appears in the call graph, e.g., "ClassName.method_name"),
        or None if no chunk matches or no symbol overlaps the chunk.
    """
    if not symbol_index:
        return None

    # Find the chunk's line range
    chunk_row = None
    for row in all_rows_dicts:
        if row.get("id") == chunk_id:
            chunk_row = row
            break

    if not chunk_row:
        return None

    chunk_rel_path = chunk_row.get("rel_path")
    chunk_start = chunk_row.get("start_line", 0)
    chunk_end = chunk_row.get("end_line", 0)

    if chunk_start == 0 and chunk_end == 0:
        return None

    # Find the first symbol whose definition range overlaps this chunk
    # (reusing the exact overlap predicate from _symbol_overlap_fallback: sym_start >= chunk_start and sym_end <= chunk_end)
    for symbol_name, ranges in symbol_index.items():
        for rel_path, sym_start, sym_end in ranges:
            if rel_path == chunk_rel_path and sym_start >= chunk_start and sym_end <= chunk_end:
                return symbol_name
    return None


def _resolve_chunk_for_symbol(
    file_path: str,
    symbol_name: str,
    all_rows_dicts: list[dict],
    symbol_index: dict[str, list[tuple[str, int, int]]] | None,
) -> dict | None:
    """Resolve a symbol to the chunk containing its definition.

    Given a file and symbol name, look up the symbol's line range in symbol_index,
    then find the chunk in that file whose [start_line, end_line] overlaps the symbol.

    Args:
        file_path: The file where the symbol is defined
        symbol_name: The symbol name (lower-cased or exact as stored in symbol_index)
        all_rows_dicts: All chunk rows as dicts
        symbol_index: dict[name_lower, list[(rel_path, start_line, end_line)]]

    Returns:
        The chunk dict that contains this symbol's definition, or None if no match.
    """
    if not symbol_index:
        return None

    # Find the symbol's definition line range
    symbol_ranges = symbol_index.get(symbol_name.lower(), [])
    sym_start, sym_end = None, None
    for rel_path, start, end in symbol_ranges:
        if rel_path == file_path:
            sym_start, sym_end = start, end
            break

    if sym_start is None:
        return None

    # Find chunks in this file that overlap the symbol's range
    chunks_for_file = [r for r in all_rows_dicts if r.get("rel_path") == file_path]
    for chunk in chunks_for_file:
        chunk_start = chunk.get("start_line", 0)
        chunk_end = chunk.get("end_line", 0)
        if chunk_start == 0 and chunk_end == 0:
            continue
        # Use the same overlap predicate: sym_start >= chunk_start and sym_end <= chunk_end
        if sym_start >= chunk_start and sym_end <= chunk_end:
            return chunk

    return None


async def _expand_function_level_1hop(
    snapshot_id: str,
    fused: list[FusedRankEntry],
    symbol_index: dict[str, list[tuple[str, int, int]]] | None,
    all_rows_dicts: list[dict],
    cap: int,
) -> list[FusedRankEntry]:
    """Collect function-level 1-hop expansion candidates (callees and callers of top-ranked functions).

    Expands the candidate pool by finding functions that call or are called by the top N fused entries,
    where N = _rerank_coverage_target(len(fused)). For each seed, fetches callees and callers via
    symbol graph edges, then resolves each to a specific chunk (not whole file). Synthetic
    FusedRankEntry placeholders are created for expansion-only candidates (never seen in original `fused`).

    Iteration is in rank order (best-first), so if the cap is hit, expansion priority deterministically
    favors the query's currently-best candidates.

    Args:
        snapshot_id: Snapshot to query
        fused: The original fused rank list
        symbol_index: Symbol definition index from load_symbol_index
        all_rows_dicts: All chunk rows as dicts
        cap: Maximum number of NEW (expansion-only) chunks to add

    Returns:
        List of synthetic FusedRankEntry for expansion-only chunks (NOT including the original fused entries).
    """
    if not fused or not symbol_index:
        return []

    # Compute N: the set of seed candidates to expand around
    N = _rerank_coverage_target(len(fused))
    seed_candidates = fused[:N]

    expansion_entries: list[FusedRankEntry] = []
    fused_chunk_ids = {e.chunk_id for e in fused}
    expansion_chunk_ids = set()

    # Iterate in rank order (best-ranked seed first)
    for seed_entry in seed_candidates:
        if len(expansion_chunk_ids) >= cap:
            break

        # Resolve seed chunk to its enclosing symbol
        seed_symbol = _resolve_symbol_for_chunk(seed_entry.chunk_id, all_rows_dicts, symbol_index)
        if not seed_symbol:
            continue

        # Get callees and callers (high_confidence_only=True, matching existing defaults)
        callees = await get_callees_of(
            snapshot_id, seed_entry.rel_path, symbol=seed_symbol, high_confidence_only=True
        )
        callers = await get_callers_of(
            snapshot_id, seed_entry.rel_path, symbol=seed_symbol, high_confidence_only=True
        )

        # Process callees: resolve dst_symbol to a chunk
        for hop in callees:
            if len(expansion_chunk_ids) >= cap:
                break
            dst_file = hop.dst_symbol.split("::")[0] if "::" in hop.dst_symbol else hop.dst_symbol
            dst_symbol = hop.dst_symbol.split("::", 1)[1] if "::" in hop.dst_symbol else ""

            if not dst_symbol:
                continue

            expansion_chunk = _resolve_chunk_for_symbol(dst_file, dst_symbol, all_rows_dicts, symbol_index)
            if expansion_chunk and expansion_chunk["id"] not in fused_chunk_ids and expansion_chunk["id"] not in expansion_chunk_ids:
                expansion_chunk_ids.add(expansion_chunk["id"])
                # Create synthetic FusedRankEntry with fused_score=0.0, per_signal_ranks={}
                expansion_entries.append(
                    FusedRankEntry(
                        chunk_id=expansion_chunk["id"],
                        rel_path=expansion_chunk.get("rel_path", ""),
                        fused_score=0.0,
                        per_signal_ranks={},
                        excerpt=expansion_chunk.get("content", "")[:200] if expansion_chunk.get("content") else "",
                        token_estimate=int(expansion_chunk.get("token_estimate") or 0),
                    )
                )

        # Process callers: resolve src_symbol to a chunk
        for hop in callers:
            if len(expansion_chunk_ids) >= cap:
                break
            src_file = hop.src_symbol.split("::")[0] if "::" in hop.src_symbol else hop.src_symbol
            src_symbol = hop.src_symbol.split("::", 1)[1] if "::" in hop.src_symbol else ""

            if not src_symbol:
                continue

            expansion_chunk = _resolve_chunk_for_symbol(src_file, src_symbol, all_rows_dicts, symbol_index)
            if expansion_chunk and expansion_chunk["id"] not in fused_chunk_ids and expansion_chunk["id"] not in expansion_chunk_ids:
                expansion_chunk_ids.add(expansion_chunk["id"])
                expansion_entries.append(
                    FusedRankEntry(
                        chunk_id=expansion_chunk["id"],
                        rel_path=expansion_chunk.get("rel_path", ""),
                        fused_score=0.0,
                        per_signal_ranks={},
                        excerpt=expansion_chunk.get("content", "")[:200] if expansion_chunk.get("content") else "",
                        token_estimate=int(expansion_chunk.get("token_estimate") or 0),
                    )
                )

    return expansion_entries


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


def _rerank_pool_in_batches(
    query: str,
    pool: list[FusedRankEntry],
    chunk_content_by_id: dict[str, str],
    batch_size: int,
) -> tuple[list, str]:
    """Rerank a pre-determined pool of candidates in sequential batches.

    This is a sibling to _rerank_in_batches with a critical difference: it does NOT
    re-compute _rerank_coverage_target(len(pool)). Instead, it batches over the full
    `pool` argument exactly as received. This avoids the "double-coverage-target footgun"
    where re-slicing an already-capped expansion pool could silently truncate away
    expansion-only candidates.

    The caller is responsible for:
    - Computing target = _rerank_coverage_target(len(fused)) once
    - Building the pool = fused[:target] + expansion_candidates (already capped)
    - Passing that explicit pool to this function

    Args:
        query: Search query
        pool: Pre-determined pool to rerank (no internal re-derivation or slicing)
        chunk_content_by_id: dict[chunk_id -> full_content_text]
        batch_size: Size of each batch (from recommended_rerank_batch_size)

    Returns:
        Tuple of (reranked_entries, status_code), sorted by rerank_score descending
    """
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
        release_gpu_cache()

    if not all_reranked:
        return [], last_status

    all_reranked.sort(key=lambda e: e.rerank_score, reverse=True)
    return all_reranked, "ok"


def _fuse_final_ranking(
    fused: list[FusedRankEntry],
    reranked: list,
    weights: tuple[float, float] = (0.6, 0.4),
) -> list[FusedRankEntry]:
    """Fuse fused_rank and cross_encoder_rank signals via RRF.

    Builds two rank lists:
    - fused_rank: from the ORIGINAL `fused` list (its existing order, with list position = rank)
    - cross_encoder_rank: from the reranked list's POST-SORT position (enumerate(reranked, start=1))

    Expansion-only candidates (in reranked but NOT in fused) only contribute via the
    cross_encoder_rank signal; they are excluded from the fused_rank list, and haystack's
    RRF naturally handles the partial-list-membership (no synthetic fallback rank needed).

    Args:
        fused: The original fused list (used only for rank signal, not mutated)
        reranked: The reranked list from _rerank_pool_in_batches (already sorted by rerank_score)
        weights: Tuple of (fused_weight, cross_encoder_weight), default (0.6, 0.4)

    Returns:
        List of FusedRankEntry sorted by the combined fused_score descending
    """
    if not fused or not reranked:
        return []

    # Build two Document lists (mimicking fuse_signal_lists' pattern at lines 444-467)
    # List 1: fused-rank signal (built from the original fused list's order)
    fused_docs = [
        Document(
            id=entry.chunk_id,
            content=entry.excerpt or entry.chunk_id,
            score=None,
        )
        for entry in fused
    ]

    # List 2: cross_encoder_rank signal (built from reranked's post-sort position)
    # HARD REQUIREMENT: use enumerate(reranked, start=1), NOT RerankedEntry.fused_rank
    reranked_docs = [
        Document(
            id=entry.chunk_id,
            content=entry.excerpt or entry.chunk_id,
            score=None,
        )
        for entry in reranked
    ]

    # Call _reciprocal_rank_fusion with the two lists and weights
    document_lists = [fused_docs, reranked_docs]
    fused_result_docs = _reciprocal_rank_fusion(document_lists, weights=list(weights))

    # Build chunk metadata from both lists (same O(total entries) pass as fuse_signal_lists)
    chunk_meta: dict[str, dict] = {}
    for rank, entry in enumerate(fused, start=1):
        meta = chunk_meta.setdefault(
            entry.chunk_id, {"ranks": {}, "excerpt": "", "rel_path": "", "token_estimate": 0}
        )
        if "fused" not in meta["ranks"]:
            meta["ranks"]["fused"] = rank
        if not meta["excerpt"] and entry.excerpt:
            meta["excerpt"] = entry.excerpt
        if not meta["rel_path"] and entry.rel_path:
            meta["rel_path"] = entry.rel_path
        if not meta["token_estimate"] and entry.token_estimate:
            meta["token_estimate"] = entry.token_estimate

    for rank, entry in enumerate(reranked, start=1):
        meta = chunk_meta.setdefault(
            entry.chunk_id, {"ranks": {}, "excerpt": "", "rel_path": "", "token_estimate": 0}
        )
        if "cross_encoder" not in meta["ranks"]:
            meta["ranks"]["cross_encoder"] = rank
        if not meta["excerpt"] and entry.excerpt:
            meta["excerpt"] = entry.excerpt
        if not meta["rel_path"] and entry.rel_path:
            meta["rel_path"] = entry.rel_path
        if not meta["token_estimate"] and entry.token_estimate:
            meta["token_estimate"] = entry.token_estimate

    # Unwrap Document results into FusedRankEntry (same pattern as fuse_signal_lists)
    final_entries: list[FusedRankEntry] = []
    for fused_doc in fused_result_docs:
        chunk_id = fused_doc.id
        fused_score = float(fused_doc.score or 0.0)
        meta = chunk_meta.get(
            chunk_id, {"ranks": {}, "excerpt": "", "rel_path": "", "token_estimate": 0}
        )

        final_entries.append(
            FusedRankEntry(
                chunk_id=chunk_id,
                rel_path=meta["rel_path"],
                fused_score=fused_score,
                per_signal_ranks=meta["ranks"],
                excerpt=meta["excerpt"],
                token_estimate=meta["token_estimate"],
            )
        )

    # Sort descending by fused_score (same fix as rrf_fusion.py:512)
    final_entries.sort(key=lambda e: e.fused_score, reverse=True)
    return final_entries


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

    # Call cross-encoder reranking (CS-254) with 1-hop expansion (CS-256) — gated by the global GPU Reranker
    # toggle (Settings). Off by default; user must explicitly opt in. Runs in
    # VRAM-sized batches (see recommended_rerank_batch_size) since the model joins
    # ALL passages in one call into a single shared sequence, not pairwise.
    final = []
    if await is_gpu_reranker_enabled():
        _, vram_gb = detect_gpu()
        batch_size = recommended_rerank_batch_size(vram_gb)

        # CS-256: Function-level 1-hop expansion
        target = _rerank_coverage_target(len(fused))
        seed_pool = fused[:target]
        expansion_cap = recommended_rerank_batch_size(vram_gb)  # Reuse VRAM-scaled ceiling for OOM-safety
        expansion_candidates = await _expand_function_level_1hop(
            snapshot_id, seed_pool, symbol_index, all_rows_dicts, expansion_cap
        )

        # Build expanded pool = fused[:target] + expansion (no re-truncation via _rerank_coverage_target)
        expanded_pool = seed_pool + expansion_candidates

        # Rerank the expanded pool (via new _rerank_pool_in_batches, avoiding the double-coverage-target footgun)
        reranked, reranker_status = _rerank_pool_in_batches(
            query, expanded_pool, chunk_content_by_id, batch_size
        )

        # CS-256: Final RRF-fuse of fused_rank (0.6) and cross_encoder_rank (0.4)
        final = _fuse_final_ranking(fused, reranked, weights=(0.6, 0.4))
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
        final=final,
    )


async def retrieve_rrf_fusion_as_bundle(
    snapshot_id: str,
    query: str,
    section: RetrievalSection,
    budget: int,
    mode: RetrievalMode = RetrievalMode.HYBRID,
    min_confidence: float | None = None,
) -> RetrievalBundle:
    """Run RRF multi-signal fusion (+ CS-254 cross-encoder + CS-256 1-hop-expand/final-fuse
    when the GPU reranker is enabled) and adapt it into the agent-compatible
    RetrievalBundle interface — mirrors retrieve_two_stage_as_bundle's shape exactly
    so callers (RetrievalService.retrieve) don't need to know which pipeline ran.

    Uses `final` (the post-expansion, post-rerank, RRF-refused ranking) when the
    cross-encoder ran; falls back to the plain 4-signal `fused` list when it didn't
    (no GPU / toggle off) — same graceful-degradation shape as the debug bundle.

    Unlike the debug bundle (deliberately unbounded, "no budget cap"), this applies
    real token-budget cropping, matching two-stage's _rank_and_budget convention,
    since this path feeds real LLM context windows.
    """
    bundle = await retrieve_rrf_fusion(snapshot_id, query, section, min_confidence=min_confidence)
    ranked_entries = bundle.final if bundle.final else bundle.fused

    # Crop to budget (token_estimate running sum), same simple-loop convention as
    # two_stage_retrieval._rank_and_budget's non-native fallback path.
    cropped: list[FusedRankEntry] = []
    used_tokens = 0
    for entry in ranked_entries:
        tok = entry.token_estimate or 1
        if used_tokens + tok > budget and cropped:
            continue
        cropped.append(entry)
        used_tokens += tok

    # chunk_index isn't carried on FusedRankEntry -- look it up for just the cropped
    # chunk_ids rather than adding a new field that would ripple through every
    # debug-panel type/UI consumer.
    chunk_index_by_id: dict[str, int] = {}
    if cropped:
        db = get_db()
        chunk_ids = [e.chunk_id for e in cropped]
        placeholders = ",".join("?" for _ in chunk_ids)
        async with db.execute(
            f"SELECT id, chunk_index FROM retrieval_chunks WHERE snapshot_id=? AND id IN ({placeholders})",
            (snapshot_id, *chunk_ids),
        ) as cur:
            rows = await cur.fetchall()
        chunk_index_by_id = {r["id"]: r["chunk_index"] for r in rows}

    evidences = [
        RetrievalEvidence(
            chunk_id=e.chunk_id,
            rel_path=e.rel_path,
            chunk_index=chunk_index_by_id.get(e.chunk_id, 0),
            reason_codes=[f"rrf-{name}" for name in e.per_signal_ranks] or ["rrf-fusion"],
            score=e.fused_score,
            token_estimate=e.token_estimate,
            excerpt=e.excerpt,
        )
        for e in cropped
    ]

    quality = None
    if cropped:
        terms = _query_terms(query)
        ranked_chunks = [
            RankedChunk(
                chunk_id=e.chunk_id,
                rel_path=e.rel_path,
                chunk_index=chunk_index_by_id.get(e.chunk_id, 0),
                score=e.fused_score,
                chunk_type="block",
                bm25_component=0.0,
                symbol_bonus=0.0,
                module_bonus=0.0,
                centrality_bonus=0.0,
                token_estimate=e.token_estimate,
                excerpt=e.excerpt,
            )
            for e in cropped
        ]
        quality = compute_retrieval_quality(terms, ranked_chunks, cropped[0].fused_score)

    return RetrievalBundle(
        snapshot_id=snapshot_id,
        mode=mode,
        section=section,
        query=query,
        budget_tokens=budget,
        used_tokens=used_tokens,
        evidences=evidences,
        quality=quality,
    )
