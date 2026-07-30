"""Signal builders for RRF multi-signal fusion.

Constructs ranked signal lists from BM25, graph-confidence, module-proximity,
and category-hint heuristics.
"""

from __future__ import annotations

from infrastructure.db.database import get_db

from .service import _SECTION_CATEGORY_HINTS
from .two_stage_retrieval import (
    _CATEGORY_HINT_BONUS,
    _MODULE_PROXIMITY_BONUS,
)
from .types import (
    RetrievalSection,
    SignalRankEntry,
)

# CAP = 2.0: anti-gaming constant. Prevents many low-confidence ambiguous edges
# from outranking genuinely high-confidence ones via edge-count gaming.
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
    same collision-safe convention as copy_unchanged_symbol_edges -- "foo.py" never
    matches "foo2.py::..."), chunked at 200 paths per round-trip, instead of fetching every
    edge in the snapshot and filtering in Python.
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
                start_line=int(best_chunk.get("start_line") or 0),
                end_line=int(best_chunk.get("end_line") or 0),
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
                    start_line=int(best_chunk.get("start_line") or 0),
                    end_line=int(best_chunk.get("end_line") or 0),
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
                    start_line=int(best_chunk.get("start_line") or 0),
                    end_line=int(best_chunk.get("end_line") or 0),
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
                    start_line=int(best_chunk.get("start_line") or 0),
                    end_line=int(best_chunk.get("end_line") or 0),
                )
            )

    return entries
