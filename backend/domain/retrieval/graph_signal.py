"""Function-level 1-hop graph expansion for cross-encoder reranking (CS-256).

Expands retrieval candidate pools with callees/callers of top-ranked functions,
resolving symbols to specific code chunks for code-level retrieval.
"""

from __future__ import annotations

from domain.qa.graph_queries import get_callees_of, get_callers_of

from .types import FusedRankEntry


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


def _rerank_coverage_target(total_fused: int) -> int:
    """Compute target number of fused entries to seed function-level 1-hop expansion.

    At least 100 candidates when available, but never more than 3/4 of the total fused pool
    (rerank is meant to refine a top slice, not replace RRF fusion's own judgment over the long tail).
    """
    _RERANK_COVERAGE_MIN = 100
    _RERANK_COVERAGE_MAX_FRACTION = 0.75
    return min(total_fused, max(_RERANK_COVERAGE_MIN, round(total_fused * _RERANK_COVERAGE_MAX_FRACTION)))


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
