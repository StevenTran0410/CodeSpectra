"""RRF Multi-Signal Fusion Query System: combines BM25 lexical signal with graph-confidence signal via reciprocal rank fusion (RRF), producing debug bundles for analysis and comparison."""

from __future__ import annotations

import time

from haystack import Document
from haystack.utils.misc import _reciprocal_rank_fusion

from infrastructure.db.database import get_db
from shared.logger import logger

from .bm25_scorer import BM25Scorer
from .cross_encoder_rerank import (
    detect_gpu,
    is_gpu_reranker_enabled,
    recommended_rerank_batch_size,
    rerank_fused_entries,
    release_gpu_cache,
)
from .service import _CHUNK_FULL_COLS
from .two_stage_retrieval import (
    _load_graph_context,
    _query_terms,
    _stage1_score_rows,
    load_symbol_index,
)
from .quality import compute_retrieval_quality
from .signal_builders import (
    _best_chunk_per_file,
    _load_confidence_weighted_edges,
    build_bm25_rank_list,
    build_category_hint_rank_list,
    build_graph_confidence_rank_list,
    build_module_proximity_rank_list,
)
from .graph_signal import (
    _expand_function_level_1hop,
    _rerank_coverage_target,
)
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


def fuse_signal_lists(
    signal_lists: list[list[SignalRankEntry]],
    weights: list[float] | None = None,
) -> list[FusedRankEntry]:
    """Fuse multiple ranked signal lists via Reciprocal Rank Fusion (using haystack's _reciprocal_rank_fusion, k=61); returns FusedRankEntry list sorted by fused_score descending."""
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

    # Pre-build chunk_id -> {ranks, excerpt, rel_path} in one O(total entries) pass instead of re-scanning every signal's entries per fused doc. Per signal, the FIRST occurrence of a chunk_id wins; for excerpt/rel_path, the first signal list (in order) yielding a truthy value wins.
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

    # _reciprocal_rank_fusion() returns results in insertion order, not sorted by fused_score -- this explicit sort is what actually re-ranks.
    fused_entries.sort(key=lambda e: e.fused_score, reverse=True)
    return fused_entries


def _rerank_pool_in_batches(
    query: str,
    pool: list[FusedRankEntry],
    chunk_content_by_id: dict[str, str],
    batch_size: int,
) -> tuple[list, str]:
    """Rerank a pre-determined pool of candidates in sequential batches. Batches over the `pool` argument exactly as received -- does NOT re-derive _rerank_coverage_target, to avoid silently truncating expansion-only candidates from an already-capped pool built by the caller. Returns (reranked_entries, status_code) sorted by rerank_score descending."""
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
    """Fuse fused_rank and cross_encoder_rank signals via RRF. `fused_rank` comes from the original `fused` list's order; `cross_encoder_rank` comes from `reranked`'s post-sort position (enumerate, not RerankedEntry.fused_rank). Expansion-only candidates (in reranked but not fused) contribute only via cross_encoder_rank. Returns FusedRankEntry list sorted by combined fused_score descending."""
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

    # List 2: cross_encoder_rank signal, built from reranked's post-sort position -- HARD REQUIREMENT: use enumerate(reranked, start=1), NOT RerankedEntry.fused_rank.
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
    """Run RRF multi-signal fusion retrieval (debug path). Loads identical all_rows/ctx/symbol_index as retrieve_two_stage, builds all 4 signal rank lists (BM25, graph-confidence, module-proximity, category-hint), fuses via RRF, and returns raw unbounded results -- does NOT route through _rank_and_budget()/_apply_diversity_filter(), since this is a debug/comparison path, not a production synthesis input."""
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

    # Confidence-weighted edges, filtered to seed files in SQL instead of fetching every edge in the snapshot and filtering in Python.
    confidence_edges = await _load_confidence_weighted_edges(snapshot_id, stage1_files)

    # Compute seed_community_ids from top-20 stage1 candidates (mirrors two_stage_retrieval.py).
    seed_community_ids: set[int] = set()
    for c in stage1_candidates[:20]:
        cid = ctx.file_community.get(c.rel_path)
        if cid is not None:
            seed_community_ids.add(cid)

    # Convert all_rows to dicts once (reuse everywhere)
    all_rows_dicts = [dict(r) for r in all_rows]

    # Build chunk_content_by_id for reranking
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

    # Call cross-encoder reranking with 1-hop expansion, gated by the global GPU Reranker toggle (off by default). Runs in VRAM-sized batches since the model joins all passages into one shared sequence per call, not pairwise.
    final = []
    if await is_gpu_reranker_enabled():
        _, vram_gb = detect_gpu()
        batch_size = recommended_rerank_batch_size(vram_gb)

        # Function-level 1-hop expansion
        target = _rerank_coverage_target(len(fused))
        seed_pool = fused[:target]
        expansion_cap = recommended_rerank_batch_size(vram_gb)  # Reuse VRAM-scaled ceiling for OOM-safety
        expansion_candidates = await _expand_function_level_1hop(
            snapshot_id, seed_pool, symbol_index, all_rows_dicts, expansion_cap
        )

        # Build expanded pool = fused[:target] + expansion (no re-truncation via _rerank_coverage_target)
        expanded_pool = seed_pool + expansion_candidates

        # Rerank the expanded pool (via new _rerank_pool_in_batches, avoiding the double-coverage-target footgun)
        t0 = time.monotonic()
        reranked, reranker_status = _rerank_pool_in_batches(
            query, expanded_pool, chunk_content_by_id, batch_size
        )
        elapsed = time.monotonic() - t0

        # Log reranking summary
        if reranked and reranker_status == "ok":
            num_batches = (len(expanded_pool) + batch_size - 1) // batch_size
            logger.info(
                "[rrf] reranked %d total candidates across %d batches in %.2fs for query=%r",
                len(reranked),
                num_batches,
                elapsed,
                query[:80],
            )

        # Final RRF-fuse of fused_rank (0.6) and cross_encoder_rank (0.4)
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
    """Run RRF multi-signal fusion (+ cross-encoder + 1-hop-expand/final-fuse when the GPU reranker is enabled) and adapt it into the agent-compatible RetrievalBundle interface, mirroring retrieve_two_stage_as_bundle's shape. Uses `final` when the cross-encoder ran, falling back to the plain 4-signal `fused` list otherwise (no GPU / toggle off). Unlike the debug bundle (unbounded), this applies real token-budget cropping since it feeds real LLM context windows."""
    bundle = await retrieve_rrf_fusion(snapshot_id, query, section, min_confidence=min_confidence)
    ranked_entries = bundle.final if bundle.final else bundle.fused

    # Crop to budget (token_estimate running sum), same simple-loop convention as two_stage_retrieval._rank_and_budget's non-native fallback path.
    cropped: list[FusedRankEntry] = []
    used_tokens = 0
    for entry in ranked_entries:
        tok = entry.token_estimate or 1
        if used_tokens + tok > budget and cropped:
            continue
        cropped.append(entry)
        used_tokens += tok

    # chunk_index isn't carried on FusedRankEntry -- look it up for just the cropped chunk_ids rather than adding a new field that would ripple through every debug-panel type/UI consumer.
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
        from .two_stage_retrieval import _query_terms

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
