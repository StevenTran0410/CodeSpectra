"""Impact-aware retrieval pipeline (RPA-060).

Runs concurrent reverse-BFS cone queries and symbol call-chain traces for each
seed file, then ranks and budgets chunks using structural graph signals.
"""
from __future__ import annotations

import asyncio
import json
import re

from infrastructure.db.database import get_db
from shared.logger import logger

from domain.qa.graph_queries import get_impact_cone, trace_call_chain, ImpactResult, TraceStep
from domain.retrieval.two_stage_retrieval import _load_graph_context, _rank_and_budget, _get_native
from domain.retrieval.bm25_scorer import BM25Scorer, _load_native_bm25, _query_terms
from domain.retrieval.types import ImpactRankedChunk, ImpactRetrievalBundle, CommunityImpact

_WORD = re.compile(r"[A-Za-z0-9_]+")

# Scoring constants
_HOP_WEIGHTS: dict[int, float] = {0: 3.0, 1: 2.0, 2: 1.2, 3: 0.6, 4: 0.3}
_IMPACT_CENTRALITY_BONUS = 2.0
_IMPACT_CENTRALITY_DECAY = 0.06
_IMPACT_COMMUNITY_BONUS = 1.0
_IMPACT_NEIGHBOR_COMMUNITY_BONUS = 0.4
_IMPACT_SYMBOL_CHAIN_BONUS = 2.5
_IMPACT_BM25_WEIGHT = 0.3


# ── Private helpers ───────────────────────────────────────────────────────────


async def _load_central_files_ranked(snapshot_id: str, db) -> list[str]:
    """Returns top_central_files in rank order for decay formula."""
    async with db.execute(
        "SELECT top_central_files FROM structural_graph_summaries WHERE snapshot_id=?",
        (snapshot_id,),
    ) as cur:
        row = await cur.fetchone()
    if not row or not row["top_central_files"]:
        return []
    try:
        arr = json.loads(row["top_central_files"])
        return [item["rel_path"] for item in arr if isinstance(item, dict) and "rel_path" in item]
    except Exception:
        return []


async def _load_neighbor_communities(snapshot_id: str, db) -> dict[int, list[int]]:
    """Returns community_id -> list[neighbor_community_id]."""
    try:
        async with db.execute(
            "SELECT community_id, neighbor_community_ids FROM graph_community_summaries WHERE snapshot_id=?",
            (snapshot_id,),
        ) as cur:
            rows = await cur.fetchall()
    except Exception:
        return {}
    result: dict[int, list[int]] = {}
    for row in rows:
        try:
            neighbors = json.loads(row["neighbor_community_ids"] or "[]")
            result[int(row["community_id"])] = [int(n) for n in neighbors]
        except Exception:
            result[int(row["community_id"])] = []
    return result


def _merge_hop_maps(results: list[ImpactResult]) -> dict[str, int]:
    """Union of hop_maps from multiple seeds; keep minimum hop distance."""
    merged: dict[str, int] = {}
    for r in results:
        for f, h in r.hop_map.items():
            merged[f] = min(merged.get(f, 999), h)
    return merged


def _collect_call_chain_files(trace_results: list[list[TraceStep]]) -> set[str]:
    """All files appearing in any call-chain traversal."""
    out: set[str] = set()
    for steps in trace_results:
        for step in steps:
            out.add(step.file)
    return out


def _score_chunk(
    rel_path: str,
    hop_distance: int | None,
    rank_in_central: int | None,
    community_id: int | None,
    seed_community_ids: set[int],
    neighbor_community_ids: set[int],
    is_in_call_chain: bool,
    bm25_score: float,
) -> tuple[float, float, float, float, float]:
    """Return (total, bm25_boost, symbol_bonus, community_bonus, centrality_bonus)."""
    # Hop weight (structural graph proximity)
    if hop_distance is not None:
        hop_w = _HOP_WEIGHTS.get(hop_distance, 0.2)
    else:
        hop_w = 0.0

    # BM25 boost
    bm25_boost = bm25_score * _IMPACT_BM25_WEIGHT

    # Symbol chain bonus (additive)
    symbol_bonus = _IMPACT_SYMBOL_CHAIN_BONUS if is_in_call_chain else 0.0

    # Community proximity bonus (additive)
    if community_id is not None and community_id in seed_community_ids:
        community_bonus = _IMPACT_COMMUNITY_BONUS
    elif community_id is not None and community_id in neighbor_community_ids:
        community_bonus = _IMPACT_NEIGHBOR_COMMUNITY_BONUS
    else:
        community_bonus = 0.0

    # Centrality bonus with exponential decay by rank
    if rank_in_central is not None:
        centrality_bonus = _IMPACT_CENTRALITY_BONUS * (1.0 - _IMPACT_CENTRALITY_DECAY) ** rank_in_central
    else:
        centrality_bonus = 0.0

    total = hop_w + bm25_boost + symbol_bonus + community_bonus + centrality_bonus
    return total, bm25_boost, symbol_bonus, community_bonus, centrality_bonus


def _build_risk_summary(
    hop_map: dict[str, int],
    central_files_set: set[str],
    file_community: dict[str, int],
    seed_community_ids: set[int],
) -> dict:
    total_affected = len(hop_map)
    high_risk_files = sorted(f for f in hop_map if f in central_files_set)
    by_hop: dict[int, int] = {}
    for h in hop_map.values():
        by_hop[h] = by_hop.get(h, 0) + 1
    affected_community_ids: set[int] = set()
    for f in hop_map:
        cid = file_community.get(f)
        if cid is not None:
            affected_community_ids.add(cid)
    return {
        "total_affected": total_affected,
        "by_hop": by_hop,
        "high_risk_files": high_risk_files,
        "affected_community_count": len(affected_community_ids),
        "seed_community_ids": sorted(seed_community_ids),
    }


def _build_affected_communities(
    hop_map: dict[str, int],
    file_community: dict[str, int],
    community_summaries: list,
) -> list[CommunityImpact]:
    # Map community_id -> summary row
    summary_map: dict[int, dict] = {}
    for row in community_summaries:
        try:
            cid = int(row["community_id"])
            summary_map[cid] = row
        except Exception:
            pass

    # Collect community ids touched by the cone
    affected: dict[int, list[str]] = {}
    for f in hop_map:
        cid = file_community.get(f)
        if cid is not None:
            affected.setdefault(cid, []).append(f)

    out: list[CommunityImpact] = []
    for cid, members in sorted(affected.items()):
        row = summary_map.get(cid)
        if row:
            try:
                hub_paths = json.loads(row.get("hub_paths") or "[]")
            except Exception:
                hub_paths = []
            try:
                member_count = int(row.get("member_count") or len(members))
            except Exception:
                member_count = len(members)
        else:
            hub_paths = []
            member_count = len(members)
        out.append(CommunityImpact(
            community_id=cid,
            member_count=member_count,
            hub_paths=hub_paths[:5],
        ))
    return out


# ── Public API ────────────────────────────────────────────────────────────────


async def retrieve_impact(
    snapshot_id: str,
    seed_files: list[str],
    query: str | None = None,
    max_hops: int = 3,
    budget: int = 8000,
) -> ImpactRetrievalBundle:
    """Graph-first retrieval for impact analysis.

    Steps:
      1-2. Concurrent cone + trace queries for each seed file.
      3.   Concurrent supplementary graph context loads.
      4.   Merge hop maps; seed files forced to hop 0.
      5.   Load chunks (cone-only IN-clause, or full corpus + BM25 if query given).
      6.   Score each chunk with _score_chunk().
      7.   _rank_and_budget() -> ImpactRankedChunk list.
      8.   Assemble and return ImpactRetrievalBundle.
    """
    db = get_db()
    native = _get_native()

    # ── Step 1-2: Concurrent cone + trace queries ─────────────────────────────
    if seed_files:
        cone_tasks = [get_impact_cone(snapshot_id, s, max_hops) for s in seed_files]
        trace_tasks = [trace_call_chain(snapshot_id, s, "backward") for s in seed_files]

        # Step 3: Overlap I/O — run all graph queries + supplementary loads concurrently
        (cone_results, trace_results), ctx, central_ranked, neighbor_cid_map = await asyncio.gather(
            asyncio.gather(
                asyncio.gather(*cone_tasks),
                asyncio.gather(*trace_tasks),
            ),
            _load_graph_context(snapshot_id),
            _load_central_files_ranked(snapshot_id, db),
            _load_neighbor_communities(snapshot_id, db),
        )
    else:
        # No seeds (plan mode): skip cone/trace, only load graph context
        ctx, central_ranked, neighbor_cid_map = await asyncio.gather(
            _load_graph_context(snapshot_id),
            _load_central_files_ranked(snapshot_id, db),
            _load_neighbor_communities(snapshot_id, db),
        )
        cone_results = []
        trace_results = []

    # ── Step 4: Merge hop maps ────────────────────────────────────────────────
    merged_hop_map = _merge_hop_maps(cone_results)
    for s in seed_files:
        merged_hop_map[s] = 0

    call_chain_files = _collect_call_chain_files(trace_results)

    # Build lookup structures for scoring
    central_rank_map: dict[str, int] = {path: i for i, path in enumerate(central_ranked)}
    seed_community_ids: set[int] = set()
    for s in seed_files:
        cid = ctx.file_community.get(s)
        if cid is not None:
            seed_community_ids.add(cid)

    # Expand neighbor_community_ids for seed communities
    neighbor_community_ids: set[int] = set()
    for scid in seed_community_ids:
        for ncid in neighbor_cid_map.get(scid, []):
            if ncid not in seed_community_ids:
                neighbor_community_ids.add(ncid)

    cone_files = set(merged_hop_map.keys())

    # ── Step 5: Load chunks ───────────────────────────────────────────────────
    terms: list[str] = _query_terms(query) if query else []
    scorer: BM25Scorer | None = None

    if query and terms:
        # Load full corpus, BM25-score, take top-50 outside cone + all cone chunks
        async with db.execute(
            "SELECT id, rel_path, chunk_index, language, category, content, token_estimate "
            "FROM retrieval_chunks WHERE snapshot_id=?",
            (snapshot_id,),
        ) as cur:
            all_rows = await cur.fetchall()

        if not all_rows:
            logger.warning("[impact_retrieval] No retrieval_chunks for snapshot %s", snapshot_id)
            all_rows = []

        async with db.execute(
            "SELECT avgdl, idf_json, k1, b FROM retrieval_bm25_stats WHERE snapshot_id=?",
            (snapshot_id,),
        ) as cur:
            bm25_row = await cur.fetchone()
        scorer = BM25Scorer.from_stats_row(bm25_row)

        # Score all rows for BM25
        bm25_scores: dict[str, float] = {}
        if scorer:
            for r in all_rows:
                content = r["content"] or ""
                s = scorer.score(terms, content.lower(), r["rel_path"].lower())
                bm25_scores[r["id"]] = s

        # Select: all cone rows + top-50 BM25 hits outside cone
        outside_cone_scored = [
            (bm25_scores.get(r["id"], 0.0), r)
            for r in all_rows
            if r["rel_path"] not in cone_files
        ]
        outside_cone_scored.sort(key=lambda x: -x[0])
        outside_top50_ids: set[str] = {r["id"] for _, r in outside_cone_scored[:50]}

        candidate_rows = [
            r for r in all_rows
            if r["rel_path"] in cone_files or r["id"] in outside_top50_ids
        ]

    else:
        # No query: load only cone files
        if not cone_files:
            # No seeds, no query — nothing to retrieve
            candidate_rows = []
        else:
            placeholders = ",".join("?" * len(cone_files))
            async with db.execute(
                f"SELECT id, rel_path, chunk_index, language, category, content, token_estimate "
                f"FROM retrieval_chunks WHERE snapshot_id=? AND rel_path IN ({placeholders})",
                (snapshot_id, *sorted(cone_files)),
            ) as cur:
                candidate_rows = await cur.fetchall()
        bm25_scores = {}

    # ── Step 6: Score each chunk ──────────────────────────────────────────────
    # _rank_and_budget expects 6-tuples: (total, c2, c3, c4, c5, row_dict)
    # We map: (total, bm25_boost, symbol_bonus, community_bonus, centrality_bonus, row)
    all_scored: list[tuple[float, float, float, float, float, dict]] = []

    native_bm25 = _load_native_bm25()
    _used_native_impact = False

    if native_bm25 is not None and candidate_rows:
        try:
            # Build inputs for native batch_impact_score
            chunk_dicts = [{"id": r["id"], "rel_path": r["rel_path"]} for r in candidate_rows]
            row_by_id = {r["id"]: dict(r) for r in candidate_rows}

            native_results = native_bm25.batch_impact_score(
                chunk_dicts,
                merged_hop_map,
                central_rank_map,
                ctx.file_community,
                seed_community_ids,
                call_chain_files,
                bm25_scores,
            )
            _used_native_impact = True

            for chunk_id, total, bm25_boost, symbol_bonus, community_bonus, centrality_bonus in native_results:
                # Apply neighbor community bonus that native doesn't handle (separate set)
                r_dict = row_by_id.get(chunk_id)
                if r_dict is None:
                    continue
                rel_path = r_dict["rel_path"]
                community_id = ctx.file_community.get(rel_path)
                if community_id is not None and community_id not in seed_community_ids and community_id in neighbor_community_ids:
                    community_bonus += _IMPACT_NEIGHBOR_COMMUNITY_BONUS
                    total += _IMPACT_NEIGHBOR_COMMUNITY_BONUS
                if total > 0:
                    all_scored.append((total, bm25_boost, symbol_bonus, community_bonus, centrality_bonus, r_dict))
        except Exception:
            _used_native_impact = False
            all_scored = []

    if not _used_native_impact:
        for r in candidate_rows:
            rel_path = r["rel_path"]
            hop_distance = merged_hop_map.get(rel_path)
            rank_in_central = central_rank_map.get(rel_path)
            community_id = ctx.file_community.get(rel_path)
            is_in_call_chain = rel_path in call_chain_files
            bm25_score = bm25_scores.get(r["id"], 0.0)

            total, bm25_boost, symbol_bonus, community_bonus, centrality_bonus = _score_chunk(
                rel_path=rel_path,
                hop_distance=hop_distance,
                rank_in_central=rank_in_central,
                community_id=community_id,
                seed_community_ids=seed_community_ids,
                neighbor_community_ids=neighbor_community_ids,
                is_in_call_chain=is_in_call_chain,
                bm25_score=bm25_score,
            )

            if total > 0:
                all_scored.append((total, bm25_boost, symbol_bonus, community_bonus, centrality_bonus, dict(r)))

    # ── Step 7: _rank_and_budget ──────────────────────────────────────────────
    # Shared budget-aware ranker — reused from two_stage_retrieval by design.
    # If this function moves, update both callers.
    ranked_base, _used_cpp = _rank_and_budget(all_scored, budget, native)

    # Convert RankedChunk -> ImpactRankedChunk (enrich with impact-specific fields)
    # Build score component lookup from all_scored
    chunk_score_map: dict[str, tuple[float, float, float, float, float]] = {}
    for total, bm25_boost, symbol_bonus, community_bonus, centrality_bonus, r in all_scored:
        chunk_score_map[r["id"]] = (total, bm25_boost, symbol_bonus, community_bonus, centrality_bonus)

    ranked_chunks: list[ImpactRankedChunk] = []
    for rc in ranked_base:
        comps = chunk_score_map.get(rc.chunk_id, (rc.score, 0.0, 0.0, 0.0, 0.0))
        _, bm25_boost, symbol_bonus, community_bonus, centrality_bonus = comps
        hop_distance = merged_hop_map.get(rc.rel_path)
        hop_weight = _HOP_WEIGHTS.get(hop_distance, 0.0) if hop_distance is not None else 0.0
        ranked_chunks.append(ImpactRankedChunk(
            chunk_id=rc.chunk_id,
            rel_path=rc.rel_path,
            chunk_index=rc.chunk_index,
            hop_distance=hop_distance,
            score=rc.score,
            hop_weight=hop_weight,
            centrality_bonus=centrality_bonus,
            community_bonus=community_bonus,
            symbol_bonus=symbol_bonus,
            bm25_boost=bm25_boost,
            token_estimate=rc.token_estimate,
            excerpt=rc.excerpt,
        ))

    # ── Step 8: Assemble ImpactRetrievalBundle ────────────────────────────────
    # Load community summaries for affected_communities enrichment
    try:
        async with db.execute(
            "SELECT community_id, member_count, hub_paths FROM graph_community_summaries WHERE snapshot_id=?",
            (snapshot_id,),
        ) as cur:
            community_summary_rows = await cur.fetchall()
    except Exception:
        community_summary_rows = []

    affected_communities = _build_affected_communities(
        merged_hop_map, ctx.file_community, community_summary_rows
    )

    risk_summary = _build_risk_summary(
        merged_hop_map, ctx.central_files, ctx.file_community, seed_community_ids
    )

    # Serialize call chains (TraceStep -> dict)
    call_chains_serial: list[dict] = []
    for steps in trace_results:
        call_chains_serial.append([
            {
                "step": step.step,
                "file": step.file,
                "symbols_involved": step.symbols_involved,
                "is_seed": step.is_seed,
            }
            for step in steps
        ])

    return ImpactRetrievalBundle(
        seed_files=seed_files,
        query=query,
        impact_cone=merged_hop_map,
        affected_communities=affected_communities,
        call_chains=call_chains_serial,
        ranked_chunks=ranked_chunks,
        risk_summary=risk_summary,
    )
