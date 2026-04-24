"""Two-stage retrieval pipeline (CS-203): BM25 stage1 → graph expansion stage2 → rank_and_budget stage3."""
from __future__ import annotations

import importlib
import json
import re
from dataclasses import dataclass, field

from infrastructure.db.database import get_db
from shared.logger import logger

from .bm25_scorer import CHUNK_TYPE_WEIGHT, BM25Scorer, _query_terms
from .service import _CHUNK_FULL_COLS
from .types import (
    RankedChunk,
    RetrievalBundle,
    RetrievalEvidence,
    RetrievalMode,
    RetrievalSection,
    StageCandidate,
    StageExpansion,
    TwoStageBundle,
    TwoStageStage3Result,
)

_WORD = re.compile(r"[A-Za-z0-9_]+")


def _row_get(row, key: str, default):
    """Safe accessor for sqlite3.Row (no .get method) and dict-like rows."""
    try:
        val = row[key]
    except (IndexError, KeyError):
        return default
    return default if val is None else val

_SYMBOL_OVERLAP_BONUS: float = 1.5
_MODULE_PROXIMITY_BONUS: float = 1.3
_CENTRALITY_BONUS_MAX: float = 2.6
_CENTRALITY_BONUS_DECAY: float = 0.08

# Mirrors service.py _SECTION_CATEGORY_HINTS — keep in sync if sections change.
_SECTION_CATEGORY_HINTS: dict[RetrievalSection, set[str]] = {
    RetrievalSection.ARCHITECTURE:    {"source", "config", "infra"},
    RetrievalSection.CONVENTIONS:     {"source", "test", "config"},
    RetrievalSection.FEATURE_MAP:     {"source", "docs"},
    RetrievalSection.IMPORTANT_FILES: {"source", "config", "infra"},
    RetrievalSection.GLOSSARY:        {"source", "docs"},
    RetrievalSection.QA:              {"source", "config", "docs"},
}
_CATEGORY_HINT_BONUS: float = 1.4

# Section-specific score cutoffs (absolute, relative) — CS-227
_SECTION_SCORE_FLOOR: dict[RetrievalSection, tuple[float, float]] = {
    RetrievalSection.QA:              (0.5, 0.35),
    RetrievalSection.ARCHITECTURE:    (0.3, 0.15),
    RetrievalSection.CONVENTIONS:     (0.5, 0.25),
    RetrievalSection.FEATURE_MAP:     (0.5, 0.25),
    RetrievalSection.IMPORTANT_FILES: (0.5, 0.25),
    RetrievalSection.GLOSSARY:        (0.4, 0.20),
}

# Symbol cache: snapshot_id -> dict[term_lower, list[(rel_path, start_line, end_line)]]
_symbol_cache: dict[str, dict[str, list[tuple[str, int, int]]]] = {}

_NATIVE = None


def _get_native():
    global _NATIVE
    if _NATIVE is None:
        try:
            _NATIVE = importlib.import_module("domain.structural_graph._native_graph")
        except Exception:
            pass
    return _NATIVE


@dataclass
class _GraphContext:
    file_symbol_refs: dict[str, set[str]] = field(default_factory=dict)
    file_community: dict[str, int] = field(default_factory=dict)
    community_members: dict[int, set[str]] = field(default_factory=dict)
    edge_tuples: list[tuple] = field(default_factory=list)
    central_files: set[str] = field(default_factory=set)


async def _load_graph_context(snapshot_id: str) -> _GraphContext:
    db = get_db()
    ctx = _GraphContext()

    async with db.execute(
        "SELECT DISTINCT src_symbol, dst_symbol FROM symbol_graph_edges WHERE snapshot_id=?",
        (snapshot_id,),
    ) as cur:
        rows = await cur.fetchall()
    for row in rows:
        src_file = row["src_symbol"].split("::")[0] if "::" in (row["src_symbol"] or "") else row["src_symbol"]
        dst_file = row["dst_symbol"].split("::")[0] if "::" in (row["dst_symbol"] or "") else row["dst_symbol"]
        if src_file and dst_file and src_file != dst_file:
            ctx.file_symbol_refs.setdefault(src_file, set()).add(dst_file)

    async with db.execute(
        "SELECT node_path, community_id FROM graph_community_members WHERE snapshot_id=?",
        (snapshot_id,),
    ) as cur:
        rows = await cur.fetchall()
    for row in rows:
        cid = int(row["community_id"])
        ctx.file_community[row["node_path"]] = cid
        ctx.community_members.setdefault(cid, set()).add(row["node_path"])

    async with db.execute(
        "SELECT src_path, dst_path, edge_type, is_external FROM structural_graph_edges WHERE snapshot_id=?",
        (snapshot_id,),
    ) as cur:
        rows = await cur.fetchall()
    for row in rows:
        ctx.edge_tuples.append((row["src_path"], row["dst_path"], row["edge_type"] or "", int(row["is_external"] or 0)))

    async with db.execute(
        "SELECT top_central_files FROM structural_graph_summaries WHERE snapshot_id=?",
        (snapshot_id,),
    ) as cur:
        row = await cur.fetchone()
    if row and row["top_central_files"]:
        try:
            arr = json.loads(row["top_central_files"])
            for item in arr:
                rp = item.get("rel_path") if isinstance(item, dict) else None
                if isinstance(rp, str):
                    ctx.central_files.add(rp)
        except Exception:
            pass

    return ctx


async def clear_symbol_cache(snapshot_id: str) -> None:
    """Clear symbol index cache for a snapshot (called on force-rebuild)."""
    _symbol_cache.pop(snapshot_id, None)


async def load_symbol_index(
    snapshot_id: str,
) -> dict[str, list[tuple[str, int, int]]]:
    """Load symbol index for a snapshot, cached in-memory.

    Returns: dict[term_lower, list[(rel_path, start_line, end_line)]]
    """
    if snapshot_id in _symbol_cache:
        return _symbol_cache[snapshot_id]

    db = get_db()
    index: dict[str, list[tuple[str, int, int]]] = {}

    async with db.execute(
        "SELECT name, rel_path, line_start, line_end FROM code_symbols WHERE snapshot_id=?",
        (snapshot_id,),
    ) as cur:
        rows = await cur.fetchall()

    for row in rows:
        name_lower = (row["name"] or "").lower()
        if name_lower:
            index.setdefault(name_lower, []).append(
                (row["rel_path"], int(row["line_start"]), int(row["line_end"]))
            )

    _symbol_cache[snapshot_id] = index
    return index


def _stage1_score_rows(rows: list, scorer: BM25Scorer | None, terms: list[str], top_k: int = 100) -> list[StageCandidate]:
    candidates: list[tuple[float, StageCandidate]] = []
    for r in rows:
        content = r["content"] or ""
        path_low = r["rel_path"].lower()
        chunk_type = _row_get(r, "chunk_type", "block")
        # Lowercase once; reused by both scorer and fallback path.
        content_low = content.lower()
        if scorer is not None:
            score = scorer.score(terms, content_low, path_low, chunk_type)
            if score <= 0.0:
                continue
        else:
            # Counter-based O(N+M) whole-word matching — aligns with the tokenisation
            # model used by the native BM25 scorer (word tokens, not substrings).
            token_counts = _WORD.findall(content_low)
            freq: dict[str, int] = {}
            for tok in token_counts:
                freq[tok] = freq.get(tok, 0) + 1
            hits = sum(freq.get(t, 0) for t in terms) + sum(2 for t in terms if t in path_low)
            if hits <= 0:
                continue
            score = float(hits)
        candidates.append((score, StageCandidate(
            chunk_id=r["id"],
            rel_path=r["rel_path"],
            chunk_index=int(r["chunk_index"]),
            bm25_score=score,
            token_estimate=int(r["token_estimate"] or max(1, len(content) // 4)),
            excerpt=content,
        )))
    candidates.sort(key=lambda x: -x[0])
    return [c for _, c in candidates[:top_k]]


def _python_1hop(seed: str, edge_tuples: list[tuple]) -> list[str]:
    neighbors: set[str] = set()
    for src, dst, _etype, is_ext in edge_tuples:
        if is_ext:
            continue
        if src == seed and dst != seed:
            neighbors.add(dst)
        elif dst == seed and src != seed:
            neighbors.add(src)
    return sorted(neighbors)[:50]


def _expand_one(seed: StageCandidate, ctx: _GraphContext, native) -> tuple[StageExpansion, set[str]]:
    symbol_refs = list(ctx.file_symbol_refs.get(seed.rel_path, set()))

    cid = ctx.file_community.get(seed.rel_path)
    if cid is not None:
        community_members = list(ctx.community_members.get(cid, set()) - {seed.rel_path})
    else:
        community_members = []

    if ctx.edge_tuples:
        if native:
            try:
                result = native.expand_neighbors(seed.rel_path, ctx.edge_tuples, 1, 50)
                neighbor_files = list(set(result["nodes"]) - {seed.rel_path})
            except Exception:
                neighbor_files = _python_1hop(seed.rel_path, ctx.edge_tuples)
        else:
            neighbor_files = _python_1hop(seed.rel_path, ctx.edge_tuples)
    else:
        neighbor_files = []

    expanded = set(symbol_refs) | set(community_members) | set(neighbor_files)
    expansion = StageExpansion(
        seed_path=seed.rel_path,
        symbol_refs=symbol_refs,
        community_members=community_members,
        neighbor_files=neighbor_files,
        net_new_count=len(expanded),
    )
    return expansion, expanded


def _compute_chunk_score(
    r: dict,
    terms: list[str],
    scorer: BM25Scorer | None,
    seed_files: set[str],
    ctx: _GraphContext,
    seed_community_ids: set[int],
    category_hints: set[str],
    symbol_index: dict[str, list[tuple[str, int, int]]] | None = None,
) -> tuple[float, float, float, float, float]:
    content = r["content"] or ""
    path_low = r["rel_path"].lower()
    chunk_type = _row_get(r, "chunk_type", "block")
    # Lowercase once; reused by both scorer and fallback path.
    content_low = content.lower()
    if scorer is not None:
        bm25 = scorer.score(terms, content_low, path_low, chunk_type)
    else:
        # Counter-based O(N+M) whole-word matching — aligns with the tokenisation
        # model used by the native BM25 scorer (word tokens, not substrings).
        token_counts = _WORD.findall(content_low)
        freq: dict[str, int] = {}
        for tok in token_counts:
            freq[tok] = freq.get(tok, 0) + 1
        bm25 = float(sum(freq.get(t, 0) for t in terms) + sum(2 for t in terms if t in path_low))

    # Definition bonus: if term matches symbol in this chunk's line range
    definition_bonus = 0.0
    if symbol_index:
        chunk_start = _row_get(r, "start_line", 0)
        chunk_end = _row_get(r, "end_line", 0)
        chunk_rel_path = r["rel_path"]
        for term in terms:
            matching_symbols = symbol_index.get(term, [])
            for sym_path, sym_start, sym_end in matching_symbols:
                if (
                    sym_path == chunk_rel_path
                    and chunk_start > 0
                    and sym_start >= chunk_start
                    and sym_end <= chunk_end
                ):
                    definition_bonus = 3.0
                    break
            if definition_bonus > 0:
                break
    bm25 += definition_bonus

    if r["category"] in category_hints:
        bm25 += _CATEGORY_HINT_BONUS

    sym_bonus = _SYMBOL_OVERLAP_BONUS if r["rel_path"] in seed_files else 1.0
    cid = ctx.file_community.get(r["rel_path"])
    mod_bonus = _MODULE_PROXIMITY_BONUS if (cid is not None and cid in seed_community_ids) else 1.0
    cent_bonus = _CENTRALITY_BONUS_MAX if r["rel_path"] in ctx.central_files else 0.0
    # Apply chunk_type weight to the FULL total (not just BM25) so graph bonuses
    # (symbol/module/centrality) are also suppressed for low-value chunk types
    # like import_group. Otherwise a central file's import-only chunk still ranks
    # top-K because cent_bonus=2.6 dominates even after BM25 is scaled down.
    chunk_weight = CHUNK_TYPE_WEIGHT.get(chunk_type, 1.0)
    total = (bm25 * sym_bonus * mod_bonus + cent_bonus) * chunk_weight
    return total, bm25, sym_bonus, mod_bonus, cent_bonus


def _rank_and_budget(
    scored: list[tuple[float, float, float, float, float, dict]],
    budget: int,
    native,
) -> tuple[list[RankedChunk], bool]:
    used_cpp = False
    chunk_map: dict[str, tuple[float, float, float, float, float, dict]] = {}
    for total, bm25, sym, mod, cent, r in scored:
        chunk_map[r["id"]] = (total, bm25, sym, mod, cent, r)

    if native:
        try:
            inputs = [(r["id"], total, int(r["token_estimate"] or 1)) for total, _, _, _, _, r in scored]
            ranked_ids: list[str] = list(native.rank_and_budget(inputs, budget))
            used_cpp = True
            out: list[RankedChunk] = []
            for cid in ranked_ids:
                if cid not in chunk_map:
                    continue
                total, bm25, sym, mod, cent, r = chunk_map[cid]
                out.append(RankedChunk(
                    chunk_id=cid,
                    rel_path=r["rel_path"],
                    chunk_index=int(r["chunk_index"]),
                    score=total,
                    bm25_component=bm25,
                    symbol_bonus=sym,
                    module_bonus=mod,
                    centrality_bonus=cent,
                    token_estimate=int(r["token_estimate"] or 1),
                    excerpt=r["content"] or "",
                ))
            return out, used_cpp
        except Exception:
            pass

    scored.sort(key=lambda x: -x[0])
    out = []
    used = 0
    for total, bm25, sym, mod, cent, r in scored:
        tok = int(r["token_estimate"] or 1)
        if used + tok > budget:
            continue
        out.append(RankedChunk(
            chunk_id=r["id"],
            rel_path=r["rel_path"],
            chunk_index=int(r["chunk_index"]),
            score=total,
            bm25_component=bm25,
            symbol_bonus=sym,
            module_bonus=mod,
            centrality_bonus=cent,
            token_estimate=tok,
            excerpt=r["content"] or "",
        ))
        used += tok
    return out, False


async def retrieve_two_stage(
    snapshot_id: str,
    query: str,
    section: RetrievalSection,
    budget: int,
) -> TwoStageBundle:
    if not query.strip():
        raise ValueError("Query is required")
    terms = _query_terms(query)
    if not terms:
        raise ValueError("Query must contain searchable terms")

    native = _get_native()
    db = get_db()

    async with db.execute(
        f"SELECT {_CHUNK_FULL_COLS} FROM retrieval_chunks WHERE snapshot_id=?",
        (snapshot_id,),
    ) as cur:
        all_rows = await cur.fetchall()

    if not all_rows:
        raise ValueError("Retrieval index not built for this snapshot")

    async with db.execute(
        "SELECT avgdl, idf_json, k1, b FROM retrieval_bm25_stats WHERE snapshot_id=?",
        (snapshot_id,),
    ) as cur:
        bm25_row = await cur.fetchone()
    scorer = BM25Scorer.from_stats_row(bm25_row)
    if scorer is None:
        logger.debug("[two_stage] BM25 stats not found for %s — using lexical fallback", snapshot_id)

    ctx = await _load_graph_context(snapshot_id)
    symbol_index = await load_symbol_index(snapshot_id)

    stage1 = _stage1_score_rows(all_rows, scorer, terms, top_k=100)

    stage1_files: set[str] = {c.rel_path for c in stage1}
    stage2_expansions: list[StageExpansion] = []
    all_expanded_files: set[str] = set()

    for candidate in stage1[:20]:
        expansion, expanded = _expand_one(candidate, ctx, native)
        stage2_expansions.append(expansion)
        all_expanded_files.update(expanded)

    new_files = all_expanded_files - stage1_files
    row_lookup: dict[str, dict] = {r["id"]: r for r in all_rows}
    file_rows: dict[str, list[dict]] = {}
    for r in all_rows:
        file_rows.setdefault(r["rel_path"], []).append(r)

    expanded_rows: list[dict] = []
    for f in new_files:
        expanded_rows.extend(file_rows.get(f, []))

    stage1_rows: list[dict] = [row_lookup[c.chunk_id] for c in stage1 if c.chunk_id in row_lookup]
    all_candidate_rows = stage1_rows + expanded_rows

    seed_community_ids: set[int] = set()
    for c in stage1[:20]:
        cid = ctx.file_community.get(c.rel_path)
        if cid is not None:
            seed_community_ids.add(cid)

    category_hints = _SECTION_CATEGORY_HINTS.get(section, set())
    all_scored: list[tuple[float, float, float, float, float, dict]] = []
    for r in all_candidate_rows:
        total, bm25, sym, mod, cent = _compute_chunk_score(
            r, terms, scorer, stage1_files, ctx, seed_community_ids, category_hints, symbol_index
        )
        if total > 0:
            all_scored.append((total, bm25, sym, mod, cent, r))

    ranked, used_cpp = _rank_and_budget(all_scored, budget, native)

    used_tokens = sum(c.token_estimate for c in ranked)

    stage3 = TwoStageStage3Result(
        ranked=ranked,
        used_tokens=used_tokens,
        budget_tokens=budget,
        used_cpp_ranker=used_cpp,
    )

    return TwoStageBundle(
        snapshot_id=snapshot_id,
        query=query,
        section=section,
        stage1={"candidates": [c.model_dump() for c in stage1]},
        stage2={"expansions": [e.model_dump() for e in stage2_expansions]},
        stage3=stage3,
    )


async def retrieve_two_stage_as_bundle(
    snapshot_id: str,
    query: str,
    section: RetrievalSection,
    budget: int,
    mode: RetrievalMode = RetrievalMode.HYBRID,
) -> RetrievalBundle:
    """Run the 2-stage pipeline and return a RetrievalBundle (agent-compatible interface)."""
    bundle = await retrieve_two_stage(snapshot_id, query, section, budget)
    evidences = [
        RetrievalEvidence(
            chunk_id=c.chunk_id,
            rel_path=c.rel_path,
            chunk_index=c.chunk_index,
            reason_codes=["2stage-bm25", "graph-expand"] if c.symbol_bonus > 1.0 or c.module_bonus > 1.0 else ["2stage-bm25"],
            score=c.score,
            token_estimate=c.token_estimate,
            excerpt=c.excerpt,
        )
        for c in bundle.stage3.ranked
    ]
    return RetrievalBundle(
        snapshot_id=snapshot_id,
        mode=mode,
        section=section,
        query=query,
        budget_tokens=budget,
        used_tokens=bundle.stage3.used_tokens,
        evidences=evidences,
    )
