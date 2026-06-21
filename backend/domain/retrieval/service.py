"""Retrieval service (RPA-034): chunking + lexical + hybrid/vectorless retrieval."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from infrastructure.db.database import get_db
from shared.errors import NotFoundError
from shared.logger import logger
from shared.sql_queries import SQL_SELECT_MANIFEST_FILES_BY_SNAPSHOT
from shared.utils import new_id, read_utf8_lenient, utc_now_iso

from .bm25_scorer import BM25Scorer, _query_terms
from .chunker_ast import ASTChunker
from .types import (
    BuildRetrievalIndexRequest,
    BuildRetrievalIndexResponse,
    RetrievalBundle,
    RetrievalCompareResponse,
    RetrievalEvidence,
    RetrievalMode,
    RetrievalSection,
    RetrieveRequest,
    TwoStageBundle,
    TwoStageRequest,
)

_WS = re.compile(r"\s+")
_WORD = re.compile(r"[A-Za-z0-9_]+")

# Languages that get AST-based semantic chunking (CS-101).
_AST_LANGS: frozenset[str] = frozenset({
    "python", "typescript", "javascript", "cpp", "go", "java", "c", "rust",
    "ruby", "php", "csharp", "kotlin", "scala", "bash", "sh", "lua", "zig",
    "haskell", "elixir", "ocaml", "julia",
    "yaml", "toml", "html", "css", "json", "markdown", "groovy", "cmake", "svelte", "sql",
})

# Module-level singleton — parser/Language objects are cached inside.
_ast_chunker = ASTChunker()

_SECTION_BUDGETS: dict[RetrievalSection, int] = {
    # Increased from ~2K to 6-10K.
    # Modern LLMs: 128K-200K context; even local 8B models have 8K+.
    # Cursor sends 8K-20K code tokens per query — our old 2600 was severe under-use.
    # Budget here = tokens reserved for evidence only (system prompt + user preamble add ~800 on top).
    RetrievalSection.ARCHITECTURE: 14_000,
    RetrievalSection.CONVENTIONS: 10_000,
    RetrievalSection.FEATURE_MAP: 14_000,
    RetrievalSection.IMPORTANT_FILES: 12_000,
    RetrievalSection.GLOSSARY: 7_000,
    RetrievalSection.QA: 12_000,
}

_SECTION_CATEGORY_HINTS: dict[RetrievalSection, set[str]] = {
    RetrievalSection.ARCHITECTURE: {"source", "config", "infra"},
    RetrievalSection.CONVENTIONS: {"source", "test", "config"},
    RetrievalSection.FEATURE_MAP: {"source", "docs"},
    RetrievalSection.IMPORTANT_FILES: {"source", "config", "infra"},
    RetrievalSection.GLOSSARY: {"source", "docs"},
    RetrievalSection.QA: {"source", "config", "docs"},
}


# Column lists for retrieval_chunks SELECT queries — avoids loading `content`
# when only metadata is needed (metadata-only queries save 50-250 MB bandwidth
# on large codebases since content can be many KB per row).
_CHUNK_FULL_COLS = "id, snapshot_id, rel_path, language, category, chunk_index, content, token_estimate, chunk_type, start_line, end_line"


async def _batch_insert_chunks(
    db,
    chunk_rows: list[tuple],
    index_rows: list[tuple],
) -> None:
    """Flush accumulated chunk and index rows in a single transaction.

    chunk_rows columns: (id, snapshot_id, rel_path, language, category,
                         chunk_index, content, token_estimate, chunk_type,
                         start_line, end_line, content_hash, created_at)
    index_rows columns: (id, snapshot_id, rel_path, chunk_index,
                         lexical_preview, created_at)
    """
    if not chunk_rows and not index_rows:
        return
    if chunk_rows:
        await db.executemany(
            """
            INSERT INTO retrieval_chunks
            (id, snapshot_id, rel_path, language, category, chunk_index, content,
             token_estimate, chunk_type, start_line, end_line, content_hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            chunk_rows,
        )
    if index_rows:
        await db.executemany(
            """
            INSERT INTO retrieval_indexes
            (id, snapshot_id, rel_path, chunk_index, lexical_preview, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            index_rows,
        )
    await db.commit()


def _normalize_text(s: str) -> str:
    return _WS.sub(" ", s).strip()


def _token_estimate(s: str) -> int:
    # Fast proxy; good enough for budgeting.
    return max(1, len(s) // 4)


def _chunk_size_for(category: str, language: str | None) -> int:
    if category == "docs":
        return 1800
    if category == "config":
        return 1200
    if category == "test":
        return 1400
    if language in {"python", "typescript", "javascript", "ruby", "elixir", "julia"}:
        return 1500
    if language in {"haskell", "ocaml", "erlang"}:
        return 1400
    return 1300


_BRACE_LANGS = {
    "javascript", "typescript", "java", "cpp", "c", "go", "rust",
    "csharp", "kotlin", "scala", "groovy",
    "php", "zig", "dart", "swift",
}


def _ends_mid_function(content: str, language: str | None) -> bool:
    """Return True if the chunk likely ends in the middle of a function/class body.

    For brace languages: unbalanced { > } means we're still inside a block.
    For Python: a def/class whose body indentation never returned to col-0.
    For unknown: fall back to brace counting.
    """
    if not content.strip():
        return False

    if (language or "").lower() in _BRACE_LANGS:
        return content.count("{") > content.count("}")

    if (language or "").lower() == "python":
        lines = content.splitlines()
        non_empty = [l for l in lines if l.strip()]
        if not non_empty:
            return False
        has_def = any(re.match(r"[ \t]*(def |class |async def )", l) for l in lines)
        last = non_empty[-1]
        # If the last line is still indented and the chunk has a def/class, likely mid-body
        return has_def and (last.startswith(" ") or last.startswith("\t"))

    # Default: brace balance
    return content.count("{") > content.count("}")


def _split_chunks(text: str, target_size: int) -> list[str]:
    clean = text.replace("\r\n", "\n")
    if len(clean) <= target_size:
        return [clean]
    out: list[str] = []
    start = 0
    overlap = max(120, target_size // 8)
    while start < len(clean):
        end = min(len(clean), start + target_size)
        out.append(clean[start:end])
        if end >= len(clean):
            break
        start = max(0, end - overlap)
    return out




def _maybe_expand_to_boundary(
    ev: RetrievalEvidence,
    chunk_lookup: dict[tuple[str, int], dict],
) -> RetrievalEvidence:
    """If the chunk ends mid-function, append the next adjacent chunk once.

    Rules:
    - Expand at most ONE hop — we never chain expansions.
    - If the next chunk exists and merging completes (or partially completes) the
      function, we include it. Even if the next chunk starts another function, we
      stop there (don't expand further).
    """
    cur_row = chunk_lookup.get((ev.rel_path, ev.chunk_index))
    language = cur_row["language"] if cur_row is not None else None
    if not _ends_mid_function(ev.excerpt, language):
        return ev

    next_row = chunk_lookup.get((ev.rel_path, ev.chunk_index + 1))
    if next_row is None or not next_row["content"]:
        return ev  # No adjacent chunk available

    merged = ev.excerpt + "\n" + (next_row["content"] or "")
    return RetrievalEvidence(
        chunk_id=ev.chunk_id,
        rel_path=ev.rel_path,
        chunk_index=ev.chunk_index,
        reason_codes=[*ev.reason_codes, "boundary-expanded"],
        score=ev.score,
        token_estimate=_token_estimate(merged),
        excerpt=merged,
    )


def _compute_content_hash(text: str) -> str:
    """Compute MD5 hash of first 4KB of content for cheap change detection (CS-229)."""
    prefix = text[:4096]
    return hashlib.md5(prefix.encode("utf-8", errors="replace")).hexdigest()


async def _apply_incremental_idf_update(
    db,
    snapshot_id: str,
    chunk_hash_map: dict[str, str],
    tokenized_corpus: list[list[str]],
    avgdl: float,
) -> tuple[dict, int]:
    """
    Attempt incremental IDF update by detecting changed chunks via content hash.
    Returns (idf_dict, next_index_version).
    If incremental is feasible (<=10% changes and within drift guard), applies delta.
    Else: full rebuild.

    Args:
        db: database connection
        snapshot_id: snapshot being indexed
        chunk_hash_map: dict[chunk_id] -> new_content_hash
        tokenized_corpus: list of tokenized chunks
        avgdl: average document length
    """
    # Fetch prior BM25 stats
    async with db.execute(
        "SELECT idf_json, index_version FROM retrieval_bm25_stats WHERE snapshot_id=?",
        (snapshot_id,),
    ) as cur:
        prior_row = await cur.fetchone()

    if prior_row is None:
        # No prior index — full rebuild required.
        return BM25Scorer.build_idf(tokenized_corpus, len(tokenized_corpus)), 0

    prior_idf = json.loads(prior_row["idf_json"])
    prior_index_version = prior_row["index_version"] or 0

    # Trigger full rebuild every 50 incremental updates to prevent silent IDF drift.
    if prior_index_version > 0 and prior_index_version % 50 == 0:
        logger.info(
            "[build_index] Forcing full rebuild at index_version=%d to prevent IDF drift",
            prior_index_version,
        )
        return BM25Scorer.build_idf(tokenized_corpus, len(tokenized_corpus)), prior_index_version + 1

    # Fetch old hashes and detect changed chunks
    async with db.execute(
        "SELECT id, content_hash FROM retrieval_chunks WHERE snapshot_id=? ORDER BY id",
        (snapshot_id,),
    ) as cur:
        old_rows = await cur.fetchall()

    old_hashes = {row["id"]: row["content_hash"] for row in old_rows}
    changed_chunk_ids = set()
    for chunk_id, new_hash in chunk_hash_map.items():
        old_hash = old_hashes.get(chunk_id)
        if old_hash != new_hash:
            changed_chunk_ids.add(chunk_id)

    # Check if incremental update is feasible
    total_chunks = len(old_hashes)
    if total_chunks == 0:
        return BM25Scorer.build_idf(tokenized_corpus, len(tokenized_corpus)), prior_index_version + 1

    change_ratio = len(changed_chunk_ids) / total_chunks
    if change_ratio >= 0.10:
        logger.info(
            "[build_index] Change ratio %.1f%% >= 10%% threshold; full rebuild",
            change_ratio * 100,
        )
        return BM25Scorer.build_idf(tokenized_corpus, len(tokenized_corpus)), prior_index_version + 1

    # Incremental IDF: only reuse prior IDF if no chunks changed (rare case)
    logger.info(
        "[build_index] Incremental path: %.1f%% changed (%d/%d chunks)",
        change_ratio * 100, len(changed_chunk_ids), total_chunks,
    )

    if not changed_chunk_ids:
        # No changed chunks — reuse prior IDF (optimization for fast re-indexes with zero changes)
        new_idf = prior_idf
        logger.info("[build_index] No changed chunks; reusing prior IDF")
    else:
        # Some chunks changed — full rebuild is safer than tracking deltas
        # (delta tracking requires matching old→new chunk positions, which is error-prone)
        new_idf = BM25Scorer.build_idf(tokenized_corpus, len(tokenized_corpus))
        logger.info("[build_index] Chunks changed; rebuilding IDF from corpus")

    return new_idf, prior_index_version + 1


class RetrievalService:
    async def build_index(self, req: BuildRetrievalIndexRequest) -> BuildRetrievalIndexResponse:
        db = get_db()
        async with db.execute("SELECT local_path FROM repo_snapshots WHERE id=?", (req.snapshot_id,)) as cur:
            snap = await cur.fetchone()
        if snap is None:
            raise NotFoundError("RepoSnapshot", req.snapshot_id)
        root = Path(snap["local_path"])
        if not root.exists():
            raise ValueError("Snapshot path does not exist")

        if req.force_rebuild:
            await db.execute("DELETE FROM retrieval_chunks WHERE snapshot_id=?", (req.snapshot_id,))
            await db.execute("DELETE FROM retrieval_indexes WHERE snapshot_id=?", (req.snapshot_id,))
            await db.execute("DELETE FROM retrieval_bm25_stats WHERE snapshot_id=?", (req.snapshot_id,))
        else:
            async with db.execute(
                "SELECT COUNT(*) as c FROM retrieval_chunks WHERE snapshot_id=?",
                (req.snapshot_id,),
            ) as cur:
                row = await cur.fetchone()
            if row and row["c"] > 0:
                # Chunks exist — also check BM25 stats are present (may be missing for old snapshots).
                async with db.execute(
                    "SELECT 1 FROM retrieval_bm25_stats WHERE snapshot_id=?",
                    (req.snapshot_id,),
                ) as cur:
                    bm25_exists = await cur.fetchone()
                if bm25_exists:
                    logger.info(
                        "[retrieval] snapshot %s already indexed (%d chunks), skipping",
                        req.snapshot_id, row["c"],
                    )
                    return BuildRetrievalIndexResponse(
                        snapshot_id=req.snapshot_id,
                        chunk_count=int(row["c"]),
                        files_indexed=0,
                        generated_at="cached",
                    )
                # BM25 stats missing — rebuild from existing chunks without re-chunking.
                logger.info(
                    "[retrieval] snapshot %s has %d chunks but no BM25 stats — rebuilding stats only",
                    req.snapshot_id, row["c"],
                )
                async with db.execute(
                    "SELECT content FROM retrieval_chunks WHERE snapshot_id=?",
                    (req.snapshot_id,),
                ) as cur:
                    chunk_rows = await cur.fetchall()
                tokenized_corpus = [_WORD.findall(r["content"].lower()) for r in chunk_rows]
                if tokenized_corpus:
                    avgdl = sum(len(t) for t in tokenized_corpus) / len(tokenized_corpus)
                    idf = BM25Scorer.build_idf(tokenized_corpus, len(tokenized_corpus))
                    now_str = datetime.utcnow().isoformat()
                    await db.execute(
                        """INSERT OR REPLACE INTO retrieval_bm25_stats
                           (snapshot_id, chunk_count, avgdl, idf_json, k1, b, generated_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (req.snapshot_id, len(tokenized_corpus), avgdl, json.dumps(idf), 2.0, 0.75, now_str),
                    )
                    await db.commit()
                    logger.info(
                        "BM25 stats rebuilt: %d docs, avgdl=%.1f, vocab=%d",
                        len(tokenized_corpus), avgdl, len(idf),
                    )
                return BuildRetrievalIndexResponse(
                    snapshot_id=req.snapshot_id,
                    chunk_count=int(row["c"]),
                    files_indexed=0,
                    generated_at="rebuilt-bm25",
                )

        async with db.execute(SQL_SELECT_MANIFEST_FILES_BY_SNAPSHOT, (req.snapshot_id,)) as cur:
            files = await cur.fetchall()

        now = utc_now_iso()
        files_indexed = 0
        chunk_count = 0
        tokenized_corpus: list[list[str]] = []
        chunk_hash_map: dict[str, str] = {}  # chunk_id -> content_hash (CS-229)

        # Accumulate rows per file; flush together after each file's chunks are ready.
        # Using the file loop as the natural batch boundary: rows per file are bounded
        # and flushing per-file avoids holding a very large in-memory list.
        for f in files:
            rel_path = f["rel_path"]
            language = f["language"]
            category = f["category"]
            if category in {"generated", "asset", "secret-risk", "other"}:
                continue
            p = root / rel_path
            if not p.exists() or not p.is_file():
                continue
            text = read_utf8_lenient(p)
            if not text.strip():
                continue
            target_size = _chunk_size_for(category, language)
            lang_key = (language or "").lower()
            if lang_key in _AST_LANGS and category not in {"docs", "config"}:
                # AST chunking: operate on raw source — normalization after.
                ast_chunks = _ast_chunker.chunk(text, lang_key, target_size)
                ast_pieces = [(c, _normalize_text(c.text)) for c in ast_chunks]
            else:
                ast_pieces = [(None, piece) for piece in _split_chunks(_normalize_text(text), target_size)]
            files_indexed += 1

            file_chunk_rows: list[tuple] = []
            file_index_rows: list[tuple] = []
            file_token_rows: list[tuple] = []
            for i, (ast_chunk, piece) in enumerate(ast_pieces):
                token_est = _token_estimate(piece)
                preview = piece[:500]
                # Extract chunk_type and line range from AST chunk if available
                chunk_type = ast_chunk.chunk_type if ast_chunk else "block"
                start_line = ast_chunk.start_line if ast_chunk else 0
                end_line = ast_chunk.end_line if ast_chunk else 0
                # Collect tokenized document for BM25 IDF computation
                tokens = _WORD.findall(piece.lower())
                tokenized_corpus.append(tokens)
                chunk_id = new_id()
                content_hash = _compute_content_hash(piece)
                chunk_hash_map[chunk_id] = content_hash  # Track for incremental IDF (CS-229)
                file_chunk_rows.append((
                    chunk_id,
                    req.snapshot_id,
                    rel_path,
                    language,
                    category,
                    i,
                    piece,
                    token_est,
                    chunk_type,
                    start_line,
                    end_line,
                    content_hash,
                    now,
                ))
                # Minimal lexical index row (prefix preview for quick debug/search metadata).
                file_index_rows.append((
                    new_id(), req.snapshot_id, rel_path, i, preview, now,
                ))
                # Token frequency rows for incremental IDF updates (CS-229).
                token_freq = Counter(tokens)
                for term, tf in token_freq.items():
                    file_token_rows.append((chunk_id, term, int(tf)))
                chunk_count += 1

            await _batch_insert_chunks(db, file_chunk_rows, file_index_rows)

            # Batch insert token frequencies (CS-229).
            if file_token_rows:
                await db.executemany(
                    """INSERT OR IGNORE INTO retrieval_chunk_tokens
                       (chunk_id, term, tf) VALUES (?, ?, ?)""",
                    file_token_rows,
                )

        # Compute and store BM25 IDF stats with incremental tracking (CS-229).
        if tokenized_corpus:
            avgdl = sum(len(t) for t in tokenized_corpus) / len(tokenized_corpus)
            idf, next_index_version = await _apply_incremental_idf_update(
                db, req.snapshot_id, chunk_hash_map, tokenized_corpus, avgdl
            )
            now_str = datetime.utcnow().isoformat()
            await db.execute(
                """INSERT OR REPLACE INTO retrieval_bm25_stats
                   (snapshot_id, chunk_count, avgdl, idf_json, k1, b, index_version, generated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (req.snapshot_id, len(tokenized_corpus), avgdl, json.dumps(idf), 2.0, 0.75, next_index_version, now_str),
            )
            logger.info("BM25 stats built: %d docs, avgdl=%.1f, vocab=%d, index_version=%d",
                       len(tokenized_corpus), avgdl, len(idf), next_index_version)

        await db.commit()
        return BuildRetrievalIndexResponse(
            snapshot_id=req.snapshot_id,
            chunk_count=chunk_count,
            files_indexed=files_indexed,
            generated_at=now,
        )

    async def retrieve(self, req: RetrieveRequest) -> RetrievalBundle:
        if not req.query.strip():
            raise ValueError("Query is required")
        terms = _query_terms(req.query)
        if not terms:
            raise ValueError("Query must contain searchable terms")

        budget = _SECTION_BUDGETS[req.section]

        # Delegate to 2-stage pipeline (CS-203); fall back to legacy single-pass on error.
        try:
            from .two_stage_retrieval import retrieve_two_stage_as_bundle
            return await retrieve_two_stage_as_bundle(
                snapshot_id=req.snapshot_id,
                query=req.query,
                section=req.section,
                budget=budget,
                mode=req.mode,
                min_confidence=req.min_confidence,
            )
        except Exception:
            logger.warning(
                "[retrieve] 2-stage pipeline failed for snapshot=%s query=%r — using fallback single-pass",
                req.snapshot_id, req.query[:60], exc_info=True,
            )

        logger.info("[retrieve] using fallback single-pass for snapshot=%s", req.snapshot_id)
        category_hints = _SECTION_CATEGORY_HINTS[req.section]

        async with get_db().execute(
            f"""
            SELECT {_CHUNK_FULL_COLS}
            FROM retrieval_chunks
            WHERE snapshot_id=?
            """,
            (req.snapshot_id,),
        ) as cur:
            rows = await cur.fetchall()

        if not rows:
            raise ValueError("Retrieval index not built for this snapshot")

        # Gather graph hints.
        async with get_db().execute(
            "SELECT top_central_files FROM structural_graph_summaries WHERE snapshot_id=?",
            (req.snapshot_id,),
        ) as cur:
            graph_summary = await cur.fetchone()
        central_rank: dict[str, int] = {}
        if graph_summary and graph_summary["top_central_files"]:
            try:
                arr = json.loads(graph_summary["top_central_files"])
            except Exception:
                arr = []
            for i, item in enumerate(arr):
                rp = item.get("rel_path")
                if isinstance(rp, str):
                    central_rank[rp] = i + 1

        # Load BM25 scorer
        bm25_row = None
        async with get_db().execute(
            "SELECT avgdl, idf_json, k1, b FROM retrieval_bm25_stats WHERE snapshot_id=?",
            (req.snapshot_id,),
        ) as cur:
            bm25_row = await cur.fetchone()
        scorer = BM25Scorer.from_stats_row(bm25_row)
        if scorer is None:
            logger.warning("BM25 stats not found for snapshot %s — using lexical fallback", req.snapshot_id)

        scored: list[tuple[float, RetrievalEvidence]] = []
        for r in rows:
            rel_path = r["rel_path"]
            cat = r["category"]
            content = r["content"] or ""
            path_low = rel_path.lower()

            if scorer is not None:
                raw = scorer.score(terms, content.lower(), path_low)
                if raw <= 0.0:
                    continue
                reason_codes: list[str] = ["bm25-hit"]
                score = raw
            else:
                # Cold-start fallback: original lexical hit counting
                low = content.lower()
                lexical_hits = 0
                for t in terms:
                    lexical_hits += low.count(t)
                    if t in path_low:
                        lexical_hits += 2
                if lexical_hits <= 0:
                    continue
                reason_codes = ["lexical-hit"]
                score = float(lexical_hits)

            if cat in category_hints:
                score += 1.4
                reason_codes.append("section-category-match")

            if rel_path in central_rank:
                # Better rank => bigger bonus.
                score += max(0.0, 2.6 - (central_rank[rel_path] * 0.08))
                reason_codes.append("graph-centrality-hint")

            if req.mode == RetrievalMode.VECTORLESS:
                # Vectorless path favors graph-shape + path semantics.
                if rel_path in central_rank:
                    score += 1.8
                if "index" in path_low or "router" in path_low or "service" in path_low:
                    score += 0.9
                reason_codes.append("vectorless-graph-prior")
            else:
                # Hybrid path: symbol-ish hints by token overlap.
                symbolish = sum(1 for t in terms if t in path_low)
                if symbolish > 0:
                    score += 0.6 * symbolish
                    reason_codes.append("symbol-path-hint")

            ev = RetrievalEvidence(
                chunk_id=r["id"],
                rel_path=rel_path,
                chunk_index=r["chunk_index"],
                reason_codes=reason_codes,
                score=score,
                token_estimate=int(r["token_estimate"] or _token_estimate(content)),
                excerpt=content,  # full chunk content — render_bundle truncates for LLM prompt
            )
            scored.append((score, ev))

        scored.sort(key=lambda x: (-x[0], x[1].rel_path, x[1].chunk_index))

        # Build a lookup for adjacent-chunk expansion: (rel_path, chunk_index) -> row
        chunk_lookup: dict[tuple[str, int], dict] = {
            (r["rel_path"], r["chunk_index"]): r for r in rows
        }

        used = 0
        picked: list[RetrievalEvidence] = []
        limit = max(1, min(req.max_results, 80))
        for _, ev in scored:
            if len(picked) >= limit:
                break
            if used + ev.token_estimate > budget:
                continue

            # Expand chunk to function boundary if needed (at most one hop).
            ev = _maybe_expand_to_boundary(ev, chunk_lookup)

            # Re-check budget after potential expansion
            if used + ev.token_estimate > budget:
                continue

            picked.append(ev)
            used += ev.token_estimate

        return RetrievalBundle(
            snapshot_id=req.snapshot_id,
            mode=req.mode,
            section=req.section,
            query=req.query,
            budget_tokens=budget,
            used_tokens=used,
            evidences=picked,
        )

    async def retrieve_two_stage(self, req: TwoStageRequest) -> TwoStageBundle:
        from .two_stage_retrieval import retrieve_two_stage as _run
        budget = req.budget or _SECTION_BUDGETS.get(req.section, 10_000)
        return await _run(
            snapshot_id=req.snapshot_id,
            query=req.query,
            section=req.section,
            budget=budget,
            min_confidence=req.min_confidence,
        )

    async def compare(self, req: RetrieveRequest) -> RetrievalCompareResponse:
        base_req = RetrieveRequest(
            snapshot_id=req.snapshot_id,
            query=req.query,
            section=req.section,
            mode=RetrievalMode.HYBRID,
            max_results=req.max_results,
        )
        vec_req = RetrieveRequest(
            snapshot_id=req.snapshot_id,
            query=req.query,
            section=req.section,
            mode=RetrievalMode.VECTORLESS,
            max_results=req.max_results,
        )
        baseline = await self.retrieve(base_req)
        vectorless = await self.retrieve(vec_req)

        # Simple comparable metrics for A/B logging.
        def _precision_at_5(bundle: RetrievalBundle) -> float:
            top = bundle.evidences[:5]
            if not top:
                return 0.0
            good = sum(1 for e in top if "section-category-match" in e.reason_codes)
            return float(good) / float(len(top))

        def _evidence_hit_rate(bundle: RetrievalBundle) -> float:
            if not bundle.evidences:
                return 0.0
            with_query = 0
            q_terms = _query_terms(bundle.query)
            for e in bundle.evidences:
                low = e.excerpt.lower()
                if any(t in low for t in q_terms):
                    with_query += 1
            return float(with_query) / float(len(bundle.evidences))

        return RetrievalCompareResponse(
            snapshot_id=req.snapshot_id,
            section=req.section,
            query=req.query,
            baseline=baseline,
            vectorless=vectorless,
            precision_at_5_delta=_precision_at_5(vectorless) - _precision_at_5(baseline),
            evidence_hit_rate_delta=_evidence_hit_rate(vectorless) - _evidence_hit_rate(baseline),
            token_cost_delta=vectorless.used_tokens - baseline.used_tokens,
        )
