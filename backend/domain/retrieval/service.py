"""Retrieval service (RPA-034): chunking + lexical + hybrid/vectorless retrieval."""
from __future__ import annotations

import json
import re
from pathlib import Path

from infrastructure.db.database import get_db
from shared.errors import NotFoundError
from shared.logger import logger
from shared.sql_queries import SQL_SELECT_MANIFEST_FILES_BY_SNAPSHOT
from shared.utils import new_id, read_utf8_lenient, utc_now_iso

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
)

_WS = re.compile(r"\s+")
_WORD = re.compile(r"[A-Za-z0-9_]+")

# Languages that get AST-based semantic chunking (CS-101).
_AST_LANGS: frozenset[str] = frozenset({
    "python", "typescript", "javascript", "cpp", "go", "java", "c", "rust",
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
}

_SECTION_CATEGORY_HINTS: dict[RetrievalSection, set[str]] = {
    RetrievalSection.ARCHITECTURE: {"source", "config", "infra"},
    RetrievalSection.CONVENTIONS: {"source", "test", "config"},
    RetrievalSection.FEATURE_MAP: {"source", "docs"},
    RetrievalSection.IMPORTANT_FILES: {"source", "config", "infra"},
    RetrievalSection.GLOSSARY: {"source", "docs"},
}


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
    if language in {"python", "typescript", "javascript"}:
        return 1500
    return 1300


_BRACE_LANGS = {"javascript", "typescript", "java", "cpp", "c", "go", "rust", "csharp", "kotlin", "swift", "scala"}


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


def _query_terms(q: str) -> list[str]:
    terms = [w.lower() for w in _WORD.findall(q) if len(w) > 1]
    # Keep distinct while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for t in terms:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
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
        else:
            async with db.execute(
                "SELECT COUNT(*) as c FROM retrieval_chunks WHERE snapshot_id=?",
                (req.snapshot_id,),
            ) as cur:
                row = await cur.fetchone()
            if row and row["c"] > 0:
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

        async with db.execute(SQL_SELECT_MANIFEST_FILES_BY_SNAPSHOT, (req.snapshot_id,)) as cur:
            files = await cur.fetchall()

        now = utc_now_iso()
        files_indexed = 0
        chunk_count = 0
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
                pieces = [_normalize_text(c.text) for c in ast_chunks]
            else:
                pieces = _split_chunks(_normalize_text(text), target_size)
            files_indexed += 1
            for i, piece in enumerate(pieces):
                token_est = _token_estimate(piece)
                preview = piece[:500]
                await db.execute(
                    """
                    INSERT INTO retrieval_chunks
                    (id, snapshot_id, rel_path, language, category, chunk_index, content, token_estimate, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        new_id(),
                        req.snapshot_id,
                        rel_path,
                        language,
                        category,
                        i,
                        piece,
                        token_est,
                        now,
                    ),
                )
                chunk_count += 1

                # Minimal lexical index row (prefix preview for quick debug/search metadata).
                await db.execute(
                    """
                    INSERT INTO retrieval_indexes
                    (id, snapshot_id, rel_path, chunk_index, lexical_preview, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (new_id(), req.snapshot_id, rel_path, i, preview, now),
                )

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
        category_hints = _SECTION_CATEGORY_HINTS[req.section]

        async with get_db().execute(
            """
            SELECT id, rel_path, chunk_index, language, category, content, token_estimate
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

        scored: list[tuple[float, RetrievalEvidence]] = []
        for r in rows:
            rel_path = r["rel_path"]
            cat = r["category"]
            content = r["content"] or ""
            low = content.lower()
            path_low = rel_path.lower()

            lexical_hits = 0
            for t in terms:
                lexical_hits += low.count(t)
                if t in path_low:
                    lexical_hits += 2
            if lexical_hits <= 0:
                continue

            reason_codes: list[str] = ["lexical-hit"]
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
