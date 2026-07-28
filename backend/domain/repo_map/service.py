"""Repo map and symbol extraction service."""
import json
from pathlib import Path

from domain.retrieval.bm25_scorer import split_identifier
from infrastructure.db.database import get_db
from shared.errors import NotFoundError
from shared.logger import logger
from shared.sql_queries import SQL_SELECT_MANIFEST_FILES_BY_SNAPSHOT
from shared.utils import new_id, read_utf8_lenient, utc_now_iso

from ._loaders import _get_ts_lang, _Symbol, _ts_parse
from ._normalizer import _dedupe_symbols
from ._walkers_config import _walk_cmake, _walk_json, _walk_sql, _walk_toml, _walk_yaml
from ._walkers_functional import _walk_elixir, _walk_haskell, _walk_julia, _walk_ocaml
from ._walkers_js_ts import _walk_js_ts
from ._walkers_jvm import _walk_groovy, _walk_java, _walk_kotlin, _walk_scala
from ._walkers_markup import _walk_css, _walk_html, _walk_markdown, _walk_svelte
from ._walkers_python import _walk_python_ts
from ._walkers_regex import _extract_lexical_symbols, _extract_python_symbols_ast
from ._walkers_scripting import _walk_csharp, _walk_lua, _walk_php, _walk_ruby
from ._walkers_systems import _walk_bash, _walk_c_cpp, _walk_go, _walk_rust, _walk_zig
from .types import (
    BuildRepoMapRequest,
    BuildRepoMapResponse,
    ExtractMode,
    ExtractSource,
    RepoMapCsvResponse,
    RepoMapSummary,
    SymbolKind,
    SymbolRecord,
    SymbolsResponse,
)

def make_qualified_name(rel_path: str, name: str, parent_name: str | None = None) -> str:
    """Standard FQN identity convention: rel_path::name (for class/function) or rel_path::parent_name.name (for method)."""
    p = f"{parent_name}." if parent_name else ""
    return f"{rel_path}::{p}{name}"


# Column list for code_symbols SELECT queries — avoids pulling all columns when
# only a subset is needed. Explicit columns are cheaper and document intent.
_SYMBOL_COLS = (
    "id, snapshot_id, rel_path, language, name, kind, "
    "line_start, line_end, signature, parent_name, extract_source, qualified_name"
)

# Batch size for executemany inserts into code_symbols.
_SYMBOL_BATCH_SIZE = 100

_WALKERS: dict[str, object] = {
    # Core
    "python":     lambda root: _walk_python_ts(root),
    "javascript": lambda root: _walk_js_ts(root, is_ts=False),
    "typescript": lambda root: _walk_js_ts(root, is_ts=True),
    "go":         _walk_go,
    "rust":       _walk_rust,
    "c":          lambda root: _walk_c_cpp(root, is_cpp=False),
    "cpp":        lambda root: _walk_c_cpp(root, is_cpp=True),
    "c++":        lambda root: _walk_c_cpp(root, is_cpp=True),
    "zig":        _walk_zig,
    "bash":       _walk_bash,
    "sh":         _walk_bash,
    # JVM
    "java":    _walk_java,
    "kotlin":  _walk_kotlin,
    "scala":   _walk_scala,
    "groovy":  _walk_groovy,
    # Scripting
    "ruby":   _walk_ruby,
    "php":    _walk_php,
    "lua":    _walk_lua,
    "csharp": _walk_csharp,
    # Functional
    "haskell": _walk_haskell,
    "ocaml":   _walk_ocaml,
    "elixir":  _walk_elixir,
    "julia":   _walk_julia,
    # Config / data
    "yaml":     _walk_yaml,
    "toml":     _walk_toml,
    "json":     _walk_json,
    "cmake":    _walk_cmake,
    "sql":      _walk_sql,
    # Markup / styling
    "html":     _walk_html,
    "css":      _walk_css,
    "markdown": _walk_markdown,
    "svelte":   _walk_svelte,
}


async def _flush_symbol_batch(db, batch: list[tuple]) -> None:
    """Insert a batch of symbol rows using executemany inside a single transaction.

    Each tuple must match the INSERT column order:
    (id, snapshot_id, rel_path, language, name, kind, line_start, line_end,
     signature, parent_name, extract_source, created_at, qualified_name)
    """
    if not batch:
        return
    await db.executemany(
        """
        INSERT INTO code_symbols
        (id, snapshot_id, rel_path, language, name, kind, line_start, line_end,
         signature, parent_name, extract_source, created_at, qualified_name)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        batch,
    )
    await db.commit()


def _extract_symbols_treesitter(content: str, language: str) -> list[_Symbol]:
    low = (language or "").lower()
    walker = _WALKERS.get(low)
    if walker is None:
        return []
    lang_obj = _get_ts_lang(low)
    if lang_obj is None:
        return []
    root = _ts_parse(content, lang_obj)
    if root is None:
        return []
    try:
        return walker(root)  # type: ignore[operator,misc]
    except Exception as exc:
        logger.warning("tree-sitter extraction failed for %s: %s", language, exc)
        return []


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def _symbol_record_from_row(r) -> SymbolRecord:
    """Build a SymbolRecord from a code_symbols row, deriving qualified_name for rows predating that column."""
    return SymbolRecord(
        id=r["id"],
        snapshot_id=r["snapshot_id"],
        rel_path=r["rel_path"],
        language=r["language"],
        name=r["name"],
        kind=SymbolKind(r["kind"]),
        line_start=r["line_start"],
        line_end=r["line_end"],
        signature=r["signature"],
        parent_name=r["parent_name"],
        extract_source=ExtractSource(
            r["extract_source"] if r["extract_source"] else "lexical"
        ),
        qualified_name=r["qualified_name"] if "qualified_name" in r.keys() else make_qualified_name(r["rel_path"], r["name"], r["parent_name"]),
    )


async def search_symbols_cascade(
    db, snapshot_id: str, q: str, limit: int = 120
) -> list[SymbolRecord]:
    """Search code symbols using a cascade: LIKE search -> bounded Levenshtein (<=2) fallback.

    Re-verifies all candidates against code_symbols table for snapshot_id before returning.
    """
    q_str = q.strip()
    if not q_str:
        return []

    like = f"%{q_str}%"
    async with db.execute(
        f"""
        SELECT {_SYMBOL_COLS} FROM code_symbols
        WHERE snapshot_id=? AND (name LIKE ? OR rel_path LIKE ?)
        ORDER BY
          CASE WHEN name = ? THEN 0 WHEN name LIKE ? THEN 1 ELSE 2 END,
          rel_path ASC,
          line_start ASC
        LIMIT ?
        """,
        (snapshot_id, like, like, q_str, f"{q_str}%", limit),
    ) as cur:
        rows = await cur.fetchall()

    if rows:
        return [_symbol_record_from_row(r) for r in rows]

    # Stage 2: Bounded Levenshtein fuzzy fallback (edit distance <= 2)
    q_low = q_str.lower()
    async with db.execute(
        "SELECT DISTINCT name FROM code_symbols WHERE snapshot_id=? AND kind NOT IN ('file', 'module')",
        (snapshot_id,),
    ) as cur:
        name_rows = await cur.fetchall()

    candidate_names: list[tuple[int, str]] = []
    for r in name_rows:
        sym_name = r["name"]
        if not sym_name:
            continue
        dist = levenshtein_distance(q_low, sym_name.lower())
        if dist <= 2:
            candidate_names.append((dist, sym_name))

    if not candidate_names:
        return []

    candidate_names.sort(key=lambda x: (x[0], x[1].lower()))

    fuzzy_records: list[SymbolRecord] = []
    seen_ids: set[str] = set()

    for _dist, sym_name in candidate_names:
        async with db.execute(
            f"SELECT {_SYMBOL_COLS} FROM code_symbols WHERE snapshot_id=? AND LOWER(name)=? LIMIT 10",
            (snapshot_id, sym_name.lower()),
        ) as cur:
            r_rows = await cur.fetchall()
            for r in r_rows:
                rec_id = r["id"]
                if rec_id not in seen_ids:
                    seen_ids.add(rec_id)
                    fuzzy_records.append(_symbol_record_from_row(r))
                if len(fuzzy_records) >= limit:
                    break
        if len(fuzzy_records) >= limit:
            break

    return fuzzy_records[:limit]


class RepoMapService:
    async def build(self, req: BuildRepoMapRequest) -> BuildRepoMapResponse:
        db = get_db()
        async with db.execute("SELECT * FROM repo_snapshots WHERE id=?", (req.snapshot_id,)) as cur:
            snap = await cur.fetchone()
        if snap is None:
            raise NotFoundError("RepoSnapshot", req.snapshot_id)

        root = Path(snap["local_path"])
        if not root.exists():
            raise ValueError("Snapshot path does not exist")

        if req.force_rebuild:
            await db.execute("DELETE FROM code_symbols WHERE snapshot_id=?", (req.snapshot_id,))
        else:
            async with db.execute(
                "SELECT COUNT(*) as c FROM code_symbols WHERE snapshot_id=?",
                (req.snapshot_id,),
            ) as cur:
                row = await cur.fetchone()
            if row and row["c"] > 0:
                logger.info(
                    "[repo_map] snapshot %s already indexed (%d symbols), skipping",
                    req.snapshot_id, row["c"],
                )
                return BuildRepoMapResponse(
                    summary=RepoMapSummary(
                        snapshot_id=req.snapshot_id,
                        total_symbols=int(row["c"]),
                        files_indexed=0,
                        parse_failures=0,
                        extract_mode=ExtractMode.HYBRID,
                        language_breakdown={},
                        kind_breakdown={},
                        generated_at="cached",
                    )
                )

        async with db.execute(
            SQL_SELECT_MANIFEST_FILES_BY_SNAPSHOT,
            (req.snapshot_id,),
        ) as cur:
            files = await cur.fetchall()

        files_indexed = 0
        parse_failures = 0
        total_symbols = 0
        inserted_keys: set[tuple[str, str, str, int, int, str | None]] = set()
        lang_breakdown: dict[str, int] = {}
        kind_breakdown: dict[str, int] = {}
        used_structural = 0
        now = utc_now_iso()

        # Accumulate rows for batch insert; flushed every _SYMBOL_BATCH_SIZE rows.
        pending_batch: list[tuple] = []
        vocab_batch: list[tuple[str, str, str]] = []

        for row in files:
            rel_path = row["rel_path"]
            language = row["language"]
            category = row["category"]
            if category not in {"source", "test", "infra"}:
                continue

            file_path = (root / rel_path).resolve()
            if not file_path.exists() or not file_path.is_file():
                continue
            content = read_utf8_lenient(file_path)
            if not content:
                continue

            files_indexed += 1
            symbols: list[_Symbol]
            try:
                if language == "python":
                    try:
                        symbols = _extract_python_symbols_ast(content)
                        used_structural += 1
                    except SyntaxError:
                        symbols = _extract_symbols_treesitter(content, "python")
                        if symbols:
                            used_structural += 1
                        else:
                            parse_failures += 1
                            symbols = _extract_lexical_symbols(content, language)
                else:
                    symbols = _extract_symbols_treesitter(content, language or "")
                    if symbols:
                        used_structural += 1
                    else:
                        parse_failures += 1
                        symbols = _extract_lexical_symbols(content, language)
            except Exception:
                parse_failures += 1
                continue

            symbols = _dedupe_symbols(symbols)[:2000]
            if not symbols:
                continue

            lang_key = language or "unknown"
            lang_breakdown[lang_key] = lang_breakdown.get(lang_key, 0) + len(symbols)

            for name, kind, line_start, line_end, signature, parent_name, extract_source in symbols:
                dedupe_key = (rel_path, name, kind.value, line_start, line_end, parent_name)
                if dedupe_key in inserted_keys:
                    continue
                inserted_keys.add(dedupe_key)
                qname = make_qualified_name(rel_path, name, parent_name)
                pending_batch.append((
                    new_id(),
                    req.snapshot_id,
                    rel_path,
                    language,
                    name,
                    kind.value,
                    line_start,
                    line_end,
                    signature,
                    parent_name,
                    extract_source.value,
                    now,
                    qname,
                ))
                total_symbols += 1
                kind_breakdown[kind.value] = kind_breakdown.get(kind.value, 0) + 1

                if kind.value not in ("file", "module"):
                    for seg in split_identifier(name):
                        vocab_batch.append((req.snapshot_id, seg, name))

                if len(pending_batch) >= _SYMBOL_BATCH_SIZE:
                    await _flush_symbol_batch(db, pending_batch)
                    pending_batch = []

        # Flush any remaining rows.
        if pending_batch:
            await _flush_symbol_batch(db, pending_batch)

        if vocab_batch:
            try:
                await db.executemany(
                    "INSERT OR IGNORE INTO name_segment_vocab (snapshot_id, segment, name) VALUES (?, ?, ?)",
                    vocab_batch,
                )
                await db.commit()
            except Exception:
                logger.exception("Failed to flush name_segment_vocab batch for %s", req.snapshot_id)

        extract_mode = ExtractMode.HYBRID if used_structural > 0 else ExtractMode.LEXICAL

        await db.execute("DELETE FROM repo_maps WHERE snapshot_id=?", (req.snapshot_id,))
        await db.execute(
            """
            INSERT INTO repo_maps
            (snapshot_id, total_symbols, files_indexed, parse_failures, extract_mode,
             language_breakdown, kind_breakdown, generated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                req.snapshot_id,
                total_symbols,
                files_indexed,
                parse_failures,
                extract_mode.value,
                json.dumps(lang_breakdown),
                json.dumps(kind_breakdown),
                now,
            ),
        )
        await db.commit()

        return BuildRepoMapResponse(
            summary=RepoMapSummary(
                snapshot_id=req.snapshot_id,
                total_symbols=total_symbols,
                files_indexed=files_indexed,
                parse_failures=parse_failures,
                extract_mode=extract_mode,
                language_breakdown=lang_breakdown,
                kind_breakdown=kind_breakdown,
                generated_at=now,
            )
        )

    async def summary(self, snapshot_id: str) -> RepoMapSummary:
        q_maps = "SELECT * FROM repo_maps WHERE snapshot_id=?"
        async with get_db().execute(q_maps, (snapshot_id,)) as cur:
            row = await cur.fetchone()
        if row is None:
            raise NotFoundError("RepoMap", snapshot_id)
        return RepoMapSummary(
            snapshot_id=row["snapshot_id"],
            total_symbols=row["total_symbols"],
            files_indexed=row["files_indexed"],
            parse_failures=row["parse_failures"],
            extract_mode=ExtractMode(row["extract_mode"]),
            language_breakdown=json.loads(row["language_breakdown"] or "{}"),
            kind_breakdown=json.loads(row["kind_breakdown"] or "{}"),
            generated_at=row["generated_at"],
        )

    async def symbols(
        self,
        snapshot_id: str,
        limit: int = 500,
        path_prefix: str | None = None,
    ) -> SymbolsResponse:
        if path_prefix:
            query = f"""
                SELECT {_SYMBOL_COLS} FROM code_symbols
                WHERE snapshot_id=? AND rel_path LIKE ?
                ORDER BY rel_path ASC, line_start ASC
                LIMIT ?
            """
            params = (snapshot_id, f"{path_prefix}%", limit)
        else:
            query = f"""
                SELECT {_SYMBOL_COLS} FROM code_symbols
                WHERE snapshot_id=?
                ORDER BY rel_path ASC, line_start ASC
                LIMIT ?
            """
            params = (snapshot_id, limit)
        async with get_db().execute(query, params) as cur:
            rows = await cur.fetchall()
        return SymbolsResponse(
            snapshot_id=snapshot_id,
            symbols=[
                SymbolRecord(
                    id=r["id"],
                    snapshot_id=r["snapshot_id"],
                    rel_path=r["rel_path"],
                    language=r["language"],
                    name=r["name"],
                    kind=SymbolKind(r["kind"]),
                    line_start=r["line_start"],
                    line_end=r["line_end"],
                    signature=r["signature"],
                    parent_name=r["parent_name"],
                    extract_source=ExtractSource(
                        r["extract_source"] if r["extract_source"] else "lexical"
                    ),
                )
                for r in rows
            ],
        )

    async def search(self, snapshot_id: str, q: str, limit: int = 120) -> SymbolsResponse:
        symbols = await search_symbols_cascade(get_db(), snapshot_id, q, limit=limit)
        return SymbolsResponse(snapshot_id=snapshot_id, symbols=symbols)

    async def export_csv(self, snapshot_id: str, exclude_tests: bool = True) -> RepoMapCsvResponse:
        q_exists = "SELECT 1 FROM repo_maps WHERE snapshot_id=?"
        async with get_db().execute(q_exists, (snapshot_id,)) as cur:
            exists = await cur.fetchone()
        if exists is None:
            raise NotFoundError("RepoMap", snapshot_id)

        async with get_db().execute(
            """
            SELECT DISTINCT rel_path, language, name, kind, line_start, line_end,
                   parent_name, signature, extract_source
            FROM code_symbols
            WHERE snapshot_id=?
            ORDER BY rel_path ASC, line_start ASC, name ASC
            """,
            (snapshot_id,),
        ) as cur:
            rows = await cur.fetchall()

        if exclude_tests:
            def _is_test_path(rel_path: str) -> bool:
                low = rel_path.lower()
                return (
                    "/test/" in low
                    or "/tests/" in low
                    or low.endswith("_test.py")
                    or ".spec." in low
                    or ".test." in low
                )

            rows = [r for r in rows if not _is_test_path(r["rel_path"])]

        def _esc(v: object | None) -> str:
            s = "" if v is None else str(v)
            s = s.replace('"', '""')
            return f'"{s}"'

        header = [
            "snapshot_id",
            "rel_path",
            "language",
            "name",
            "kind",
            "line_start",
            "line_end",
            "parent_name",
            "signature",
            "extract_source",
        ]
        lines = [",".join(header)]
        for r in rows:
            lines.append(
                ",".join(
                    [
                        _esc(snapshot_id),
                        _esc(r["rel_path"]),
                        _esc(r["language"]),
                        _esc(r["name"]),
                        _esc(r["kind"]),
                        _esc(r["line_start"]),
                        _esc(r["line_end"]),
                        _esc(r["parent_name"]),
                        _esc(r["signature"]),
                        _esc(r["extract_source"]),
                    ]
                )
            )

        return RepoMapCsvResponse(
            snapshot_id=snapshot_id,
            row_count=len(rows),
            csv="\n".join(lines) + ("\n" if lines else ""),
        )
