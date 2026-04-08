"""Repo map and symbol extraction service (RPA-032)."""
import json
from pathlib import Path

from infrastructure.db.database import get_db
from shared.errors import NotFoundError
from shared.logger import logger
from shared.sql_queries import SQL_SELECT_MANIFEST_FILES_BY_SNAPSHOT
from shared.utils import new_id, read_utf8_lenient, utc_now_iso

from ._loaders import _get_ts_lang, _Symbol, _ts_parse
from ._normalizer import _dedupe_symbols
from ._walkers_js_ts import _walk_js_ts
from ._walkers_python import _walk_python_ts
from ._walkers_regex import _extract_lexical_symbols, _extract_python_symbols_ast
from ._walkers_systems import _walk_c_cpp, _walk_go, _walk_java, _walk_rust
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

_WALKERS: dict[str, object] = {
    "python": lambda root: _walk_python_ts(root),
    "javascript": lambda root: _walk_js_ts(root, is_ts=False),
    "typescript": lambda root: _walk_js_ts(root, is_ts=True),
    "go": _walk_go,
    "java": _walk_java,
    "rust": _walk_rust,
    "c": lambda root: _walk_c_cpp(root, is_cpp=False),
    "cpp": lambda root: _walk_c_cpp(root, is_cpp=True),
    "c++": lambda root: _walk_c_cpp(root, is_cpp=True),
}


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
                elif language in {"typescript", "javascript", "go", "java", "rust", "c", "cpp"}:
                    symbols = _extract_symbols_treesitter(content, language)
                    if symbols:
                        used_structural += 1
                    else:
                        parse_failures += 1
                        symbols = _extract_lexical_symbols(content, language)
                else:
                    symbols = _extract_symbols_treesitter(content, language or "")
                    if not symbols:
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
                await db.execute(
                    """
                    INSERT INTO code_symbols
                    (id, snapshot_id, rel_path, language, name, kind, line_start, line_end,
                     signature, parent_name, extract_source, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
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
                    ),
                )
                total_symbols += 1
                kind_breakdown[kind.value] = kind_breakdown.get(kind.value, 0) + 1

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
            query = """
                SELECT * FROM code_symbols
                WHERE snapshot_id=? AND rel_path LIKE ?
                ORDER BY rel_path ASC, line_start ASC
                LIMIT ?
            """
            params = (snapshot_id, f"{path_prefix}%", limit)
        else:
            query = """
                SELECT * FROM code_symbols
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
        q = q.strip()
        if not q:
            return SymbolsResponse(snapshot_id=snapshot_id, symbols=[])
        like = f"%{q}%"
        async with get_db().execute(
            """
            SELECT * FROM code_symbols
            WHERE snapshot_id=? AND (name LIKE ? OR rel_path LIKE ?)
            ORDER BY
              CASE WHEN name = ? THEN 0 WHEN name LIKE ? THEN 1 ELSE 2 END,
              rel_path ASC,
              line_start ASC
            LIMIT ?
            """,
            (snapshot_id, like, like, q, f"{q}%", limit),
        ) as cur:
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
