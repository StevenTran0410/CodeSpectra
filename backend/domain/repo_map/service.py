"""Repo map and symbol extraction service (RPA-032)."""
import ast
import json
import re
from pathlib import Path
from typing import Any

from infrastructure.db.database import get_db
from shared.errors import NotFoundError
from shared.sql_queries import SQL_SELECT_MANIFEST_FILES_BY_SNAPSHOT
from shared.utils import new_id, read_utf8_lenient, utc_now_iso

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

_Symbol = tuple[str, SymbolKind, int, int, str | None, str | None, ExtractSource]

_RE_PY_CLASS = re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
_RE_PY_FUNC = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)

_RE_JS_CLASS = re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z_$][A-Za-z0-9_$]*)", re.MULTILINE)
_RE_JS_FUNC = re.compile(
    r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\(",
    re.MULTILINE,
)
_RE_JS_INTERFACE = re.compile(
    r"^\s*(?:export\s+)?interface\s+([A-Za-z_$][A-Za-z0-9_$]*)",
    re.MULTILINE,
)
_RE_JS_ENUM = re.compile(r"^\s*(?:export\s+)?enum\s+([A-Za-z_$][A-Za-z0-9_$]*)", re.MULTILINE)
_RE_JS_TYPE = re.compile(r"^\s*(?:export\s+)?type\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=", re.MULTILINE)
_RE_JS_ARROW = re.compile(
    r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*(?:async\s*)?\(",
    re.MULTILINE,
)

_RE_GO_FUNC = re.compile(r"^\s*func\s+(?:\([^)]+\)\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)
_RE_GO_TYPE = re.compile(r"^\s*type\s+([A-Za-z_][A-Za-z0-9_]*)\s+(?:struct|interface)\b", re.MULTILINE)

_RE_JAVA_CLASS = re.compile(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)")
_RE_JAVA_INTERFACE = re.compile(r"\binterface\s+([A-Za-z_][A-Za-z0-9_]*)")
_RE_JAVA_ENUM = re.compile(r"\benum\s+([A-Za-z_][A-Za-z0-9_]*)")
_RE_JAVA_METHOD = re.compile(
    r"^\s*(?:public|private|protected)?\s*(?:static\s+)?[A-Za-z0-9_<>\[\]]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
    re.MULTILINE,
)


class _PythonSymbolVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.symbols: list[_Symbol] = []
        self._class_stack: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self.symbols.append(
            (
                node.name,
                SymbolKind.CLASS,
                int(getattr(node, "lineno", 1)),
                int(getattr(node, "end_lineno", getattr(node, "lineno", 1))),
                None,
                self._class_stack[-1] if self._class_stack else None,
                ExtractSource.AST,
            )
        )
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._visit_function(node, is_async=True)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, is_async: bool) -> None:
        args = [a.arg for a in node.args.args]
        sig_prefix = "async " if is_async else ""
        sig = f"{sig_prefix}{node.name}({', '.join(args)})"
        kind = SymbolKind.METHOD if self._class_stack else SymbolKind.FUNCTION
        parent_name = self._class_stack[-1] if self._class_stack else None
        self.symbols.append(
            (
                node.name,
                kind,
                int(getattr(node, "lineno", 1)),
                int(getattr(node, "end_lineno", getattr(node, "lineno", 1))),
                sig,
                parent_name,
                ExtractSource.AST,
            )
        )
        self.generic_visit(node)


def _line_for_match(content: str, start: int) -> int:
    return content.count("\n", 0, start) + 1


def _extract_python_symbols_ast(content: str) -> list[_Symbol]:
    tree = ast.parse(content)
    visitor = _PythonSymbolVisitor()
    visitor.visit(tree)
    return visitor.symbols


def _extract_lexical_symbols(content: str, language: str | None) -> list[_Symbol]:
    out: list[_Symbol] = []
    low = (language or "").lower()

    def add_from(regex: re.Pattern[str], kind: SymbolKind) -> None:
        for m in regex.finditer(content):
            line = _line_for_match(content, m.start())
            out.append((m.group(1), kind, line, line, None, None, ExtractSource.LEXICAL))

    if low in {"javascript", "typescript"}:
        add_from(_RE_JS_CLASS, SymbolKind.CLASS)
        add_from(_RE_JS_FUNC, SymbolKind.FUNCTION)
        add_from(_RE_JS_INTERFACE, SymbolKind.INTERFACE)
        add_from(_RE_JS_ENUM, SymbolKind.ENUM)
        add_from(_RE_JS_TYPE, SymbolKind.TYPE)
        add_from(_RE_JS_ARROW, SymbolKind.FUNCTION)
        return out

    if low == "go":
        add_from(_RE_GO_TYPE, SymbolKind.TYPE)
        add_from(_RE_GO_FUNC, SymbolKind.FUNCTION)
        return out

    if low == "java":
        add_from(_RE_JAVA_CLASS, SymbolKind.CLASS)
        add_from(_RE_JAVA_INTERFACE, SymbolKind.INTERFACE)
        add_from(_RE_JAVA_ENUM, SymbolKind.ENUM)
        add_from(_RE_JAVA_METHOD, SymbolKind.METHOD)
        return out

    add_from(_RE_PY_CLASS, SymbolKind.CLASS)
    add_from(_RE_PY_FUNC, SymbolKind.FUNCTION)
    return out


def _dedupe_symbols(symbols: list[_Symbol]) -> list[_Symbol]:
    seen: set[tuple[str, str, int, int, str | None]] = set()
    out: list[_Symbol] = []
    for name, kind, line_start, line_end, signature, parent, source in symbols:
        key = (name, kind.value, line_start, line_end, parent)
        if key in seen:
            continue
        seen.add(key)
        out.append((name, kind, line_start, line_end, signature, parent, source))
    return out


def _load_treesitter_lang(name: str) -> Any:
    try:
        from tree_sitter_languages import get_language  # type: ignore

        return get_language(name)
    except Exception:
        return None


def _node_text(content: bytes, node: Any) -> str | None:
    try:
        return content[node.start_byte:node.end_byte].decode("utf-8", errors="ignore").strip()
    except Exception:
        return None


def _extract_ts_js_treesitter(content: str, language: str) -> list[_Symbol]:
    try:
        from tree_sitter import Parser  # type: ignore
    except Exception:
        return []

    lang = _load_treesitter_lang("typescript" if language == "typescript" else "javascript")
    if lang is None:
        return []

    parser = Parser()
    try:
        parser.set_language(lang)  # old API
    except Exception:
        parser.language = lang  # new API

    raw = content.encode("utf-8", errors="ignore")
    tree = parser.parse(raw)
    out: list[_Symbol] = []

    def _line(node: Any) -> tuple[int, int]:
        return int(node.start_point[0]) + 1, int(node.end_point[0]) + 1

    def walk(node: Any, class_name: str | None = None) -> None:
        t = node.type
        if t == "class_declaration":
            name_node = node.child_by_field_name("name")
            name = _node_text(raw, name_node) if name_node else None
            if name:
                s, e = _line(node)
                out.append((name, SymbolKind.CLASS, s, e, None, None, ExtractSource.AST))
                for ch in node.children:
                    walk(ch, class_name=name)
                return
        elif t in {"function_declaration", "generator_function_declaration"}:
            name_node = node.child_by_field_name("name")
            name = _node_text(raw, name_node) if name_node else None
            if name:
                s, e = _line(node)
                out.append((name, SymbolKind.FUNCTION, s, e, None, None, ExtractSource.AST))
        elif t == "method_definition":
            name_node = node.child_by_field_name("name")
            name = _node_text(raw, name_node) if name_node else None
            if name:
                s, e = _line(node)
                out.append((name, SymbolKind.METHOD, s, e, None, class_name, ExtractSource.AST))
        elif t == "interface_declaration":
            name_node = node.child_by_field_name("name")
            name = _node_text(raw, name_node) if name_node else None
            if name:
                s, e = _line(node)
                out.append((name, SymbolKind.INTERFACE, s, e, None, None, ExtractSource.AST))
        elif t == "type_alias_declaration":
            name_node = node.child_by_field_name("name")
            name = _node_text(raw, name_node) if name_node else None
            if name:
                s, e = _line(node)
                out.append((name, SymbolKind.TYPE, s, e, None, None, ExtractSource.AST))
        elif t == "enum_declaration":
            name_node = node.child_by_field_name("name")
            name = _node_text(raw, name_node) if name_node else None
            if name:
                s, e = _line(node)
                out.append((name, SymbolKind.ENUM, s, e, None, None, ExtractSource.AST))

        for ch in node.children:
            walk(ch, class_name=class_name)

    walk(tree.root_node)
    return out


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
                    except Exception:
                        parse_failures += 1
                        symbols = _extract_lexical_symbols(content, language)
                elif language in {"typescript", "javascript"}:
                    symbols = _extract_ts_js_treesitter(content, language)
                    if symbols:
                        used_structural += 1
                    else:
                        symbols = _extract_lexical_symbols(content, language)
                else:
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
                    (id, snapshot_id, rel_path, language, name, kind, line_start, line_end, signature, parent_name, extract_source, created_at)
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
            (snapshot_id, total_symbols, files_indexed, parse_failures, extract_mode, language_breakdown, kind_breakdown, generated_at)
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
        async with get_db().execute("SELECT * FROM repo_maps WHERE snapshot_id=?", (snapshot_id,)) as cur:
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

    async def symbols(self, snapshot_id: str, limit: int = 500, path_prefix: str | None = None) -> SymbolsResponse:
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
                    extract_source=ExtractSource(r["extract_source"] if r["extract_source"] else "lexical"),
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
                    extract_source=ExtractSource(r["extract_source"] if r["extract_source"] else "lexical"),
                )
                for r in rows
            ],
        )

    async def export_csv(self, snapshot_id: str, exclude_tests: bool = True) -> RepoMapCsvResponse:
        async with get_db().execute("SELECT 1 FROM repo_maps WHERE snapshot_id=?", (snapshot_id,)) as cur:
            exists = await cur.fetchone()
        if exists is None:
            raise NotFoundError("RepoMap", snapshot_id)

        async with get_db().execute(
            """
            SELECT DISTINCT rel_path, language, name, kind, line_start, line_end, parent_name, signature, extract_source
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
