"""Repo map and symbol extraction service (RPA-032)."""
import ast
import json
import re
from pathlib import Path
from typing import Any

from infrastructure.db.database import get_db
from shared.errors import NotFoundError
from shared.logger import logger
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

# ── Tree-sitter language loader (new API: tree-sitter >= 0.22) ────────────────

def _load_ts_language(name: str) -> Any | None:
    """Load a tree-sitter Language object from the individual language packages."""
    try:
        from tree_sitter import Language
        if name == "python":
            import tree_sitter_python as m
        elif name in ("javascript", "js"):
            import tree_sitter_javascript as m  # type: ignore[no-redef]
        elif name == "typescript":
            import tree_sitter_typescript as m  # type: ignore[no-redef]
            return Language(m.language_typescript())
        elif name == "go":
            import tree_sitter_go as m  # type: ignore[no-redef]
        elif name == "java":
            import tree_sitter_java as m  # type: ignore[no-redef]
        elif name == "rust":
            import tree_sitter_rust as m  # type: ignore[no-redef]
        elif name == "c":
            import tree_sitter_c as m  # type: ignore[no-redef]
        elif name in ("cpp", "c++"):
            import tree_sitter_cpp as m  # type: ignore[no-redef]
        else:
            return None
        return Language(m.language())
    except Exception:
        return None


def _ts_parse(content: str, lang: Any) -> Any | None:
    """Parse source text with tree-sitter, return root Node or None."""
    try:
        from tree_sitter import Parser
        p = Parser(lang)
        tree = p.parse(content.encode("utf-8", errors="ignore"))
        return tree.root_node
    except Exception:
        return None


def _node_name(node: Any) -> str | None:
    """Extract the text of the 'name' field child, decoded as str."""
    try:
        n = node.child_by_field_name("name")
        if n and n.text:
            return n.text.decode("utf-8", errors="ignore").strip()
    except Exception:
        pass
    return None


def _lines(node: Any) -> tuple[int, int]:
    return node.start_point[0] + 1, node.end_point[0] + 1


# ── Language-specific tree-sitter walkers ─────────────────────────────────────

def _walk_python_ts(root: Any) -> list[_Symbol]:
    """Walk tree-sitter Python AST (fallback only; prefer stdlib ast module)."""
    out: list[_Symbol] = []

    def walk(node: Any, class_stack: list[str]) -> None:
        if node.type == "decorated_definition":
            for ch in node.children:
                if ch.type in ("function_definition", "class_definition"):
                    walk(ch, class_stack)
            return
        if node.type == "function_definition":
            name = _node_name(node)
            if name:
                s, e = _lines(node)
                kind = SymbolKind.METHOD if class_stack else SymbolKind.FUNCTION
                out.append((name, kind, s, e, None, class_stack[-1] if class_stack else None, ExtractSource.AST))
            for ch in node.children:
                walk(ch, class_stack)
        elif node.type == "class_definition":
            name = _node_name(node)
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.CLASS, s, e, None, class_stack[-1] if class_stack else None, ExtractSource.AST))
                new_stack = [*class_stack, name]
                for ch in node.children:
                    walk(ch, new_stack)
        else:
            for ch in node.children:
                walk(ch, class_stack)

    walk(root, [])
    return out


def _walk_js_ts(root: Any, is_ts: bool = False) -> list[_Symbol]:
    out: list[_Symbol] = []

    _DEF_NODES = {
        "function_declaration": SymbolKind.FUNCTION,
        "generator_function_declaration": SymbolKind.FUNCTION,
        "function_expression": SymbolKind.FUNCTION,
    }
    _CLASS_NODES = {"class_declaration", "class"}
    _IFACE_NODES = {"interface_declaration"} if is_ts else set()
    _TYPE_NODES = {"type_alias_declaration"} if is_ts else set()
    _ENUM_NODES = {"enum_declaration"}
    _METHOD_NODES = {"method_definition", "method_signature"}

    def walk(node: Any, class_stack: list[str]) -> None:
        t = node.type
        if t in _DEF_NODES:
            name = _node_name(node)
            if name:
                s, e = _lines(node)
                out.append((name, _DEF_NODES[t], s, e, None, class_stack[-1] if class_stack else None, ExtractSource.AST))
        elif t in _CLASS_NODES:
            name = _node_name(node)
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.CLASS, s, e, None, None, ExtractSource.AST))
                for ch in node.children:
                    walk(ch, [*class_stack, name])
                return
        elif t in _IFACE_NODES:
            name = _node_name(node)
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.INTERFACE, s, e, None, None, ExtractSource.AST))
        elif t in _TYPE_NODES:
            name = _node_name(node)
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.TYPE, s, e, None, None, ExtractSource.AST))
        elif t in _ENUM_NODES:
            name = _node_name(node)
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.ENUM, s, e, None, None, ExtractSource.AST))
        elif t in _METHOD_NODES:
            name = _node_name(node)
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.METHOD, s, e, None, class_stack[-1] if class_stack else None, ExtractSource.AST))
        elif t == "lexical_declaration":
            # Arrow functions: const foo = async (x) => ...
            for ch in node.children:
                if ch.type == "variable_declarator":
                    var_name_node = ch.child_by_field_name("name")
                    val = ch.child_by_field_name("value")
                    if var_name_node and val and val.type in ("arrow_function", "function_expression"):
                        try:
                            vname = var_name_node.text.decode("utf-8", errors="ignore").strip()
                            if vname:
                                s, e = _lines(ch)
                                out.append((vname, SymbolKind.FUNCTION, s, e, None, class_stack[-1] if class_stack else None, ExtractSource.AST))
                        except Exception:
                            pass
        for ch in node.children:
            walk(ch, class_stack)

    walk(root, [])
    return out


def _walk_go(root: Any) -> list[_Symbol]:
    out: list[_Symbol] = []
    for node in root.children:
        if node.type == "function_declaration":
            name = _node_name(node)
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.FUNCTION, s, e, None, None, ExtractSource.AST))
        elif node.type == "method_declaration":
            name_node = node.child_by_field_name("name")
            recv = node.child_by_field_name("receiver")
            recv_type = None
            if recv:
                # Extract receiver type name
                for ch in recv.children:
                    if ch.type == "parameter_declaration":
                        for tc in ch.children:
                            if tc.type in ("type_identifier", "pointer_type"):
                                try:
                                    recv_type = tc.text.decode("utf-8", errors="ignore").strip().lstrip("*")
                                except Exception:
                                    pass
            if name_node:
                try:
                    name = name_node.text.decode("utf-8", errors="ignore").strip()
                    s, e = _lines(node)
                    out.append((name, SymbolKind.METHOD, s, e, None, recv_type, ExtractSource.AST))
                except Exception:
                    pass
        elif node.type == "type_declaration":
            for ch in node.children:
                if ch.type == "type_spec":
                    name = _node_name(ch)
                    if not name:
                        continue
                    body_type = ch.child_by_field_name("type")
                    s, e = _lines(ch)
                    if body_type and body_type.type == "struct_type":
                        out.append((name, SymbolKind.CLASS, s, e, None, None, ExtractSource.AST))
                    elif body_type and body_type.type == "interface_type":
                        out.append((name, SymbolKind.INTERFACE, s, e, None, None, ExtractSource.AST))
                    else:
                        out.append((name, SymbolKind.TYPE, s, e, None, None, ExtractSource.AST))
    return out


def _walk_java(root: Any) -> list[_Symbol]:
    out: list[_Symbol] = []

    def walk(node: Any, class_stack: list[str]) -> None:
        t = node.type
        if t == "class_declaration":
            name = _node_name(node)
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.CLASS, s, e, None, class_stack[-1] if class_stack else None, ExtractSource.AST))
                for ch in node.children:
                    walk(ch, [*class_stack, name])
                return
        elif t == "interface_declaration":
            name = _node_name(node)
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.INTERFACE, s, e, None, None, ExtractSource.AST))
        elif t == "enum_declaration":
            name = _node_name(node)
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.ENUM, s, e, None, None, ExtractSource.AST))
        elif t in ("method_declaration", "constructor_declaration"):
            name = _node_name(node)
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.METHOD, s, e, None, class_stack[-1] if class_stack else None, ExtractSource.AST))
        for ch in node.children:
            walk(ch, class_stack)

    walk(root, [])
    return out


def _walk_rust(root: Any) -> list[_Symbol]:
    out: list[_Symbol] = []

    def walk(node: Any, impl_type: str | None = None) -> None:
        t = node.type
        if t == "function_item":
            name = _node_name(node)
            if name:
                s, e = _lines(node)
                kind = SymbolKind.METHOD if impl_type else SymbolKind.FUNCTION
                out.append((name, kind, s, e, None, impl_type, ExtractSource.AST))
        elif t == "struct_item":
            name = _node_name(node)
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.CLASS, s, e, None, None, ExtractSource.AST))
        elif t == "enum_item":
            name = _node_name(node)
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.ENUM, s, e, None, None, ExtractSource.AST))
        elif t == "trait_item":
            name = _node_name(node)
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.INTERFACE, s, e, None, None, ExtractSource.AST))
        elif t == "impl_item":
            type_node = node.child_by_field_name("type")
            try:
                impl_name = type_node.text.decode("utf-8", errors="ignore").strip() if type_node else None
            except Exception:
                impl_name = None
            for ch in node.children:
                walk(ch, impl_type=impl_name)
            return
        elif t == "type_item":
            name = _node_name(node)
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.TYPE, s, e, None, None, ExtractSource.AST))
        for ch in node.children:
            walk(ch, impl_type)

    walk(root)
    return out


def _walk_c_cpp(root: Any, is_cpp: bool = False) -> list[_Symbol]:
    out: list[_Symbol] = []

    def walk(node: Any) -> None:
        t = node.type
        if t == "function_definition":
            # C/C++ function_definition has a declarator field
            decl = node.child_by_field_name("declarator")
            name = _resolve_c_declarator_name(decl)
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.FUNCTION, s, e, None, None, ExtractSource.AST))
        elif t in ("struct_specifier", "union_specifier"):
            name = _node_name(node)
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.CLASS, s, e, None, None, ExtractSource.AST))
        elif is_cpp and t == "class_specifier":
            name = _node_name(node)
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.CLASS, s, e, None, None, ExtractSource.AST))
                for ch in node.children:
                    walk(ch)
                return
        elif t == "enum_specifier":
            name = _node_name(node)
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.ENUM, s, e, None, None, ExtractSource.AST))
        elif t == "type_definition":
            for ch in node.children:
                if ch.type == "type_identifier":
                    try:
                        tname = ch.text.decode("utf-8", errors="ignore").strip()
                        if tname:
                            s, e = _lines(node)
                            out.append((tname, SymbolKind.TYPE, s, e, None, None, ExtractSource.AST))
                    except Exception:
                        pass
                    break
        for ch in node.children:
            walk(ch)

    walk(root)
    return out


def _resolve_c_declarator_name(node: Any) -> str | None:
    """Recursively unwrap nested declarators to find the innermost identifier name."""
    if node is None:
        return None
    t = node.type
    if t == "identifier":
        try:
            return node.text.decode("utf-8", errors="ignore").strip()
        except Exception:
            return None
    if t in ("function_declarator", "pointer_declarator", "reference_declarator",
              "abstract_reference_declarator", "scoped_identifier"):
        inner = node.child_by_field_name("declarator")
        if inner:
            return _resolve_c_declarator_name(inner)
        # fallback: first child that is an identifier or nested declarator
        for ch in node.children:
            result = _resolve_c_declarator_name(ch)
            if result:
                return result
    return None


# ── Cached language objects (loaded once per process) ─────────────────────────

_TS_CACHE: dict[str, Any] = {}


def _get_ts_lang(name: str) -> Any | None:
    if name not in _TS_CACHE:
        _TS_CACHE[name] = _load_ts_language(name)
    return _TS_CACHE[name]


# ── Dispatcher ────────────────────────────────────────────────────────────────

def _extract_symbols_treesitter(content: str, language: str) -> list[_Symbol]:
    """Extract symbols using tree-sitter (new API). Returns empty list on failure."""
    low = (language or "").lower()
    lang_obj = _get_ts_lang(low)
    if lang_obj is None:
        return []
    root = _ts_parse(content, lang_obj)
    if root is None:
        return []
    try:
        if low == "python":
            return _walk_python_ts(root)
        if low in ("javascript",):
            return _walk_js_ts(root, is_ts=False)
        if low == "typescript":
            return _walk_js_ts(root, is_ts=True)
        if low == "go":
            return _walk_go(root)
        if low == "java":
            return _walk_java(root)
        if low == "rust":
            return _walk_rust(root)
        if low == "c":
            return _walk_c_cpp(root, is_cpp=False)
        if low in ("cpp", "c++"):
            return _walk_c_cpp(root, is_cpp=True)
    except Exception as exc:
        logger.warning("tree-sitter extraction failed for %s: %s", language, exc)
    return []


# ── Legacy regex fallback (kept for unsupported languages) ────────────────────

_RE_PY_CLASS = re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
_RE_PY_FUNC = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)


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
    """Last-resort regex fallback for languages without tree-sitter support."""
    out: list[_Symbol] = []

    def add_from(regex: re.Pattern[str], kind: SymbolKind) -> None:
        for m in regex.finditer(content):
            line = _line_for_match(content, m.start())
            out.append((m.group(1), kind, line, line, None, None, ExtractSource.LEXICAL))

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
                    # Python stdlib ast is the ground truth; tree-sitter as fallback.
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
                    # Unknown language: try tree-sitter by name, then regex
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
