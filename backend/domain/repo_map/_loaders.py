"""Tree-sitter language loading and low-level node helpers."""

from typing import Any

from .types import ExtractSource, SymbolKind

_Symbol = tuple[str, SymbolKind, int, int, str | None, str | None, ExtractSource]

_TS_CACHE: dict[str, Any] = {}


def _load_ts_language(name: str) -> Any | None:
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
    try:
        from tree_sitter import Parser
        p = Parser(lang)
        tree = p.parse(content.encode("utf-8", errors="ignore"))
        return tree.root_node
    except Exception:
        return None


def _node_name(node: Any) -> str | None:
    try:
        n = node.child_by_field_name("name")
        return n.text.decode("utf-8", errors="ignore").strip() if n and n.text else None
    except Exception:
        return None


def _node_text(node: Any) -> str | None:
    try:
        return node.text.decode("utf-8", errors="ignore").strip() if node and node.text else None
    except Exception:
        return None


def _lines(node: Any) -> tuple[int, int]:
    return node.start_point[0] + 1, node.end_point[0] + 1


def _emit(out: list[_Symbol], node: Any, kind: SymbolKind, parent: str | None = None) -> str | None:
    name = _node_name(node)
    if name:
        s, e = _lines(node)
        out.append((name, kind, s, e, None, parent, ExtractSource.AST))
    return name


def _get_ts_lang(name: str) -> Any | None:
    if name not in _TS_CACHE:
        _TS_CACHE[name] = _load_ts_language(name)
    return _TS_CACHE[name]
