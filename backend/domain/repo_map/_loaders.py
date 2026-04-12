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
        elif name == "ruby":
            import tree_sitter_ruby as m  # type: ignore[no-redef]
        elif name == "php":
            import tree_sitter_php as m  # type: ignore[no-redef]
            return Language(m.language_php())
        elif name in ("csharp", "c_sharp", "cs"):
            import tree_sitter_c_sharp as m  # type: ignore[no-redef]
        elif name == "kotlin":
            import tree_sitter_kotlin as m  # type: ignore[no-redef]
        elif name == "scala":
            import tree_sitter_scala as m  # type: ignore[no-redef]
        elif name in ("bash", "sh", "shell"):
            import tree_sitter_bash as m  # type: ignore[no-redef]
        elif name == "lua":
            import tree_sitter_lua as m  # type: ignore[no-redef]
        elif name == "zig":
            import tree_sitter_zig as m  # type: ignore[no-redef]
        elif name == "haskell":
            import tree_sitter_haskell as m  # type: ignore[no-redef]
        elif name == "elixir":
            import tree_sitter_elixir as m  # type: ignore[no-redef]
        elif name == "ocaml":
            import tree_sitter_ocaml as m  # type: ignore[no-redef]
            return Language(m.language_ocaml())
        elif name == "julia":
            import tree_sitter_julia as m  # type: ignore[no-redef]
        elif name == "yaml":
            import tree_sitter_yaml as m  # type: ignore[no-redef]
        elif name == "toml":
            import tree_sitter_toml as m  # type: ignore[no-redef]
        elif name == "html":
            import tree_sitter_html as m  # type: ignore[no-redef]
        elif name == "css":
            import tree_sitter_css as m  # type: ignore[no-redef]
        elif name == "json":
            import tree_sitter_json as m  # type: ignore[no-redef]
        elif name == "markdown":
            import tree_sitter_markdown as m  # type: ignore[no-redef]
        elif name == "groovy":
            import tree_sitter_groovy as m  # type: ignore[no-redef]
        elif name == "cmake":
            import tree_sitter_cmake as m  # type: ignore[no-redef]
        elif name == "svelte":
            import tree_sitter_svelte as m  # type: ignore[no-redef]
        elif name == "sql":
            import tree_sitter_sql as m  # type: ignore[no-redef]
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
