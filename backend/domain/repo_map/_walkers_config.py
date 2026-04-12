"""Config/data-language tree-sitter walkers: YAML, TOML, JSON, CMake, SQL."""

from typing import Any

from ._loaders import _lines, _node_text, _Symbol
from .types import ExtractSource, SymbolKind


# ---------------------------------------------------------------------------
# YAML — top-level keys as symbols
# ---------------------------------------------------------------------------

def _walk_yaml(root: Any) -> list[_Symbol]:
    out: list[_Symbol] = []

    def walk_mapping(node: Any) -> None:
        for ch in node.children:
            if ch.type == "block_mapping_pair":
                key_node = ch.child_by_field_name("key")
                if key_node:
                    name = _node_text(key_node)
                    if name:
                        s, e = _lines(ch)
                        out.append((name, SymbolKind.VARIABLE, s, e, None, None, ExtractSource.AST))

    def walk(node: Any) -> None:
        if node.type == "block_mapping":
            walk_mapping(node)
            return  # only emit top-level keys
        for ch in node.children:
            if ch.is_named:
                walk(ch)

    walk(root)
    return out


# ---------------------------------------------------------------------------
# TOML — tables and array tables as symbols
# ---------------------------------------------------------------------------

def _walk_toml(root: Any) -> list[_Symbol]:
    out: list[_Symbol] = []

    for node in root.children:
        if node.type in ("table", "table_array_element"):
            name_node = next((c for c in node.children if c.type == "bare_key"), None)
            name = _node_text(name_node) if name_node else None
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.MODULE, s, e, None, None, ExtractSource.AST))

    return out


# ---------------------------------------------------------------------------
# JSON — top-level object keys as symbols
# ---------------------------------------------------------------------------

def _walk_json(root: Any) -> list[_Symbol]:
    out: list[_Symbol] = []

    # root → document → object → pairs
    doc = next((c for c in root.children if c.is_named), None)
    if doc is None:
        return out
    obj = next((c for c in doc.children if c.type == "object"), None) if doc.type == "document" else (
        doc if doc.type == "object" else None
    )
    if obj is None:
        return out

    for pair in obj.children:
        if pair.type == "pair":
            key_node = next((c for c in pair.children if c.type == "string"), None)
            if key_node:
                name = _node_text(key_node).strip('"\'')
                if name:
                    s, e = _lines(pair)
                    out.append((name, SymbolKind.VARIABLE, s, e, None, None, ExtractSource.AST))

    return out


# ---------------------------------------------------------------------------
# CMake — functions and macros
# ---------------------------------------------------------------------------

def _walk_cmake(root: Any) -> list[_Symbol]:
    out: list[_Symbol] = []

    for node in root.children:
        if node.type == "function_def":
            cmd = next((c for c in node.children if c.type == "function_command"), None)
            if cmd:
                args = cmd.child_by_field_name("arguments") or next(
                    (c for c in cmd.children if c.type == "argument_list"), None
                )
                if args:
                    name_node = next((c for c in args.children if c.type == "argument"), None)
                    name = _node_text(name_node) if name_node else None
                    if name:
                        s, e = _lines(node)
                        out.append((name, SymbolKind.FUNCTION, s, e, None, None, ExtractSource.AST))
        elif node.type == "macro_def":
            cmd = next((c for c in node.children if c.type == "macro_command"), None)
            if cmd:
                args = cmd.child_by_field_name("arguments") or next(
                    (c for c in cmd.children if c.type == "argument_list"), None
                )
                if args:
                    name_node = next((c for c in args.children if c.type == "argument"), None)
                    name = _node_text(name_node) if name_node else None
                    if name:
                        s, e = _lines(node)
                        out.append((name, SymbolKind.FUNCTION, s, e, None, None, ExtractSource.AST))

    return out


# ---------------------------------------------------------------------------
# SQL — CREATE statements as symbols (tables, views, functions, procedures)
# ---------------------------------------------------------------------------

def _walk_sql(root: Any) -> list[_Symbol]:
    out: list[_Symbol] = []

    _CREATE_KIND: dict[str, SymbolKind] = {
        "create_table": SymbolKind.CLASS,
        "create_view": SymbolKind.CLASS,
        "create_function": SymbolKind.FUNCTION,
        "create_procedure": SymbolKind.FUNCTION,
        "create_index": SymbolKind.VARIABLE,
    }

    def walk(node: Any) -> None:
        t = node.type
        if t == "statement":
            for ch in node.children:
                if ch.type in _CREATE_KIND:
                    kind = _CREATE_KIND[ch.type]
                    # object_reference child holds the name
                    ref = next(
                        (c for c in ch.children if c.type == "object_reference"), None
                    )
                    name = _node_text(ref) if ref else None
                    if name:
                        s, e = _lines(ch)
                        out.append((name, kind, s, e, None, None, ExtractSource.AST))
        for ch in node.children:
            if ch.is_named:
                walk(ch)

    walk(root)
    return out
