"""Systems-language tree-sitter walkers: Go, Rust, C/C++, Zig, Bash/sh.

Java has been moved to _walkers_jvm.py.
"""

from typing import Any

from ._loaders import _emit, _lines, _node_name, _node_text, _Symbol
from .types import ExtractSource, SymbolKind


def _walk_go(root: Any) -> list[_Symbol]:
    out: list[_Symbol] = []
    for node in root.children:
        t = node.type
        if t == "function_declaration":
            _emit(out, node, SymbolKind.FUNCTION)
        elif t == "method_declaration":
            name = _node_text(node.child_by_field_name("name"))
            recv_type = None
            recv = node.child_by_field_name("receiver")
            if recv:
                for ch in recv.children:
                    if ch.type == "parameter_declaration":
                        for tc in ch.children:
                            if tc.type in ("type_identifier", "pointer_type"):
                                recv_type = (_node_text(tc) or "").lstrip("*") or None
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.METHOD, s, e, None, recv_type, ExtractSource.AST))
        elif t == "type_declaration":
            for ch in node.children:
                if ch.type == "type_spec":
                    name = _node_name(ch)
                    if not name:
                        continue
                    body = ch.child_by_field_name("type")
                    kind = (
                        SymbolKind.CLASS
                        if body and body.type == "struct_type"
                        else SymbolKind.INTERFACE
                        if body and body.type == "interface_type"
                        else SymbolKind.TYPE
                    )
                    s, e = _lines(ch)
                    out.append((name, kind, s, e, None, None, ExtractSource.AST))
    return out


def _walk_java(root: Any) -> list[_Symbol]:
    out: list[_Symbol] = []

    def walk(node: Any, cs: list[str]) -> None:
        t = node.type
        if t == "class_declaration":
            name = _emit(out, node, SymbolKind.CLASS, cs[-1] if cs else None)
            if name:
                for ch in node.children:
                    walk(ch, [*cs, name])
                return
        elif t == "interface_declaration":
            _emit(out, node, SymbolKind.INTERFACE)
        elif t == "enum_declaration":
            _emit(out, node, SymbolKind.ENUM)
        elif t in ("method_declaration", "constructor_declaration"):
            _emit(out, node, SymbolKind.METHOD, cs[-1] if cs else None)
        for ch in node.children:
            walk(ch, cs)

    walk(root, [])
    return out


_RUST_FLAT: dict[str, SymbolKind] = {
    "struct_item": SymbolKind.CLASS,
    "enum_item": SymbolKind.ENUM,
    "trait_item": SymbolKind.INTERFACE,
    "type_item": SymbolKind.TYPE,
}


def _walk_rust(root: Any) -> list[_Symbol]:
    out: list[_Symbol] = []

    def walk(node: Any, impl_ctx: str | None = None) -> None:
        t = node.type
        if t == "function_item":
            _emit(out, node, SymbolKind.METHOD if impl_ctx else SymbolKind.FUNCTION, impl_ctx)
        elif t in _RUST_FLAT:
            _emit(out, node, _RUST_FLAT[t])
        elif t == "impl_item":
            impl_name = _node_text(node.child_by_field_name("type"))
            for ch in node.children:
                walk(ch, impl_ctx=impl_name)
            return
        for ch in node.children:
            walk(ch, impl_ctx)

    walk(root)
    return out


def _resolve_c_decl_name(node: Any) -> str | None:
    if node is None:
        return None
    if node.type == "identifier":
        return _node_text(node)
    if node.type in (
        "function_declarator",
        "pointer_declarator",
        "reference_declarator",
        "abstract_reference_declarator",
        "scoped_identifier",
    ):
        inner = node.child_by_field_name("declarator")
        if inner:
            return _resolve_c_decl_name(inner)
        for ch in node.children:
            result = _resolve_c_decl_name(ch)
            if result:
                return result
    return None


def _walk_c_cpp(root: Any, is_cpp: bool = False) -> list[_Symbol]:
    out: list[_Symbol] = []

    def walk(node: Any) -> None:
        t = node.type
        if t == "function_definition":
            name = _resolve_c_decl_name(node.child_by_field_name("declarator"))
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.FUNCTION, s, e, None, None, ExtractSource.AST))
        elif t in ("struct_specifier", "union_specifier"):
            _emit(out, node, SymbolKind.CLASS)
        elif t == "enum_specifier":
            _emit(out, node, SymbolKind.ENUM)
        elif is_cpp and t == "class_specifier":
            name = _emit(out, node, SymbolKind.CLASS)
            if name:
                for ch in node.children:
                    walk(ch)
                return
        elif t == "type_definition":
            for ch in node.children:
                if ch.type == "type_identifier":
                    tname = _node_text(ch)
                    if tname:
                        s, e = _lines(node)
                        out.append((tname, SymbolKind.TYPE, s, e, None, None, ExtractSource.AST))
                    break
        for ch in node.children:
            walk(ch)

    walk(root)
    return out


# ---------------------------------------------------------------------------
# Zig
# ---------------------------------------------------------------------------

def _walk_zig(root: Any) -> list[_Symbol]:
    out: list[_Symbol] = []

    def walk(node: Any) -> None:
        t = node.type
        if t == "function_declaration":
            name_node = node.child_by_field_name("name")
            name = _node_text(name_node) if name_node else None
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.FUNCTION, s, e, None, None, ExtractSource.AST))
        elif t == "variable_declaration":
            # const Foo = struct/union/enum { ... }
            name_node = node.child_by_field_name("name")
            name = _node_text(name_node) if name_node else None
            if name:
                val = next(
                    (c for c in node.children if c.type in (
                        "struct_declaration", "union_declaration", "enum_declaration"
                    )),
                    None,
                )
                if val:
                    kind_map = {
                        "struct_declaration": SymbolKind.CLASS,
                        "union_declaration": SymbolKind.CLASS,
                        "enum_declaration": SymbolKind.ENUM,
                    }
                    s, e = _lines(node)
                    out.append((name, kind_map[val.type], s, e, None, None, ExtractSource.AST))
        for ch in node.children:
            walk(ch)

    walk(root)
    return out


# ---------------------------------------------------------------------------
# Bash / sh
# ---------------------------------------------------------------------------

def _walk_bash(root: Any) -> list[_Symbol]:
    out: list[_Symbol] = []

    for node in root.children:
        if node.type == "function_definition":
            # name is a 'word' child
            name_node = next((c for c in node.children if c.type == "word"), None)
            name = _node_text(name_node) if name_node else None
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.FUNCTION, s, e, None, None, ExtractSource.AST))

    return out
