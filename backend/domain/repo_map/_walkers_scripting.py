"""Scripting-language tree-sitter walkers: Ruby, PHP, Lua, C#."""

from typing import Any

from ._loaders import _lines, _node_text, _Symbol
from .types import ExtractSource, SymbolKind


# ---------------------------------------------------------------------------
# Ruby
# ---------------------------------------------------------------------------

def _walk_ruby(root: Any) -> list[_Symbol]:
    out: list[_Symbol] = []

    def walk(node: Any, parent: str | None = None) -> None:
        t = node.type
        if t == "class":
            # name is a 'constant' child
            name_node = next((c for c in node.children if c.type == "constant"), None)
            name = _node_text(name_node) if name_node else None
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.CLASS, s, e, None, parent, ExtractSource.AST))
                for ch in node.children:
                    walk(ch, name)
            return
        elif t == "module":
            name_node = next((c for c in node.children if c.type == "constant"), None)
            name = _node_text(name_node) if name_node else None
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.MODULE, s, e, None, parent, ExtractSource.AST))
                for ch in node.children:
                    walk(ch, name)
            return
        elif t == "method":
            name_node = node.child_by_field_name("name")
            name = _node_text(name_node) if name_node else None
            if name:
                kind = SymbolKind.METHOD if parent else SymbolKind.FUNCTION
                s, e = _lines(node)
                out.append((name, kind, s, e, None, parent, ExtractSource.AST))
        elif t == "singleton_method":
            # def self.foo — name field
            name_node = node.child_by_field_name("name")
            name = _node_text(name_node) if name_node else None
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.METHOD, s, e, None, parent, ExtractSource.AST))
        for ch in node.children:
            walk(ch, parent)

    walk(root)
    return out


# ---------------------------------------------------------------------------
# PHP
# ---------------------------------------------------------------------------

def _walk_php(root: Any) -> list[_Symbol]:
    out: list[_Symbol] = []

    def walk(node: Any, parent: str | None = None) -> None:
        t = node.type
        if t == "class_declaration":
            name_node = node.child_by_field_name("name")
            name = _node_text(name_node) if name_node else None
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.CLASS, s, e, None, parent, ExtractSource.AST))
                for ch in node.children:
                    walk(ch, name)
            return
        elif t == "interface_declaration":
            name_node = node.child_by_field_name("name")
            name = _node_text(name_node) if name_node else None
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.INTERFACE, s, e, None, parent, ExtractSource.AST))
        elif t == "trait_declaration":
            name_node = node.child_by_field_name("name")
            name = _node_text(name_node) if name_node else None
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.CLASS, s, e, None, parent, ExtractSource.AST))
        elif t == "enum_declaration":
            name_node = node.child_by_field_name("name")
            name = _node_text(name_node) if name_node else None
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.ENUM, s, e, None, parent, ExtractSource.AST))
        elif t == "function_definition":
            name_node = node.child_by_field_name("name")
            name = _node_text(name_node) if name_node else None
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.FUNCTION, s, e, None, parent, ExtractSource.AST))
        elif t == "method_declaration":
            name_node = node.child_by_field_name("name")
            name = _node_text(name_node) if name_node else None
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.METHOD, s, e, None, parent, ExtractSource.AST))
        for ch in node.children:
            walk(ch, parent)

    walk(root)
    return out


# ---------------------------------------------------------------------------
# Lua
# ---------------------------------------------------------------------------

def _walk_lua(root: Any) -> list[_Symbol]:
    out: list[_Symbol] = []

    def walk(node: Any) -> None:
        t = node.type
        if t == "function_declaration":
            # name field may be identifier or method_index_expression (MyClass:method)
            name_node = node.child_by_field_name("name")
            if name_node:
                name = _node_text(name_node)
                if name:
                    s, e = _lines(node)
                    kind = SymbolKind.METHOD if ":" in name else SymbolKind.FUNCTION
                    # For MyClass:method, parent = MyClass
                    if ":" in name:
                        parts = name.split(":", 1)
                        out.append((parts[1], kind, s, e, None, parts[0], ExtractSource.AST))
                    else:
                        out.append((name, kind, s, e, None, None, ExtractSource.AST))
        elif t == "local_function":
            name_node = node.child_by_field_name("name")
            name = _node_text(name_node) if name_node else None
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.FUNCTION, s, e, None, None, ExtractSource.AST))
        elif t == "assignment_statement":
            # Detect table-as-class: MyClass = {}
            var_list = node.child_by_field_name("variables")
            expr_list = node.child_by_field_name("values")
            if var_list and expr_list:
                var_name = _node_text(next((c for c in var_list.children if c.is_named), None))
                val = next((c for c in expr_list.children if c.is_named), None)
                if var_name and val and val.type == "table_constructor":
                    s, e = _lines(node)
                    out.append((var_name, SymbolKind.CLASS, s, e, None, None, ExtractSource.AST))
        for ch in node.children:
            walk(ch)

    walk(root)
    return out


# ---------------------------------------------------------------------------
# C# (csharp)
# ---------------------------------------------------------------------------

def _walk_csharp(root: Any) -> list[_Symbol]:
    out: list[_Symbol] = []

    def walk(node: Any, parent: str | None = None) -> None:
        t = node.type
        if t == "namespace_declaration":
            name_node = node.child_by_field_name("name")
            name = _node_text(name_node) if name_node else None
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.MODULE, s, e, None, None, ExtractSource.AST))
                for ch in node.children:
                    walk(ch, name)
            return
        elif t == "class_declaration":
            name_node = node.child_by_field_name("name")
            name = _node_text(name_node) if name_node else None
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.CLASS, s, e, None, parent, ExtractSource.AST))
                for ch in node.children:
                    walk(ch, name)
            return
        elif t == "record_declaration":
            name_node = node.child_by_field_name("name")
            name = _node_text(name_node) if name_node else None
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.CLASS, s, e, None, parent, ExtractSource.AST))
        elif t == "struct_declaration":
            name_node = node.child_by_field_name("name")
            name = _node_text(name_node) if name_node else None
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.CLASS, s, e, None, parent, ExtractSource.AST))
        elif t == "interface_declaration":
            name_node = node.child_by_field_name("name")
            name = _node_text(name_node) if name_node else None
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.INTERFACE, s, e, None, parent, ExtractSource.AST))
        elif t == "enum_declaration":
            name_node = node.child_by_field_name("name")
            name = _node_text(name_node) if name_node else None
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.ENUM, s, e, None, parent, ExtractSource.AST))
        elif t in ("method_declaration", "constructor_declaration", "destructor_declaration"):
            name_node = node.child_by_field_name("name")
            name = _node_text(name_node) if name_node else None
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.METHOD, s, e, None, parent, ExtractSource.AST))
        for ch in node.children:
            walk(ch, parent)

    walk(root)
    return out
