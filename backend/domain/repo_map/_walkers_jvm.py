"""JVM-family tree-sitter walkers: Java, Kotlin, Scala, Groovy."""

from typing import Any

from ._loaders import _emit, _lines, _node_name, _node_text, _Symbol
from .types import ExtractSource, SymbolKind


# ---------------------------------------------------------------------------
# Java (moved from _walkers_systems.py)
# ---------------------------------------------------------------------------

def _walk_java(root: Any) -> list[_Symbol]:
    out: list[_Symbol] = []

    def walk(node: Any, class_stack: list[str]) -> None:
        t = node.type
        if t == "class_declaration":
            name = _emit(out, node, SymbolKind.CLASS, class_stack[-1] if class_stack else None)
            if name:
                for ch in node.children:
                    walk(ch, [*class_stack, name])
                return
        elif t == "interface_declaration":
            _emit(out, node, SymbolKind.INTERFACE)
        elif t == "enum_declaration":
            _emit(out, node, SymbolKind.ENUM)
        elif t in ("method_declaration", "constructor_declaration"):
            _emit(out, node, SymbolKind.METHOD, class_stack[-1] if class_stack else None)
        for ch in node.children:
            walk(ch, class_stack)

    walk(root, [])
    return out


# ---------------------------------------------------------------------------
# Kotlin
# ---------------------------------------------------------------------------

def _walk_kotlin(root: Any) -> list[_Symbol]:
    out: list[_Symbol] = []

    def _kotlin_name(node: Any) -> str | None:
        """Kotlin uses 'identifier' as a direct named child for the name."""
        n = node.child_by_field_name("name")
        if n:
            return _node_text(n)
        # Fallback: first identifier child
        for ch in node.children:
            if ch.type == "identifier":
                return _node_text(ch)
        return None

    def walk(node: Any, parent: str | None = None) -> None:
        t = node.type
        if t == "class_declaration":
            name = _kotlin_name(node)
            # Determine kind: check raw text for 'interface' or 'enum' modifier
            raw = (node.text or b"").decode()[:30]
            kind = SymbolKind.INTERFACE if "interface" in raw else (
                SymbolKind.ENUM if "enum" in raw else SymbolKind.CLASS
            )
            if name:
                s, e = _lines(node)
                out.append((name, kind, s, e, None, parent, ExtractSource.AST))
                for ch in node.children:
                    walk(ch, name)
            return
        elif t == "object_declaration":
            name = _kotlin_name(node)
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.CLASS, s, e, None, parent, ExtractSource.AST))
        elif t == "function_declaration":
            name = _kotlin_name(node)
            if name:
                kind = SymbolKind.METHOD if parent else SymbolKind.FUNCTION
                s, e = _lines(node)
                out.append((name, kind, s, e, None, parent, ExtractSource.AST))
        for ch in node.children:
            walk(ch, parent)

    walk(root)
    return out


# ---------------------------------------------------------------------------
# Scala
# ---------------------------------------------------------------------------

def _walk_scala(root: Any) -> list[_Symbol]:
    out: list[_Symbol] = []

    def walk(node: Any, parent: str | None = None) -> None:
        t = node.type
        if t == "class_definition":
            name = _emit(out, node, SymbolKind.CLASS, parent)
            for ch in node.children:
                walk(ch, name)
            return
        elif t == "object_definition":
            name = _emit(out, node, SymbolKind.CLASS, parent)
            for ch in node.children:
                walk(ch, name)
            return
        elif t == "trait_definition":
            name = _emit(out, node, SymbolKind.INTERFACE, parent)
            for ch in node.children:
                walk(ch, name)
            return
        elif t in ("function_definition", "function_declaration"):
            kind = SymbolKind.METHOD if parent else SymbolKind.FUNCTION
            _emit(out, node, kind, parent)
        elif t == "type_definition":
            _emit(out, node, SymbolKind.TYPE, parent)
        for ch in node.children:
            walk(ch, parent)

    walk(root)
    return out


# ---------------------------------------------------------------------------
# Groovy
# ---------------------------------------------------------------------------

def _walk_groovy(root: Any) -> list[_Symbol]:
    out: list[_Symbol] = []

    def walk(node: Any, parent: str | None = None) -> None:
        t = node.type
        if t == "class_declaration":
            name = _emit(out, node, SymbolKind.CLASS, parent)
            for ch in node.children:
                walk(ch, name)
            return
        elif t == "interface_declaration":
            name = _emit(out, node, SymbolKind.INTERFACE, parent)
            for ch in node.children:
                walk(ch, name)
            return
        elif t == "enum_declaration":
            _emit(out, node, SymbolKind.ENUM, parent)
        elif t == "method_declaration":
            _emit(out, node, SymbolKind.METHOD if parent else SymbolKind.FUNCTION, parent)
        elif t == "function_definition":
            _emit(out, node, SymbolKind.FUNCTION, parent)
        for ch in node.children:
            walk(ch, parent)

    walk(root)
    return out
