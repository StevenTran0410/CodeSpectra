"""Functional-language tree-sitter walkers: Haskell, OCaml, Elixir, Julia."""

from typing import Any

from ._loaders import _lines, _node_text, _Symbol
from .types import ExtractSource, SymbolKind


# ---------------------------------------------------------------------------
# Haskell
# ---------------------------------------------------------------------------

def _walk_haskell(root: Any) -> list[_Symbol]:
    out: list[_Symbol] = []

    def walk(node: Any) -> None:
        t = node.type
        if t == "function":
            # Top-level function binding: variable child is the name
            var = next((c for c in node.children if c.type == "variable"), None)
            name = _node_text(var) if var else None
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.FUNCTION, s, e, None, None, ExtractSource.AST))
        elif t == "signature":
            # Type signature: variable child is the name — skip to avoid duplicates
            pass
        elif t == "data_type":
            name_node = node.child_by_field_name("name") or next(
                (c for c in node.children if c.type == "name"), None
            )
            name = _node_text(name_node) if name_node else None
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.TYPE, s, e, None, None, ExtractSource.AST))
        elif t == "newtype":
            name_node = node.child_by_field_name("name") or next(
                (c for c in node.children if c.type == "name"), None
            )
            name = _node_text(name_node) if name_node else None
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.TYPE, s, e, None, None, ExtractSource.AST))
        elif t == "type_synomym":  # note: typo is in tree-sitter-haskell grammar
            name_node = node.child_by_field_name("name") or next(
                (c for c in node.children if c.type == "name"), None
            )
            name = _node_text(name_node) if name_node else None
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.TYPE, s, e, None, None, ExtractSource.AST))
        elif t == "class":
            name_node = node.child_by_field_name("name") or next(
                (c for c in node.children if c.type == "name"), None
            )
            name = _node_text(name_node) if name_node else None
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.CLASS, s, e, None, None, ExtractSource.AST))
        elif t == "instance":
            pass  # skip instances — too noisy without type tracing
        for ch in node.children:
            walk(ch)

    # The root for Haskell is 'haskell' > 'declarations'
    for ch in root.children:
        if ch.type == "declarations":
            for decl in ch.children:
                walk(decl)
        else:
            walk(ch)
    return out


# ---------------------------------------------------------------------------
# OCaml
# ---------------------------------------------------------------------------

def _walk_ocaml(root: Any) -> list[_Symbol]:
    out: list[_Symbol] = []

    def walk(node: Any, parent: str | None = None) -> None:
        t = node.type
        if t == "value_definition":
            # let foo x = ... → let_binding > value_name
            for bind in node.children:
                if bind.type == "let_binding":
                    name_node = next((c for c in bind.children if c.type == "value_name"), None)
                    name = _node_text(name_node) if name_node else None
                    if name:
                        # Determine function vs plain value by checking for parameter child
                        has_params = any(c.type == "parameter" for c in bind.children)
                        kind = SymbolKind.FUNCTION if has_params else SymbolKind.VARIABLE
                        s, e = _lines(bind)
                        out.append((name, kind, s, e, None, parent, ExtractSource.AST))
        elif t == "type_definition":
            for bind in node.children:
                if bind.type == "type_binding":
                    name_node = next((c for c in bind.children if c.type == "type_constructor"), None)
                    name = _node_text(name_node) if name_node else None
                    if name:
                        s, e = _lines(bind)
                        out.append((name, SymbolKind.TYPE, s, e, None, parent, ExtractSource.AST))
        elif t == "module_definition":
            for bind in node.children:
                if bind.type == "module_binding":
                    name_node = next((c for c in bind.children if c.type == "module_name"), None)
                    name = _node_text(name_node) if name_node else None
                    if name:
                        s, e = _lines(bind)
                        out.append((name, SymbolKind.MODULE, s, e, None, parent, ExtractSource.AST))
                        for ch in bind.children:
                            walk(ch, name)
                        return
        elif t == "class_definition":
            name_node = next((c for c in node.children if c.type == "class_name"), None)
            name = _node_text(name_node) if name_node else None
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.CLASS, s, e, None, parent, ExtractSource.AST))
        for ch in node.children:
            walk(ch, parent)

    walk(root)
    return out


# ---------------------------------------------------------------------------
# Elixir
# ---------------------------------------------------------------------------

def _walk_elixir(root: Any) -> list[_Symbol]:
    """
    Elixir AST: everything is a 'call' node.
    We detect defmodule/def/defp/defmacro by checking the identifier child.
    """
    out: list[_Symbol] = []

    _DEF_KINDS = {
        "def": SymbolKind.FUNCTION,
        "defp": SymbolKind.FUNCTION,
        "defmacro": SymbolKind.FUNCTION,
        "defmacrop": SymbolKind.FUNCTION,
        "defguard": SymbolKind.FUNCTION,
        "defguardp": SymbolKind.FUNCTION,
    }

    def _first_arg_name(args_node: Any) -> str | None:
        """Extract the function/module name from the first argument."""
        if args_node is None:
            return None
        for ch in args_node.children:
            if ch.is_named:
                # Module name: alias node (e.g. Foo.Bar)
                if ch.type == "alias":
                    return _node_text(ch)
                # Function call signature: call > identifier
                if ch.type in ("call", "identifier"):
                    id_node = next(
                        (c for c in ch.children if c.type == "identifier"), None
                    ) if ch.type == "call" else ch
                    return _node_text(id_node) if id_node else None
                if ch.type == "binary_operator":
                    # def name, do: body pattern
                    left = ch.children[0] if ch.children else None
                    if left and left.is_named:
                        return _first_arg_name(left)
        return None

    def walk(node: Any, module: str | None = None) -> None:
        if node.type != "call":
            for ch in node.children:
                if ch.is_named:
                    walk(ch, module)
            return

        # Get the macro name (def/defmodule/etc.)
        id_node = next((c for c in node.children if c.type == "identifier"), None)
        macro = _node_text(id_node) if id_node else None

        args_node = next((c for c in node.children if c.type == "arguments"), None)
        do_block = next((c for c in node.children if c.type == "do_block"), None)

        if macro == "defmodule":
            name = _first_arg_name(args_node)
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.MODULE, s, e, None, None, ExtractSource.AST))
                if do_block:
                    for ch in do_block.children:
                        walk(ch, name)
            return

        if macro in _DEF_KINDS:
            name = _first_arg_name(args_node)
            if name:
                kind = _DEF_KINDS[macro]
                s, e = _lines(node)
                out.append((name, kind, s, e, None, module, ExtractSource.AST))
            return

        if macro in ("defstruct", "defexception"):
            # No meaningful name to extract, skip
            pass

        for ch in node.children:
            if ch.is_named:
                walk(ch, module)

    walk(root)
    return out


# ---------------------------------------------------------------------------
# Julia
# ---------------------------------------------------------------------------

def _walk_julia(root: Any) -> list[_Symbol]:
    out: list[_Symbol] = []

    def walk(node: Any, parent: str | None = None) -> None:
        t = node.type
        if t == "function_definition":
            # 'signature' is a named child (not a field) → use next()
            sig = next((c for c in node.children if c.is_named and c.type == "signature"), None)
            name = None
            if sig:
                # signature > call_expression > identifier
                call = next((c for c in sig.children if c.type == "call_expression"), None)
                if call:
                    id_node = next((c for c in call.children if c.type == "identifier"), None)
                    name = _node_text(id_node) if id_node else None
                else:
                    id_node = next((c for c in sig.children if c.type == "identifier"), None)
                    name = _node_text(id_node) if id_node else None
            if name:
                kind = SymbolKind.METHOD if parent else SymbolKind.FUNCTION
                s, e = _lines(node)
                out.append((name, kind, s, e, None, parent, ExtractSource.AST))
        elif t == "short_function_definition":
            sig = next((c for c in node.children if c.is_named and c.type == "signature"), None)
            name = None
            if sig:
                call = next((c for c in sig.children if c.type == "call_expression"), None)
                if call:
                    id_node = next((c for c in call.children if c.type == "identifier"), None)
                    name = _node_text(id_node) if id_node else None
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.FUNCTION, s, e, None, parent, ExtractSource.AST))
        elif t == "macro_definition":
            name_node = next((c for c in node.children if c.type == "identifier"), None)
            name = _node_text(name_node) if name_node else None
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.FUNCTION, s, e, None, parent, ExtractSource.AST))
        elif t == "struct_definition":
            # type_head > identifier
            head = next((c for c in node.children if c.type == "type_head"), None)
            name = None
            if head:
                id_node = next((c for c in head.children if c.type == "identifier"), None)
                name = _node_text(id_node) if id_node else None
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.CLASS, s, e, None, parent, ExtractSource.AST))
        elif t == "abstract_definition":
            id_node = next((c for c in node.children if c.type == "identifier"), None)
            name = _node_text(id_node) if id_node else None
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.TYPE, s, e, None, parent, ExtractSource.AST))
        elif t == "module_definition":
            id_node = next((c for c in node.children if c.type == "identifier"), None)
            name = _node_text(id_node) if id_node else None
            if name:
                s, e = _lines(node)
                out.append((name, SymbolKind.MODULE, s, e, None, None, ExtractSource.AST))
                for ch in node.children:
                    walk(ch, name)
            return
        for ch in node.children:
            walk(ch, parent)

    walk(root)
    return out
