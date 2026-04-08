"""Python tree-sitter walker."""

from typing import Any

from ._loaders import _emit, _Symbol
from .types import SymbolKind


def _walk_python_ts(root: Any) -> list[_Symbol]:
    out: list[_Symbol] = []

    def walk(node: Any, cs: list[str]) -> None:
        t = node.type
        if t == "decorated_definition":
            for ch in node.children:
                if ch.type in ("function_definition", "class_definition"):
                    walk(ch, cs)
            return
        if t == "function_definition":
            kind = SymbolKind.METHOD if cs else SymbolKind.FUNCTION
            _emit(out, node, kind, cs[-1] if cs else None)
        elif t == "class_definition":
            name = _emit(out, node, SymbolKind.CLASS, cs[-1] if cs else None)
            if name:
                for ch in node.children:
                    walk(ch, [*cs, name])
                return
        for ch in node.children:
            walk(ch, cs)

    walk(root, [])
    return out
