"""JavaScript / TypeScript tree-sitter walkers."""

from typing import Any

from ._loaders import _emit, _lines, _node_text, _Symbol
from .types import ExtractSource, SymbolKind

_JS_FLAT: dict[str, tuple[SymbolKind, bool]] = {
    "function_declaration": (SymbolKind.FUNCTION, True),
    "generator_function_declaration": (SymbolKind.FUNCTION, True),
    "function_expression": (SymbolKind.FUNCTION, True),
    "method_definition": (SymbolKind.METHOD, True),
    "method_signature": (SymbolKind.METHOD, True),
    "enum_declaration": (SymbolKind.ENUM, False),
}
_TS_ONLY_FLAT: dict[str, tuple[SymbolKind, bool]] = {
    "interface_declaration": (SymbolKind.INTERFACE, False),
    "type_alias_declaration": (SymbolKind.TYPE, False),
}


def _walk_js_ts(root: Any, is_ts: bool = False) -> list[_Symbol]:
    out: list[_Symbol] = []
    flat = {**_JS_FLAT, **(_TS_ONLY_FLAT if is_ts else {})}

    def walk(node: Any, cs: list[str]) -> None:
        t = node.type
        if t in flat:
            kind, inherit = flat[t]
            _emit(out, node, kind, cs[-1] if cs and inherit else None)
        elif t in ("class_declaration", "class"):
            name = _emit(out, node, SymbolKind.CLASS)
            if name:
                for ch in node.children:
                    walk(ch, [*cs, name])
                return
        elif t == "lexical_declaration":
            for ch in node.children:
                if ch.type == "variable_declarator":
                    val = ch.child_by_field_name("value")
                    if val and val.type in ("arrow_function", "function_expression"):
                        vn = ch.child_by_field_name("name")
                        name = _node_text(vn)
                        if name:
                            s, e = _lines(ch)
                            out.append(
                                (
                                    name,
                                    SymbolKind.FUNCTION,
                                    s,
                                    e,
                                    None,
                                    cs[-1] if cs else None,
                                    ExtractSource.AST,
                                )
                            )
        for ch in node.children:
            walk(ch, cs)

    walk(root, [])
    return out
