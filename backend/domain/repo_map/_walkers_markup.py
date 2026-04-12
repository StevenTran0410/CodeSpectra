"""Markup/styling language tree-sitter walkers: HTML, CSS, Markdown, Svelte."""

from typing import Any

from ._loaders import _lines, _node_text, _Symbol
from .types import ExtractSource, SymbolKind


# ---------------------------------------------------------------------------
# HTML — id/class attributes and heading text as symbols
# ---------------------------------------------------------------------------

def _walk_html(root: Any) -> list[_Symbol]:
    out: list[_Symbol] = []

    def walk(node: Any) -> None:
        if node.type == "element":
            start = next((c for c in node.children if c.type == "start_tag"), None)
            if start:
                tag_node = next((c for c in start.children if c.type == "tag_name"), None)
                tag = _node_text(tag_node) if tag_node else ""
                # Emit id attribute as a named symbol
                for attr in start.children:
                    if attr.type == "attribute":
                        attr_name_node = next((c for c in attr.children if c.type == "attribute_name"), None)
                        attr_val_node = next((c for c in attr.children if c.type == "quoted_attribute_value"), None)
                        if attr_name_node and _node_text(attr_name_node) == "id" and attr_val_node:
                            name = _node_text(attr_val_node).strip('"\'')
                            if name:
                                s, e = _lines(node)
                                out.append((f"#{name}", SymbolKind.VARIABLE, s, e, None, None, ExtractSource.AST))
                # Emit headings (h1-h6) text as symbols
                if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
                    text_parts = []
                    for ch in node.children:
                        if ch.type == "text":
                            t = _node_text(ch)
                            if t:
                                text_parts.append(t.strip())
                    name = " ".join(text_parts)[:60]
                    if name:
                        s, e = _lines(node)
                        out.append((name, SymbolKind.MODULE, s, e, None, None, ExtractSource.AST))
        for ch in node.children:
            if ch.is_named:
                walk(ch)

    walk(root)
    return out


# ---------------------------------------------------------------------------
# CSS — rule selectors and @keyframes names as symbols
# ---------------------------------------------------------------------------

def _walk_css(root: Any) -> list[_Symbol]:
    out: list[_Symbol] = []

    for node in root.children:
        if node.type == "rule_set":
            sel_node = next((c for c in node.children if c.type == "selectors"), None)
            name = _node_text(sel_node).split(",")[0].strip() if sel_node else None
            if name and len(name) <= 80:
                s, e = _lines(node)
                out.append((name, SymbolKind.CLASS, s, e, None, None, ExtractSource.AST))
        elif node.type == "media_statement":
            s, e = _lines(node)
            out.append(("@media", SymbolKind.MODULE, s, e, None, None, ExtractSource.AST))
        elif node.type == "keyframes_statement":
            name_node = next((c for c in node.children if c.type == "keyframes_name"), None)
            name = _node_text(name_node) if name_node else None
            if name:
                s, e = _lines(node)
                out.append((f"@keyframes {name}", SymbolKind.FUNCTION, s, e, None, None, ExtractSource.AST))
        elif node.type == "import_statement":
            s, e = _lines(node)
            text = _node_text(node)
            if text:
                out.append((text[:60], SymbolKind.VARIABLE, s, e, None, None, ExtractSource.AST))

    return out


# ---------------------------------------------------------------------------
# Markdown — headings as symbols
# ---------------------------------------------------------------------------

def _walk_markdown(root: Any) -> list[_Symbol]:
    out: list[_Symbol] = []

    def walk(node: Any, depth: int = 0) -> None:
        if node.type in ("atx_heading", "setext_heading"):
            inline = next((c for c in node.children if c.type == "inline"), None)
            name = _node_text(inline).strip() if inline else _node_text(node).lstrip("#").strip()
            if name:
                s, e = _lines(node)
                # Use heading level to determine kind: h1/h2 → MODULE, h3+ → FUNCTION
                marker = next(
                    (c for c in node.children if "marker" in c.type or "h1" in c.type or "h2" in c.type),
                    None,
                )
                kind = SymbolKind.MODULE if depth == 0 else SymbolKind.FUNCTION
                out.append((name, kind, s, e, None, None, ExtractSource.AST))
        for ch in node.children:
            if ch.is_named:
                walk(ch, depth + 1 if node.type == "section" else depth)

    walk(root)
    return out


# ---------------------------------------------------------------------------
# Svelte — component-level exports and script symbols
# ---------------------------------------------------------------------------

def _walk_svelte(root: Any) -> list[_Symbol]:
    """
    Svelte files have script/style/element blocks. We emit:
    - 'script' as a MODULE-level marker
    - exported let/const names from <script> raw_text via simple pattern
    """
    import re

    out: list[_Symbol] = []

    _EXPORT_RE = re.compile(r"\bexport\s+(?:let|const|function)\s+([A-Za-z_$][A-Za-z0-9_$]*)")
    _FUNC_RE = re.compile(r"\bfunction\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\(")

    for node in root.children:
        if node.type == "script_element":
            s, e = _lines(node)
            raw = next((c for c in node.children if c.type == "raw_text"), None)
            if raw:
                text = (raw.text or b"").decode()
                for m in _EXPORT_RE.finditer(text):
                    line = s + text[:m.start()].count("\n")
                    out.append((m.group(1), SymbolKind.VARIABLE, line, line, None, None, ExtractSource.AST))
                for m in _FUNC_RE.finditer(text):
                    if f"export function {m.group(1)}" not in text[:m.start() + 20]:
                        line = s + text[:m.start()].count("\n")
                        out.append((m.group(1), SymbolKind.FUNCTION, line, line, None, None, ExtractSource.AST))
        elif node.type == "style_element":
            s, e = _lines(node)
            out.append(("<style>", SymbolKind.MODULE, s, e, None, None, ExtractSource.AST))

    return out
