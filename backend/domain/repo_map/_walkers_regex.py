"""Regex and stdlib-ast fallback extractors."""

import ast
import re

from ._loaders import _Symbol
from .types import ExtractSource, SymbolKind

_RE_PY_CLASS = re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
_RE_PY_FUNC = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)

# Generic patterns used as lexical fallback for languages without a dedicated walker
_RE_FUNC_KEYWORD = re.compile(r"\bfun(?:ction)?\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*[\(\{]", re.MULTILINE)
_RE_CLASS_KEYWORD = re.compile(r"\b(?:class|interface|struct|trait|object|module|record)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)

# Language-specific patterns
_LEXICAL_PATTERNS: dict[str, list[tuple[re.Pattern[str], SymbolKind]]] = {
    "ruby":     [
        (re.compile(r"^\s*(?:def\s+self\.)?def\s+([A-Za-z_][A-Za-z0-9_?!]*)", re.MULTILINE), SymbolKind.FUNCTION),
        (re.compile(r"^\s*(?:class|module)\s+([A-Za-z][A-Za-z0-9_:]*)", re.MULTILINE), SymbolKind.CLASS),
    ],
    "php":      [
        (re.compile(r"\bfunction\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE), SymbolKind.FUNCTION),
        (re.compile(r"\b(?:class|interface|trait|enum)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE), SymbolKind.CLASS),
    ],
    "kotlin":   [
        (re.compile(r"\bfun\s+([A-Za-z_][A-Za-z0-9_]*)\s*[\(<]", re.MULTILINE), SymbolKind.FUNCTION),
        (re.compile(r"\b(?:class|object|interface|data class)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE), SymbolKind.CLASS),
    ],
    "scala":    [
        (re.compile(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*[\[(]", re.MULTILINE), SymbolKind.FUNCTION),
        (re.compile(r"\b(?:class|object|trait|case class)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE), SymbolKind.CLASS),
    ],
    "haskell":  [
        (re.compile(r"^([a-z_][A-Za-z0-9_']*)\s+::", re.MULTILINE), SymbolKind.FUNCTION),
        (re.compile(r"\b(?:data|newtype|type)\s+([A-Z][A-Za-z0-9_]*)", re.MULTILINE), SymbolKind.TYPE),
    ],
    "elixir":   [
        (re.compile(r"\bdef(?:p|macro|guard)?\s+([A-Za-z_][A-Za-z0-9_?!]*)", re.MULTILINE), SymbolKind.FUNCTION),
        (re.compile(r"\bdefmodule\s+([A-Za-z][A-Za-z0-9_.]*)", re.MULTILINE), SymbolKind.MODULE),
    ],
    "lua":      [
        (re.compile(r"\bfunction\s+([A-Za-z_][A-Za-z0-9_.:']*)\s*\(", re.MULTILINE), SymbolKind.FUNCTION),
    ],
    "bash":     [
        (re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*\)", re.MULTILINE), SymbolKind.FUNCTION),
    ],
    "csharp":   [
        (re.compile(r"\b(?:class|interface|struct|record|enum)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE), SymbolKind.CLASS),
        (re.compile(r"\b(?:public|private|protected|internal|static|virtual|override|async)[\s\w<>\[\]]*\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE), SymbolKind.FUNCTION),
    ],
    "groovy":   [
        (re.compile(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE), SymbolKind.FUNCTION),
        (re.compile(r"\b(?:class|interface|trait|enum)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE), SymbolKind.CLASS),
    ],
}


class _PythonSymbolVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.symbols: list[_Symbol] = []
        self._class_stack: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self.symbols.append(
            (
                node.name,
                SymbolKind.CLASS,
                int(getattr(node, "lineno", 1)),
                int(getattr(node, "end_lineno", getattr(node, "lineno", 1))),
                None,
                self._class_stack[-1] if self._class_stack else None,
                ExtractSource.AST,
            )
        )
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._visit_function(node, is_async=True)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, is_async: bool) -> None:
        args = [a.arg for a in node.args.args]
        sig_prefix = "async " if is_async else ""
        sig = f"{sig_prefix}{node.name}({', '.join(args)})"
        kind = SymbolKind.METHOD if self._class_stack else SymbolKind.FUNCTION
        parent_name = self._class_stack[-1] if self._class_stack else None
        self.symbols.append(
            (
                node.name,
                kind,
                int(getattr(node, "lineno", 1)),
                int(getattr(node, "end_lineno", getattr(node, "lineno", 1))),
                sig,
                parent_name,
                ExtractSource.AST,
            )
        )
        self.generic_visit(node)


def _line_for_match(content: str, start: int) -> int:
    return content.count("\n", 0, start) + 1


def _extract_python_symbols_ast(content: str) -> list[_Symbol]:
    tree = ast.parse(content)
    visitor = _PythonSymbolVisitor()
    visitor.visit(tree)
    return visitor.symbols


def _extract_lexical_symbols(content: str, language: str | None) -> list[_Symbol]:
    out: list[_Symbol] = []
    lang = (language or "").lower()

    def add_from(regex: re.Pattern[str], kind: SymbolKind) -> None:
        for m in regex.finditer(content):
            line = _line_for_match(content, m.start())
            out.append((m.group(1), kind, line, line, None, None, ExtractSource.LEXICAL))

    # Language-specific patterns
    if lang in _LEXICAL_PATTERNS:
        for regex, kind in _LEXICAL_PATTERNS[lang]:
            add_from(regex, kind)
        return out

    # Python-family fallback (also catches Ruby def/class, Groovy def)
    add_from(_RE_PY_CLASS, SymbolKind.CLASS)
    add_from(_RE_PY_FUNC, SymbolKind.FUNCTION)

    # Generic function/class keywords for anything else
    if not out:
        add_from(_RE_FUNC_KEYWORD, SymbolKind.FUNCTION)
        add_from(_RE_CLASS_KEYWORD, SymbolKind.CLASS)

    return out
