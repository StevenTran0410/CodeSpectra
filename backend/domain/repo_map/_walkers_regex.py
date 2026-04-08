"""Regex and stdlib-ast fallback extractors."""

import ast
import re

from ._loaders import _Symbol
from .types import ExtractSource, SymbolKind

_RE_PY_CLASS = re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
_RE_PY_FUNC = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)


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

    def add_from(regex: re.Pattern[str], kind: SymbolKind) -> None:
        for m in regex.finditer(content):
            line = _line_for_match(content, m.start())
            out.append((m.group(1), kind, line, line, None, None, ExtractSource.LEXICAL))

    add_from(_RE_PY_CLASS, SymbolKind.CLASS)
    add_from(_RE_PY_FUNC, SymbolKind.FUNCTION)
    return out
