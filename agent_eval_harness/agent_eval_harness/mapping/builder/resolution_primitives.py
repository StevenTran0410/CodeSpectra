"""Entry-method recognition + model-call provenance vocabulary. Primitives layer (like
roles.VALID_ROLES / system_types.py's DECIDED_BY_*): dependency-free, importable from discovery/*
and mapping/* without a cycle.
"""
from __future__ import annotations

import ast
from typing import Literal

DECLARED_BOUNDARY = "declared_boundary"   # resolved callee's import module matches a config provider identity
DERIVED_BOUNDARY = "derived_boundary"     # resolved callee matches a config in-repo interface/Protocol method
STRUCTURAL_CLOSURE = "structural_closure"  # complete closure, no ast.Call at all, no boundary reached
LLM_VERIFIED = "llm_verified"  # LLM residue pass, cite-verified against the closure bundle it was given
LLM_UNVERIFIED_DEFAULT = "llm_unverified_default"  # residue pass gave no verifiable verdict -- defaulted to step
ModelCallSource = Literal[
    "declared_boundary", "derived_boundary", "structural_closure", "llm_verified", "llm_unverified_default",
]


def recognized_entry_methods(tree: ast.AST, class_name: str) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """A class's own methods that no OTHER method of the same class calls (`__init__` excluded,
    self-recursion excluded from the caller tally) -- the structural replacement for a kept
    run/run_async/__call__ name list. Returned in source (lineno) order."""
    cls = next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == class_name), None)
    if cls is None:
        return []
    methods = [
        m for m in cls.body
        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name != "__init__"
    ]
    called: set[str] = set()
    for m in methods:
        for node in ast.walk(m):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            if (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                    and f.value.id == "self" and f.attr != m.name):
                called.add(f.attr)
    return sorted((m for m in methods if m.name not in called), key=lambda m: m.lineno)


_CONVENTIONAL_ENTRY_NAMES = ("run", "run_async", "__call__")


def _is_stub_body(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True when `fn` carries no real logic: its body, minus an optional leading docstring, is only
    `pass`, `...`, and/or `raise NotImplementedError(...)` statements -- the common shape for a
    placeholder that defers to a sibling entry method for the real implementation."""
    body = fn.body
    if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    if not body:
        return True
    for stmt in body:
        if isinstance(stmt, ast.Pass):
            continue
        if (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
                and stmt.value.value is Ellipsis):
            continue
        if isinstance(stmt, ast.Raise) and stmt.exc is not None:
            callee = stmt.exc.func if isinstance(stmt.exc, ast.Call) else stmt.exc
            if isinstance(callee, ast.Name) and callee.id == "NotImplementedError":
                continue
        return False
    return True


def preferred_entry_method(tree: ast.AST, class_name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """For a PICK-ONE consumer (contract metadata, never the call-closure walk, which takes the
    whole recognized_entry_methods() set): among the recognized methods, prefer a conventional name
    when one is present (preserving today's selection everywhere that convention holds) and fall back
    to source order otherwise -- a preference among already-structural candidates, not a second gate.
    A stub body (_is_stub_body) never outranks a real implementation, conventionally named or not."""
    methods = recognized_entry_methods(tree, class_name)
    by_name = {m.name: m for m in methods}
    real_by_name = {name: m for name, m in by_name.items() if not _is_stub_body(m)}
    for name in _CONVENTIONAL_ENTRY_NAMES:
        if name in real_by_name:
            return real_by_name[name]
    real_methods = [m for m in methods if not _is_stub_body(m)]
    if real_methods:
        return real_methods[0]
    for name in _CONVENTIONAL_ENTRY_NAMES:
        if name in by_name:
            return by_name[name]
    return methods[0] if methods else None
