"""Generic AST resolver for prompt-constant string values — no module-name assumption, no
target-specific pattern. Handles Constant/JoinedStr/BinOp(Add)/Name with bounded recursion."""
from __future__ import annotations

import ast
import logging
from pathlib import Path

logger = logging.getLogger("agent_eval_harness.discovery.prompt_resolver")

_MAX_RESOLVE_DEPTH = 4


def resolve_constant(
    node: ast.expr | None,
    consts: dict[str, ast.expr],
    depth: int = 0,
    visited: frozenset[str] | None = None,
) -> str | None:
    """Resolve an expression node to its string value, or None if it isn't statically knowable.
    Constant(str) -> value. JoinedStr -> literal parts verbatim, '{expr}' placeholder per
    FormattedValue. BinOp(Add) -> both sides resolved and concatenated, else None. Name -> looked
    up in `consts` and recursed. Anything else (a call, a subscript, ...) -> None; never guessed."""
    if node is None or depth > _MAX_RESOLVE_DEPTH:
        return None
    if visited is None:
        visited = frozenset()

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value

    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                try:
                    parts.append("{" + ast.unparse(value.value) + "}")
                except Exception as e:
                    logger.debug(f"Could not unparse f-string expression, using placeholder: {e}")
                    parts.append("{...}")
            else:
                return None
        return "".join(parts)

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = resolve_constant(node.left, consts, depth + 1, visited)
        right = resolve_constant(node.right, consts, depth + 1, visited)
        return left + right if left is not None and right is not None else None

    if isinstance(node, ast.Name):
        if node.id in visited or node.id not in consts:
            return None
        return resolve_constant(consts[node.id], consts, depth + 1, visited | {node.id})

    return None


def build_module_constants(tree: ast.Module) -> dict[str, ast.expr]:
    """Index every top-level `NAME = <expr>` / `NAME += <expr>` in source order — no name
    pattern, no module assumption. AugAssign folds into the running value so a later lookup
    (or a later AugAssign) sees the cumulative result."""
    consts: dict[str, ast.expr] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            consts[node.targets[0].id] = node.value
        elif (
            isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and isinstance(node.op, ast.Add)
        ):
            name = node.target.id
            prior_value = resolve_constant(consts.get(name), consts) if name in consts else None
            rhs_value = resolve_constant(node.value, consts)
            if prior_value is not None and rhs_value is not None:
                consts[name] = ast.Constant(value=prior_value + rhs_value)
    return consts


def _resolve_import_module_file(
    node: ast.ImportFrom, importing_file: Path, repo_root: Path
) -> Path | None:
    """Resolve `from X import Y` (absolute, under repo_root) or `from .X import Y`
    (relative, `level>0`, honouring the importing file's own package depth) to a real .py file."""
    if node.level and node.level > 0:
        base = importing_file.parent
        for _ in range(node.level - 1):
            base = base.parent
        module_parts = node.module.split(".") if node.module else []
    else:
        if not node.module:
            return None
        base = repo_root
        module_parts = node.module.split(".")

    candidate = base
    for part in module_parts:
        candidate = candidate / part

    py_file = candidate.with_suffix(".py")
    if py_file.is_file():
        return py_file
    init_file = candidate / "__init__.py"
    if init_file.is_file():
        return init_file
    return None


def resolve_import_site(
    node: ast.ImportFrom, importing_file: Path, repo_root: Path
) -> dict[str, str | None]:
    """Resolve every alias of an `ImportFrom` to the string value it points at in the target
    module. Never raises — an unresolvable import (3rd-party, missing file) yields None per
    alias, not an exception."""
    aliases = {alias.asname or alias.name: alias.name for alias in node.names}

    target_file = _resolve_import_module_file(node, importing_file, repo_root)
    if target_file is None:
        return {local: None for local in aliases}

    try:
        source = target_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(target_file))
    except (OSError, UnicodeDecodeError, SyntaxError) as e:
        logger.warning(f"Could not resolve import target {target_file}: {e}")
        return {local: None for local in aliases}

    consts = build_module_constants(tree)
    return {
        local: resolve_constant(consts.get(original), consts)
        for local, original in aliases.items()
    }
