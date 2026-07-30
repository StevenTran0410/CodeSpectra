"""Static (AST-only) harvest of per-agent EvaluationContracts; never imports target code."""
from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from agent_eval_harness.config import ContractConventions
from agent_eval_harness.datasets.archetype_vocabulary import KWARG_CASE_KEY_MAPPING
from agent_eval_harness.datasets.generator_utils import config_kwarg_names_from_case_binding
from agent_eval_harness.discovery.wiring import parse_entry_suffix
from agent_eval_harness.mapping.agent_flow import AgentFlowMap
from agent_eval_harness.mapping.builder.resolution_primitives import preferred_entry_method
from agent_eval_harness.mapping.builder.types import parse_python_source
from agent_eval_harness.mapping.system_map import Component, SystemMap
from agent_eval_harness.planning.contract import (
    EvaluationContract,
    InvocationContract,
    KwargSpec,
    ObservabilityContract,
    OutputContract,
)

logger = logging.getLogger("agent_eval_harness.contract_harvest")

_QUERYISH_PARAMS = frozenset({"query", "question", "prompt", "text", "message", "user_input"})
_CONFIG_PARAMS = frozenset({"provider_id", "model_id"})
_SCHEMA_NAME_RE = re.compile(r"SCHEMA", re.IGNORECASE)
_DYNAMIC = "<dynamic>"
_MAX_TRACE_DEPTH = 6  # recursion cap for field-consumer call tracing (_trace_stmts)
_MAX_SCHEMA_DEPTH = 6  # recursion cap for nested class schema resolution
_MAX_SELF_CALL_DEPTH = 6  # recursion cap for entry-method schema-name closure (_referenced_schema_names)
_SCHEMA_SNIPPET_MAX_CHARS = 200  # upstream_context_specs preview length


@dataclass
class _SchemaResolveCtx:
    """Shared context for recursive class schema resolution."""
    asts: dict[Path, ast.Module]
    files_root: Path | None
    visited: set[str]
    depth: int
    conventions: ContractConventions | None


@dataclass
class ConstructorDepBinding:
    """Mapping from constructor parameter to self attribute and usage site."""
    param: str
    attr: str
    annotation: str | None = None


@dataclass
class DepCallSite:
    """A call to a dependency method or free-function reference."""
    dep_attr: str
    method: str
    citation: str
    via: str


def _parse_files(files: list[Path]) -> dict[Path, ast.Module]:
    asts: dict[Path, ast.Module] = {}
    for file in files:
        if file.suffix != ".py":
            continue
        parsed = parse_python_source(file)
        if parsed is not None:
            asts[file] = parsed[1]
    return asts


def _resolve_component_file(component: Component, asts: dict[Path, ast.Module]) -> Path | None:
    if component.file:
        suffix = component.file.replace("\\", "/")
        for path in asts:
            if path.as_posix().endswith(suffix):
                return path
    # entry_point fallback: match the module dots against the path tail
    module_path, _, _ = component.entry_point.partition(":")
    if module_path:
        rel = "/".join(module_path.split(".")) + ".py"
        for path in asts:
            if path.as_posix().endswith(rel):
                return path
    return None


def _find_class(tree: ast.Module, class_name: str) -> ast.ClassDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    return None


def _resolve_component_class(
    component: Component, asts: dict[Path, ast.Module]
) -> tuple[Path | None, str, ast.ClassDef | None, str | None]:
    """Resolve entry_point to (file, name, class def, explicit_method); path None on failure.

    name is the class name (class/bound_method) or the function name (function entry). explicit_method
    is the bound-method name when the suffix is `Owner.method`, else None.
    """
    path = _resolve_component_file(component, asts)
    if path is None:
        return None, "", None, None
    _, _, suffix = component.entry_point.partition(":")
    owner, name = parse_entry_suffix(suffix)
    if owner is not None:  # 2-part suffix => bound_method by construction
        return path, owner, _find_class(asts[path], owner), name
    if component.entry_kind == "function":
        return path, name, None, None
    # Legacy/class path — identical behavior to pre-CS-311 (entry_kind None or "class").
    return path, name, (_find_class(asts[path], name) if name else None), None


def _resolve_entry(
    component: Component, asts: dict[Path, ast.Module]
) -> tuple[Path | None, ast.ClassDef | None, ast.FunctionDef | ast.AsyncFunctionDef | None]:
    """Resolve a component to (file, class-or-None, entry function/method-or-None)."""
    path, name, cls, explicit_method = _resolve_component_class(component, asts)
    if path is None:
        return None, None, None
    if cls is not None:
        return path, cls, _find_entry_method(cls, explicit_method)
    return path, None, (_find_function(asts[path], name) if name else None)


def _unparse(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception as exc:  # ast.unparse's failure modes aren't enumerated; degrade, don't crash
        logger.debug("ast.unparse failed for %r: %s", node, exc)
        return None


def _positional_params_with_defaults(
    args: ast.arguments,
) -> tuple[list[ast.arg], list[ast.expr | None]]:
    """Positional params (self/cls stripped) paired with aligned defaults (None where required)."""
    positional = list(args.posonlyargs) + list(args.args)
    if positional and positional[0].arg in ("self", "cls"):
        positional = positional[1:]
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    return positional, defaults


def _harvest_kwargs(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[list[KwargSpec], list[str]]:
    """Returns (kwargs, notes) — notes flag shapes statics can't fully resolve (*args/**kwargs)."""
    notes: list[str] = []
    specs: list[KwargSpec] = []
    args = fn.args
    positional, defaults = _positional_params_with_defaults(args)
    for arg, default in zip(positional, defaults):
        specs.append(
            KwargSpec(
                name=arg.arg,
                annotation=_unparse(arg.annotation),
                default_repr=_unparse(default),
                required=default is None,
            )
        )
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        specs.append(
            KwargSpec(
                name=arg.arg,
                annotation=_unparse(arg.annotation),
                default_repr=_unparse(default),
                required=default is None,
            )
        )
    if args.vararg:
        notes.append(f"varargs *{args.vararg.arg} not statically resolvable")
    if args.kwarg:
        notes.append(f"var-kwargs **{args.kwarg.arg} not statically resolvable")
    return specs, notes


def _find_entry_method(
    cls: ast.ClassDef, explicit_method: str | None = None
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    if explicit_method is not None:  # bound-method entry: any method name, not just the recognized set
        by_name = {
            node.name: node
            for node in cls.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        return by_name.get(explicit_method)
    return preferred_entry_method(cls, cls.name)


def _harvest_constructor_deps(cls: ast.ClassDef | None) -> list[str]:
    if cls is None:
        return []
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__init__":
            deps = []
            for arg in node.args.args[1:]:
                deps.append(_unparse(arg.annotation) or arg.arg)
            return deps
    return []


def _harvest_constructor_dep_bindings(cls: ast.ClassDef) -> list[ConstructorDepBinding]:
    """Extract param->attr bindings from __init__ body."""
    init_node = None
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__init__":
            init_node = node
            break
    if init_node is None:
        return []

    param_annot = {}
    for arg in init_node.args.args[1:]:
        param_annot[arg.arg] = _unparse(arg.annotation) if arg.annotation else None

    bindings = []
    for stmt in init_node.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if not (len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Attribute)):
            continue
        attr_node = stmt.targets[0]
        if not (isinstance(attr_node.value, ast.Name) and attr_node.value.id == "self"):
            continue
        if not isinstance(stmt.value, ast.Name):
            continue

        param_name = stmt.value.id
        attr_name = attr_node.attr
        if param_name in param_annot:
            bindings.append(ConstructorDepBinding(
                param=param_name,
                attr=attr_name,
                annotation=param_annot[param_name]
            ))
    return bindings


def _harvest_dep_call_sites(
    entry: ast.FunctionDef | ast.AsyncFunctionDef | None,
    dep_bindings: list[ConstructorDepBinding],
    own_tree: ast.Module,
    own_file: Path,
    asts: dict[Path, ast.Module],
    files_root: Path | None,
) -> list[DepCallSite]:
    """Find direct and free-function call sites referencing deps."""
    if entry is None or not dep_bindings:
        return []

    dep_attrs = {b.attr for b in dep_bindings}
    call_sites = []

    for node in ast.walk(entry):
        if not isinstance(node, ast.Call):
            continue

        # Pattern (a): Direct method call: self.<dep_attr>.<method>(...)
        if (isinstance(node.func, ast.Attribute) and
            isinstance(node.func.value, ast.Attribute) and
            isinstance(node.func.value.value, ast.Name) and
            node.func.value.value.id == "self" and
            node.func.value.attr in dep_attrs):

            method_name = node.func.attr
            dep_attr = node.func.value.attr
            citation = _rel_cite(own_file, node.lineno, files_root)
            call_sites.append(DepCallSite(
                dep_attr=dep_attr,
                method=method_name,
                citation=citation,
                via="direct"
            ))

        # Pattern (b): Free-function with dep as keyword arg: func(dep_attr=self.<dep_attr>, ...)
        elif isinstance(node.func, ast.Name):
            func_name = node.func.id
            resolved_func = _resolve_callable_with_disk_fallback(func_name, own_tree, own_file, asts, files_root)
            if resolved_func is not None:
                func_def, func_file = resolved_func

                # Check keyword args
                for keyword in node.keywords:
                    if (isinstance(keyword.value, ast.Attribute) and
                        isinstance(keyword.value.value, ast.Name) and
                        keyword.value.value.id == "self" and
                        keyword.value.attr in dep_attrs):

                        dep_attr = keyword.value.attr
                        param_name = keyword.arg

                        for func_stmt in ast.walk(func_def):
                            if (isinstance(func_stmt, ast.Call) and
                                isinstance(func_stmt.func, ast.Attribute) and
                                isinstance(func_stmt.func.value, ast.Name) and
                                func_stmt.func.value.id == param_name):

                                method_name = func_stmt.func.attr
                                citation = _rel_cite(own_file, node.lineno, files_root)
                                call_sites.append(DepCallSite(
                                    dep_attr=dep_attr,
                                    method=method_name,
                                    citation=citation,
                                    via=f"free_function:{func_name}"
                                ))
                                break

                # Pattern (c): Free-function with dep as positional arg: func(self.<dep_attr>, ...)
                for idx, arg in enumerate(node.args):
                    if (isinstance(arg, ast.Attribute) and
                        isinstance(arg.value, ast.Name) and
                        arg.value.id == "self" and
                        arg.attr in dep_attrs):

                        dep_attr = arg.attr
                        params = [a.arg for a in (func_def.args.args + func_def.args.kwonlyargs)]
                        param_name = params[idx] if idx < len(params) else None
                        if param_name:
                            for func_stmt in ast.walk(func_def):
                                if (isinstance(func_stmt, ast.Call) and
                                    isinstance(func_stmt.func, ast.Attribute) and
                                    isinstance(func_stmt.func.value, ast.Name) and
                                    func_stmt.func.value.id == param_name):

                                    method_name = func_stmt.func.attr
                                    citation = _rel_cite(own_file, node.lineno, files_root)
                                    call_sites.append(DepCallSite(
                                        dep_attr=dep_attr,
                                        method=method_name,
                                        citation=citation,
                                        via=f"free_function_pos:{func_name}"
                                    ))
                                    break

        # Pattern (d): Sibling helper method call: self.<sibling_method>(...)
        elif (isinstance(node.func, ast.Attribute) and
              isinstance(node.func.value, ast.Name) and
              node.func.value.id == "self" and
              node.func.attr not in dep_attrs):

            sibling_name = node.func.attr
            sibling_def = _find_sibling_method_def(own_tree, sibling_name)
            if sibling_def is not None and sibling_def is not entry:
                sub_sites = _harvest_dep_call_sites(sibling_def, dep_bindings, own_tree, own_file, asts, files_root)
                if sub_sites:
                    for ss in sub_sites:
                        call_sites.append(DepCallSite(
                            dep_attr=ss.dep_attr,
                            method=ss.method,
                            citation=_rel_cite(own_file, node.lineno, files_root),
                            via=f"sibling:{sibling_name}"
                        ))

    return call_sites


def _find_sibling_method_def(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for stmt in ast.walk(tree):
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == name:
            return stmt
    return None


def _harvest_dep_usage_fields(
    entry: ast.FunctionDef | ast.AsyncFunctionDef | None,
    call_site_vars: dict[tuple[str, str], str],
) -> set[str]:
    """Extract field names accessed via .get() or subscript on dep result variables."""
    if entry is None or not call_site_vars:
        return set()

    fields = set()
    target_vars = set(call_site_vars.values())

    for node in ast.walk(entry):
        if isinstance(node, ast.Call):
            if (isinstance(node.func, ast.Attribute) and
                node.func.attr == "get" and
                isinstance(node.func.value, ast.Name) and
                node.func.value.id in target_vars and
                node.args and
                isinstance(node.args[0], ast.Constant) and
                isinstance(node.args[0].value, str)):
                fields.add(node.args[0].value)

        elif isinstance(node, ast.Subscript):
            if (isinstance(node.value, ast.Name) and
                node.value.id in target_vars and
                isinstance(node.slice, ast.Constant) and
                isinstance(node.slice.value, str)):
                fields.add(node.slice.value)

    return fields


def _dep_role_for_annotation(annotation: str | None, keywords_by_role: dict[str, list[str]]) -> str:
    """Match annotation against role keywords."""
    if not annotation:
        return "unknown"
    for role, keywords in keywords_by_role.items():
        for keyword in keywords:
            if keyword in annotation:
                return role
    return "unknown"


def _derive_case_binding(kwargs: list[KwargSpec]) -> dict[str, str]:
    """Derive case→kwarg bindings from archetype field-vocabulary (KWARG_CASE_KEY_MAPPING in archetype_vocabulary.py — the one canonical source), falling back to a same-name guess only when the vocabulary has no mapping."""
    binding: dict[str, str] = {}
    for spec in kwargs:
        if spec.name in _CONFIG_PARAMS:
            binding[spec.name] = f"config:{spec.name}"
        elif spec.required or spec.name in KWARG_CASE_KEY_MAPPING:
            if spec.name in KWARG_CASE_KEY_MAPPING:
                case_key = KWARG_CASE_KEY_MAPPING[spec.name]
                binding[spec.name] = f"case:$.input.{case_key}"
            else:
                binding[spec.name] = f"case:$.input.{spec.name}"
    return binding


def _derive_input_kind(kwargs: list[KwargSpec]) -> str:
    names = {s.name for s in kwargs}
    if names & _QUERYISH_PARAMS:
        return "query"
    if names - _CONFIG_PARAMS:
        return "structured"
    return "unknown"


def _archetype_for(contract: EvaluationContract) -> str:
    """Deterministic archetype classifier from harvested contract signals only — never a guess."""
    if contract.field_downstream_consumers:
        return "fan_in_judge"
    kwarg_names = {k.name for k in contract.invocation.kwargs} if contract.invocation else set()
    if not contract.has_retrieval_signal:
        # A non-retrieval agent with a real input contract still gets a dataset — "unimplemented" only when there's truly no usable input.
        case_binding = contract.invocation.case_binding if contract.invocation else {}
        config_names = config_kwarg_names_from_case_binding(case_binding)
        if not (kwarg_names - config_names):
            return "unimplemented"
    has_query_planning = contract.query_planning_subcall
    if has_query_planning:
        return "rag_query_planning"
    # Stage-2 truth: upstream_context_specs (populated in harvest_contracts pass 2) replaces the old hand-written kwarg blocklist.
    if contract.upstream_context_specs:
        return "rag_upstream"
    return "rag_single_shot"


def _rel_cite(path: Path, lineno: int, files_root: Path | None) -> str:
    p = path.as_posix()
    if files_root is not None:
        try:
            p = path.relative_to(files_root).as_posix()
        except ValueError:
            pass
    return f"{p}:{lineno}"


def _find_validator_letter(cls: ast.ClassDef | None) -> str | None:
    """First validate_*(<str-const>, ...) call inside the class body, e.g. validate_section("A", data)."""
    if cls is None:
        return None
    for node in ast.walk(cls):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
        if name.startswith("validate") and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                return first.value
    return None


def _entry_schema_scope(
    cls: ast.ClassDef, entry: ast.FunctionDef | ast.AsyncFunctionDef
) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """`entry` plus the transitive closure of in-class self.<helper>() methods it calls -- so a
    sibling entry method's schema constant on a multi-entry class (e.g. LangGraph nodes sharing one
    class) is never picked up."""
    methods_by_name = {
        node.name: node for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    scope = [entry]
    visited = {entry.name}
    frontier = [entry]
    depth = 0
    while frontier and depth < _MAX_SELF_CALL_DEPTH:
        depth += 1
        next_frontier = []
        for method in frontier:
            for called_name in _self_called_names(method):
                if called_name in visited:
                    continue
                callee = methods_by_name.get(called_name)
                if callee is None:
                    continue
                visited.add(called_name)
                scope.append(callee)
                next_frontier.append(callee)
        frontier = next_frontier
    return scope


def _referenced_schema_names(
    cls: ast.ClassDef | None, entry: ast.FunctionDef | ast.AsyncFunctionDef | None = None
) -> list[str]:
    """Schema-constant-shaped names (*SCHEMA*, case-insensitive) referenced within scope. Scoped to `entry`'s
    own closure when available; falls back to the whole class only when no entry is resolvable at all."""
    scope_nodes: list[ast.AST] = []
    if cls is not None:
        scope_nodes = _entry_schema_scope(cls, entry) if entry is not None else [cls]
    elif entry is not None:
        scope_nodes = [entry]
    names = []
    for scope_node in scope_nodes:
        for node in ast.walk(scope_node):
            if isinstance(node, ast.Name) and _SCHEMA_NAME_RE.search(node.id):
                names.append(node.id)
            elif isinstance(node, ast.Attribute) and _SCHEMA_NAME_RE.search(node.attr):
                names.append(node.attr)
    return names


def _import_source_file(
    tree: ast.Module, name: str, asts: dict[Path, ast.Module]
) -> Path | None:
    """Resolve `from a.b import NAME [as alias]` in `tree` to the file backing it, if present."""
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if (alias.asname or alias.name) == name:
                    rel = "/".join(node.module.split(".")) + ".py"
                    for path in asts:
                        if path.as_posix().endswith(rel):
                            return path
    return None


def _load_import_target_from_disk(
    own_tree: ast.Module, name: str, own_file: Path | None,
    files_root: Path | None, asts: dict[Path, ast.Module],
) -> bool:
    """Best-effort: resolves `from [.]mod import name` to a .py on disk and adds it to `asts` so `_resolve_callable` can follow into an imported free function. Returns True if newly added."""
    if files_root is None or own_file is None:
        return False
    for node in own_tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if not any((a.asname or a.name) == name for a in node.names):
            continue
        mod = node.module or ""
        level = node.level or 0
        candidates: list[Path] = []
        if level > 0:  # relative import — resolve against own_file's package
            base = (files_root / own_file).parent
            for _ in range(level - 1):
                base = base.parent
            parts = mod.split(".") if mod else []
            candidates.append(base.joinpath(*parts).with_suffix(".py") if parts else base / f"{name}.py")
        elif mod:  # absolute import — the module's source root may be the repo root or any
            # top-level source directory (src, app, lib, server, ...). Try each generically —
            # no framework/layout-specific name is assumed.
            rel = Path(*mod.split(".")).with_suffix(".py")
            candidates.append(files_root / rel)
            try:
                for child in sorted(files_root.iterdir()):
                    if (child.is_dir() and not child.name.startswith(".")
                            and child.name not in ("node_modules", "__pycache__", "venv")):
                        candidates.append(child / rel)
            except Exception:
                pass
        for disk in candidates:
            try:
                if disk.exists():
                    tree_mod = ast.parse(disk.read_text(encoding="utf-8"))
                    key = Path("/".join(mod.split("."))).with_suffix(".py") if mod else Path(f"{name}.py")
                    if key not in asts:
                        asts[key] = tree_mod
                    return True
            except Exception:
                continue
    return False


def _find_name_assignment(tree: ast.Module, name: str) -> tuple[ast.expr, int] | None:
    """Returns (value_node, lineno) for a module-level `NAME = ...` / `NAME: T = ...`."""
    for node in tree.body:
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        for target in targets:
            if isinstance(target, ast.Name) and target.id == name:
                return value, node.lineno
    return None


def _name_search_order(
    asts: dict[Path, ast.Module], name: str, own_file: Path | None
) -> list[Path]:
    """File search order for resolving a module-level NAME: own_file's import source, then own_file, then every other parsed file — shared by the str-constant/dict-literal resolvers below."""
    search_order: list[Path] = []
    if own_file is not None and own_file in asts:
        imported_from = _import_source_file(asts[own_file], name, asts)
        if imported_from is not None:
            search_order.append(imported_from)
        search_order.append(own_file)
    for path in asts:
        if path not in search_order:
            search_order.append(path)
    return search_order


def _find_module_dict_or_str_constant(
    asts: dict[Path, ast.Module],
    name: str,
    files_root: Path | None,
    *,
    own_file: Path | None = None,
) -> tuple[dict | str, str] | None:
    """Find a module-level `NAME = <str literal | dict literal>`, preferring own_file's import source then own_file, else scan others."""
    for path in _name_search_order(asts, name, own_file):
        found = _find_name_assignment(asts[path], name)
        if found is None:
            continue
        value, lineno = found
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value, _rel_cite(path, lineno, files_root)
        if isinstance(value, ast.Dict):
            try:
                dict_val = ast.literal_eval(value)
                if isinstance(dict_val, dict):
                    return dict_val, _rel_cite(path, lineno, files_root)
            except Exception:
                pass
        return None
    return None


def _find_module_str_constant(
    asts: dict[Path, ast.Module],
    name: str,
    files_root: Path | None,
    *,
    own_file: Path | None = None,
) -> tuple[str, str] | None:
    res = _find_module_dict_or_str_constant(asts, name, files_root, own_file=own_file)
    if res and isinstance(res[0], str):
        return res[0], res[1]
    return None


_TYPE_MAP = {
    "str": {"type": "string"},
    "int": {"type": "integer"},
    "float": {"type": "number"},
    "bool": {"type": "boolean"},
    "dict": {"type": "object"},
    "list": {"type": "array"},
}


def _annotation_to_schema(node: ast.expr, ctx: _SchemaResolveCtx | None = None) -> dict:
    # X | None  (PEP 604, Python 3.10+) — resolve the non-None branch
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left, right = node.left, node.right
        is_none_right = isinstance(right, ast.Constant) and right.value is None
        is_none_left = isinstance(left, ast.Constant) and left.value is None
        if is_none_right:
            return _annotation_to_schema(left, ctx)
        if is_none_left:
            return _annotation_to_schema(right, ctx)
        # Both non-None union: resolve left (first branch)
        return _annotation_to_schema(left, ctx)

    if isinstance(node, ast.Name):
        base_schema = dict(_TYPE_MAP.get(node.id, {}))
        if not base_schema and ctx is not None and node.id not in ctx.visited:
            resolved = _resolve_class_schema(node.id, ctx)
            if resolved:
                return resolved[0]
        return base_schema
    if isinstance(node, ast.Subscript):
        base_name = _subscript_base_name(node)
        if base_name in ("list", "List"):
            item = _annotation_to_schema(node.slice, ctx) if not isinstance(node.slice, ast.Tuple) else {}
            schema: dict = {"type": "array"}
            if item:
                schema["items"] = item
            return schema
        if base_name in ("dict", "Dict"):
            # dict[K, V] — emit additionalProperties for the value type
            if isinstance(node.slice, ast.Tuple) and len(node.slice.elts) == 2:
                val_schema = _annotation_to_schema(node.slice.elts[1], ctx)
                if val_schema:
                    return {"type": "object", "additionalProperties": val_schema}
            return {"type": "object"}
        if base_name == "Literal":
            elts = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            values = [e.value for e in elts if isinstance(e, ast.Constant)]
            if values and all(isinstance(v, str) for v in values):
                return {"type": "string", "enum": values}
        if base_name in ("NotRequired", "Required"):
            return _annotation_to_schema(node.slice, ctx)
        if base_name == "Optional":
            # Optional[X] → resolve X (same as X | None)
            return _annotation_to_schema(node.slice, ctx)
    return {}



def _infer_schema_from_literal(literal: dict, ctx: _SchemaResolveCtx | None = None) -> dict | None:
    """Infer a JSON schema from a dict literal by examining value types."""
    if not literal:
        return None
    properties: dict = {}
    for key, val in literal.items():
        if val == _DYNAMIC:
            properties[key] = {}
        elif isinstance(val, bool):
            properties[key] = {"type": "boolean"}
        elif isinstance(val, int):
            properties[key] = {"type": "integer"}
        elif isinstance(val, float):
            properties[key] = {"type": "number"}
        elif isinstance(val, str):
            properties[key] = {"type": "string"}
        elif isinstance(val, list):
            inner = {"type": "array"}
            if val and isinstance(val[0], dict) and ctx:
                inner_schema = _infer_schema_from_literal(val[0], ctx)
                if inner_schema:
                    inner["items"] = inner_schema
            properties[key] = inner
        elif isinstance(val, dict):
            inner_schema = {"type": "object"}
            if val and ctx:
                nested = _infer_schema_from_literal(val, ctx)
                if nested:
                    inner_schema = nested
            properties[key] = inner_schema
        else:
            properties[key] = {}
    return {"type": "object", "properties": properties, "required": list(literal.keys())}


def _subscript_base_name(node: ast.expr) -> str:
    if isinstance(node, ast.Subscript):
        base = node.value
        return base.id if isinstance(base, ast.Name) else base.attr if isinstance(base, ast.Attribute) else ""
    return ""


def _is_notrequired(node: ast.expr) -> bool:
    return _subscript_base_name(node) == "NotRequired"


def _is_required(node: ast.expr) -> bool:
    return _subscript_base_name(node) == "Required"


def _extract_state_assigned_keys(entry: ast.AST) -> set[str]:
    """Extract set of state keys assigned/mutated in a state-node function body."""
    assigned: set[str] = set()
    parent_map: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(entry):
        for child in ast.iter_child_nodes(parent):
            parent_map[child] = parent

    for sub in ast.walk(entry):
        if isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name) and sub.value.id in ("state", "s"):
            key = sub.slice.value if isinstance(sub.slice, ast.Constant) and isinstance(sub.slice.value, str) else None
            if key:
                if isinstance(getattr(sub, "ctx", None), ast.Store):
                    assigned.add(key)
                else:
                    parent = parent_map.get(sub)
                    if isinstance(parent, ast.Attribute) and parent.attr in ("append", "extend", "update", "setdefault"):
                        grandparent = parent_map.get(parent)
                        if isinstance(grandparent, ast.Call) and grandparent.func == parent:
                            assigned.add(key)
                    elif isinstance(parent, ast.AugAssign) and parent.target == sub:
                        assigned.add(key)
                    elif isinstance(parent, ast.Subscript) and parent.value == sub and isinstance(getattr(parent, "ctx", None), ast.Store):
                        assigned.add(key)
        elif isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and isinstance(sub.func.value, ast.Name) and sub.func.value.id in ("state", "s"):
            if sub.func.attr == "update" and sub.args and isinstance(sub.args[0], ast.Dict):
                for k in sub.args[0].keys:
                    if isinstance(k, ast.Constant) and isinstance(k.value, str):
                        assigned.add(k.value)
    return assigned


def _extract_state_read_keys(entry: ast.AST) -> set[str]:
    """State keys a state-node function READS: Load-context `state["k"]` subscripts and
    `state.get("k")` calls. Read-modify-write keys (their subscript is Load before the mutation)
    are naturally included. Used to restrict the node's input schema to what it actually consumes."""
    read: set[str] = set()
    for sub in ast.walk(entry):
        if isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name) and sub.value.id in ("state", "s"):
            if isinstance(sub.slice, ast.Constant) and isinstance(sub.slice.value, str):
                if not isinstance(getattr(sub, "ctx", None), ast.Store):
                    read.add(sub.slice.value)
        elif (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr == "get"
            and isinstance(sub.func.value, ast.Name)
            and sub.func.value.id in ("state", "s")
            and sub.args
            and isinstance(sub.args[0], ast.Constant)
            and isinstance(sub.args[0].value, str)
        ):
            read.add(sub.args[0].value)
    return read


def _infer_untyped_list_item_schema(
    field_name: str, ctx: _SchemaResolveCtx | None, prefer_path: Path | None = None
) -> dict | None:
    """Infer item schema for an untyped list[dict] state field from nearby AST append literals or prompt JSON templates.
    Searches prefer_path's file first (the file that owns the field's TypedDict) so a same-file template wins over a
    same-shaped template in an unrelated file (e.g. another agent's `"steps": [...]` prompt)."""
    if not ctx or not ctx.asts or not field_name:
        return None
    ordered = ([ctx.asts[prefer_path]] if prefer_path is not None and prefer_path in ctx.asts else []) + [
        t for p, t in ctx.asts.items() if p != prefer_path
    ]
    for tree in ordered:
        for sub in ast.walk(tree):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr == "append":
                target_name = ""
                if isinstance(sub.func.value, ast.Subscript) and isinstance(sub.func.value.slice, ast.Constant):
                    target_name = str(sub.func.value.slice.value)
                elif isinstance(sub.func.value, ast.Attribute):
                    target_name = sub.func.value.attr
                elif isinstance(sub.func.value, ast.Name):
                    target_name = sub.func.value.id
                if target_name == field_name and sub.args and isinstance(sub.args[0], ast.Dict):
                    literal = {}
                    for k, v in zip(sub.args[0].keys, sub.args[0].values):
                        if isinstance(k, ast.Constant) and isinstance(k.value, str):
                            literal[k.value] = _literal_eval_or_dynamic(v)
                    if literal:
                        inferred = _infer_schema_from_literal(literal, ctx)
                        if inferred:
                            return inferred

    import re
    if field_name == "plan":
        pattern = r'"(?:steps|plan)"\s*:\s*\[\s*(\{.*?\})\s*\]'
    else:
        field_pattern = re.escape(field_name)
        pattern = rf'"{field_pattern}"\s*:\s*\[\s*(\{{.*?\}})\s*\]'
    for tree in ordered:
        for sub in ast.walk(tree):
            if isinstance(sub, ast.Assign) and isinstance(sub.value, ast.Constant) and isinstance(sub.value.value, str):
                text = sub.value.value
                steps_match = re.search(pattern, text, re.DOTALL)
                if steps_match:
                    raw_template = steps_match.group(1)
                    prop_keys = []
                    try:
                        parsed_item = json.loads(raw_template)
                        if isinstance(parsed_item, dict):
                            prop_keys = list(parsed_item.keys())
                    except Exception:
                        prop_keys = re.findall(r'"([a-zA-Z_][a-zA-Z0-9_]*)"\s*:', raw_template)
                    if prop_keys:
                        props = {}
                        for pk in prop_keys:
                            if pk == "type" and "trace_forward" in text:
                                props[pk] = {"type": "string", "enum": ["retrieve", "trace_forward", "trace_backward", "impact"]}
                            else:
                                props[pk] = {"type": "string"}
                        return {"type": "object", "properties": props, "required": list(props.keys())}
    return None


def _resolve_class_schema(name: str, ctx: _SchemaResolveCtx) -> tuple[dict, str] | None:
    """Resolve a class schema by name (TypedDict/Pydantic/dataclass); returns (schema, source) or None if unresolvable. Cycle-guarded by visited set, depth-capped."""
    if ctx.depth >= _MAX_SCHEMA_DEPTH or name in ctx.visited:
        return None
    ctx.visited.add(name)
    cls = None
    for path, tree in list(ctx.asts.items()):
        cls = _find_class(tree, name)
        if cls is not None:
            break

    if cls is None and ctx.files_root and ctx.files_root.exists():
        candidate_paths = [
            ctx.files_root / "backend/domain/analysis/schemas.py",
            ctx.files_root / "schemas.py",
        ]
        for cand in candidate_paths:
            if cand.is_file():
                try:
                    rel_p = cand.relative_to(ctx.files_root) if ctx.files_root in cand.parents else cand
                    if rel_p not in ctx.asts and cand not in ctx.asts:
                        parsed = parse_python_source(cand)
                        if parsed:
                            ctx.asts[rel_p] = parsed[1]
                except Exception:
                    pass

        for search_dir in [ctx.files_root / "backend", ctx.files_root / "src"]:
            if search_dir.is_dir():
                for py_file in search_dir.rglob("*.py"):
                    try:
                        rel_p = py_file.relative_to(ctx.files_root) if ctx.files_root in py_file.parents else py_file
                        if rel_p not in ctx.asts and py_file not in ctx.asts:
                            content = py_file.read_text(encoding="utf-8", errors="ignore")
                            if f"class {name}" in content:
                                parsed = parse_python_source(py_file)
                                if parsed:
                                    ctx.asts[rel_p] = parsed[1]
                                    break
                    except Exception:
                        pass

    for path, tree in list(ctx.asts.items()):
        cls = _find_class(tree, name)
        if cls is None:
            continue
        base_names = {b.id if isinstance(b, ast.Name) else getattr(b, "attr", "") for b in cls.bases}
        properties: dict = {}
        required: list[str] = []
        is_typeddict = "TypedDict" in base_names
        is_pydantic = "BaseModel" in base_names
        is_dataclass = any(
            isinstance(d, ast.Name) and d.id == "dataclass"
            or isinstance(d, ast.Call) and (isinstance(d.func, ast.Name) and d.func.id == "dataclass")
            for d in cls.decorator_list
        )
        # Class-syntax NamedTuple only; the functional form (X = NamedTuple("X", [...])) is an
        # ast.Assign, never an ast.ClassDef, so _find_class already can't match it -- degrades to
        # unresolvable (needs_human downstream), never guessed.
        is_namedtuple = "NamedTuple" in base_names
        if not (is_typeddict or is_pydantic or is_dataclass or is_namedtuple):
            continue
        total = True
        for kw in cls.keywords:
            if kw.arg == "total" and isinstance(kw.value, ast.Constant):
                total = bool(kw.value.value)
        for item in cls.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                key = item.target.id
                # Fresh visited copy per field so two sibling fields of the SAME nested type both
                # resolve -- a shared visited set would let the first field's type-name block the
                # second (e.g. all six ConventionAspect aspects). Cycle detection still holds down
                # each field's own recursion path.
                field_ctx = _SchemaResolveCtx(
                    asts=ctx.asts, files_root=ctx.files_root, visited=set(ctx.visited),
                    depth=ctx.depth + 1, conventions=ctx.conventions,
                )
                prop_schema = _annotation_to_schema(item.annotation, field_ctx)
                items = prop_schema.get("items") if isinstance(prop_schema.get("items"), dict) else None
                item_is_opaque = items is not None and items.get("type") == "object" and not items.get("properties")
                if prop_schema.get("type") == "array" and (items is None or item_is_opaque):
                    inferred_items = _infer_untyped_list_item_schema(key, field_ctx, prefer_path=path)
                    if inferred_items:
                        prop_schema = dict(prop_schema)
                        prop_schema["items"] = inferred_items
                properties[key] = prop_schema
                if is_typeddict:
                    is_req = not _is_notrequired(item.annotation) if total else _is_required(item.annotation)
                elif is_pydantic or is_dataclass or is_namedtuple:
                    is_req = item.value is None
                else:
                    is_req = False
                if is_req:
                    required.append(key)
        schema = {"type": "object", "properties": properties, "required": required}
        return schema, _rel_cite(path, cls.lineno, ctx.files_root) + f" {name}"
    return None


def _self_called_names(method: ast.FunctionDef | ast.AsyncFunctionDef | None) -> set[str]:
    """Method names invoked as `self.<name>(...)` inside `method`, to prefer the fallback actually called by the entry point."""
    if method is None:
        return set()
    return {
        node.func.attr
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
    }


def _literal_eval_or_dynamic(node: ast.expr) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return _DYNAMIC


def _extract_dict_return_literal(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict | None:
    """First `return {...}` dict literal in a function body; non-constant values become '<dynamic>'."""
    for ret in ast.walk(node):
        if isinstance(ret, ast.Return) and isinstance(ret.value, ast.Dict):
            literal: dict = {}
            for k, v in zip(ret.value.keys, ret.value.values):
                if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                    continue
                literal[k.value] = _literal_eval_or_dynamic(v)
            if literal:
                return literal
    return None


def _extract_list_return_literal(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list | None:
    """First `return [...]` list literal in a function body -- the list-return sibling of
    _extract_dict_return_literal, so a list-returning entry point can still get an output schema. A
    dict element resolves key-by-key like the dict-return case; any other element becomes its own
    ast.literal_eval() or '<dynamic>'."""
    for ret in ast.walk(node):
        if isinstance(ret, ast.Return) and isinstance(ret.value, ast.List):
            items = []
            for elt in ret.value.elts:
                if isinstance(elt, ast.Dict):
                    item: dict = {}
                    for k, v in zip(elt.keys, elt.values):
                        if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                            continue
                        item[k.value] = _literal_eval_or_dynamic(v)
                    items.append(item)
                else:
                    items.append(_literal_eval_or_dynamic(elt))
            if items:
                return items
    return None


def _infer_array_schema_from_literal(items: list, ctx: _SchemaResolveCtx | None = None) -> dict | None:
    """Infer a JSON schema from a `return [...]` list literal: array-of-object when the first element
    is a (non-empty) dict literal, resolved the same way _infer_schema_from_literal resolves a
    dict-return's own fields; else a bare array with no item shape."""
    if not items:
        return None
    schema: dict = {"type": "array"}
    first = items[0]
    if isinstance(first, dict) and first:
        item_schema = _infer_schema_from_literal(first, ctx)
        if item_schema:
            schema["items"] = item_schema
    return schema


def _find_pipeline_fallback_function(
    asts: dict[Path, ast.Module], letter: str, files_root: Path | None,
    conventions: ContractConventions | None = None,
) -> tuple[dict, str] | None:
    """Fallthrough for agents without their own try/except (e.g. K/L): they have a module-level `_section_<letter>_pipeline_fallback()` instead."""
    pattern = (conventions or ContractConventions()).pipeline_fallback_name_pattern
    if pattern is None:
        return None
    target_name = pattern.format(letter=letter.lower())
    for path, tree in asts.items():
        fn = _find_function(tree, target_name)
        if fn is None:
            continue
        literal = _extract_dict_return_literal(fn)
        if literal:
            return literal, _rel_cite(path, fn.lineno, files_root)
    return None


def _harvest_fallback(
    cls: ast.ClassDef | None,
    path: Path,
    files_root: Path | None,
    *,
    letter: str | None = None,
    asts: dict[Path, ast.Module] | None = None,
    conventions: ContractConventions | None = None,
) -> tuple[dict | None, str | None, bool]:
    """Fallback method returning a dict literal; disambiguates candidates by entry-point call, then name match, then order. Returned bool flags genuine ambiguity for human review."""
    candidates = [
        node for node in (cls.body if cls is not None else [])
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and "fallback" in node.name.lower()
    ]
    if not candidates:
        if letter and asts:
            found = _find_pipeline_fallback_function(asts, letter, files_root, conventions)
            if found:
                literal, source = found
                return literal, source, False
        return None, None, False

    called_names = _self_called_names(_find_entry_method(cls))

    def _rank(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
        if node.name in called_names:
            return 0
        if node.name.lower() in ("fallback", "_fallback"):
            return 1
        return 2

    candidates.sort(key=_rank)
    ambiguous = len(candidates) > 1

    for node in candidates:
        literal = _extract_dict_return_literal(node)
        if literal:
            return literal, _rel_cite(path, node.lineno, files_root), ambiguous
    return None, None, False


def _resolve_callable_with_disk_fallback(
    name: str, own_tree: ast.Module, own_file: Path, asts: dict[Path, ast.Module], files_root: Path | None,
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, Path] | None:
    """_resolve_callable, retried once after loading the call's likely import target from disk --
    shared by dependency-call-site harvesting and enum-normalizer-function resolution."""
    resolved = _resolve_callable(name, own_tree, own_file, asts)
    if resolved is None and _load_import_target_from_disk(own_tree, name, own_file, files_root, asts):
        resolved = _resolve_callable(name, own_tree, own_file, asts)
    return resolved


def _string_set_literal(node: ast.expr) -> set[str] | None:
    """An inline set()/frozenset() call, or a bare {...} literal, made entirely of string constants
    -- the value-domain of an enum-whitelist guard. None when the shape or element types don't match."""
    target = node
    if isinstance(target, ast.Call) and isinstance(target.func, ast.Name) and target.func.id in ("set", "frozenset"):
        if not target.args:
            return set()
        target = target.args[0]
    if isinstance(target, ast.Set):
        values = [e.value for e in target.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
        if values and len(values) == len(target.elts):
            return set(values)
    return None


def _resolve_enum_set_ref(
    node: ast.expr, own_tree: ast.Module, cls: ast.ClassDef | None
) -> set[str] | None:
    """Resolves the RHS of an `in`/`not in` enum-whitelist check: an inline set/frozenset literal
    (_string_set_literal), or a bare Name bound to a module- or class-level frozenset/set-of-strings
    constant."""
    inline = _string_set_literal(node)
    if inline is not None:
        return inline
    if not isinstance(node, ast.Name):
        return None
    for body in [own_tree.body] + ([cls.body] if cls is not None else []):
        for stmt in body:
            target = value = None
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target, value = stmt.targets[0], stmt.value
            elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                target, value = stmt.target, stmt.value
            if isinstance(target, ast.Name) and target.id == node.id:
                resolved = _string_set_literal(value)
                if resolved is not None:
                    return resolved
    return None


def _dict_key_for_name_in_stmts(stmts: list[ast.stmt], var_name: str) -> str | None:
    """First dict-literal string key, within this statement list only, whose value is the bare name
    `var_name` or a single-arg call wrapping it (e.g. str(var_name)) -- the output field a coerced
    local variable is written into."""
    for stmt in stmts:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for node in ast.walk(stmt):
            if not isinstance(node, ast.Dict):
                continue
            for k, v in zip(node.keys, node.values):
                if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                    continue
                target = v.args[0] if (isinstance(v, ast.Call) and len(v.args) == 1) else v
                if isinstance(target, ast.Name) and target.id == var_name:
                    return k.value
    return None


def _scan_stmts_for_enum_domains(
    stmts: list[ast.stmt], own_tree: ast.Module, cls: ast.ClassDef | None, result: dict[str, set[str]],
) -> None:
    """Recursively finds `if X not in ENUM: X = "fallback"` re-coercion guards (e.g. a folder-role or
    rule-strength normalizer written inline in a post-processing loop) and links each to its OWN
    nearest dict-literal write of X, one statement list (e.g. one for-loop body) at a time -- so two
    sibling loops guarding same-named locals (e.g. two loops both using a local named `sev`) never
    cross-link to the wrong enum. Mirrors _trace_stmts's per-block recursive-descent shape."""
    guards: list[tuple[str, set[str]]] = []
    for stmt in stmts:
        if isinstance(stmt, ast.If) and stmt.body:
            test = stmt.test
            if (isinstance(test, ast.Compare) and len(test.ops) == 1
                    and isinstance(test.ops[0], ast.NotIn) and isinstance(test.left, ast.Name)):
                first = stmt.body[0]
                if (isinstance(first, ast.Assign) and len(first.targets) == 1
                        and isinstance(first.targets[0], ast.Name)
                        and first.targets[0].id == test.left.id
                        and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)):
                    values = _resolve_enum_set_ref(test.comparators[0], own_tree, cls)
                    if values:
                        guards.append((test.left.id, values | {first.value.value}))
        for attr in ("body", "orelse", "finalbody"):
            child = getattr(stmt, attr, None)
            if child:
                _scan_stmts_for_enum_domains(child, own_tree, cls, result)
        for handler in getattr(stmt, "handlers", None) or []:
            _scan_stmts_for_enum_domains(handler.body, own_tree, cls, result)

    for var_name, values in guards:
        field = _dict_key_for_name_in_stmts(stmts, var_name)
        if field:
            result.setdefault(field, values)


def _enum_normalizer_call_sites(scope_nodes: list[ast.AST]) -> list[tuple[str, str]]:
    """(function_name, output_field) pairs for every place in scope a bare-name call's result is
    written to a specific output field -- directly (`data["x"] = fn(...)`), inside a dict-literal
    value (`{"x": fn(...)}`), or as a dict-comprehension's per-value transform for a dict[str, str]
    -shaped field (`data["x"] = {k: fn(v) for k, v in ...}`)."""
    pairs: list[tuple[str, str]] = []
    for scope_node in scope_nodes:
        for node in ast.walk(scope_node):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if not (isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant)
                        and isinstance(target.slice.value, str)):
                    continue
                value = node.value
                if isinstance(value, ast.DictComp):
                    value = value.value
                if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
                    pairs.append((value.func.id, target.slice.value))
            elif isinstance(node, ast.Dict):
                for k, v in zip(node.keys, node.values):
                    if (isinstance(k, ast.Constant) and isinstance(k.value, str)
                            and isinstance(v, ast.Call) and isinstance(v.func, ast.Name)):
                        pairs.append((v.func.id, k.value))
    return pairs


def _return_based_enum_normalizer_values(
    fn: ast.FunctionDef | ast.AsyncFunctionDef, fn_module_tree: ast.Module,
) -> set[str] | None:
    """Detects the coerce-to-enum shape `if X in ENUM: return X` immediately followed by a
    fallthrough `return "<fallback>"` (e.g. a `_normalize_conf`-style helper). Returns the
    allowed-value set (whitelist plus the fallback) or None when the body doesn't match."""
    body = fn.body
    for i, stmt in enumerate(body[:-1]):
        if not isinstance(stmt, ast.If) or stmt.orelse or len(stmt.body) != 1:
            continue
        test = stmt.test
        if not (isinstance(test, ast.Compare) and len(test.ops) == 1
                and isinstance(test.ops[0], ast.In) and isinstance(test.left, ast.Name)):
            continue
        ret = stmt.body[0]
        if not (isinstance(ret, ast.Return) and isinstance(ret.value, ast.Name)
                and ret.value.id == test.left.id):
            continue
        fallthrough = body[i + 1]
        if not (isinstance(fallthrough, ast.Return) and isinstance(fallthrough.value, ast.Constant)
                and isinstance(fallthrough.value.value, str)):
            continue
        values = _resolve_enum_set_ref(test.comparators[0], fn_module_tree, None)
        if values:
            return values | {fallthrough.value.value}
    return None


def _harvest_schema_enum_values(
    scope_nodes: list[ast.AST],
    cls: ast.ClassDef | None,
    own_tree: ast.Module,
    own_file: Path,
    asts: dict[Path, ast.Module],
    files_root: Path | None,
) -> dict[str, list[str]]:
    """Field -> allowed literal values, from post-processing enum-whitelist guards reachable from the
    entry method's own scope: an inline `if X not in ENUM: X = "fallback"` re-coercion, or a call to a
    small `if X in ENUM: return X` / `return "fallback"` normalizer helper, resolved wherever it's
    actually defined (same file or another already-parsed one). AST/type-driven only -- no field or
    agent name is ever assumed."""
    domains: dict[str, set[str]] = {}
    for scope_node in scope_nodes:
        body = getattr(scope_node, "body", None)
        if body:
            _scan_stmts_for_enum_domains(list(body), own_tree, cls, domains)

    for func_name, field in _enum_normalizer_call_sites(scope_nodes):
        if field in domains:
            continue
        resolved = _resolve_callable_with_disk_fallback(func_name, own_tree, own_file, asts, files_root)
        if resolved is None:
            continue
        fn_def, fn_file = resolved
        values = _return_based_enum_normalizer_values(fn_def, asts.get(fn_file, own_tree))
        if values:
            domains[field] = values

    return {field: sorted(values) for field, values in domains.items()}


def _scope_has_streaming_call(scope_nodes: list[ast.AST]) -> bool:
    """A generic, name-based signal (not framework-specific): any call in scope whose own
    function/method name contains 'stream' -- the same naming convention _GENERATION_VERBS already
    treats as a generic generation verb elsewhere in this harvest (discovery/enrichment.py)."""
    for scope_node in scope_nodes:
        for node in ast.walk(scope_node):
            if not isinstance(node, ast.Call):
                continue
            name = (
                node.func.attr if isinstance(node.func, ast.Attribute)
                else node.func.id if isinstance(node.func, ast.Name)
                else ""
            )
            if "stream" in name.lower():
                return True
    return False


def _harvest_constants(tree: ast.Module, cls: ast.ClassDef | None) -> dict[str, int]:
    """UPPER_CASE int constants at module level and on the agent's class."""
    constants: dict[str, int] = {}
    scopes: list[list[ast.stmt]] = [tree.body] + ([cls.body] if cls is not None else [])
    for body in scopes:
        for node in body:
            target: ast.expr | None = None
            value: ast.expr | None = None
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target, value = node.targets[0], node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                target, value = node.target, node.value
            if (
                isinstance(target, ast.Name)
                and target.id.isupper()
                and isinstance(value, ast.Constant)
                and isinstance(value.value, int)
                and not isinstance(value.value, bool)
            ):
                constants[target.id] = value.value
    return constants


def harvest_component_contract(
    component: Component,
    asts: dict[Path, ast.Module],
    files_root: Path | None = None,
    conventions: ContractConventions | None = None,
) -> tuple[InvocationContract | None, OutputContract | None, dict[str, int], list[str], str]:
    """Returns (invocation, output, constants, needs_human_notes, input_kind) for one component."""
    notes: list[str] = []
    ctx = _SchemaResolveCtx(
        asts=asts, files_root=files_root, visited=set(), depth=0, conventions=conventions
    )
    path, cls, entry = _resolve_entry(component, asts)
    if path is None:
        return None, None, {}, [f"{component.id}: source file not found for static harvest"], "unknown"
    tree = asts[path]
    if cls is None and entry is None:
        return None, None, {}, [f"{component.id}: entry point not resolvable in {path.name}"], "unknown"
    entry_is_async = isinstance(entry, ast.AsyncFunctionDef) if entry is not None else False

    # A validator letter also identifies a per-agent-route invocation path, not just the output TypedDict.
    letter = _find_validator_letter(cls)  # None-tolerant for function entry
    invocation: InvocationContract | None = None
    input_kind = "unknown"
    if entry is not None:
        kwargs, kw_notes = _harvest_kwargs(entry)
        notes.extend(f"{component.id}: {n}" for n in kw_notes)
        constructor_deps = _harvest_constructor_deps(cls)
        input_kind = _derive_input_kind(kwargs)
        # In-process is the only mode that can pass a case's kwargs AND swap a retrieval dependency for a stub — a per-section route resolves its own state from a stored report and would run against the real repo, ignoring the case entirely.
        invocation_mode = "in_harness"
        route = (conventions or ContractConventions()).rerun_section_route if letter else None
        case_binding = _derive_case_binding(kwargs)
        if route:
            notes.append(
                f"{component.id}: a per-section route ({route}) exists but is not used for "
                "evaluation — it rebuilds live dependencies and cannot accept a case's input"
            )
        invocation = InvocationContract(
            callable=component.entry_point,
            method=entry.name,
            kwargs=kwargs,
            constructor_deps=constructor_deps,
            invocation_mode=invocation_mode,
            route=route,
            case_binding=case_binding,
            source="ast",
            citations=[_rel_cite(path, entry.lineno, files_root)],
        )
    else:
        notes.append(f"{component.id}: no entry function/method found")

    # Input contract: for a state-node (entry takes and returns the same state TypedDict), restrict the
    # state kwarg's schema to the keys the node actually READS -- symmetric to the output write-delta --
    # so the generator never has to synthesise the full multi-field state (which bloats the prompt and
    # risks truncation). Fresh ctx (own visited set) so this resolution never poisons the output block's.
    if invocation is not None and isinstance(entry, (ast.FunctionDef, ast.AsyncFunctionDef)):
        param_types = {
            arg.arg: (arg.annotation.id if isinstance(arg.annotation, ast.Name) else "")
            for arg in (entry.args.args + entry.args.kwonlyargs)
            if arg.annotation is not None
        }
        ret_type_name = entry.returns.id if isinstance(entry.returns, ast.Name) else ""
        state_arg_names = {a for a, t in param_types.items() if t == ret_type_name}
        if ret_type_name and state_arg_names:
            input_ctx = _SchemaResolveCtx(
                asts=asts, files_root=files_root, visited=set(), depth=0, conventions=conventions
            )
            state_res = _resolve_class_schema(ret_type_name, input_ctx)
            if state_res and isinstance(state_res[0], dict) and state_res[0].get("properties"):
                all_props = state_res[0]["properties"]
                read_keys = _extract_state_read_keys(entry)
                subset = {k: all_props[k] for k in read_keys if k in all_props}
                if subset:
                    reqs = [k for k in state_res[0].get("required", []) if k in subset]
                    restricted = {"type": "object", "properties": subset, "required": reqs}
                    for kw in invocation.kwargs:
                        if kw.name in state_arg_names:
                            kw.resolved_schema = restricted

    # Output contract: validator letter -> Section{letter} TypedDict; else state mutation delta; else referenced schema constant; else return dict/annotation fallback.
    json_schema: dict | None = None
    schema_source: str | None = None
    if letter:
        found = _resolve_class_schema(f"Section{letter}", ctx)
        if found:
            json_schema, schema_source = found

    # State-node check: entry function taking and returning state TypedDict -> state mutation delta
    if json_schema is None and entry is not None and isinstance(entry, (ast.FunctionDef, ast.AsyncFunctionDef)):
        param_type_names = [
            arg.annotation.id if isinstance(arg.annotation, ast.Name) else ""
            for arg in (entry.args.args + entry.args.kwonlyargs)
            if arg.annotation is not None
        ]
        ret_type_name = entry.returns.id if isinstance(entry.returns, ast.Name) else ""
        if ret_type_name and ret_type_name in param_type_names:
            state_schema = _resolve_class_schema(ret_type_name, ctx)
            if state_schema and isinstance(state_schema[0], dict) and state_schema[0].get("properties"):
                assigned_keys = _extract_state_assigned_keys(entry)
                if assigned_keys:
                    all_props = state_schema[0]["properties"]
                    delta_props = {k: all_props[k] for k in assigned_keys if k in all_props}
                    if delta_props:
                        reqs = [k for k in state_schema[0].get("required", []) if k in delta_props]
                        json_schema = {"type": "object", "properties": delta_props, "required": reqs}
                        schema_source = _rel_cite(path, entry.lineno, files_root) + " (state mutation delta)"

    if json_schema is None:
        for name in _referenced_schema_names(cls, entry):
            found_const = _find_module_dict_or_str_constant(asts, name, files_root, own_file=path)
            if found_const is None:
                notes.append(f"{component.id}: schema constant {name} is dynamically assembled — unresolvable")
                continue
            val, cite = found_const
            if isinstance(val, dict):
                json_schema, schema_source = val, cite + f" {name}"
                break
            elif isinstance(val, str):
                try:
                    parsed = json.loads(val)
                    if isinstance(parsed, dict):
                        json_schema, schema_source = parsed, cite + f" {name}"
                        break
                except json.JSONDecodeError:
                    schema_source = schema_source or (cite + f" {name} (non-JSON template)")

    if json_schema is None and entry is not None:
        dict_literal = _extract_dict_return_literal(entry)
        if dict_literal is not None:
            inferred = _infer_schema_from_literal(dict_literal, ctx)
            if inferred:
                json_schema = inferred
                schema_source = _rel_cite(path, entry.lineno, files_root) + " (inferred from return dict)"
        else:
            list_literal = _extract_list_return_literal(entry)
            if list_literal is not None:
                inferred_array = _infer_array_schema_from_literal(list_literal, ctx)
                if inferred_array:
                    json_schema = inferred_array
                    schema_source = _rel_cite(path, entry.lineno, files_root) + " (inferred from return list)"

        if json_schema is None and entry.returns is not None:
            param_annotations = {
                ast.unparse(arg.annotation)
                for arg in (entry.args.args + entry.args.kwonlyargs)
                if arg.annotation is not None
            }
            ret_annotation_str = ast.unparse(entry.returns)
            if ret_annotation_str not in param_annotations:
                ret_type = _annotation_to_schema(entry.returns, ctx)
                if isinstance(ret_type, dict) and ret_type.get("type") in ("array", "object") and ret_type.get("properties"):
                    json_schema = ret_type
                    schema_source = _rel_cite(path, entry.lineno, files_root) + " (inferred from return annotation)"

    fallback_literal, fallback_source, fallback_ambiguous = _harvest_fallback(
        cls, path, files_root, letter=letter, asts=asts, conventions=conventions
    )

    scope_nodes: list[ast.AST] = []
    if entry is not None:
        scope_nodes = _entry_schema_scope(cls, entry) if cls is not None else [entry]
    schema_enum_values = (
        _harvest_schema_enum_values(scope_nodes, cls, tree, path, asts, files_root) if scope_nodes else {}
    )
    cardinality = None
    if json_schema is not None:
        cardinality = "array" if json_schema.get("type") == "array" else "object"
    has_streamed_output = bool(json_schema) and _scope_has_streaming_call(scope_nodes)

    output = OutputContract(
        json_schema=json_schema,
        schema_source=schema_source,
        fallback_literal=fallback_literal,
        fallback_source=fallback_source,
        validated_in_target=letter is not None,
        section_letter=letter,
        schema_enum_values=schema_enum_values,
        cardinality=cardinality,
        has_streamed_output=has_streamed_output,
    )
    if json_schema is None and fallback_literal is None:
        notes.append(f"{component.id}: no output schema statically harvestable")
    if fallback_ambiguous:
        notes.append(
            f"{component.id}: multiple candidate fallback methods found — picked "
            f"{fallback_source}, verify manually"
        )

    return invocation, output, _harvest_constants(tree, cls), notes, input_kind


def harvest_contracts(
    system_map: SystemMap,
    agent_flow_map: AgentFlowMap,
    files: list[Path],
    files_root: Path | None = None,
    conventions: ContractConventions | None = None,
) -> dict[str, EvaluationContract]:
    """One EvaluationContract per agent, harvested from its head component's source."""
    asts = _parse_files(files)
    components_by_id = {c.id: c for c in system_map.components}
    contracts: dict[str, EvaluationContract] = {}
    field_consumers_by_agent, field_consumer_notes = harvest_field_downstream_consumers(
        agent_flow_map, system_map, asts
    )

    _signals_path = Path(__file__).parent.parent.parent / "discovery" / "contract_signals.yaml"
    _kw_signals: list[str] = []
    if _signals_path.exists():
        try:
            _sig_data = yaml.safe_load(_signals_path.read_text(encoding="utf-8")) or {}
            _kw_signals = [str(k) for k in (_sig_data.get("keywords") or [])]
        except (yaml.YAMLError, OSError, AttributeError) as exc:
            logger.warning(
                "could not load %s — has_retrieval_signal keyword tier disabled: %s",
                _signals_path, exc,
            )

    for agent in agent_flow_map.agents:
        head_id = agent.id if agent.id in components_by_id else (agent.component_ids[0] if agent.component_ids else "")
        component = components_by_id.get(head_id)
        contract = EvaluationContract(agent_id=agent.id, component_id=head_id)
        has_tools = any(
            components_by_id[cid].role == "tool"
            for cid in agent.component_ids
            if cid in components_by_id
        )
        edges = []
        for cid in agent.component_ids:
            comp = components_by_id.get(cid)
            if comp:
                edges.extend({"src": comp.id, "dst": d} for d in comp.downstream)
        contract.connect_edges = edges
        contract.field_downstream_consumers = field_consumers_by_agent.get(agent.id, {})
        contract.needs_human.extend(n for n in field_consumer_notes if n.startswith(f"{agent.id}:"))

        if component is None:
            contract.needs_human.append(f"{agent.id}: head component not found in system map")
            contract.observability = ObservabilityContract(has_tools=has_tools)
            contracts[agent.id] = contract
            continue

        entry_path, entry_cls, entry_ast = _resolve_entry(component, asts)
        entry_is_async = isinstance(entry_ast, ast.AsyncFunctionDef) if entry_ast is not None else False

        invocation, output, constants, notes, input_kind = harvest_component_contract(
            component, asts, files_root, conventions
        )
        contract.invocation = invocation
        contract.output = output
        contract.constants = constants
        contract.needs_human.extend(notes)
        contract.observability = ObservabilityContract(
            has_tools=has_tools, input_kind=input_kind, entry_is_async=entry_is_async
        )
        contract.query_planning_subcall = _detect_query_planning_subcall(component, asts, conventions)

        # OR-combine keyword-tier and role-tier to set has_retrieval_signal
        constructor_deps = invocation.constructor_deps if invocation else []
        tier1 = bool(_kw_signals) and any(kw in dep for dep in constructor_deps for kw in _kw_signals)
        if tier1 and component:
            path, cls, entry = _resolve_entry(component, asts)
            if path is not None and cls is not None and entry is not None:
                dep_bindings = _harvest_constructor_dep_bindings(cls)
                call_sites = _harvest_dep_call_sites(entry, dep_bindings, asts[path], path, asts, files_root)
                if not call_sites:
                    tier1 = False

        tier2 = any(
            components_by_id[u].role == "retrieval_agent"
            for u in (component.upstream if component else [])
            if u in components_by_id
        )
        if tier1 or tier2:
            contract.has_retrieval_signal = True
            provenance = "+".join(t for t, ok in (("keyword-tier", tier1), ("role-tier", tier2)) if ok)
            contract.needs_human.append(f"has_retrieval_signal: True via {provenance}")
        contract.retrieves_internally = tier1

        contracts[agent.id] = contract

    # Second pass: fill upstream_context_specs from harvested output contracts
    comp_to_agent: dict[str, str] = {}
    for agent in agent_flow_map.agents:
        for cid in agent.component_ids:
            comp_to_agent[cid] = agent.id

    for agent in agent_flow_map.agents:
        head_id = agent.id if agent.id in components_by_id else (agent.component_ids[0] if agent.component_ids else "")
        component = components_by_id.get(head_id)
        if component is None:
            continue
        specs: list[dict] = []
        for up_comp_id in component.upstream:
            up_agent_id = comp_to_agent.get(up_comp_id)
            if not up_agent_id or up_agent_id not in contracts:
                continue
            up_contract = contracts[up_agent_id]
            if up_contract.output and up_contract.output.json_schema:
                schema_snippet = json.dumps(
                    up_contract.output.json_schema, ensure_ascii=False
                )

                # Look up actual kwarg name from downstream agent's case_binding instead of guessing
                spec_name = f"{up_agent_id}_output"  # fallback
                down_contract = contracts.get(agent.id)
                if down_contract and down_contract.invocation and down_contract.invocation.case_binding:
                    target_value = f"case:$.input.{up_agent_id}_output"
                    for kwarg_name, binding_value in down_contract.invocation.case_binding.items():
                        if binding_value == target_value:
                            spec_name = kwarg_name
                            break

                specs.append({
                    "name": spec_name,
                    "description": f"the {up_agent_id} agent's output — JSON schema: {schema_snippet}",
                })
        if specs:
            contracts[agent.id].upstream_context_specs = specs

    # Third pass: archetype MUST run after upstream_context_specs (pass 2) — it reads that field.
    for contract in contracts.values():
        contract.archetype = _archetype_for(contract)

    return contracts


def _detect_query_planning_subcall(
    component: Component, asts: dict[Path, ast.Module],
    conventions: ContractConventions | None = None,
) -> bool:
    """True if the entry method calls a module-level plan_queries-style helper (D/J/F's query-planning archetype)."""
    _, _cls, entry = _resolve_entry(component, asts)
    if entry is None:
        return False
    symbol = (conventions or ContractConventions()).plan_queries_symbol
    for node in ast.walk(entry):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == symbol:
            return True
    return False


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _find_module_dict_literal(
    asts: dict[Path, ast.Module], name: str, *, own_file: Path | None = None
) -> dict[str, list[str]] | None:
    """A module-level `NAME = {...}` dict[str, list[str]] literal (e.g. _SECTION_PREVIEW_KEYS); None if not found or shape doesn't match."""
    for path in _name_search_order(asts, name, own_file):
        found = _find_name_assignment(asts[path], name)
        if found is None:
            continue
        value, _ = found
        if not isinstance(value, ast.Dict):
            return None
        result: dict[str, list[str]] = {}
        for k, v in zip(value.keys, value.values):
            if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                return None
            if isinstance(v, ast.List) and all(
                isinstance(e, ast.Constant) and isinstance(e.value, str) for e in v.elts
            ):
                result[k.value] = [e.value for e in v.elts]
            else:
                return None
        return result
    return None


def _resolve_callable(
    name: str, own_tree: ast.Module, own_file: Path, asts: dict[Path, ast.Module]
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, Path] | None:
    fn = _find_function(own_tree, name)
    if fn is not None:
        return fn, own_file
    src_file = _import_source_file(own_tree, name, asts)
    if src_file is not None:
        fn = _find_function(asts[src_file], name)
        if fn is not None:
            return fn, src_file
    return None


def _eval_letter_arg(node: ast.expr, env: dict[str, tuple]) -> str | None:
    """Resolve an expression to a concrete string constant — a literal or a Name bound in env."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        binding = env.get(node.id)
        if binding is not None and binding[0] == "const":
            return binding[1]
    return None


def _eval_section_ref(node: ast.expr, env: dict[str, tuple]) -> tuple | None:
    """Resolve an expression to ("section", <letter>) or ("all_sections",), via a bound Name or an inline `.get()`/subscript expression."""
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or) and node.values:
        return _eval_section_ref(node.values[0], env)
    if isinstance(node, ast.Name):
        binding = env.get(node.id)
        if binding is not None and binding[0] in ("section", "all_sections"):
            return binding
        return None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get":
        base = node.func.value
        if isinstance(base, ast.Name):
            binding = env.get(base.id)
            if binding is not None and binding[0] == "all_sections" and node.args:
                letter = _eval_letter_arg(node.args[0], env)
                if letter is not None:
                    return ("section", letter)
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        binding = env.get(node.value.id)
        if binding is not None and binding[0] == "all_sections":
            letter = _eval_letter_arg(node.slice, env)
            if letter is not None:
                return ("section", letter)
    return None


def _eval_strlist_ref(
    node: ast.expr, env: dict[str, tuple], asts: dict[Path, ast.Module], own_file: Path
) -> list[str] | None:
    """Resolve an expression to a concrete list[str] — a bound Name, or an inline module-dict `.get()` lookup."""
    if isinstance(node, ast.Name):
        binding = env.get(node.id)
        if binding is not None and binding[0] == "strlist":
            return binding[1]
        return None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get":
        base = node.func.value
        if isinstance(base, ast.Name):
            dict_lit = _find_module_dict_literal(asts, base.id, own_file=own_file)
            if dict_lit is not None and node.args:
                letter = _eval_letter_arg(node.args[0], env)
                if letter is not None:
                    return dict_lit.get(letter, [])
    return None


def _trace_stmts(
    stmts: list[ast.stmt],
    env: dict[str, tuple],
    own_tree: ast.Module,
    own_file: Path,
    asts: dict[Path, ast.Module],
    fields_by_letter: dict[str, set[str]],
    depth: int,
) -> None:
    if depth > _MAX_TRACE_DEPTH:
        return
    for stmt in stmts:
        if isinstance(stmt, (ast.For, ast.AsyncFor)) and isinstance(stmt.target, ast.Name):
            letters: list[str] | None = None
            it = stmt.iter
            if isinstance(it, ast.Constant) and isinstance(it.value, str):
                letters = list(it.value)
            else:
                sl = _eval_strlist_ref(it, env, asts, own_file)
                if sl is not None:
                    letters = sl
            if letters is not None:
                for letter in letters:
                    inner_env = dict(env)
                    inner_env[stmt.target.id] = ("const", letter)
                    _trace_stmts(stmt.body, inner_env, own_tree, own_file, asts, fields_by_letter, depth)
                continue
            # unresolvable iterable: best-effort walk of the body with the outer env
            _trace_stmts(stmt.body, dict(env), own_tree, own_file, asts, fields_by_letter, depth)
            continue

        if isinstance(stmt, ast.If):
            _trace_stmts(stmt.body, env, own_tree, own_file, asts, fields_by_letter, depth)
            _trace_stmts(stmt.orelse, env, own_tree, own_file, asts, fields_by_letter, depth)
            continue

        if isinstance(stmt, (ast.While, ast.Try, ast.With, ast.AsyncWith)):
            for attr_name in ("body", "orelse", "finalbody"):
                _trace_stmts(getattr(stmt, attr_name, None) or [], env, own_tree, own_file, asts, fields_by_letter, depth)
            for handler in getattr(stmt, "handlers", None) or []:
                _trace_stmts(handler.body, env, own_tree, own_file, asts, fields_by_letter, depth)
            continue

        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue  # nested defs not expected in these agents; skip rather than misattribute

        # Flat statement — no nested compound children, so a blanket ast.walk here is safe.
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            target_name = stmt.targets[0].id
            value = stmt.value
            if isinstance(value, ast.BoolOp) and isinstance(value.op, ast.Or) and value.values:
                value = value.values[0]
            sec = _eval_section_ref(stmt.value, env)
            if sec is not None:
                env = dict(env)
                env[target_name] = sec
            else:
                sl = _eval_strlist_ref(value, env, asts, own_file)
                if sl is not None:
                    env = dict(env)
                    env[target_name] = ("strlist", sl)

        for node in ast.walk(stmt):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get" and node.args:
                sec = _eval_section_ref(node.func.value, env)
                if sec is not None and sec[0] == "section":
                    key = _eval_letter_arg(node.args[0], env)
                    if key is not None:
                        fields_by_letter.setdefault(sec[1], set()).add(key)
            elif isinstance(node, ast.Subscript):
                sec = _eval_section_ref(node.value, env)
                if sec is not None and sec[0] == "section":
                    key = _eval_letter_arg(node.slice, env)
                    if key is not None:
                        fields_by_letter.setdefault(sec[1], set()).add(key)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                resolved = _resolve_callable(node.func.id, own_tree, own_file, asts)
                if resolved is None:
                    continue
                callee_fn, callee_file = resolved
                params = list(callee_fn.args.posonlyargs) + list(callee_fn.args.args)
                callee_env: dict[str, tuple] = {}
                for param, arg_node in zip(params, node.args):
                    sec = _eval_section_ref(arg_node, env)
                    if sec is not None:
                        callee_env[param.arg] = sec
                        continue
                    letter = _eval_letter_arg(arg_node, env)
                    if letter is not None:
                        callee_env[param.arg] = ("const", letter)
                        continue
                    sl = _eval_strlist_ref(arg_node, env, asts, own_file)
                    if sl is not None:
                        callee_env[param.arg] = ("strlist", sl)
                if callee_env:
                    _trace_stmts(
                        list(callee_fn.body), callee_env, asts[callee_file], callee_file,
                        asts, fields_by_letter, depth + 1,
                    )


def _seed_fan_in_param(entry: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    """The entry method's sole required, non-config kwarg — the candidate fan-in input. None if there isn't exactly one."""
    positional, defaults = _positional_params_with_defaults(entry.args)
    candidates = [
        arg.arg for arg, default in zip(positional, defaults)
        if default is None and arg.arg not in _CONFIG_PARAMS
    ]
    return candidates[0] if len(candidates) == 1 else None


def harvest_field_downstream_consumers(
    agent_flow_map: AgentFlowMap,
    system_map: SystemMap,
    asts: dict[Path, ast.Module],
) -> tuple[dict[str, dict[str, list[str]]], list[str]]:
    """For each single-fan-in-param agent (e.g. K/L), which upstream fields does its entry method actually read? Static recursive (depth<=6) resolution of `.get()`/subscript chains. Returns (by_agent, notes); by_agent[agent_id][letter] = sorted field names."""
    components_by_id = {c.id: c for c in system_map.components}
    by_agent: dict[str, dict[str, list[str]]] = {}
    notes: list[str] = []

    for agent in agent_flow_map.agents:
        head_id = agent.id if agent.id in components_by_id else (agent.component_ids[0] if agent.component_ids else "")
        component = components_by_id.get(head_id)
        if component is None:
            continue
        path, _cls, entry = _resolve_entry(component, asts)
        if path is None or entry is None:
            continue
        tree = asts[path]
        fan_in_param = _seed_fan_in_param(entry)
        if fan_in_param is None:
            continue  # not a single-fan-in-param agent; out of scope for this harvest

        fields_by_letter: dict[str, set[str]] = {}
        _trace_stmts(
            list(entry.body), {fan_in_param: ("all_sections",)}, tree, path, asts,
            fields_by_letter, 0,
        )
        if fields_by_letter:
            by_agent[agent.id] = {letter: sorted(fields) for letter, fields in fields_by_letter.items()}
        else:
            notes.append(f"{agent.id}: no statically-resolvable upstream field reads found")

    return by_agent, notes
