"""CS-287 — static (AST-only) harvest of per-agent EvaluationContracts; never imports target code."""
from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path

from agent_eval_harness.mapping.agent_flow import AgentFlowMap
from agent_eval_harness.mapping.system_map import Component, SystemMap
from agent_eval_harness.planning.contract import (
    EvaluationContract,
    InvocationContract,
    KwargSpec,
    ObservabilityContract,
    OutputContract,
)

logger = logging.getLogger("agent_eval_harness.contract_harvest")

_ENTRY_METHOD_NAMES = ("run", "run_async", "__call__")
_QUERYISH_PARAMS = frozenset({"query", "question", "prompt", "text", "message", "user_input"})
_CONFIG_PARAMS = frozenset({"provider_id", "model_id"})
_SCHEMA_NAME_RE = re.compile(r"SCHEMA", re.IGNORECASE)
_DYNAMIC = "<dynamic>"
# CodeSpectra's own analysis pipeline (v1 harvest target — CS-287 scope) exposes exactly
# one generic per-section rerun route; `section` accepts any of the 12 section letters.
_RERUN_SECTION_ROUTE = "/api/analysis/rerun_section"


def _parse_files(files: list[Path]) -> dict[Path, ast.Module]:
    asts: dict[Path, ast.Module] = {}
    for file in files:
        if file.suffix != ".py":
            continue
        try:
            asts[file] = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        except (SyntaxError, OSError, UnicodeDecodeError):
            continue
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


def _unparse(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None


def _harvest_kwargs(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[list[KwargSpec], list[str]]:
    """Returns (kwargs, notes) — notes flag shapes statics can't fully resolve (*args/**kwargs)."""
    notes: list[str] = []
    specs: list[KwargSpec] = []
    args = fn.args
    positional = list(args.posonlyargs) + list(args.args)
    if positional and positional[0].arg in ("self", "cls"):
        positional = positional[1:]
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
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


def _find_entry_method(cls: ast.ClassDef) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    by_name = {
        node.name: node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for name in _ENTRY_METHOD_NAMES:
        if name in by_name:
            return by_name[name]
    return None


def _harvest_constructor_deps(cls: ast.ClassDef) -> list[str]:
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__init__":
            deps = []
            for arg in node.args.args[1:]:  # skip self
                deps.append(_unparse(arg.annotation) or arg.arg)
            return deps
    return []


def _derive_case_binding(kwargs: list[KwargSpec]) -> dict[str, str]:
    binding: dict[str, str] = {}
    for spec in kwargs:
        if spec.name in _CONFIG_PARAMS:
            binding[spec.name] = f"config:{spec.name}"
        elif spec.required:
            binding[spec.name] = f"case:$.input.{spec.name}"
    return binding


def _derive_input_kind(kwargs: list[KwargSpec]) -> str:
    names = {s.name for s in kwargs}
    if names & _QUERYISH_PARAMS:
        return "query"
    if names - _CONFIG_PARAMS:
        return "structured"
    return "unknown"


def _rel_cite(path: Path, lineno: int, files_root: Path | None) -> str:
    p = path.as_posix()
    if files_root is not None:
        try:
            p = path.relative_to(files_root).as_posix()
        except ValueError:
            pass
    return f"{p}:{lineno}"


def _find_validator_letter(cls: ast.ClassDef) -> str | None:
    """First validate_*(<str-const>, ...) call inside the class body, e.g. validate_section("A", data)."""
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


def _referenced_schema_names(cls: ast.ClassDef) -> list[str]:
    names = []
    for node in ast.walk(cls):
        if isinstance(node, ast.Name) and _SCHEMA_NAME_RE.search(node.id) and node.id.isupper():
            names.append(node.id)
        elif isinstance(node, ast.Attribute) and _SCHEMA_NAME_RE.search(node.attr) and node.attr.isupper():
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


def _find_module_str_constant(
    asts: dict[Path, ast.Module],
    name: str,
    files_root: Path | None,
    *,
    own_file: Path | None = None,
) -> tuple[str, str] | None:
    """Find a module-level `NAME = <str literal>` assignment.

    Resolution order: the file an explicit `from x.y import NAME` in `own_file`
    points to, then `own_file` itself, then (last resort) every other parsed
    file. Stops at the FIRST file where `name` is assigned at all — even if
    that assignment is dynamic (non-literal) — rather than skipping past it to
    grab a same-named constant from an unrelated file, which would silently
    misattribute one component's schema to another.
    """
    search_order: list[Path] = []
    if own_file is not None and own_file in asts:
        imported_from = _import_source_file(asts[own_file], name, asts)
        if imported_from is not None:
            search_order.append(imported_from)
        search_order.append(own_file)
    for path in asts:
        if path not in search_order:
            search_order.append(path)

    for path in search_order:
        found = _find_name_assignment(asts[path], name)
        if found is None:
            continue
        value, lineno = found
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value, _rel_cite(path, lineno, files_root)
        # found (dynamically assembled) in the resolved source file — stop here, never guess
        return None
    return None


_TYPE_MAP = {
    "str": {"type": "string"},
    "int": {"type": "integer"},
    "float": {"type": "number"},
    "bool": {"type": "boolean"},
    "dict": {"type": "object"},
    "list": {"type": "array"},
}


def _annotation_to_schema(node: ast.expr) -> dict:
    if isinstance(node, ast.Name):
        return dict(_TYPE_MAP.get(node.id, {}))
    if isinstance(node, ast.Subscript):
        base = node.value
        base_name = base.id if isinstance(base, ast.Name) else base.attr if isinstance(base, ast.Attribute) else ""
        if base_name in ("list", "List"):
            item = _annotation_to_schema(node.slice) if not isinstance(node.slice, ast.Tuple) else {}
            schema: dict = {"type": "array"}
            if item:
                schema["items"] = item
            return schema
        if base_name in ("dict", "Dict"):
            return {"type": "object"}
        if base_name == "Literal":
            elts = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            values = [e.value for e in elts if isinstance(e, ast.Constant)]
            if values and all(isinstance(v, str) for v in values):
                return {"type": "string", "enum": values}
        if base_name in ("NotRequired", "Required"):
            return _annotation_to_schema(node.slice)
    return {}


def _subscript_base_name(node: ast.expr) -> str:
    if isinstance(node, ast.Subscript):
        base = node.value
        return base.id if isinstance(base, ast.Name) else base.attr if isinstance(base, ast.Attribute) else ""
    return ""


def _is_notrequired(node: ast.expr) -> bool:
    return _subscript_base_name(node) == "NotRequired"


def _is_required(node: ast.expr) -> bool:
    return _subscript_base_name(node) == "Required"


def _find_typeddict(
    asts: dict[Path, ast.Module], name: str, files_root: Path | None
) -> tuple[dict, str] | None:
    for path, tree in asts.items():
        cls = _find_class(tree, name)
        if cls is None:
            continue
        base_names = {b.id if isinstance(b, ast.Name) else getattr(b, "attr", "") for b in cls.bases}
        if "TypedDict" not in base_names:
            continue
        total = True
        for kw in cls.keywords:
            if kw.arg == "total" and isinstance(kw.value, ast.Constant):
                total = bool(kw.value.value)
        properties: dict = {}
        required: list[str] = []
        for item in cls.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                key = item.target.id
                properties[key] = _annotation_to_schema(item.annotation)
                # PEP 655: total=True fields are required unless NotRequired[]; total=False
                # fields are optional unless explicitly wrapped in Required[].
                is_required = (
                    not _is_notrequired(item.annotation) if total else _is_required(item.annotation)
                )
                if is_required:
                    required.append(key)
        schema = {"type": "object", "properties": properties, "required": required}
        return schema, _rel_cite(path, cls.lineno, files_root) + f" {name}"
    return None


def _self_called_names(method: ast.FunctionDef | ast.AsyncFunctionDef | None) -> set[str]:
    """Method names invoked as `self.<name>(...)` inside `method` — used to prefer the
    fallback method the entry point actually calls over an unrelated same-named helper."""
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


def _harvest_fallback(
    cls: ast.ClassDef, path: Path, files_root: Path | None
) -> tuple[dict | None, str | None, bool]:
    """A *fallback* method returning a dict literal; non-constant values become '<dynamic>'.

    A class can have more than one method with "fallback" in its name (e.g. an output
    fallback alongside an unrelated `_fallback_provider` config helper). Disambiguate by
    preferring the method actually called from the entry point, then an exact
    `fallback`/`_fallback` name, then declaration order. Returns whether the pick was
    genuinely ambiguous (>1 candidate) so the caller can flag it for human review.
    """
    candidates = [
        node for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and "fallback" in node.name.lower()
    ]
    if not candidates:
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
        for ret in ast.walk(node):
            if isinstance(ret, ast.Return) and isinstance(ret.value, ast.Dict):
                literal: dict = {}
                for k, v in zip(ret.value.keys, ret.value.values):
                    if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                        continue
                    try:
                        literal[k.value] = ast.literal_eval(v)
                    except (ValueError, SyntaxError):
                        literal[k.value] = _DYNAMIC
                if literal:
                    return literal, _rel_cite(path, node.lineno, files_root), ambiguous
    return None, None, False


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
) -> tuple[InvocationContract | None, OutputContract | None, dict[str, int], list[str], str]:
    """Returns (invocation, output, constants, needs_human_notes, input_kind) for one component."""
    notes: list[str] = []
    path = _resolve_component_file(component, asts)
    if path is None:
        return None, None, {}, [f"{component.id}: source file not found for static harvest"], "unknown"
    tree = asts[path]
    _, _, class_name = component.entry_point.partition(":")
    class_name = class_name.split(".")[0]
    cls = _find_class(tree, class_name) if class_name else None
    if cls is None:
        return None, None, {}, [f"{component.id}: class {class_name!r} not found in {path.name}"], "unknown"

    # Computed early — a validator letter also identifies a per-agent-route invocation
    # path (see below), not just the output TypedDict.
    letter = _find_validator_letter(cls)

    entry = _find_entry_method(cls)
    invocation: InvocationContract | None = None
    input_kind = "unknown"
    if entry is not None:
        kwargs, kw_notes = _harvest_kwargs(entry)
        notes.extend(f"{component.id}: {n}" for n in kw_notes)
        constructor_deps = _harvest_constructor_deps(cls)
        input_kind = _derive_input_kind(kwargs)
        if not constructor_deps:
            invocation_mode = "in_harness"
            route = None
            case_binding = _derive_case_binding(kwargs)
        elif letter:
            # CodeSpectra's own analysis pipeline exposes exactly this: rerun one
            # section by letter through an existing endpoint. Live services the
            # constructor needs (ProviderConfigService, RetrievalService, ...) run
            # inside that endpoint's process — AEH never constructs them itself.
            invocation_mode = "per_agent_route"
            route = _RERUN_SECTION_ROUTE
            case_binding = {
                "report_id": "case:$.input.report_id",
                "section": f"const:{letter}",
                "provider_id": "config:provider_id",
                "model_id": "config:model_id",
            }
            notes.append(
                f"{component.id}: invocation via {route} requires an existing analysis "
                "report for the snapshot — report_id is not the same identifier as snapshot_id"
            )
        else:
            # Live constructor deps and no known route to reach this component through —
            # statics genuinely can't resolve an invocation path here.
            invocation_mode = "unsupported"
            route = None
            case_binding = _derive_case_binding(kwargs)
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
        notes.append(f"{component.id}: no run/run_async/__call__ method found")

    # Output contract: validator letter -> Section{letter} TypedDict; else SCHEMA_STR-as-JSON.
    json_schema: dict | None = None
    schema_source: str | None = None
    if letter:
        found = _find_typeddict(asts, f"Section{letter}", files_root)
        if found:
            json_schema, schema_source = found
    if json_schema is None:
        for name in _referenced_schema_names(cls):
            found_str = _find_module_str_constant(asts, name, files_root, own_file=path)
            if found_str is None:
                notes.append(f"{component.id}: schema constant {name} is dynamically assembled — unresolvable")
                continue
            text, cite = found_str
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    json_schema, schema_source = parsed, cite + f" {name}"
                    break
            except json.JSONDecodeError:
                schema_source = schema_source or (cite + f" {name} (non-JSON template)")

    fallback_literal, fallback_source, fallback_ambiguous = _harvest_fallback(cls, path, files_root)
    output = OutputContract(
        json_schema=json_schema,
        schema_source=schema_source,
        fallback_literal=fallback_literal,
        fallback_source=fallback_source,
        validated_in_target=letter is not None,
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
) -> dict[str, EvaluationContract]:
    """One EvaluationContract per agent, harvested from its head component's source."""
    asts = _parse_files(files)
    components_by_id = {c.id: c for c in system_map.components}
    contracts: dict[str, EvaluationContract] = {}

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

        if component is None:
            contract.needs_human.append(f"{agent.id}: head component not found in system map")
            contract.observability = ObservabilityContract(has_tools=has_tools)
            contracts[agent.id] = contract
            continue

        invocation, output, constants, notes, input_kind = harvest_component_contract(
            component, asts, files_root
        )
        contract.invocation = invocation
        contract.output = output
        contract.constants = constants
        contract.needs_human.extend(notes)
        contract.observability = ObservabilityContract(has_tools=has_tools, input_kind=input_kind)
        contracts[agent.id] = contract

    return contracts
