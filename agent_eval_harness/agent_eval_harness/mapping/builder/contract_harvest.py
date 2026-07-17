"""Static (AST-only) harvest of per-agent EvaluationContracts; never imports target code."""
from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path

import yaml

from agent_eval_harness.config import ContractConventions
from agent_eval_harness.mapping.agent_flow import AgentFlowMap
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

_ENTRY_METHOD_NAMES = ("run", "run_async", "__call__")
_QUERYISH_PARAMS = frozenset({"query", "question", "prompt", "text", "message", "user_input"})
_CONFIG_PARAMS = frozenset({"provider_id", "model_id"})
_SCHEMA_NAME_RE = re.compile(r"SCHEMA", re.IGNORECASE)
_DYNAMIC = "<dynamic>"
_MAX_TRACE_DEPTH = 6  # recursion cap for field-consumer call tracing (_trace_stmts)
_SCHEMA_SNIPPET_MAX_CHARS = 200  # upstream_context_specs preview length


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


def _name_search_order(
    asts: dict[Path, ast.Module], name: str, own_file: Path | None
) -> list[Path]:
    """File search order for resolving a module-level NAME: own_file's import source, then
    own_file itself, then every other parsed file — shared by the str-constant and dict-literal
    resolvers below since they both need "prefer where NAME was imported from."""
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


def _find_module_str_constant(
    asts: dict[Path, ast.Module],
    name: str,
    files_root: Path | None,
    *,
    own_file: Path | None = None,
) -> tuple[str, str] | None:
    """Find a module-level `NAME = <str literal>`, preferring own_file's import source then own_file, else scan others."""
    for path in _name_search_order(asts, name, own_file):
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
                # PEP 655: total=True required unless NotRequired[]; total=False optional unless Required[].
                is_required = (
                    not _is_notrequired(item.annotation) if total else _is_required(item.annotation)
                )
                if is_required:
                    required.append(key)
        schema = {"type": "object", "properties": properties, "required": required}
        return schema, _rel_cite(path, cls.lineno, files_root) + f" {name}"
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


def _extract_dict_return_literal(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict | None:
    """First `return {...}` dict literal in a function body; non-constant values become '<dynamic>'."""
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
                return literal
    return None


def _find_pipeline_fallback_function(
    asts: dict[Path, ast.Module], letter: str, files_root: Path | None,
    conventions: ContractConventions | None = None,
) -> tuple[dict, str] | None:
    """Fallthrough for agents without their own try/except (e.g. K/L): they have a module-level `_section_<letter>_pipeline_fallback()` instead."""
    pattern = (conventions or ContractConventions()).pipeline_fallback_name_pattern
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
    cls: ast.ClassDef,
    path: Path,
    files_root: Path | None,
    *,
    letter: str | None = None,
    asts: dict[Path, ast.Module] | None = None,
    conventions: ContractConventions | None = None,
) -> tuple[dict | None, str | None, bool]:
    """Fallback method returning a dict literal; disambiguates candidates by entry-point call, then name match, then order. Returned bool flags genuine ambiguity for human review."""
    candidates = [
        node for node in cls.body
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
    path = _resolve_component_file(component, asts)
    if path is None:
        return None, None, {}, [f"{component.id}: source file not found for static harvest"], "unknown"
    tree = asts[path]
    _, _, class_name = component.entry_point.partition(":")
    class_name = class_name.split(".")[0]
    cls = _find_class(tree, class_name) if class_name else None
    if cls is None:
        return None, None, {}, [f"{component.id}: class {class_name!r} not found in {path.name}"], "unknown"

    # A validator letter also identifies a per-agent-route invocation path, not just the output TypedDict.
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
            # Rerun via the pipeline's per-section endpoint — live constructor deps run in that process, not AEH.
            invocation_mode = "per_agent_route"
            route = (conventions or ContractConventions()).rerun_section_route
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
            # Live constructor deps with no known route — statics can't resolve invocation here.
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

    fallback_literal, fallback_source, fallback_ambiguous = _harvest_fallback(
        cls, path, files_root, letter=letter, asts=asts, conventions=conventions
    )
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
    conventions: ContractConventions | None = None,
) -> dict[str, EvaluationContract]:
    """One EvaluationContract per agent, harvested from its head component's source."""
    asts = _parse_files(files)
    components_by_id = {c.id: c for c in system_map.components}
    contracts: dict[str, EvaluationContract] = {}
    field_consumers_by_agent, field_consumer_notes = harvest_field_downstream_consumers(
        agent_flow_map, system_map, asts
    )

    # Load keyword signals from discovery/contract_signals.yaml
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

        invocation, output, constants, notes, input_kind = harvest_component_contract(
            component, asts, files_root, conventions
        )
        contract.invocation = invocation
        contract.output = output
        contract.constants = constants
        contract.needs_human.extend(notes)
        contract.observability = ObservabilityContract(has_tools=has_tools, input_kind=input_kind)
        contract.query_planning_subcall = _detect_query_planning_subcall(component, asts, conventions)

        # OR-combine keyword-tier and role-tier to set has_retrieval_signal
        constructor_deps = invocation.constructor_deps if invocation else []
        tier1 = bool(_kw_signals) and any(kw in dep for dep in constructor_deps for kw in _kw_signals)
        tier2 = any(
            components_by_id[u].role == "retrieval_agent"
            for u in (component.upstream if component else [])
            if u in components_by_id
        )
        if tier1 or tier2:
            contract.has_retrieval_signal = True
            provenance = "+".join(t for t, ok in (("keyword-tier", tier1), ("role-tier", tier2)) if ok)
            contract.needs_human.append(f"has_retrieval_signal: True via {provenance}")

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
                )[:_SCHEMA_SNIPPET_MAX_CHARS]
                specs.append({
                    "name": f"{up_agent_id}_output",
                    "description": f"the {up_agent_id} agent's output — JSON schema: {schema_snippet}",
                })
        if specs:
            contracts[agent.id].upstream_context_specs = specs

    return contracts


def _detect_query_planning_subcall(
    component: Component, asts: dict[Path, ast.Module],
    conventions: ContractConventions | None = None,
) -> bool:
    """True if the entry method calls a module-level plan_queries-style helper (D/J/F's query-planning archetype)."""
    path = _resolve_component_file(component, asts)
    if path is None:
        return False
    tree = asts[path]
    _, _, class_name = component.entry_point.partition(":")
    class_name = class_name.split(".")[0]
    cls = _find_class(tree, class_name) if class_name else None
    if cls is None:
        return False
    entry = _find_entry_method(cls)
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
    args = entry.args
    positional = list(args.posonlyargs) + list(args.args)
    if positional and positional[0].arg in ("self", "cls"):
        positional = positional[1:]
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
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
        path = _resolve_component_file(component, asts)
        if path is None:
            continue
        tree = asts[path]
        _, _, class_name = component.entry_point.partition(":")
        class_name = class_name.split(".")[0]
        cls = _find_class(tree, class_name) if class_name else None
        if cls is None:
            continue
        entry = _find_entry_method(cls)
        if entry is None:
            continue
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
