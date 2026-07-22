"""Framework scanner protocol and Haystack implementation."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Protocol, runtime_checkable

from agent_eval_harness.discovery.wiring import (
    _build_enclosing_class_map,
    _build_self_attr_classes,
    _build_stategraph_var_names,
    _call_func_name,
    _const_str,
    _decorator_effective_name,
    _flatten_bitor,
    _imports_langgraph,
    _receiver_name,
    _resolve_component_class_and_file,
    _resolve_node_target,
)
from agent_eval_harness.mapping.builder.types import (
    CandidateComponent,
    ManualSpanHint,
    parse_python_source,
)

_SNIPPET_LINE_COUNT = 30  # first N lines of a class/function body kept as its source_snippet


@runtime_checkable
class FrameworkScanner(Protocol):
    """Interface for framework-specific component scanners."""

    framework: str

    def scan(self, source_files: list[Path]) -> list[CandidateComponent]: ...


class HaystackScanner:
    """Scans a Haystack-based target for component candidates."""

    framework = "haystack"

    def scan(self, source_files: list[Path]) -> list[CandidateComponent]:
        """Scan source files and return deduplicated, sorted component candidates."""
        candidates = []

        # Cache ASTs upfront to avoid quadratic re-parsing in the second pass
        asts: dict[Path, ast.Module] = {}
        file_contents: dict[str, str] = {}
        known_classes: set[str] = set()
        add_component_names: dict[str, str] = {}  # {haystack_name: class_name}
        module_level_async_functions: dict[str, ast.AsyncFunctionDef] = {}

        for file in source_files:
            parsed = parse_python_source(file)
            if parsed is None:
                continue
            source, tree = parsed
            asts[file] = tree
            file_contents[str(file)] = source

        # First pass: build known_classes and add_component_names
        for file, tree in asts.items():
            self_attr_classes = _build_self_attr_classes(tree)

            for node in tree.body:
                if isinstance(node, ast.AsyncFunctionDef):
                    module_level_async_functions[node.name] = node

            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    for decorator in node.decorator_list:
                        if self._is_component_decorator(decorator):
                            known_classes.add(node.name)
                            break

                # (e.g. add_component(alias, _WrapperComponent(..., self._real_agent, ...)))
                # to the real wrapped class, same as the wiring detector does for Stage 1.
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if self._is_add_component_call(child):
                            if (
                                len(child.args) >= 2
                                and isinstance(child.args[0], ast.Constant)
                                and isinstance(child.args[0].value, str)
                                and isinstance(child.args[1], ast.Call)
                            ):
                                haystack_name = child.args[0].value
                                class_name, _resolved_file = _resolve_component_class_and_file(
                                    child.args[1], str(file), self_attr_classes, file_contents
                                )
                                if class_name:
                                    add_component_names[haystack_name] = class_name
                                    known_classes.add(class_name)

                processed_binops = set()
                for child in ast.walk(node):
                    if isinstance(child, (ast.Assign, ast.AnnAssign, ast.Expr, ast.Return, ast.Yield)):
                        val_node = None
                        if isinstance(child, (ast.Assign, ast.Expr, ast.Return, ast.Yield)):
                            val_node = child.value
                        elif isinstance(child, ast.AnnAssign):
                            val_node = child.value

                        if val_node:
                            for sub_child in ast.walk(val_node):
                                if isinstance(sub_child, ast.BinOp) and isinstance(sub_child.op, ast.BitOr):
                                    if sub_child in processed_binops:
                                        continue

                                    operands = _flatten_bitor(sub_child, processed_binops)
                                    for op in operands:
                                        class_name = None
                                        if isinstance(op, ast.Name):
                                            class_name = op.id
                                        elif isinstance(op, ast.Call):
                                            class_name = _call_func_name(op)
                                        elif isinstance(op, ast.Attribute):
                                            class_name = op.attr

                                        if class_name:
                                            known_classes.add(class_name)
                                            add_component_names[class_name] = class_name

        # Second pass: grow known_classes with composed components using cached ASTs
        for file, tree in asts.items():
            # Grow known_classes with composed components
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name in known_classes:
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                            for arg in item.args.args:
                                if arg.annotation:
                                    for name_node in ast.walk(arg.annotation):
                                        if isinstance(name_node, ast.Name):
                                            for check_tree in asts.values():
                                                for check_node in check_tree.body:
                                                    if (
                                                        isinstance(check_node, ast.ClassDef)
                                                        and check_node.name
                                                        == name_node.id
                                                    ):
                                                        known_classes.add(name_node.id)

            # Now extract candidates
            source_lines = file_contents[str(file)].splitlines()

            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name in known_classes:
                    haystack_name = None
                    for name, class_name in add_component_names.items():
                        if class_name == node.name:
                            haystack_name = name
                            break

                    source_snippet = _extract_source_snippet(
                        file, node.lineno, source_lines
                    )
                    manual_span_hints = _extract_manual_spans(node, file)

                    candidate = CandidateComponent(
                        file=file,
                        line=node.lineno,
                        class_name=node.name,
                        haystack_name=haystack_name,
                        source_snippet=source_snippet,
                        manual_span_hints=manual_span_hints,
                    )
                    candidates.append(candidate)

            # Tool discovery: find dict literals with async function values
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    owner_class_name = None
                    if isinstance(node.func, ast.Name):
                        owner_class_name = node.func.id

                    for arg in node.args:
                        if isinstance(arg, ast.Dict):
                            _extract_tool_candidates(
                                file,
                                arg,
                                module_level_async_functions,
                                candidates,
                                owner_class_name,
                                source_lines,
                            )
                    for keyword in node.keywords:
                        if isinstance(keyword.value, ast.Dict):
                            _extract_tool_candidates(
                                file,
                                keyword.value,
                                module_level_async_functions,
                                candidates,
                                owner_class_name,
                                source_lines,
                            )

        # Sub-span split and dedup/sort
        candidates = _split_sub_spans(candidates)
        candidates = _dedup_and_sort(candidates)

        return candidates

    def _is_component_decorator(self, decorator: ast.expr) -> bool:
        """Check if a decorator is @component or @haystack.component."""
        if isinstance(decorator, ast.Name):
            return decorator.id == "component"
        if isinstance(decorator, ast.Attribute):
            return decorator.attr == "component"
        return False

    def _is_add_component_call(self, node: ast.Call) -> bool:
        """Check if a call is <obj>.add_component(...)."""
        if isinstance(node.func, ast.Attribute):
            return node.func.attr == "add_component"
        return False

def _extract_source_snippet(file: Path, line: int, lines: list[str]) -> str:
    """Extract first ~30 lines of a class/function body."""
    if line <= 1 or line > len(lines):
        return ""
    start = line
    end = min(line + _SNIPPET_LINE_COUNT - 1, len(lines))
    return "\n".join(lines[start : end + 1])


def _extract_manual_spans(node: ast.ClassDef, file: Path) -> list[ManualSpanHint]:
    """Extract manual_span() calls from a class's methods."""
    hints = []

    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(item):
                if isinstance(child, ast.With):
                    for with_item in child.items:
                        context_expr = with_item.context_expr
                        if isinstance(context_expr, ast.Call):
                            if isinstance(context_expr.func, ast.Name):
                                if context_expr.func.id == "manual_span":
                                    hint = _parse_manual_span_call(
                                        context_expr, file, context_expr.lineno
                                    )
                                    if hint:
                                        hints.append(hint)

    return hints


def _parse_manual_span_call(
    call: ast.Call, file: Path, line: int
) -> ManualSpanHint | None:
    """Parse a manual_span(op_name, component_name, tags=...) call."""
    if len(call.args) < 2:
        return None

    op_name = ast.unparse(call.args[0])
    component_name_arg = call.args[1]

    if not isinstance(component_name_arg, ast.Constant) or not isinstance(
        component_name_arg.value, str
    ):
        return None

    component_name = component_name_arg.value
    tags = {}

    dict_arg = None
    if len(call.args) >= 3 and isinstance(call.args[2], ast.Dict):
        dict_arg = call.args[2]
    else:
        for keyword in call.keywords:
            if keyword.arg == "tags" and isinstance(keyword.value, ast.Dict):
                dict_arg = keyword.value
                break

    if dict_arg:
        for key, value in zip(dict_arg.keys, dict_arg.values):
            # Only keep literal string values
            if isinstance(key, ast.Constant) and isinstance(value, ast.Constant):
                if isinstance(key.value, str) and isinstance(value.value, str):
                    tags[key.value] = value.value

    return ManualSpanHint(
        op_name=op_name,
        component_name=component_name,
        tags=tags,
        file=file,
        line=line,
    )


def _extract_tool_candidates(
    file: Path,
    dict_node: ast.Dict,
    module_level_async_functions: dict[str, ast.AsyncFunctionDef],
    candidates: list[CandidateComponent],
    owner_class_name: str | None = None,
    source_lines: list[str] | None = None,
) -> None:
    """Extract tool candidates from a dict literal with async function values."""
    all_async_funcs = True
    for value in dict_node.values:
        if not (
            isinstance(value, ast.Name) and value.id in module_level_async_functions
        ):
            all_async_funcs = False
            break

    if not all_async_funcs:
        return

    for key, value in zip(dict_node.keys, dict_node.values):
        if not (
            isinstance(key, ast.Constant)
            and isinstance(key.value, str)
            and isinstance(value, ast.Name)
        ):
            continue

        registered_name = key.value
        func_name = value.id

        if func_name not in module_level_async_functions:
            continue

        func_node = module_level_async_functions[func_name]
        # Use pre-parsed source_lines if available, else read from file
        if source_lines is None:
            source_lines = file.read_text(encoding="utf-8").splitlines()
        source_snippet = _extract_source_snippet(file, func_node.lineno, source_lines)

        candidate = CandidateComponent(
            file=file,
            line=func_node.lineno,
            class_name=func_name,
            is_tool=True,
            registered_name=registered_name,
            owner_class_name=owner_class_name,
            entry_kind="function",
            source_snippet=source_snippet,
        )
        candidates.append(candidate)


def _split_sub_spans(candidates: list[CandidateComponent]) -> list[CandidateComponent]:
    """Split candidates with multi-valued tag keys in their manual_span_hints."""
    result = []

    for candidate in candidates:
        hints_by_component_name: dict[str, list[ManualSpanHint]] = {}
        for hint in candidate.manual_span_hints:
            if hint.component_name not in hints_by_component_name:
                hints_by_component_name[hint.component_name] = []
            hints_by_component_name[hint.component_name].append(hint)

        should_split = False
        split_info: dict[str, dict[str, list[ManualSpanHint]]] = {}

        for comp_name, hints in hints_by_component_name.items():
            if len(hints) == 1:
                continue

            tag_values: dict[str, set[str]] = {}
            for hint in hints:
                for key, value in hint.tags.items():
                    if key not in tag_values:
                        tag_values[key] = set()
                    tag_values[key].add(value)

            for key, values in tag_values.items():
                if len(values) > 1:
                    should_split = True
                    if key not in split_info:
                        split_info[key] = {}
                    for value in values:
                        if value not in split_info[key]:
                            split_info[key][value] = []
                        split_info[key][value].extend(
                            [h for h in hints if h.tags.get(key) == value]
                        )

        if not should_split:
            result.append(candidate)
        else:
            for tag_key, value_dict in split_info.items():
                for tag_value, hints_for_value in value_dict.items():
                    split_candidate = CandidateComponent(
                        file=candidate.file,
                        line=candidate.line,
                        class_name=candidate.class_name,
                        tag_suffix=tag_value,
                        haystack_name=candidate.haystack_name,
                        is_tool=candidate.is_tool,
                        registered_name=candidate.registered_name,
                        owner_class_name=candidate.owner_class_name,
                        entry_kind=candidate.entry_kind,
                        source_snippet=candidate.source_snippet,
                        model_hints=candidate.model_hints,
                        manual_span_hints=hints_for_value,
                    )
                    result.append(split_candidate)
                break  # Only split on the first multi-valued key

    return result


def _dedup_and_sort(candidates: list[CandidateComponent]) -> list[CandidateComponent]:
    """Deduplicate by (file, class_name, tag_suffix) and sort by (file, line)."""
    seen: set[tuple[str, str, str | None]] = set()
    deduped = []

    for candidate in candidates:
        key = (str(candidate.file), candidate.class_name, candidate.tag_suffix)
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)

    deduped.sort(key=lambda c: (str(c.file), c.line))

    return deduped


class LangGraphScanner:
    """Scan a StateGraph-based target: add_node() (function / class / bound-method node, keeping the
    owner class), create_react_agent() call sites, and the @task/@entrypoint functional API. Purely
    structural — matches any StateGraph/add_node/self.<method>, never a specific target's names."""

    framework = "langgraph"

    def scan(self, source_files: list[Path]) -> list[CandidateComponent]:
        candidates: list[CandidateComponent] = []
        asts: dict[Path, ast.Module] = {}
        file_contents: dict[str, str] = {}

        for file in source_files:
            parsed = parse_python_source(file)
            if parsed is None:
                continue
            source, tree = parsed
            asts[file] = tree
            file_contents[str(file)] = source

        for file, tree in asts.items():
            imports_lg = _imports_langgraph(tree)
            stategraph_names = _build_stategraph_var_names(tree)
            if not imports_lg and not stategraph_names:
                continue  # file never touches langgraph — scan nothing (keep zero false positives)

            source_lines = file_contents[str(file)].splitlines()
            self_attr_classes = _build_self_attr_classes(tree)
            enclosing_class_map = _build_enclosing_class_map(tree)
            known_class_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}

            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_node"
                    and len(node.args) >= 2
                    and _receiver_name(node.func.value) in stategraph_names
                ):
                    alias = _const_str(node.args[0])
                    callee, entry_kind, owner = _resolve_node_target(
                        node.args[1], enclosing_class_map, self_attr_classes,
                        known_class_names, str(file), file_contents,
                    )
                    if alias and callee:
                        arg1 = node.args[1]
                        line = arg1.lineno if hasattr(arg1, "lineno") else node.lineno
                        candidates.append(CandidateComponent(
                            file=file, line=line, class_name=callee,
                            entry_kind=entry_kind, owner_class_name=owner,
                            source_snippet=_extract_source_snippet(file, node.lineno, source_lines),
                        ))

                elif (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Call)
                    and _call_func_name(node.value) == "create_react_agent"
                    and imports_lg
                ):
                    candidates.append(CandidateComponent(
                        file=file, line=node.lineno, class_name=node.targets[0].id,
                        entry_kind="function",
                        source_snippet=_extract_source_snippet(file, node.lineno, source_lines),
                    ))

            if imports_lg:
                for node in tree.body:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if any(
                            _decorator_effective_name(d) in ("task", "entrypoint")
                            for d in node.decorator_list
                        ):
                            candidates.append(CandidateComponent(
                                file=file, line=node.lineno, class_name=node.name,
                                entry_kind="function",
                                source_snippet=_extract_source_snippet(file, node.lineno, source_lines),
                            ))

        candidates = _split_sub_spans(candidates)
        candidates = _dedup_and_sort(candidates)
        return candidates


# Tuple of classes (not a dict) so the lookup key IS the class's own `framework` attr — no
# second source of truth to drift. CS-312/313 append their scanner class here.
_REGISTERED_SCANNERS: tuple[type[FrameworkScanner], ...] = (HaystackScanner, LangGraphScanner)


def get_scanner(framework: str | None) -> FrameworkScanner:
    """Return the scanner whose `framework` matches; falls back to HaystackScanner (never raises)."""
    for cls in _REGISTERED_SCANNERS:
        if cls.framework == framework:
            return cls()
    return HaystackScanner()
