"""Framework scanner protocol and Haystack implementation."""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Protocol, runtime_checkable

from agent_eval_harness.mapping.builder.types import (
    CandidateComponent,
    ManualSpanHint,
)


@runtime_checkable
class FrameworkScanner(Protocol):
    """Interface for framework-specific component scanners."""

    def scan(self, source_files: list[Path]) -> list[CandidateComponent]: ...


class HaystackScanner:
    """Scans a Haystack-based target for component candidates."""

    def scan(self, source_files: list[Path]) -> list[CandidateComponent]:
        """Scan source files and return deduplicated, sorted component candidates."""
        candidates = []

        # Cache ASTs upfront to avoid quadratic re-parsing in the second pass
        asts: dict[Path, ast.Module] = {}
        known_classes: set[str] = set()
        add_component_names: dict[str, str] = {}  # {haystack_name: class_name}
        module_level_async_functions: dict[str, ast.AsyncFunctionDef] = {}

        # Parse all files once upfront
        for file in source_files:
            try:
                source = file.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(file))
                asts[file] = tree
            except SyntaxError as exc:
                print(f"[aeh map] skipping {file}: {exc}", file=sys.stderr)
                continue

        # First pass: build known_classes and add_component_names
        for file, tree in asts.items():
            # Collect top-level async functions for tool discovery
            for node in tree.body:
                if isinstance(node, ast.AsyncFunctionDef):
                    module_level_async_functions[node.name] = node

            # Find @component classes and add_component() calls
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    # Check for @component decorator
                    for decorator in node.decorator_list:
                        if self._is_component_decorator(decorator):
                            known_classes.add(node.name)
                            break

                # Find add_component() calls
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if self._is_add_component_call(child):
                            if (
                                len(child.args) >= 2
                                and isinstance(child.args[0], ast.Constant)
                                and isinstance(child.args[0].value, str)
                            ):
                                haystack_name = child.args[0].value
                                if isinstance(child.args[1], ast.Call) and isinstance(
                                    child.args[1].func, ast.Name
                                ):
                                    class_name = child.args[1].func.id
                                    add_component_names[haystack_name] = class_name

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
                                            # Check if this name is an actual class in asts
                                            for check_tree in asts.values():
                                                for check_node in check_tree.body:
                                                    if (
                                                        isinstance(check_node, ast.ClassDef)
                                                        and check_node.name
                                                        == name_node.id
                                                    ):
                                                        known_classes.add(name_node.id)

            # Now extract candidates
            source = file.read_text(encoding="utf-8")
            source_lines = source.splitlines()

            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name in known_classes:
                    # Check if it has @component decorator
                    is_component = False
                    for decorator in node.decorator_list:
                        if self._is_component_decorator(decorator):
                            is_component = True
                            break

                    # Get haystack_name if it's registered via add_component
                    haystack_name = None
                    for name, class_name in add_component_names.items():
                        if class_name == node.name:
                            haystack_name = name
                            break

                    docstring = ast.get_docstring(node) or ""
                    source_snippet = self._extract_source_snippet(
                        file, node.lineno, source_lines
                    )
                    manual_span_hints = self._extract_manual_spans(node, file)

                    # If not @component-decorated but in known_classes, it's composed
                    is_composed = not is_component

                    candidate = CandidateComponent(
                        file=file,
                        line=node.lineno,
                        class_name=node.name,
                        haystack_name=haystack_name,
                        is_haystack_component=is_component,
                        is_composed=is_composed,
                        docstring=docstring,
                        source_snippet=source_snippet,
                        manual_span_hints=manual_span_hints,
                    )
                    candidates.append(candidate)

            # Tool discovery: find dict literals with async function values
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    # Track the owner class name (e.g., for WorkerComponent(...))
                    owner_class_name = None
                    if isinstance(node.func, ast.Name):
                        owner_class_name = node.func.id

                    for arg in node.args:
                        if isinstance(arg, ast.Dict):
                            self._extract_tool_candidates(
                                file,
                                arg,
                                module_level_async_functions,
                                candidates,
                                owner_class_name,
                                source_lines,
                            )
                    for keyword in node.keywords:
                        if isinstance(keyword.value, ast.Dict):
                            self._extract_tool_candidates(
                                file,
                                keyword.value,
                                module_level_async_functions,
                                candidates,
                                owner_class_name,
                                source_lines,
                            )

        # Sub-span split and dedup/sort
        candidates = self._split_sub_spans(candidates)
        candidates = self._dedup_and_sort(candidates)

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

    def _extract_source_snippet(self, file: Path, line: int, lines: list[str]) -> str:
        """Extract first ~30 lines of a class/function body."""
        if line <= 1 or line > len(lines):
            return ""
        # Start from the line after the def/class line
        start = line
        end = min(line + 29, len(lines))
        return "\n".join(lines[start : end + 1])

    def _extract_manual_spans(self, node: ast.ClassDef, file: Path) -> list[ManualSpanHint]:
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
                                        hint = self._parse_manual_span_call(
                                            context_expr, file, context_expr.lineno
                                        )
                                        if hint:
                                            hints.append(hint)

        return hints

    def _parse_manual_span_call(
        self, call: ast.Call, file: Path, line: int
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

        # Extract tags from the dict (args[2] or keyword arg)
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
        self,
        file: Path,
        dict_node: ast.Dict,
        module_level_async_functions: dict[str, ast.AsyncFunctionDef],
        candidates: list[CandidateComponent],
        owner_class_name: str | None = None,
        source_lines: list[str] | None = None,
    ) -> None:
        """Extract tool candidates from a dict literal with async function values."""
        # Check if ALL values are Name nodes referencing async functions
        all_async_funcs = True
        for value in dict_node.values:
            if not (
                isinstance(value, ast.Name) and value.id in module_level_async_functions
            ):
                all_async_funcs = False
                break

        if not all_async_funcs:
            return

        # Extract each key/value pair
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
            docstring = ast.get_docstring(func_node) or ""
            # Use pre-parsed source_lines if available, else read from file
            if source_lines is None:
                source_lines = file.read_text(encoding="utf-8").splitlines()
            source_snippet = self._extract_source_snippet(file, func_node.lineno, source_lines)

            candidate = CandidateComponent(
                file=file,
                line=func_node.lineno,
                class_name=func_name,
                is_tool=True,
                registered_name=registered_name,
                owner_class_name=owner_class_name,
                docstring=docstring,
                source_snippet=source_snippet,
            )
            candidates.append(candidate)

    def _split_sub_spans(self, candidates: list[CandidateComponent]) -> list[CandidateComponent]:
        """Split candidates with multi-valued tag keys in their manual_span_hints."""
        result = []

        for candidate in candidates:
            # Group hints by component_name
            hints_by_component_name: dict[str, list[ManualSpanHint]] = {}
            for hint in candidate.manual_span_hints:
                if hint.component_name not in hints_by_component_name:
                    hints_by_component_name[hint.component_name] = []
                hints_by_component_name[hint.component_name].append(hint)

            # Check if any component_name group has a multi-valued tag key
            should_split = False
            split_info: dict[str, dict[str, list[ManualSpanHint]]] = {}

            for comp_name, hints in hints_by_component_name.items():
                if len(hints) == 1:
                    continue

                # Check for multi-valued tag keys
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
                # Create split candidates
                for tag_key, value_dict in split_info.items():
                    for tag_value, hints_for_value in value_dict.items():
                        split_candidate = CandidateComponent(
                            file=candidate.file,
                            line=candidate.line,
                            class_name=candidate.class_name,
                            tag_suffix=tag_value,
                            haystack_name=candidate.haystack_name,
                            is_haystack_component=candidate.is_haystack_component,
                            is_composed=candidate.is_composed,
                            is_tool=candidate.is_tool,
                            registered_name=candidate.registered_name,
                            docstring=candidate.docstring,
                            source_snippet=candidate.source_snippet,
                            model_hints=candidate.model_hints,
                            manual_span_hints=hints_for_value,
                        )
                        result.append(split_candidate)
                    break  # Only split on the first multi-valued key

        return result

    def _dedup_and_sort(self, candidates: list[CandidateComponent]) -> list[CandidateComponent]:
        """Deduplicate by (file, class_name, tag_suffix) and sort by (file, line)."""
        seen: set[tuple[str, str, str | None]] = set()
        deduped = []

        for candidate in candidates:
            key = (str(candidate.file), candidate.class_name, candidate.tag_suffix)
            if key not in seen:
                seen.add(key)
                deduped.append(candidate)

        # Sort by (file, line)
        deduped.sort(key=lambda c: (str(c.file), c.line))

        return deduped
