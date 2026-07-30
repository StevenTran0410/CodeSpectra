"""PlainPythonScanner — surfaces hand-rolled, framework-free agents (no Haystack/LangGraph/LangChain)
so a plain-python target reaches the map instead of having zero scanner match.

Signal-based, NOT a rigid SDK template: real plain-python agents vary across SDKs (OpenAI/Anthropic/raw
HTTP/provider-abstraction), loop structures, and tool-registration styles, so a template tuned to one
vendor's `.chat.completions.create` shape would miss the common case (e.g. a provider-abstraction agent
that calls an inherited `self._call(...)`).

Coverage:
  AST-covered, deterministic, 0 LLM:
    - CORE: any class with a recognized entry method (run/run_async/__call__) whose call closure
      contains at least one call — every such call is either a resolved boundary or an unresolved
      seam, so a class is emitted without reading any identifier's English meaning. This
      deliberately widens candidate emission; narrowing happens at the verdict layer, which can
      degrade to `unknown` instead of silently dropping. A bare stub entry method with zero calls
      (e.g. a mock/dataclass) is excluded — there is nothing to be evidence for or against.
    - SECONDARY (lower-confidence, one vendor's shape): a `tools=[fn, ...]` list literal of module-level
      functions → tool components. Validated only synthetically; ~0 recall on provider-abstraction agents.
  llm_fallback (wiring_block, source="llm_fallback"): shapes the AST can't fit — free-function agents,
    exotic dispatch — best-effort via Stage-1 Pass D.
  Out of scope: multi-file agent graphs, decorator-registered tool frameworks (CrewAI/AutoGen), and
    runtime-dynamic tool registration.

Framework-generic: matches structural call-closure activity, never any one target's symbols.
"""
from __future__ import annotations

import ast
from pathlib import Path

from agent_eval_harness.discovery.wiring import (
    _class_def_node,
    _container_call_sites,
    _imports_langgraph,
    _is_pure_data_classdef,
    _iter_method_calls,
    _self_attr_containers,
)
from agent_eval_harness.mapping.builder.resolution_primitives import recognized_entry_methods
from agent_eval_harness.mapping.builder.types import (
    CandidateComponent,
    _extract_source_snippet,
    find_symbol_node,
    parse_python_source,
)


def _dedup_and_sort(candidates: list[CandidateComponent]) -> list[CandidateComponent]:
    seen: set[tuple[str, str, str | None]] = set()
    deduped: list[CandidateComponent] = []
    for c in candidates:
        key = (str(c.file), c.class_name, c.tag_suffix)
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    deduped.sort(key=lambda c: (str(c.file), c.line))
    return deduped


class PlainPythonScanner:
    """Scan a framework-free target for classes with entry-method call-closure activity + their
    tool functions."""

    framework = "plain_python"

    def scan(
        self, source_files: list[Path], wiring_block: object | None = None
    ) -> list[CandidateComponent]:
        candidates: list[CandidateComponent] = []
        asts: dict[Path, ast.Module] = {}
        sources: dict[Path, str] = {}

        for file in source_files:
            parsed = parse_python_source(file)
            if parsed is None:
                continue
            sources[file], asts[file] = parsed[0], parsed[1]

        for file, tree in asts.items():
            source_lines = sources[file].splitlines()
            module_functions = {
                n.name for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            # A LangGraph file's node/router methods are referenced (never self-called) as callback
            # arguments to add_node/add_conditional_edges, so the "no intra-class caller" rule would
            # wrongly re-admit them here -- that graph is LangGraphScanner's own territory.
            owned_by_langgraph = _imports_langgraph(tree)

            for node in tree.body:
                if (isinstance(node, ast.ClassDef) and not owned_by_langgraph
                        and self._has_closure_activity(tree, node.name)):
                    snippet, truncated = _extract_source_snippet(node, source_lines)
                    candidates.append(CandidateComponent(
                        file=file, line=node.lineno, class_name=node.name,
                        entry_kind="class",
                        source_snippet=snippet, snippet_truncated=truncated,
                    ))

            self._scan_tool_lists(tree, file, module_functions, source_lines, candidates)

        if wiring_block is not None:
            self._scan_wiring_fallback(wiring_block, asts, sources, candidates)

        return _dedup_and_sort(candidates)

    def _has_closure_activity(self, tree: ast.AST, class_name: str) -> bool:
        """Structural emission gate: a class is a candidate iff it has a recognized entry method
        (no intra-class caller) whose call closure contains at least one call -- any call is either
        a resolved boundary or an unresolved seam, so no resolution is needed here, only the same
        shape-recognition the generic call closure also uses. Replaces the Agent$-suffix /
        LLM-verb-attr / prompt-const regex gates, which decided from an identifier's English
        meaning. A pure-data class (dataclass/TypedDict/NamedTuple/BaseModel) is excluded even when
        one of its methods has calls (e.g. a classmethod factory) -- it is state, not an agent."""
        cls_node = _class_def_node(tree, class_name)
        if cls_node is not None and _is_pure_data_classdef(cls_node):
            return False
        entry_methods = recognized_entry_methods(tree, class_name)
        if not entry_methods:
            return False
        self_containers = _self_attr_containers(tree, class_name)
        return any(
            _iter_method_calls(m) or _container_call_sites(m, self_containers)
            for m in entry_methods
        )

    def _scan_tool_lists(
        self, tree: ast.AST, file: Path, module_functions: set[str],
        source_lines: list[str], candidates: list[CandidateComponent],
    ) -> None:
        """SECONDARY: `tools = [fn, ...]` list of module-level functions → tool components."""
        enclosing = self._enclosing_class_map(tree)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.List)):
                continue
            elts = node.value.elts
            if not elts or not all(isinstance(e, ast.Name) and e.id in module_functions for e in elts):
                continue
            owner = enclosing.get(id(node))
            for elt in elts:
                def_node = find_symbol_node(tree, [elt.id])
                snippet, truncated = (
                    _extract_source_snippet(def_node, source_lines) if def_node is not None else ("", False)
                )
                candidates.append(CandidateComponent(
                    file=file, line=elt.lineno, class_name=elt.id, is_tool=True,
                    registered_name=elt.id, owner_class_name=owner, entry_kind="function",
                    source_snippet=snippet, snippet_truncated=truncated,
                ))

    def _enclosing_class_map(self, tree: ast.AST) -> dict[int, str]:
        mapping: dict[int, str] = {}
        for cls in ast.walk(tree):
            if isinstance(cls, ast.ClassDef):
                for child in ast.walk(cls):
                    mapping[id(child)] = cls.name
        return mapping

    def _scan_wiring_fallback(
        self, wiring_block: object, asts: dict, sources: dict,
        candidates: list[CandidateComponent],
    ) -> None:
        """A persisted wiring_block names steps the AST heuristic missed; resolve each to a real
        ClassDef/FunctionDef anywhere in the scanned set and emit it. The whole scanned set is
        searched (not just the hint file) because a step's DEFINING file — an imported helper or an
        inherited base class — is admitted to the scan set by the CS-318 unscanned-step resolver, so
        the definer is present under a different path than the call-site hint."""
        for w_node in getattr(wiring_block, "nodes", []):
            name = getattr(w_node, "callee_name", None)
            if not name:
                continue
            for file, tree in asts.items():
                source_lines = sources[file].splitlines()
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and node.name == name:
                        snippet, truncated = _extract_source_snippet(node, source_lines)
                        candidates.append(CandidateComponent(
                            file=file, line=node.lineno, class_name=name, entry_kind="class",
                            source_snippet=snippet, snippet_truncated=truncated,
                        ))
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                        snippet, truncated = _extract_source_snippet(node, source_lines)
                        candidates.append(CandidateComponent(
                            file=file, line=node.lineno, class_name=name, is_tool=True,
                            registered_name=name, entry_kind="function",
                            source_snippet=snippet, snippet_truncated=truncated,
                        ))
