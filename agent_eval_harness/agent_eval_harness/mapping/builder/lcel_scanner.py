"""LCELScanner — surfaces LangChain (LCEL) chain components so a LangChain target maps to real
components + explicit degrade instead of vanishing silently.

Coverage (measured on a real LangChain retrieval-chain app + synthetic fixtures):
  AST-covered, deterministic, 0 LLM:
    - Factory-function idiom (dominant production style): `V = create_retrieval_chain(a, b)`,
      `V = create_history_aware_retriever(llm, retriever, prompt)`, `V = x.as_retriever(...)`, and the
      Runnable* wrappers. Each produced variable V is a chain component (entry_kind="function"); the
      linkage/edges are derived by the Stage-1 wiring detector from the call ARGS (see
      discovery/wiring.py::_langchain_chain_factories) and joined here by variable name.
    - Pipe idiom: `a | b | c`. Each operand becomes a component. A library Call (ChatOpenAI(),
      StrOutputParser()) degrades explicitly via is_library_object=True (never harvestable — there is no
      ClassDef in the target). A `RunnableLambda(user_fn)` wrapping a module-level function unwraps to a
      component for that FUNCTION (is_library_object=False) so real user code is not lost.
  Out of scope (fall to Stage-1 llm_fallback or a later ticket): chains assembled by dynamic/looped
  composition, custom Runnable subclasses declared elsewhere, cross-file partial application.

Framework-generic to the LangChain idiom (factory names, `.as_retriever`, `RunnableLambda`), never to any
one target's symbols. Import-gated so non-LangChain files are skipped.
"""
from __future__ import annotations

import ast
from pathlib import Path

from agent_eval_harness.discovery.wiring import (
    _LANGCHAIN_CHAIN_FACTORIES,
    _RETRIEVER_FACTORY_METHOD,
    _call_func_name,
    _flatten_bitor,
    _imports_langchain,
    _looks_like_type_union,
)
from agent_eval_harness.mapping.builder.types import CandidateComponent, parse_python_source

_SNIPPET_LINE_COUNT = 30


def _snippet(lines: list[str], line: int) -> str:
    if line <= 1 or line > len(lines):
        return ""
    start = line
    end = min(line + _SNIPPET_LINE_COUNT - 1, len(lines))
    return "\n".join(lines[start : end + 1])


def _dedup_and_sort(candidates: list[CandidateComponent]) -> list[CandidateComponent]:
    """Deduplicate by (file, class_name, tag_suffix), preserving the first (a factory/user component
    outranks a later library duplicate of the same name), then sort by (file, line)."""
    seen: set[tuple[str, str, str | None]] = set()
    deduped: list[CandidateComponent] = []
    for c in candidates:
        key = (str(c.file), c.class_name, c.tag_suffix)
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    deduped.sort(key=lambda c: (str(c.file), c.line))
    return deduped


class LCELScanner:
    """Scan a LangChain target for LCEL chain components (factory + pipe idioms)."""

    framework = "langchain"

    def scan(
        self, source_files: list[Path], wiring_block: object | None = None
    ) -> list[CandidateComponent]:
        candidates: list[CandidateComponent] = []

        for file in source_files:
            parsed = parse_python_source(file)
            if parsed is None:
                continue
            source, tree = parsed
            if not _imports_langchain(tree):
                continue  # keep zero false positives on non-LangChain files

            source_lines = source.splitlines()
            target_functions = {
                n.name: n
                for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            target_classes = {
                n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
            }

            self._scan_factory_idiom(tree, file, source_lines, candidates)
            self._scan_pipe_idiom(
                tree, file, source_lines, target_functions, target_classes, candidates
            )

        return _dedup_and_sort(candidates)

    def _scan_factory_idiom(
        self, tree: ast.AST, file: Path, source_lines: list[str],
        candidates: list[CandidateComponent],
    ) -> None:
        """`V = <factory>(...)` / `V = x.as_retriever(...)` anywhere (module or function body) → V is a
        Runnable component. Edges come from Stage-1 wiring; here we only emit the node."""
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call)
            ):
                continue
            callee = _call_func_name(node.value)
            if callee in _LANGCHAIN_CHAIN_FACTORIES or callee == _RETRIEVER_FACTORY_METHOD:
                candidates.append(CandidateComponent(
                    file=file, line=node.value.lineno, class_name=node.targets[0].id,
                    entry_kind="function", is_library_object=False,
                    source_snippet=_snippet(source_lines, node.value.lineno),
                ))

    def _scan_pipe_idiom(
        self, tree: ast.AST, file: Path, source_lines: list[str],
        target_functions: dict, target_classes: dict, candidates: list[CandidateComponent],
    ) -> None:
        processed: set = set()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.Expr, ast.Return, ast.Yield)):
                continue
            val = node.value
            if val is None:
                continue
            for sub in ast.walk(val):
                if not (isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.BitOr)):
                    continue
                if sub in processed:
                    continue
                operands = _flatten_bitor(sub, processed)
                if len(operands) < 2 or _looks_like_type_union(operands):
                    continue
                for op in operands:
                    cand = self._candidate_for_operand(
                        op, file, source_lines, target_functions, target_classes
                    )
                    if cand is not None:
                        candidates.append(cand)

    def _candidate_for_operand(
        self, op: ast.expr, file: Path, source_lines: list[str],
        target_functions: dict, target_classes: dict,
    ) -> CandidateComponent | None:
        line = getattr(op, "lineno", 1)
        # RunnableLambda(user_fn) → unwrap to the wrapped module-level function (real user code).
        if (
            isinstance(op, ast.Call)
            and _call_func_name(op) == "RunnableLambda"
            and len(op.args) == 1
            and isinstance(op.args[0], ast.Name)
            and op.args[0].id in target_functions
        ):
            fn = target_functions[op.args[0].id]
            return CandidateComponent(
                file=file, line=fn.lineno, class_name=op.args[0].id,
                entry_kind="function", is_library_object=False,
                source_snippet=_snippet(source_lines, fn.lineno),
            )

        name = None
        if isinstance(op, ast.Name):
            name = op.id
        elif isinstance(op, ast.Call):
            name = _call_func_name(op)
        elif isinstance(op, ast.Attribute):
            name = op.attr

        if name and name in target_classes:
            return CandidateComponent(
                file=file, line=line, class_name=name, entry_kind="class",
                is_library_object=False, source_snippet=_snippet(source_lines, line),
            )
        if name and name in target_functions:
            return CandidateComponent(
                file=file, line=line, class_name=name, entry_kind="function",
                is_library_object=False, source_snippet=_snippet(source_lines, line),
            )
        # Unresolved or library-defined → surface explicitly, then degrade (never harvestable).
        return CandidateComponent(
            file=file, line=line, class_name=name or "unknown",
            entry_kind="function", is_library_object=True,
            source_snippet=_snippet(source_lines, line),
        )
