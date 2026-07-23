"""PlainPythonScanner — surfaces hand-rolled, framework-free agents (no Haystack/LangGraph/LangChain)
so a plain-python target reaches the map instead of having zero scanner match.

Signal-based, NOT a rigid SDK template: real plain-python agents vary across SDKs (OpenAI/Anthropic/raw
HTTP/provider-abstraction), loop structures, and tool-registration styles, so a template tuned to one
vendor's `.chat.completions.create` shape would miss the common case (e.g. a provider-abstraction agent
that calls an inherited `self._call(...)`).

Coverage:
  AST-covered, deterministic, 0 LLM:
    - CORE: a class named `*Agent` corroborated by ANY of {an entry method (run/run_async/__call__), a
      `self.<attr>(...)` call whose attr looks like an LLM call, a referenced module-level system-prompt
      constant}. The corroborator holds precision — a bare `*Agent` dataclass/mock is not emitted.
    - SECONDARY (lower-confidence, one vendor's shape): a `tools=[fn, ...]` list literal of module-level
      functions → tool components. Validated only synthetically; ~0 recall on provider-abstraction agents.
  llm_fallback (wiring_block, source="llm_fallback"): shapes the AST can't fit — non-`*Agent` names,
    free-function agents, exotic dispatch — best-effort via Stage-1 Pass D.
  Out of scope: multi-file agent graphs, decorator-registered tool frameworks (CrewAI/AutoGen), and
    runtime-dynamic tool registration.

Framework-generic: matches the `*Agent` naming idiom + generic LLM-call/prompt signals, never any one
target's symbols.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

from agent_eval_harness.mapping.builder.types import CandidateComponent, parse_python_source

_SNIPPET_LINE_COUNT = 30
_AGENT_CLASS_SUFFIX_RE = re.compile(r"Agent$")
# Generic LLM-call method-name signal (provider-abstraction `self._call_stream`, raw `chat.completions
# .create`, `client.messages.create`, `.invoke`, `.generate`, …) — a corroborator, never sufficient alone.
_LLM_CALL_ATTR_RE = re.compile(r"(?i)(call|chat|complete|completion|stream|invoke|generate|predict|messages)")
# System-prompt-shaped module constant: ALL-CAPS name ending in _SYSTEM/_PROMPT/_INSTRUCTION(S).
_PROMPT_CONST_RE = re.compile(r"(?:^[A-Z0-9_]+$)")
_PROMPT_CONST_SUFFIX = ("_SYSTEM", "_PROMPT", "_INSTRUCTION", "_INSTRUCTIONS")
_ENTRY_METHOD_NAMES = ("run", "run_async", "__call__")


def _snippet(lines: list[str], line: int) -> str:
    if line <= 1 or line > len(lines):
        return ""
    start = line
    end = min(line + _SNIPPET_LINE_COUNT - 1, len(lines))
    return "\n".join(lines[start : end + 1])


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
    """Scan a framework-free target for `*Agent`-shaped classes + their tool functions."""

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
            prompt_consts = self._module_prompt_consts(tree)

            for node in tree.body:
                if isinstance(node, ast.ClassDef) and _AGENT_CLASS_SUFFIX_RE.search(node.name):
                    if self._corroborated(node, prompt_consts):
                        candidates.append(CandidateComponent(
                            file=file, line=node.lineno, class_name=node.name,
                            entry_kind="class",
                            source_snippet=_snippet(source_lines, node.lineno),
                        ))

            self._scan_tool_lists(tree, file, module_functions, source_lines, candidates)

        if wiring_block is not None:
            self._scan_wiring_fallback(wiring_block, asts, sources, candidates)

        return _dedup_and_sort(candidates)

    def _module_prompt_consts(self, tree: ast.AST) -> set[str]:
        consts: set[str] = set()
        for node in getattr(tree, "body", []):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and (
                        tgt.id.endswith(_PROMPT_CONST_SUFFIX) or _PROMPT_CONST_RE.match(tgt.id)
                    ):
                        consts.add(tgt.id)
        return consts

    def _corroborated(self, cls: ast.ClassDef, prompt_consts: set[str]) -> bool:
        """A `*Agent` class is emitted only with a second signal, to keep precision."""
        for item in cls.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name in _ENTRY_METHOD_NAMES:
                return True
        for child in ast.walk(cls):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                if _LLM_CALL_ATTR_RE.search(child.func.attr):
                    return True
            if isinstance(child, ast.Name) and child.id in prompt_consts:
                return True
        return False

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
                candidates.append(CandidateComponent(
                    file=file, line=elt.lineno, class_name=elt.id, is_tool=True,
                    registered_name=elt.id, owner_class_name=owner, entry_kind="function",
                    source_snippet=_snippet(source_lines, elt.lineno),
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
        """llm_fallback path: a persisted wiring_block (source='llm_fallback') names components the AST
        heuristic missed; resolve each to a real ClassDef/FunctionDef in the target and emit it."""
        for w_node in getattr(wiring_block, "nodes", []):
            name = getattr(w_node, "callee_name", None)
            hint = getattr(w_node, "source_hint_file", "") or ""
            if not name:
                continue
            for file, tree in asts.items():
                if hint and not str(file).endswith(hint) and Path(hint).name != Path(file).name:
                    continue
                source_lines = sources[file].splitlines()
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and node.name == name:
                        candidates.append(CandidateComponent(
                            file=file, line=node.lineno, class_name=name, entry_kind="class",
                            source_snippet=_snippet(source_lines, node.lineno),
                        ))
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                        candidates.append(CandidateComponent(
                            file=file, line=node.lineno, class_name=name, is_tool=True,
                            registered_name=name, entry_kind="function",
                            source_snippet=_snippet(source_lines, node.lineno),
                        ))
