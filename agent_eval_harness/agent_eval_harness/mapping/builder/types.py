"""Shared types for the map builder pipeline."""
from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("agent_eval_harness.mapping.builder.types")


def parse_python_source(file: Path) -> tuple[str, ast.Module] | None:
    """Read+parse one Python file; None (and a logged skip) if it can't be read or parsed —
    shared by every pass so a bad file is dropped consistently instead of crashing the scan."""
    try:
        source = file.read_text(encoding="utf-8")
        return source, ast.parse(source, filename=str(file))
    except (SyntaxError, OSError, UnicodeDecodeError) as exc:
        logger.warning("skipping unparseable file %s: %s", file, exc)
        return None


def find_symbol_node(tree: ast.AST, names: list[str]) -> ast.AST | None:
    """Resolve `Owner.member` / `Owner` / `func` to its OWN definition node, never a caller's
    reference to it. Shared by the LangGraph scanner (anchor a node candidate on its own def, not
    the add_node() call site) and agent_flow's uncapped re-extraction."""
    owner, member = (names[0], names[1]) if len(names) >= 2 else (names[0], None)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != owner:
            continue
        if member is None:
            return node
        for sub in getattr(node, "body", []):
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name == member:
                return sub
    return None


_DEFAULT_SNIPPET_CAP = 30  # line cap for a function/method body
_SNIPPET_CAP_BY_KIND = {"ClassDef": 200}  # a class body may legitimately be long; a function may not


def _extract_source_snippet(node: ast.AST, lines: list[str]) -> tuple[str, bool]:
    """Extract a symbol's OWN body: anchored on `node`'s own def/class line, ending at its real
    `end_lineno` (never a caller's reference to it), capped by kind so one huge symbol can't blow
    the prompt budget. Returns (snippet, truncated) so the consumer can tell a short symbol from
    one the cap actually clipped. Shared by every scanner (Haystack/LangGraph/LCEL/plain-python) so
    there is one anchor-and-bound implementation, not a per-scanner copy."""
    start_line = getattr(node, "lineno", None)
    end_lineno = getattr(node, "end_lineno", None)
    if not start_line or not end_lineno or start_line > len(lines):
        return "", False
    cap = _SNIPPET_CAP_BY_KIND.get(type(node).__name__, _DEFAULT_SNIPPET_CAP)
    capped_end = min(end_lineno, start_line + cap)
    return "\n".join(lines[start_line:capped_end]), capped_end < end_lineno


@dataclass
class ManualSpanHint:
    """A single manual_span() call detected in source code."""

    op_name: str  # raw source text of the first manual_span() arg
    component_name: str  # second positional arg (always a string literal)
    tags: dict[str, str]  # third arg dict — only literal string values kept
    file: Path
    line: int
    end_line: int | None = None  # enclosing `with` block's end_lineno — scopes a split child's own evidence region


@dataclass
class CandidateComponent:
    """A source-code class or function that might be an agentic component."""

    file: Path
    line: int
    class_name: str  # Python class or function name
    tag_suffix: str | None = None  # e.g. "rule" / "llm" for split candidates
    haystack_name: str | None = None  # name from add_component()
    is_tool: bool = False  # async def referenced in dict at call site
    registered_name: str | None = None  # dict key for tool candidates
    owner_class_name: str | None = None  # class owning this tool OR bound-method; entry_kind disambiguates, the two never co-occur
    entry_kind: str = "class"  # "class" | "function" | "bound_method"
    is_library_object: bool = False  # LCEL library/framework link (ChatOpenAI, StrOutputParser) — surfaced then degraded, never harvestable
    framework: str | None = None  # producing scanner's framework; set by scan_all, used to scope a split per-system map to its own components
    source_snippet: str = ""  # the symbol's own body, anchored+bounded by _extract_source_snippet
    snippet_truncated: bool = False  # True when the by-kind cap actually clipped the body
    manual_span_hints: list[ManualSpanHint] = field(default_factory=list)

    @property
    def candidate_id(self) -> str:
        """Simplified class_name (removing 'Component' suffix) + optional _<tag_suffix>."""
        # Use haystack_name if available (preferred), else strip "Component" from class_name
        if self.haystack_name:
            base = self.haystack_name
        else:
            base = self.class_name
            if base.endswith("Component"):
                base = base[:-9]  # Remove "Component" suffix
            base = base.lower()
        return f"{base}_{self.tag_suffix}" if self.tag_suffix else base


@dataclass
class TopologyEdges:
    """Upstream and downstream nodes in the component graph."""

    upstream: list[str] = field(default_factory=list)
    downstream: list[str] = field(default_factory=list)
    constructor_downstream: list[str] = field(default_factory=list)  # sibling components this one constructs/owns
    # CS-319 (additive; empty on every non-langgraph map so Haystack is byte-identical): the subset of
    # `downstream` reached by a "call" edge (intra-node call to a gray target), and the subset of
    # `downstream` whose hard edge is a conditional router branch (add_conditional_edges source).
    call_downstream: list[str] = field(default_factory=list)
    conditional_downstream: list[str] = field(default_factory=list)
    # CS-321: motif classification over the downstream graph (linear/branch/loop, shared-state reserved).
    motif: str | None = None
