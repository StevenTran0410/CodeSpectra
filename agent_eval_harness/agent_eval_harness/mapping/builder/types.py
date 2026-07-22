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


@dataclass
class ManualSpanHint:
    """A single manual_span() call detected in source code."""

    op_name: str  # raw source text of the first manual_span() arg
    component_name: str  # second positional arg (always a string literal)
    tags: dict[str, str]  # third arg dict — only literal string values kept
    file: Path
    line: int


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
    source_snippet: str = ""  # first ~30 lines of the class/function body
    model_hints: list[str] = field(default_factory=list)
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
