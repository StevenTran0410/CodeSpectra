"""Symbol reference edge extraction — CS-202.

Extracts symbol-level call edges from Python and TypeScript source files.
Cross-file import resolution narrows which symbol is being called when
multiple candidates share the same name.

Storage table::

    CREATE TABLE symbol_graph_edges (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        snapshot_id    TEXT    NOT NULL,
        src_symbol     TEXT    NOT NULL,
        dst_symbol     TEXT    NOT NULL,
        edge_type      TEXT    NOT NULL DEFAULT 'calls',
        confidence     TEXT    NOT NULL DEFAULT 'high',
        evidence_lines TEXT    NOT NULL DEFAULT '[]'
    );
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SymbolEdge:
    """A directed call/reference edge between two symbols.

    Attributes:
        src_symbol:         Fully-qualified caller in ``file.py::Class.method`` form.
        dst_symbol:         Fully-qualified callee in the same form.
        edge_type:          Relationship kind — ``"calls"``, ``"returns"``, or
                            ``"param_type"``.
        confidence_score:   Numeric confidence (0.0-1.0) — 1.0 is certain,
                            0.0 is no confidence.  Stored as REAL in DB.
        resolution_method:  Label describing how the edge was resolved (e.g.
                            'import_path_match', 'same_file_scope',
                            'mro_resolved', 'constructor_type_trace',
                            'name_heuristic_ambiguous', 'unknown').
        confidence:         (Legacy) Derived from confidence_score: 'high' if
                            score >= 0.7 else 'low' (for back-compat with
                            existing string-based filters).
        evidence_lines:     Line numbers in the *source* file where the reference
                            was observed.
    """

    src_symbol: str
    dst_symbol: str
    edge_type: str = "calls"
    confidence_score: float = 0.7
    resolution_method: str = "unknown"
    evidence_lines: list[int] = field(default_factory=list)

    @property
    def confidence(self) -> str:
        """Legacy confidence string derived from confidence_score for back-compat.

        Returns 'high' if score >= 0.7 else 'low' to maintain backward compatibility
        with existing string-based filters.
        """
        return "high" if self.confidence_score >= 0.7 else "low"


class SymbolGraphBuilder:
    """Extract symbol-level call edges from a set of source files.

    Usage::

        builder = SymbolGraphBuilder()
        edges = builder.build({
            "auth.py": open("auth.py").read(),
            "api.py":  open("api.py").read(),
        })

    All files are analysed as a unit so that import chains can be traced
    across file boundaries.
    """

    def build(self, sources: dict[str, str]) -> list[SymbolEdge]:
        """Return symbol reference edges for the given source files.

        Args:
            sources: Mapping of ``filename → source code``.  All files are
                     analysed as a unit so that import chains can be traced.

        Returns:
            List of :class:`SymbolEdge` objects.  Unresolvable call sites
            (duck typing, star imports, ambiguous DI) are omitted — never
            silently guessed.
        """
        from ._symbol_parser import parse_file
        from ._symbol_resolver import resolve_edges

        parsed_files = {}
        for filename, source in sources.items():
            try:
                pf = parse_file(filename, source)
                if pf is not None:
                    parsed_files[filename] = pf
            except Exception:
                logger.exception("SymbolGraphBuilder.build: failed to parse %s", filename)

        return resolve_edges(parsed_files)
