"""Graph query helpers for deep research — call chains and impact cones."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from domain.structural_graph.service import _expand_neighbors_python, _load_native_graph
from infrastructure.db.database import get_db
from shared.logger import logger


@dataclass
class SymbolHop:
    """A single edge in the symbol call graph."""

    src_symbol: str  # "file.py::Class.method" or "file.py"
    dst_symbol: str
    edge_type: str  # "calls" | "returns" | "param_type"
    confidence: str  # "high" | "low" | "none"
    evidence_lines: list[int] = field(default_factory=list)
    confidence_score: float = 1.0  # numeric score (0.0-1.0); default for safety if not populated

    @property
    def src_file(self) -> str:
        """Extract file path from source symbol."""
        return self.src_symbol.split("::")[0] if "::" in self.src_symbol else self.src_symbol

    @property
    def dst_file(self) -> str:
        """Extract file path from destination symbol."""
        return self.dst_symbol.split("::")[0] if "::" in self.dst_symbol else self.dst_symbol


@dataclass
class TraceStep:
    """One hop in a call chain traversal."""

    step: int
    file: str
    symbols_involved: list[str]
    hops: list[SymbolHop]
    is_seed: bool = False
    path_confidence: float = 1.0  # multiplicative confidence from chain start to this step


@dataclass
class ImpactResult:
    """File-level reverse transitive closure result."""

    seed_file: str
    impacted_files: list[str]  # ordered by discovery hop
    hop_map: dict[str, int]  # file → hop at which it was discovered


async def get_callees_of(
    snapshot_id: str,
    file_path: str,
    symbol: str | None = None,
    high_confidence_only: bool = True,
    min_confidence: float | None = None,
) -> list[SymbolHop]:
    """Forward lookup: what symbols does this file (or symbol) call?

    Args:
        snapshot_id: The snapshot to query.
        file_path: The file path to query.
        symbol: Optional specific symbol; if None, returns all symbols in the file.
        high_confidence_only: (Legacy) If True, filters to confidence='high' edges.
        min_confidence: Optional numeric threshold (0.0-1.0); when set, filters to
                        confidence_score >= min_confidence. Orthogonal to high_confidence_only.
    """
    db = get_db()

    if symbol:
        prefix = f"{file_path}::{symbol}"
        # exact match on src_symbol
        query = """
            SELECT src_symbol, dst_symbol, edge_type, confidence, confidence_score, evidence_lines
            FROM symbol_graph_edges
            WHERE snapshot_id=? AND src_symbol=?
        """
        params: list = [snapshot_id, prefix]
    else:
        # All symbols defined in this file (src_symbol starts with "file_path::")
        prefix = f"{file_path}::"
        query = """
            SELECT src_symbol, dst_symbol, edge_type, confidence, confidence_score, evidence_lines
            FROM symbol_graph_edges
            WHERE snapshot_id=? AND (src_symbol LIKE ? ESCAPE '\\' OR src_symbol=?)
        """
        escaped = prefix.replace("%", "\\%").replace("_", "\\_") + "%"
        params = [snapshot_id, escaped, prefix.rstrip("::")]

    if high_confidence_only:
        query += " AND confidence_score >= 0.7"

    if min_confidence is not None:
        query += " AND confidence_score >= ?"
        params.append(min_confidence)

    async with db.execute(query, tuple(params)) as cur:
        rows = await cur.fetchall()

    return [
        SymbolHop(
            src_symbol=r["src_symbol"],
            dst_symbol=r["dst_symbol"],
            edge_type=r["edge_type"],
            confidence=r["confidence"],
            evidence_lines=json.loads(r["evidence_lines"] or "[]"),
            confidence_score=r["confidence_score"],
        )
        for r in rows
    ]


async def get_callers_of(
    snapshot_id: str,
    file_path: str,
    symbol: str | None = None,
    high_confidence_only: bool = True,
    min_confidence: float | None = None,
) -> list[SymbolHop]:
    """Reverse lookup: who calls symbols defined in this file?

    Args:
        snapshot_id: The snapshot to query.
        file_path: The file path to query.
        symbol: Optional specific symbol; if None, returns all callers to symbols in the file.
        high_confidence_only: (Legacy) If True, filters to confidence='high' edges.
        min_confidence: Optional numeric threshold (0.0-1.0); when set, filters to
                        confidence_score >= min_confidence. Orthogonal to high_confidence_only.
    """
    db = get_db()

    if symbol:
        prefix = f"{file_path}::{symbol}"
        query = """
            SELECT src_symbol, dst_symbol, edge_type, confidence, confidence_score, evidence_lines
            FROM symbol_graph_edges
            WHERE snapshot_id=? AND dst_symbol=?
        """
        params: list = [snapshot_id, prefix]
    else:
        prefix = f"{file_path}::"
        query = """
            SELECT src_symbol, dst_symbol, edge_type, confidence, confidence_score, evidence_lines
            FROM symbol_graph_edges
            WHERE snapshot_id=? AND (dst_symbol LIKE ? ESCAPE '\\' OR dst_symbol=?)
        """
        escaped = prefix.replace("%", "\\%").replace("_", "\\_") + "%"
        params = [snapshot_id, escaped, prefix.rstrip("::")]

    if high_confidence_only:
        query += " AND confidence_score >= 0.7"

    if min_confidence is not None:
        query += " AND confidence_score >= ?"
        params.append(min_confidence)

    async with db.execute(query, tuple(params)) as cur:
        rows = await cur.fetchall()

    return [
        SymbolHop(
            src_symbol=r["src_symbol"],
            dst_symbol=r["dst_symbol"],
            edge_type=r["edge_type"],
            confidence=r["confidence"],
            evidence_lines=json.loads(r["evidence_lines"] or "[]"),
            confidence_score=r["confidence_score"],
        )
        for r in rows
    ]


async def trace_call_chain(
    snapshot_id: str,
    start_file: str,
    direction: str,  # "forward" | "backward"
    max_hops: int = 4,
    high_confidence_only: bool = True,
    min_confidence: float | None = None,
) -> list[TraceStep]:
    """Multi-hop call chain traversal starting from start_file.

    Each hop discovers new files reachable via symbol-level call edges.
    Returns ordered list of TraceStep, one per file discovered.
    Stops when max_hops reached, no new files, or > 30 total files visited.

    Args:
        min_confidence: Optional numeric threshold (0.0-1.0); when set, filters to
                        confidence_score >= min_confidence. Orthogonal to high_confidence_only.
    """
    visited: set[str] = {start_file}
    steps: list[TraceStep] = []
    frontier: set[str] = {start_file}

    # Step 0: seed
    steps.append(TraceStep(step=0, file=start_file, symbols_involved=[], hops=[], is_seed=True))

    for hop_num in range(1, max_hops + 1):
        if len(visited) > 30:
            break
        next_frontier: set[str] = set()
        all_hops: list[SymbolHop] = []

        for current_file in frontier:
            if direction == "forward":
                hops = await get_callees_of(snapshot_id, current_file, high_confidence_only=high_confidence_only, min_confidence=min_confidence)
            else:
                hops = await get_callers_of(snapshot_id, current_file, high_confidence_only=high_confidence_only, min_confidence=min_confidence)

            all_hops.extend(hops)
            for hop in hops:
                target = hop.dst_file if direction == "forward" else hop.src_file
                if target and target != current_file and target not in visited:
                    next_frontier.add(target)

        if not next_frontier:
            break

        for new_file in sorted(next_frontier):
            visited.add(new_file)
            file_hops = [
                h
                for h in all_hops
                if (direction == "forward" and h.dst_file == new_file) or (direction == "backward" and h.src_file == new_file)
            ]
            symbols = list({h.dst_symbol if direction == "forward" else h.src_symbol for h in file_hops})
            # Path confidence: multiply parent step's confidence by the max confidence among parallel hops
            # that discover this file (one strong edge suffices — OR relationship), multiply across step depth
            # (a chain requires every hop to hold — AND relationship).
            parent_step = steps[-1]  # Last step added is the parent frontier
            max_hop_confidence = max((h.confidence_score for h in file_hops), default=1.0)
            new_path_confidence = parent_step.path_confidence * max_hop_confidence
            steps.append(
                TraceStep(
                    step=hop_num,
                    file=new_file,
                    symbols_involved=symbols[:5],
                    hops=file_hops[:10],
                    path_confidence=new_path_confidence,
                )
            )

        frontier = next_frontier

    return steps


def _get_impact_cone_from_graph(seed_path: str, graph_data: dict, max_hops: int) -> ImpactResult:
    """BFS using pre-built reverse adjacency dict from graph_data edges."""
    # Build reverse adjacency: tgt -> [src] (who imports me)
    rev_adj: dict[str, list[str]] = {}
    for edge in graph_data.get("edges", []):
        if edge.get("external"):
            continue
        tgt = edge.get("dst") or edge.get("tgt")
        src = edge.get("src")
        if src and tgt:
            rev_adj.setdefault(tgt, []).append(src)

    visited: set[str] = set()
    queue: list[str] = [seed_path]
    hop_map: dict[str, int] = {}

    for hop_num in range(1, max_hops + 1):
        if not queue:
            break
        next_queue: list[str] = []
        for node in queue:
            for caller in rev_adj.get(node, []):
                if caller not in visited:
                    visited.add(caller)
                    hop_map[caller] = hop_num
                    next_queue.append(caller)
        queue = next_queue

    return ImpactResult(seed_file=seed_path, impacted_files=list(visited), hop_map=hop_map)


async def get_impact_cone(
    snapshot_id: str,
    file_path: str,
    hops: int = 3,
    graph_data: dict | None = None,
) -> ImpactResult:
    """File-level reverse transitive closure: all files that depend on file_path.

    Uses structural_graph_edges (file-level import graph). A→B means A imports B.
    We want all A that directly or transitively import file_path (reverse direction).

    Args:
        graph_data: Optional pre-loaded graph.json dict. If provided, uses in-memory
                    adjacency lookup instead of DB queries (faster for repeated calls).
                    Build from graph_json.read_graph_json(snapshot_id, data_dir).
                    Falls back to DB queries if None or if graph_data lacks edge data.

    Implementation: load all internal edges, swap src↔dst to build a reversed edge
    set, then delegate BFS to expand_neighbors (C++ native if available, Python
    fallback otherwise). This is the same engine used by two-stage retrieval.
    """
    if graph_data and graph_data.get("edges"):
        return _get_impact_cone_from_graph(file_path, graph_data, hops)

    db = get_db()
    async with db.execute(
        """SELECT src_path, dst_path, edge_type, is_external
           FROM structural_graph_edges
           WHERE snapshot_id=? AND is_external=0""",
        (snapshot_id,),
    ) as cur:
        rows = await cur.fetchall()

    if not rows:
        return ImpactResult(seed_file=file_path, impacted_files=[], hop_map={})

    # Swap src↔dst so that expand_neighbors traverses in the reverse import direction:
    # original: A→B means "A imports B"; reversed: B→A means "B is imported by A"
    # expand_neighbors from seed=file_path on reversed edges discovers all files
    # that (directly or transitively) import file_path.
    reversed_edge_inputs = [(r["dst_path"], r["src_path"], r["edge_type"], int(r["is_external"])) for r in rows]

    clamped_hops = min(max(hops, 1), 4)
    native_graph = _load_native_graph()
    if native_graph and hasattr(native_graph, "expand_neighbors"):
        expanded = native_graph.expand_neighbors(file_path, reversed_edge_inputs, clamped_hops, 500)
    else:
        logger.debug("[graph_queries] native expand_neighbors unavailable; using Python fallback")
        expanded = _expand_neighbors_python(file_path, reversed_edge_inputs, clamped_hops, 500)

    discovered = [n for n in expanded.get("nodes", []) if n != file_path]

    # Reconstruct hop_map via BFS over the returned edges (expand_neighbors does
    # not return per-node depths, so we do one Python BFS pass over the edge list).
    ret_edges = expanded.get("edges", [])  # list of (src, dst, edge_type)
    adj: dict[str, list[str]] = {}
    for src, dst, *_ in ret_edges:
        adj.setdefault(src, []).append(dst)

    hop_map: dict[str, int] = {}
    bfs_visited: set[str] = {file_path}
    frontier_bfs: list[str] = [file_path]
    for hop_num in range(1, clamped_hops + 1):
        next_bfs: list[str] = []
        for node in frontier_bfs:
            for neighbor in adj.get(node, []):
                if neighbor not in bfs_visited:
                    bfs_visited.add(neighbor)
                    hop_map[neighbor] = hop_num
                    next_bfs.append(neighbor)
        frontier_bfs = next_bfs
        if not frontier_bfs:
            break

    return ImpactResult(seed_file=file_path, impacted_files=discovered, hop_map=hop_map)
