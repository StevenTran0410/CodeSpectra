"""Structural graph service (RPA-033)."""
from __future__ import annotations

import ast
import importlib
import json
import re
from pathlib import Path

from infrastructure.db.database import get_db
from shared.errors import NotFoundError
from shared.logger import logger
from shared.sql_queries import SQL_SELECT_MANIFEST_FILES_BY_SNAPSHOT
from shared.toolchain import detect_cpp_toolchain
from shared.utils import read_utf8_lenient, utc_now_iso

from .types import (
    BuildGraphRequest,
    BuildGraphResponse,
    GraphEdge,
    GraphEdgesResponse,
    GraphNeighborsResponse,
    GraphNodeScore,
    StructuralGraphSummary,
)

_RE_TS_IMPORT = re.compile(
    r"""(?m)^\s*import\s+(?:[^'"]+?\s+from\s+)?['"]([^'"]+)['"]\s*;?"""
)
_RE_TS_REQUIRE = re.compile(r"""(?m)require\(\s*['"]([^'"]+)['"]\s*\)""")


def _load_native_graph():
    try:
        return importlib.import_module("domain.structural_graph._native_graph")
    except Exception:  # pragma: no cover - platform/build dependent
        return None


def _compute_scores_python(
    nodes: list[str], edge_inputs: list[tuple[str, str, str, int]]
) -> list[dict[str, int | str]]:
    indeg: dict[str, int] = {n: 0 for n in nodes}
    outdeg: dict[str, int] = {n: 0 for n in nodes}
    for src, dst, _edge_type, _is_external in edge_inputs:
        outdeg[src] = outdeg.get(src, 0) + 1
        indeg[dst] = indeg.get(dst, 0) + 1

    seen = set(indeg.keys()) | set(outdeg.keys())
    items: list[dict[str, int | str]] = []
    for rel_path in seen:
        indegree = indeg.get(rel_path, 0)
        outdegree = outdeg.get(rel_path, 0)
        items.append(
            {
                "rel_path": rel_path,
                "indegree": indegree,
                "outdegree": outdegree,
                "score": (indegree * 3) + outdegree,
            }
        )
    items.sort(
        key=lambda x: (-int(x["score"]), -int(x["indegree"]), str(x["rel_path"]))
    )
    return items


def _expand_neighbors_python(
    seed: str, edge_inputs: list[tuple[str, str, str, int]], hops: int, limit: int
) -> dict[str, list]:
    if hops < 1:
        hops = 1
    if hops > 4:
        hops = 4
    if limit < 10:
        limit = 10
    if limit > 2000:
        limit = 2000

    edges = [e for e in edge_inputs if int(e[3]) == 0]
    adjacency: dict[str, list[int]] = {}
    for i, e in enumerate(edges):
        src = str(e[0])
        dst = str(e[1])
        adjacency.setdefault(src, []).append(i)
        adjacency.setdefault(dst, []).append(i)

    visited: set[str] = {seed}
    kept_edge_indexes: set[int] = set()
    frontier: set[str] = {seed}

    for _ in range(hops):
        next_frontier: set[str] = set()
        for node in frontier:
            for edge_idx in adjacency.get(node, []):
                if len(kept_edge_indexes) < limit:
                    kept_edge_indexes.add(edge_idx)
                src, dst, _edge_type, _is_external = edges[edge_idx]
                nxt = str(dst) if str(src) == node else str(src)
                if nxt not in visited and len(visited) < limit:
                    visited.add(nxt)
                    next_frontier.add(nxt)
        if not next_frontier:
            break
        frontier = next_frontier

    nodes = sorted(visited)
    out_edges: list[tuple[str, str, str]] = []
    for idx in sorted(kept_edge_indexes):
        src, dst, edge_type, _is_external = edges[idx]
        out_edges.append((str(src), str(dst), str(edge_type)))
    return {"nodes": nodes, "edges": out_edges}


def _normalize(path: str) -> str:
    return path.replace("\\", "/")


def _is_entrypoint(rel_path: str) -> bool:
    low = rel_path.lower()
    name = Path(low).name
    if name in {"main.py", "__main__.py", "app.py", "server.py", "cli.py", "index.ts", "index.tsx"}:
        return True
    if "/bin/" in low or "/cmd/" in low:
        return True
    if low.endswith("/manage.py"):
        return True
    if "/routes/" in low and ("index." in name or "router" in name):
        return True
    return False


def _resolve_relative_import(src_rel_path: str, target: str, candidates: set[str]) -> str | None:
    src = Path(src_rel_path)
    base = src.parent
    candidate = _normalize(str((base / target).as_posix()))

    options = [
        candidate,
        f"{candidate}.py",
        f"{candidate}.ts",
        f"{candidate}.tsx",
        f"{candidate}.js",
        f"{candidate}.jsx",
        _normalize(str(Path(candidate) / "index.py")),
        _normalize(str(Path(candidate) / "index.ts")),
        _normalize(str(Path(candidate) / "index.tsx")),
        _normalize(str(Path(candidate) / "index.js")),
    ]
    for opt in options:
        if opt in candidates:
            return opt
    return None


def _extract_python_imports(content: str) -> list[str]:
    out: list[str] = []
    try:
        tree = ast.parse(content)
    except Exception:
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.append(node.module)
    return out


def _extract_ts_js_imports(content: str) -> list[str]:
    out = [m.group(1) for m in _RE_TS_IMPORT.finditer(content)]
    out.extend(m.group(1) for m in _RE_TS_REQUIRE.finditer(content))
    return out


class StructuralGraphService:
    async def build(self, req: BuildGraphRequest) -> BuildGraphResponse:
        db = get_db()
        async with db.execute("SELECT * FROM repo_snapshots WHERE id=?", (req.snapshot_id,)) as cur:
            snap = await cur.fetchone()
        if snap is None:
            raise NotFoundError("RepoSnapshot", req.snapshot_id)

        root = Path(snap["local_path"])
        if not root.exists():
            raise ValueError("Snapshot path does not exist")

        if req.force_rebuild:
            await db.execute("DELETE FROM structural_graph_edges WHERE snapshot_id=?", (req.snapshot_id,))
            await db.execute("DELETE FROM structural_graph_summaries WHERE snapshot_id=?", (req.snapshot_id,))
        else:
            async with db.execute(
                "SELECT 1 FROM structural_graph_summaries WHERE snapshot_id=? LIMIT 1",
                (req.snapshot_id,),
            ) as cur:
                exists = await cur.fetchone()
            if exists:
                logger.info(
                    "[structural_graph] snapshot %s already built, skipping", req.snapshot_id
                )
                return BuildGraphResponse(summary=await self.summary(req.snapshot_id))

        async with db.execute(SQL_SELECT_MANIFEST_FILES_BY_SNAPSHOT, (req.snapshot_id,)) as cur:
            files = await cur.fetchall()

        file_set = {r["rel_path"] for r in files}
        native_graph = _load_native_graph()
        entrypoints: list[str] = []
        total_edges = 0
        external_edges = 0
        edge_inputs: list[tuple[str, str, str, int]] = []
        now = utc_now_iso()

        for r in files:
            rel_path = r["rel_path"]
            language = r["language"]
            category = r["category"]
            if category not in {"source", "test", "infra"}:
                continue
            if _is_entrypoint(rel_path):
                entrypoints.append(rel_path)
            src = (root / rel_path).resolve()
            if not src.exists() or not src.is_file():
                continue

            content = read_utf8_lenient(src)
            if not content:
                continue

            imports: list[str] = []
            if language == "python":
                imports = _extract_python_imports(content)
            elif language in {"typescript", "javascript"}:
                imports = _extract_ts_js_imports(content)

            if not imports:
                continue

            for imp in imports:
                dst_path: str | None = None
                is_external = True
                if imp.startswith("."):
                    dst_path = _resolve_relative_import(rel_path, imp, file_set)
                    is_external = dst_path is None
                else:
                    # Python module-like imports may map to local path.
                    py_guess = _normalize(imp.replace(".", "/") + ".py")
                    if py_guess in file_set:
                        dst_path = py_guess
                        is_external = False

                total_edges += 1
                if is_external:
                    external_edges += 1
                dst_store = dst_path if dst_path else imp
                edge_inputs.append((rel_path, dst_store, "import", int(is_external)))
                await db.execute(
                    """
                    INSERT INTO structural_graph_edges
                    (snapshot_id, src_path, dst_path, edge_type, is_external, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (req.snapshot_id, rel_path, dst_store, "import", int(is_external), now),
                )

        all_nodes = set(file_set)
        if native_graph and hasattr(native_graph, "compute_scores"):
            scored_raw = native_graph.compute_scores(sorted(all_nodes), edge_inputs)
        else:
            logger.info(
                "[structural_graph] native compute_scores unavailable; using Python fallback"
            )
            scored_raw = _compute_scores_python(sorted(all_nodes), edge_inputs)
        scored: list[GraphNodeScore] = [GraphNodeScore(**item) for item in scored_raw]
        top_central = scored[:30]

        native_toolchain = detect_cpp_toolchain()
        await db.execute(
            """
            INSERT INTO structural_graph_summaries
            (snapshot_id, total_nodes, total_edges, external_edges, entrypoints, top_central_files, generated_at, native_toolchain)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                req.snapshot_id,
                len(all_nodes),
                total_edges,
                external_edges,
                json.dumps(sorted(set(entrypoints))),
                json.dumps([s.model_dump() for s in top_central]),
                now,
                native_toolchain,
            ),
        )
        await db.commit()

        return BuildGraphResponse(
            summary=StructuralGraphSummary(
                snapshot_id=req.snapshot_id,
                total_nodes=len(all_nodes),
                total_edges=total_edges,
                external_edges=external_edges,
                entrypoints=sorted(set(entrypoints)),
                top_central_files=top_central,
                generated_at=now,
                native_toolchain=native_toolchain,
            )
        )

    async def summary(self, snapshot_id: str) -> StructuralGraphSummary:
        async with get_db().execute(
            "SELECT * FROM structural_graph_summaries WHERE snapshot_id=?",
            (snapshot_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            raise NotFoundError("StructuralGraphSummary", snapshot_id)
        # Always return live toolchain status — DB value is stale if module was
        # built after the graph was last computed.
        return StructuralGraphSummary(
            snapshot_id=row["snapshot_id"],
            total_nodes=row["total_nodes"],
            total_edges=row["total_edges"],
            external_edges=row["external_edges"],
            entrypoints=json.loads(row["entrypoints"] or "[]"),
            top_central_files=[GraphNodeScore(**x) for x in json.loads(row["top_central_files"] or "[]")],
            generated_at=row["generated_at"],
            native_toolchain=detect_cpp_toolchain(),
        )

    async def edges(self, snapshot_id: str, limit: int = 2000) -> GraphEdgesResponse:
        async with get_db().execute(
            """
            SELECT snapshot_id, src_path, dst_path, edge_type, is_external
            FROM structural_graph_edges
            WHERE snapshot_id=?
            ORDER BY src_path ASC, dst_path ASC
            LIMIT ?
            """,
            (snapshot_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        return GraphEdgesResponse(
            snapshot_id=snapshot_id,
            edges=[
                GraphEdge(
                    snapshot_id=r["snapshot_id"],
                    src_path=r["src_path"],
                    dst_path=r["dst_path"],
                    edge_type=r["edge_type"],
                    is_external=bool(r["is_external"]),
                )
                for r in rows
            ],
        )

    async def neighbors(
        self, snapshot_id: str, seed_path: str, hops: int = 1, limit: int = 300
    ) -> GraphNeighborsResponse:
        seed = seed_path.strip()
        if not seed:
            raise ValueError("seed path is required")
        if hops < 1:
            hops = 1
        if hops > 4:
            hops = 4
        if limit < 10:
            limit = 10
        if limit > 2000:
            limit = 2000

        native_graph = _load_native_graph()

        async with get_db().execute(
            """
            SELECT snapshot_id, src_path, dst_path, edge_type, is_external
            FROM structural_graph_edges
            WHERE snapshot_id=? AND is_external=0
            """,
            (snapshot_id,),
        ) as cur:
            rows = await cur.fetchall()

        if not rows:
            return GraphNeighborsResponse(
                snapshot_id=snapshot_id,
                seed_path=seed,
                hops=hops,
                nodes=[seed],
                edges=[],
            )

        edge_inputs = [
            (r["src_path"], r["dst_path"], r["edge_type"], int(r["is_external"]))
            for r in rows
        ]
        if native_graph and hasattr(native_graph, "expand_neighbors"):
            expanded = native_graph.expand_neighbors(seed, edge_inputs, hops, limit)
        else:
            logger.info(
                "[structural_graph] native expand_neighbors unavailable; using Python fallback"
            )
            expanded = _expand_neighbors_python(seed, edge_inputs, hops, limit)
        nodes = [str(x) for x in expanded.get("nodes", [])]
        edges: list[GraphEdge] = []
        for t in expanded.get("edges", []):
            src_path = str(t[0])
            dst_path = str(t[1])
            edge_type = str(t[2]) if len(t) > 2 else "import"
            edges.append(
                GraphEdge(
                    snapshot_id=snapshot_id,
                    src_path=src_path,
                    dst_path=dst_path,
                    edge_type=edge_type,
                    is_external=False,
                )
            )

        return GraphNeighborsResponse(
            snapshot_id=snapshot_id,
            seed_path=seed,
            hops=hops,
            nodes=nodes,
            edges=edges,
        )
