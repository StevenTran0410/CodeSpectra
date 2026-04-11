"""Structural graph service (RPA-033)."""
from __future__ import annotations

import asyncio
import ast
import importlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from infrastructure.db.database import get_db
from shared.errors import NotFoundError
from shared.logger import logger
from shared.sql_queries import SQL_SELECT_MANIFEST_FILES_BY_SNAPSHOT
from shared.toolchain import detect_cpp_toolchain
from shared.utils import new_id, read_utf8_lenient, utc_now_iso

from .types import (
    BuildGraphRequest,
    BuildGraphResponse,
    CommunityInfo,
    CyclesResponse,
    GraphCommunitiesResponse,
    GraphEdge,
    GraphEdgesResponse,
    GraphNeighborsResponse,
    GraphNodeScore,
    NodeCommunityResponse,
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


def _is_init_file(path: str) -> bool:
    """Return True for Python __init__.py files.

    These are namespace-package markers that carry no structural information —
    they're either empty or just re-export symbols from sub-modules.  Including
    them as graph nodes adds noise: they generate singleton communities because
    they have near-zero import edges.
    """
    return path == "__init__.py" or path.endswith("/__init__.py")


def _build_py_suffix_index(file_paths: set[str] | list[str]) -> dict[str, str]:
    """Build a suffix-lookup table for Python files.

    Enables resolving absolute imports like "domain.x" → "backend/domain/x.py"
    regardless of the source-root prefix used in the project layout.
    First-seen suffix wins (insertion order = iteration order of the input).
    """
    index: dict[str, str] = {}
    for f in file_paths:
        if f.endswith(".py"):
            parts = f.split("/")
            for i in range(len(parts)):
                suffix = "/".join(parts[i:])
                if suffix not in index:
                    index[suffix] = f
    return index


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
    # os.path.normpath collapses '..' segments that Path.as_posix() leaves raw.
    # Without this, "screens/analysis/../../store/foo" never matches file_set.
    candidate = _normalize(os.path.normpath(str(base / target)))

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
        test_file_set = {r["rel_path"] for r in files if r["category"] == "test"}

        py_suffix_index = _build_py_suffix_index(file_set)

        native_graph = _load_native_graph()
        entrypoints: list[str] = []
        total_edges = 0
        external_edges = 0
        edge_inputs: list[tuple[str, str, str, int]] = []
        _edge_rows: list[tuple] = []
        now = utc_now_iso()

        for r in files:
            rel_path = r["rel_path"]
            language = r["language"]
            category = r["category"]
            if category not in {"source", "infra"}:
                # Test files are excluded from the structural graph.  They import
                # everything they test, creating noisy cross-community edges that
                # distort Louvain clustering.  Test coverage is tracked separately
                # via manifest_files (category='test') and surfaced in the export.
                continue
            # __init__.py files are namespace markers with no structural content.
            # Excluding them prevents singleton communities for every package dir.
            if _is_init_file(rel_path):
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
                    # Use suffix index so imports resolve regardless of source-root prefix.
                    py_guess = _normalize(imp.replace(".", "/") + ".py")
                    resolved_path = py_suffix_index.get(py_guess)
                    if not resolved_path:
                        # 'from domain.pkg import X' targets the package dir, not a .py file.
                        # Fall back to the package's __init__.py — but we exclude __init__.py
                        # from the graph (they are namespace markers with no structural content),
                        # so this fallback is intentionally left as external/unresolved.
                        pass
                    if resolved_path and not _is_init_file(resolved_path):
                        dst_path = resolved_path
                        is_external = False

                total_edges += 1
                if is_external:
                    external_edges += 1
                dst_store = dst_path if dst_path else imp
                edge_inputs.append((rel_path, dst_store, "import", int(is_external)))
                _edge_rows.append(
                    (req.snapshot_id, rel_path, dst_store, "import", int(is_external), now)
                )

        # Batch-insert all edges in one round-trip instead of one await per edge.
        await db.executemany(
            """
            INSERT INTO structural_graph_edges
            (snapshot_id, src_path, dst_path, edge_type, is_external, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            _edge_rows,
        )

        all_nodes = {
            f for f in file_set
            if not _is_init_file(f) and f not in test_file_set
        }
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

        # Fire community detection in the background — does not block build response.
        # WAL mode allows concurrent writes; failure is logged but non-fatal.
        async def _background_communities() -> None:
            try:
                await self.detect_communities(req.snapshot_id)
                logger.info("[structural_graph] community detection done for %s", req.snapshot_id)
            except Exception as exc:
                logger.warning("[structural_graph] community detection failed: %s", exc)

        asyncio.create_task(_background_communities())

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

    async def edges(
        self, snapshot_id: str, limit: int = 2000, internal_only: bool = False
    ) -> GraphEdgesResponse:
        where = "WHERE snapshot_id=?" + (" AND is_external=0" if internal_only else "")
        async with get_db().execute(
            f"SELECT snapshot_id, src_path, dst_path, edge_type, is_external"
            f" FROM structural_graph_edges {where}"
            f" ORDER BY src_path ASC, dst_path ASC LIMIT ?",
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

    # ── CS-102: community detection ───────────────────────────────────────────

    async def detect_communities(
        self, snapshot_id: str, resolution: float = 1.0
    ) -> GraphCommunitiesResponse:
        """Run Louvain community detection and persist results.

        Uses C++ native compute_louvain if available, falls back to Python.
        Safe to call concurrently with reads (WAL mode).
        """
        db = get_db()

        # Load ALL edges — we re-resolve "external" Python absolute imports below,
        # which fixes existing graph data built before the suffix-index fix.
        async with db.execute(
            "SELECT src_path, dst_path, is_external FROM structural_graph_edges WHERE snapshot_id=?",
            (snapshot_id,),
        ) as cur:
            edge_rows = await cur.fetchall()

        # Load all manifest nodes (ensures isolated nodes are included)
        async with db.execute(
            "SELECT rel_path FROM manifest_files WHERE snapshot_id=? AND category IN ('source','infra')",
            (snapshot_id,),
        ) as cur:
            node_rows = await cur.fetchall()

        # dict.fromkeys preserves insertion order while deduplicating.
        # manifest_files may have duplicate rows (pre-migration-20 DBs); passing
        # duplicates to Louvain inflates iteration count and can skew convergence.
        # __init__.py files and test files are excluded:
        #   - __init__.py: namespace markers with no edges → singleton communities
        #   - test files: import everything they test → noisy cross-community edges
        node_ids = list(dict.fromkeys(
            r["rel_path"] for r in node_rows if not _is_init_file(r["rel_path"])
        ))
        node_id_set = set(node_ids)

        # Suffix index — resolve Python absolute imports stored as unresolved externals.
        # E.g. edge dst_path "domain.structural_graph.service" → "backend/domain/structural_graph/service.py"
        py_suffix_index = _build_py_suffix_index(node_ids)

        # Build de-duplicated edge list; attempt to re-resolve external Python imports.
        edge_set: set[tuple[str, str]] = set()
        edge_tuples: list[tuple[str, str, float]] = []

        for r in edge_rows:
            src, dst = r["src_path"], r["dst_path"]
            if src not in node_id_set or src == dst:
                continue
            if not r["is_external"]:
                # Already-resolved internal edge
                if dst in node_id_set and (src, dst) not in edge_set:
                    edge_tuples.append((src, dst, 1.0))
                    edge_set.add((src, dst))
            elif src.endswith(".py") and "/" not in dst and not dst.startswith("."):
                # Unresolved Python absolute import stored pre-suffix-fix — re-resolve now.
                # Also handles package imports: "domain.pkg" → "backend/domain/pkg/__init__.py".
                py_guess = dst.replace(".", "/") + ".py"
                resolved = py_suffix_index.get(py_guess)
                if not resolved:
                    pkg_init = dst.replace(".", "/") + "/__init__.py"
                    resolved = py_suffix_index.get(pkg_init)
                if resolved and resolved != src and (src, resolved) not in edge_set:
                    edge_tuples.append((src, resolved, 1.0))
                    edge_set.add((src, resolved))
            elif dst.startswith("."):
                # Unresolved relative TS/JS import stored pre-normpath-fix — re-resolve now.
                # dst holds the raw import string (e.g. "../../store/local-repo.store").
                resolved = _resolve_relative_import(src, dst, node_id_set)
                if resolved and resolved != src and (src, resolved) not in edge_set:
                    edge_tuples.append((src, resolved, 1.0))
                    edge_set.add((src, resolved))

        # Run Louvain — C++ native first, Python fallback
        native_graph = _load_native_graph()
        if native_graph and hasattr(native_graph, "compute_louvain"):
            try:
                raw: dict[str, int] = native_graph.compute_louvain(
                    edge_tuples, node_ids, resolution, 42
                )
            except Exception as e:
                logger.debug("[structural_graph] native compute_louvain failed: %s", e)
                raw = None
        else:
            raw = None

        if raw is None:
            from ._louvain_fallback import compute_louvain_python
            raw = compute_louvain_python(edge_tuples, node_ids, resolution)

        # ── Singleton absorption ──────────────────────────────────────────────
        # Louvain often isolates weakly-connected nodes (e.g. __init__.py files,
        # config files) into singleton communities.  For each singleton, count
        # "votes" from its internal neighbours and reassign it to the majority
        # community.  We work from a snapshot of `raw` so absorbed nodes don't
        # influence each other.  Nodes with no internal neighbours are left as
        # true singletons and flagged later.
        _comm_size: dict[int, int] = defaultdict(int)
        for _cid in raw.values():
            _comm_size[_cid] += 1

        _node_nbrs: dict[str, list[str]] = defaultdict(list)
        for _s, _d, _w in edge_tuples:
            _node_nbrs[_s].append(_d)
            _node_nbrs[_d].append(_s)

        _raw_snap = dict(raw)
        for _node, _cid in _raw_snap.items():
            if _comm_size[_cid] != 1:
                continue
            _votes: dict[int, int] = defaultdict(int)
            for _nbr in _node_nbrs[_node]:
                _nbr_cid = _raw_snap.get(_nbr, -1)
                if _nbr_cid >= 0 and _nbr_cid != _cid:
                    _votes[_nbr_cid] += 1
            if _votes:
                raw[_node] = max(_votes, key=_votes.__getitem__)

        # Compute hub scores (intra-community in-degree) and inter-community edges.
        comm_adj: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        # inter_comm_neighbors[cid] = set of community IDs that cid shares an edge with.
        inter_comm_neighbors: dict[int, set[int]] = defaultdict(set)

        for s, d, w in edge_tuples:
            cs, cd = raw.get(s, -1), raw.get(d, -1)
            if cs < 0 or cd < 0:
                continue
            if cs == cd:
                # Intra-community: count in-degree for hub scoring
                comm_adj[cs][d] += w
            else:
                # Inter-community: record adjacency (undirected)
                inter_comm_neighbors[cs].add(cd)
                inter_comm_neighbors[cd].add(cs)

        # Aggregate: member_count, hub_paths, hub_score per node
        comm_members: dict[int, list[str]] = defaultdict(list)
        for node, cid in raw.items():
            comm_members[cid].append(node)

        now = utc_now_iso()

        # Replace old community rows for this snapshot
        await db.execute("DELETE FROM graph_community_members WHERE snapshot_id=?", (snapshot_id,))
        await db.execute("DELETE FROM graph_community_summaries WHERE snapshot_id=?", (snapshot_id,))

        communities: list[CommunityInfo] = []
        _member_rows: list[tuple] = []
        _summary_rows: list[tuple] = []

        for cid, members in sorted(comm_members.items()):
            hub_scores = {n: comm_adj[cid].get(n, 0.0) for n in members}
            top_hubs = sorted(members, key=lambda n: -hub_scores[n])[:3]
            neighbor_ids = sorted(inter_comm_neighbors.get(cid, set()))
            mod_contrib = float(len(members)) / max(len(node_ids), 1)

            for node in members:
                _member_rows.append((snapshot_id, node, cid, hub_scores[node], now))

            _summary_rows.append((
                snapshot_id, cid, len(members), json.dumps(top_hubs), mod_contrib,
                json.dumps(neighbor_ids), now,
            ))

            communities.append(CommunityInfo(
                community_id=cid,
                member_count=len(members),
                hub_paths=top_hubs,
                modularity_contribution=mod_contrib,
                neighbor_community_ids=neighbor_ids,
                is_singleton=len(members) == 1,
                llm_summary=None,
                generated_at=now,
            ))

        # Batch-insert members and summaries in two round-trips.
        await db.executemany(
            """
            INSERT OR REPLACE INTO graph_community_members
            (snapshot_id, node_path, community_id, hub_score, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            _member_rows,
        )
        await db.executemany(
            """
            INSERT OR REPLACE INTO graph_community_summaries
            (snapshot_id, community_id, member_count, hub_paths, modularity_contribution,
             neighbor_community_ids, llm_summary, generated_at)
            VALUES (?, ?, ?, ?, ?, ?, NULL, ?)
            """,
            _summary_rows,
        )

        await db.commit()

        return GraphCommunitiesResponse(
            snapshot_id=snapshot_id,
            total_communities=len(communities),
            communities=communities,
            node_index=raw,
        )

    async def list_communities(self, snapshot_id: str) -> GraphCommunitiesResponse:
        """Return cached community data from DB (no recomputation)."""
        db = get_db()

        async with db.execute(
            """
            SELECT community_id, member_count, hub_paths, modularity_contribution,
                   neighbor_community_ids, llm_summary, generated_at
            FROM graph_community_summaries WHERE snapshot_id=?
            ORDER BY community_id ASC
            """,
            (snapshot_id,),
        ) as cur:
            rows = await cur.fetchall()

        async with db.execute(
            "SELECT node_path, community_id FROM graph_community_members WHERE snapshot_id=?",
            (snapshot_id,),
        ) as cur:
            member_rows = await cur.fetchall()

        node_index = {r["node_path"]: r["community_id"] for r in member_rows}

        communities = [
            CommunityInfo(
                community_id=r["community_id"],
                member_count=r["member_count"],
                hub_paths=json.loads(r["hub_paths"] or "[]"),
                modularity_contribution=float(r["modularity_contribution"]),
                neighbor_community_ids=json.loads(r["neighbor_community_ids"] or "[]"),
                is_singleton=r["member_count"] == 1,
                llm_summary=r["llm_summary"],
                generated_at=r["generated_at"],
            )
            for r in rows
        ]

        return GraphCommunitiesResponse(
            snapshot_id=snapshot_id,
            total_communities=len(communities),
            communities=communities,
            node_index=node_index,
        )

    async def community_for_node(
        self, snapshot_id: str, rel_path: str
    ) -> NodeCommunityResponse:
        """Return community ID and all members for a given node."""
        db = get_db()

        async with db.execute(
            "SELECT community_id FROM graph_community_members WHERE snapshot_id=? AND node_path=?",
            (snapshot_id, rel_path),
        ) as cur:
            row = await cur.fetchone()

        if row is None:
            raise ValueError(f"Node '{rel_path}' not found in community index for snapshot {snapshot_id}")

        cid = row["community_id"]

        async with db.execute(
            "SELECT node_path FROM graph_community_members WHERE snapshot_id=? AND community_id=? ORDER BY node_path",
            (snapshot_id, cid),
        ) as cur:
            member_rows = await cur.fetchall()

        return NodeCommunityResponse(
            snapshot_id=snapshot_id,
            node_path=rel_path,
            community_id=cid,
            members=[r["node_path"] for r in member_rows],
        )

    async def cycles(self, snapshot_id: str) -> CyclesResponse:
        """Return circular import cycles (SCCs) via C++ native or Python fallback."""
        db = get_db()

        async with db.execute(
            "SELECT src_path, dst_path FROM structural_graph_edges WHERE snapshot_id=? AND is_external=0",
            (snapshot_id,),
        ) as cur:
            rows = await cur.fetchall()

        edge_tuples = [(r["src_path"], r["dst_path"]) for r in rows]

        native_graph = _load_native_graph()
        if native_graph and hasattr(native_graph, "compute_scc"):
            try:
                sccs = native_graph.compute_scc(edge_tuples)
            except Exception as e:
                logger.debug("[structural_graph] native compute_scc failed: %s", e)
                sccs = None
        else:
            sccs = None

        if sccs is None:
            from ._scc_fallback import compute_scc_python
            sccs = compute_scc_python(edge_tuples)

        return CyclesResponse(snapshot_id=snapshot_id, cycles=sccs)

    async def export_graph_json(self, snapshot_id: str) -> dict:
        """Export full graph structure as a single JSON-serialisable dict.

        Includes nodes, all edges (internal + external), community assignments,
        per-community member lists, and circular import cycles.
        Designed for copy-paste debugging: share the output to diagnose clustering.
        """
        db = get_db()

        async with db.execute(
            "SELECT src_path, dst_path, is_external FROM structural_graph_edges WHERE snapshot_id=? ORDER BY src_path, dst_path",
            (snapshot_id,),
        ) as cur:
            edge_rows = await cur.fetchall()

        async with db.execute(
            "SELECT node_path, community_id FROM graph_community_members WHERE snapshot_id=? ORDER BY community_id, node_path",
            (snapshot_id,),
        ) as cur:
            member_rows = await cur.fetchall()

        async with db.execute(
            "SELECT rel_path FROM manifest_files WHERE snapshot_id=? AND category IN ('source','infra') ORDER BY rel_path",
            (snapshot_id,),
        ) as cur:
            node_rows = await cur.fetchall()

        async with db.execute(
            "SELECT rel_path FROM manifest_files WHERE snapshot_id=? AND category='test' ORDER BY rel_path",
            (snapshot_id,),
        ) as cur:
            test_rows = await cur.fetchall()

        cycles_resp = await self.cycles(snapshot_id)

        # Test files are tracked separately in the manifest (category='test') and
        # surfaced as `test_files` metadata in the export.  They are excluded from
        # the structural graph because their import edges (test→source) distort
        # community detection without adding architectural information.
        test_files = list(dict.fromkeys(r["rel_path"] for r in test_rows))

        # Deduplicate: pre-migration-20 DBs may have 2× rows per file/edge.
        # __init__.py files are excluded from the graph (namespace markers, no edges).
        nodes = list(dict.fromkeys(
            r["rel_path"] for r in node_rows if not _is_init_file(r["rel_path"])
        ))
        node_set = set(nodes)
        seen_edges: set[tuple[str, str, bool]] = set()
        edges = []
        for r in edge_rows:
            if _is_init_file(r["src_path"]) or _is_init_file(r["dst_path"]):
                continue
            # Drop edges involving nodes excluded from the graph (test files, etc.)
            if r["src_path"] not in node_set and not bool(r["is_external"]):
                continue
            key = (r["src_path"], r["dst_path"], bool(r["is_external"]))
            if key not in seen_edges:
                seen_edges.add(key)
                edges.append({"src": r["src_path"], "dst": r["dst_path"], "external": key[2]})

        communities: dict[str, int] = {}
        community_groups: dict[str, list[str]] = {}
        for r in member_rows:
            path, cid = r["node_path"], str(r["community_id"])
            communities[path] = r["community_id"]
            community_groups.setdefault(cid, []).append(path)

        internal_count = sum(1 for e in edges if not e["external"])

        # Compute community-level adjacency: which communities are connected by import edges.
        # This lets tools and humans see "Community A uses Community B" at a glance,
        # without having to manually trace individual file edges.
        inter_comm_counts: dict[tuple[int, int], int] = defaultdict(int)
        for e in edges:
            if e["external"]:
                continue
            src_cid = communities.get(e["src"])
            dst_cid = communities.get(e["dst"])
            if src_cid is None or dst_cid is None or src_cid == dst_cid:
                continue
            # Directed: src_community imports dst_community
            inter_comm_counts[(src_cid, dst_cid)] += 1

        community_edges = [
            {"src_community": src, "dst_community": dst, "edge_count": cnt}
            for (src, dst), cnt in sorted(
                inter_comm_counts.items(), key=lambda kv: -kv[1]
            )
        ]

        singleton_community_count = sum(
            1 for members in community_groups.values() if len(members) == 1
        )

        return {
            "snapshot_id": snapshot_id,
            "exported_at": utc_now_iso(),
            "stats": {
                "node_count": len(nodes),
                "total_edge_count": len(edges),
                "internal_edge_count": internal_count,
                "community_count": len(community_groups),
                "singleton_community_count": singleton_community_count,
                "cycle_count": len(cycles_resp.cycles),
                "test_file_count": len(test_files),
            },
            "nodes": nodes,
            "edges": edges,
            "communities": communities,
            "community_groups": community_groups,
            "community_edges": community_edges,
            "cycles": cycles_resp.cycles,
            # Test files are tracked here for agent context (so agents know tests exist)
            # but are NOT part of the structural graph — they distort community detection.
            "test_files": test_files,
        }
