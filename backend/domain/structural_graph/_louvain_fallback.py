"""Pure-Python Louvain community detection fallback (CS-102).

Uses networkx if available (already in pyproject.toml optional deps),
otherwise falls back to a minimal greedy modularity implementation
using only stdlib — adequate for repos up to ~1,000 internal nodes.

Public API:
    compute_louvain_python(
        edge_tuples: list[tuple[str, str, float]],   # (src, dst, weight)
        node_ids:    list[str],
        resolution:  float = 1.0,
    ) -> dict[str, int]                               # node_path -> community_id
"""
from __future__ import annotations

import random
from collections import defaultdict

from shared.logger import logger


# ── networkx-backed path ──────────────────────────────────────────────────────

def _louvain_via_networkx(
    edge_tuples: list[tuple[str, str, float]],
    node_ids: list[str],
    resolution: float,
) -> dict[str, int]:
    import networkx as nx  # type: ignore[import]
    import networkx.algorithms.community as nx_comm  # type: ignore[import]

    G = nx.Graph()
    G.add_nodes_from(node_ids)
    for src, dst, w in edge_tuples:
        if G.has_edge(src, dst):
            G[src][dst]["weight"] = G[src][dst].get("weight", 1.0) + w
        else:
            G.add_edge(src, dst, weight=w)

    communities = nx_comm.louvain_communities(G, resolution=resolution, seed=42)
    result: dict[str, int] = {}
    for cid, members in enumerate(communities):
        for node in members:
            result[node] = cid
    # assign isolated nodes (not in any edge) their own communities
    next_cid = len(communities)
    for n in node_ids:
        if n not in result:
            result[n] = next_cid
            next_cid += 1
    return result


# ── stdlib-only greedy modularity (Phase 1 Louvain) ──────────────────────────

def _louvain_stdlib(
    edge_tuples: list[tuple[str, str, float]],
    node_ids: list[str],
    resolution: float,
) -> dict[str, int]:
    """Greedy Phase-1 Louvain using only stdlib.

    Sufficient for repos up to ~1,000 internal nodes; logs a warning above that.
    """
    if len(node_ids) > 1_000:
        logger.warning(
            "[louvain] stdlib fallback running on %d nodes — install networkx for better quality",
            len(node_ids),
        )

    # Build string → int intern table for speed
    node_set = set(node_ids)
    for s, d, _ in edge_tuples:
        node_set.add(s)
        node_set.add(d)
    all_nodes = sorted(node_set)
    nid: dict[str, int] = {n: i for i, n in enumerate(all_nodes)}
    N = len(all_nodes)

    # Weighted adjacency: adj[u] = list[(v, w)]
    adj: list[list[tuple[int, float]]] = [[] for _ in range(N)]
    m2 = 0.0  # 2 * total edge weight
    for src, dst, w in edge_tuples:
        u, v = nid.get(src, -1), nid.get(dst, -1)
        if u < 0 or v < 0 or u == v:
            continue
        adj[u].append((v, w))
        adj[v].append((u, w))
        m2 += 2.0 * w

    if m2 == 0.0:
        # No edges — each node is its own community
        return {n: i for i, n in enumerate(all_nodes)}

    # Weighted degrees
    k: list[float] = [sum(w for _, w in adj[u]) for u in range(N)]

    # Initial assignment: every node in its own community
    community: list[int] = list(range(N))
    sigma_tot: list[float] = k[:]  # sum of degrees of nodes in community c

    rng = random.Random(42)
    order = list(range(N))

    for _pass in range(20):
        moved = False
        rng.shuffle(order)

        for u in order:
            old_c = community[u]

            # Weights to each neighbouring community
            w_to_comm: dict[int, float] = defaultdict(float)
            for v, w in adj[u]:
                w_to_comm[community[v]] += w

            # Remove u from its current community
            sigma_tot[old_c] -= k[u]
            community[u] = -1  # temporarily unassigned

            best_c = old_c
            best_dq = 0.0

            for c, w_in in w_to_comm.items():
                if c == old_c:
                    # Re-evaluate gain of staying
                    dq = w_in / m2 - resolution * k[u] * sigma_tot[c] / (m2 * m2)
                else:
                    dq = w_in / m2 - resolution * k[u] * sigma_tot[c] / (m2 * m2)
                if dq > best_dq:
                    best_dq = dq
                    best_c = c

            community[u] = best_c
            sigma_tot[best_c] += k[u]

            if best_c != old_c:
                moved = True

        if not moved:
            break

    # Renumber communities 0..K-1 ordered by first appearance
    remap: dict[int, int] = {}
    next_id = 0
    result: dict[str, int] = {}
    for i, n in enumerate(all_nodes):
        c = community[i]
        if c not in remap:
            remap[c] = next_id
            next_id += 1
        result[n] = remap[c]

    # Nodes in node_ids but with no edges get isolated communities
    for n in node_ids:
        if n not in result:
            result[n] = next_id
            next_id += 1

    return result


# ── public entry point ────────────────────────────────────────────────────────

def compute_louvain_python(
    edge_tuples: list[tuple[str, str, float]],
    node_ids: list[str],
    resolution: float = 1.0,
) -> dict[str, int]:
    """Louvain community detection — networkx if available, stdlib otherwise."""
    try:
        return _louvain_via_networkx(edge_tuples, node_ids, resolution)
    except ImportError:
        pass
    except Exception as e:
        logger.debug("[louvain] networkx path failed: %s — falling back to stdlib", e)

    return _louvain_stdlib(edge_tuples, node_ids, resolution)
