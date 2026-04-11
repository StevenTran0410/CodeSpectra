"""Pure-Python iterative Tarjan SCC fallback.

Accepts the same interface as the C++ native compute_scc:
  edge_tuples: list of (src, dst) string pairs
Returns: list of SCCs with >= 2 nodes, each SCC sorted, largest first.
"""
from __future__ import annotations

from collections import defaultdict


def compute_scc_python(edge_tuples: list[tuple[str, str]]) -> list[list[str]]:
    """Iterative Tarjan SCC — avoids Python recursion limits via explicit call stack."""
    graph: dict[str, set[str]] = defaultdict(set)
    all_nodes: set[str] = set()
    for s, d in edge_tuples:
        graph[s].add(d)
        all_nodes.add(s)
        all_nodes.add(d)

    index_counter = [0]
    stack: list[str] = []
    on_stack: set[str] = set()
    index: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    sccs: list[list[str]] = []

    for root in list(all_nodes):
        if root in index:
            continue
        index[root] = lowlink[root] = index_counter[0]
        index_counter[0] += 1
        stack.append(root)
        on_stack.add(root)
        call_stack: list[tuple[str, list[str], int]] = [
            (root, list(graph.get(root, set())), 0)
        ]

        while call_stack:
            v, neighbours, ei = call_stack[-1]
            if ei < len(neighbours):
                call_stack[-1] = (v, neighbours, ei + 1)
                w = neighbours[ei]
                if w not in index:
                    index[w] = lowlink[w] = index_counter[0]
                    index_counter[0] += 1
                    stack.append(w)
                    on_stack.add(w)
                    call_stack.append((w, list(graph.get(w, set())), 0))
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], index[w])
            else:
                call_stack.pop()
                if call_stack:
                    parent = call_stack[-1][0]
                    lowlink[parent] = min(lowlink[parent], lowlink[v])
                if lowlink[v] == index[v]:
                    scc: list[str] = []
                    while True:
                        w = stack.pop()
                        on_stack.discard(w)
                        scc.append(w)
                        if w == v:
                            break
                    if len(scc) >= 2:
                        sccs.append(sorted(scc))

    sccs.sort(key=lambda s: -len(s))
    return sccs
