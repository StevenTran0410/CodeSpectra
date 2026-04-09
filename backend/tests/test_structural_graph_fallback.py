from __future__ import annotations

from domain.structural_graph.service import _compute_scores_python, _expand_neighbors_python


def test_compute_scores_python_ranking_and_tiebreak() -> None:
    nodes = ["a.py", "b.py", "c.py", "d.py", "e.py"]
    edges = [
        ("a.py", "b.py", "import", 0),
        ("a.py", "c.py", "import", 0),
        ("d.py", "b.py", "import", 0),
        ("e.py", "b.py", "import", 1),  # external import still counts for centrality
    ]

    scored = _compute_scores_python(nodes, edges)
    by_path = {str(x["rel_path"]): x for x in scored}

    assert int(by_path["b.py"]["indegree"]) == 3
    assert int(by_path["b.py"]["outdegree"]) == 0
    assert int(by_path["b.py"]["score"]) == 9

    assert int(by_path["a.py"]["indegree"]) == 0
    assert int(by_path["a.py"]["outdegree"]) == 2
    assert int(by_path["a.py"]["score"]) == 2

    # d.py and e.py both score 1; alphabetical rel_path tie-break applies.
    rank = [str(x["rel_path"]) for x in scored]
    assert rank.index("d.py") < rank.index("e.py")


def test_expand_neighbors_python_internal_bounded_bfs() -> None:
    edges = [
        ("src/a.py", "src/b.py", "import", 0),
        ("src/a.py", "src/c.py", "import", 0),
        ("src/b.py", "src/d.py", "import", 0),
        ("src/c.py", "src/e.py", "import", 0),
        ("src/e.py", "src/f.py", "import", 0),
        ("src/a.py", "requests", "import", 1),  # external edge should be ignored
    ]

    out = _expand_neighbors_python(seed="src/a.py", edge_inputs=edges, hops=2, limit=300)
    nodes = set(out["nodes"])
    out_edges = set(tuple(e) for e in out["edges"])

    assert "src/a.py" in nodes
    assert "src/b.py" in nodes
    assert "src/c.py" in nodes
    assert "src/d.py" in nodes
    assert "src/e.py" in nodes
    assert "src/f.py" not in nodes  # 3 hops away
    assert "requests" not in nodes  # external-only node excluded

    assert ("src/a.py", "src/b.py", "import") in out_edges
    assert ("src/a.py", "src/c.py", "import") in out_edges
    assert ("src/b.py", "src/d.py", "import") in out_edges
    assert ("src/c.py", "src/e.py", "import") in out_edges
    assert ("src/e.py", "src/f.py", "import") not in out_edges
