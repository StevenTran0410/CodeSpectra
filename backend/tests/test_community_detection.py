"""Tests for CS-102 community detection — Python fallback and C++ native path."""
from __future__ import annotations

import importlib

import pytest

from domain.structural_graph._louvain_fallback import compute_louvain_python
from domain.structural_graph._scc_fallback import compute_scc_python


# ── SCC fallback tests ────────────────────────────────────────────────────────

def test_scc_fallback_simple_cycle() -> None:
    """a→b→a forms a 2-cycle."""
    result = compute_scc_python([("a", "b"), ("b", "a")])
    assert len(result) == 1
    assert sorted(result[0]) == ["a", "b"]


def test_scc_fallback_two_independent_cycles() -> None:
    """Two separate 2-cycles → 2 SCCs."""
    edges = [("a", "b"), ("b", "a"), ("c", "d"), ("d", "c")]
    result = compute_scc_python(edges)
    assert len(result) == 2
    paths = [sorted(s) for s in result]
    assert ["a", "b"] in paths
    assert ["c", "d"] in paths


def test_scc_fallback_no_cycle() -> None:
    """Acyclic graph → no SCCs returned."""
    edges = [("a", "b"), ("b", "c"), ("c", "d")]
    result = compute_scc_python(edges)
    assert result == []


def test_scc_fallback_larger_scc() -> None:
    """3-node cycle a→b→c→a."""
    edges = [("a", "b"), ("b", "c"), ("c", "a")]
    result = compute_scc_python(edges)
    assert len(result) == 1
    assert sorted(result[0]) == ["a", "b", "c"]


def test_scc_fallback_bridge_between_two_sccs() -> None:
    """Two cliques joined by a one-way bridge; bridge should not merge SCCs."""
    edges = [
        ("a", "b"), ("b", "a"),           # SCC 1
        ("c", "d"), ("d", "c"),           # SCC 2
        ("b", "c"),                        # one-way bridge — does not merge
    ]
    result = compute_scc_python(edges)
    assert len(result) == 2


def test_scc_fallback_empty() -> None:
    result = compute_scc_python([])
    assert result == []


# ── Louvain Python fallback tests ─────────────────────────────────────────────

def test_louvain_two_cliques_two_communities() -> None:
    """Two cliques with a weak bridge → two distinct communities."""
    adj = [
        ("a", "b", 1.0), ("b", "c", 1.0), ("c", "a", 1.0),   # clique 1
        ("d", "e", 1.0), ("e", "f", 1.0), ("f", "d", 1.0),   # clique 2
        ("c", "d", 0.05),                                        # weak bridge
    ]
    nodes = ["a", "b", "c", "d", "e", "f"]
    result = compute_louvain_python(adj, nodes)

    assert result["a"] == result["b"] == result["c"], "clique 1 must share a community"
    assert result["d"] == result["e"] == result["f"], "clique 2 must share a community"
    assert result["a"] != result["d"], "the two cliques must be in different communities"


def test_louvain_star_single_community() -> None:
    """Hub + spokes with no inter-spoke edges — all may end up in one community."""
    adj = [(f"leaf{i}", "hub", 1.0) for i in range(5)]
    adj += [("hub", f"leaf{i}", 1.0) for i in range(5)]
    nodes = ["hub"] + [f"leaf{i}" for i in range(5)]
    result = compute_louvain_python(adj, nodes)
    # All nodes connected to hub — expect at most 2 communities
    assert len(set(result.values())) <= 2


def test_louvain_disconnected_components() -> None:
    """Fully disconnected components must end up in separate communities."""
    adj = [("a", "b", 1.0), ("c", "d", 1.0)]
    nodes = ["a", "b", "c", "d"]
    result = compute_louvain_python(adj, nodes)
    assert result["a"] == result["b"]
    assert result["c"] == result["d"]
    assert result["a"] != result["c"]


def test_louvain_isolated_nodes_get_communities() -> None:
    """Nodes with no edges still appear in result."""
    adj: list[tuple[str, str, float]] = []
    nodes = ["x", "y", "z"]
    result = compute_louvain_python(adj, nodes)
    assert set(result.keys()) == {"x", "y", "z"}


def test_louvain_returns_int_community_ids() -> None:
    adj = [("a", "b", 1.0), ("b", "c", 1.0), ("c", "a", 1.0)]
    nodes = ["a", "b", "c"]
    result = compute_louvain_python(adj, nodes)
    for v in result.values():
        assert isinstance(v, int)


def test_louvain_community_ids_start_at_zero() -> None:
    adj = [("a", "b", 1.0), ("b", "a", 1.0)]
    nodes = ["a", "b"]
    result = compute_louvain_python(adj, nodes)
    assert min(result.values()) == 0


# ── C++ native Louvain tests ──────────────────────────────────────────────────

def _native_available() -> bool:
    try:
        importlib.import_module("domain.structural_graph._native_graph")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _native_available(), reason="C++ native not built")
def test_native_louvain_two_cliques() -> None:
    """Native compute_louvain must separate two dense cliques."""
    native = importlib.import_module("domain.structural_graph._native_graph")
    adj = [
        ("a", "b", 1.0), ("b", "c", 1.0), ("c", "a", 1.0),
        ("d", "e", 1.0), ("e", "f", 1.0), ("f", "d", 1.0),
        ("c", "d", 0.05),
    ]
    nodes = ["a", "b", "c", "d", "e", "f"]
    result: dict[str, int] = native.compute_louvain(adj, nodes, 1.0, 42)

    assert result["a"] == result["b"] == result["c"]
    assert result["d"] == result["e"] == result["f"]
    assert result["a"] != result["d"]


@pytest.mark.skipif(not _native_available(), reason="C++ native not built")
def test_native_louvain_matches_python_on_simple_graph() -> None:
    """Native and Python fallback must produce structurally equivalent partitions."""
    native = importlib.import_module("domain.structural_graph._native_graph")
    adj = [
        ("a", "b", 1.0), ("b", "c", 1.0), ("c", "a", 1.0),
        ("d", "e", 1.0), ("e", "f", 1.0), ("f", "d", 1.0),
        ("c", "d", 0.05),
    ]
    nodes = ["a", "b", "c", "d", "e", "f"]

    native_result = native.compute_louvain(adj, nodes, 1.0, 42)
    python_result = compute_louvain_python(adj, nodes, 1.0)

    # Both must agree that clique 1 and clique 2 are in different communities
    assert native_result["a"] == native_result["b"] == native_result["c"]
    assert python_result["a"] == python_result["b"] == python_result["c"]
    assert native_result["a"] != native_result["d"]
    assert python_result["a"] != python_result["d"]


@pytest.mark.skipif(not _native_available(), reason="C++ native not built")
def test_native_louvain_isolated_nodes() -> None:
    """Native must include isolated nodes in output."""
    native = importlib.import_module("domain.structural_graph._native_graph")
    result: dict[str, int] = native.compute_louvain([], ["x", "y", "z"], 1.0, 42)
    assert set(result.keys()) == {"x", "y", "z"}


@pytest.mark.skipif(not _native_available(), reason="C++ native not built")
def test_native_louvain_empty_graph() -> None:
    native = importlib.import_module("domain.structural_graph._native_graph")
    result = native.compute_louvain([], [], 1.0, 42)
    assert result == {}
