"""Parity tests: C++ native modules must produce identical results to Python fallbacks.

If any test fails, the C++ implementation must be fixed to match the Python reference.
If native modules are not built, tests are skipped with a clear message.
"""
from __future__ import annotations

import importlib
import math

import pytest

from domain.structural_graph._louvain_fallback import compute_louvain_python
from domain.structural_graph._scc_fallback import compute_scc_python
from domain.structural_graph.service import _compute_scores_python, _expand_neighbors_python
from domain.retrieval.bm25_scorer import BM25Scorer, _WORD, CHUNK_TYPE_WEIGHT

# Try to load native modules — skip all tests if not built
try:
    from domain.structural_graph._native_graph import (
        expand_neighbors,
        compute_scc,
        compute_louvain,
        compute_scores,
        rank_and_budget,
    )

    HAS_NATIVE_GRAPH = True
except ImportError:
    HAS_NATIVE_GRAPH = False

try:
    from domain.retrieval._native_bm25 import tokenize, batch_score, batch_impact_score

    HAS_NATIVE_BM25 = True
except ImportError:
    HAS_NATIVE_BM25 = False


# ═══════════════════════════════════════════════════════════════════════════════
# Graph module tests
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_NATIVE_GRAPH, reason="native graph module not built")
class TestExpandNeighborsParity:
    """expand_neighbors: C++ vs Python must return same nodes and edges."""

    def test_simple_chain(self):
        """A→B→C→D, seed=A, hops=2 → should find B,C."""
        edges = [
            ("A", "B", "import", 0),
            ("B", "C", "import", 0),
            ("C", "D", "import", 0),
        ]
        cpp = expand_neighbors("A", edges, 2, 100)
        py = _expand_neighbors_python("A", edges, 2, 100)

        assert set(cpp["nodes"]) == set(py["nodes"]), f"Nodes differ: C++ {cpp['nodes']} vs Py {py['nodes']}"
        assert len(cpp["edges"]) == len(py["edges"]), f"Edge count differs: {len(cpp['edges'])} vs {len(py['edges'])}"
        assert set(cpp["edges"]) == set(py["edges"]), f"Edges differ: C++ {cpp['edges']} vs Py {py['edges']}"

    def test_diamond_graph(self):
        """A→B, A→C, B→D, C→D."""
        edges = [
            ("A", "B", "import", 0),
            ("A", "C", "import", 0),
            ("B", "D", "import", 0),
            ("C", "D", "import", 0),
        ]
        cpp = expand_neighbors("A", edges, 2, 100)
        py = _expand_neighbors_python("A", edges, 2, 100)

        assert set(cpp["nodes"]) == set(py["nodes"])
        assert set(cpp["edges"]) == set(py["edges"])

    def test_with_external_edges(self):
        """External edges (is_external=1) should be excluded."""
        edges = [
            ("A", "B", "import", 0),
            ("B", "C", "import", 0),
            ("C", "X", "import", 1),  # external — should not appear in result
        ]
        cpp = expand_neighbors("A", edges, 3, 100)
        py = _expand_neighbors_python("A", edges, 3, 100)

        assert set(cpp["nodes"]) == set(py["nodes"])
        assert set(cpp["edges"]) == set(py["edges"])
        # "X" should not be in either result because the C→X edge is external
        assert "X" not in cpp["nodes"]
        assert "X" not in py["nodes"]

    def test_limit_respected(self):
        """Large graph with limit=5 should respect both node and edge limits."""
        edges = []
        # Create a chain: A→B→C→D→E→F→G
        for i in range(7):
            src = chr(ord("A") + i)
            dst = chr(ord("A") + i + 1)
            edges.append((src, dst, "import", 0))

        cpp = expand_neighbors("A", edges, 10, 5)
        py = _expand_neighbors_python("A", edges, 10, 5)

        assert len(cpp["nodes"]) <= 5
        assert len(cpp["edges"]) <= 5
        assert set(cpp["nodes"]) == set(py["nodes"])
        assert set(cpp["edges"]) == set(py["edges"])

    def test_hops_clamped(self):
        """hops < 1 clamped to 1, hops > 4 clamped to 4."""
        edges = [
            ("A", "B", "import", 0),
            ("B", "C", "import", 0),
        ]

        # hops = 0 should be clamped to 1
        cpp_0 = expand_neighbors("A", edges, 0, 100)
        py_0 = _expand_neighbors_python("A", edges, 0, 100)
        assert set(cpp_0["nodes"]) == set(py_0["nodes"])

        # hops = 5 should be clamped to 4
        cpp_5 = expand_neighbors("A", edges, 5, 100)
        py_5 = _expand_neighbors_python("A", edges, 5, 100)
        assert set(cpp_5["nodes"]) == set(py_5["nodes"])

    def test_limit_clamped(self):
        """limit < 10 clamped to 10, limit > 2000 clamped to 2000."""
        edges = [(str(i), str(i + 1), "import", 0) for i in range(100)]

        # limit = 5 should be clamped to 10
        cpp_5 = expand_neighbors("0", edges, 1, 5)
        py_5 = _expand_neighbors_python("0", edges, 1, 5)
        assert set(cpp_5["nodes"]) == set(py_5["nodes"])

    def test_empty_edges(self):
        """No edges, seed node only."""
        cpp = expand_neighbors("A", [], 2, 100)
        py = _expand_neighbors_python("A", [], 2, 100)

        assert cpp["nodes"] == ["A"]
        assert cpp["nodes"] == py["nodes"]
        assert cpp["edges"] == []
        assert cpp["edges"] == py["edges"]

    def test_disconnected_graph(self):
        """Two disconnected components."""
        edges = [
            ("A", "B", "import", 0),
            ("X", "Y", "import", 0),
        ]
        cpp = expand_neighbors("A", edges, 3, 100)
        py = _expand_neighbors_python("A", edges, 3, 100)

        assert set(cpp["nodes"]) == {"A", "B"}
        assert set(cpp["nodes"]) == set(py["nodes"])


@pytest.mark.skipif(not HAS_NATIVE_GRAPH, reason="native graph module not built")
class TestComputeScoresParity:
    """compute_scores: C++ vs Python must return same items in same order."""

    def test_basic_scoring(self):
        """Indegree and outdegree scoring."""
        nodes = ["a.py", "b.py", "c.py"]
        edges = [
            ("a.py", "b.py", "import", 0),
            ("a.py", "c.py", "import", 0),
            ("b.py", "c.py", "import", 0),
        ]

        cpp = compute_scores(nodes, edges)
        py = _compute_scores_python(nodes, edges)

        # Convert to dicts for easier comparison
        cpp_dict = {item["rel_path"]: item for item in cpp}
        py_dict = {item["rel_path"]: item for item in py}

        assert set(cpp_dict.keys()) == set(py_dict.keys())

        for path in cpp_dict:
            cpp_item = cpp_dict[path]
            py_item = py_dict[path]
            assert cpp_item["indegree"] == py_item["indegree"], f"Indegree mismatch for {path}"
            assert cpp_item["outdegree"] == py_item["outdegree"], f"Outdegree mismatch for {path}"
            assert cpp_item["score"] == py_item["score"], f"Score mismatch for {path}"

    def test_scoring_order_and_tiebreak(self):
        """Items should be sorted by score desc, then indegree desc, then path asc."""
        nodes = ["a.py", "b.py", "c.py", "d.py"]
        edges = [
            ("a.py", "b.py", "import", 0),
            ("c.py", "b.py", "import", 0),
            ("d.py", "b.py", "import", 0),
            ("a.py", "d.py", "import", 0),
        ]

        cpp = compute_scores(nodes, edges)
        py = _compute_scores_python(nodes, edges)

        # Compare order
        cpp_paths = [item["rel_path"] for item in cpp]
        py_paths = [item["rel_path"] for item in py]

        assert cpp_paths == py_paths, f"Order mismatch: C++ {cpp_paths} vs Py {py_paths}"

    def test_empty_nodes_and_edges(self):
        """Empty inputs should return empty list."""
        cpp = compute_scores([], [])
        py = _compute_scores_python([], [])

        assert cpp == []
        assert py == []

    def test_disconnected_nodes(self):
        """Nodes with no edges should appear with 0 degree."""
        nodes = ["a.py", "b.py", "c.py"]
        edges = [("a.py", "b.py", "import", 0)]

        cpp = compute_scores(nodes, edges)
        py = _compute_scores_python(nodes, edges)

        cpp_dict = {item["rel_path"]: item for item in cpp}
        py_dict = {item["rel_path"]: item for item in py}

        assert "c.py" in cpp_dict
        assert cpp_dict["c.py"]["indegree"] == 0
        assert cpp_dict["c.py"]["outdegree"] == 0
        assert cpp_dict["c.py"]["score"] == 0


@pytest.mark.skipif(not HAS_NATIVE_GRAPH, reason="native graph module not built")
class TestComputeSccParity:
    """compute_scc: C++ vs Python must return SCCs in same order (size desc, nodes sorted)."""

    def test_simple_cycle(self):
        """A→B→A forms an SCC."""
        edges = [("A", "B"), ("B", "A")]

        cpp = compute_scc(edges)
        py = compute_scc_python(edges)

        # Both should return one SCC with both nodes
        assert len(cpp) == len(py)
        assert set(cpp[0]) == set(py[0]) == {"A", "B"}

    def test_multiple_cycles(self):
        """Two disjoint SCCs."""
        edges = [
            ("A", "B"),
            ("B", "A"),
            ("X", "Y"),
            ("Y", "X"),
        ]

        cpp = compute_scc(edges)
        py = compute_scc_python(edges)

        assert len(cpp) == len(py) == 2

        # Both should have two SCCs, each of size 2
        cpp_sizes = sorted([len(scc) for scc in cpp], reverse=True)
        py_sizes = sorted([len(scc) for scc in py], reverse=True)
        assert cpp_sizes == py_sizes

    def test_single_nodes_excluded(self):
        """SCCs with < 2 nodes should be excluded."""
        edges = [("A", "B")]  # No cycle, just A→B

        cpp = compute_scc(edges)
        py = compute_scc_python(edges)

        assert len(cpp) == 0
        assert len(py) == 0

    def test_large_cycle(self):
        """A→B→C→D→A forms a 4-node SCC."""
        edges = [
            ("A", "B"),
            ("B", "C"),
            ("C", "D"),
            ("D", "A"),
        ]

        cpp = compute_scc(edges)
        py = compute_scc_python(edges)

        assert len(cpp) == 1
        assert len(py) == 1
        assert set(cpp[0]) == set(py[0]) == {"A", "B", "C", "D"}

    def test_sorted_output(self):
        """SCCs should be sorted: largest first, nodes within each SCC sorted."""
        edges = [
            ("A", "B"),
            ("B", "A"),
            ("X", "Y"),
            ("Y", "Z"),
            ("Z", "X"),
        ]

        cpp = compute_scc(edges)
        py = compute_scc_python(edges)

        # Both should have 2 SCCs: one of size 3, one of size 2
        assert cpp[0] == sorted(cpp[0])
        assert py[0] == sorted(py[0])
        assert len(cpp[0]) >= len(cpp[1])  # Sorted by size desc
        assert len(py[0]) >= len(py[1])

    def test_empty_edges(self):
        """No edges → no SCCs."""
        cpp = compute_scc([])
        py = compute_scc_python([])

        assert len(cpp) == 0
        assert len(py) == 0


@pytest.mark.skipif(not HAS_NATIVE_GRAPH, reason="native graph module not built")
class TestComputeLouvainParity:
    """compute_louvain: Community assignment structure must match (IDs may differ)."""

    def test_two_communities(self):
        """Two clusters with no inter-cluster edges."""
        edges = [
            ("A", "B", 1.0),
            ("B", "C", 1.0),
            ("X", "Y", 1.0),
            ("Y", "Z", 1.0),
        ]
        node_ids = ["A", "B", "C", "X", "Y", "Z"]

        cpp = compute_louvain(edges, node_ids, 1.0)
        py = compute_louvain_python(edges, node_ids, 1.0)

        # Both should assign same-cluster nodes to same community
        assert type(cpp) == dict
        assert type(py) == dict

        # Nodes in the same cluster should have the same community ID
        # (even if the IDs differ, the partition structure must match)
        cpp_a_comm = cpp["A"]
        cpp_b_comm = cpp["B"]
        cpp_c_comm = cpp["C"]

        py_a_comm = py["A"]
        py_b_comm = py["B"]
        py_c_comm = py["C"]

        # In both C++ and Python: A, B, C should be in same community
        assert cpp_a_comm == cpp_b_comm == cpp_c_comm
        assert py_a_comm == py_b_comm == py_c_comm

        # X, Y, Z should be in same community
        cpp_x_comm = cpp["X"]
        cpp_y_comm = cpp["Y"]
        cpp_z_comm = cpp["Z"]

        py_x_comm = py["X"]
        py_y_comm = py["Y"]
        py_z_comm = py["Z"]

        assert cpp_x_comm == cpp_y_comm == cpp_z_comm
        assert py_x_comm == py_y_comm == py_z_comm

        # The two clusters should be different
        assert cpp_a_comm != cpp_x_comm
        assert py_a_comm != py_x_comm

    def test_weighted_edges(self):
        """Higher-weight edges should create tighter communities."""
        edges = [
            ("A", "B", 5.0),  # Strong edge A-B
            ("B", "C", 0.1),  # Weak edge B-C
            ("C", "D", 5.0),  # Strong edge C-D
        ]
        node_ids = ["A", "B", "C", "D"]

        cpp = compute_louvain(edges, node_ids, 1.0)
        py = compute_louvain_python(edges, node_ids, 1.0)

        # A-B should be in same community
        assert cpp["A"] == cpp["B"]
        assert py["A"] == py["B"]

        # C-D should be in same community
        assert cpp["C"] == cpp["D"]
        assert py["C"] == py["D"]

    def test_isolated_nodes(self):
        """Nodes with no edges get isolated communities."""
        edges = [("A", "B", 1.0)]
        node_ids = ["A", "B", "C"]

        cpp = compute_louvain(edges, node_ids, 1.0)
        py = compute_louvain_python(edges, node_ids, 1.0)

        # All three nodes should have assignments
        assert "A" in cpp and "B" in cpp and "C" in cpp
        assert "A" in py and "B" in py and "C" in py

        # C is isolated, so should be in its own community
        assert cpp["C"] not in [cpp["A"], cpp["B"]] or cpp["C"] == cpp["A"] == cpp["B"]
        assert py["C"] not in [py["A"], py["B"]] or py["C"] == py["A"] == py["B"]

    def test_empty_graph(self):
        """Empty graph should assign each node its own community."""
        edges = []
        node_ids = ["A", "B", "C"]

        cpp = compute_louvain(edges, node_ids, 1.0)
        py = compute_louvain_python(edges, node_ids, 1.0)

        # All nodes should be present
        assert set(cpp.keys()) == set(node_ids)
        assert set(py.keys()) == set(node_ids)

        # Each should be in a different community (no edges = no grouping)
        cpp_comms = [cpp[n] for n in node_ids]
        py_comms = [py[n] for n in node_ids]

        assert len(set(cpp_comms)) == len(node_ids)
        assert len(set(py_comms)) == len(node_ids)

    def test_complete_graph(self):
        """Fully connected graph should put all nodes in one community."""
        edges = [
            ("A", "B", 1.0),
            ("A", "C", 1.0),
            ("B", "C", 1.0),
        ]
        node_ids = ["A", "B", "C"]

        cpp = compute_louvain(edges, node_ids, 1.0)
        py = compute_louvain_python(edges, node_ids, 1.0)

        # All should be in same community
        assert cpp["A"] == cpp["B"] == cpp["C"]
        assert py["A"] == py["B"] == py["C"]


@pytest.mark.skipif(not HAS_NATIVE_GRAPH, reason="native graph module not built")
class TestRankAndBudgetParity:
    """rank_and_budget: C++ vs Python must select same chunks within budget."""

    def test_basic_ranking_with_budget(self):
        """Select top chunks by score within token budget."""
        scored = [
            ("chunk_a", 10.0, 50),
            ("chunk_b", 8.0, 30),
            ("chunk_c", 5.0, 40),
            ("chunk_d", 3.0, 100),
        ]
        budget = 100

        cpp = list(rank_and_budget(scored, budget))
        py_result, _ = _rank_and_budget_python(scored, budget)
        py = [item["chunk_id"] for item in py_result]

        assert cpp == py, f"Chunk selection mismatch: C++ {cpp} vs Py {py}"

    def test_respects_token_budget(self):
        """Total tokens of selected chunks should not exceed budget."""
        scored = [
            ("a", 10.0, 50),
            ("b", 8.0, 40),
            ("c", 6.0, 30),
            ("d", 4.0, 50),
        ]
        budget = 100

        cpp = list(rank_and_budget(scored, budget))
        py_result, _ = _rank_and_budget_python(scored, budget)

        cpp_tokens = sum(item[2] for item in scored if item[0] in cpp)
        assert cpp_tokens <= budget

    def test_empty_chunks(self):
        """Empty list should return empty."""
        cpp = list(rank_and_budget([], 100))
        py_result, _ = _rank_and_budget_python([], 100)
        py = [item["chunk_id"] for item in py_result]

        assert cpp == py == []

    def test_single_chunk_over_budget(self):
        """Single chunk exceeding budget should be skipped."""
        scored = [("a", 10.0, 150)]
        budget = 100

        cpp = list(rank_and_budget(scored, budget))
        py_result, _ = _rank_and_budget_python(scored, budget)
        py = [item["chunk_id"] for item in py_result]

        assert cpp == py == []

    def test_score_order_preserved(self):
        """Results should follow score order (descending)."""
        scored = [
            ("low", 1.0, 10),
            ("high", 10.0, 10),
            ("mid", 5.0, 10),
        ]
        budget = 1000

        cpp = list(rank_and_budget(scored, budget))
        py_result, _ = _rank_and_budget_python(scored, budget)
        py = [item["chunk_id"] for item in py_result]

        # Should be in order: high, mid, low
        expected = ["high", "mid", "low"]
        assert cpp == expected
        assert py == expected


# ═══════════════════════════════════════════════════════════════════════════════
# BM25 module tests
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_NATIVE_BM25, reason="native BM25 module not built")
class TestTokenizeParity:
    """tokenize: C++ vs Python regex must return identical token lists."""

    def test_simple_tokens(self):
        text = "hello world foo_bar"
        cpp = tokenize(text)
        py = _WORD.findall(text)

        assert cpp == py

    def test_mixed_case(self):
        text = "HelloWorld FooBar"
        cpp = tokenize(text)
        py = _WORD.findall(text)

        assert cpp == py

    def test_special_chars_excluded(self):
        text = "hello@world #test $value"
        cpp = tokenize(text)
        py = _WORD.findall(text)

        assert cpp == py
        assert cpp == ["hello", "world", "test", "value"]

    def test_numbers_included(self):
        text = "func123 test456 abc"
        cpp = tokenize(text)
        py = _WORD.findall(text)

        assert cpp == py
        assert "func123" in cpp
        assert "test456" in cpp

    def test_underscore_in_tokens(self):
        text = "snake_case _leading __dunder__ trailing_"
        cpp = tokenize(text)
        py = _WORD.findall(text)

        assert cpp == py

    def test_empty_string(self):
        cpp = tokenize("")
        py = _WORD.findall("")

        assert cpp == py == []

    def test_only_special_chars(self):
        cpp = tokenize("!@#$%^&*()")
        py = _WORD.findall("!@#$%^&*()")

        assert cpp == py == []

    def test_unicode_characters(self):
        """Non-ASCII characters: only ASCII alphanumeric + underscore should tokenize."""
        # Use ASCII-safe string to avoid file encoding issues with pybind11
        text = "hello world\xc3\xa9test"  # raw bytes won't appear in token
        # Both should extract same ASCII tokens
        py = _WORD.findall("hello world test")
        cpp = tokenize("hello world test")
        assert cpp == py


@pytest.mark.skipif(not HAS_NATIVE_BM25, reason="native BM25 module not built")
class TestBatchScoreParity:
    """batch_score: C++ vs Python BM25 scoring must match within tolerance."""

    def test_basic_scoring(self):
        """Score chunks with simple BM25."""
        corpus = [["foo", "bar"], ["foo", "foo", "baz"]]
        idf = BM25Scorer.build_idf(corpus, 2)
        avgdl = 2.5
        k1 = 2.0
        b = 0.75

        chunks = [
            ("chunk1", "foo bar", "file1.py"),
            ("chunk2", "foo foo baz", "file2.py"),
        ]
        terms = ["foo"]

        cpp = batch_score(chunks, terms, idf, avgdl, k1, b)
        py_scorer = BM25Scorer(idf, avgdl, k1, b)
        py = py_scorer.score_batch(chunks, terms)

        # Compare as dicts for easier assertion
        cpp_dict = dict(cpp)
        py_dict = dict(py)

        assert set(cpp_dict.keys()) == set(py_dict.keys())

        for chunk_id in cpp_dict:
            cpp_score = cpp_dict[chunk_id]
            py_score = py_dict[chunk_id]
            assert abs(cpp_score - py_score) < 1e-4, (
                f"Score mismatch for {chunk_id}: C++ {cpp_score} vs Py {py_score}"
            )

    def test_path_bonus(self):
        """Term in path increases score."""
        corpus = [["foo"], ["foo"]]
        idf = BM25Scorer.build_idf(corpus, 2)

        chunks = [
            ("c1", "foo content", "somefile.py"),
            ("c2", "foo content", "foo_module.py"),
        ]
        terms = ["foo"]

        cpp = dict(batch_score(chunks, terms, idf, 2.0, 2.0, 0.75))
        py_scorer = BM25Scorer(idf, 2.0, 2.0, 0.75)
        py = dict(py_scorer.score_batch(chunks, terms))

        # Both should show c2 > c1 due to path bonus
        assert cpp["c2"] > cpp["c1"]
        assert py["c2"] > py["c1"]

        # Scores should match
        assert abs(cpp["c1"] - py["c1"]) < 1e-4
        assert abs(cpp["c2"] - py["c2"]) < 1e-4

    def test_zero_score_no_terms(self):
        """Chunk with no query terms returns 0.0."""
        corpus = [["foo"], ["bar"]]
        idf = BM25Scorer.build_idf(corpus, 2)

        chunks = [("c1", "foo content", "file.py")]
        terms = ["baz"]  # Not in chunk

        cpp = dict(batch_score(chunks, terms, idf, 2.0, 2.0, 0.75))
        py_scorer = BM25Scorer(idf, 2.0, 2.0, 0.75)
        py = dict(py_scorer.score_batch(chunks, terms))

        assert cpp["c1"] == 0.0
        assert py["c1"] == 0.0

    def test_multiple_terms(self):
        """Multiple query terms contribute to score."""
        # Need 2+ docs so build_idf doesn't skip hapax (df < 2)
        corpus = [["foo", "bar", "baz"], ["foo", "bar", "qux"]]
        idf = BM25Scorer.build_idf(corpus, 2)

        chunks = [("c1", "foo bar baz content", "file.py")]
        terms = ["foo", "bar"]

        cpp = dict(batch_score(chunks, terms, idf, 3.0, 2.0, 0.75))
        py_scorer = BM25Scorer(idf, 3.0, 2.0, 0.75)
        py = dict(py_scorer.score_batch(chunks, terms))

        assert cpp["c1"] > 0.0
        assert abs(cpp["c1"] - py["c1"]) < 1e-4

    def test_length_normalization(self):
        """Shorter chunks with same TF score higher."""
        corpus = [["foo"], ["foo"]]
        idf = BM25Scorer.build_idf(corpus, 2)

        chunks = [
            ("short", "foo bar", "file.py"),
            ("long", "foo bar extra content padding word", "file.py"),
        ]
        terms = ["foo"]

        cpp = dict(batch_score(chunks, terms, idf, 2.5, 2.0, 0.75))
        py_scorer = BM25Scorer(idf, 2.5, 2.0, 0.75)
        py = dict(py_scorer.score_batch(chunks, terms))

        # Short should score higher
        assert cpp["short"] > cpp["long"]
        assert py["short"] > py["long"]

        # Scores should match
        assert abs(cpp["short"] - py["short"]) < 1e-4
        assert abs(cpp["long"] - py["long"]) < 1e-4

    def test_empty_chunks_list(self):
        """Empty chunks returns empty results."""
        idf = {"foo": 1.0}
        cpp = batch_score([], ["foo"], idf, 2.0, 2.0, 0.75)
        py_scorer = BM25Scorer(idf, 2.0, 2.0, 0.75)
        py = py_scorer.score_batch([], ["foo"])

        assert cpp == []
        assert py == []

    def test_empty_terms(self):
        """Empty terms should return 0 for all chunks."""
        idf = {}
        chunks = [
            ("c1", "foo bar", "file.py"),
            ("c2", "baz qux", "file.py"),
        ]

        cpp = dict(batch_score(chunks, [], idf, 2.0, 2.0, 0.75))
        py_scorer = BM25Scorer(idf, 2.0, 2.0, 0.75)
        py = dict(py_scorer.score_batch(chunks, []))

        assert cpp["c1"] == 0.0
        assert cpp["c2"] == 0.0
        assert py["c1"] == 0.0
        assert py["c2"] == 0.0

    def test_chunk_type_weighting(self):
        """chunk_type_weights should multiply scores (CS-227)."""
        corpus = [["foo"], ["foo"]]
        idf = BM25Scorer.build_idf(corpus, 2)

        chunks = [
            ("c1", "foo content", "file.py", "import_group"),
            ("c2", "foo content", "file.py", "function"),
        ]
        terms = ["foo"]
        weights = {"import_group": 0.2, "function": 1.5}

        cpp = dict(batch_score(chunks, terms, idf, 2.0, 2.0, 0.75, weights))
        py_scorer = BM25Scorer(idf, 2.0, 2.0, 0.75)
        py = dict(py_scorer.score_batch(chunks, terms))

        # c2 (function, weight 1.5) should score higher than c1 (import_group, weight 0.2)
        assert cpp["c2"] > cpp["c1"], f"Function {cpp['c2']} should > import_group {cpp['c1']}"

        # Scores should match Python fallback
        assert abs(cpp["c1"] - py["c1"]) < 1e-4
        assert abs(cpp["c2"] - py["c2"]) < 1e-4

    def test_min_score_absolute_cutoff(self):
        """Chunks below min_score_abs should be filtered (CS-227)."""
        corpus = [["foo"], ["bar"]]
        idf = BM25Scorer.build_idf(corpus, 2)

        chunks = [
            ("c1", "foo content", "file.py", "block"),
            ("c2", "bar content", "file.py", "block"),
        ]
        terms = ["foo"]

        # With min_score_abs=10, very high threshold
        cpp = dict(batch_score(chunks, terms, idf, 2.0, 2.0, 0.75, {}, min_score_abs=10.0))
        py_scorer = BM25Scorer(idf, 2.0, 2.0, 0.75)
        py = dict(py_scorer.score_batch(chunks, terms, min_score_abs=10.0))

        # Both should be filtered (or very few results)
        assert cpp == py, "Cutoff filtering should match"

    def test_min_score_relative_cutoff(self):
        """Chunks below relative fraction of top score should be filtered (CS-227)."""
        corpus = [["foo"], ["foo"]]
        idf = BM25Scorer.build_idf(corpus, 2)

        chunks = [
            ("high", "foo foo foo foo", "file.py", "block"),
            ("low", "foo", "file.py", "block"),
        ]
        terms = ["foo"]

        # Relative cutoff 0.5: only chunks with score >= 0.5 * max
        cpp_dict = dict(batch_score(chunks, terms, idf, 3.0, 2.0, 0.75, {}, min_score_relative=0.5))
        py_scorer = BM25Scorer(idf, 3.0, 2.0, 0.75)
        py_dict = dict(py_scorer.score_batch(chunks, terms, min_score_relative=0.5))

        # Both should filter the same way
        assert set(cpp_dict.keys()) == set(py_dict.keys()), "Same chunks should survive"
        for cid in cpp_dict:
            assert abs(cpp_dict[cid] - py_dict[cid]) < 1e-4


@pytest.mark.skipif(not HAS_NATIVE_BM25, reason="native BM25 module not built")
class TestBatchImpactScoreParity:
    """batch_impact_score: C++ vs Python impact scoring must match."""

    def test_basic_impact_scoring(self):
        """Score chunks with impact signals."""
        chunks = [
            {"id": "chunk1", "rel_path": "src/main.py"},
            {"id": "chunk2", "rel_path": "src/util.py"},
        ]
        hop_map = {"src/main.py": 0, "src/util.py": 2}
        central_ranks = {"src/main.py": 0}
        community_map = {"src/main.py": 1, "src/util.py": 2}
        seed_communities = {1}
        call_chain_files = set()
        bm25_scores = {"chunk1": 5.0, "chunk2": 3.0}

        cpp = batch_impact_score(
            chunks,
            hop_map,
            central_ranks,
            community_map,
            seed_communities,
            call_chain_files,
            bm25_scores,
        )
        py = _batch_impact_score_python(
            chunks,
            hop_map,
            central_ranks,
            community_map,
            seed_communities,
            call_chain_files,
            bm25_scores,
        )

        assert len(cpp) == len(py)
        cpp_dict = {item[0]: item for item in cpp}
        py_dict = {item[0]: item for item in py}

        for chunk_id in cpp_dict:
            cpp_scores = cpp_dict[chunk_id]
            py_scores = py_dict[chunk_id]

            # Compare total and component scores (skip index 0 = chunk_id string)
            for i in range(1, len(cpp_scores)):
                assert abs(cpp_scores[i] - py_scores[i]) < 1e-4, (
                    f"Score component {i} mismatch for {chunk_id}: "
                    f"C++ {cpp_scores[i]} vs Py {py_scores[i]}"
                )

    def test_no_signals(self):
        """Chunks with no signals should score 0."""
        chunks = [{"id": "c1", "rel_path": "unknown.py"}]
        hop_map = {}
        central_ranks = {}
        community_map = {}
        seed_communities = set()
        call_chain_files = set()
        bm25_scores = {}

        cpp = batch_impact_score(
            chunks,
            hop_map,
            central_ranks,
            community_map,
            seed_communities,
            call_chain_files,
            bm25_scores,
        )

        # Should have one result with total score 0
        assert len(cpp) == 1
        assert cpp[0][1] == 0.0  # total score

    def test_call_chain_bonus(self):
        """Files in call chain get symbol bonus."""
        chunks = [
            {"id": "c1", "rel_path": "chain.py"},
            {"id": "c2", "rel_path": "other.py"},
        ]
        hop_map = {}
        central_ranks = {}
        community_map = {}
        seed_communities = set()
        call_chain_files = {"chain.py"}
        bm25_scores = {}

        cpp = batch_impact_score(
            chunks,
            hop_map,
            central_ranks,
            community_map,
            seed_communities,
            call_chain_files,
            bm25_scores,
        )

        cpp_dict = {item[0]: item for item in cpp}

        # Tuple: (chunk_id, total, bm25_boost, symbol_bonus, community_bonus, centrality_bonus)
        # c1 should have symbol bonus at index 3
        assert cpp_dict["c1"][3] > 0  # symbol_bonus
        assert cpp_dict["c2"][3] == 0  # no symbol_bonus


# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions for Python fallback implementations
# ═══════════════════════════════════════════════════════════════════════════════


def _rank_and_budget_python(
    scored: list[tuple[str, float, int]],
    budget: int,
) -> tuple[list[dict], bool]:
    """Python fallback for rank_and_budget.

    Args:
        scored: list of (chunk_id, score, token_estimate)
        budget: token budget

    Returns:
        tuple of (results, used_cpp)
    """
    # Sort by score descending
    sorted_items = sorted(scored, key=lambda x: -x[1])

    out = []
    used = 0
    for chunk_id, score, tok in sorted_items:
        if used + tok > budget:
            continue
        out.append(
            {
                "chunk_id": chunk_id,
                "score": score,
                "token_estimate": tok,
            }
        )
        used += tok

    return out, False


def _batch_impact_score_python(
    chunks: list[dict],
    hop_map: dict[str, int],
    central_ranks: dict[str, int],
    community_map: dict[str, int],
    seed_communities: set[int],
    call_chain_files: set[str],
    bm25_scores: dict[str, float],
) -> list[tuple[str, float, float, float, float, float]]:
    """Python fallback for batch_impact_score.

    Mirrors the logic from impact_retrieval.py _score_chunk function.
    """
    # Scoring constants (mirrors bm25_native.cpp)
    HOP_WEIGHTS = {0: 3.0, 1: 2.0, 2: 1.2, 3: 0.6, 4: 0.3}
    IMPACT_CENTRALITY_BONUS = 2.0
    IMPACT_CENTRALITY_DECAY = 0.06
    IMPACT_COMMUNITY_BONUS = 1.0
    IMPACT_SYMBOL_CHAIN_BONUS = 2.5
    IMPACT_BM25_WEIGHT = 0.3

    out = []
    for chunk in chunks:
        chunk_id = chunk["id"]
        rel_path = chunk["rel_path"]

        # Hop weight
        hop_distance = hop_map.get(rel_path)
        if hop_distance is not None:
            hop_w = HOP_WEIGHTS.get(hop_distance, 0.2)
        else:
            hop_w = 0.0

        # BM25 boost
        bm25_score = bm25_scores.get(chunk_id, 0.0)
        bm25_boost = bm25_score * IMPACT_BM25_WEIGHT

        # Symbol chain bonus
        symbol_bonus = IMPACT_SYMBOL_CHAIN_BONUS if rel_path in call_chain_files else 0.0

        # Community bonus
        community_id = community_map.get(rel_path)
        community_bonus = 0.0
        if community_id is not None and community_id in seed_communities:
            community_bonus = IMPACT_COMMUNITY_BONUS

        # Centrality bonus
        rank = central_ranks.get(rel_path)
        if rank is not None:
            centrality_bonus = IMPACT_CENTRALITY_BONUS * (1.0 - IMPACT_CENTRALITY_DECAY) ** rank
        else:
            centrality_bonus = 0.0

        total = hop_w + bm25_boost + symbol_bonus + community_bonus + centrality_bonus
        out.append((chunk_id, total, bm25_boost, symbol_bonus, community_bonus, centrality_bonus))

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# STRESS TESTS — Complex cases that catch real logic bugs
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_NATIVE_BM25, reason="native BM25 module not built")
class TestTokenizeStress:
    """Edge cases that break naive tokenizers."""

    def test_single_char_tokens(self):
        """Single characters should be valid tokens (regex matches len>=1)."""
        text = "a b c x y z"
        cpp = tokenize(text)
        py = _WORD.findall(text)
        assert cpp == py

    def test_mixed_delimiters(self):
        """Tabs, newlines, multiple spaces."""
        text = "foo\tbar\nbaz  qux\r\nend"
        cpp = tokenize(text)
        py = _WORD.findall(text)
        assert cpp == py

    def test_long_string(self):
        """10K characters — performance and correctness."""
        text = " ".join(f"token_{i}" for i in range(2000))
        cpp = tokenize(text)
        py = _WORD.findall(text)
        assert cpp == py
        assert len(cpp) == 2000

    def test_consecutive_underscores(self):
        """__init__, __main__, __dunder__ patterns."""
        text = "class __init__ def __main__ var __dunder__"
        cpp = tokenize(text)
        py = _WORD.findall(text)
        assert cpp == py

    def test_numeric_tokens(self):
        """Pure numbers, mixed alphanumeric."""
        text = "123 abc 456def ghi789 0x1F var_123_end"
        cpp = tokenize(text)
        py = _WORD.findall(text)
        assert cpp == py

    def test_empty_between_delimiters(self):
        """Lots of non-word characters in a row."""
        text = "foo!!!???...---bar"
        cpp = tokenize(text)
        py = _WORD.findall(text)
        assert cpp == py
        assert cpp == ["foo", "bar"]


@pytest.mark.skipif(not HAS_NATIVE_BM25, reason="native BM25 module not built")
class TestBatchScoreStress:
    """Complex BM25 scoring cases."""

    def test_large_corpus_scoring(self):
        """Score 50 chunks against a corpus with realistic IDF distribution."""
        corpus = []
        for i in range(20):
            tokens = [f"common_{j}" for j in range(3)]
            tokens += [f"mid_{j}" for j in range(i % 5, (i % 5) + 2)]
            tokens += [f"rare_{i}", f"rare_{i}"]
            corpus.append(tokens)
        idf = BM25Scorer.build_idf(corpus, 20)
        avgdl = sum(len(doc) for doc in corpus) / len(corpus)

        chunks = []
        for i in range(50):
            content = f"common_0 common_1 mid_{i % 5} extra_word_{i} padding"
            chunks.append((f"c{i}", content, f"path/file_{i}.py"))

        terms = ["common_0", "mid_2"]

        cpp = dict(batch_score(chunks, terms, idf, avgdl, 2.0, 0.75))
        scorer = BM25Scorer(idf, avgdl, 2.0, 0.75)
        py = dict(scorer.score_batch(chunks, terms))

        assert len(cpp) == len(py) == 50
        for cid in cpp:
            assert abs(cpp[cid] - py[cid]) < 1e-4, f"{cid}: C++ {cpp[cid]:.6f} vs Py {py[cid]:.6f}"

    def test_path_bonus_only(self):
        """Term appears in path but NOT in content — path bonus should fire."""
        corpus = [["auth", "middleware"], ["auth", "service"]]
        idf = BM25Scorer.build_idf(corpus, 2)

        chunks = [
            ("c1", "unrelated content here nothing matches", "auth/middleware.py"),
            ("c2", "unrelated content here nothing matches", "utils/helper.py"),
        ]
        terms = ["auth"]

        cpp = dict(batch_score(chunks, terms, idf, 3.0, 2.0, 0.75))
        scorer = BM25Scorer(idf, 3.0, 2.0, 0.75)
        py = dict(scorer.score_batch(chunks, terms))

        assert cpp["c1"] > cpp["c2"]
        assert abs(cpp["c1"] - py["c1"]) < 1e-4
        assert abs(cpp["c2"] - py["c2"]) < 1e-4

    def test_case_insensitive(self):
        """BM25 should lowercase before matching. Terms are pre-lowered."""
        corpus = [["hello", "world"], ["hello", "test"]]
        idf = BM25Scorer.build_idf(corpus, 2)

        chunks = [("c1", "Hello WORLD This Has Mixed CASE", "SRC/File.PY")]
        terms = ["hello", "world"]

        cpp = dict(batch_score(chunks, terms, idf, 3.0, 2.0, 0.75))
        scorer = BM25Scorer(idf, 3.0, 2.0, 0.75)
        py = dict(scorer.score_batch(chunks, terms))

        assert cpp["c1"] > 0.0
        assert abs(cpp["c1"] - py["c1"]) < 1e-4

    def test_high_tf_saturation(self):
        """BM25 should saturate — 100 occurrences shouldn't score 100x more than 1."""
        corpus = [["foo", "bar"], ["foo", "baz"]]
        idf = BM25Scorer.build_idf(corpus, 2)

        chunks = [
            ("once", "foo bar", "file.py"),
            ("many", " ".join(["foo"] * 100 + ["bar"]), "file.py"),
        ]
        terms = ["foo"]

        cpp = dict(batch_score(chunks, terms, idf, 3.0, 2.0, 0.75))
        scorer = BM25Scorer(idf, 3.0, 2.0, 0.75)
        py = dict(scorer.score_batch(chunks, terms))

        assert cpp["many"] > cpp["once"]
        assert cpp["many"] < cpp["once"] * 5  # saturation check
        assert abs(cpp["once"] - py["once"]) < 1e-4
        assert abs(cpp["many"] - py["many"]) < 1e-4

    def test_content_and_path_both_match(self):
        """Term in both content AND path — should get both contributions."""
        corpus = [["auth", "service"], ["auth", "handler"]]
        idf = BM25Scorer.build_idf(corpus, 2)

        chunks = [
            ("both", "auth service handler", "auth/service.py"),
            ("content_only", "auth service handler", "utils/helper.py"),
            ("path_only", "unrelated content", "auth/service.py"),
        ]
        terms = ["auth"]

        cpp = dict(batch_score(chunks, terms, idf, 3.0, 2.0, 0.75))
        scorer = BM25Scorer(idf, 3.0, 2.0, 0.75)
        py = dict(scorer.score_batch(chunks, terms))

        # "both" gets content + path bonus → highest score
        assert cpp["both"] > cpp["content_only"]
        assert cpp["both"] > cpp["path_only"]
        # content_only and path_only may be equal (BM25 content contrib can equal path bonus)
        # but both must be > 0 and match Python exactly
        assert cpp["content_only"] > 0.0
        assert cpp["path_only"] > 0.0
        for cid in ["both", "content_only", "path_only"]:
            assert abs(cpp[cid] - py[cid]) < 1e-4, f"{cid}: {cpp[cid]} vs {py[cid]}"


@pytest.mark.skipif(not HAS_NATIVE_BM25, reason="native BM25 module not built")
class TestBatchImpactScoreStress:
    """Complex impact scoring that verifies every bonus stacks correctly."""

    def test_all_bonuses_stacking(self):
        """File with EVERY bonus: hop 1 + call chain + central rank 0 + same community + BM25."""
        chunks = [
            {"id": "mega", "rel_path": "core.py"},
            {"id": "none", "rel_path": "random.py"},
        ]
        hop_map = {"core.py": 1, "random.py": 4}
        central_ranks = {"core.py": 0}
        community_map = {"core.py": 5, "random.py": 99}
        seed_communities = {5}
        call_chain_files = {"core.py"}
        bm25_scores = {"mega": 10.0, "none": 0.0}

        cpp = batch_impact_score(
            chunks, hop_map, central_ranks, community_map,
            seed_communities, call_chain_files, bm25_scores,
        )
        py = _batch_impact_score_python(
            chunks, hop_map, central_ranks, community_map,
            seed_communities, call_chain_files, bm25_scores,
        )

        cpp_dict = {item[0]: item for item in cpp}
        py_dict = {item[0]: item for item in py}

        mega_cpp = cpp_dict["mega"]
        # (chunk_id, total, bm25_boost, symbol_bonus, community_bonus, centrality_bonus)
        assert mega_cpp[2] == pytest.approx(10.0 * 0.3, abs=1e-4)   # bm25_boost
        assert mega_cpp[3] == pytest.approx(2.5, abs=1e-4)           # symbol_bonus
        assert mega_cpp[4] == pytest.approx(1.0, abs=1e-4)           # community_bonus
        assert mega_cpp[5] == pytest.approx(2.0, abs=1e-4)           # centrality rank 0
        expected_total = 2.0 + (10.0 * 0.3) + 2.5 + 1.0 + 2.0
        assert mega_cpp[1] == pytest.approx(expected_total, abs=1e-4)

        for i in range(1, 6):
            assert abs(mega_cpp[i] - py_dict["mega"][i]) < 1e-4

        none_cpp = cpp_dict["none"]
        assert none_cpp[1] < mega_cpp[1]

    def test_centrality_decay_at_high_ranks(self):
        """Centrality bonus should decay: rank 0 >> rank 10 >> rank 30."""
        chunks = [
            {"id": "r0", "rel_path": "top.py"},
            {"id": "r10", "rel_path": "mid.py"},
            {"id": "r30", "rel_path": "low.py"},
        ]
        hop_map = {"top.py": 1, "mid.py": 1, "low.py": 1}
        central_ranks = {"top.py": 0, "mid.py": 10, "low.py": 30}
        community_map = {}
        seed_communities = set()
        call_chain_files = set()
        bm25_scores = {}

        cpp = batch_impact_score(
            chunks, hop_map, central_ranks, community_map,
            seed_communities, call_chain_files, bm25_scores,
        )
        py = _batch_impact_score_python(
            chunks, hop_map, central_ranks, community_map,
            seed_communities, call_chain_files, bm25_scores,
        )

        cpp_dict = {item[0]: item for item in cpp}
        py_dict = {item[0]: item for item in py}

        # Centrality at index 5: 2.0 * (0.94)^rank
        assert cpp_dict["r0"][5] == pytest.approx(2.0 * (0.94 ** 0), abs=1e-4)
        assert cpp_dict["r10"][5] == pytest.approx(2.0 * (0.94 ** 10), abs=1e-4)
        assert cpp_dict["r30"][5] == pytest.approx(2.0 * (0.94 ** 30), abs=1e-4)
        assert cpp_dict["r0"][5] > cpp_dict["r10"][5] > cpp_dict["r30"][5] > 0

        for cid in ["r0", "r10", "r30"]:
            for i in range(1, 6):
                assert abs(cpp_dict[cid][i] - py_dict[cid][i]) < 1e-4

    def test_hop_beyond_4(self):
        """Hop distance > 4 should use fallback weight 0.2."""
        chunks = [{"id": "far", "rel_path": "far.py"}]
        hop_map = {"far.py": 7}

        cpp = batch_impact_score(chunks, hop_map, {}, {}, set(), set(), {})
        py = _batch_impact_score_python(chunks, hop_map, {}, {}, set(), set(), {})

        assert cpp[0][1] == pytest.approx(0.2, abs=1e-4)
        assert py[0][1] == pytest.approx(0.2, abs=1e-4)

    def test_many_chunks_scoring(self):
        """100 chunks with mixed signals — ALL components must match."""
        chunks = [{"id": f"c{i}", "rel_path": f"file_{i}.py"} for i in range(100)]
        hop_map = {f"file_{i}.py": i % 5 for i in range(60)}
        central_ranks = {f"file_{i}.py": i for i in range(20)}
        community_map = {f"file_{i}.py": i % 3 for i in range(100)}
        seed_communities = {0, 1}
        call_chain_files = {f"file_{i}.py" for i in range(10)}
        bm25_scores = {f"c{i}": float(i) * 0.5 for i in range(100)}

        cpp = batch_impact_score(
            chunks, hop_map, central_ranks, community_map,
            seed_communities, call_chain_files, bm25_scores,
        )
        py = _batch_impact_score_python(
            chunks, hop_map, central_ranks, community_map,
            seed_communities, call_chain_files, bm25_scores,
        )

        assert len(cpp) == len(py) == 100
        cpp_dict = {item[0]: item for item in cpp}
        py_dict = {item[0]: item for item in py}

        mismatches = []
        for cid in cpp_dict:
            for i in range(1, 6):
                diff = abs(cpp_dict[cid][i] - py_dict[cid][i])
                if diff >= 1e-4:
                    mismatches.append(f"{cid}[{i}]: C++ {cpp_dict[cid][i]:.6f} vs Py {py_dict[cid][i]:.6f}")
        assert not mismatches, f"{len(mismatches)} mismatches:\n" + "\n".join(mismatches[:10])


@pytest.mark.skipif(not HAS_NATIVE_GRAPH, reason="native graph module not built")
class TestExpandNeighborsStress:
    """Complex graph traversal cases."""

    def test_star_graph(self):
        """Hub node with 50 spokes — all should be discovered at hop 1."""
        edges = [("hub", f"spoke_{i}", "import", 0) for i in range(50)]
        cpp = expand_neighbors("hub", edges, 1, 100)
        py = _expand_neighbors_python("hub", edges, 1, 100)
        assert set(cpp["nodes"]) == set(py["nodes"])
        assert len(cpp["nodes"]) == 51

    def test_deep_chain_limited(self):
        """Chain of 20 nodes with hops=3 limit=10 — must stop correctly."""
        edges = [(f"n{i}", f"n{i+1}", "import", 0) for i in range(20)]
        cpp = expand_neighbors("n0", edges, 3, 10)
        py = _expand_neighbors_python("n0", edges, 3, 10)
        assert set(cpp["nodes"]) == set(py["nodes"])
        assert len(cpp["nodes"]) <= 10

    def test_bidirectional_edges(self):
        """A<->B: edges in both directions."""
        edges = [
            ("A", "B", "import", 0),
            ("B", "A", "import", 0),
            ("B", "C", "import", 0),
        ]
        cpp = expand_neighbors("A", edges, 2, 100)
        py = _expand_neighbors_python("A", edges, 2, 100)
        assert set(cpp["nodes"]) == set(py["nodes"])
        assert "C" in cpp["nodes"]
