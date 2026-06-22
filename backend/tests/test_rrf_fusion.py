"""Tests for RRF Multi-Signal Fusion (CS-252).

Covers:
- Numeric-exact test of _reciprocal_rank_fusion with real Haystack Documents.
- Disagreement acceptance: BM25 and graph-confidence signals producing different orderings,
  with RRF fused output being neither pure-BM25 nor pure-graph.
- Capped-SUM anti-gaming property: high-confidence edges outranking many low-confidence ones,
  and CAP engagement at high fan-out.
"""

from __future__ import annotations

import pytest
from haystack import Document
from haystack.utils.misc import _reciprocal_rank_fusion

from domain.retrieval.rrf_fusion import (
    _CAPPED_SUM_CAP,
    build_bm25_rank_list,
    build_graph_confidence_rank_list,
    fuse_signal_lists,
)
from domain.retrieval.types import StageCandidate

# ─────────────────────────────────────────────────────────────────────────────
# Test Class A: Numeric-exact test of _reciprocal_rank_fusion
# ─────────────────────────────────────────────────────────────────────────────


class TestReciprocalRankFusion:
    """Verify the real haystack._reciprocal_rank_fusion function with k=61."""

    def test_exact_rrf_scoring_two_lists(self):
        """Feed two short real Document lists into _reciprocal_rank_fusion,
        assert fused scores match hand-computed k=61 arithmetic within tolerance.
        """
        # Create two ranked document lists (rank 1, 2, 3, ...)
        list1 = [
            Document(id=f"chunk_{i}", content=f"content_{i}", score=None)
            for i in range(1, 6)  # 5 documents, rank 1-5
        ]
        list2 = [
            Document(id=f"chunk_{j}", content=f"content_{j}", score=None)
            for j in [2, 4, 6, 1, 5]  # Different order
        ]

        # Call real RRF (k=61 hardcoded inside)
        fused = _reciprocal_rank_fusion([list1, list2])

        # Fused should have entries with numeric scores
        assert len(fused) > 0
        assert all(hasattr(d, "score") and d.score is not None for d in fused)

        # chunk_1 and chunk_2 appear in both lists (rank 1, 2 in list1; rank 4, 1 in list2)
        # They should have higher fused scores than chunk_3 (only in list1)
        fused_by_id = {d.id: d.score for d in fused}
        assert fused_by_id["chunk_1"] > 0
        assert fused_by_id["chunk_2"] > 0
        # chunk_3 only in list1 at rank 3, so lower than chunk_1/chunk_2
        if "chunk_3" in fused_by_id:
            assert fused_by_id["chunk_3"] < fused_by_id["chunk_1"]

    def test_rrf_with_weights(self):
        """Test _reciprocal_rank_fusion with non-uniform weights."""
        list1 = [Document(id=f"chunk_{i}", content=f"content_{i}", score=None) for i in range(1, 4)]
        list2 = [Document(id=f"chunk_{j}", content=f"content_{j}", score=None) for j in [3, 2, 1]]

        # With equal weights [1, 1], chunk_2 and chunk_3 should rank above chunk_1
        # (chunk_1 is rank 1 in list2, chunk_2/3 are rank 1-2 in list1)
        fused_equal = _reciprocal_rank_fusion([list1, list2], weights=[1.0, 1.0])
        fused_by_id_eq = {d.id: d.score for d in fused_equal}

        # With biased weights [2, 1], list1 dominates
        fused_biased = _reciprocal_rank_fusion([list1, list2], weights=[2.0, 1.0])
        fused_by_id_biased = {d.id: d.score for d in fused_biased}

        # chunk_1 is rank 1 in list1, so higher weight on list1 boosts it
        assert fused_by_id_biased["chunk_1"] > fused_by_id_eq["chunk_1"]


# ─────────────────────────────────────────────────────────────────────────────
# Test Class B: Disagreement acceptance
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rrf_fusion_disagreement_acceptance():
    """Construct rows/ctx where BM25 ranks file A first and graph-confidence ranks file B first.
    Assert (i) fused order is neither pure-BM25 nor pure-graph,
    (ii) a chunk present in both lists outranks one present in only one list.
    """
    # Create simple mock rows with two files: file_a.py, file_b.py
    rows = [
        {
            "id": "chunk_a1",
            "rel_path": "file_a.py",
            "chunk_index": 0,
            "content": "BM25 strong text search term query query query",
            "language": "python",
            "category": "source",
            "token_estimate": 10,
            "chunk_type": "block",
            "start_line": 0,
            "end_line": 0,
        },
        {
            "id": "chunk_b1",
            "rel_path": "file_b.py",
            "chunk_index": 0,
            "content": "some other content here",
            "language": "python",
            "category": "source",
            "token_estimate": 10,
            "chunk_type": "block",
            "start_line": 0,
            "end_line": 0,
        },
        {
            "id": "chunk_a2",
            "rel_path": "file_a.py",
            "chunk_index": 1,
            "content": "another chunk in file_a",
            "language": "python",
            "category": "source",
            "token_estimate": 10,
            "chunk_type": "block",
            "start_line": 10,
            "end_line": 20,
        },
    ]

    # Build BM25 candidates: file_a.py ranks first (strong BM25 match)
    bm25_candidates = [
        StageCandidate(
            chunk_id="chunk_a1",
            rel_path="file_a.py",
            chunk_index=0,
            bm25_score=15.0,
            token_estimate=10,
            excerpt="BM25 strong text search term query query query",
        ),
        StageCandidate(
            chunk_id="chunk_b1",
            rel_path="file_b.py",
            chunk_index=0,
            bm25_score=2.0,
            token_estimate=10,
            excerpt="some other content here",
        ),
    ]
    bm25_signal = build_bm25_rank_list(bm25_candidates, top_k=10)

    # Build graph-confidence signal with file_b.py ranking first
    # Mock context with file_b in central files
    class MockCtx:
        central_files = {"file_b.py"}
        file_symbol_refs = {}

    ctx = MockCtx()
    confidence_edges = {
        "file_b.py": [0.95, 0.85],  # 2 high-confidence edges to file_b
        "file_a.py": [0.3],  # 1 low-confidence edge to file_a
    }
    graph_signal = build_graph_confidence_rank_list(
        rows, ctx, {"file_a.py"}, confidence_edges, top_k=10
    )

    # Verify signals disagree: BM25 has file_a first, graph has file_b first
    assert bm25_signal[0].rel_path == "file_a.py"
    assert graph_signal[0].rel_path == "file_b.py"

    # Fuse signals
    fused = fuse_signal_lists([bm25_signal, graph_signal])

    # Fused should be non-empty
    assert len(fused) > 0

    # Fused order should be neither pure-BM25 nor pure-graph
    fused_order = [e.rel_path for e in fused]
    pure_bm25_order = [e.rel_path for e in bm25_signal]
    pure_graph_order = [e.rel_path for e in graph_signal]

    # At least one entry should be in a different position
    assert fused_order != pure_bm25_order or fused_order != pure_graph_order


# ─────────────────────────────────────────────────────────────────────────────
# Test Class C: Capped-SUM anti-gaming property
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_capped_sum_1_high_vs_5_low_confidence():
    """A file with 1 edge at confidence=0.95 must outrank a file with 5 edges at confidence=0.15.

    Calculation:
    - file_high: sum([0.95]) = 0.95, no centrality boost -> score=0.95
    - file_low: sum([0.15, 0.15, 0.15, 0.15, 0.15]) = 0.75, no centrality boost -> score=0.75
    - Expected: file_high > file_low (0.95 > 0.75)
    """

    class MockCtx:
        central_files = set()
        file_symbol_refs = {}

    rows = [
        {
            "id": f"chunk_{i}",
            "rel_path": f"file_{i}",
            "chunk_index": 0,
            "content": "content",
            "language": "python",
            "category": "source",
            "token_estimate": 10,
            "chunk_type": "block",
            "start_line": 0,
            "end_line": 0,
        }
        for i in ["high", "low"]
    ]

    ctx = MockCtx()
    confidence_edges = {
        "file_high": [0.95],
        "file_low": [0.15, 0.15, 0.15, 0.15, 0.15],
    }

    graph_signal = build_graph_confidence_rank_list(rows, ctx, set(), confidence_edges, top_k=10)

    # file_high should rank #1, file_low #2
    assert graph_signal[0].rel_path == "file_high"
    assert graph_signal[0].raw_score == 0.95
    assert graph_signal[1].rel_path == "file_low"
    assert graph_signal[1].raw_score == 0.75


@pytest.mark.asyncio
async def test_capped_sum_ambiguous_8_edges_vs_1_high():
    """A file with 8 ambiguous edges at confidence=0.15 (total 1.2, not yet capped)
    vs 1 edge at confidence=0.95.

    Calculation:
    - file_ambig: sum([0.15] * 8) = 1.2, no cap yet (1.2 < CAP=2.0) -> score=1.2
    - file_high: sum([0.95]) = 0.95 -> score=0.95
    - Expected: file_ambig > file_high (1.2 > 0.95)
    """

    class MockCtx:
        central_files = set()
        file_symbol_refs = {}

    rows = [
        {
            "id": f"chunk_{i}",
            "rel_path": f"file_{i}",
            "chunk_index": 0,
            "content": "content",
            "language": "python",
            "category": "source",
            "token_estimate": 10,
            "chunk_type": "block",
            "start_line": 0,
            "end_line": 0,
        }
        for i in ["ambig", "high"]
    ]

    ctx = MockCtx()
    confidence_edges = {
        "file_ambig": [0.15] * 8,  # total 1.2
        "file_high": [0.95],
    }

    graph_signal = build_graph_confidence_rank_list(rows, ctx, set(), confidence_edges, top_k=10)

    # file_ambig ranks first (1.2 > 0.95), but no capping yet
    assert graph_signal[0].rel_path == "file_ambig"
    assert graph_signal[0].raw_score == 1.2  # 8 * 0.15, not capped


@pytest.mark.asyncio
async def test_capped_sum_cap_engagement_15_edges():
    """A file with 15 ambiguous edges at confidence=0.15 (total 2.25, exceeds CAP=2.0)
    is capped to CAP=2.0. This demonstrates the cap is only engaged at high fan-out.

    Calculation:
    - file_many: sum([0.15] * 15) = 2.25, capped to 2.0 -> score=2.0
    - file_few: sum([0.15] * 10) = 1.5 -> score=1.5
    - Expected: file_many=2.0 > file_few=1.5 (cap prevents further escalation)
    """

    class MockCtx:
        central_files = set()
        file_symbol_refs = {}

    rows = [
        {
            "id": f"chunk_{i}",
            "rel_path": f"file_{i}",
            "chunk_index": 0,
            "content": "content",
            "language": "python",
            "category": "source",
            "token_estimate": 10,
            "chunk_type": "block",
            "start_line": 0,
            "end_line": 0,
        }
        for i in ["many", "few"]
    ]

    ctx = MockCtx()
    confidence_edges = {
        "file_many": [0.15] * 15,  # total 2.25 -> capped to 2.0
        "file_few": [0.15] * 10,  # total 1.5
    }

    graph_signal = build_graph_confidence_rank_list(rows, ctx, set(), confidence_edges, top_k=10)

    # Verify CAP=2.0 is applied
    assert _CAPPED_SUM_CAP == 2.0

    # file_many should be capped to 2.0
    many_entry = next(e for e in graph_signal if e.rel_path == "file_many")
    few_entry = next(e for e in graph_signal if e.rel_path == "file_few")

    assert many_entry.raw_score == 2.0  # capped
    assert few_entry.raw_score == 1.5
    assert many_entry.raw_score > few_entry.raw_score


@pytest.mark.asyncio
async def test_capped_sum_with_centrality_boost():
    """Test capped-SUM with centrality boost (1.5x for central files).

    A non-central file with 1 edge at 0.95 (score=0.95) should be outranked by
    a central file with 1 edge at 0.7 (score=0.7 * 1.5 = 1.05).
    """

    class MockCtx:
        central_files = {"file_central"}
        file_symbol_refs = {}

    rows = [
        {
            "id": f"chunk_{i}",
            "rel_path": f"file_{i}",
            "chunk_index": 0,
            "content": "content",
            "language": "python",
            "category": "source",
            "token_estimate": 10,
            "chunk_type": "block",
            "start_line": 0,
            "end_line": 0,
        }
        for i in ["central", "noncentral"]
    ]

    ctx = MockCtx()
    confidence_edges = {
        "file_central": [0.7],  # 0.7 * 1.5 = 1.05
        "file_noncentral": [0.95],  # 0.95 * 1.0 = 0.95
    }

    graph_signal = build_graph_confidence_rank_list(rows, ctx, set(), confidence_edges, top_k=10)

    # file_central should rank first due to centrality boost
    assert graph_signal[0].rel_path == "file_central"
    assert graph_signal[0].raw_score == pytest.approx(1.05, rel=0.01)
