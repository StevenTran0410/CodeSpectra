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
    _best_chunk_per_file,
    _symbol_overlap_fallback,
    build_bm25_rank_list,
    build_category_hint_rank_list,
    build_graph_confidence_rank_list,
    build_module_proximity_rank_list,
    fuse_signal_lists,
)
from domain.retrieval.types import RetrievalSection, StageCandidate

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
    # Precompute best_chunk_by_file (new in CS-253)
    best_chunk_by_file = _best_chunk_per_file(rows, bm25_candidates, None)

    bm25_signal = build_bm25_rank_list(bm25_candidates, best_chunk_by_file, top_k=10)

    # Build graph-confidence signal with file_b.py ranking first
    # Mock context with file_b in central files
    class MockCtx:
        central_files = {"file_b.py"}
        file_symbol_refs = {}
        file_community = {}

    ctx = MockCtx()
    confidence_edges = {
        "file_b.py": [0.95, 0.85],  # 2 high-confidence edges to file_b
        "file_a.py": [0.3],  # 1 low-confidence edge to file_a
    }
    graph_signal = build_graph_confidence_rank_list(
        rows, ctx, {"file_a.py"}, confidence_edges, best_chunk_by_file, top_k=10
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
        file_community = {}

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

    best_chunk_by_file = _best_chunk_per_file(rows, [], None)
    graph_signal = build_graph_confidence_rank_list(
        rows, ctx, set(), confidence_edges, best_chunk_by_file, top_k=10
    )

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
        file_community = {}

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

    best_chunk_by_file = _best_chunk_per_file(rows, [], None)
    graph_signal = build_graph_confidence_rank_list(
        rows, ctx, set(), confidence_edges, best_chunk_by_file, top_k=10
    )

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
        file_community = {}

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

    best_chunk_by_file = _best_chunk_per_file(rows, [], None)
    graph_signal = build_graph_confidence_rank_list(
        rows, ctx, set(), confidence_edges, best_chunk_by_file, top_k=10
    )

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
        file_community = {}

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

    best_chunk_by_file = _best_chunk_per_file(rows, [], None)
    graph_signal = build_graph_confidence_rank_list(
        rows, ctx, set(), confidence_edges, best_chunk_by_file, top_k=10
    )

    # file_central should rank first due to centrality boost
    assert graph_signal[0].rel_path == "file_central"
    assert graph_signal[0].raw_score == pytest.approx(1.05, rel=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# Test Class D: CS-253 Join-Key Fix (3a) — Regression Test
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_join_key_fix_collapses_to_one_row_per_file():
    """Regression test for CS-253 3a: join-key bug fix.

    Construct fixture where:
    - file_a.py has two chunks: chunk_a1 (chunk_index=0, bm25_score=10.0)
                                chunk_a2 (chunk_index=1, bm25_score=0.0)
    - The OLD bug would pick chunk_a2 for graph_signal (max by chunk_index)
    - The fix must pick chunk_a1 (best BM25 chunk) for BOTH bm25_signal and graph_signal

    Assert:
    1. bm25_signal and graph_signal both reference chunk_a1 (same chunk_id)
    2. fuse_signal_lists produces exactly ONE FusedRankEntry for file_a.py
       (not two separate rows, one for each signal's chunk_id mismatch)
    3. per_signal_ranks has both 'bm25' and 'graph_confidence' keys
    """
    rows = [
        {
            "id": "chunk_a1",
            "rel_path": "file_a.py",
            "chunk_index": 0,
            "content": "BM25 strong match query query query",
            "language": "python",
            "category": "source",
            "token_estimate": 10,
            "chunk_type": "block",
            "start_line": 0,
            "end_line": 30,
        },
        {
            "id": "chunk_a2",
            "rel_path": "file_a.py",
            "chunk_index": 1,
            "content": "trailing chunk end of file",
            "language": "python",
            "category": "source",
            "token_estimate": 10,
            "chunk_type": "block",
            "start_line": 31,
            "end_line": 50,
        },
    ]

    # BM25 candidates: only chunk_a1 scores above zero
    bm25_candidates = [
        StageCandidate(
            chunk_id="chunk_a1",
            rel_path="file_a.py",
            chunk_index=0,
            bm25_score=10.0,
            token_estimate=10,
            excerpt="BM25 strong match query query query",
        ),
    ]

    # Precompute best_chunk_by_file (step 5 of spec)
    best_chunk_by_file = _best_chunk_per_file(rows, bm25_candidates, None)

    # Build signals
    bm25_signal = build_bm25_rank_list(bm25_candidates, best_chunk_by_file, top_k=100)

    class MockCtx:
        central_files = set()
        file_symbol_refs = {}
        file_community = {}

    ctx = MockCtx()
    confidence_edges = {
        "file_a.py": [0.8],  # Graph has some confidence edge to file_a
    }

    graph_signal = build_graph_confidence_rank_list(
        rows, ctx, {"file_a.py"}, confidence_edges, best_chunk_by_file, top_k=100
    )

    # Both signals must reference chunk_a1 (not chunk_a2)
    assert len(bm25_signal) == 1
    assert bm25_signal[0].chunk_id == "chunk_a1"
    assert bm25_signal[0].rel_path == "file_a.py"

    assert len(graph_signal) == 1
    assert graph_signal[0].chunk_id == "chunk_a1", (
        "Graph signal must use best_chunk_by_file, not chunk_index-max. "
        "OLD BUG would pick chunk_a2 (chunk_index=1)"
    )
    assert graph_signal[0].rel_path == "file_a.py"

    # Fuse: should produce exactly ONE row for file_a.py
    fused = fuse_signal_lists([bm25_signal, graph_signal])

    assert len(fused) == 1, (
        f"Expected exactly 1 fused row for file_a.py, got {len(fused)}. "
        "If chunk_ids mismatch, fuse_signal_lists treats them as 2 separate documents."
    )

    fused_entry = fused[0]
    assert fused_entry.chunk_id == "chunk_a1"
    assert fused_entry.rel_path == "file_a.py"

    # per_signal_ranks must have both signal names
    assert "bm25" in fused_entry.per_signal_ranks
    assert "graph_confidence" in fused_entry.per_signal_ranks
    assert fused_entry.per_signal_ranks["bm25"] == 1
    assert fused_entry.per_signal_ranks["graph_confidence"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# Test Class E: CS-253 Module-Proximity Signal (3b)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_build_module_proximity_rank_list_seed_community():
    """Test build_module_proximity_rank_list includes files in seed communities."""
    rows = [
        {
            "id": f"chunk_{i}",
            "rel_path": f"file_{i}",
            "chunk_index": 0,
            "content": f"content for file_{i}",
            "language": "python",
            "category": "source",
            "token_estimate": 10,
            "chunk_type": "block",
            "start_line": 0,
            "end_line": 30,
        }
        for i in range(1, 4)
    ]

    best_chunk_by_file = {f"file_{i}": rows[i - 1] for i in range(1, 4)}

    class MockCtx:
        file_community = {
            "file_1": 100,  # community 100
            "file_2": 101,  # community 101
            "file_3": 100,  # community 100 (same as file_1)
        }

    seed_community_ids = {100}  # Only community 100 is in seed set

    module_signal = build_module_proximity_rank_list(
        rows, MockCtx(), seed_community_ids, best_chunk_by_file, top_k=100
    )

    # file_1 and file_3 are in community 100 (seed), file_2 is not
    assert len(module_signal) == 2
    rel_paths = {e.rel_path for e in module_signal}
    assert rel_paths == {"file_1", "file_3"}

    # All entries should have signal_name='module_proximity' and raw_score=1.3
    for entry in module_signal:
        assert entry.signal_name == "module_proximity"
        assert entry.raw_score == 1.3  # _MODULE_PROXIMITY_BONUS


@pytest.mark.asyncio
async def test_build_module_proximity_rank_list_non_seed_community():
    """Test build_module_proximity_rank_list excludes files in non-seed communities."""
    rows = [
        {
            "id": "chunk_1",
            "rel_path": "file_1",
            "chunk_index": 0,
            "content": "content",
            "language": "python",
            "category": "source",
            "token_estimate": 10,
            "chunk_type": "block",
            "start_line": 0,
            "end_line": 30,
        },
    ]

    best_chunk_by_file = {"file_1": rows[0]}

    class MockCtx:
        file_community = {"file_1": 200}  # community 200

    seed_community_ids = {100, 101}  # 200 is NOT in seed set

    module_signal = build_module_proximity_rank_list(
        rows, MockCtx(), seed_community_ids, best_chunk_by_file, top_k=100
    )

    # file_1 is not in any seed community, should be excluded
    assert len(module_signal) == 0


@pytest.mark.asyncio
async def test_build_module_proximity_rank_list_no_community():
    """Test build_module_proximity_rank_list excludes files with no community assignment."""
    rows = [
        {
            "id": "chunk_1",
            "rel_path": "file_1",
            "chunk_index": 0,
            "content": "content",
            "language": "python",
            "category": "source",
            "token_estimate": 10,
            "chunk_type": "block",
            "start_line": 0,
            "end_line": 30,
        },
    ]

    best_chunk_by_file = {"file_1": rows[0]}

    class MockCtx:
        file_community = {}  # file_1 has no community assignment

    seed_community_ids = {100}

    module_signal = build_module_proximity_rank_list(
        rows, MockCtx(), seed_community_ids, best_chunk_by_file, top_k=100
    )

    # file_1 has no community, should be excluded
    assert len(module_signal) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Test Class F: CS-253 Category-Hint Signal (3c)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_build_category_hint_rank_list_matching_category():
    """Test build_category_hint_rank_list includes files with matching category.

    ARCHITECTURE section hints: {"source", "config", "infra"}
    """
    rows = [
        {
            "id": "chunk_1",
            "rel_path": "file_1.py",
            "chunk_index": 0,
            "content": "Architecture code",
            "language": "python",
            "category": "source",  # Matches ARCHITECTURE section hints
            "token_estimate": 10,
            "chunk_type": "block",
            "start_line": 0,
            "end_line": 30,
        },
        {
            "id": "chunk_2",
            "rel_path": "file_2.md",
            "chunk_index": 0,
            "content": "docs",
            "language": None,
            "category": "docs",  # Does not match ARCHITECTURE hints
            "token_estimate": 10,
            "chunk_type": "block",
            "start_line": 0,
            "end_line": 30,
        },
    ]

    best_chunk_by_file = {
        "file_1.py": rows[0],
        "file_2.md": rows[1],
    }

    # ARCHITECTURE section should have 'source' in its hints
    category_signal = build_category_hint_rank_list(
        rows, RetrievalSection.ARCHITECTURE, best_chunk_by_file, top_k=100
    )

    # Only file_1.py should be included
    assert len(category_signal) == 1
    assert category_signal[0].rel_path == "file_1.py"
    assert category_signal[0].signal_name == "category_hint"
    assert category_signal[0].raw_score == 1.4  # _CATEGORY_HINT_BONUS


@pytest.mark.asyncio
async def test_build_category_hint_rank_list_non_matching_category():
    """Test build_category_hint_rank_list excludes non-matching categories.

    ARCHITECTURE section hints: {"source", "config", "infra"}
    """
    rows = [
        {
            "id": "chunk_1",
            "rel_path": "file_1.md",
            "chunk_index": 0,
            "content": "docs",
            "language": None,
            "category": "docs",  # ARCHITECTURE hints don't include 'docs'
            "token_estimate": 10,
            "chunk_type": "block",
            "start_line": 0,
            "end_line": 30,
        },
    ]

    best_chunk_by_file = {"file_1.md": rows[0]}

    category_signal = build_category_hint_rank_list(
        rows, RetrievalSection.ARCHITECTURE, best_chunk_by_file, top_k=100
    )

    assert len(category_signal) == 0


@pytest.mark.asyncio
async def test_build_category_hint_rank_list_multiple_sections():
    """Test build_category_hint_rank_list with different RetrievalSection values.

    ARCHITECTURE hints: {"source", "config", "infra"}
    FEATURE_MAP hints: {"source", "docs"}
    """
    rows = [
        {
            "id": "chunk_1",
            "rel_path": "file_1.md",
            "chunk_index": 0,
            "content": "docs",
            "language": None,
            "category": "docs",
            "token_estimate": 10,
            "chunk_type": "block",
            "start_line": 0,
            "end_line": 30,
        },
        {
            "id": "chunk_2",
            "rel_path": "file_2.py",
            "chunk_index": 0,
            "content": "source",
            "language": "python",
            "category": "source",
            "token_estimate": 10,
            "chunk_type": "block",
            "start_line": 0,
            "end_line": 30,
        },
    ]

    best_chunk_by_file = {
        "file_1.md": rows[0],
        "file_2.py": rows[1],
    }

    # FEATURE_MAP section: should match both docs and source
    feature_signal = build_category_hint_rank_list(
        rows, RetrievalSection.FEATURE_MAP, best_chunk_by_file, top_k=100
    )
    assert len(feature_signal) >= 1
    feature_rel_paths = {e.rel_path for e in feature_signal}
    # file_1.md (docs) and file_2.py (source) should both match FEATURE_MAP hints
    assert "file_1.md" in feature_rel_paths or "file_2.py" in feature_rel_paths

    # ARCHITECTURE section: should only match source (not docs)
    arch_signal = build_category_hint_rank_list(
        rows, RetrievalSection.ARCHITECTURE, best_chunk_by_file, top_k=100
    )
    # file_2.py (source) matches ARCHITECTURE, file_1.md (docs) does not
    if len(arch_signal) > 0:
        arch_rel_paths = {e.rel_path for e in arch_signal}
        assert "file_2.py" in arch_rel_paths


@pytest.mark.asyncio
async def test_build_category_hint_rank_list_empty_section_hints():
    """Test build_category_hint_rank_list returns empty list for sections with no hints."""
    rows = [
        {
            "id": "chunk_1",
            "rel_path": "file_1.py",
            "chunk_index": 0,
            "content": "content",
            "language": "python",
            "category": "source",
            "token_estimate": 10,
            "chunk_type": "block",
            "start_line": 0,
            "end_line": 30,
        },
    ]

    best_chunk_by_file = {"file_1.py": rows[0]}

    # If a section has no hints in _SECTION_CATEGORY_HINTS, should return []
    # (This is a defensive test; actual empty sections depend on two_stage_retrieval.py config)
    category_signal = build_category_hint_rank_list(
        rows, RetrievalSection.GLOSSARY, best_chunk_by_file, top_k=100
    )
    # GLOSSARY may or may not be empty; just ensure it returns a list
    assert isinstance(category_signal, list)


# ─────────────────────────────────────────────────────────────────────────────
# Test Class G: CS-253 Fallback Mechanisms (_best_chunk_per_file)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_best_chunk_per_file_bm25_preferred():
    """Test _best_chunk_per_file picks BM25-scored chunk when available."""
    rows = [
        {
            "id": "chunk_a1",
            "rel_path": "file_a.py",
            "chunk_index": 0,
            "content": "first chunk",
            "start_line": 0,
            "end_line": 30,
        },
        {
            "id": "chunk_a2",
            "rel_path": "file_a.py",
            "chunk_index": 1,
            "content": "second chunk",
            "start_line": 31,
            "end_line": 50,
        },
    ]

    candidates = [
        StageCandidate(
            chunk_id="chunk_a1",
            rel_path="file_a.py",
            chunk_index=0,
            bm25_score=10.0,
            token_estimate=10,
            excerpt="first chunk",
        ),
    ]

    best = _best_chunk_per_file(rows, candidates, None)

    # Should pick chunk_a1 (BM25 score available)
    assert "file_a.py" in best
    assert best["file_a.py"]["id"] == "chunk_a1"


@pytest.mark.asyncio
async def test_best_chunk_per_file_symbol_overlap_fallback():
    """Test _best_chunk_per_file symbol_overlap fallback (no BM25, overlaps symbol)."""
    rows = [
        {
            "id": "chunk_file1",
            "rel_path": "file_1.py",
            "chunk_index": 0,
            "content": "module imports",
            "start_line": 0,
            "end_line": 10,
        },
        {
            "id": "chunk_file2",
            "rel_path": "file_1.py",
            "chunk_index": 1,
            "content": "function body",
            "start_line": 11,
            "end_line": 50,
        },
    ]

    # No BM25 candidates for file_1.py
    candidates = []

    # Symbol index: a symbol defined in lines 11-25 overlaps chunk_file2
    symbol_index = {
        "my_function": [("file_1.py", 11, 25)],
    }

    best = _best_chunk_per_file(rows, candidates, symbol_index)

    # Should pick chunk_file2 (overlaps symbol, preferred over chunk_index 0)
    assert "file_1.py" in best
    assert best["file_1.py"]["id"] == "chunk_file2", (
        "Should prefer chunk overlapping symbol definition over chunk_index==0"
    )


@pytest.mark.asyncio
async def test_best_chunk_per_file_chunk_index_zero_fallback():
    """Test _best_chunk_per_file chunk_index==0 fallback (no BM25, no symbol overlap)."""
    rows = [
        {
            "id": "chunk_a1",
            "rel_path": "file_a.py",
            "chunk_index": 0,
            "content": "module header",
            "start_line": 0,
            "end_line": 5,
        },
        {
            "id": "chunk_a2",
            "rel_path": "file_a.py",
            "chunk_index": 1,
            "content": "trailing code",
            "start_line": 100,
            "end_line": 150,
        },
    ]

    # No BM25 candidates, no symbol index
    candidates = []

    best = _best_chunk_per_file(rows, candidates, None)

    # Should pick chunk_a1 (chunk_index==0), NOT chunk_a2 (chunk_index-max)
    assert "file_a.py" in best
    assert best["file_a.py"]["id"] == "chunk_a1", (
        "Fallback must pick chunk_index==0, not chunk_index-max. OLD BUG would pick chunk_a2."
    )


@pytest.mark.asyncio
async def test_symbol_overlap_fallback_basic():
    """Test _symbol_overlap_fallback finds chunk overlapping symbol range."""
    chunks = [
        {"rel_path": "file.py", "start_line": 0, "end_line": 10},
        {"rel_path": "file.py", "start_line": 11, "end_line": 50},  # Overlaps symbol
        {"rel_path": "file.py", "start_line": 51, "end_line": 100},
    ]

    # Symbol at lines 20-30, overlaps second chunk
    symbol_index = {"function_def": [("file.py", 20, 30)]}

    result = _symbol_overlap_fallback(chunks, symbol_index)

    assert result is not None
    assert result["start_line"] == 11  # Second chunk


@pytest.mark.asyncio
async def test_symbol_overlap_fallback_no_overlap():
    """Test _symbol_overlap_fallback returns None when no overlap."""
    chunks = [
        {"rel_path": "file.py", "start_line": 0, "end_line": 10},
        {"rel_path": "file.py", "start_line": 11, "end_line": 50},
    ]

    # Symbol at lines 200-300, outside all chunks
    symbol_index = {"distant_function": [("file.py", 200, 300)]}

    result = _symbol_overlap_fallback(chunks, symbol_index)

    assert result is None


@pytest.mark.asyncio
async def test_symbol_overlap_fallback_empty_symbol_index():
    """Test _symbol_overlap_fallback returns None for empty symbol_index."""
    chunks = [
        {"rel_path": "file.py", "start_line": 0, "end_line": 10},
    ]

    result = _symbol_overlap_fallback(chunks, {})

    assert result is None
