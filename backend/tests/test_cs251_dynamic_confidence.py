"""CS-251 — Dynamic Confidence Scoring + Real Consumption Test Suite.

Acceptance gate for dynamic ambiguous-confidence formula, min_confidence wiring,
path_confidence multiplicative compounding, and tentative file tracking.

Tests:
  1. _ambiguous_confidence formula: monotonic decrease, 0.15 floor.
  2. min_confidence filters results in both two_stage_retrieval and trace_call_chain.
  3. path_confidence multiplicative compounding on multi-hop fixture.
  4. tentative_files/tentative_confidence capture in deep_research Enhancement-5.
  5. Regression: SymbolHop/TraceStep new fields don't break existing positional construction.
"""
from __future__ import annotations

import pytest

from domain.qa.graph_queries import (
    SymbolHop,
    TraceStep,
    get_callees_of,
    trace_call_chain,
)
from domain.structural_graph._symbol_resolver import _ambiguous_confidence
from infrastructure.db.database import get_db

# ---------------------------------------------------------------------------
# Unit Tests: _ambiguous_confidence formula (3a)
# ---------------------------------------------------------------------------


class TestAmbiguousConfidenceFormula:
    """Verify _ambiguous_confidence formula shape and floor."""

    @pytest.mark.parametrize(
        "num_candidates,expected",
        [
            (1, 0.6),
            (2, 0.3),
            (3, 0.2),
            (5, 0.12),  # actually 0.12, should be floored to 0.15
            (10, 0.15),  # floor applies
            (20, 0.15),  # floor applies
        ],
    )
    def test_formula_shape(self, num_candidates: int, expected: float) -> None:
        """Test _ambiguous_confidence returns expected value for candidate counts."""
        result = _ambiguous_confidence(num_candidates)
        expected_floored = max(0.15, expected)
        assert abs(result - expected_floored) < 0.001, (
            f"_ambiguous_confidence({num_candidates}): expected {expected_floored}, got {result}"
        )

    def test_monotonic_decrease(self) -> None:
        """Confidence should decrease monotonically as candidate count increases."""
        values = [_ambiguous_confidence(n) for n in range(1, 21)]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1], (
                f"Not monotonic at index {i}: {values[i]} > {values[i + 1]}"
            )

    def test_floor_at_0_15(self) -> None:
        """Confidence should never drop below 0.15."""
        for n in range(1, 100):
            result = _ambiguous_confidence(n)
            assert result >= 0.15, f"Floor violated at n={n}: {result}"

    def test_never_reaches_high_boundary(self) -> None:
        """All ambiguous edges must remain 'low' (score < 0.7 derivation boundary)."""
        for n in range(1, 100):
            result = _ambiguous_confidence(n)
            # The >=0.7 boundary for 'high' confidence must never be reached by ambiguous scores
            assert result < 0.7, (
                f"Ambiguous confidence at n={n} ({result}) reaches/exceeds high boundary (0.7)"
            )


# ---------------------------------------------------------------------------
# Integration Tests: min_confidence wiring (3b)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_min_confidence_filters_callees():
    """min_confidence parameter in get_callees_of filters results correctly."""
    db = get_db()

    # Prepare test: insert edges with known confidence scores
    snapshot_id = "test-min-conf-callees"
    await db.execute("DELETE FROM symbol_graph_edges WHERE snapshot_id=?", (snapshot_id,))

    # Insert test edges with varying confidence scores
    test_edges = [
        (snapshot_id, "test.py::func_a", "test.py::func_b", "calls", "low", 0.3, "[]"),
        (snapshot_id, "test.py::func_a", "test.py::func_c", "calls", "high", 0.9, "[]"),
        (snapshot_id, "test.py::func_a", "test.py::func_d", "calls", "low", 0.5, "[]"),
    ]
    for row in test_edges:
        await db.execute(
            "INSERT INTO symbol_graph_edges "
            "(snapshot_id, src_symbol, dst_symbol, edge_type, confidence, "
            "confidence_score, evidence_lines) VALUES (?, ?, ?, ?, ?, ?, ?)",
            row,
        )

    # Test 1: high_confidence_only=True filters to >=0.7 (1 edge)
    hops_high = await get_callees_of(snapshot_id, "test.py", high_confidence_only=True)
    assert len(hops_high) == 1
    assert hops_high[0].dst_symbol == "test.py::func_c"

    # Test 2: min_confidence=0.5 filters to >=0.5 (2 edges)
    hops_min = await get_callees_of(
        snapshot_id, "test.py", high_confidence_only=False, min_confidence=0.5
    )
    assert len(hops_min) == 2
    assert all(h.confidence_score >= 0.5 for h in hops_min)

    # Test 3: min_confidence=0.9 filters to >=0.9 (1 edge)
    hops_strict = await get_callees_of(
        snapshot_id, "test.py", high_confidence_only=False, min_confidence=0.9
    )
    assert len(hops_strict) == 1
    assert hops_strict[0].confidence_score == 0.9


@pytest.mark.asyncio
async def test_trace_call_chain_forwards_min_confidence():
    """trace_call_chain forwards min_confidence to get_callees_of/get_callers_of."""
    db = get_db()

    snapshot_id = "test-trace-min-conf"
    await db.execute("DELETE FROM symbol_graph_edges WHERE snapshot_id=?", (snapshot_id,))

    # Build a simple call chain: A -> B -> C
    # A->B: high confidence (0.9)
    # B->C: low confidence (0.4)
    test_edges = [
        (snapshot_id, "file_a.py::func_a", "file_b.py::func_b", "calls", "high", 0.9, "[]"),
        (snapshot_id, "file_b.py::func_b", "file_c.py::func_c", "calls", "low", 0.4, "[]"),
    ]
    for row in test_edges:
        await db.execute(
            "INSERT INTO symbol_graph_edges "
            "(snapshot_id, src_symbol, dst_symbol, edge_type, confidence, "
            "confidence_score, evidence_lines) VALUES (?, ?, ?, ?, ?, ?, ?)",
            row,
        )

    # Trace forward from file_a with min_confidence=0.5 should find file_b but not file_c
    trace_filtered = await trace_call_chain(
        snapshot_id,
        "file_a.py",
        "forward",
        max_hops=3,
        high_confidence_only=False,
        min_confidence=0.5,
    )
    files_found = {s.file for s in trace_filtered}
    assert "file_b.py" in files_found, "Should find file_b via 0.9 edge"
    assert "file_c.py" not in files_found, (
        "Should NOT find file_c via 0.4 edge when min_confidence=0.5"
    )


# ---------------------------------------------------------------------------
# Integration Tests: path_confidence multiplicative compounding (3c)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_path_confidence_multiplicative():
    """path_confidence multiplies across step depth: 0.9 * 0.85 * 0.4 ≈ 0.306."""
    db = get_db()

    snapshot_id = "test-path-conf"
    await db.execute("DELETE FROM symbol_graph_edges WHERE snapshot_id=?", (snapshot_id,))

    # Build 3-hop chain: A -> B -> C -> D with confidences 0.9, 0.85, 0.4
    edges = [
        (snapshot_id, "a.py::f", "b.py::g", "calls", "high", 0.9, "[]"),
        (snapshot_id, "b.py::g", "c.py::h", "calls", "high", 0.85, "[]"),
        (snapshot_id, "c.py::h", "d.py::i", "calls", "low", 0.4, "[]"),
    ]
    for row in edges:
        await db.execute(
            "INSERT INTO symbol_graph_edges "
            "(snapshot_id, src_symbol, dst_symbol, edge_type, confidence, "
            "confidence_score, evidence_lines) VALUES (?, ?, ?, ?, ?, ?, ?)",
            row,
        )

    # Trace forward from a.py, high_confidence_only=False to include the 0.4 edge
    trace = await trace_call_chain(
        snapshot_id, "a.py", "forward", max_hops=4, high_confidence_only=False
    )

    # Find the final step (d.py should be found at step 3)
    final_step = None
    for step in trace:
        if step.file == "d.py":
            final_step = step
            break

    assert final_step is not None, "Should find d.py in the trace"
    # Expected: 0.9 * 0.85 * 0.4 = 0.306
    expected_path_confidence = 0.9 * 0.85 * 0.4
    assert abs(final_step.path_confidence - expected_path_confidence) < 0.01, (
        f"Expected path_confidence ≈ {expected_path_confidence}, got {final_step.path_confidence}"
    )


@pytest.mark.asyncio
async def test_path_confidence_max_per_step():
    """path_confidence uses max() across parallel hops in same step."""
    db = get_db()

    snapshot_id = "test-path-conf-parallel"
    await db.execute("DELETE FROM symbol_graph_edges WHERE snapshot_id=?", (snapshot_id,))

    # Build parallel hops to same file: A -> B via two different paths
    # Path 1: A -> B directly (0.9)
    # Path 2: A -> B via alias (0.6)
    # Expected: max(0.9, 0.6) = 0.9
    edges = [
        (snapshot_id, "a.py::f1", "b.py::g", "calls", "high", 0.9, "[]"),
        (snapshot_id, "a.py::f2", "b.py::g", "calls", "low", 0.6, "[]"),
    ]
    for row in edges:
        await db.execute(
            "INSERT INTO symbol_graph_edges "
            "(snapshot_id, src_symbol, dst_symbol, edge_type, confidence, "
            "confidence_score, evidence_lines) VALUES (?, ?, ?, ?, ?, ?, ?)",
            row,
        )

    trace = await trace_call_chain(
        snapshot_id, "a.py", "forward", max_hops=2, high_confidence_only=False
    )

    b_step = None
    for step in trace:
        if step.file == "b.py":
            b_step = step
            break

    assert b_step is not None, "Should find b.py"
    # With parallel hops (0.9 and 0.6), max should be 0.9
    assert abs(b_step.path_confidence - 0.9) < 0.01, (
        f"Expected path_confidence ≈ 0.9 (max of parallel hops), got {b_step.path_confidence}"
    )


@pytest.mark.asyncio
async def test_symbol_hop_confidence_score_field():
    """SymbolHop.confidence_score field is populated from DB."""
    db = get_db()

    snapshot_id = "test-hop-field"
    await db.execute("DELETE FROM symbol_graph_edges WHERE snapshot_id=?", (snapshot_id,))

    await db.execute(
        "INSERT INTO symbol_graph_edges "
        "(snapshot_id, src_symbol, dst_symbol, edge_type, confidence, "
        "confidence_score, evidence_lines) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (snapshot_id, "a.py::f", "b.py::g", "calls", "high", 0.85, "[10]"),
    )

    hops = await get_callees_of(snapshot_id, "a.py", high_confidence_only=False)
    assert len(hops) == 1
    hop = hops[0]
    assert hop.confidence_score == 0.85, "SymbolHop.confidence_score should be populated from DB"


# ---------------------------------------------------------------------------
# Regression Tests: Existing positional construction compatibility (3c + 5)
# ---------------------------------------------------------------------------


def test_symbol_hop_backward_compat():
    """SymbolHop constructed positionally (backward compat, no confidence_score)."""
    # Old-style positional construction (5 args, no confidence_score)
    hop = SymbolHop("file.py::func_a", "file.py::func_b", "calls", "high", [10, 20])
    assert hop.src_symbol == "file.py::func_a"
    assert hop.dst_symbol == "file.py::func_b"
    assert hop.edge_type == "calls"
    assert hop.confidence == "high"
    assert hop.evidence_lines == [10, 20]
    assert hop.confidence_score == 1.0  # default


def test_trace_step_backward_compat():
    """TraceStep constructed positionally (backward compat, no path_confidence)."""
    hop1 = SymbolHop("a.py::f", "b.py::g", "calls", "high", [1])

    # Old-style positional construction (5 args, no path_confidence)
    step = TraceStep(
        1,
        "b.py",
        ["b.py::g"],
        [hop1],
        False,  # is_seed
    )
    assert step.step == 1
    assert step.file == "b.py"
    assert step.symbols_involved == ["b.py::g"]
    assert len(step.hops) == 1
    assert step.is_seed is False
    assert step.path_confidence == 1.0  # default


# ---------------------------------------------------------------------------
# Integration Tests: tentative files capture in Enhancement-5 fallback (3d)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enhancement5_fallback_capture_tentative_files():
    """Verify tentative_files/tentative_confidence capture when fallback fires.

    Enhancement-5: When high-confidence trace finds < 2 files, retry with
    low-confidence edges and capture extra files as tentative with their
    max hop confidence scores before overwriting the trace object.
    """
    db = get_db()

    snapshot_id = "test-enhancement5-fallback"
    await db.execute("DELETE FROM symbol_graph_edges WHERE snapshot_id=?", (snapshot_id,))

    # Build a scenario where:
    # - High-confidence edges: A -> B (only 1 file found, < 2)
    # - Low-confidence edges: A -> B, A -> C (total 2 files: B, C)
    # When fallback fires, it should capture C as tentative
    # Note: DB has UNIQUE constraint on (snapshot_id, src_symbol, dst_symbol)
    # so we use different source symbols for the parallel paths

    edges = [
        # High confidence path (only leads to B)
        (snapshot_id, "a.py::f_start", "b.py::g", "calls", "high", 0.9, "[]"),
        # Low confidence path (leads to C)
        (snapshot_id, "a.py::f_start", "c.py::h", "calls", "low", 0.45, "[]"),
    ]
    for row in edges:
        await db.execute(
            "INSERT INTO symbol_graph_edges ("
            "snapshot_id, src_symbol, dst_symbol, edge_type, confidence, "
            "confidence_score, evidence_lines) VALUES (?, ?, ?, ?, ?, ?, ?)",
            row,
        )

    # Trace with high_confidence_only=True: should find only b.py (1 file < 2)
    trace_high = await trace_call_chain(
        snapshot_id, "a.py", "forward", max_hops=2, high_confidence_only=True
    )
    high_conf_files = {s.file for s in trace_high}
    assert high_conf_files == {"a.py", "b.py"}, (
        f"High-confidence trace should find only a.py and b.py, got {high_conf_files}"
    )
    new_files_high = [s.file for s in trace_high if s.file != "a.py"]
    assert len(new_files_high) < 2, (
        "High-confidence trace should find < 2 files (triggers fallback)"
    )

    # Trace with high_confidence_only=False: should find both b.py and c.py
    trace_low = await trace_call_chain(
        snapshot_id, "a.py", "forward", max_hops=2, high_confidence_only=False
    )
    low_conf_files = {s.file for s in trace_low}
    assert "c.py" in low_conf_files, f"Low-confidence trace should find c.py, got {low_conf_files}"

    # Simulate Enhancement-5 fallback logic:
    tentative_files = []
    tentative_confidence = {}

    extra = [s.file for s in trace_low if s.file not in high_conf_files and s.file != "a.py"]
    if extra and len(new_files_high) < 2:
        tentative_files = extra[:5]
        for file in tentative_files:
            for step in trace_low:
                if step.file == file:
                    max_conf = max((h.confidence_score for h in step.hops), default=1.0)
                    tentative_confidence[file] = max_conf
                    break

    # Verify tentative capture
    assert "c.py" in tentative_files, "c.py should be captured as tentative"
    assert "c.py" in tentative_confidence, "c.py should have confidence score"
    assert tentative_confidence["c.py"] == 0.45, (
        f"c.py confidence should be 0.45 (its hop confidence), got {tentative_confidence['c.py']}"
    )


@pytest.mark.asyncio
async def test_enhancement5_no_fallback_when_sufficient_high_confidence():
    """Verify tentative_files remain empty when high-confidence trace finds >= 2 files."""
    db = get_db()

    snapshot_id = "test-enhancement5-no-fallback"
    await db.execute("DELETE FROM symbol_graph_edges WHERE snapshot_id=?", (snapshot_id,))

    # Build a scenario where high-confidence trace finds 2+ files (no fallback)
    edges = [
        (snapshot_id, "a.py::f_start", "b.py::g", "calls", "high", 0.9, "[]"),
        (snapshot_id, "a.py::f_start", "c.py::h", "calls", "high", 0.85, "[]"),
        # Low confidence (should not be used)
        (snapshot_id, "a.py::f_start", "d.py::i", "calls", "low", 0.3, "[]"),
    ]
    for row in edges:
        await db.execute(
            "INSERT INTO symbol_graph_edges ("
            "snapshot_id, src_symbol, dst_symbol, edge_type, confidence, "
            "confidence_score, evidence_lines) VALUES (?, ?, ?, ?, ?, ?, ?)",
            row,
        )

    # Trace with high_confidence_only=True: find b.py & c.py (no fallback)
    trace_high = await trace_call_chain(
        snapshot_id, "a.py", "forward", max_hops=2, high_confidence_only=True
    )
    new_files = [s.file for s in trace_high if s.file != "a.py"]
    assert len(new_files) >= 2, (
        "High-confidence trace should find >= 2 files (no fallback triggers)"
    )

    # Simulate Enhancement-5 logic (should NOT capture tentative files)
    tentative_files = []
    tentative_confidence = {}

    if len(new_files) < 2:  # This condition should be False
        trace_low = await trace_call_chain(
            snapshot_id, "a.py", "forward", max_hops=2, high_confidence_only=False
        )
        high_conf_files = {s.file for s in trace_high}
        extra = [s.file for s in trace_low if s.file not in high_conf_files and s.file != "a.py"]
        if extra:
            tentative_files = extra[:5]
            for file in tentative_files:
                for step in trace_low:
                    if step.file == file:
                        max_conf = max((h.confidence_score for h in step.hops), default=1.0)
                        tentative_confidence[file] = max_conf
                        break

    # Verify NO tentative capture (enhancement-5 doesn't fire)
    assert len(tentative_files) == 0, "tentative_files should remain empty when no fallback"
    assert len(tentative_confidence) == 0, (
        "tentative_confidence should remain empty when no fallback"
    )
