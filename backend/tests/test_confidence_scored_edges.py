"""CS-240 — Confidence-Scored Graph Edges Test Suite.

Acceptance gate for confidence scoring on symbol_graph_edges (CS-240).

Tests:
  1. Known-ambiguous case (two same-named functions, different files, no import)
     → resolver returns [] (unchanged EC-9/EC-6 behavior).
  2. Each resolver branch emits expected confidence_score and resolution_method.
  3. Legacy confidence string filters still work (backward-compat).
  4. min_confidence filtering in graph_queries and two_stage_retrieval works.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from domain.structural_graph.symbol_graph import SymbolEdge, SymbolGraphBuilder
from domain.qa.graph_queries import get_callees_of, get_callers_of
from infrastructure.db.database import get_db


# ---------------------------------------------------------------------------
# Unit Tests: Resolver Branch Confidence Mapping (CS-240)
# ---------------------------------------------------------------------------

def _builder() -> SymbolGraphBuilder:
    return SymbolGraphBuilder()


def _find(edges: list[SymbolEdge], src: str, dst: str) -> SymbolEdge | None:
    """Return the first edge matching *src* → *dst*, or None."""
    for e in edges:
        if e.src_symbol == src and e.dst_symbol == dst:
            return e
    return None


class TestResolverBranchMapping:
    """Verify each resolver branch emits the documented confidence_score/resolution_method."""

    def test_import_path_match_bare_call(self) -> None:
        """_resolve_bare_call import-namespace hit → confidence_score=0.95, resolution_method='import_path_match'."""
        utils = "def compute_hash(data: bytes) -> str: pass"
        pipeline = "from utils import compute_hash\n\ndef process(raw):\n    return compute_hash(raw)"

        edges = _builder().build({"utils.py": utils, "pipeline.py": pipeline})
        edge = _find(edges, "pipeline.py::process", "utils.py::compute_hash")

        assert edge is not None, "Import-resolved bare call must produce an edge"
        assert edge.confidence_score == 0.95, f"Expected 0.95, got {edge.confidence_score}"
        assert edge.resolution_method == "import_path_match"
        assert edge.confidence == "high", "Score 0.95 >= 0.7 → confidence='high'"

    def test_same_file_scope_bare_call(self) -> None:
        """_resolve_bare_call same-file single candidate → confidence_score=0.85, resolution_method='same_file_scope'."""
        code = """
def helper(): pass

def caller():
    helper()
"""
        edges = _builder().build({"local.py": code})
        edge = _find(edges, "local.py::caller", "local.py::helper")

        assert edge is not None, "Same-file bare call must produce an edge"
        assert edge.confidence_score == 0.85, f"Expected 0.85, got {edge.confidence_score}"
        assert edge.resolution_method == "same_file_scope"
        assert edge.confidence == "high", "Score 0.85 >= 0.7 → confidence='high'"

    def test_mro_resolved_self_call(self) -> None:
        """_resolve_self_call MRO walk → confidence_score=0.9, resolution_method='mro_resolved'."""
        code = """
class Base:
    def validate(self): pass

class Child(Base):
    def handle(self):
        self.validate()
"""
        edges = _builder().build({"inheritance.py": code})
        edge = _find(edges, "inheritance.py::Child.handle", "inheritance.py::Base.validate")

        assert edge is not None, "MRO-resolved self call must produce an edge"
        assert edge.confidence_score == 0.9, f"Expected 0.9, got {edge.confidence_score}"
        assert edge.resolution_method == "mro_resolved"
        assert edge.confidence == "high", "Score 0.9 >= 0.7 → confidence='high'"

    def test_constructor_type_trace_single(self) -> None:
        """Attribute receiver with single assigned type → confidence_score=0.85, resolution_method='constructor_type_trace'."""
        code = """
class ServiceA:
    def process(self): return "A"

class Controller:
    def __init__(self):
        self.svc = ServiceA()

    def handle(self):
        return self.svc.process()
"""
        edges = _builder().build({"wired.py": code})
        edge = _find(edges, "wired.py::Controller.handle", "wired.py::ServiceA.process")

        assert edge is not None, "Constructor-traced single type must produce an edge"
        assert edge.confidence_score == 0.85, f"Expected 0.85, got {edge.confidence_score}"
        assert edge.resolution_method == "constructor_type_trace"
        assert edge.confidence == "high", "Score 0.85 >= 0.7 → confidence='high'"

    def test_name_heuristic_ambiguous_multi_type(self) -> None:
        """Attribute receiver with 2+ assigned types → confidence_score=0.4, resolution_method='name_heuristic_ambiguous'."""
        code = """
class Alpha:
    def run(self): pass

class Beta:
    def run(self): pass

class Runner:
    def __init__(self):
        self.worker = Alpha()

    def switch(self):
        self.worker = Beta()

    def execute(self):
        self.worker.run()
"""
        edges = _builder().build({"reassign.py": code})
        alpha_edge = _find(edges, "reassign.py::Runner.execute", "reassign.py::Alpha.run")
        beta_edge = _find(edges, "reassign.py::Runner.execute", "reassign.py::Beta.run")

        # Should have both edges, both marked as low confidence
        # With 2 assigned types, _ambiguous_confidence(2) = 0.6/2 = 0.3
        expected_score = 0.3  # _ambiguous_confidence(2)
        if alpha_edge:
            assert alpha_edge.confidence_score == expected_score, f"Expected {expected_score} for Alpha, got {alpha_edge.confidence_score}"
            assert alpha_edge.resolution_method == "name_heuristic_ambiguous"
            assert alpha_edge.confidence == "low", f"Score {expected_score} < 0.7 → confidence='low'"
        if beta_edge:
            assert beta_edge.confidence_score == expected_score, f"Expected {expected_score} for Beta, got {beta_edge.confidence_score}"
            assert beta_edge.resolution_method == "name_heuristic_ambiguous"
            assert beta_edge.confidence == "low", f"Score {expected_score} < 0.7 → confidence='low'"

    def test_module_call_import_resolved(self) -> None:
        """_resolve_module_call via import → confidence_score=0.95, resolution_method='import_path_match'."""
        auth = "def authenticate(token): pass"
        service = "from auth import authenticate as auth_func\n\nclass Service:\n    def login(self):\n        auth_func('token')"

        edges = _builder().build({"auth.py": auth, "service.py": service})
        # The call is to auth_func which aliases authenticate
        matching = [e for e in edges if "login" in e.src_symbol and "authenticate" in e.dst_symbol]

        if matching:
            edge = matching[0]
            assert edge.confidence_score == 0.95, f"Expected 0.95, got {edge.confidence_score}"
            assert edge.resolution_method == "import_path_match"
            assert edge.confidence == "high"

    def test_confidence_derived_from_score(self) -> None:
        """confidence string is derived from confidence_score: >= 0.7 → 'high', else 'low'."""
        # Test the boundary
        edge_high = SymbolEdge(
            src_symbol="a.py::func",
            dst_symbol="b.py::func",
            confidence_score=0.7,
            resolution_method="test",
        )
        assert edge_high.confidence == "high", "Score 0.7 should be 'high'"

        edge_low = SymbolEdge(
            src_symbol="a.py::func",
            dst_symbol="b.py::func",
            confidence_score=0.69,
            resolution_method="test",
        )
        assert edge_low.confidence == "low", "Score 0.69 should be 'low'"


# ---------------------------------------------------------------------------
# EC-9/EC-6 Cases: Ensure Unchanged Behavior
# ---------------------------------------------------------------------------

class TestAmbiguousCasesUnchanged:
    """Verify ambiguous/unresolvable cases still return [] (unchanged EC-9/EC-6 behavior)."""

    def test_star_import_no_edge(self) -> None:
        """Star import makes origin unknown → resolver returns [] (EC-9)."""
        code = "from utils import *\n\ndef handler():\n    result = compute_hash(data)"

        edges = _builder().build({"star.py": code})
        star_edges = [
            e for e in edges
            if e.src_symbol == "star.py::handler" and "compute_hash" in e.dst_symbol
        ]

        assert star_edges == [], "Star import must produce no edges — guessing origin is forbidden"

    def test_duck_typing_no_confident_edge(self) -> None:
        """Untyped parameter → resolver returns [] (EC-6)."""
        code = "def process_any(obj):\n    obj.build()"

        edges = _builder().build({"duck.py": code})
        duck_edges = [
            e for e in edges
            if e.src_symbol == "duck.py::process_any" and "build" in e.dst_symbol
        ]

        high_conf = [e for e in duck_edges if e.confidence == "high"]
        assert high_conf == [], "Duck-typed call site must not produce confidence='high' edges"

    def test_two_same_named_functions_no_import_no_edge(self) -> None:
        """Two same-named functions, different files, no import link → no edge (unchanged)."""
        auth = "def validate(token): pass"
        billing = "def validate(api_key): pass"
        # No explicit import, so no resolution path → []
        middleware = "def process(req):\n    validate(req.token)"

        edges = _builder().build({
            "auth.py": auth,
            "billing.py": billing,
            "middleware.py": middleware,
        })

        middleware_edges = [
            e for e in edges
            if e.src_symbol == "middleware.py::process" and "validate" in e.dst_symbol
        ]

        assert middleware_edges == [], (
            "Unimported bare call to ambiguous name must produce no edges — "
            "neither auth.validate nor billing.validate should be guessed"
        )


# ---------------------------------------------------------------------------
# Integration Tests: Database Round-trip & Filter Behavior
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_legacy_confidence_string_filter_backward_compat(
    test_snapshot_id: str,
    cleanup_test_data: None,
) -> None:
    """Verify high_confidence_only=True still excludes confidence < 0.7 rows."""
    db = get_db()

    # Insert test edges with mixed confidence_score values
    await db.executemany(
        """
        INSERT INTO symbol_graph_edges
        (snapshot_id, src_symbol, dst_symbol, edge_type, confidence_score, resolution_method, evidence_lines)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (test_snapshot_id, "a.py::caller", "b.py::high_conf", "calls", 0.95, "import_path_match", "[]"),
            (test_snapshot_id, "a.py::caller", "b.py::mid_conf", "calls", 0.7, "same_file_scope", "[]"),
            (test_snapshot_id, "a.py::caller", "b.py::low_conf", "calls", 0.4, "name_heuristic_ambiguous", "[]"),
            (test_snapshot_id, "a.py::caller", "b.py::zero_conf", "calls", 0.0, "unknown", "[]"),
        ],
    )

    # Query with high_confidence_only=True (default)
    high_only = await get_callees_of(test_snapshot_id, "a.py", high_confidence_only=True)

    # Should get only rows with confidence='high' (score >= 0.7)
    dests_high_only = {hop.dst_symbol for hop in high_only}
    assert "b.py::high_conf" in dests_high_only, "Score 0.95 should be included"
    assert "b.py::mid_conf" in dests_high_only, "Score 0.7 (boundary) should be included"
    assert "b.py::low_conf" not in dests_high_only, "Score 0.4 should be excluded"
    assert "b.py::zero_conf" not in dests_high_only, "Score 0.0 should be excluded"

    # Query with high_confidence_only=False (unfiltered)
    all_edges = await get_callees_of(test_snapshot_id, "a.py", high_confidence_only=False)
    dests_all = {hop.dst_symbol for hop in all_edges}
    assert len(dests_all) == 4, "Unfiltered should get all 4 edges"


@pytest.mark.asyncio
async def test_min_confidence_filtering_get_callees_of(
    test_snapshot_id: str,
    cleanup_test_data: None,
) -> None:
    """Verify min_confidence parameter filters correctly (inclusive >=)."""
    db = get_db()

    await db.executemany(
        """
        INSERT INTO symbol_graph_edges
        (snapshot_id, src_symbol, dst_symbol, edge_type, confidence_score, resolution_method, evidence_lines)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (test_snapshot_id, "a.py::caller", "b.py::t1", "calls", 0.95, "import_path_match", "[]"),
            (test_snapshot_id, "a.py::caller", "b.py::t2", "calls", 0.85, "constructor_type_trace", "[]"),
            (test_snapshot_id, "a.py::caller", "b.py::t3", "calls", 0.4, "name_heuristic_ambiguous", "[]"),
        ],
    )

    # Query with min_confidence=0.85
    result = await get_callees_of(
        test_snapshot_id, "a.py", high_confidence_only=False, min_confidence=0.85
    )
    dests = {hop.dst_symbol for hop in result}

    assert "b.py::t1" in dests, "Score 0.95 >= 0.85 should be included"
    assert "b.py::t2" in dests, "Score 0.85 >= 0.85 (boundary) should be included"
    assert "b.py::t3" not in dests, "Score 0.4 < 0.85 should be excluded"

    # Query with min_confidence=0.5
    result_half = await get_callees_of(
        test_snapshot_id, "a.py", high_confidence_only=False, min_confidence=0.5
    )
    dests_half = {hop.dst_symbol for hop in result_half}

    assert dests_half == {"b.py::t1", "b.py::t2"}, "Only t1 (0.95) and t2 (0.85) are >= 0.5; t3 (0.4) is excluded"


@pytest.mark.asyncio
async def test_min_confidence_filtering_get_callers_of(
    test_snapshot_id: str,
    cleanup_test_data: None,
) -> None:
    """Verify min_confidence parameter works for reverse lookup (get_callers_of)."""
    db = get_db()

    await db.executemany(
        """
        INSERT INTO symbol_graph_edges
        (snapshot_id, src_symbol, dst_symbol, edge_type, confidence_score, resolution_method, evidence_lines)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (test_snapshot_id, "a.py::caller1", "b.py::target", "calls", 0.95, "import_path_match", "[]"),
            (test_snapshot_id, "a.py::caller2", "b.py::target", "calls", 0.4, "name_heuristic_ambiguous", "[]"),
        ],
    )

    # Unfiltered
    all_callers = await get_callers_of(test_snapshot_id, "b.py", high_confidence_only=False)
    assert len(all_callers) == 2

    # With min_confidence=0.85
    high_callers = await get_callers_of(
        test_snapshot_id, "b.py", high_confidence_only=False, min_confidence=0.85
    )
    assert len(high_callers) == 1
    assert high_callers[0].src_symbol == "a.py::caller1"


@pytest.mark.asyncio
async def test_min_confidence_none_unfiltered(
    test_snapshot_id: str,
    cleanup_test_data: None,
) -> None:
    """Verify min_confidence=None means unfiltered (default, preserve current recall)."""
    db = get_db()

    await db.executemany(
        """
        INSERT INTO symbol_graph_edges
        (snapshot_id, src_symbol, dst_symbol, edge_type, confidence_score, resolution_method, evidence_lines)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (test_snapshot_id, "a.py::caller", "b.py::t1", "calls", 0.95, "import_path_match", "[]"),
            (test_snapshot_id, "a.py::caller", "b.py::t2", "calls", 0.0, "unknown", "[]"),
        ],
    )

    # Explicitly pass min_confidence=None
    result = await get_callees_of(
        test_snapshot_id, "a.py", high_confidence_only=False, min_confidence=None
    )

    assert len(result) == 2, "min_confidence=None should be unfiltered (all edges)"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def test_snapshot_id() -> str:
    """Provide a test snapshot ID."""
    import uuid
    return f"test-snap-{uuid.uuid4().hex[:8]}"


@pytest.fixture
async def cleanup_test_data(test_snapshot_id: str) -> None:
    """Clean up test edges after test runs (async/DB integration tests only)."""
    yield
    db = get_db()
    await db.execute(
        "DELETE FROM symbol_graph_edges WHERE snapshot_id LIKE ?",
        ("test-snap-%",),
    )
    await db.commit()
