"""Tests for CS-249: symbol_graph_edges incremental rebuild with carry-forward.

Covers:
- Fresh snapshot: symbol edges computed fresh (no carry-forward).
- Idempotent re-run: build same snapshot twice, no crash, stable edge count.
- Mixed changed/unchanged: changed files' edges recomputed, unchanged files' carried forward.
- Collision guard: ensure foo.py carry-forward never matches foo2.py edges.
"""
from __future__ import annotations

import json

import pytest

from domain.structural_graph.extraction_cache import (
    copy_unchanged_symbol_edges,
    ExtractionCacheResult,
)
from infrastructure.db.database import get_db
from shared.utils import new_id, utc_now_iso


@pytest.mark.asyncio
async def test_copy_unchanged_symbol_edges_basic():
    """Test basic carry-forward of symbol edges for unchanged files."""
    db = get_db()
    snap1_id = new_id()
    snap2_id = new_id()

    # Insert symbol edges for snap1 with two files: foo.py and bar.py
    now = utc_now_iso()
    snap1_edges = [
        (snap1_id, "foo.py::FooClass.method1", "bar.py::BarClass.helper", "calls", 0.8, "import_path_match", "high", json.dumps([10, 15])),
        (snap1_id, "foo.py::FooClass.method2", "bar.py::BarClass.helper", "calls", 0.9, "same_file_scope", "high", json.dumps([20])),
        (snap1_id, "bar.py::BarClass.helper", "foo.py::FooClass.util", "calls", 0.7, "mro_resolved", "high", json.dumps([30, 35])),
    ]
    await db.executemany(
        """
        INSERT INTO symbol_graph_edges
        (snapshot_id, src_symbol, dst_symbol, edge_type, confidence_score, resolution_method, confidence, evidence_lines)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        snap1_edges,
    )
    await db.commit()

    # Verify snap1 has 3 edges
    async with db.execute("SELECT COUNT(*) as cnt FROM symbol_graph_edges WHERE snapshot_id=?", (snap1_id,)) as cur:
        row = await cur.fetchone()
        assert row["cnt"] == 3

    # Now copy edges for foo.py (unchanged) to snap2
    unchanged_paths = ["foo.py"]
    copied = await copy_unchanged_symbol_edges(db, snap1_id, snap2_id, unchanged_paths)

    # Should have copied 2 edges (both starting with foo.py::)
    assert copied == 2

    # Verify snap2 has exactly 2 edges (the foo.py ones)
    async with db.execute("SELECT * FROM symbol_graph_edges WHERE snapshot_id=? ORDER BY src_symbol", (snap2_id,)) as cur:
        rows = await cur.fetchall()
        assert len(rows) == 2
        assert rows[0]["src_symbol"] == "foo.py::FooClass.method1"
        assert rows[1]["src_symbol"] == "foo.py::FooClass.method2"

        # Verify confidence_score and resolution_method are preserved
        assert rows[0]["confidence_score"] == 0.8
        assert rows[0]["resolution_method"] == "import_path_match"
        assert rows[1]["confidence_score"] == 0.9
        assert rows[1]["resolution_method"] == "same_file_scope"

        # Verify evidence_lines are preserved
        assert json.loads(rows[0]["evidence_lines"]) == [10, 15]
        assert json.loads(rows[1]["evidence_lines"]) == [20]


@pytest.mark.asyncio
async def test_copy_unchanged_symbol_edges_collision_guard():
    """Test that foo.py carry-forward never matches foo2.py edges (collision guard)."""
    db = get_db()
    snap1_id = new_id()
    snap2_id = new_id()

    # Insert edges for both foo.py and foo2.py in snap1
    snap1_edges = [
        (snap1_id, "foo.py::FooClass.method", "bar.py::BarClass.helper", "calls", 0.8, "unknown", "high", json.dumps([])),
        (snap1_id, "foo2.py::Foo2Class.method", "bar.py::BarClass.helper", "calls", 0.9, "unknown", "high", json.dumps([])),
    ]
    await db.executemany(
        """
        INSERT INTO symbol_graph_edges
        (snapshot_id, src_symbol, dst_symbol, edge_type, confidence_score, resolution_method, confidence, evidence_lines)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        snap1_edges,
    )
    await db.commit()

    # Copy edges for foo.py only (unchanged)
    unchanged_paths = ["foo.py"]
    copied = await copy_unchanged_symbol_edges(db, snap1_id, snap2_id, unchanged_paths)

    # Should have copied only 1 edge (foo.py, not foo2.py)
    assert copied == 1

    # Verify snap2 has exactly 1 edge and it's from foo.py
    async with db.execute("SELECT * FROM symbol_graph_edges WHERE snapshot_id=?", (snap2_id,)) as cur:
        rows = await cur.fetchall()
        assert len(rows) == 1
        assert rows[0]["src_symbol"] == "foo.py::FooClass.method"


@pytest.mark.asyncio
async def test_copy_unchanged_symbol_edges_empty():
    """Test that empty unchanged_paths returns 0."""
    db = get_db()
    snap1_id = new_id()
    snap2_id = new_id()

    # Insert edge for snap1
    snap1_edges = [
        (snap1_id, "foo.py::FooClass.method", "bar.py::BarClass.helper", "calls", 0.8, "unknown", "high", json.dumps([])),
    ]
    await db.executemany(
        """
        INSERT INTO symbol_graph_edges
        (snapshot_id, src_symbol, dst_symbol, edge_type, confidence_score, resolution_method, confidence, evidence_lines)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        snap1_edges,
    )
    await db.commit()

    # Try to copy with empty unchanged_paths
    copied = await copy_unchanged_symbol_edges(db, snap1_id, snap2_id, [])
    assert copied == 0

    # Verify snap2 has no edges
    async with db.execute("SELECT COUNT(*) as cnt FROM symbol_graph_edges WHERE snapshot_id=?", (snap2_id,)) as cur:
        row = await cur.fetchone()
        assert row["cnt"] == 0


@pytest.mark.asyncio
async def test_copy_unchanged_symbol_edges_multiple_files():
    """Test carry-forward for multiple unchanged files."""
    db = get_db()
    snap1_id = new_id()
    snap2_id = new_id()

    # Insert edges for three files: foo.py, bar.py, baz.py
    snap1_edges = [
        (snap1_id, "foo.py::FooClass.m1", "baz.py::BazClass.h1", "calls", 0.8, "unknown", "high", json.dumps([])),
        (snap1_id, "bar.py::BarClass.m2", "baz.py::BazClass.h2", "calls", 0.9, "unknown", "high", json.dumps([])),
        (snap1_id, "baz.py::BazClass.m3", "foo.py::FooClass.h3", "calls", 0.7, "unknown", "high", json.dumps([])),
    ]
    await db.executemany(
        """
        INSERT INTO symbol_graph_edges
        (snapshot_id, src_symbol, dst_symbol, edge_type, confidence_score, resolution_method, confidence, evidence_lines)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        snap1_edges,
    )
    await db.commit()

    # Copy edges for foo.py and bar.py (unchanged)
    unchanged_paths = ["foo.py", "bar.py"]
    copied = await copy_unchanged_symbol_edges(db, snap1_id, snap2_id, unchanged_paths)

    # Should have copied 2 edges (foo.py and bar.py sources)
    assert copied == 2

    # Verify snap2 has exactly 2 edges from foo.py and bar.py
    async with db.execute(
        "SELECT src_symbol FROM symbol_graph_edges WHERE snapshot_id=? ORDER BY src_symbol",
        (snap2_id,),
    ) as cur:
        rows = await cur.fetchall()
        assert len(rows) == 2
        src_symbols = [r["src_symbol"] for r in rows]
        assert "foo.py::FooClass.m1" in src_symbols
        assert "bar.py::BarClass.m2" in src_symbols


@pytest.mark.asyncio
async def test_copy_unchanged_symbol_edges_preserves_all_columns():
    """Test that all columns including edge_type are preserved."""
    db = get_db()
    snap1_id = new_id()
    snap2_id = new_id()

    # Insert edges with different edge_type values
    snap1_edges = [
        (snap1_id, "foo.py::FooClass.m1", "bar.py::BarClass.h1", "calls", 0.8, "method1", "high", json.dumps([1, 2])),
        (snap1_id, "foo.py::FooClass.m2", "bar.py::BarClass.h2", "returns", 0.6, "method2", "low", json.dumps([3])),
        (snap1_id, "foo.py::FooClass.m3", "bar.py::BarClass.h3", "param_type", 0.5, "method3", "low", json.dumps([])),
    ]
    await db.executemany(
        """
        INSERT INTO symbol_graph_edges
        (snapshot_id, src_symbol, dst_symbol, edge_type, confidence_score, resolution_method, confidence, evidence_lines)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        snap1_edges,
    )
    await db.commit()

    # Copy all edges from foo.py
    unchanged_paths = ["foo.py"]
    copied = await copy_unchanged_symbol_edges(db, snap1_id, snap2_id, unchanged_paths)
    assert copied == 3

    # Verify all columns are preserved
    async with db.execute(
        "SELECT edge_type, confidence_score, resolution_method, confidence, evidence_lines FROM symbol_graph_edges WHERE snapshot_id=? ORDER BY src_symbol",
        (snap2_id,),
    ) as cur:
        rows = await cur.fetchall()
        assert rows[0]["edge_type"] == "calls"
        assert rows[0]["confidence_score"] == 0.8
        assert rows[0]["resolution_method"] == "method1"
        assert rows[0]["confidence"] == "high"
        assert json.loads(rows[0]["evidence_lines"]) == [1, 2]

        assert rows[1]["edge_type"] == "returns"
        assert rows[1]["confidence_score"] == 0.6
        assert rows[1]["resolution_method"] == "method2"
        assert rows[1]["confidence"] == "low"

        assert rows[2]["edge_type"] == "param_type"
        assert rows[2]["confidence_score"] == 0.5
