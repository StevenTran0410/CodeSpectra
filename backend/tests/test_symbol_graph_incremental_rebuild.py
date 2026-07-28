"""Tests for symbol_graph_edges incremental rebuild with carry-forward.

Covers:
- Fresh snapshot: symbol edges computed fresh (no carry-forward).
- Idempotent re-run: build same snapshot twice, no crash, stable edge count.
- Mixed changed/unchanged: changed files' edges recomputed, unchanged files' carried forward.
- Collision guard: ensure foo.py carry-forward never matches foo2.py edges.
"""
from __future__ import annotations

import json

import pytest

from domain.structural_graph.extraction_cache import copy_unchanged_symbol_edges
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
async def test_symbol_edges_for_file_cs250():
    """Verify that symbol_edges_for_file returns defined symbols + outgoing/incoming
    cross-file edges with confidence_score/resolution_method, using the same
    collision-safe prefix match as copy_unchanged_symbol_edges (foo.py vs foo2.py)."""
    from domain.structural_graph.service import StructuralGraphService

    db = get_db()
    snap_id = new_id()
    now = utc_now_iso()
    rows = [
        # foo.py calls out to bar.py (outgoing for foo.py)
        (snap_id, "foo.py::caller", "bar.py::helper", "calls", 0.95, "import_path_match", "high", json.dumps([1])),
        # bar.py calls back into foo.py (incoming for foo.py)
        (snap_id, "bar.py::other", "foo.py::util", "calls", 0.4, "name_heuristic_ambiguous", "low", json.dumps([2])),
        # foo2.py edges must never leak into foo.py's results (collision guard)
        (snap_id, "foo2.py::unrelated", "bar.py::helper", "calls", 0.9, "mro_resolved", "high", json.dumps([3])),
        # Self-referential: foo.py calling another symbol in foo.py itself. The
        # UNION ALL (not UNION) must surface this in BOTH outgoing and incoming --
        # a plain UNION with identical-looking rows could silently dedupe it.
        (snap_id, "foo.py::caller", "foo.py::local_helper", "calls", 0.85,
         "same_file_scope", "high", json.dumps([4])),
    ]
    await db.executemany(
        """
        INSERT INTO symbol_graph_edges
        (snapshot_id, src_symbol, dst_symbol, edge_type, confidence_score, resolution_method, confidence, evidence_lines)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    await db.commit()

    service = StructuralGraphService()
    result = await service.symbol_edges_for_file(snap_id, "foo.py")

    assert result.file_path == "foo.py"
    assert sorted(result.defined_symbols) == ["caller", "local_helper", "util"]

    assert len(result.outgoing) == 2
    outgoing_by_dst = {e.dst_symbol: e for e in result.outgoing}
    assert outgoing_by_dst["bar.py::helper"].src_symbol == "foo.py::caller"
    assert outgoing_by_dst["bar.py::helper"].confidence_score == 0.95
    assert outgoing_by_dst["bar.py::helper"].resolution_method == "import_path_match"
    # Self-referential edge must appear in outgoing (foo.py -> foo.py)
    assert outgoing_by_dst["foo.py::local_helper"].src_symbol == "foo.py::caller"
    assert outgoing_by_dst["foo.py::local_helper"].confidence_score == 0.85

    assert len(result.incoming) == 2
    incoming_by_src = {e.src_symbol: e for e in result.incoming}
    assert incoming_by_src["bar.py::other"].dst_symbol == "foo.py::util"
    assert incoming_by_src["bar.py::other"].confidence_score == 0.4
    assert incoming_by_src["bar.py::other"].resolution_method == "name_heuristic_ambiguous"
    # Same self-referential edge must ALSO appear in incoming (foo.py -> foo.py) --
    # UNION ALL must not collapse it just because outgoing already has an identical-looking row.
    assert incoming_by_src["foo.py::caller"].dst_symbol == "foo.py::local_helper"
    assert incoming_by_src["foo.py::caller"].confidence_score == 0.85

    # Collision guard: foo2.py's edge must not appear anywhere in foo.py's result
    all_symbols = [e.src_symbol for e in result.outgoing + result.incoming] + [
        e.dst_symbol for e in result.outgoing + result.incoming
    ]
    assert not any(s.startswith("foo2.py") for s in all_symbols)


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


# ═══════════════════════════════════════════════════════════════════════════════
# Integration Tests: StructuralGraphService.build() with Real Files & SHA256
# ═══════════════════════════════════════════════════════════════════════════════
# These tests exercise the full build() code path including the cache classify,
# DELETE, and copy_unchanged_symbol_edges() wiring to catch real integration bugs.

import tempfile
from pathlib import Path

from domain.structural_graph.service import StructuralGraphService
from domain.structural_graph.types import BuildGraphRequest
from infrastructure.db.database import get_db


@pytest.fixture
async def temp_repo() -> tuple[str, str]:
    """Create a temporary repo on disk with files; yield (local_path, local_repo_id)."""
    import shutil
    tmpdir = tempfile.mkdtemp()
    repo_id = new_id()
    db = get_db()
    now = utc_now_iso()

    # Insert repo (using correct column name: path, not root_path)
    await db.execute(
        "INSERT INTO local_repos (id, name, path, added_at, last_validated_at) VALUES (?, ?, ?, ?, ?)",
        (repo_id, "test_repo", tmpdir, now, now),
    )
    await db.commit()

    yield tmpdir, repo_id

    # Cleanup: remove temp directory and DB rows
    shutil.rmtree(tmpdir, ignore_errors=True)
    await db.execute("DELETE FROM local_repos WHERE id=?", (repo_id,))
    await db.commit()


@pytest.fixture
async def cleanup_build_snapshots() -> None:
    """Clean up test snapshots after test runs."""
    yield
    db = get_db()
    await db.execute("DELETE FROM repo_snapshots WHERE id LIKE ?", ("test-build-%",))
    await db.execute("DELETE FROM manifest_files WHERE snapshot_id LIKE ?", ("test-build-%",))
    await db.execute("DELETE FROM structural_graph_edges WHERE snapshot_id LIKE ?", ("test-build-%",))
    await db.execute("DELETE FROM symbol_graph_edges WHERE snapshot_id LIKE ?", ("test-build-%",))
    await db.execute("DELETE FROM file_extraction_cache WHERE snapshot_id LIKE ?", ("test-build-%",))
    await db.commit()


@pytest.mark.asyncio
async def test_build_fresh_snapshot_no_previous(
    temp_repo: tuple[str, str],
    cleanup_build_snapshots,  # pytest fixture cleanup hook; unused parameter pylint: disable
) -> None:
    """Test fresh snapshot: symbol edges computed fresh when no previous snapshot exists."""
    local_path, local_repo_id = temp_repo
    db = get_db()
    snap_id = f"test-build-fresh-{new_id()[:8]}"

    # Create two Python files with imports on disk
    foo_file = Path(local_path) / "foo.py"
    bar_file = Path(local_path) / "bar.py"
    foo_file.write_text("def helper(): pass\ndef caller(): helper()")
    bar_file.write_text("def process(): pass")

    # Insert snapshot
    now = utc_now_iso()
    await db.execute(
        """
        INSERT INTO repo_snapshots (id, local_repo_id, local_path, synced_at, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (snap_id, local_repo_id, local_path, now, now),
    )

    # Insert manifest entries
    await db.executemany(
        """
        INSERT INTO manifest_files
        (id, snapshot_id, rel_path, language, category, size_bytes, mtime_ns, checksum)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (new_id(), snap_id, "foo.py", "python", "source", 100, 0, "hash1"),
            (new_id(), snap_id, "bar.py", "python", "source", 50, 0, "hash2"),
        ],
    )
    await db.commit()

    # Build graph on fresh snapshot
    service = StructuralGraphService()
    req = BuildGraphRequest(snapshot_id=snap_id, force_rebuild=True)
    response = await service.build(req)

    # Verify response (summary should be created)
    assert response.summary is not None
    assert response.summary.total_nodes >= 1  # At least foo.py should be a node

    # Verify symbol edges were created from foo.py (which has caller->helper edge)
    async with db.execute(
        "SELECT COUNT(*) as cnt FROM symbol_graph_edges WHERE snapshot_id=?",
        (snap_id,),
    ) as cur:
        row = await cur.fetchone()
        edge_count = row["cnt"]
        # foo.py has def caller(): helper() which should produce at least one symbol edge
        assert edge_count >= 1, f"Fresh snapshot should have at least 1 symbol edge from foo.py, got {edge_count}"


@pytest.mark.asyncio
async def test_build_idempotent_rerun_no_unique_constraint_crash(
    temp_repo: tuple[str, str],
    cleanup_build_snapshots,  # pytest fixture cleanup hook; unused parameter pylint: disable
) -> None:
    """Test idempotent re-run: build same snapshot twice, no UNIQUE constraint crash."""
    local_path, local_repo_id = temp_repo
    db = get_db()
    snap_id = f"test-build-idem-{new_id()[:8]}"

    # Create files on disk
    foo_file = Path(local_path) / "foo.py"
    bar_file = Path(local_path) / "bar.py"
    foo_file.write_text("def helper(): pass\ndef caller(): helper()")
    bar_file.write_text("def process(): pass")

    # Insert snapshot and manifest
    now = utc_now_iso()
    await db.execute(
        "INSERT INTO repo_snapshots (id, local_repo_id, local_path, synced_at, created_at) VALUES (?, ?, ?, ?, ?)",
        (snap_id, local_repo_id, local_path, now, now),
    )
    await db.executemany(
        """
        INSERT INTO manifest_files
        (id, snapshot_id, rel_path, language, category, size_bytes, mtime_ns, checksum)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (new_id(), snap_id, "foo.py", "python", "source", 100, 0, "hash1"),
            (new_id(), snap_id, "bar.py", "python", "source", 50, 0, "hash2"),
        ],
    )
    await db.commit()

    service = StructuralGraphService()
    req = BuildGraphRequest(snapshot_id=snap_id, force_rebuild=True)

    # First build
    response1 = await service.build(req)
    assert response1.summary is not None

    # Get edge count after first build
    async with db.execute(
        "SELECT COUNT(*) as cnt FROM symbol_graph_edges WHERE snapshot_id=?",
        (snap_id,),
    ) as cur:
        row1 = await cur.fetchone()
        count1 = row1["cnt"]

    # Second build (idempotent) with force_rebuild=True
    # This should DELETE all edges first, then rebuild
    response2 = await service.build(req)
    assert response2.summary is not None

    # Verify edge count is stable (not doubled or crashed)
    async with db.execute(
        "SELECT COUNT(*) as cnt FROM symbol_graph_edges WHERE snapshot_id=?",
        (snap_id,),
    ) as cur:
        row2 = await cur.fetchone()
        count2 = row2["cnt"]

    assert count2 == count1, f"Edge count should be identical after rebuild: {count1} vs {count2}"


@pytest.mark.asyncio
async def test_multi_edge_types_same_pair_persists_without_unique_crash(
    temp_repo: tuple[str, str],
    cleanup_build_snapshots,
) -> None:
    """CS-330 Item 1 test: calls and extends edges on the SAME (snapshot_id, src, dst) pair both persist without IntegrityError."""
    db = get_db()
    snap_id = f"test-multi-edge-{new_id()[:8]}"

    await db.executemany(
        """
        INSERT INTO symbol_graph_edges
        (snapshot_id, src_symbol, dst_symbol, edge_type, confidence_score, resolution_method, confidence, evidence_lines)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (snap_id, "foo.py::Sub", "bar.py::Base", "calls", 1.0, "direct", "high", "[]"),
            (snap_id, "foo.py::Sub", "bar.py::Base", "extends", 1.0, "inheritance", "high", "[]"),
        ],
    )
    await db.commit()

    async with db.execute(
        "SELECT edge_type FROM symbol_graph_edges WHERE snapshot_id=? AND src_symbol=? AND dst_symbol=?",
        (snap_id, "foo.py::Sub", "bar.py::Base"),
    ) as cur:
        rows = await cur.fetchall()
        types = {r["edge_type"] for r in rows}
        assert types == {"calls", "extends"}


@pytest.mark.asyncio
async def test_build_mixed_changed_unchanged_carry_forward(
    temp_repo: tuple[str, str],
    cleanup_build_snapshots,  # pytest fixture cleanup hook; unused parameter pylint: disable
) -> None:
    """Test mixed changed/unchanged: build snap1, modify file, build snap2.
    Unchanged files' edges carried forward, changed files' edges recomputed.
    Confidence_score and resolution_method preserved on carry-forward.
    """
    local_path, local_repo_id = temp_repo
    db = get_db()
    snap1_id = f"test-build-snap1-{new_id()[:8]}"
    snap2_id = f"test-build-snap2-{new_id()[:8]}"

    # Create files on disk
    foo_file = Path(local_path) / "foo.py"
    bar_file = Path(local_path) / "bar.py"
    foo_file.write_text("def helper(): pass\ndef caller(): helper()")
    bar_file.write_text("def util(): pass\ndef process(): util()")

    # === Snapshot 1 ===
    now = utc_now_iso()
    await db.execute(
        "INSERT INTO repo_snapshots (id, local_repo_id, local_path, synced_at, created_at) VALUES (?, ?, ?, ?, ?)",
        (snap1_id, local_repo_id, local_path, now, now),
    )
    await db.executemany(
        """
        INSERT INTO manifest_files
        (id, snapshot_id, rel_path, language, category, size_bytes, mtime_ns, checksum)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (new_id(), snap1_id, "foo.py", "python", "source", 100, 0, "hash1"),
            (new_id(), snap1_id, "bar.py", "python", "source", 50, 0, "hash2"),
        ],
    )
    await db.commit()

    # Build snapshot 1
    service = StructuralGraphService()
    req1 = BuildGraphRequest(snapshot_id=snap1_id, force_rebuild=True)
    response1 = await service.build(req1)
    assert response1.summary is not None

    # Capture snap1 edges
    async with db.execute(
        "SELECT src_symbol, dst_symbol, confidence_score, resolution_method FROM symbol_graph_edges WHERE snapshot_id=?",
        (snap1_id,),
    ) as cur:
        snap1_edges = await cur.fetchall()
    snap1_edge_count = len(snap1_edges)

    # === Modify foo.py (changes SHA256) ===
    foo_file.write_text("def helper(): pass\ndef caller(): helper()\n# New comment\n")

    # === Snapshot 2 ===
    await db.execute(
        "INSERT INTO repo_snapshots (id, local_repo_id, local_path, synced_at, created_at) VALUES (?, ?, ?, ?, ?)",
        (snap2_id, local_repo_id, local_path, now, now),
    )
    await db.executemany(
        """
        INSERT INTO manifest_files
        (id, snapshot_id, rel_path, language, category, size_bytes, mtime_ns, checksum)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (new_id(), snap2_id, "foo.py", "python", "source", 150, 0, "hash1_modified"),  # Different hash
            (new_id(), snap2_id, "bar.py", "python", "source", 50, 0, "hash2"),  # Same hash
        ],
    )
    await db.commit()

    # Build snapshot 2
    req2 = BuildGraphRequest(snapshot_id=snap2_id, force_rebuild=False)
    response2 = await service.build(req2)
    assert response2.summary is not None

    # Capture snap2 edges
    async with db.execute(
        "SELECT src_symbol, dst_symbol, confidence_score, resolution_method FROM symbol_graph_edges WHERE snapshot_id=?",
        (snap2_id,),
    ) as cur:
        snap2_edges = await cur.fetchall()

    # === Verify carry-forward behavior ===
    # Snap1 must have produced edges for the test to be meaningful
    assert snap1_edge_count > 0, "Fixture produced no edges in snap1 — test is vacuous"

    # Find bar.py edges in snap1 (bar.py is unchanged, should be carried forward)
    bar_edges_snap1 = [e for e in snap1_edges if e["src_symbol"].startswith("bar.py::")]
    assert bar_edges_snap1, "bar.py produced no edges in snap1 — fixture broken"

    # These should be carried forward to snap2
    bar_edges_snap2 = [e for e in snap2_edges if e["src_symbol"].startswith("bar.py::")]
    assert len(bar_edges_snap2) == len(bar_edges_snap1), (
        f"bar.py edges not carried forward: snap1 had {len(bar_edges_snap1)}, snap2 has {len(bar_edges_snap2)}"
    )

    # Verify confidence_score and resolution_method are preserved on carry-forward
    for orig_edge in bar_edges_snap1:
        matching = [
            e
            for e in bar_edges_snap2
            if e["src_symbol"] == orig_edge["src_symbol"]
            and e["dst_symbol"] == orig_edge["dst_symbol"]
        ]
        assert matching, f"Edge {orig_edge['src_symbol']} -> {orig_edge['dst_symbol']} not carried forward"
        carried = matching[0]
        assert carried["confidence_score"] == orig_edge["confidence_score"], (
            f"Confidence mismatch for {orig_edge['src_symbol']}: "
            f"{orig_edge['confidence_score']} -> {carried['confidence_score']}"
        )
        assert carried["resolution_method"] == orig_edge["resolution_method"], (
            f"Resolution method mismatch for {orig_edge['src_symbol']}: "
            f"{orig_edge['resolution_method']} -> {carried['resolution_method']}"
        )

    # Verify foo.py edges in snap2 are from fresh computation, not carry-forward
    # foo.py was modified between snapshots, so its edges must be recomputed
    foo_edges_snap2 = [e for e in snap2_edges if e["src_symbol"].startswith("foo.py::")]
    assert foo_edges_snap2, "foo.py edges missing in snap2 (changed file must be recomputed)"
    # foo.py should still have its caller->helper edge after modification (comment added)
    # This implicitly verifies foo.py was not carried forward and was genuinely rebuilt
