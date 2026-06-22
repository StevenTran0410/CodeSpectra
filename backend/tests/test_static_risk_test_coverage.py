"""Tests for static_risk.py's detect_test_coverage_shape and the include_tests guard.

When a user explicitly excludes test files from indexing (include_tests=False, the
default), manifest_files never contains any test files for that snapshot -- every
module would otherwise look like it has zero coverage, which is a false positive
caused by the user's own indexing choice, not a real test gap.
"""
from __future__ import annotations

import pytest

from domain.analysis.static_risk import detect_test_coverage_shape
from infrastructure.db.database import get_db
from shared.utils import new_id, utc_now_iso


async def _seed_repo_and_snapshot(include_tests: bool) -> str:
    db = get_db()
    repo_id = new_id()
    snap_id = new_id()
    now = utc_now_iso()

    await db.execute(
        """
        INSERT INTO local_repos
        (id, path, name, added_at, last_validated_at, include_tests)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (repo_id, f"/tmp/{repo_id}", "test_repo", now, now, int(include_tests)),
    )
    await db.execute(
        """
        INSERT INTO repo_snapshots
        (id, local_repo_id, local_path, synced_at, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (snap_id, repo_id, f"/tmp/{repo_id}", now, now),
    )

    # A module with 10 source files and zero test files -- would normally trigger
    # a "No test coverage" finding (len(src_files) >= 10 -> severity "high").
    manifest_rows = [
        (
            new_id(), snap_id, f"backend/domain/widgets/file_{i}.py",
            "python", "source", 100, 0, f"hash{i}",
        )
        for i in range(10)
    ]
    await db.executemany(
        """
        INSERT INTO manifest_files
        (id, snapshot_id, rel_path, language, category, size_bytes, mtime_ns, checksum)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        manifest_rows,
    )
    await db.commit()
    return snap_id


@pytest.mark.asyncio
async def test_no_findings_when_include_tests_false():
    """include_tests=False -> detector must skip entirely, no false 'no test coverage'
    findings, even though this fixture has 10 source files and zero test files."""
    snap_id = await _seed_repo_and_snapshot(include_tests=False)
    db = get_db()

    findings = await detect_test_coverage_shape(snap_id, db)

    assert findings == []


@pytest.mark.asyncio
async def test_findings_still_emitted_when_include_tests_true():
    """include_tests=True -> detector behaves as before, real gaps still reported."""
    snap_id = await _seed_repo_and_snapshot(include_tests=True)
    db = get_db()

    findings = await detect_test_coverage_shape(snap_id, db)

    assert len(findings) >= 1
    assert any(f.category == "test_gap" for f in findings)
    assert any("No test coverage" in f.title for f in findings)
