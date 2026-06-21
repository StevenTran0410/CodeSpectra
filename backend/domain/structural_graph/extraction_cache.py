"""SHA256 incremental extraction cache for structural graph builds (RPA-058 Part A)."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass

import aiosqlite

from shared.logger import logger
from shared.utils import utc_now_iso


@dataclass
class ExtractionCacheResult:
    """Result of file classification against previous cache."""

    unchanged_paths: list[str]  # rel_paths where SHA256 matched — skip extraction
    changed_files: list  # file objects (from manifest) needing extraction
    sha256_map: dict[str, str]  # rel_path -> sha256 for ALL files
    previous_snapshot_id: str | None  # source for edge copy, None if first run


def compute_sha256(abs_path: str) -> str:
    """Compute SHA256 of file by streaming 64KB chunks."""
    hasher = hashlib.sha256()
    try:
        with open(abs_path, "rb") as f:
            while True:
                chunk = f.read(65536)  # 64KB chunks
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger.warning(f"Failed to compute SHA256 for {abs_path}: {e}")
        return ""


async def load_previous_cache(
    db: aiosqlite.Connection, local_repo_id: str, current_snapshot_id: str
) -> dict[str, str]:
    """Load SHA256 cache from previous snapshot for same repo.

    Returns {rel_path: sha256} mapping for most recent snapshot before current.
    Returns {} if no previous snapshot or on any error (graceful degradation).
    """
    try:
        # Find the most recent snapshot before this one
        query = """
            SELECT id FROM repo_snapshots
            WHERE local_repo_id=? AND id<>?
            ORDER BY created_at DESC
            LIMIT 1
        """
        async with db.execute(query, (local_repo_id, current_snapshot_id)) as cur:
            row = await cur.fetchone()

        if row is None:
            return {}

        prev_snapshot_id = row["id"]

        # Load cache entries for previous snapshot
        query = "SELECT file_path, sha256 FROM file_extraction_cache WHERE snapshot_id=?"
        async with db.execute(query, (prev_snapshot_id,)) as cur:
            rows = await cur.fetchall()

        return {r["file_path"]: r["sha256"] for r in rows}
    except Exception as e:
        logger.warning(f"Failed to load previous cache: {e}")
        return {}


async def _get_previous_snapshot_id(
    db: aiosqlite.Connection, local_repo_id: str, current_snapshot_id: str
) -> str | None:
    """Get the ID of the previous snapshot for edge copying."""
    try:
        query = """
            SELECT id FROM repo_snapshots
            WHERE local_repo_id=? AND id<>?
            ORDER BY created_at DESC
            LIMIT 1
        """
        async with db.execute(query, (local_repo_id, current_snapshot_id)) as cur:
            row = await cur.fetchone()
        return row["id"] if row else None
    except Exception:
        return None


async def classify_files(
    files: list,
    previous_cache: dict[str, str],
    previous_snapshot_id: str | None,
    executor,
    root_path: str,
) -> ExtractionCacheResult:
    """Classify files as changed/unchanged by comparing SHA256 against previous cache.

    Computes SHA256 for all files in parallel via executor, then compares against
    previous_cache to identify unchanged files.

    ``files`` are ``sqlite3.Row`` objects with at least a ``rel_path`` key.
    ``root_path`` is the absolute project root used to build full file paths.
    """
    import os

    loop = asyncio.get_event_loop()

    # Compute SHA256 for all files in parallel
    sha256_tasks = [
        loop.run_in_executor(executor, compute_sha256, os.path.join(root_path, f["rel_path"]))
        for f in files
    ]
    sha256_list = await asyncio.gather(*sha256_tasks)

    sha256_map: dict[str, str] = {}
    unchanged_paths: list[str] = []
    changed_files: list = []

    for f, sha256 in zip(files, sha256_list):
        rel = f["rel_path"]
        if not sha256:
            # Compute failed — treat as changed
            changed_files.append(f)
            continue

        sha256_map[rel] = sha256

        # Check if unchanged
        if rel in previous_cache and previous_cache[rel] == sha256:
            unchanged_paths.append(rel)
        else:
            changed_files.append(f)

    logger.info(
        f"Classified files: {len(unchanged_paths)} unchanged, {len(changed_files)} changed"
    )

    return ExtractionCacheResult(
        unchanged_paths=unchanged_paths,
        changed_files=changed_files,
        sha256_map=sha256_map,
        previous_snapshot_id=previous_snapshot_id,
    )


async def copy_unchanged_edges(
    db: aiosqlite.Connection,
    previous_snapshot_id: str,
    current_snapshot_id: str,
    unchanged_paths: list[str],
) -> int:
    """Copy edges for unchanged files from previous snapshot via bulk INSERT SELECT.

    Chunks unchanged_paths into groups of 500 (SQLite variable limit).
    Returns total edges copied.
    """
    if not unchanged_paths:
        return 0

    total_copied = 0
    chunk_size = 500

    for i in range(0, len(unchanged_paths), chunk_size):
        chunk = unchanged_paths[i : i + chunk_size]
        placeholders = ", ".join("?" * len(chunk))

        query = f"""
            INSERT INTO structural_graph_edges
            (snapshot_id, src_path, dst_path, edge_type, is_external, created_at)
            SELECT ?, src_path, dst_path, edge_type, is_external, ?
            FROM structural_graph_edges
            WHERE snapshot_id=? AND src_path IN ({placeholders})
        """
        params = (
            current_snapshot_id,
            utc_now_iso(),
            previous_snapshot_id,
            *chunk,
        )

        await db.execute(query, params)
        rows_affected = db.total_changes - total_copied
        total_copied += rows_affected

    await db.commit()
    return total_copied


async def copy_unchanged_symbol_edges(
    db: aiosqlite.Connection,
    previous_snapshot_id: str,
    current_snapshot_id: str,
    unchanged_paths: list[str],
) -> int:
    """Copy symbol edges for unchanged files from previous snapshot via bulk INSERT SELECT (CS-249).

    Matches symbol edges belonging to unchanged files using substr-based prefix matching
    on src_symbol (format: 'path::symbol_name'). Preserves all edge properties including
    confidence_score and resolution_method.

    Chunks unchanged_paths into groups of 200 to account for 2 bind params per path
    (path and path:: prefix), staying well under SQLite's variable limit.

    Returns total symbol edges copied.
    """
    if not unchanged_paths:
        return 0

    total_copied = 0
    chunk_size = 200  # Conservative: 2 params per path + 2 fixed params = 402 params max

    for i in range(0, len(unchanged_paths), chunk_size):
        chunk = unchanged_paths[i : i + chunk_size]

        # Build OR'd conditions for each unchanged path:
        # substr(src_symbol, 1, length(?) + 2) = ? || '::'
        # This ensures 'foo.py' only matches 'foo.py::*', not 'foo2.py::*'
        conditions = []
        params_list = []
        for path in chunk:
            conditions.append("substr(src_symbol, 1, length(?) + 2) = ? || '::'")
            params_list.extend([path, path])

        condition_str = " OR ".join(f"({c})" for c in conditions)

        query = f"""
            INSERT INTO symbol_graph_edges
            (snapshot_id, src_symbol, dst_symbol, edge_type, confidence_score, resolution_method, confidence, evidence_lines)
            SELECT ?, src_symbol, dst_symbol, edge_type, confidence_score, resolution_method, confidence, evidence_lines
            FROM symbol_graph_edges
            WHERE snapshot_id=? AND ({condition_str})
        """
        params = (current_snapshot_id, previous_snapshot_id, *params_list)

        # Count rows to be inserted before executing
        count_query = f"""
            SELECT COUNT(*) as cnt
            FROM symbol_graph_edges
            WHERE snapshot_id=? AND ({condition_str})
        """
        count_params = (previous_snapshot_id, *params_list)
        async with db.execute(count_query, count_params) as cur:
            count_row = await cur.fetchone()
            chunk_count = count_row["cnt"] if count_row is not None else 0

        await db.execute(query, params)
        total_copied += chunk_count

    await db.commit()
    return total_copied


async def write_cache(
    db: aiosqlite.Connection, snapshot_id: str, sha256_map: dict[str, str]
) -> None:
    """Write SHA256 cache entries to DB via bulk INSERT OR REPLACE."""
    if not sha256_map:
        return

    now = utc_now_iso()
    rows = [
        (snapshot_id, rel_path, sha256, now) for rel_path, sha256 in sha256_map.items()
    ]

    await db.executemany(
        """
        INSERT OR REPLACE INTO file_extraction_cache
        (snapshot_id, file_path, sha256, extracted_at)
        VALUES (?, ?, ?, ?)
        """,
        rows,
    )

    await db.commit()
    logger.info(f"Wrote {len(rows)} cache entries for snapshot {snapshot_id}")
