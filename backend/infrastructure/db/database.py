"""Async SQLite database access via aiosqlite with sequential migrations."""
import os
from pathlib import Path
from typing import Any

import aiosqlite

from shared.logger import logger

_db: aiosqlite.Connection | None = None

# ──────────────────────────────────────────────────────────────────────────────
# Migrations — add new dicts at the end; never modify existing ones.
# ──────────────────────────────────────────────────────────────────────────────
_MIGRATIONS: list[dict[str, Any]] = [
    {
        "version": 0,
        "description": "Initial schema: app_metadata + workspaces",
        "sql": """
            CREATE TABLE IF NOT EXISTS app_metadata (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS workspaces (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL UNIQUE,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                settings    TEXT NOT NULL DEFAULT '{}'
            );

            INSERT OR IGNORE INTO app_metadata (key, value)
                VALUES ('first_launched_at', datetime('now'));
        """,
    },
    {
        "version": 1,
        "description": "Add provider_configs table",
        "sql": """
            CREATE TABLE IF NOT EXISTS provider_configs (
                id           TEXT PRIMARY KEY,
                kind         TEXT NOT NULL,
                display_name TEXT NOT NULL,
                base_url     TEXT NOT NULL,
                model_id     TEXT NOT NULL,
                capabilities TEXT NOT NULL DEFAULT '{}',
                extra        TEXT NOT NULL DEFAULT '{}',
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL
            );
        """,
    },
    {
        "version": 2,
        "description": "Add local_repos table for local folder import",
        "sql": """
            CREATE TABLE IF NOT EXISTS local_repos (
                id                TEXT PRIMARY KEY,
                path              TEXT NOT NULL UNIQUE,
                name              TEXT NOT NULL,
                source_type       TEXT NOT NULL DEFAULT 'local_folder',
                is_git_repo       INTEGER NOT NULL DEFAULT 0,
                git_branch        TEXT,
                git_head_hash     TEXT,
                git_remote_url    TEXT,
                has_size_warning  INTEGER NOT NULL DEFAULT 0,
                added_at          TEXT NOT NULL,
                last_validated_at TEXT NOT NULL
            );
        """,
    },
    {
        "version": 3,
        "description": "Add selected_branch to local_repos (user-chosen branch for analysis)",
        "sql": """
            ALTER TABLE local_repos ADD COLUMN selected_branch TEXT;
        """,
    },
    {
        "version": 4,
        "description": "Add github_accounts table for GitHub OAuth device flow",
        "sql": """
            CREATE TABLE IF NOT EXISTS github_accounts (
                id           TEXT PRIMARY KEY,
                login        TEXT NOT NULL,
                display_name TEXT,
                avatar_url   TEXT,
                access_token TEXT NOT NULL,
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL
            );
        """,
    },
    {
        "version": 5,
        "description": "Add jobs table for analysis pipeline tracking",
        "sql": """
            CREATE TABLE IF NOT EXISTS jobs (
                id           TEXT PRIMARY KEY,
                type         TEXT NOT NULL,
                repo_id      TEXT,
                status       TEXT NOT NULL DEFAULT 'pending',
                steps        TEXT NOT NULL DEFAULT '{}',
                current_step TEXT,
                error        TEXT,
                started_at   TEXT NOT NULL,
                finished_at  TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_jobs_repo_id ON jobs(repo_id);
            CREATE INDEX IF NOT EXISTS idx_jobs_started_at ON jobs(started_at DESC);
        """,
    },
    {
        "version": 6,
        "description": "Add repo_snapshots table for sync engine",
        "sql": """
            CREATE TABLE IF NOT EXISTS repo_snapshots (
                id             TEXT PRIMARY KEY,
                local_repo_id  TEXT NOT NULL,
                branch         TEXT,
                commit_hash    TEXT,
                local_path     TEXT NOT NULL,
                status         TEXT NOT NULL DEFAULT 'pending',
                error          TEXT,
                clone_policy   TEXT NOT NULL DEFAULT 'full',
                synced_at      TEXT NOT NULL,
                created_at     TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_snapshots_repo ON repo_snapshots(local_repo_id);
        """,
    },
    {
        "version": 7,
        "description": "Add per-repository setup settings to local_repos",
        "sql": """
            ALTER TABLE local_repos ADD COLUMN sync_mode TEXT NOT NULL DEFAULT 'latest';
            ALTER TABLE local_repos ADD COLUMN pinned_ref TEXT;
            ALTER TABLE local_repos ADD COLUMN ignore_overrides TEXT NOT NULL DEFAULT '[]';
            ALTER TABLE local_repos ADD COLUMN detect_submodules INTEGER NOT NULL DEFAULT 1;
        """,
    },
    {
        "version": 8,
        "description": "Add manifest_files table for file manifest and delta detection",
        "sql": """
            CREATE TABLE IF NOT EXISTS manifest_files (
                id          TEXT PRIMARY KEY,
                snapshot_id TEXT NOT NULL,
                rel_path    TEXT NOT NULL,
                language    TEXT,
                category    TEXT NOT NULL,
                size_bytes  INTEGER NOT NULL,
                mtime_ns    INTEGER NOT NULL,
                checksum    TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_manifest_snapshot ON manifest_files(snapshot_id);
            CREATE INDEX IF NOT EXISTS idx_manifest_rel_path ON manifest_files(snapshot_id, rel_path);
        """,
    },
    {
        "version": 9,
        "description": "Add active_snapshot_id to local_repos",
        "sql": """
            ALTER TABLE local_repos ADD COLUMN active_snapshot_id TEXT;
        """,
    },
    {
        "version": 10,
        "description": "Add manual_ignores to repo_snapshots",
        "sql": """
            ALTER TABLE repo_snapshots ADD COLUMN manual_ignores TEXT NOT NULL DEFAULT '[]';
        """,
    },
    {
        "version": 11,
        "description": "Add repo_map and code_symbols tables for RPA-032",
        "sql": """
            CREATE TABLE IF NOT EXISTS code_symbols (
                id          TEXT PRIMARY KEY,
                snapshot_id TEXT NOT NULL,
                rel_path    TEXT NOT NULL,
                language    TEXT,
                name        TEXT NOT NULL,
                kind        TEXT NOT NULL,
                line_start  INTEGER NOT NULL,
                line_end    INTEGER NOT NULL,
                signature   TEXT,
                parent_name TEXT,
                created_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_symbols_snapshot ON code_symbols(snapshot_id);
            CREATE INDEX IF NOT EXISTS idx_symbols_file ON code_symbols(snapshot_id, rel_path);
            CREATE INDEX IF NOT EXISTS idx_symbols_name ON code_symbols(snapshot_id, name);

            CREATE TABLE IF NOT EXISTS repo_maps (
                snapshot_id        TEXT PRIMARY KEY,
                total_symbols      INTEGER NOT NULL,
                files_indexed      INTEGER NOT NULL,
                parse_failures     INTEGER NOT NULL,
                extract_mode       TEXT NOT NULL,
                language_breakdown TEXT NOT NULL DEFAULT '{}',
                kind_breakdown     TEXT NOT NULL DEFAULT '{}',
                generated_at       TEXT NOT NULL
            );
        """,
    },
    {
        "version": 12,
        "description": "Add extract_source to code_symbols for quality tracing",
        "sql": """
            ALTER TABLE code_symbols ADD COLUMN extract_source TEXT NOT NULL DEFAULT 'lexical';
        """,
    },
    {
        "version": 13,
        "description": "Add structural graph tables for RPA-033",
        "sql": """
            CREATE TABLE IF NOT EXISTS structural_graph_edges (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id TEXT NOT NULL,
                src_path    TEXT NOT NULL,
                dst_path    TEXT NOT NULL,
                edge_type   TEXT NOT NULL,
                is_external INTEGER NOT NULL DEFAULT 0,
                created_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_graph_edges_snapshot ON structural_graph_edges(snapshot_id);
            CREATE INDEX IF NOT EXISTS idx_graph_edges_src ON structural_graph_edges(snapshot_id, src_path);

            CREATE TABLE IF NOT EXISTS structural_graph_summaries (
                snapshot_id        TEXT PRIMARY KEY,
                total_nodes        INTEGER NOT NULL,
                total_edges        INTEGER NOT NULL,
                external_edges     INTEGER NOT NULL,
                entrypoints        TEXT NOT NULL DEFAULT '[]',
                top_central_files  TEXT NOT NULL DEFAULT '[]',
                generated_at       TEXT NOT NULL,
                native_toolchain   TEXT
            );
        """,
    },
    {
        "version": 14,
        "description": "Add retrieval chunk/index tables for RPA-034",
        "sql": """
            CREATE TABLE IF NOT EXISTS retrieval_chunks (
                id             TEXT PRIMARY KEY,
                snapshot_id    TEXT NOT NULL,
                rel_path       TEXT NOT NULL,
                language       TEXT,
                category       TEXT NOT NULL,
                chunk_index    INTEGER NOT NULL,
                content        TEXT NOT NULL,
                token_estimate INTEGER NOT NULL,
                created_at     TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_retrieval_chunks_snapshot ON retrieval_chunks(snapshot_id);
            CREATE INDEX IF NOT EXISTS idx_retrieval_chunks_path ON retrieval_chunks(snapshot_id, rel_path);

            CREATE TABLE IF NOT EXISTS retrieval_indexes (
                id              TEXT PRIMARY KEY,
                snapshot_id     TEXT NOT NULL,
                rel_path        TEXT NOT NULL,
                chunk_index     INTEGER NOT NULL,
                lexical_preview TEXT NOT NULL,
                created_at      TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_retrieval_index_snapshot ON retrieval_indexes(snapshot_id);
        """,
    },
]

TARGET_VERSION = len(_MIGRATIONS) - 1


async def init_db() -> None:
    """Open the database, enable WAL mode, and run pending migrations."""
    global _db

    data_dir = os.getenv("CODESPECTRA_DATA_DIR", ".")
    db_path = Path(data_dir) / "codespectra.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Opening database: {db_path}")
    _db = await aiosqlite.connect(str(db_path))
    _db.row_factory = aiosqlite.Row

    await _db.execute("PRAGMA journal_mode=WAL")
    await _db.execute("PRAGMA foreign_keys=ON")
    await _db.commit()

    await _run_migrations(_db)


async def close_db() -> None:
    global _db
    if _db is not None:
        await _db.close()
        _db = None
        logger.info("Database connection closed")


def get_db() -> aiosqlite.Connection:
    if _db is None:
        raise RuntimeError("Database has not been initialised — call init_db() first")
    return _db


# ──────────────────────────────────────────────────────────────────────────────
# Internal migration runner
# ──────────────────────────────────────────────────────────────────────────────
async def _run_migrations(db: aiosqlite.Connection) -> None:
    # Determine current schema version
    table_exists = await db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='app_metadata'"
    )
    row = await table_exists.fetchone()

    current_version: int
    if row is None:
        current_version = -1
    else:
        cur = await db.execute("SELECT value FROM app_metadata WHERE key='schema_version'")
        ver_row = await cur.fetchone()
        current_version = int(ver_row["value"]) if ver_row else -1

    if current_version >= TARGET_VERSION:
        logger.debug(f"Database schema up-to-date (version {current_version})")
        return

    for migration in _MIGRATIONS:
        if migration["version"] <= current_version:
            continue
        v = migration["version"]
        desc = migration["description"]
        logger.info(f"Running migration {v}: {desc}")
        await db.executescript(migration["sql"])
        await db.execute(
            "INSERT OR REPLACE INTO app_metadata (key, value) VALUES ('schema_version', ?)",
            (str(v),),
        )
        await db.commit()
        logger.info(f"Migration {v} complete")
