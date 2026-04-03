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
