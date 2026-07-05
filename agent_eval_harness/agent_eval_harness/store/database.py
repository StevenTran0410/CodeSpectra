"""Async SQLite database access via aiosqlite with sequential migrations.

Mirrors backend/infrastructure/db/database.py's pattern exactly (module-level
connection, migrations-as-data, schema_version in app_metadata, WAL mode) —
this is AEH's own store (aeh.db), fully independent of codespectra.db.
"""
import logging
import os
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger("agent_eval_harness.store")

_db: aiosqlite.Connection | None = None

# ──────────────────────────────────────────────────────────────────────────────
# Migrations — add new dicts at the end; never modify existing ones.
# ──────────────────────────────────────────────────────────────────────────────
_MIGRATIONS: list[dict[str, Any]] = [
    {
        "version": 0,
        "description": "Initial AEH schema (CS-260 §4b): runs/traces/spans/evaluations/cases",
        "sql": """
            CREATE TABLE IF NOT EXISTS app_metadata (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS runs (
                id                TEXT PRIMARY KEY,
                target_system_id  TEXT NOT NULL,
                eval_plan_id      TEXT,
                started_at        TEXT NOT NULL,
                finished_at       TEXT,
                status            TEXT NOT NULL DEFAULT 'running'
            );

            CREATE TABLE IF NOT EXISTS traces (
                id                TEXT PRIMARY KEY,
                run_id            TEXT NOT NULL REFERENCES runs(id),
                dataset_case_id   TEXT,
                root_input        TEXT NOT NULL,
                final_output      TEXT,
                total_tokens      INTEGER NOT NULL DEFAULT 0,
                total_latency_ms  INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS spans (
                id              TEXT PRIMARY KEY,
                trace_id        TEXT NOT NULL REFERENCES traces(id),
                parent_span_id  TEXT,
                component_id    TEXT,
                span_type       TEXT NOT NULL,
                input_json      TEXT,
                output_json     TEXT,
                model           TEXT,
                tokens_in       INTEGER,
                tokens_out      INTEGER,
                latency_ms      INTEGER,
                started_at      TEXT NOT NULL,
                details_json    TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_spans_trace_id ON spans(trace_id);

            CREATE TABLE IF NOT EXISTS evaluations (
                id           TEXT PRIMARY KEY,
                run_id       TEXT NOT NULL REFERENCES runs(id),
                span_id      TEXT,
                trace_id     TEXT,
                component_id TEXT,
                metric_name  TEXT NOT NULL,
                metric_class TEXT NOT NULL,
                score        REAL,
                passed       INTEGER,
                details_json TEXT NOT NULL DEFAULT '{}',
                evaluator    TEXT,
                cost_tokens  INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_evaluations_run_component
                ON evaluations(run_id, component_id);

            CREATE TABLE IF NOT EXISTS dataset_cases (
                id             TEXT PRIMARY KEY,
                dataset_id     TEXT NOT NULL,
                input_json     TEXT NOT NULL,
                expected_json  TEXT,
                labels_json    TEXT,
                provenance     TEXT
            );

            INSERT OR IGNORE INTO app_metadata (key, value)
                VALUES ('first_launched_at', datetime('now'));
        """,
    },
    {
        "version": 1,
        "description": "Add map_path and active_defects columns to runs table",
        "sql": """
            ALTER TABLE runs ADD COLUMN map_path TEXT;
            ALTER TABLE runs ADD COLUMN active_defects TEXT;
        """,
    },
]

TARGET_VERSION = len(_MIGRATIONS) - 1


async def init_db() -> None:
    """Open the database, enable WAL mode, and run pending migrations."""
    global _db

    data_dir = os.getenv("AEH_DATA_DIR", ".")
    db_path = Path(data_dir) / "aeh.db"
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
