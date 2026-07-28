"""Async SQLite store for aeh.db — module-level connection, migrations-as-data, WAL mode, fully independent of codespectra.db."""
import logging
import os
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger("agent_eval_harness.store")

DB_FILENAME = "aeh.db"

_db: aiosqlite.Connection | None = None

# Migrations — add new dicts at the end; never modify existing ones.
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
    {
        "version": 2,
        "description": "Add target/suite_path/parent_run_id/model_overrides for reruns",
        "sql": """
            ALTER TABLE runs ADD COLUMN target TEXT;
            ALTER TABLE runs ADD COLUMN suite_path TEXT;
            ALTER TABLE runs ADD COLUMN parent_run_id TEXT;
            ALTER TABLE runs ADD COLUMN model_overrides TEXT NOT NULL DEFAULT '{}';
        """,
    },
    {
        "version": 3,
        "description": "Add discovery_sessions and discovery_candidates tables",
        "sql": """
            CREATE TABLE IF NOT EXISTS discovery_sessions (
                id           TEXT PRIMARY KEY,
                repo_ref     TEXT NOT NULL,
                snapshot_id  TEXT NOT NULL,
                status       TEXT NOT NULL,
                error        TEXT,
                created_at   TEXT NOT NULL,
                finished_at  TEXT
            );

            CREATE TABLE IF NOT EXISTS discovery_candidates (
                id                TEXT PRIMARY KEY,
                session_id        TEXT NOT NULL REFERENCES discovery_sessions(id) ON DELETE CASCADE,
                name              TEXT NOT NULL,
                frameworks_json   TEXT NOT NULL,
                entry_points_json TEXT NOT NULL,
                evidence_json     TEXT NOT NULL,
                confidence        TEXT NOT NULL,
                verdict           TEXT NOT NULL DEFAULT 'proposed'
            );
            CREATE INDEX IF NOT EXISTS idx_candidates_session_id ON discovery_candidates(session_id);
        """,
    },
    {
        "version": 4,
        "description": "Add needs_human flag to discovery_candidates — distinguishes "
        "'reviewed, low confidence' from 'not confidently classified' (epic §4e)",
        "sql": """
            ALTER TABLE discovery_candidates ADD COLUMN needs_human INTEGER NOT NULL DEFAULT 0;
        """,
    },
    {
        "version": 5,
        "description": "Add community_id/cluster_files/hub_paths to discovery_candidates — "
        "Pass B already computes these per cluster (engine.py), this just stops discarding "
        "them so the UI can show the full community a candidate came from, not just its "
        "fingerprint-hit subset",
        "sql": """
            ALTER TABLE discovery_candidates ADD COLUMN community_id TEXT;
            ALTER TABLE discovery_candidates ADD COLUMN cluster_files_json TEXT NOT NULL DEFAULT '[]';
            ALTER TABLE discovery_candidates ADD COLUMN hub_paths_json TEXT NOT NULL DEFAULT '[]';
        """,
    },
    {
        "version": 6,
        "description": "Add expansion_sessions table (CS-271) for Stage 2 iterative expansion",
        "sql": """
            CREATE TABLE IF NOT EXISTS expansion_sessions (
                id            TEXT PRIMARY KEY,
                candidate_id  TEXT NOT NULL,
                snapshot_id   TEXT NOT NULL,
                status        TEXT NOT NULL,
                error         TEXT,
                map_path      TEXT,
                accepted_json TEXT NOT NULL DEFAULT '[]',
                boundary_json TEXT NOT NULL DEFAULT '[]',
                stop_reason   TEXT,
                created_at    TEXT NOT NULL,
                finished_at   TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_expansion_sessions_candidate ON expansion_sessions(candidate_id);
        """,
    },
    {
        "version": 7,
        "description": "Add wiring_block_json to discovery_candidates — Pass D's detected "
        "framework wiring block (Haystack connect()/LangGraph add_edge()/LangChain LCEL), "
        "or null if none found",
        "sql": """
            ALTER TABLE discovery_candidates ADD COLUMN wiring_block_json TEXT;
        """,
    },
    {
        "version": 8,
        "description": "Add excluded_files_json to discovery_candidates — lets a reviewer "
        "remove one bad file from a candidate's cluster without rejecting the whole candidate",
        "sql": """
            ALTER TABLE discovery_candidates ADD COLUMN excluded_files_json TEXT NOT NULL DEFAULT '[]';
        """,
    },
    {
        "version": 9,
        "description": "Add pause_info_json to discovery_sessions and 'paused_rate_limit' as a "
        "possible status — lets a run pause on a sustained rate limit instead of silently "
        "degrading or failing, so the user can wait or switch model and resume",
        "sql": """
            ALTER TABLE discovery_sessions ADD COLUMN pause_info_json TEXT;
        """,
    },
    {
        "version": 10,
        "description": "Add matched_files_json and file_provenance_json to discovery_candidates — "
        "Pass E's cross-candidate core-seed consolidation, corrects Pass B's Louvain "
        "community assignment using Pass D's structural wiring evidence",
        "sql": """
            ALTER TABLE discovery_candidates ADD COLUMN matched_files_json TEXT NOT NULL DEFAULT '[]';
            ALTER TABLE discovery_candidates ADD COLUMN file_provenance_json TEXT NOT NULL DEFAULT '{}';
        """,
    },
    {
        "version": 11,
        "description": "Add accepted_edges_json to expansion_sessions — the real file-to-file "
        "edges discovered during BFS expansion, so Stage 2's graph can show every accepted "
        "file (not just the narrow framework-recognized component subset)",
        "sql": """
            ALTER TABLE expansion_sessions ADD COLUMN accepted_edges_json TEXT NOT NULL DEFAULT '[]';
        """,
    },
    {
        "version": 12,
        "description": "Add plan_path to expansion_sessions (CS-273) and pipeline_stage to "
        "discovery_sessions (CS-274) - explicit pipeline state tracking instead of "
        "inferring from child record presence",
        "sql": """
            ALTER TABLE expansion_sessions ADD COLUMN plan_path TEXT;
            ALTER TABLE discovery_sessions ADD COLUMN pipeline_stage TEXT NOT NULL DEFAULT 'fingerprinting';
        """,
    },
    {
        "version": 13,
        "description": "Add agent_flows_path to expansion_sessions - the LLM-2 holistic "
        "agent-flow separation pass output (Stage 2 agent-flow view), a sibling YAML next "
        "to the map, same pattern as plan_path",
        "sql": """
            ALTER TABLE expansion_sessions ADD COLUMN agent_flows_path TEXT;
        """,
    },
    {
        "version": 14,
        "description": "Add plan_report_path to expansion_sessions - the Stage 3 agentic "
        "eval planner's per-agent report (data profiles, gates, advisory notes), a sibling "
        "YAML next to the map, same pattern as plan_path/agent_flows_path",
        "sql": """
            ALTER TABLE expansion_sessions ADD COLUMN plan_report_path TEXT;
        """,
    },
    {
        "version": 15,
        "description": "Add entry_id and agent_id to evaluations (CS-281 §6) — the run↔plan "
        "linkage CS-283 ingest scoring and the reports UI join on",
        "sql": """
            ALTER TABLE evaluations ADD COLUMN entry_id TEXT;
            ALTER TABLE evaluations ADD COLUMN agent_id TEXT;
        """,
    },
    {
        "version": 16,
        "description": "Add datasets metadata table (CS-282 §2) — kills the kind-sniffing "
        "hack (_resolve_dataset_ref reading input_json['kind'] off every case of every "
        "dataset); dataset_cases has no FK to it (pre-existing rows have no metadata row, "
        "left as-is — repository functions treat a missing metadata row as legacy/unknown-kind)",
        "sql": """
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id           TEXT PRIMARY KEY,
                kind                 TEXT NOT NULL,
                instructions_json    TEXT NOT NULL DEFAULT '{}',
                source_gate_ids_json TEXT NOT NULL DEFAULT '[]',
                min_cases            INTEGER NOT NULL DEFAULT 1,
                created_at           TEXT NOT NULL
            );
        """,
    },
    {
        "version": 17,
        "description": "Add runs.source (CS-283 ingest) — distinguishes a live in-process "
        "sweep run from a run ingested from an injected target's own spanlog; existing rows "
        "default to 'live' since that's the only kind that existed before this.",
        "sql": """
            ALTER TABLE runs ADD COLUMN source TEXT NOT NULL DEFAULT 'live';
        """,
    },
    {
        "version": 18,
        "description": "Add eval branch/plan columns to expansion_sessions (CS-284)",
        "sql": (
            "ALTER TABLE expansion_sessions ADD COLUMN eval_branch_name TEXT;\n"
            "ALTER TABLE expansion_sessions ADD COLUMN eval_original_branch TEXT;\n"
            "ALTER TABLE expansion_sessions ADD COLUMN eval_plan_md_path TEXT;"
        ),
    },
    {
        "version": 19,
        "description": "Add eval_run_id/eval_manifest_run_id to expansion_sessions — lets "
        "eval-results be idempotent per target-side manifest run_id (re-clicking Load "
        "Results with an unchanged manifest returns the existing AEH run instead of "
        "inserting a duplicate) and lets the UI recall the last loaded run across "
        "navigation instead of forcing a fresh click every visit.",
        "sql": (
            "ALTER TABLE expansion_sessions ADD COLUMN eval_run_id TEXT;\n"
            "ALTER TABLE expansion_sessions ADD COLUMN eval_manifest_run_id TEXT;"
        ),
    },
    {
        "version": 20,
        "description": "Add project_context_json to discovery_sessions and risk_flags_json "
        "to discovery_candidates — persist code-analysis report context (CS-290 §B1) and "
        "risk flag propagation from analysis to candidate profiles (CS-290 §B3)",
        "sql": (
            "ALTER TABLE discovery_sessions ADD COLUMN project_context_json TEXT;\n"
            "ALTER TABLE discovery_candidates ADD COLUMN risk_flags_json TEXT NOT NULL DEFAULT '[]';"
        ),
    },
    {
        "version": 21,
        "description": "Add agent_knowledge table for enrichment results (CS-291 Phase 2)",
        "sql": (
            "CREATE TABLE IF NOT EXISTS agent_knowledge (\n"
            "    session_id TEXT NOT NULL,\n"
            "    agent_id TEXT NOT NULL,\n"
            "    md_path TEXT NOT NULL,\n"
            "    json_path TEXT NOT NULL,\n"
            "    evidence_hash TEXT NOT NULL,\n"
            "    confidence TEXT NOT NULL,\n"
            "    query_count INTEGER NOT NULL DEFAULT 0,\n"
            "    generated_at TEXT NOT NULL,\n"
            "    PRIMARY KEY (session_id, agent_id)\n"
            ");"
        ),
    },
    {
        "version": 22,
        "description": "Persist per-system map scoping on discovery_candidates (CS-319): a split "
        "candidate's map_scope_framework + excluded_component_classes were computed at discovery "
        "but never stored, so expansion rebuilt with no scope filter and co-located sibling classes "
        "leaked into the map. Store them so the scope reaches build_from_files.",
        "sql": (
            "ALTER TABLE discovery_candidates ADD COLUMN map_scope_framework TEXT;\n"
            "ALTER TABLE discovery_candidates ADD COLUMN excluded_component_classes_json TEXT NOT NULL DEFAULT '[]';"
        ),
    },
    {
        "version": 23,
        "description": "Persist structural system-type classification (CS-320): each discovered "
        "system is classified into an agentic-system type (pipeline/orchestrator/routing/etc.) "
        "and agent type (single-flow/tool-loop/etc.) via structural signals + optional LLM "
        "confirmation. system_type stores the primary type (or NULL for ambiguous/unresolved); "
        "system_type_signals_json stores the full struct with confidence/candidates/signals.",
        "sql": (
            "ALTER TABLE discovery_candidates ADD COLUMN system_type TEXT;\n"
            "ALTER TABLE discovery_candidates ADD COLUMN system_type_signals_json TEXT NOT NULL DEFAULT '{}';"
        ),
    },
    {
        "version": 24,
        "description": "Add model_call_verdicts table - global cache for the LLM residue pass over "
        "an incomplete-closure component; keyed on a content hash of its evidence (never "
        "snapshot_id or session_id), so an unrelated commit or re-run reuses the verdict at zero "
        "token cost",
        "sql": (
            "CREATE TABLE IF NOT EXISTS model_call_verdicts (\n"
            "    cache_key        TEXT PRIMARY KEY,\n"
            "    makes_model_call INTEGER NOT NULL,\n"
            "    source           TEXT NOT NULL,\n"
            "    citation         TEXT,\n"
            "    evidence_kind    TEXT,\n"
            "    created_at       TEXT NOT NULL\n"
            ");"
        ),
    },
    {
        "version": 25,
        "description": "Add expansion_session_id to datasets - scopes the Dataset Review list to "
        "the system whose expansion session produced it (snapshot_id is shared across sibling "
        "systems, so it can't be used as the scope key); nullable, existing rows stay unscoped.",
        "sql": (
            "ALTER TABLE datasets ADD COLUMN expansion_session_id TEXT;"
        ),
    },
]

TARGET_VERSION = len(_MIGRATIONS) - 1


async def init_db() -> None:
    """Open the database, enable WAL mode, and run pending migrations."""
    global _db

    data_dir = os.getenv("AEH_DATA_DIR", ".")
    db_path = Path(data_dir) / DB_FILENAME
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


# Internal migration runner
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
