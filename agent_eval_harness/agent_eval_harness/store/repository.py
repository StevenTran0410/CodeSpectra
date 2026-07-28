"""Result-store repository functions."""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Literal

from agent_eval_harness.instrumentation._extract import utc_now_iso
from agent_eval_harness.instrumentation.base import CapturedSpan
from agent_eval_harness.store.database import get_db

logger = logging.getLogger("agent_eval_harness.store.repository")


def new_id() -> str:
    return str(uuid.uuid4())


async def insert_run(
    target_system_id: str,
    eval_plan_id: str | None = None,
    map_path: str | None = None,
    active_defects: list[str] | None = None,
    target: str | None = None,
    suite_path: str | None = None,
    parent_run_id: str | None = None,
    model_overrides: dict[str, str] | None = None,
    source: Literal["live", "ingested"] = "live",
) -> str:
    run_id = new_id()
    db = get_db()
    active_defects_json = json.dumps(active_defects) if active_defects is not None else None
    model_overrides_json = json.dumps(model_overrides or {})
    await db.execute(
        "INSERT INTO runs (id, target_system_id, eval_plan_id, started_at, status, "
        "map_path, active_defects, target, suite_path, parent_run_id, model_overrides, source) "
        "VALUES (?, ?, ?, ?, 'running', ?, ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            target_system_id,
            eval_plan_id,
            utc_now_iso(),
            map_path,
            active_defects_json,
            target,
            suite_path,
            parent_run_id,
            model_overrides_json,
            source,
        ),
    )
    await db.commit()
    return run_id


async def finish_run(run_id: str, status: Literal["completed", "failed", "partial"]) -> None:
    db = get_db()
    await db.execute(
        "UPDATE runs SET status = ?, finished_at = ? WHERE id = ?",
        (status, utc_now_iso(), run_id),
    )
    await db.commit()


async def insert_trace(
    run_id: str, root_input: str, dataset_case_id: str | None = None
) -> str:
    trace_id = new_id()
    db = get_db()
    await db.execute(
        "INSERT INTO traces (id, run_id, dataset_case_id, root_input) VALUES (?, ?, ?, ?)",
        (trace_id, run_id, dataset_case_id, root_input),
    )
    await db.commit()
    return trace_id


async def finalize_trace(
    trace_id: str, final_output: str, total_tokens: int, total_latency_ms: int
) -> None:
    db = get_db()
    await db.execute(
        "UPDATE traces SET final_output = ?, total_tokens = ?, total_latency_ms = ? WHERE id = ?",
        (final_output, total_tokens, total_latency_ms, trace_id),
    )
    await db.commit()


async def insert_spans_bulk(trace_id: str, spans: list[CapturedSpan]) -> None:
    """Bulk insert spans with stable span_id."""
    db = get_db()
    rows = [
        (
            span.span_id,
            trace_id,
            span.parent_span_id,
            span.component_id,
            span.span_type,
            span.input_json,
            span.output_json,
            span.model,
            span.tokens_in,
            span.tokens_out,
            span.latency_ms,
            span.started_at,
            json.dumps(
                {
                    "tier": span.tier,
                    "token_source": span.token_source,
                    "raw_tags": span.tags,
                }
            ),
        )
        for span in spans
    ]
    await db.executemany(
        "INSERT INTO spans (id, trace_id, parent_span_id, component_id, span_type, "
        "input_json, output_json, model, tokens_in, tokens_out, latency_ms, started_at, "
        "details_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    await db.commit()


async def get_run(run_id: str) -> dict | None:
    db = get_db()
    async with db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)) as cur:
        row = await cur.fetchone()
    return dict(row) if row else None


async def get_traces_for_run(run_id: str) -> list[dict]:
    db = get_db()
    async with db.execute("SELECT * FROM traces WHERE run_id = ?", (run_id,)) as cur:
        rows = await cur.fetchall()
    return [dict(row) for row in rows]


async def list_ingested_runs_for_plan(eval_plan_id: str) -> list[dict]:
    """Every ingested run under one eval plan, newest first, with case and scored-case counts —
    the set Stage 5's run picker switches between so multiple runs of the same cases coexist."""
    db = get_db()
    query = """
        SELECT r.id, r.started_at, r.status,
            (SELECT COUNT(*) FROM traces t WHERE t.run_id = r.id) AS case_count,
            (SELECT COUNT(DISTINCT e.trace_id) FROM evaluations e
             WHERE e.run_id = r.id AND e.metric_name = 'semantic_match') AS scored_count
        FROM runs r
        WHERE r.eval_plan_id = ? AND r.source = 'ingested'
        ORDER BY r.started_at DESC
    """
    async with db.execute(query, (eval_plan_id,)) as cur:
        rows = await cur.fetchall()
    return [dict(row) for row in rows]


async def get_dataset_cases_by_ids(case_ids: list[str]) -> dict[str, dict]:
    """Batch-fetch dataset cases keyed by id, avoiding one query per case in a loop."""
    if not case_ids:
        return {}
    db = get_db()
    placeholders = ",".join("?" * len(case_ids))
    async with db.execute(
        f"SELECT * FROM dataset_cases WHERE id IN ({placeholders})", case_ids
    ) as cur:
        rows = await cur.fetchall()
    return {row["id"]: dict(row) for row in rows}


async def get_spans_for_trace(trace_id: str) -> list[dict]:
    """Get spans for trace ordered by rowid insertion order."""
    db = get_db()
    async with db.execute(
        "SELECT * FROM spans WHERE trace_id = ? ORDER BY rowid", (trace_id,)
    ) as cur:
        rows = await cur.fetchall()
    return [dict(row) for row in rows]


async def get_span_tree(trace_id: str) -> list[dict]:
    """Get flat spans for a run."""
    return await get_spans_for_trace(trace_id)


async def count_unmatched_spans(trace_id: str) -> int:
    db = get_db()
    async with db.execute(
        "SELECT COUNT(*) as n FROM spans WHERE trace_id = ? AND component_id IS NULL",
        (trace_id,),
    ) as cur:
        row = await cur.fetchone()
    return row["n"] if row else 0


async def insert_dataset_cases_bulk(dataset_id: str, cases: list) -> None:
    db = get_db()
    rows = []
    for case in cases:
        if hasattr(case, "model_dump"):
            case_dict = case.model_dump()
        else:
            case_dict = case

        cid = case_dict["id"]
        kind = case_dict["kind"]
        inp = case_dict["input"]
        # kind is stored in input_json as {"kind": kind, **input}
        input_dict = {"kind": kind, **inp}
        input_json = json.dumps(input_dict)

        expected = case_dict.get("expected")
        expected_json = json.dumps(expected) if expected is not None else None

        labels = case_dict.get("labels")
        labels_json = json.dumps(labels) if labels is not None else None

        provenance = case_dict["provenance"]
        rows.append((cid, dataset_id, input_json, expected_json, labels_json, provenance))

    await db.executemany(
        "INSERT INTO dataset_cases "
        "(id, dataset_id, input_json, expected_json, labels_json, provenance) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    await db.commit()


async def get_dataset_cases(dataset_id: str) -> list[dict]:
    db = get_db()
    async with db.execute(
        "SELECT * FROM dataset_cases WHERE dataset_id = ?", (dataset_id,)
    ) as cur:
        rows = await cur.fetchall()
    return [dict(row) for row in rows]


async def get_dataset_case_by_id(case_id: str) -> dict | None:
    db = get_db()
    async with db.execute(
        "SELECT * FROM dataset_cases WHERE id = ?", (case_id,)
    ) as cur:
        row = await cur.fetchone()
    return dict(row) if row else None


async def list_dataset_ids(expansion_session_id: str | None = None) -> list[dict]:
    """Per-dataset case counts, left-joined with `datasets` metadata (a missing row means legacy/unknown-kind).
    `expansion_session_id`, when given, scopes to datasets created by that expansion session only."""
    db = get_db()
    query = """
        SELECT
            c.dataset_id,
            COUNT(*) as total_count,
            SUM(CASE WHEN c.provenance = 'synthetic' THEN 1 ELSE 0 END) as synthetic_count,
            SUM(CASE WHEN c.provenance = 'handwritten' THEN 1 ELSE 0 END) as handwritten_count,
            SUM(CASE WHEN c.provenance = 'generated+reviewed' THEN 1 ELSE 0 END) as reviewed_count,
            d.kind as kind,
            COALESCE(d.min_cases, 1) as min_cases
        FROM dataset_cases c
        LEFT JOIN datasets d ON d.dataset_id = c.dataset_id
    """
    params: tuple = ()
    if expansion_session_id is not None:
        query += " WHERE d.expansion_session_id = ?"
        params = (expansion_session_id,)
    query += " GROUP BY c.dataset_id"

    async with db.execute(query, params) as cur:
        rows = await cur.fetchall()
    results = [dict(row) for row in rows]
    for r in results:
        r["review_complete"] = r["synthetic_count"] == 0 and r["total_count"] >= r["min_cases"]
    return results


async def insert_dataset_metadata(
    dataset_id: str,
    kind: str,
    *,
    instructions: dict | None = None,
    source_gate_ids: list[str] | None = None,
    min_cases: int = 1,
    metrics: dict | None = None,
    expansion_session_id: str | None = None,
) -> None:
    stored = dict(instructions or {})
    if metrics:
        stored["_metrics"] = metrics
    db = get_db()
    await db.execute(
        "INSERT OR REPLACE INTO datasets "
        "(dataset_id, kind, instructions_json, source_gate_ids_json, min_cases, created_at, expansion_session_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            dataset_id,
            kind,
            json.dumps(stored),
            json.dumps(source_gate_ids or []),
            min_cases,
            utc_now_iso(),
            expansion_session_id,
        ),
    )
    await db.commit()


async def get_dataset_metadata(dataset_id: str) -> dict | None:
    db = get_db()
    async with db.execute(
        "SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,)
    ) as cur:
        row = await cur.fetchone()
    return dict(row) if row else None


async def update_case_provenance(
    case_id: str,
    provenance: str,
    *,
    expected_json: str | None = None,
    input_json: str | None = None,
    labels_json: str | None = None,
) -> None:
    """Flip a case's provenance, optionally editing any of its three JSON fields."""
    fields = ["provenance = ?"]
    values: list[str] = [provenance]
    for col, val in (("expected_json", expected_json), ("input_json", input_json), ("labels_json", labels_json)):
        if val is not None:
            fields.append(f"{col} = ?")
            values.append(val)
    values.append(case_id)

    db = get_db()
    await db.execute(
        f"UPDATE dataset_cases SET {', '.join(fields)} WHERE id = ?",
        values,
    )
    await db.commit()


async def delete_dataset_case(case_id: str) -> None:
    db = get_db()
    await db.execute("DELETE FROM dataset_cases WHERE id = ?", (case_id,))
    await db.commit()


async def delete_dataset(dataset_id: str) -> None:
    """Delete a dataset's cases and its metadata row. Idempotent."""
    db = get_db()
    await db.execute("DELETE FROM dataset_cases WHERE dataset_id = ?", (dataset_id,))
    await db.execute("DELETE FROM datasets WHERE dataset_id = ?", (dataset_id,))
    await db.commit()


async def delete_evaluations_for_component(run_id: str, component_id: str) -> int:
    """Clears one component's evaluations so a re-judge replaces them instead of stacking a
    second set of rows on top. Returns how many were removed. Idempotent."""
    db = get_db()
    cursor = await db.execute(
        "DELETE FROM evaluations WHERE run_id = ? AND component_id = ?", (run_id, component_id)
    )
    await db.commit()
    return cursor.rowcount or 0


async def insert_evaluation(
    run_id: str,
    metric_name: str,
    metric_class: str,
    *,
    span_id: str | None = None,
    trace_id: str | None = None,
    component_id: str | None = None,
    score: float | None = None,
    passed: bool | None = None,
    details: dict | None = None,
    evaluator: str | None = None,
    cost_tokens: int | None = None,
    entry_id: str | None = None,
    agent_id: str | None = None,
) -> str:
    eval_id = new_id()
    db = get_db()
    passed_int = int(passed) if passed is not None else None
    await db.execute(
        "INSERT INTO evaluations (id, run_id, span_id, trace_id, component_id, "
        "metric_name, metric_class, score, passed, details_json, evaluator, cost_tokens, "
        "entry_id, agent_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            eval_id,
            run_id,
            span_id,
            trace_id,
            component_id,
            metric_name,
            metric_class,
            score,
            passed_int,
            json.dumps(details or {}),
            evaluator,
            cost_tokens,
            entry_id,
            agent_id,
        ),
    )
    await db.commit()
    return eval_id


async def get_evaluations_for_run(run_id: str) -> list[dict]:
    db = get_db()
    async with db.execute(
        "SELECT * FROM evaluations WHERE run_id = ? ORDER BY rowid", (run_id,)
    ) as cur:
        rows = await cur.fetchall()
    return [dict(row) for row in rows]


async def get_evaluations_for_component(run_id: str, component_id: str) -> list[dict]:
    db = get_db()
    async with db.execute(
        "SELECT * FROM evaluations WHERE run_id = ? AND component_id = ? ORDER BY rowid",
        (run_id, component_id),
    ) as cur:
        rows = await cur.fetchall()
    return [dict(row) for row in rows]


async def list_runs(target_system_id: str | None = None) -> list[dict]:
    db = get_db()
    query = """
        SELECT
            r.id,
            r.target_system_id,
            r.eval_plan_id,
            r.started_at,
            r.finished_at,
            r.status,
            r.map_path,
            r.active_defects,
            COALESCE(
                CAST(SUM(CASE WHEN e.passed = 1 THEN 1 ELSE 0 END) AS REAL) /
                NULLIF(COUNT(CASE WHEN e.passed IS NOT NULL THEN 1 END), 0),
                0.0
            ) as pass_rate,
            COALESCE(
                SUM(CASE WHEN e.metric_class = 'llm_judge' THEN e.cost_tokens ELSE 0 END),
                0
            ) as judge_cost
        FROM runs r
        LEFT JOIN evaluations e ON r.id = e.run_id
    """
    if target_system_id:
        query += " WHERE r.target_system_id = ?"
        query += " GROUP BY r.id ORDER BY r.started_at DESC"
        async with db.execute(query, (target_system_id,)) as cur:
            rows = await cur.fetchall()
    else:
        query += " GROUP BY r.id ORDER BY r.started_at DESC"
        async with db.execute(query) as cur:
            rows = await cur.fetchall()
    return [dict(row) for row in rows]


async def insert_discovery_session(
    repo_ref: str,
    snapshot_id: str,
) -> str:
    session_id = new_id()
    db = get_db()
    await db.execute(
        "INSERT INTO discovery_sessions (id, repo_ref, snapshot_id, status, created_at) "
        "VALUES (?, ?, ?, 'running', ?)",
        (session_id, repo_ref, snapshot_id, utc_now_iso()),
    )
    await db.commit()
    return session_id


async def finish_discovery_session(
    session_id: str,
    status: Literal["completed", "failed"],
    error: str | None = None,
) -> None:
    db = get_db()
    if status == "completed":
        await db.execute(
            "UPDATE discovery_sessions SET status = ?, error = ?, pipeline_stage = 'awaiting_candidate_review', finished_at = ? WHERE id = ?",
            (status, error, utc_now_iso(), session_id),
        )
    else:
        await db.execute(
            "UPDATE discovery_sessions SET status = ?, error = ?, finished_at = ? WHERE id = ?",
            (status, error, utc_now_iso(), session_id),
        )
    await db.commit()


async def update_discovery_session_project_context(
    session_id: str, ctx: Any,
) -> None:
    """Update a discovery session's project context JSON; a no-op if ctx is None (degrade-don't-break)."""
    if ctx is None:
        return

    import dataclasses
    db = get_db()
    ctx_dict = dataclasses.asdict(ctx)
    ctx_json = json.dumps(ctx_dict)
    await db.execute(
        "UPDATE discovery_sessions SET project_context_json = ? WHERE id = ?",
        (ctx_json, session_id),
    )
    await db.commit()


async def insert_discovery_candidates_bulk(
    session_id: str,
    candidates: list[dict],
) -> None:
    db = get_db()
    rows = [
        (
            new_id(),
            session_id,
            c["name"],
            json.dumps(c.get("frameworks", [])),
            json.dumps(c.get("entry_points", [])),
            json.dumps(c.get("evidence", [])),
            c.get("confidence", "low"),
            c.get("verdict", "proposed"),
            int(bool(c.get("needs_human", False))),
            c.get("community_id"),
            json.dumps(c.get("cluster_files", [])),
            json.dumps(c.get("hub_paths", [])),
            json.dumps(c.get("wiring_block")) if c.get("wiring_block") is not None else None,
            json.dumps(c.get("excluded_files", [])),
            json.dumps(c.get("matched_files", [])),
            json.dumps(c.get("file_provenance", {})),
            json.dumps(c.get("risk_flags", [])),
            c.get("map_scope_framework"),
            json.dumps(c.get("excluded_component_classes", [])),
            c.get("system_type"),
            json.dumps(c.get("system_type_signals", {})),
        )
        for c in candidates
    ]
    if rows:
        await db.executemany(
            "INSERT INTO discovery_candidates (id, session_id, name, frameworks_json, "
            "entry_points_json, evidence_json, confidence, verdict, needs_human, "
            "community_id, cluster_files_json, hub_paths_json, wiring_block_json, "
            "excluded_files_json, matched_files_json, file_provenance_json, risk_flags_json, "
            "map_scope_framework, excluded_component_classes_json, system_type, "
            "system_type_signals_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        await db.commit()


async def _decode_discovery_session_row(d: dict) -> dict:
    """Expand a discovery_sessions row's *_json column and resolve its derived pipeline_stage."""
    d["pause_info"] = json.loads(d["pause_info_json"]) if d.get("pause_info_json") else None
    d["analysis_context"] = "available" if d.get("project_context_json") else "unavailable"
    d["pipeline_stage"] = await _resolve_effective_pipeline_stage(d)
    return d


def _decode_discovery_candidate_row(d: dict) -> dict:
    """Expand a discovery_candidates row's *_json columns into their parsed list/dict fields."""
    d["frameworks"] = json.loads(d["frameworks_json"] or "[]")
    d["entry_points"] = json.loads(d["entry_points_json"] or "[]")
    d["evidence"] = json.loads(d["evidence_json"] or "[]")
    d["needs_human"] = bool(d.get("needs_human", 0))
    d["cluster_files"] = json.loads(d.get("cluster_files_json") or "[]")
    d["hub_paths"] = json.loads(d.get("hub_paths_json") or "[]")
    d["wiring_block"] = json.loads(d["wiring_block_json"]) if d.get("wiring_block_json") else None
    d["excluded_files"] = json.loads(d.get("excluded_files_json") or "[]")
    d["matched_files"] = json.loads(d.get("matched_files_json") or "[]")
    d["file_provenance"] = json.loads(d.get("file_provenance_json") or "{}")
    d["risk_flags"] = json.loads(d.get("risk_flags_json") or "[]")
    d["map_scope_framework"] = d.get("map_scope_framework")
    d["excluded_component_classes"] = json.loads(d.get("excluded_component_classes_json") or "[]")
    d["system_type"] = d.get("system_type")
    d["system_type_signals"] = json.loads(d.get("system_type_signals_json") or "{}")
    return d


def _decode_expansion_session_row(d: dict) -> dict:
    """Expand an expansion_sessions row's *_json columns into their parsed list fields."""
    d["accepted"] = json.loads(d.get("accepted_json") or "[]")
    # Backward-compat: convert old str format to new dict format
    d["accepted"] = [
        item if isinstance(item, dict)
        else {"file": item, "role_hint": None, "key_symbols": [], "follow": False}
        for item in d["accepted"]
    ]
    d["boundary"] = json.loads(d.get("boundary_json") or "[]")
    d["accepted_edges"] = json.loads(d.get("accepted_edges_json") or "[]")
    return d


async def _candidate_ids_with_completed_expansion(candidate_ids: list[str]) -> set[str]:
    """Batch-check which of these candidate ids have >=1 completed expansion session (one query, not one per id)."""
    if not candidate_ids:
        return set()
    db = get_db()
    placeholders = ",".join("?" * len(candidate_ids))
    async with db.execute(
        f"SELECT DISTINCT candidate_id FROM expansion_sessions "
        f"WHERE candidate_id IN ({placeholders}) AND status = 'completed'",
        candidate_ids,
    ) as cur:
        rows = await cur.fetchall()
    return {r[0] for r in rows}


async def _resolve_effective_pipeline_stage(d: dict) -> str:
    db = get_db()
    stage = d.get("pipeline_stage", "fingerprinting")
    if stage != "fingerprinting":
        return stage
    if d.get("status") == "running":
        return "fingerprinting"

    async with db.execute(
        "SELECT id FROM discovery_candidates WHERE session_id = ? AND verdict = 'confirmed'",
        (d["id"],)
    ) as cur:
        confirmed_rows = await cur.fetchall()
    confirmed_ids = [r[0] for r in confirmed_rows]

    if not confirmed_ids:
        return "awaiting_candidate_review"

    completed = await _candidate_ids_with_completed_expansion(confirmed_ids)
    return "awaiting_map_review" if completed else "expanding"


async def list_discovery_sessions(
    repo_ref: str | None = None, snapshot_id: str | None = None
) -> list[dict]:
    """List discovery sessions with optional repo_ref and snapshot_id filtering."""
    db = get_db()
    if snapshot_id:
        async with db.execute(
            "SELECT * FROM discovery_sessions WHERE snapshot_id = ? ORDER BY created_at DESC",
            (snapshot_id,),
        ) as cur:
            rows = await cur.fetchall()
    elif repo_ref:
        async with db.execute(
            "SELECT * FROM discovery_sessions WHERE repo_ref = ? ORDER BY created_at DESC",
            (repo_ref,),
        ) as cur:
            rows = await cur.fetchall()
    else:
        async with db.execute(
            "SELECT * FROM discovery_sessions ORDER BY created_at DESC"
        ) as cur:
            rows = await cur.fetchall()
    return [await _decode_discovery_session_row(dict(r)) for r in rows]


async def get_discovery_session(session_id: str) -> dict | None:
    db = get_db()
    async with db.execute(
        "SELECT * FROM discovery_sessions WHERE id = ?",
        (session_id,),
    ) as cur:
        row = await cur.fetchone()
    if not row:
        return None
    return await _decode_discovery_session_row(dict(row))


async def pause_discovery_session(
    session_id: str,
    provider_id: str,
    model_id: str | None,
    reasoning_effort: str | None = None,
    thinking_budget: int | None = None,
) -> None:
    db = get_db()
    pause_info = json.dumps({
        "reason": "rate_limited",
        "provider_id": provider_id,
        "model_id": model_id,
        "reasoning_effort": reasoning_effort,
        "thinking_budget": thinking_budget,
    })
    await db.execute(
        "UPDATE discovery_sessions SET status = 'paused_rate_limit', pause_info_json = ? WHERE id = ?",
        (pause_info, session_id),
    )
    await db.commit()


async def resume_discovery_session(session_id: str) -> None:
    db = get_db()
    await db.execute(
        "UPDATE discovery_sessions SET status = 'running', pause_info_json = NULL WHERE id = ?",
        (session_id,),
    )
    await db.commit()


async def replace_discovery_candidates(session_id: str, candidates: list[dict]) -> None:
    """Re-derive a session's candidates (on completion, and on rate-limit pause/resume) without
    wiping a verdict the user already set -- insert_discovery_candidates_bulk defaults verdict to
    'proposed', so a naive replace silently un-confirms/un-rejects every candidate on resume."""
    db = get_db()
    async with db.execute(
        "SELECT name, verdict FROM discovery_candidates WHERE session_id = ?", (session_id,),
    ) as cur:
        existing_verdicts = {row["name"]: row["verdict"] for row in await cur.fetchall()}
    await db.execute("DELETE FROM discovery_candidates WHERE session_id = ?", (session_id,))
    await db.commit()
    carried = [
        {**c, "verdict": existing_verdicts.get(c["name"], c.get("verdict", "proposed"))}
        for c in candidates
    ]
    await insert_discovery_candidates_bulk(session_id, carried)


async def get_discovery_candidates(session_id: str) -> list[dict]:
    db = get_db()
    async with db.execute(
        "SELECT * FROM discovery_candidates WHERE session_id = ? ORDER BY name ASC",
        (session_id,),
    ) as cur:
        rows = await cur.fetchall()
    return [_decode_discovery_candidate_row(dict(r)) for r in rows]


async def update_candidate_verdict(
    candidate_id: str,
    verdict: Literal["proposed", "confirmed", "rejected"],
) -> None:
    db = get_db()
    await db.execute(
        "UPDATE discovery_candidates SET verdict = ? WHERE id = ?",
        (verdict, candidate_id),
    )
    await db.commit()


async def update_candidate_excluded_files(candidate_id: str, excluded_files: list[str]) -> None:
    db = get_db()
    await db.execute(
        "UPDATE discovery_candidates SET excluded_files_json = ? WHERE id = ?",
        (json.dumps(excluded_files), candidate_id),
    )
    await db.commit()


async def get_discovery_candidate(candidate_id: str) -> dict | None:
    db = get_db()
    async with db.execute(
        "SELECT * FROM discovery_candidates WHERE id = ?",
        (candidate_id,),
    ) as cur:
        row = await cur.fetchone()
    if not row:
        return None
    return _decode_discovery_candidate_row(dict(row))


async def insert_expansion_session(session_id: str, candidate_id: str, snapshot_id: str) -> None:
    db = get_db()
    await db.execute(
        "INSERT INTO expansion_sessions (id, candidate_id, snapshot_id, status, created_at) "
        "VALUES (?, ?, ?, 'running', ?)",
        (session_id, candidate_id, snapshot_id, utc_now_iso()),
    )
    await db.commit()


async def finish_expansion_session(
    session_id: str,
    status: Literal["completed", "failed"],
    error: str | None = None,
    map_path: str | None = None,
    accepted: list[str] = [],
    boundary: list[str] = [],
    stop_reason: str | None = None,
    accepted_edges: list[dict] = [],
) -> None:
    db = get_db()
    await db.execute(
        "UPDATE expansion_sessions "
        "SET status = ?, error = ?, map_path = ?, accepted_json = ?, boundary_json = ?, "
        "    stop_reason = ?, accepted_edges_json = ?, finished_at = ? "
        "WHERE id = ?",
        (
            status,
            error,
            map_path,
            json.dumps(accepted),
            json.dumps(boundary),
            stop_reason,
            json.dumps(accepted_edges),
            utc_now_iso(),
            session_id,
        ),
    )
    await db.commit()

    if status == "completed":
        try:
            async with db.execute("SELECT candidate_id FROM expansion_sessions WHERE id = ?", (session_id,)) as cur:
                row = await cur.fetchone()
            if row:
                candidate_id = row[0]
                candidate = await get_discovery_candidate(candidate_id)
                if candidate:
                    parent_session_id = candidate["session_id"]
                    all_candidates = await get_discovery_candidates(parent_session_id)
                    confirmed_cands = [c for c in all_candidates if c["verdict"] == "confirmed"]

                    if confirmed_cands:
                        confirmed_ids = [c["id"] for c in confirmed_cands]
                        completed_ids = await _candidate_ids_with_completed_expansion(confirmed_ids)
                        all_completed = set(confirmed_ids) <= completed_ids

                        if all_completed:
                            await db.execute(
                                "UPDATE discovery_sessions SET pipeline_stage = 'awaiting_map_review' WHERE id = ?",
                                (parent_session_id,)
                            )
                            await db.commit()
        except Exception as e:
            logger.error(f"Error checking sibling expansion sessions completion: {e}", exc_info=True)


async def get_expansion_session(session_id: str) -> dict | None:
    db = get_db()
    async with db.execute(
        "SELECT * FROM expansion_sessions WHERE id = ?",
        (session_id,),
    ) as cur:
        row = await cur.fetchone()
    if not row:
        return None
    return _decode_expansion_session_row(dict(row))


async def list_expansion_sessions_for_candidate(candidate_id: str) -> list[dict]:
    db = get_db()
    async with db.execute(
        "SELECT * FROM expansion_sessions WHERE candidate_id = ? ORDER BY created_at DESC",
        (candidate_id,),
    ) as cur:
        rows = await cur.fetchall()
    return [_decode_expansion_session_row(dict(r)) for r in rows]


async def update_expansion_session_plan_path(session_id: str, plan_path: str | None) -> None:
    db = get_db()
    await db.execute(
        "UPDATE expansion_sessions SET plan_path = ? WHERE id = ?",
        (plan_path, session_id),
    )
    await db.commit()


async def update_expansion_session_agentflows_path(session_id: str, agent_flows_path: str) -> None:
    db = get_db()
    await db.execute(
        "UPDATE expansion_sessions SET agent_flows_path = ? WHERE id = ?",
        (agent_flows_path, session_id),
    )
    await db.commit()


async def update_expansion_session_plan_report_path(session_id: str, plan_report_path: str | None) -> None:
    db = get_db()
    await db.execute(
        "UPDATE expansion_sessions SET plan_report_path = ? WHERE id = ?",
        (plan_report_path, session_id),
    )
    await db.commit()


async def update_expansion_session_eval_branch(
    session_id: str, branch_name: str, original_branch: str
) -> None:
    db = get_db()
    await db.execute(
        "UPDATE expansion_sessions SET eval_branch_name = ?, eval_original_branch = ? WHERE id = ?",
        (branch_name, original_branch, session_id),
    )
    await db.commit()


async def update_expansion_session_eval_plan_md_path(
    session_id: str, plan_md_path: str
) -> None:
    db = get_db()
    await db.execute(
        "UPDATE expansion_sessions SET eval_plan_md_path = ? WHERE id = ?",
        (plan_md_path, session_id),
    )
    await db.commit()


async def update_expansion_session_eval_run(
    session_id: str, run_id: str, manifest_run_id: str | None
) -> None:
    db = get_db()
    await db.execute(
        "UPDATE expansion_sessions SET eval_run_id = ?, eval_manifest_run_id = ? WHERE id = ?",
        (run_id, manifest_run_id, session_id),
    )
    await db.commit()


async def cancel_orphaned_running_sessions() -> None:
    """Mark any status='running' session as failed — background tasks don't persist across process restarts, so a 'running' row at startup is always a leftover from a killed process."""
    db = get_db()
    now = utc_now_iso()
    error = "Interrupted: app closed before this run finished."
    await db.execute(
        "UPDATE discovery_sessions SET status = 'failed', error = ?, finished_at = ? "
        "WHERE status = 'running'",
        (error, now),
    )
    await db.execute(
        "UPDATE expansion_sessions SET status = 'failed', error = ?, finished_at = ? "
        "WHERE status = 'running'",
        (error, now),
    )
    await db.commit()


async def update_discovery_session_pipeline_stage(session_id: str, stage: str) -> None:
    db = get_db()
    await db.execute(
        "UPDATE discovery_sessions SET pipeline_stage = ? WHERE id = ?",
        (stage, session_id),
    )
    await db.commit()


async def upsert_agent_knowledge(
    session_id: str,
    agent_id: str,
    md_path: str,
    json_path: str,
    evidence_hash: str,
    confidence: str,
    query_count: int,
) -> None:
    """Insert or replace agent knowledge record."""
    db = get_db()
    await db.execute(
        "INSERT OR REPLACE INTO agent_knowledge "
        "(session_id, agent_id, md_path, json_path, evidence_hash, confidence, query_count, generated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (session_id, agent_id, md_path, json_path, evidence_hash, confidence, query_count, utc_now_iso()),
    )
    await db.commit()


async def get_agent_knowledge(session_id: str, agent_id: str) -> dict | None:
    """Retrieve agent knowledge record."""
    db = get_db()
    async with db.execute(
        "SELECT * FROM agent_knowledge WHERE session_id = ? AND agent_id = ?",
        (session_id, agent_id),
    ) as cur:
        row = await cur.fetchone()
    if not row:
        return None
    return dict(row)


async def list_agent_knowledge(session_id: str) -> list[dict]:
    """List all agent knowledge records for a session."""
    db = get_db()
    async with db.execute(
        "SELECT * FROM agent_knowledge WHERE session_id = ? ORDER BY agent_id",
        (session_id,),
    ) as cur:
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def get_model_call_verdict(cache_key: str) -> dict | None:
    """Retrieve a cached LLM residue-pass verdict, keyed on evidence content, not run identity."""
    db = get_db()
    async with db.execute(
        "SELECT * FROM model_call_verdicts WHERE cache_key = ?", (cache_key,),
    ) as cur:
        row = await cur.fetchone()
    return dict(row) if row else None


async def upsert_model_call_verdict(
    cache_key: str, makes_model_call: bool, source: str, citation: str | None, evidence_kind: str | None,
) -> None:
    """Insert or replace a cited-and-verified LLM residue-pass verdict."""
    db = get_db()
    await db.execute(
        "INSERT OR REPLACE INTO model_call_verdicts "
        "(cache_key, makes_model_call, source, citation, evidence_kind, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (cache_key, int(makes_model_call), source, citation, evidence_kind, utc_now_iso()),
    )
    await db.commit()
