import json
from pathlib import Path

import pytest

from agent_eval_harness.ingest.spanlog_ingest import (
    IngestError,
    parse_spanlog,
    persist_spanlog,
    referential_dry_run,
)
from agent_eval_harness.store import repository
from agent_eval_harness.store.database import close_db, init_db


@pytest.fixture(autouse=True)
async def _setup_db(tmp_path, monkeypatch):
    monkeypatch.setenv("AEH_DATA_DIR", str(tmp_path))
    await init_db()
    yield
    await close_db()


def _write_log(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def _clean_log_records() -> list[dict]:
    return [
        {"record": "header", "schema": "aeh.spanlog/1", "tracer_version": "1", "plan_id": "p1", "run_id": "r1"},
        {"record": "case_start", "trace_id": "t1", "dataset_case_id": "case-1", "input": {"snapshot_id": "s1"}},
        {
            "record": "span", "trace_id": "t1", "span_id": "sp1", "parent_span_id": None,
            "component_id": "project_identity", "span_type": "agent", "operation": "haystack.component.run",
            "started_at": "2026-01-01T00:00:00.000Z", "latency_ms": 120,
            "input_json": '{"snapshot_id": "s1"}', "output_json": '{"domain": "web"}',
            "model": "gpt-5", "tokens_in": 100, "tokens_out": 50, "token_source": "measured",
        },
        {"record": "case_end", "trace_id": "t1", "status": "ok", "final_output_json": '{"domain": "web"}'},
        {"record": "run_summary", "attempted": 1, "succeeded": 1},
    ]


async def test_parse_and_persist_clean_run_marks_completed(tmp_path: Path) -> None:
    log_path = tmp_path / "eval_log.123.jsonl"
    _write_log(log_path, _clean_log_records())

    parsed = parse_spanlog(log_path)
    run_id = await persist_spanlog(parsed, target_system_id="codespectra", eval_plan_id="p1")

    run = await repository.get_run(run_id)
    assert run["status"] == "completed"
    assert run["source"] == "ingested"


async def test_dataset_case_id_is_populated_on_the_trace(tmp_path: Path) -> None:
    """Regression guard: traces.dataset_case_id has existed since migration v0 but was never populated until ingest."""
    log_path = tmp_path / "eval_log.124.jsonl"
    _write_log(log_path, _clean_log_records())
    parsed = parse_spanlog(log_path)

    run_id = await persist_spanlog(parsed, target_system_id="codespectra")

    from agent_eval_harness.store.database import get_db
    db = get_db()
    async with db.execute("SELECT * FROM traces WHERE run_id = ?", (run_id,)) as cur:
        row = await cur.fetchone()
    assert dict(row)["dataset_case_id"] == "case-1"


async def test_span_input_output_are_persisted_and_redacted(tmp_path: Path) -> None:
    records = _clean_log_records()
    for r in records:
        if r.get("record") == "span":
            r["output_json"] = '{"api_key": "sk-abcdef1234567890", "domain": "web"}'
    log_path = tmp_path / "eval_log.125.jsonl"
    _write_log(log_path, records)
    parsed = parse_spanlog(log_path)

    run_id = await persist_spanlog(parsed, target_system_id="codespectra")

    from agent_eval_harness.store.database import get_db
    db = get_db()
    async with db.execute(
        "SELECT s.output_json FROM spans s JOIN traces t ON s.trace_id = t.id WHERE t.run_id = ?",
        (run_id,),
    ) as cur:
        row = await cur.fetchone()
    assert "sk-abcdef1234567890" not in row["output_json"]
    assert "[REDACTED]" in row["output_json"]
    assert "web" in row["output_json"]  # non-secret content untouched


async def test_case_with_error_status_marks_run_partial_not_completed(tmp_path: Path) -> None:
    records = _clean_log_records()
    for r in records:
        if r.get("record") == "case_end":
            r["status"] = "error"
    log_path = tmp_path / "eval_log.126.jsonl"
    _write_log(log_path, records)
    parsed = parse_spanlog(log_path)

    run_id = await persist_spanlog(parsed, target_system_id="codespectra")

    run = await repository.get_run(run_id)
    assert run["status"] == "partial"


async def test_case_with_zero_spans_marks_run_partial() -> None:
    from agent_eval_harness.ingest.spanlog_ingest import ParsedSpanlog

    parsed = ParsedSpanlog(
        header={"schema": "aeh.spanlog/1"},
        cases={"t1": {"dataset_case_id": "c1", "root_input": "{}", "final_output": "{}", "status": "ok", "spans": []}},
    )

    run_id = await persist_spanlog(parsed, target_system_id="codespectra")

    run = await repository.get_run(run_id)
    assert run["status"] == "partial"


def test_parse_rejects_unknown_schema_major(tmp_path: Path) -> None:
    log_path = tmp_path / "eval_log.127.jsonl"
    _write_log(log_path, [{"record": "header", "schema": "aeh.spanlog/2"}])

    with pytest.raises(IngestError, match="unsupported spanlog schema"):
        parse_spanlog(log_path)


def test_parse_rejects_missing_header(tmp_path: Path) -> None:
    log_path = tmp_path / "eval_log.128.jsonl"
    _write_log(log_path, [{"record": "case_start", "trace_id": "t1"}])

    with pytest.raises(IngestError, match="no header record"):
        parse_spanlog(log_path)


def test_referential_dry_run_counts_unmatched_component_ids(tmp_path: Path) -> None:
    log_path = tmp_path / "eval_log.129.jsonl"
    records = _clean_log_records()
    records.append({
        "record": "span", "trace_id": "t1", "span_id": "sp2", "parent_span_id": None,
        "component_id": "totally_unknown_agent", "span_type": "agent", "operation": "haystack.component.run",
        "started_at": "2026-01-01T00:00:01.000Z", "latency_ms": 50,
    })
    _write_log(log_path, records)
    parsed = parse_spanlog(log_path)

    total, unmatched = referential_dry_run(parsed, known_component_ids={"project_identity"})

    assert total == 2
    assert unmatched == 1
