"""Result-store round-trip tests."""
from __future__ import annotations

import pytest

from agent_eval_harness.instrumentation.base import CapturedSpan
from agent_eval_harness.store import repository
from agent_eval_harness.store.database import _run_migrations, close_db, get_db, init_db

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
async def _setup_db(tmp_path, monkeypatch):
    monkeypatch.setenv("AEH_DATA_DIR", str(tmp_path))
    await init_db()
    yield
    await close_db()


async def test_run_trace_span_roundtrip() -> None:
    run_id = await repository.insert_run("test_target")
    trace_id = await repository.insert_trace(run_id, root_input="hello")

    spans = [
        CapturedSpan(
            span_id="rt-s1",
            parent_span_id=None,
            operation_name="op1",
            span_type="agent",
            component_name="a",
            component_id="a",
            started_at="2026-01-01T00:00:00Z",
            latency_ms=10,
        ),
        CapturedSpan(
            span_id="rt-s2",
            parent_span_id="rt-s1",
            operation_name="op2",
            span_type="llm_call",
            component_name="b",
            component_id="b",
            model="fake-model",
            tokens_in=5,
            tokens_out=3,
            token_source="measured",
            started_at="2026-01-01T00:00:01Z",
            latency_ms=20,
        ),
    ]
    await repository.insert_spans_bulk(trace_id, spans)
    await repository.finalize_trace(trace_id, "final answer", total_tokens=8, total_latency_ms=30)
    await repository.finish_run(run_id, "completed")

    run = await repository.get_run(run_id)
    assert run is not None
    assert run["status"] == "completed"

    traces = await repository.get_traces_for_run(run_id)
    assert len(traces) == 1
    assert traces[0]["final_output"] == "final answer"
    assert traces[0]["total_tokens"] == 8

    stored_spans = await repository.get_spans_for_trace(trace_id)
    assert len(stored_spans) == 2
    by_id = {s["id"]: s for s in stored_spans}
    assert by_id["rt-s2"]["parent_span_id"] == "rt-s1"
    assert by_id["rt-s2"]["model"] == "fake-model"

    unmatched_count = await repository.count_unmatched_spans(trace_id)
    assert unmatched_count == 0


async def test_unmatched_span_counted_via_null_component_id() -> None:
    run_id = await repository.insert_run("test_target")
    trace_id = await repository.insert_trace(run_id, root_input="q")
    span = CapturedSpan(
        span_id="rt-s3",
        parent_span_id=None,
        operation_name="op",
        span_type="agent",
        component_id=None,
        started_at="2026-01-01T00:00:00Z",
        latency_ms=1,
    )
    await repository.insert_spans_bulk(trace_id, [span])

    assert await repository.count_unmatched_spans(trace_id) == 1


async def test_migration_is_idempotent() -> None:
    db = get_db()
    await _run_migrations(db)  # calling again must not error or duplicate rows

    async with db.execute("PRAGMA index_list('spans')") as cur:
        indexes = await cur.fetchall()
    assert len(indexes) >= 1
