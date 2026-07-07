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


async def test_discovery_candidates_persistence() -> None:
    session_id = await repository.insert_discovery_session("repo-test", "snap-test")
    candidates = [
        {
            "name": "Test Candidate",
            "frameworks": ["haystack"],
            "entry_points": ["app.py"],
            "evidence": [{"file": "app.py", "snippet": "import haystack", "token_estimate": 4}],
            "confidence": "high",
            "verdict": "proposed",
            "needs_human": False,
            "community_id": "42",
            "cluster_files": ["app.py", "utils.py"],
            "hub_paths": ["app.py"],
            "wiring_block": {
                "nodes": [{"alias": "n1", "class_name": "C1", "source_hint_file": "app.py"}],
                "edges": [{"src": "n1", "dst": "n1"}],
                "framework": "haystack",
                "source": "static",
            },
            "excluded_files": ["bad_file.py"],
        }
    ]
    await repository.insert_discovery_candidates_bulk(session_id, candidates)
    res = await repository.get_discovery_candidates(session_id)
    assert len(res) == 1
    c = res[0]
    assert c["name"] == "Test Candidate"
    assert c["frameworks"] == ["haystack"]
    assert c["entry_points"] == ["app.py"]
    assert c["evidence"] == [{"file": "app.py", "snippet": "import haystack", "token_estimate": 4}]
    assert c["confidence"] == "high"
    assert c["verdict"] == "proposed"
    assert c["excluded_files"] == ["bad_file.py"]

    # Test update
    await repository.update_candidate_excluded_files(c["id"], ["bad_file.py", "another.py"])
    updated = await repository.get_discovery_candidate(c["id"])
    assert updated is not None
    assert updated["excluded_files"] == ["bad_file.py", "another.py"]
    assert c["needs_human"] is False
    assert c["community_id"] == "42"
    assert c["cluster_files"] == ["app.py", "utils.py"]
    assert c["hub_paths"] == ["app.py"]
    assert c["wiring_block"] == {
        "nodes": [{"alias": "n1", "class_name": "C1", "source_hint_file": "app.py"}],
        "edges": [{"src": "n1", "dst": "n1"}],
        "framework": "haystack",
        "source": "static",
    }


async def test_list_discovery_sessions_finds_session_by_snapshot_id_even_when_repo_ref_mismatches() -> None:
    """Verify session lookup works by snapshot_id even when repo_ref doesn't match."""
    session_id = await repository.insert_discovery_session(
        repo_ref="C:\\Users\\PC\\CodeSpectra\\repos\\CodeSpectra",
        snapshot_id="snap-mismatch-test",
    )

    # repo_ref filtering with the "other" value a caller might reasonably use
    by_wrong_repo_ref = await repository.list_discovery_sessions(repo_ref="snap-mismatch-test")
    assert session_id not in [s["id"] for s in by_wrong_repo_ref]

    # snapshot_id filtering must find it regardless.
    by_snapshot_id = await repository.list_discovery_sessions(snapshot_id="snap-mismatch-test")
    assert session_id in [s["id"] for s in by_snapshot_id]


async def test_discovery_session_pause_resume_roundtrip() -> None:
    session_id = await repository.insert_discovery_session("repo-test", "snap-test")
    session = await repository.get_discovery_session(session_id)
    assert session is not None
    assert session["status"] == "running"
    assert session["pause_info"] is None

    # Pause it
    await repository.pause_discovery_session(session_id, "prov-123", "model-abc")
    paused = await repository.get_discovery_session(session_id)
    assert paused is not None
    assert paused["status"] == "paused_rate_limit"
    assert paused["pause_info"] == {
        "reason": "rate_limited",
        "provider_id": "prov-123",
        "model_id": "model-abc",
    }

    # Resume it
    await repository.resume_discovery_session(session_id)
    resumed = await repository.get_discovery_session(session_id)
    assert resumed is not None
    assert resumed["status"] == "running"
    assert resumed["pause_info"] is None


async def test_expansion_session_roundtrip_includes_accepted_edges() -> None:
    session_id = "sess-roundtrip-123"
    await repository.insert_expansion_session(session_id, "cand-123", "snap-123")
    
    edges = [{"src": "a.py", "dst": "b.py"}]
    await repository.finish_expansion_session(
        session_id,
        "completed",
        map_path="/tmp/map.yaml",
        accepted=["a.py", "b.py"],
        boundary=["c.py"],
        stop_reason="frontier_exhausted",
        accepted_edges=edges
    )
    
    sess = await repository.get_expansion_session(session_id)
    assert sess is not None
    assert sess["status"] == "completed"
    assert sess["accepted"] == ["a.py", "b.py"]
    assert sess["boundary"] == ["c.py"]
    assert sess["accepted_edges"] == edges

    list_sessions = await repository.list_expansion_sessions_for_candidate("cand-123")
    assert len(list_sessions) == 1
    assert list_sessions[0]["accepted_edges"] == edges

