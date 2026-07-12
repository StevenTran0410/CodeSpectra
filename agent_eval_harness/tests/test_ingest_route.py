"""Route-level test for POST /api/eval-runs/ingest, following test_eval_run_route.py's
ASGITransport pattern (no real server process needed)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from agent_eval_harness.store import repository
from agent_eval_harness.store.database import close_db, init_db
from agent_eval_harness.ui.server import app

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
async def _setup_db(tmp_path, monkeypatch):
    monkeypatch.setenv("AEH_DATA_DIR", str(tmp_path))
    await init_db()
    yield
    await close_db()


def _write_clean_spanlog(tmp_path: Path) -> tuple[Path, Path]:
    log_path = tmp_path / "eval_log.999.jsonl"
    records = [
        {"record": "header", "schema": "aeh.spanlog/1", "tracer_version": "1", "plan_id": "plan-xyz", "run_id": "r1"},
        {"record": "case_start", "trace_id": "t1", "dataset_case_id": "case-1", "input": {"snapshot_id": "s1"}},
        {
            "record": "span", "trace_id": "t1", "span_id": "sp1", "parent_span_id": None,
            "component_id": "project_identity", "span_type": "agent", "operation": "haystack.component.run",
            "started_at": "2026-01-01T00:00:00.000Z", "latency_ms": 100,
            "input_json": "{}", "output_json": '{"domain": "web"}',
        },
        {"record": "case_end", "trace_id": "t1", "status": "ok", "final_output_json": '{"domain": "web"}'},
        {"record": "run_summary", "attempted": 1, "succeeded": 1},
    ]
    log_path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"plan_id": "plan-xyz", "log_path": str(log_path)}), encoding="utf-8")
    return manifest_path, log_path


async def test_ingest_route_persists_a_run_and_returns_its_status(tmp_path: Path) -> None:
    manifest_path, _ = _write_clean_spanlog(tmp_path)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/eval-runs/ingest", json={"manifest_path": str(manifest_path)})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"

    run = await repository.get_run(body["run_id"])
    assert run["source"] == "ingested"
    assert run["eval_plan_id"] == "plan-xyz"


async def test_ingest_route_404s_on_missing_manifest(tmp_path: Path) -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/eval-runs/ingest", json={"manifest_path": str(tmp_path / "does_not_exist.json")}
        )

    assert resp.status_code == 400


async def test_ingest_route_422s_on_unsupported_schema(tmp_path: Path) -> None:
    log_path = tmp_path / "eval_log.jsonl"
    log_path.write_text(json.dumps({"record": "header", "schema": "aeh.spanlog/2"}) + "\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"plan_id": "p", "log_path": str(log_path)}), encoding="utf-8")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/eval-runs/ingest", json={"manifest_path": str(manifest_path)})

    assert resp.status_code == 422
