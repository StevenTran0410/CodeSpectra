"""Unit tests for the FastAPI UI backend server endpoints (CS-266)."""
from __future__ import annotations

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


async def test_ui_api_endpoints() -> None:
    # 1. Populate dummy data
    run_id = await repository.insert_run(
        "test_target", map_path="test_targets/t3_reranker/system_map.yaml"
    )

    trace_id = await repository.insert_trace(
        run_id, root_input="hello query", dataset_case_id="case-1"
    )

    from agent_eval_harness.instrumentation.base import CapturedSpan

    dummy_span = CapturedSpan(
        span_id="span-1",
        parent_span_id=None,
        operation_name="write_op",
        span_type="llm_call",
        component_id="writer",
        input_json='{"prompt": "hello"}',
        output_json='{"response": "world"}',
        model="fake-mini",
        tokens_in=10,
        tokens_out=20,
        latency_ms=100,
        started_at="2026-07-05T12:00:00Z",
        tags={"tier": "1"},
    )
    await repository.insert_spans_bulk(trace_id, [dummy_span])

    await repository.insert_evaluation(
        run_id,
        "ragas.faithfulness",
        "llm_judge",
        span_id="span-1",
        trace_id=trace_id,
        component_id="writer",
        score=0.9,
        passed=True,
        details={"reason": "good answer"},
        evaluator="ragas",
        cost_tokens=100,
    )

    # Complete run
    await repository.finish_run(run_id, "completed")

    # 2. Test API client
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # GET /api/runs
        res = await client.get("/api/runs")
        assert res.status_code == 200
        runs_list = res.json()
        assert len(runs_list) == 1
        assert runs_list[0]["id"] == run_id
        assert runs_list[0]["pass_rate"] == 1.0
        assert runs_list[0]["judge_cost"] == 100

        # GET /api/runs/{run_id}
        res = await client.get(f"/api/runs/{run_id}")
        assert res.status_code == 200
        run_detail = res.json()
        assert run_detail["id"] == run_id
        assert "writer" in run_detail["component_aggregates"]
        assert run_detail["component_aggregates"]["writer"]["total"] == 1
        assert run_detail["component_aggregates"]["writer"]["passed"] == 1

        # GET /api/runs/{run_id}/components/{component_id}
        res = await client.get(f"/api/runs/{run_id}/components/writer")
        assert res.status_code == 200
        component_evals = res.json()
        assert len(component_evals) == 1
        assert component_evals[0]["metric_name"] == "ragas.faithfulness"
        assert component_evals[0]["root_input"] == "hello query"
        assert component_evals[0]["passed"] is True

        # GET /api/traces/{trace_id}
        res = await client.get(f"/api/traces/{trace_id}")
        assert res.status_code == 200
        trace_detail = res.json()
        assert len(trace_detail["spans"]) == 1
        assert trace_detail["spans"][0]["id"] == "span-1"
        assert trace_detail["spans"][0]["component_id"] == "writer"


async def test_ui_api_rerun_and_providers() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Test 1: GET /api/providers
        res = await client.get("/api/providers")
        assert res.status_code == 200
        assert isinstance(res.json(), list)

        # Test 2: Rerun 404
        res = await client.post("/api/runs/nonexistent-run/rerun", json={"model_overrides": {}})
        assert res.status_code == 404

        # Test 3: Rerun 409 for legacy run (missing target/suite_path)
        legacy_run_id = await repository.insert_run("test_target")
        res = await client.post(f"/api/runs/{legacy_run_id}/rerun", json={"model_overrides": {}})
        assert res.status_code == 409
        assert "predates rerun support" in res.json()["detail"]

        # Test 4: Successful rerun trigger
        valid_run_id = await repository.insert_run(
            target_system_id="test_target",
            map_path="test_targets/t3_reranker/system_map.yaml",
            target="test_targets.t3_reranker.pipeline:build_pipeline",
            suite_path="configs/qa_testset.yaml"
        )
        res = await client.post(
            f"/api/runs/{valid_run_id}/rerun",
            json={
                "model_overrides": {"writer": "fake-provider:gpt-4"},
                "active_defects": ["no_retry"]
            }
        )
        assert res.status_code == 200
        new_run_id = res.json()["run_id"]
        assert new_run_id != valid_run_id

        # Verify the database contains the correct rerun fields
        new_run = await repository.get_run(new_run_id)
        assert new_run is not None
        assert new_run["parent_run_id"] == valid_run_id
        import json
        overrides = json.loads(new_run["model_overrides"])
        assert overrides == {"writer": "fake-provider:gpt-4"}
        defects = json.loads(new_run["active_defects"])
        assert defects == ["no_retry"]

