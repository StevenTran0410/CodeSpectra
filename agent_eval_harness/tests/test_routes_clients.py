"""Merged route/client/CLI/store/consolidation tests (see section markers for origin)."""
from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import yaml
from httpx import ASGITransport, AsyncClient

from agent_eval_harness import cli
from agent_eval_harness.datasets.types import DatasetCase
from agent_eval_harness.discovery.client import CodeSpectraClient
from agent_eval_harness.discovery.consolidation import consolidate_candidates
from agent_eval_harness.ingest.spanlog_ingest import (
    IngestError,
    parse_spanlog,
    persist_spanlog,
    referential_dry_run,
)
from agent_eval_harness.instrumentation.base import CapturedSpan
from agent_eval_harness.llm.client import LLMMessage, LLMResponse, RateLimitExceeded
from agent_eval_harness.llm.fake_client import FakeLLMClient
from agent_eval_harness.llm.proxy_client import CodeSpectraProxyClient
from agent_eval_harness.llm.routing_client import RoutingLLMClient, current_component_id_var
from agent_eval_harness.mapping.agent_flow import AgentFlow, AgentFlowMap, save_agent_flow_map
from agent_eval_harness.mapping.system_map import Component, SystemMap, save_system_map
from agent_eval_harness.store import repository
from agent_eval_harness.store.database import _run_migrations, close_db, get_db, init_db
from agent_eval_harness.ui.server import app
from tests._stubs import FakeCodeSpectraClient as _StubClient


@pytest.fixture
async def _setup_db(tmp_path, monkeypatch):
    """Shared by every group below that originally declared its own autouse copy of this fixture."""
    monkeypatch.setenv("AEH_DATA_DIR", str(tmp_path))
    await init_db()
    yield
    await close_db()


# --- test_eval_run_route.py ---


def _write_widget_source(tmp_path: Path) -> None:
    (tmp_path / "widget.py").write_text(
        "from haystack import component\n\n"
        "@component\n"
        "class WidgetComponent:\n"
        "    \"\"\"Does widget things.\"\"\"\n"
        "    def run(self):\n"
        "        return 'ok'\n"
    )


async def _seed_completed_expansion_session(tmp_path: Path, session_id: str = "sess-1") -> str:
    await repository.insert_expansion_session(session_id, "cand-1", "snap-1")

    system_map = SystemMap(
        target_system_id="widget_system",
        components=[
            Component(id="widget", role="writer", entry_point="widget:WidgetComponent", file="widget.py"),
        ],
    )
    map_path = tmp_path / f"{session_id}.yaml"
    save_system_map(system_map, map_path)

    await repository.finish_expansion_session(
        session_id, "completed", map_path=str(map_path), accepted=["widget.py"],
        boundary=[], stop_reason="frontier_exhausted", accepted_edges=[],
    )

    agent_flow_map = AgentFlowMap(
        target_system_id="widget_system",
        agents=[AgentFlow(id="widget_agent", role="writer", label="Widget Agent", component_ids=["widget"])],
        entry_agent_ids=["widget_agent"],
    )
    agent_flows_path = tmp_path / f"{session_id}_agentflows.yaml"
    save_agent_flow_map(agent_flow_map, agent_flows_path)
    await repository.update_expansion_session_agentflows_path(session_id, str(agent_flows_path))

    # /plan 400s unless Stage 2.5 enrichment has produced at least one row.
    await repository.upsert_agent_knowledge(
        session_id=session_id, agent_id="widget_agent",
        md_path="", json_path="", evidence_hash="seed", confidence="high", query_count=0,
    )

    return session_id


async def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def _generate_plan(client: AsyncClient, session_id: str):
    resp = await client.post(
        f"/api/discovery/expansion-sessions/{session_id}/plan",
        json={"provider_id": "prov-1", "backend_url": "http://fake-backend", "backend_token": "tok"},
    )
    assert resp.status_code == 200, resp.text
    return resp


@pytest.mark.usefixtures("_setup_db")
class TestEvalRunRoute:
    """PUT plan-report and POST eval-run endpoints; follows test_agentic_plan_routes.py's fake-external-calls pattern."""

    @pytest.fixture(autouse=True)
    def _patch_external_calls(self, monkeypatch, tmp_path):
        async def fake_get_snapshot(self, snapshot_id: str) -> dict:
            return {"local_path": str(tmp_path)}

        async def fake_complete(self, messages, *, max_tokens=512, temperature=0.2, json_mode=False):
            return LLMResponse(content="{}", model="fake-llm")

        monkeypatch.setattr(CodeSpectraClient, "get_snapshot", fake_get_snapshot)
        monkeypatch.setattr(CodeSpectraProxyClient, "complete", fake_complete)

    async def test_put_plan_report_route_persists_eval_enabled(self, tmp_path: Path) -> None:
        _write_widget_source(tmp_path)
        session_id = await _seed_completed_expansion_session(tmp_path)

        async with await _client() as client:
            await _generate_plan(client, session_id)
            get_resp = await client.get(f"/api/discovery/expansion-sessions/{session_id}/plan-report")
            report = get_resp.json()
            # Fix-plan #6: eval_enabled now defaults True (Stage3Screen toggle is the opt-OUT).
            assert report["agents"][0]["eval_enabled"] is True

            report["agents"][0]["eval_enabled"] = False
            put_resp = await client.put(
                f"/api/discovery/expansion-sessions/{session_id}/plan-report", json=report
            )
            assert put_resp.status_code == 200, put_resp.text
            assert put_resp.json() == {"success": True}

            reget_resp = await client.get(f"/api/discovery/expansion-sessions/{session_id}/plan-report")
            assert reget_resp.json()["agents"][0]["eval_enabled"] is False

        sess = await repository.get_expansion_session(session_id)
        on_disk = yaml.safe_load(Path(sess["plan_report_path"]).read_text(encoding="utf-8"))
        assert on_disk["agents"][0]["eval_enabled"] is False

    async def test_put_plan_report_route_422_on_invalid_body(self, tmp_path: Path) -> None:
        _write_widget_source(tmp_path)
        session_id = await _seed_completed_expansion_session(tmp_path)

        async with await _client() as client:
            await _generate_plan(client, session_id)
            resp = await client.put(
                f"/api/discovery/expansion-sessions/{session_id}/plan-report",
                json={"agents": "not-a-list"},
            )

        assert resp.status_code == 422

    async def test_put_plan_report_route_400_before_generation(self, tmp_path: Path) -> None:
        session_id = await _seed_completed_expansion_session(tmp_path)

        async with await _client() as client:
            resp = await client.put(
                f"/api/discovery/expansion-sessions/{session_id}/plan-report",
                json={"target_system_id": "widget_system", "agents": []},
            )

        assert resp.status_code == 400

    async def test_eval_run_route_no_agents_enabled(self, tmp_path: Path) -> None:
        _write_widget_source(tmp_path)
        session_id = await _seed_completed_expansion_session(tmp_path)

        async with await _client() as client:
            await _generate_plan(client, session_id)
            # Fix-plan #6: default is now enabled — explicitly opt every agent OUT to test the empty case.
            report = (await client.get(f"/api/discovery/expansion-sessions/{session_id}/plan-report")).json()
            for agent_report in report["agents"]:
                agent_report["eval_enabled"] = False
            await client.put(f"/api/discovery/expansion-sessions/{session_id}/plan-report", json=report)
            resp = await client.post(
                f"/api/discovery/expansion-sessions/{session_id}/eval-run",
                json={"provider_id": "prov-1", "backend_url": "http://fake-backend", "backend_token": "tok"},
            )

        assert resp.status_code == 200, resp.text
        assert resp.json()["agents"] == {}

    async def test_eval_run_route_needs_human_when_no_fulfilled_dataset(self, tmp_path: Path) -> None:
        _write_widget_source(tmp_path)
        session_id = await _seed_completed_expansion_session(tmp_path)

        async with await _client() as client:
            await _generate_plan(client, session_id)
            get_resp = await client.get(f"/api/discovery/expansion-sessions/{session_id}/plan-report")
            report = get_resp.json()
            report["agents"][0]["eval_enabled"] = True
            put_resp = await client.put(
                f"/api/discovery/expansion-sessions/{session_id}/plan-report", json=report
            )
            assert put_resp.status_code == 200

            resp = await client.post(
                f"/api/discovery/expansion-sessions/{session_id}/eval-run",
                json={"provider_id": "prov-1", "backend_url": "http://fake-backend", "backend_token": "tok"},
            )

        assert resp.status_code == 200, resp.text
        agents = resp.json()["agents"]
        assert agents["widget_agent"]["status"] == "needs_human"

    async def test_eval_run_route_404_for_unknown_session(self) -> None:
        async with await _client() as client:
            resp = await client.post(
                "/api/discovery/expansion-sessions/does-not-exist/eval-run",
                json={"provider_id": "prov-1", "backend_url": "http://fake-backend", "backend_token": "tok"},
            )
        assert resp.status_code == 404

    async def test_reset_stage3_deletes_plan_report_and_fulfilled_datasets(self, tmp_path: Path) -> None:
        _write_widget_source(tmp_path)
        session_id = await _seed_completed_expansion_session(tmp_path)

        async with await _client() as client:
            await _generate_plan(client, session_id)

        sess_before = await repository.get_expansion_session(session_id)
        plan_path = Path(sess_before["plan_path"])
        plan_report_path = Path(sess_before["plan_report_path"])
        assert plan_path.exists()
        assert plan_report_path.exists()

        # Simulate a fulfilled synthetic_agent_io dataset referenced by the plan.
        dataset_id = "ds-widget-agent-synth"
        await repository.insert_dataset_cases_bulk(
            dataset_id,
            [
                DatasetCase(
                    id="case-1", dataset=dataset_id, kind="synthetic_agent_io",
                    input={"shape": "retrieval_only", "bundle": {}}, expected=None, labels={},
                    provenance="synthetic",
                )
            ],
        )
        await repository.insert_dataset_metadata(dataset_id, "synthetic_agent_io", source_gate_ids=["x"], min_cases=1)
        suite_text = plan_path.read_text(encoding="utf-8")
        plan_data = yaml.safe_load(suite_text)
        plan_data["entries"][0]["dataset"] = {"ref": dataset_id, "required": None}
        plan_path.write_text(yaml.dump(plan_data, allow_unicode=True), encoding="utf-8")

        assert await repository.get_dataset_cases(dataset_id)  # sanity: seeded

        async with await _client() as client:
            resp = await client.delete(f"/api/discovery/expansion-sessions/{session_id}/stage3")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["success"] is True
        assert body["deleted_dataset_ids"] == [dataset_id]

        assert await repository.get_dataset_cases(dataset_id) == []
        assert not plan_path.exists()
        assert not plan_report_path.exists()

        sess_after = await repository.get_expansion_session(session_id)
        assert sess_after["plan_path"] is None
        assert sess_after["plan_report_path"] is None

    async def test_reset_stage3_graceful_when_no_plan_generated_yet(self, tmp_path: Path) -> None:
        session_id = await _seed_completed_expansion_session(tmp_path)

        async with await _client() as client:
            resp = await client.delete(f"/api/discovery/expansion-sessions/{session_id}/stage3")

        assert resp.status_code == 200, resp.text
        assert resp.json() == {"success": True, "deleted_dataset_ids": []}

    async def test_reset_stage3_404_for_unknown_session(self) -> None:
        async with await _client() as client:
            resp = await client.delete("/api/discovery/expansion-sessions/does-not-exist/stage3")
        assert resp.status_code == 404


# --- test_ingest_route.py ---


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


@pytest.mark.usefixtures("_setup_db")
class TestIngestRoute:
    """POST /api/eval-runs/ingest, using the ASGITransport pattern (no real server process needed)."""

    async def test_ingest_route_persists_a_run_and_returns_its_status(self, tmp_path: Path) -> None:
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

    async def test_ingest_route_404s_on_missing_manifest(self, tmp_path: Path) -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/eval-runs/ingest", json={"manifest_path": str(tmp_path / "does_not_exist.json")}
            )

        assert resp.status_code == 400

    async def test_ingest_route_422s_on_unsupported_schema(self, tmp_path: Path) -> None:
        log_path = tmp_path / "eval_log.jsonl"
        log_path.write_text(json.dumps({"record": "header", "schema": "aeh.spanlog/2"}) + "\n", encoding="utf-8")
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps({"plan_id": "p", "log_path": str(log_path)}), encoding="utf-8")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/eval-runs/ingest", json={"manifest_path": str(manifest_path)})

        assert resp.status_code == 422


# --- test_ingest_spanlog.py ---


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


@pytest.mark.usefixtures("_setup_db")
class TestIngestSpanlog:
    async def test_parse_and_persist_clean_run_marks_completed(self, tmp_path: Path) -> None:
        log_path = tmp_path / "eval_log.123.jsonl"
        _write_log(log_path, _clean_log_records())

        parsed = parse_spanlog(log_path)
        run_id = await persist_spanlog(parsed, target_system_id="codespectra", eval_plan_id="p1")

        run = await repository.get_run(run_id)
        assert run["status"] == "completed"
        assert run["source"] == "ingested"

    async def test_dataset_case_id_is_populated_on_the_trace(self, tmp_path: Path) -> None:
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

    async def test_span_input_output_are_persisted_and_redacted(self, tmp_path: Path) -> None:
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

    async def test_case_with_error_status_marks_run_partial_not_completed(self, tmp_path: Path) -> None:
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

    async def test_case_with_zero_spans_marks_run_partial(self) -> None:
        from agent_eval_harness.ingest.spanlog_ingest import ParsedSpanlog

        parsed = ParsedSpanlog(
            header={"schema": "aeh.spanlog/1"},
            cases={"t1": {"dataset_case_id": "c1", "root_input": "{}", "final_output": "{}", "status": "ok", "spans": []}},
        )

        run_id = await persist_spanlog(parsed, target_system_id="codespectra")

        run = await repository.get_run(run_id)
        assert run["status"] == "partial"

    def test_parse_rejects_unknown_schema_major(self, tmp_path: Path) -> None:
        log_path = tmp_path / "eval_log.127.jsonl"
        _write_log(log_path, [{"record": "header", "schema": "aeh.spanlog/2"}])

        with pytest.raises(IngestError, match="unsupported spanlog schema"):
            parse_spanlog(log_path)

    def test_parse_rejects_missing_header(self, tmp_path: Path) -> None:
        log_path = tmp_path / "eval_log.128.jsonl"
        _write_log(log_path, [{"record": "case_start", "trace_id": "t1"}])

        with pytest.raises(IngestError, match="no header record"):
            parse_spanlog(log_path)

    def test_referential_dry_run_counts_unmatched_component_ids(self, tmp_path: Path) -> None:
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


# --- test_ui_api.py ---


@pytest.mark.usefixtures("_setup_db")
class TestUiApi:
    """Unit tests for the FastAPI UI backend server endpoints."""

    async def test_ui_api_endpoints(self) -> None:
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

        await repository.finish_run(run_id, "completed")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            res = await client.get("/api/runs")
            assert res.status_code == 200
            runs_list = res.json()
            assert len(runs_list) == 1
            assert runs_list[0]["id"] == run_id
            assert runs_list[0]["pass_rate"] == 1.0
            assert runs_list[0]["judge_cost"] == 100

            res = await client.get(f"/api/runs/{run_id}")
            assert res.status_code == 200
            run_detail = res.json()
            assert run_detail["id"] == run_id
            assert "writer" in run_detail["component_aggregates"]
            assert run_detail["component_aggregates"]["writer"]["total"] == 1
            assert run_detail["component_aggregates"]["writer"]["passed"] == 1

            res = await client.get(f"/api/runs/{run_id}/components/writer")
            assert res.status_code == 200
            component_evals = res.json()
            assert len(component_evals) == 1
            assert component_evals[0]["metric_name"] == "ragas.faithfulness"
            assert component_evals[0]["root_input"] == "hello query"
            assert component_evals[0]["passed"] is True

            res = await client.get(f"/api/traces/{trace_id}")
            assert res.status_code == 200
            trace_detail = res.json()
            assert len(trace_detail["spans"]) == 1
            assert trace_detail["spans"][0]["id"] == "span-1"
            assert trace_detail["spans"][0]["component_id"] == "writer"

    async def test_ui_api_rerun_and_providers(self) -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            res = await client.get("/api/providers")
            assert res.status_code == 200
            assert isinstance(res.json(), list)

            res = await client.post("/api/runs/nonexistent-run/rerun", json={"model_overrides": {}})
            assert res.status_code == 404

            # Legacy run missing target/suite_path must 409, not crash.
            legacy_run_id = await repository.insert_run("test_target")
            res = await client.post(f"/api/runs/{legacy_run_id}/rerun", json={"model_overrides": {}})
            assert res.status_code == 409
            assert "predates rerun support" in res.json()["detail"]

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

            new_run = await repository.get_run(new_run_id)
            assert new_run is not None
            assert new_run["parent_run_id"] == valid_run_id
            import json
            overrides = json.loads(new_run["model_overrides"])
            assert overrides == {"writer": "fake-provider:gpt-4"}
            defects = json.loads(new_run["active_defects"])
            assert defects == ["no_retry"]


# --- test_proxy_client_reasoning.py ---


class _CapturingTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.last_body: dict | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.last_body = json.loads(request.content)
        return httpx.Response(
            200,
            json={"content": "ok", "model_id": "m", "prompt_tokens": 1, "completion_tokens": 1},
            request=request,
        )


async def test_reasoning_fields_attached_when_configured() -> None:
    transport = _CapturingTransport()
    http_client = httpx.AsyncClient(transport=transport)
    client = CodeSpectraProxyClient(
        "http://test", "tok", "provider-x", "model-y",
        http_client=http_client,
        reasoning_effort="high",
        thinking_budget=4096,
    )

    await client.complete([LLMMessage(role="user", content="hi")])

    assert transport.last_body is not None
    assert transport.last_body["reasoning_effort"] == "high"
    assert transport.last_body["thinking_budget"] == 4096
    await client.aclose()


async def test_reasoning_fields_absent_by_default() -> None:
    transport = _CapturingTransport()
    http_client = httpx.AsyncClient(transport=transport)
    client = CodeSpectraProxyClient(
        "http://test", "tok", "provider-x", "model-y", http_client=http_client,
    )

    await client.complete([LLMMessage(role="user", content="hi")])

    assert transport.last_body is not None
    assert transport.last_body["reasoning_effort"] is None
    assert transport.last_body["thinking_budget"] is None
    await client.aclose()


async def test_per_call_reasoning_effort_overrides_constructor_default() -> None:
    """A caller doing bounded structured JSON extraction can force a low effort for just its own call, regardless of the client's configured default."""
    transport = _CapturingTransport()
    http_client = httpx.AsyncClient(transport=transport)
    client = CodeSpectraProxyClient(
        "http://test", "tok", "provider-x", "model-y",
        http_client=http_client,
        reasoning_effort="high",
    )

    await client.complete([LLMMessage(role="user", content="hi")], reasoning_effort="low")

    assert transport.last_body is not None
    assert transport.last_body["reasoning_effort"] == "low"
    await client.aclose()


# --- test_proxy_client_retry.py ---


def _response(status_code: int, json_body: dict) -> httpx.Response:
    return httpx.Response(
        status_code, json=json_body, request=httpx.Request("POST", "http://test/complete")
    )


class _ScriptedTransport(httpx.AsyncBaseTransport):
    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = list(responses)
        self.call_count = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.call_count += 1
        return self._responses.pop(0)


async def test_retries_429_then_succeeds(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(
        "asyncio.sleep", lambda seconds: sleeps.append(seconds) or _noop_coro()
    )

    transport = _ScriptedTransport(
        [
            _response(429, {"detail": "rate limited"}),
            _response(429, {"detail": "rate limited"}),
            _response(
                200,
                {"content": "ok", "model_id": "m", "prompt_tokens": 1, "completion_tokens": 1},
            ),
        ]
    )
    http_client = httpx.AsyncClient(transport=transport)
    client = CodeSpectraProxyClient("http://test", "tok", "provider-x", http_client=http_client)

    result = await client.complete([LLMMessage(role="user", content="hi")])

    assert result.content == "ok"
    assert transport.call_count == 3
    assert len(sleeps) == 2
    await client.aclose()


async def test_exhausts_retries_and_raises(monkeypatch) -> None:
    monkeypatch.setattr("asyncio.sleep", lambda seconds: _noop_coro())

    # 1 initial attempt + 3 retries = 4 total 429 responses needed to exhaust retries
    transport = _ScriptedTransport([_response(429, {"detail": "rate limited"}) for _ in range(4)])
    http_client = httpx.AsyncClient(transport=transport)
    client = CodeSpectraProxyClient("http://test", "tok", "provider-x", http_client=http_client)

    with pytest.raises(RateLimitExceeded):
        await client.complete([LLMMessage(role="user", content="hi")])

    assert transport.call_count == 4
    await client.aclose()


async def test_non_429_error_fails_immediately_no_retry(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(
        "asyncio.sleep", lambda seconds: sleeps.append(seconds) or _noop_coro()
    )

    transport = _ScriptedTransport([_response(500, {"detail": "server error"})])
    http_client = httpx.AsyncClient(transport=transport)
    client = CodeSpectraProxyClient("http://test", "tok", "provider-x", http_client=http_client)

    with pytest.raises(httpx.HTTPStatusError):
        await client.complete([LLMMessage(role="user", content="hi")])

    assert transport.call_count == 1
    assert sleeps == []
    await client.aclose()


async def _noop_coro() -> None:
    return None


# --- test_routing_client.py ---


@pytest.mark.asyncio
async def test_routing_llm_client_contextvar() -> None:
    default_client = FakeLLMClient(LLMResponse(content="default", model="default-model"))
    override_client = FakeLLMClient(LLMResponse(content="override", model="override-model"))

    routing_client = RoutingLLMClient(default=default_client, overrides={"writer": override_client})

    res = await routing_client.complete([LLMMessage(role="user", content="hello")])
    assert res.content == "default"
    assert len(default_client.calls) == 1
    assert len(override_client.calls) == 0

    token = current_component_id_var.set("writer")
    try:
        res = await routing_client.complete([LLMMessage(role="user", content="hello")])
        assert res.content == "override"
        assert len(default_client.calls) == 1
        assert len(override_client.calls) == 1
    finally:
        current_component_id_var.reset(token)

    token = current_component_id_var.set("retriever")
    try:
        res = await routing_client.complete([LLMMessage(role="user", content="hello")])
        assert res.content == "default"
        assert len(default_client.calls) == 2
        assert len(override_client.calls) == 1
    finally:
        current_component_id_var.reset(token)


class _MockSpan:
    def __init__(self, tags):
        self.tags = tags


@contextmanager
def _patched_current_span(mock_span):
    """Temporarily replace haystack's global tracer.current_span() return value."""
    import haystack.tracing

    original = getattr(haystack.tracing.tracer, "current_span", None)
    haystack.tracing.tracer.current_span = MagicMock(return_value=mock_span)
    try:
        yield
    finally:
        if original is not None:
            haystack.tracing.tracer.current_span = original
        else:
            del haystack.tracing.tracer.current_span


@pytest.mark.asyncio
async def test_routing_llm_client_haystack_automatic_component_span() -> None:
    """Haystack's own automatic per-component span tag (haystack.component.name) must be picked up for routing, not just the manual span tag."""
    default_client = FakeLLMClient(LLMResponse(content="default", model="default-model"))
    override_client = FakeLLMClient(LLMResponse(content="override", model="override-model"))
    routing_client = RoutingLLMClient(default=default_client, overrides={"writer": override_client})

    with _patched_current_span(_MockSpan(tags={"haystack.component.name": "writer"})):
        res = await routing_client.complete([LLMMessage(role="user", content="hello")])
        assert res.content == "override"
        assert len(override_client.calls) == 1
        assert len(default_client.calls) == 0


@pytest.mark.asyncio
async def test_routing_llm_client_manual_span_tag() -> None:
    """Every real T1/T2 test-target component wraps its LLM call in a manual span carrying the component_name tag; routing must key off that tag too."""
    default_client = FakeLLMClient(LLMResponse(content="default", model="default-model"))
    override_client = FakeLLMClient(LLMResponse(content="override", model="override-model"))
    routing_client = RoutingLLMClient(default=default_client, overrides={"writer": override_client})

    with _patched_current_span(_MockSpan(tags={"component_name": "writer"})):
        res = await routing_client.complete([LLMMessage(role="user", content="hello")])
        assert res.content == "override"
        assert len(override_client.calls) == 1
        assert len(default_client.calls) == 0


@pytest.mark.asyncio
async def test_routing_llm_client_real_target_tier1_e2e(tmp_path, monkeypatch) -> None:
    """End-to-end regression guard: run T1's real Haystack pipeline (not a mocked component) to verify the writer's model override actually reaches its LLM call span."""
    import os

    from agent_eval_harness.metrics.sweep import run_sweep
    from agent_eval_harness.store import repository
    from agent_eval_harness.store.database import close_db, get_db, init_db

    monkeypatch.setenv("AEH_DATA_DIR", str(tmp_path))
    await init_db()
    try:
        override_client = FakeLLMClient(
            LLMResponse(content="override answer", model="override-fake-model-xyz")
        )
        routing_client = RoutingLLMClient(
            default=default_client_for_e2e(), overrides={"writer": override_client}
        )

        result = await run_sweep(
            target="test_targets.linear_rag.pipeline:build_pipeline",
            map_path="test_targets/linear_rag/system_map.yaml",
            suite_path="test_targets/linear_rag/suite.yaml",
            llm_client=routing_client,
            tier="1",
        )
        assert result.errors == []

        db = get_db()
        async with db.execute(
            "SELECT s.model FROM spans s JOIN traces t ON s.trace_id = t.id "
            "WHERE t.run_id = ? AND s.component_id = 'writer' AND s.span_type = 'llm_call'",
            (result.run_id,),
        ) as cur:
            rows = await cur.fetchall()

        assert rows, "expected at least one writer llm_call span"
        assert all(r["model"] == "override-fake-model-xyz" for r in rows), (
            "writer's LLM span did not record the overridden model — the override "
            "was silently ignored"
        )
    finally:
        await close_db()


def default_client_for_e2e() -> FakeLLMClient:
    return FakeLLMClient(LLMResponse(content="default answer", model="fake-default"))


# --- test_cli_run.py ---


_T1_MAP = str(Path(__file__).parent.parent / "test_targets" / "linear_rag" / "system_map.yaml")


class TestCliRun:
    """CLI smoke tests — invoke cli.main() directly (no subprocess), capture stdout."""

    @pytest.fixture(autouse=True)
    def _restore_shared_db_after_cli_closes_it(self):
        """cli.main() tears down the module-level DB connection via its own close_db()
        lifecycle, breaking later tests; reopen it once this test's process-lifecycle
        simulation is done."""
        yield
        asyncio.run(init_db())

    def test_cli_run_linear_rag(self, capsys) -> None:
        exit_code = cli.main(
            [
                "run",
                "--target",
                "test_targets.linear_rag.pipeline:build_pipeline",
                "--map",
                _T1_MAP,
                "--query",
                "What is the vacation policy?",
            ]
        )

        assert exit_code == 0
        out = capsys.readouterr().out
        assert "retriever" in out
        assert "writer" in out
        assert "mapped" in out
        assert "run completed" in out

    def test_cli_run_prints_unmatched_warning_for_broken_map(self, tmp_path, capsys) -> None:
        broken_map = tmp_path / "broken_map.yaml"
        broken_map.write_text(
            """
target_system_id: linear_rag
discrepancies: []
components:
  - id: retriever
    role: retrieval_agent
    entry_point: "test_targets.linear_rag.pipeline:run_retrieve"
    span_match: [{component_name: "nonexistent_1"}]
    constraints: []
    upstream: []
    downstream: [writer]
  - id: writer
    role: writer
    entry_point: "test_targets.linear_rag.pipeline:run_write"
    span_match: [{component_name: "nonexistent_2"}]
    constraints: []
    upstream: [retriever]
    downstream: []
""",
            encoding="utf-8",
        )

        exit_code = cli.main(
            [
                "run",
                "--target",
                "test_targets.linear_rag.pipeline:build_pipeline",
                "--map",
                str(broken_map),
                "--query",
                "What is the vacation policy?",
            ]
        )

        assert exit_code == 0
        out = capsys.readouterr().out
        assert "unmatched" in out.lower()
        assert "mapped 0/" in out


# --- test_store_roundtrip.py ---


@pytest.mark.usefixtures("_setup_db")
class TestStoreRoundtrip:
    """Result-store round-trip tests."""

    async def test_run_trace_span_roundtrip(self) -> None:
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

    async def test_unmatched_span_counted_via_null_component_id(self) -> None:
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

    async def test_migration_is_idempotent(self) -> None:
        db = get_db()
        await _run_migrations(db)  # calling again must not error or duplicate rows

        async with db.execute("PRAGMA index_list('spans')") as cur:
            indexes = await cur.fetchall()
        assert len(indexes) >= 1

    async def test_discovery_candidates_persistence(self) -> None:
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

    async def test_list_discovery_sessions_finds_session_by_snapshot_id_even_when_repo_ref_mismatches(self) -> None:
        """Verify session lookup works by snapshot_id even when repo_ref doesn't match."""
        session_id = await repository.insert_discovery_session(
            repo_ref="C:\\Users\\PC\\CodeSpectra\\repos\\CodeSpectra",
            snapshot_id="snap-mismatch-test",
        )

        # repo_ref filtering with the "other" value a caller might reasonably use
        by_wrong_repo_ref = await repository.list_discovery_sessions(repo_ref="snap-mismatch-test")
        assert session_id not in [s["id"] for s in by_wrong_repo_ref]

        by_snapshot_id = await repository.list_discovery_sessions(snapshot_id="snap-mismatch-test")
        assert session_id in [s["id"] for s in by_snapshot_id]

    async def test_discovery_session_pause_resume_roundtrip(self) -> None:
        session_id = await repository.insert_discovery_session("repo-test", "snap-test")
        session = await repository.get_discovery_session(session_id)
        assert session is not None
        assert session["status"] == "running"
        assert session["pause_info"] is None

        await repository.pause_discovery_session(
            session_id, "prov-123", "model-abc", reasoning_effort="high", thinking_budget=4096
        )
        paused = await repository.get_discovery_session(session_id)
        assert paused is not None
        assert paused["status"] == "paused_rate_limit"
        assert paused["pause_info"] == {
            "reason": "rate_limited",
            "provider_id": "prov-123",
            "model_id": "model-abc",
            "reasoning_effort": "high",
            "thinking_budget": 4096,
        }

        await repository.resume_discovery_session(session_id)
        resumed = await repository.get_discovery_session(session_id)
        assert resumed is not None
        assert resumed["status"] == "running"
        assert resumed["pause_info"] is None

    async def test_expansion_session_roundtrip_includes_accepted_edges(self) -> None:
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
        # accepted is now a list of dicts with file, role_hint, key_symbols, follow fields
        accepted_files = [item["file"] for item in sess["accepted"]]
        assert accepted_files == ["a.py", "b.py"]
        assert sess["boundary"] == ["c.py"]
        assert sess["accepted_edges"] == edges

        list_sessions = await repository.list_expansion_sessions_for_candidate("cand-123")
        assert len(list_sessions) == 1
        assert list_sessions[0]["accepted_edges"] == edges

    async def test_expansion_session_agent_flows_path_roundtrip(self) -> None:
        session_id = "sess-agentflows-123"
        await repository.insert_expansion_session(session_id, "cand-123", "snap-123")

        sess = await repository.get_expansion_session(session_id)
        assert sess is not None
        assert sess["agent_flows_path"] is None

        await repository.update_expansion_session_agentflows_path(
            session_id, "/tmp/map_agentflows.yaml"
        )

        sess = await repository.get_expansion_session(session_id)
        assert sess is not None
        assert sess["agent_flows_path"] == "/tmp/map_agentflows.yaml"

    async def test_cancel_orphaned_running_sessions_only_touches_running(self) -> None:
        running_discovery = await repository.insert_discovery_session("repo-x", "snap-x")
        paused_discovery = await repository.insert_discovery_session("repo-y", "snap-y")
        await repository.pause_discovery_session(paused_discovery, "prov", "model")

        running_expansion = "exp-running"
        completed_expansion = "exp-completed"
        await repository.insert_expansion_session(running_expansion, "cand-1", "snap-x")
        await repository.insert_expansion_session(completed_expansion, "cand-1", "snap-x")
        await repository.finish_expansion_session(completed_expansion, "completed")

        await repository.cancel_orphaned_running_sessions()

        running_disc = await repository.get_discovery_session(running_discovery)
        assert running_disc is not None
        assert running_disc["status"] == "failed"
        assert running_disc["error"] is not None
        assert running_disc["finished_at"] is not None

        paused_disc = await repository.get_discovery_session(paused_discovery)
        assert paused_disc is not None
        assert paused_disc["status"] == "paused_rate_limit"  # untouched — intentionally paused

        running_exp = await repository.get_expansion_session(running_expansion)
        assert running_exp is not None
        assert running_exp["status"] == "failed"
        assert running_exp["error"] is not None

        completed_exp = await repository.get_expansion_session(completed_expansion)
        assert completed_exp is not None
        assert completed_exp["status"] == "completed"

    async def test_evaluation_entry_id_agent_id_roundtrip(self) -> None:
        run_id = await repository.insert_run("test_target")
        eval_id = await repository.insert_evaluation(
            run_id=run_id,
            metric_name="ragas.faithfulness",
            metric_class="llm_judge",
            entry_id="writer.faithfulness.123",
            agent_id="writer_agent",
        )

        db = get_db()
        async with db.execute("SELECT * FROM evaluations WHERE id = ?", (eval_id,)) as cur:
            row = await cur.fetchone()

        assert row is not None
        assert row["entry_id"] == "writer.faithfulness.123"
        assert row["agent_id"] == "writer_agent"


# --- test_consolidation.py ---


@pytest.mark.anyio
async def test_consolidation_clear_import_match() -> None:
    # Candidate 1: has wiring block (>=2 nodes) -> core seed is {"a.py", "b.py"}
    # Candidate 2: has only 1 node in wiring -> falls back to hub_paths {"c.py"}
    candidates = [
        {
            "community_id": 1,
            "name": "Block 1",
            "frameworks": ["haystack"],
            "cluster_files": ["a.py", "b.py", "shared.py"],
            "hub_paths": ["a.py"],
            "wiring_block": {
                "nodes": [
                    {"source_hint_file": "a.py"},
                    {"source_hint_file": "b.py"}
                ],
                "edges": []
            }
        },
        {
            "community_id": 2,
            "name": "Block 2",
            "frameworks": ["langgraph"],
            "cluster_files": ["c.py", "other.py"],
            "hub_paths": ["c.py"],
            "wiring_block": {
                "nodes": [
                    {"source_hint_file": "c.py"}
                ],
                "edges": []
            }
        }
    ]

    # "shared.py" has a symbol edge to "b.py" (Candidate 1's core seed)
    edges_map = {
        "shared.py": {
            "outgoing": [{"src_symbol": "shared.py::helper", "dst_symbol": "b.py::run"}],
            "incoming": []
        }
    }
    client = _StubClient(edges_map=edges_map)
    llm_client = AsyncMock()

    res = await consolidate_candidates(candidates, client, "snap", llm_client)
    assert len(res) == 2

    # Candidate 1 should match "shared.py" via import
    c1 = next(c for c in res if c["community_id"] == 1)
    assert "shared.py" in c1["matched_files"]
    assert c1["file_provenance"]["shared.py"] == "import_matched"
    assert c1["seed_anchor_kind"] == "wiring"

    # Candidate 2: "shared.py" does not match since it's import_matched to 1
    c2 = next(c for c in res if c["community_id"] == 2)
    assert "shared.py" not in c2["matched_files"]
    assert c2["seed_anchor_kind"] == "hub"  # only 1 node -> fallback to hub


@pytest.mark.anyio
async def test_consolidation_tied_and_unresolved() -> None:
    # No path or symbol edge signal connecting shared.py to either seed -> stays unresolved
    candidates = [
        {
            "community_id": 1,
            "name": "Block 1",
            "cluster_files": ["a.py", "shared.py"],
            "hub_paths": ["a.py"],
            "wiring_block": None
        },
        {
            "community_id": 2,
            "name": "Block 2",
            "cluster_files": ["b.py", "shared.py"],
            "hub_paths": ["b.py"],
            "wiring_block": None
        }
    ]

    client = _StubClient()  # no edges
    llm_client = AsyncMock()

    res = await consolidate_candidates(candidates, client, "snap", llm_client)
    # shared.py has no signals -> no LLM call is made, remains unmatched
    llm_client.complete.assert_not_called()
    for c in res:
        assert "shared.py" not in c.get("matched_files", [])


@pytest.mark.anyio
async def test_consolidation_lone_candidate_does_not_claim_unrelated_file_by_repo_root_alone() -> None:
    """Regression: a lone candidate must not claim a file via shared top-level dir alone (e.g. "backend/") absent real symbol-edge signal."""
    candidates = [
        {
            "community_id": 1,
            "name": "Analysis Pipeline",
            "cluster_files": ["backend/domain/analysis/agents/agent_conventions.py", "backend/domain/qa/agent.py"],
            "hub_paths": ["backend/domain/analysis/agents/agent_conventions.py"],
            "wiring_block": {
                "nodes": [
                    {"source_hint_file": "backend/domain/analysis/agents/agent_conventions.py"},
                    {"source_hint_file": "backend/domain/analysis/agents/agent_risk.py"},
                ],
                "edges": [],
            },
        }
    ]

    client = _StubClient()  # no symbol edges at all
    llm_client = AsyncMock()

    res = await consolidate_candidates(candidates, client, "snap", llm_client)
    llm_client.complete.assert_not_called()
    assert "backend/domain/qa/agent.py" not in res[0]["matched_files"]


@pytest.mark.anyio
async def test_consolidation_llm_judge_resolves_tie() -> None:
    candidates = [
        {
            "community_id": 1,
            "name": "Block 1",
            "frameworks": ["haystack"],
            "cluster_files": ["dir/a.py", "dir/shared.py"],
            "hub_paths": ["dir/a.py"],
        },
        {
            "community_id": 2,
            "name": "Block 2",
            "frameworks": ["langgraph"],
            "cluster_files": ["dir/b.py", "dir/shared.py"],
            "hub_paths": ["dir/b.py"],
        }
    ]

    # dir/shared.py has a tie: path overlap is equal (best=2 for both since directories match)
    client = _StubClient(files={"dir/shared.py": "some content"})
    llm_client = AsyncMock()
    llm_client.complete.return_value = LLMResponse(
        content='{"candidate_id": "1", "confidence": "high"}',
        model="fake"
    )

    res = await consolidate_candidates(candidates, client, "snap", llm_client)
    # llm_client should be called to resolve the tie
    llm_client.complete.assert_called_once()
    c1 = next(c for c in res if c["community_id"] == 1)
    assert "dir/shared.py" in c1["matched_files"]
    assert c1["file_provenance"]["dir/shared.py"] == "llm_linked"


@pytest.mark.anyio
async def test_consolidation_propagates_rate_limit() -> None:
    candidates = [
        {
            "community_id": 1,
            "name": "Block 1",
            "cluster_files": ["dir/a.py", "dir/shared.py"],
            "hub_paths": ["dir/a.py"],
        },
        {
            "community_id": 2,
            "name": "Block 2",
            "cluster_files": ["dir/b.py", "dir/shared.py"],
            "hub_paths": ["dir/b.py"],
        }
    ]

    client = _StubClient()
    llm_client = AsyncMock()
    llm_client.complete.side_effect = RateLimitExceeded("prov", "model")

    with pytest.raises(RateLimitExceeded):
        await consolidate_candidates(candidates, client, "snap", llm_client)


@pytest.mark.anyio
async def test_consolidation_split_siblings_same_community_id_do_not_clobber() -> None:
    """Two per-system candidates split from one community share community_id but have distinct system_id; consolidation must key on system_id (via _cand_key) so their matched_files/provenance never collide."""
    candidates = [
        {
            "community_id": 10, "system_id": "10#haystack-aaaa", "name": "S — haystack",
            "frameworks": ["haystack"],
            "cluster_files": ["hay1.py", "hay2.py", "shared.py"], "hub_paths": ["hay1.py"],
            "wiring_block": {"nodes": [{"source_hint_file": "hay1.py"},
                                        {"source_hint_file": "hay2.py"}], "edges": []},
        },
        {
            "community_id": 10, "system_id": "10#langgraph-bbbb", "name": "S — langgraph",
            "frameworks": ["langgraph"],
            "cluster_files": ["lg1.py", "lg2.py", "shared.py"], "hub_paths": ["lg1.py"],
            "wiring_block": {"nodes": [{"source_hint_file": "lg1.py"},
                                        {"source_hint_file": "lg2.py"}], "edges": []},
        },
    ]
    # shared.py imports into the langgraph seed (lg2.py) -> must land ONLY on the langgraph sibling.
    edges_map = {
        "shared.py": {
            "outgoing": [{"src_symbol": "shared.py::h", "dst_symbol": "lg2.py::run"}],
            "incoming": [],
        }
    }
    client = _StubClient(edges_map=edges_map)
    res = await consolidate_candidates(candidates, client, "snap", AsyncMock())

    hay = next(c for c in res if c["system_id"] == "10#haystack-aaaa")
    lg = next(c for c in res if c["system_id"] == "10#langgraph-bbbb")
    # Each keeps its OWN core seed intact — no clobber.
    assert set(hay["matched_files"]) >= {"hay1.py", "hay2.py"}
    assert set(lg["matched_files"]) >= {"lg1.py", "lg2.py"}
    # shared.py resolves to the langgraph sibling only.
    assert "shared.py" in lg["matched_files"]
    assert "shared.py" not in hay["matched_files"]
