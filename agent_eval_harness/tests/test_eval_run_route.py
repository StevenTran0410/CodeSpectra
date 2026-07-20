"""Route-level tests for the PUT plan-report and POST eval-run endpoints.
Follows test_agentic_plan_routes.py's fake-external-calls pattern."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from httpx import ASGITransport, AsyncClient

from agent_eval_harness.datasets.types import DatasetCase
from agent_eval_harness.discovery.client import CodeSpectraClient
from agent_eval_harness.llm.client import LLMResponse
from agent_eval_harness.llm.proxy_client import CodeSpectraProxyClient
from agent_eval_harness.mapping.agent_flow import AgentFlow, AgentFlowMap, save_agent_flow_map
from agent_eval_harness.mapping.system_map import Component, SystemMap, save_system_map
from agent_eval_harness.store import repository
from agent_eval_harness.store.database import close_db, init_db
from agent_eval_harness.ui.server import app


@pytest.fixture(autouse=True)
async def _setup_db(tmp_path, monkeypatch):
    monkeypatch.setenv("AEH_DATA_DIR", str(tmp_path))
    await init_db()
    yield
    await close_db()


@pytest.fixture(autouse=True)
def _patch_external_calls(monkeypatch, tmp_path):
    async def fake_get_snapshot(self, snapshot_id: str) -> dict:
        return {"local_path": str(tmp_path)}

    async def fake_complete(self, messages, *, max_tokens=512, temperature=0.2, json_mode=False):
        return LLMResponse(content="{}", model="fake-llm")

    monkeypatch.setattr(CodeSpectraClient, "get_snapshot", fake_get_snapshot)
    monkeypatch.setattr(CodeSpectraProxyClient, "complete", fake_complete)


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


async def test_put_plan_report_route_persists_eval_enabled(tmp_path: Path) -> None:
    _write_widget_source(tmp_path)
    session_id = await _seed_completed_expansion_session(tmp_path)

    async with await _client() as client:
        await _generate_plan(client, session_id)
        get_resp = await client.get(f"/api/discovery/expansion-sessions/{session_id}/plan-report")
        report = get_resp.json()
        assert report["agents"][0]["eval_enabled"] is False

        report["agents"][0]["eval_enabled"] = True
        put_resp = await client.put(
            f"/api/discovery/expansion-sessions/{session_id}/plan-report", json=report
        )
        assert put_resp.status_code == 200, put_resp.text
        assert put_resp.json() == {"success": True}

        reget_resp = await client.get(f"/api/discovery/expansion-sessions/{session_id}/plan-report")
        assert reget_resp.json()["agents"][0]["eval_enabled"] is True

    sess = await repository.get_expansion_session(session_id)
    on_disk = yaml.safe_load(Path(sess["plan_report_path"]).read_text(encoding="utf-8"))
    assert on_disk["agents"][0]["eval_enabled"] is True


async def test_put_plan_report_route_422_on_invalid_body(tmp_path: Path) -> None:
    _write_widget_source(tmp_path)
    session_id = await _seed_completed_expansion_session(tmp_path)

    async with await _client() as client:
        await _generate_plan(client, session_id)
        resp = await client.put(
            f"/api/discovery/expansion-sessions/{session_id}/plan-report",
            json={"agents": "not-a-list"},
        )

    assert resp.status_code == 422


async def test_put_plan_report_route_400_before_generation(tmp_path: Path) -> None:
    session_id = await _seed_completed_expansion_session(tmp_path)

    async with await _client() as client:
        resp = await client.put(
            f"/api/discovery/expansion-sessions/{session_id}/plan-report",
            json={"target_system_id": "widget_system", "agents": []},
        )

    assert resp.status_code == 400


async def test_eval_run_route_no_agents_enabled(tmp_path: Path) -> None:
    _write_widget_source(tmp_path)
    session_id = await _seed_completed_expansion_session(tmp_path)

    async with await _client() as client:
        await _generate_plan(client, session_id)
        resp = await client.post(
            f"/api/discovery/expansion-sessions/{session_id}/eval-run",
            json={"provider_id": "prov-1", "backend_url": "http://fake-backend", "backend_token": "tok"},
        )

    assert resp.status_code == 200, resp.text
    assert resp.json()["agents"] == {}


async def test_eval_run_route_needs_human_when_no_fulfilled_dataset(tmp_path: Path) -> None:
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


async def test_eval_run_route_404_for_unknown_session() -> None:
    async with await _client() as client:
        resp = await client.post(
            "/api/discovery/expansion-sessions/does-not-exist/eval-run",
            json={"provider_id": "prov-1", "backend_url": "http://fake-backend", "backend_token": "tok"},
        )
    assert resp.status_code == 404


async def test_reset_stage3_deletes_plan_report_and_fulfilled_datasets(tmp_path: Path) -> None:
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


async def test_reset_stage3_graceful_when_no_plan_generated_yet(tmp_path: Path) -> None:
    session_id = await _seed_completed_expansion_session(tmp_path)

    async with await _client() as client:
        resp = await client.delete(f"/api/discovery/expansion-sessions/{session_id}/stage3")

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"success": True, "deleted_dataset_ids": []}


async def test_reset_stage3_404_for_unknown_session() -> None:
    async with await _client() as client:
        resp = await client.delete("/api/discovery/expansion-sessions/does-not-exist/stage3")
    assert resp.status_code == 404
