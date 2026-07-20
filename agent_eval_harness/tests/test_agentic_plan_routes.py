"""Route-level tests for the Stage 3 agentic planner endpoints; fakes the snapshot lookup and LLM calls, everything else runs for real."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from httpx import ASGITransport, AsyncClient

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
    """Every DAG node's LLM call is answered with "{}", so every node degrades gracefully — enough to exercise routing/persistence/wiring without per-node canned JSON."""

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


async def _seed_completed_expansion_session(
    tmp_path: Path, session_id: str = "sess-1", *, with_agent_flows: bool = True
) -> str:
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
        session_id,
        "completed",
        map_path=str(map_path),
        accepted=["widget.py"],
        boundary=[],
        stop_reason="frontier_exhausted",
        accepted_edges=[],
    )

    if with_agent_flows:
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


async def test_generate_plan_blocked_without_agent_flow_map(tmp_path: Path) -> None:
    """Stage 3's hard prerequisite (plan §2): no AgentFlowMap -> 400, actionable message."""
    session_id = await _seed_completed_expansion_session(tmp_path, with_agent_flows=False)

    async with await _client() as client:
        resp = await client.post(
            f"/api/discovery/expansion-sessions/{session_id}/plan",
            json={"provider_id": "prov-1", "backend_url": "http://fake-backend", "backend_token": "tok"},
        )

    assert resp.status_code == 400
    assert "Stage 2" in resp.json()["detail"]


async def test_generate_plan_success_persists_suite_and_report(tmp_path: Path) -> None:
    _write_widget_source(tmp_path)
    session_id = await _seed_completed_expansion_session(tmp_path)

    async with await _client() as client:
        resp = await client.post(
            f"/api/discovery/expansion-sessions/{session_id}/plan",
            json={"provider_id": "prov-1", "backend_url": "http://fake-backend", "backend_token": "tok"},
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["entries"], "expected the writer-role baseline gates (ragas.faithfulness/answer_relevancy)"
    metrics = {e["metric"] for e in body["entries"]}
    assert "ragas.faithfulness" in metrics
    assert all(e["agent_id"] == "widget_agent" for e in body["entries"])

    sess = await repository.get_expansion_session(session_id)
    assert sess is not None
    assert sess["plan_path"] is not None and Path(sess["plan_path"]).exists()
    assert sess["plan_report_path"] is not None
    report_path = Path(sess["plan_report_path"])
    assert report_path.exists()
    on_disk = yaml.safe_load(report_path.read_text(encoding="utf-8"))
    assert on_disk["target_system_id"] == "widget_system"
    assert [a["agent_id"] for a in on_disk["agents"]] == ["widget_agent"]


async def test_get_plan_report_404_before_generation(tmp_path: Path) -> None:
    session_id = await _seed_completed_expansion_session(tmp_path)

    async with await _client() as client:
        resp = await client.get(f"/api/discovery/expansion-sessions/{session_id}/plan-report")

    assert resp.status_code == 404


async def test_get_plan_report_returns_saved_report_after_generation(tmp_path: Path) -> None:
    _write_widget_source(tmp_path)
    session_id = await _seed_completed_expansion_session(tmp_path)

    async with await _client() as client:
        post_resp = await client.post(
            f"/api/discovery/expansion-sessions/{session_id}/plan",
            json={"provider_id": "prov-1", "backend_url": "http://fake-backend", "backend_token": "tok"},
        )
        assert post_resp.status_code == 200

        get_resp = await client.get(f"/api/discovery/expansion-sessions/{session_id}/plan-report")

    assert get_resp.status_code == 200
    assert get_resp.json()["target_system_id"] == "widget_system"


async def test_generate_plan_404_for_unknown_session() -> None:
    async with await _client() as client:
        resp = await client.post(
            "/api/discovery/expansion-sessions/does-not-exist/plan",
            json={"provider_id": "prov-1", "backend_url": "http://fake-backend", "backend_token": "tok"},
        )
    assert resp.status_code == 404


async def test_generate_plan_missing_local_path_returns_400_not_500(tmp_path: Path, monkeypatch) -> None:
    """Regression guard: an inner HTTPException must not get re-wrapped into a 500."""
    session_id = await _seed_completed_expansion_session(tmp_path)

    async def fake_get_snapshot_missing_local_path(self, snapshot_id: str) -> dict:
        return {}  # no "local_path" key

    monkeypatch.setattr(CodeSpectraClient, "get_snapshot", fake_get_snapshot_missing_local_path)

    async with await _client() as client:
        resp = await client.post(
            f"/api/discovery/expansion-sessions/{session_id}/plan",
            json={"provider_id": "prov-1", "backend_url": "http://fake-backend", "backend_token": "tok"},
        )
    assert resp.status_code == 400
    assert "local_path" in resp.json()["detail"]


async def test_generate_plan_field_downstream_consumers_resolves_boundary_only_helper(tmp_path: Path) -> None:
    """Regression: contract harvest must still see a field read delegated to a helper file that
    landed in `boundary` (not `accepted`), or field_downstream_consumers comes back empty and
    the synthetic_agent_io gate never gets emitted for that agent."""
    (tmp_path / "myapp").mkdir()
    (tmp_path / "myapp" / "fan_in_agent.py").write_text(
        "from myapp.helper import build_input\n\n"
        "class FanInAgent:\n"
        "    async def run(self, provider_id: str, model_id: str, all_sections: dict) -> dict:\n"
        "        return build_input(all_sections)\n",
        encoding="utf-8",
    )
    (tmp_path / "myapp" / "helper.py").write_text(
        "def build_input(sections):\n"
        "    out = {}\n"
        "    for letter in 'AB':\n"
        "        s = sections.get(letter) or {}\n"
        "        out[letter] = {'confidence': s.get('confidence'), 'purpose': s.get('purpose')}\n"
        "    return out\n",
        encoding="utf-8",
    )

    session_id = "sess-boundary"
    await repository.insert_expansion_session(session_id, "cand-1", "snap-1")
    system_map = SystemMap(
        target_system_id="t",
        components=[
            Component(
                id="fanin", role="writer",
                entry_point="myapp.fan_in_agent:FanInAgent", file="myapp/fan_in_agent.py",
            )
        ],
    )
    map_path = tmp_path / f"{session_id}.yaml"
    save_system_map(system_map, map_path)
    await repository.finish_expansion_session(
        session_id, "completed", map_path=str(map_path),
        accepted=["myapp/fan_in_agent.py"],  # helper.py deliberately NOT accepted
        boundary=["myapp/helper.py"],
        stop_reason="frontier_exhausted", accepted_edges=[],
    )
    agent_flow_map = AgentFlowMap(
        target_system_id="t",
        agents=[AgentFlow(id="fanin", role="writer", label="FanIn", component_ids=["fanin"])],
        entry_agent_ids=["fanin"],
    )
    agent_flows_path = tmp_path / f"{session_id}_agentflows.yaml"
    save_agent_flow_map(agent_flow_map, agent_flows_path)
    await repository.update_expansion_session_agentflows_path(session_id, str(agent_flows_path))
    await repository.upsert_agent_knowledge(
        session_id=session_id, agent_id="fanin",
        md_path="", json_path="", evidence_hash="seed", confidence="high", query_count=0,
    )

    async with await _client() as client:
        resp = await client.post(
            f"/api/discovery/expansion-sessions/{session_id}/plan",
            json={"provider_id": "prov-1", "backend_url": "http://fake-backend", "backend_token": "tok"},
        )
    assert resp.status_code == 200, resp.text

    sess = await repository.get_expansion_session(session_id)
    report = yaml.safe_load(Path(sess["plan_report_path"]).read_text(encoding="utf-8"))
    contract = next(a["contract"] for a in report["agents"] if a["agent_id"] == "fanin")
    assert contract["field_downstream_consumers"] == {
        "A": ["confidence", "purpose"],
        "B": ["confidence", "purpose"],
    }

    plan = yaml.safe_load(Path(sess["plan_path"]).read_text(encoding="utf-8"))
    synth_entries = [
        e for e in plan["entries"]
        if ((e.get("dataset") or {}).get("required") or {}).get("kind") == "synthetic_agent_io"
    ]
    assert len(synth_entries) == 1
    assert synth_entries[0]["agent_id"] == "fanin"
