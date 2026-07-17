"""Route-level tests for the agent-flow endpoints (POST/GET .../agent-flows)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from httpx import ASGITransport, AsyncClient

from agent_eval_harness.discovery.client import CodeSpectraClient
from agent_eval_harness.llm.client import LLMResponse
from agent_eval_harness.llm.proxy_client import CodeSpectraProxyClient
from agent_eval_harness.mapping.system_map import Component, SystemMap, save_system_map
from agent_eval_harness.store import repository
from agent_eval_harness.store.database import close_db, init_db
from agent_eval_harness.ui.server import app

pytestmark = pytest.mark.asyncio

CANNED_GROUPING = json.dumps({
    "agents": [
        {
            "id": "widget",
            "label": "Widget Agent",
            "role": "orchestrator",
            "summary": "Does widget things.",
            "component_ids": ["widget"],
            "upstream_agents": [],
            "downstream_agents": [],
            "parent_agent": None,
        }
    ],
    "entry_agent_ids": ["widget"],
    "unassigned_component_ids": [],
})


@pytest.fixture(autouse=True)
async def _setup_db(tmp_path, monkeypatch):
    monkeypatch.setenv("AEH_DATA_DIR", str(tmp_path))
    await init_db()
    yield
    await close_db()


@pytest.fixture(autouse=True)
def _patch_external_calls(monkeypatch, tmp_path):
    """Both external boundaries (snapshot lookup + LLM-2 call) are faked; everything else
    (route guards, scanning, storage) runs for real."""

    async def fake_get_snapshot(self, snapshot_id: str) -> dict:
        return {"local_path": str(tmp_path)}

    async def fake_complete(self, messages, *, max_tokens=512, temperature=0.2, json_mode=False, **kwargs):
        return LLMResponse(content=CANNED_GROUPING, model="fake-llm-2")

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
            Component(
                id="widget",
                role="unknown",
                entry_point="widget:WidgetComponent",
                file="widget.py",
            )
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
    )
    return session_id


async def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_generate_agent_flows_success_persists_and_returns_map(tmp_path: Path) -> None:
    _write_widget_source(tmp_path)
    session_id = await _seed_completed_expansion_session(tmp_path)

    async with await _client() as client:
        resp = await client.post(
            f"/api/discovery/expansion-sessions/{session_id}/agent-flows",
            json={
                "provider_id": "prov-1",
                "model_id": "strong-model",
                "backend_url": "http://fake-backend",
                "backend_token": "tok",
            },
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["target_system_id"] == "widget_system"
    assert [a["id"] for a in body["agents"]] == ["widget"]
    assert body["agents"][0]["component_ids"] == ["widget"]
    assert body["unassigned_component_ids"] == []

    # Persisted: sibling YAML written + DB pointer updated.
    sess = await repository.get_expansion_session(session_id)
    assert sess is not None
    assert sess["agent_flows_path"] is not None
    agent_flows_path = Path(sess["agent_flows_path"])
    assert agent_flows_path.exists()
    assert agent_flows_path.name == f"{session_id}_agentflows.yaml"
    on_disk = yaml.safe_load(agent_flows_path.read_text(encoding="utf-8"))
    assert on_disk["target_system_id"] == "widget_system"


async def test_get_agent_flows_returns_404_before_generation(tmp_path: Path) -> None:
    session_id = await _seed_completed_expansion_session(tmp_path)

    async with await _client() as client:
        resp = await client.get(f"/api/discovery/expansion-sessions/{session_id}/agent-flows")

    assert resp.status_code == 404


async def test_get_agent_flows_returns_saved_map_after_generation(tmp_path: Path) -> None:
    _write_widget_source(tmp_path)
    session_id = await _seed_completed_expansion_session(tmp_path)

    async with await _client() as client:
        post_resp = await client.post(
            f"/api/discovery/expansion-sessions/{session_id}/agent-flows",
            json={"provider_id": "prov-1", "backend_url": "http://fake-backend", "backend_token": "tok"},
        )
        assert post_resp.status_code == 200

        get_resp = await client.get(f"/api/discovery/expansion-sessions/{session_id}/agent-flows")

    assert get_resp.status_code == 200
    assert get_resp.json() == post_resp.json()


async def test_generate_agent_flows_404_for_unknown_session() -> None:
    async with await _client() as client:
        resp = await client.post(
            "/api/discovery/expansion-sessions/does-not-exist/agent-flows",
            json={"provider_id": "prov-1", "backend_url": "http://fake-backend", "backend_token": "tok"},
        )
    assert resp.status_code == 404


async def test_generate_agent_flows_400_when_expansion_not_completed() -> None:
    session_id = "sess-running"
    await repository.insert_expansion_session(session_id, "cand-1", "snap-1")  # status='running'

    async with await _client() as client:
        resp = await client.post(
            f"/api/discovery/expansion-sessions/{session_id}/agent-flows",
            json={"provider_id": "prov-1", "backend_url": "http://fake-backend", "backend_token": "tok"},
        )
    assert resp.status_code == 400


async def test_generate_agent_flows_missing_backend_config_returns_400_not_500(
    tmp_path: Path,
) -> None:
    session_id = await _seed_completed_expansion_session(tmp_path)

    async with await _client() as client:
        resp = await client.post(
            f"/api/discovery/expansion-sessions/{session_id}/agent-flows",
            json={"provider_id": "prov-1"},  # no backend_url/backend_token, no .aeh/config.yaml
        )
    assert resp.status_code == 400


async def test_generate_agent_flows_missing_local_path_returns_400_not_500(
    tmp_path: Path, monkeypatch
) -> None:
    """Regression guard: an inner HTTPException must not get re-wrapped into a 500."""
    session_id = await _seed_completed_expansion_session(tmp_path)

    async def fake_get_snapshot_missing_local_path(self, snapshot_id: str) -> dict:
        return {}  # no "local_path" key

    monkeypatch.setattr(CodeSpectraClient, "get_snapshot", fake_get_snapshot_missing_local_path)

    async with await _client() as client:
        resp = await client.post(
            f"/api/discovery/expansion-sessions/{session_id}/agent-flows",
            json={"provider_id": "prov-1", "backend_url": "http://fake-backend", "backend_token": "tok"},
        )
    assert resp.status_code == 400
    assert "local_path" in resp.json()["detail"]
