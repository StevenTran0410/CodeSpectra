from __future__ import annotations

import json
from pathlib import Path
import pytest
import yaml
from httpx import ASGITransport, AsyncClient
from fastapi import HTTPException

from agent_eval_harness.llm.client import LLMResponse
from agent_eval_harness.llm.fake_client import FakeLLMClient
from agent_eval_harness.metrics.suite import Suite, load_suite
from agent_eval_harness.planning.planner import generate_plan
from agent_eval_harness.store import repository
from agent_eval_harness.store.database import close_db, init_db, get_db
from agent_eval_harness.ui.server import app


@pytest.fixture(autouse=True)
async def _setup_db(tmp_path, monkeypatch):
    monkeypatch.setenv("AEH_DATA_DIR", str(tmp_path))
    await init_db()
    yield
    await close_db()


async def test_generate_plan_returns_valid_suite(tmp_path) -> None:
    map_content = {
        "target_system_id": "test_system",
        "discrepancies": [],
        "components": [
            {
                "id": "my_orchestrator",
                "role": "orchestrator",
                "model": "gpt-4",
                "entry_point": "dummy_module:DummyClass",
                "constraints": [],
                "upstream": [],
                "downstream": [],
            }
        ],
    }
    map_file = tmp_path / "system_map.yaml"
    map_file.write_text(yaml.dump(map_content))

    llm_client = FakeLLMClient(LLMResponse(content='{"rubric_text": "Fake rubric info"}', model="fake"))
    suite = await generate_plan(map_file, llm_client)

    assert len(suite.entries) > 0
    for entry in suite.entries:
        assert entry.id is not None
        assert entry.component == "my_orchestrator"
        assert entry.metric is not None
        assert entry.provenance in ("rule", "human_added", "llm_suggested")


async def test_put_round_trip_flips_provenance_correctly(tmp_path) -> None:
    plan_file = tmp_path / "test_map_plan.yaml"
    initial_suite = {
        "entries": [
            {
                "id": "writer.faithfulness",
                "component": "writer",
                "metric": "ragas.faithfulness",
                "metric_class": "llm_judge",
                "rationale": "Initial rule rationale",
                "provenance": "rule",
                "params": {},
            },
            {
                "id": "writer.answer_relevancy",
                "component": "writer",
                "metric": "ragas.answer_relevancy",
                "metric_class": "llm_judge",
                "rationale": "Unchanged rule rationale",
                "provenance": "rule",
                "params": {},
            },
        ]
    }
    plan_file.write_text(yaml.dump(initial_suite), encoding="utf-8")

    db = get_db()
    session_id = "test-expansion-session"
    await db.execute(
        "INSERT INTO expansion_sessions (id, candidate_id, snapshot_id, status, map_path, plan_path, created_at) "
        "VALUES (?, 'cand-1', 'snap-1', 'completed', 'test_map.yaml', ?, '2026-07-07T12:00:00Z')",
        (session_id, str(plan_file)),
    )
    await db.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        update_payload = {
            "entries": [
                {
                    "id": "writer.faithfulness",
                    "component": "writer",
                    "metric": "ragas.faithfulness",
                    "metric_class": "llm_judge",
                    "rationale": "Changed rationale by human!",
                    "provenance": "rule",
                    "params": {},
                },
                {
                    "id": "writer.answer_relevancy",
                    "component": "writer",
                    "metric": "ragas.answer_relevancy",
                    "metric_class": "llm_judge",
                    "rationale": "Unchanged rule rationale",
                    "provenance": "rule",
                    "params": {},
                },
            ]
        }
        res = await client.put(
            f"/api/discovery/expansion-sessions/{session_id}/plan",
            json=update_payload,
        )
        assert res.status_code == 200
        assert res.json() == {"success": True}

        updated_suite = load_suite(plan_file)
        entries_by_id = {e.id: e for e in updated_suite.entries}

        assert entries_by_id["writer.faithfulness"].provenance == "human_added"
        assert entries_by_id["writer.answer_relevancy"].provenance == "rule"


async def test_advance_refuses_to_skip_review_gate() -> None:
    db = get_db()
    session_id = "test-discovery-session"
    await db.execute(
        "INSERT INTO discovery_sessions (id, repo_ref, snapshot_id, status, pipeline_stage, created_at) "
        "VALUES (?, 'repo-1', 'snap-1', 'completed', 'awaiting_candidate_review', '2026-07-07T12:00:00Z')",
        (session_id,),
    )
    await db.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        payload = {}
        res = await client.post(
            f"/api/discovery/sessions/{session_id}/advance",
            json=payload,
        )
        assert res.status_code == 400
        assert "Must provide confirmed_candidates" in res.json()["detail"]

        payload_ok = {"confirmed_candidates": ["cand-1"]}
        res_ok = await client.post(
            f"/api/discovery/sessions/{session_id}/advance",
            json=payload_ok,
        )
        assert res_ok.status_code == 200
        assert res_ok.json() == {"pipeline_stage": "expanding"}
