"""Injection runner — top-level suite entry point (CS-289 Phase 0, Workstream D5).

Requires the backend source importable via the sys.path bridge, same gate as
test_injection_agent_invokers.py.
"""
from __future__ import annotations

import json

import pytest

from agent_eval_harness.injection.backend_bridge import ensure_backend_importable
from agent_eval_harness.store.database import close_db, init_db

try:
    ensure_backend_importable()
    from domain.model_connector.service import ProviderConfigService
    from domain.model_connector.types import ChatMessage, ChatRequest, ChatResponse
    _BACKEND = True
except Exception:
    _BACKEND = False

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not _BACKEND, reason="backend source not importable (set backend_source_path)"),
]


@pytest.fixture(autouse=True)
async def _setup_db(tmp_path, monkeypatch):
    """A DB-writing test file must own its lifecycle (init/close), not rely on the session
    fixture — other files in this suite close the shared module-level connection in their own
    teardown (test_defect_leak_prevention.py, test_ui_api.py, ...), so relying on session state
    alone is order-dependent and flaky."""
    monkeypatch.setenv("AEH_DATA_DIR", str(tmp_path))
    await init_db()
    yield
    await close_db()

_K_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_confidence": {"type": "string"},
        "section_scores": {"type": "object"},
        "weakest_sections": {"type": "array"},
        "coverage_percentage": {"type": "number"},
        "notes": {"type": "string"},
        "blind_spots": {"type": "array"},
    },
    "required": ["overall_confidence"],
}


async def test_proxy_provider_service_delegates_to_llm_client() -> None:
    from agent_eval_harness.injection.proxy_provider_service import ProxyProviderService
    from agent_eval_harness.llm.client import LLMResponse
    from agent_eval_harness.llm.fake_client import FakeLLMClient

    fake = FakeLLMClient(LLMResponse(content='{"ok": true}', model="fake-model"))
    svc = ProxyProviderService(fake)

    resp = await svc.chat(
        ChatRequest(
            provider_id="p1",
            model_id="m1",
            messages=[ChatMessage(role="system", content="sys"), ChatMessage(role="user", content="hi")],
            max_completion_tokens=500,
            temperature=0.3,
            json_mode=True,
        )
    )

    assert isinstance(resp, ChatResponse)
    assert resp.content == '{"ok": true}'
    assert resp.provider_id == "p1"
    sent = fake.calls[0]
    assert [m.content for m in sent] == ["sys", "hi"]


async def test_run_synthetic_agent_io_suite_invokes_scores_and_persists() -> None:
    pytest.importorskip("deepeval")
    from agent_eval_harness.injection.suite_runner import run_synthetic_agent_io_suite
    from agent_eval_harness.llm.client import LLMResponse
    from agent_eval_harness.llm.fake_client import FakeLLMClient
    from agent_eval_harness.planning.contract import EvaluationContract, OutputContract
    from agent_eval_harness.store import repository

    agent_output = {
        "overall_confidence": "high",
        "section_scores": {"A": "high"},
        "weakest_sections": ["B"],
        "coverage_percentage": 80.0,
        "notes": "fine",
        "blind_spots": [],
    }
    agent_llm_client = FakeLLMClient(LLMResponse(content=json.dumps(agent_output), model="agent-model"))

    steps_resp = LLMResponse(content=json.dumps({"steps": ["Check groundedness."]}), model="judge-model")
    score_resp = LLMResponse(content=json.dumps({"score": 9, "reason": "grounded"}), model="judge-model")
    judge_llm_client = FakeLLMClient([steps_resp, score_resp])

    contract = EvaluationContract(agent_id="auditor", output=OutputContract(json_schema=_K_SCHEMA))

    dataset_cases = [
        {
            "id": "case-1",
            "input_json": json.dumps({
                "shape": "all_sections",
                "all_sections": {L: {"confidence": "high", "blind_spots": []} for L in "ABCDEFGHIJ"},
            }),
            "expected_json": None,
        }
    ]

    result = await run_synthetic_agent_io_suite(
        "auditor", "target-1", contract, dataset_cases, "p1", "m1",
        agent_llm_client, judge_llm_client,
    )

    assert result.errors == []
    assert len(agent_llm_client.calls) == 1  # the real agent's one LLM call
    metric_names = [r.metric_name for r in result.results]
    assert "assertion.schema_valid" in metric_names
    assert any("groundedness" in n for n in metric_names)

    persisted = []
    db = repository.get_db()
    async with db.execute("SELECT * FROM evaluations WHERE run_id = ?", (result.run_id,)) as cur:
        rows = await cur.fetchall()
        persisted = [dict(r) for r in rows]
    assert len(persisted) == len(result.results)
    assert all(p["agent_id"] == "auditor" for p in persisted)

    runs = []
    async with db.execute("SELECT * FROM runs WHERE id = ?", (result.run_id,)) as cur:
        rows = await cur.fetchall()
        runs = [dict(r) for r in rows]
    assert runs[0]["status"] == "completed"


async def test_run_synthetic_agent_io_suite_generalizes_to_rag_single_shot_archetype() -> None:
    """Proves the shared invoke->score->persist pipeline isn't fan_in_judge-specific --
    exercises glossary (rag_single_shot), a real mocked-retrieval agent."""
    pytest.importorskip("deepeval")
    from agent_eval_harness.injection.suite_runner import run_synthetic_agent_io_suite
    from agent_eval_harness.llm.client import LLMResponse
    from agent_eval_harness.llm.fake_client import FakeLLMClient
    from agent_eval_harness.planning.contract import EvaluationContract, OutputContract

    i_schema = {
        "type": "object",
        "properties": {"terms": {"type": "array"}, "confidence": {"type": "string"}, "blind_spots": {"type": "array"}},
        "required": ["terms", "confidence"],
    }
    agent_output = {
        "terms": [{"term": "Widget", "definition": "a thing", "evidence_files": ["src/widget.py"]}],
        "confidence": "high",
        "blind_spots": [],
    }
    agent_llm_client = FakeLLMClient(LLMResponse(content=json.dumps(agent_output), model="agent-model"))
    judge_llm_client = FakeLLMClient([
        LLMResponse(content=json.dumps({"steps": ["Check groundedness."]}), model="judge-model"),
        LLMResponse(content=json.dumps({"score": 9, "reason": "grounded"}), model="judge-model"),
    ])

    contract = EvaluationContract(agent_id="glossary", output=OutputContract(json_schema=i_schema))
    dataset_cases = [
        {
            "id": "case-1",
            "input_json": json.dumps({
                "shape": "retrieval_only",
                "bundle": {"evidences": [{"rel_path": "src/widget.py", "excerpt": "class Widget: pass"}]},
            }),
            "expected_json": None,
        }
    ]

    result = await run_synthetic_agent_io_suite(
        "glossary", "target-1", contract, dataset_cases, "p1", "m1",
        agent_llm_client, judge_llm_client,
    )

    assert result.errors == []
    assert len(agent_llm_client.calls) == 1
    metric_names = [r.metric_name for r in result.results]
    assert "assertion.schema_valid" in metric_names
    assert result.results[0].passed is True


async def test_run_synthetic_agent_io_suite_records_invocation_errors_without_aborting() -> None:
    from agent_eval_harness.injection.suite_runner import run_synthetic_agent_io_suite
    from agent_eval_harness.llm.fake_client import FakeLLMClient
    from agent_eval_harness.llm.client import LLMResponse
    from agent_eval_harness.planning.contract import EvaluationContract

    contract = EvaluationContract(agent_id="auditor")
    dataset_cases = [
        {"id": "bad-case", "input_json": json.dumps({"shape": "all_sections"}), "expected_json": None},
    ]
    agent_llm_client = FakeLLMClient(LLMResponse(content="{}", model="agent-model"))
    judge_llm_client = FakeLLMClient(LLMResponse(content="{}", model="judge-model"))

    result = await run_synthetic_agent_io_suite(
        "auditor", "target-1", contract, dataset_cases, "p1", "m1",
        agent_llm_client, judge_llm_client,
    )

    assert len(result.errors) == 1
    assert result.errors[0]["case_id"] == "bad-case"
    assert result.results == []
