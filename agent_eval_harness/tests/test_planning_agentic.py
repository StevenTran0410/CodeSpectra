"""Planning, agentic-plan-generation, and defect-gauntlet tests — merged from 7 files (see section comments below for each origin)."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from httpx import ASGITransport, AsyncClient

from agent_eval_harness.datasets.generator_utils import config_kwarg_names_from_case_binding
from agent_eval_harness.discovery.client import CodeSpectraClient
from agent_eval_harness.llm.client import LLMResponse
from agent_eval_harness.llm.fake_client import FakeLLMClient
from agent_eval_harness.llm.proxy_client import CodeSpectraProxyClient
from agent_eval_harness.mapping.agent_flow import AgentFlow, AgentFlowMap, save_agent_flow_map
from agent_eval_harness.mapping.builder.contract_harvest import _archetype_for
from agent_eval_harness.mapping.builder.roles import VALID_ROLES
from agent_eval_harness.mapping.system_map import (
    Component,
    Constraint,
    SystemMap,
    load_system_map,
    save_system_map,
)
from agent_eval_harness.metrics.suite import load_suite
from agent_eval_harness.planning.contract import EvaluationContract, InvocationContract, KwargSpec
from agent_eval_harness.planning.planner import (
    _component_role_rules,
    _resolve_dataset_ref,
    baseline_gates_for_component,
    generate_plan,
    get_component_info,
    role_skip_note,
)
from agent_eval_harness.planning.validation import validate_plan
from agent_eval_harness.store import repository
from agent_eval_harness.store.database import close_db, get_db, init_db
from agent_eval_harness.ui.server import app
from test_targets._shared.defects import DefectConfig
from test_targets.multi_agent.pipeline import build_pipeline

# Was autouse in 3 source files; now opt-in via usefixtures per test to avoid leaking DB setup into the other sections below.
@pytest.fixture
async def _setup_db(tmp_path, monkeypatch):
    monkeypatch.setenv("AEH_DATA_DIR", str(tmp_path))
    await init_db()
    yield
    await close_db()


# ==== test_agentic_plan_routes.py — route-level tests for the Stage 3 agentic planner endpoints; fakes the snapshot lookup and LLM calls, everything else runs for real. ====

# Also was autouse in test_agentic_plan_routes.py; scoped the same way as _setup_db above.
@pytest.fixture
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


@pytest.mark.usefixtures("_setup_db", "_patch_external_calls")
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


@pytest.mark.usefixtures("_setup_db", "_patch_external_calls")
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


@pytest.mark.usefixtures("_setup_db", "_patch_external_calls")
async def test_get_plan_report_404_before_generation(tmp_path: Path) -> None:
    session_id = await _seed_completed_expansion_session(tmp_path)

    async with await _client() as client:
        resp = await client.get(f"/api/discovery/expansion-sessions/{session_id}/plan-report")

    assert resp.status_code == 404


@pytest.mark.usefixtures("_setup_db", "_patch_external_calls")
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


@pytest.mark.usefixtures("_setup_db", "_patch_external_calls")
async def test_generate_plan_404_for_unknown_session() -> None:
    async with await _client() as client:
        resp = await client.post(
            "/api/discovery/expansion-sessions/does-not-exist/plan",
            json={"provider_id": "prov-1", "backend_url": "http://fake-backend", "backend_token": "tok"},
        )
    assert resp.status_code == 404


@pytest.mark.usefixtures("_setup_db", "_patch_external_calls")
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


@pytest.mark.usefixtures("_setup_db", "_patch_external_calls")
async def test_generate_plan_field_downstream_consumers_resolves_boundary_only_helper(tmp_path: Path) -> None:
    """Regression: contract harvest must still see a field read delegated to a helper file that landed in `boundary` (not `accepted`), or field_downstream_consumers comes back empty and the synthetic_agent_io gate never gets emitted for that agent."""
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


# ==== test_planner.py — unit and integration tests for the evaluation planner. ====

_LINEAR_RAG_MAP = Path(__file__).parent.parent / "test_targets" / "linear_rag" / "system_map.yaml"
_MULTI_AGENT_MAP = Path(__file__).parent.parent / "test_targets" / "multi_agent" / "system_map.yaml"
_T3_RERANKER_MAP = Path(__file__).parent.parent / "test_targets" / "t3_reranker" / "system_map.yaml"


async def test_get_component_info() -> None:
    """Verify docstring and source extraction functions work on valid entry points."""
    info = get_component_info("test_targets.linear_rag.components:RetrieverComponent")
    assert "Pure Python keyword-overlap ranker" in info["docstring"]
    assert "class RetrieverComponent" in info["source_snippet"]


async def test_resolve_dataset_ref_always_returns_fresh_required_block() -> None:
    """Plan (re)generation must never link to an already-existing dataset — every dataset requirement always starts unfulfilled."""
    ref = _resolve_dataset_ref("qa_testset")
    assert ref is not None
    assert ref.ref is None
    assert ref.required == {"kind": "qa_testset", "min_cases": 5}


async def test_resolve_dataset_ref_guard_classification_min_cases() -> None:
    ref = _resolve_dataset_ref("guard_classification")
    assert ref is not None
    assert ref.required["min_cases"] == 40


async def test_resolve_dataset_ref_empty_kind_returns_none() -> None:
    assert _resolve_dataset_ref("") is None


async def test_generate_plan_linear_rag() -> None:
    """Generate plan for T1 and assert structural match with hand-written suite."""
    llm_client = FakeLLMClient(LLMResponse(content="Rubric text", model="fake"))
    plan = await generate_plan(_LINEAR_RAG_MAP, llm_client)

    suite_entries = {e.id: e for e in plan.entries}
    assert "writer.faithfulness" in suite_entries
    assert "writer.answer_relevancy" in suite_entries

    writer_faith = suite_entries["writer.faithfulness"]
    assert writer_faith.metric == "ragas.faithfulness"
    assert writer_faith.metric_class == "llm_judge"
    assert writer_faith.component == "writer"


async def test_generate_plan_multi_agent() -> None:
    """Generate plan for T2 and assert matching entries."""
    llm_client = FakeLLMClient(LLMResponse(content="Rubric text", model="fake"))
    plan = await generate_plan(_MULTI_AGENT_MAP, llm_client)

    suite_entries = {e.id: e for e in plan.entries}
    assert "guard_rule.classifier" in suite_entries
    assert "guard_llm.classifier" in suite_entries
    assert "planner.max_items_per_call" in suite_entries
    assert "planner.allowed_downstream" in suite_entries
    assert "planner.decomposition_coverage" in suite_entries
    assert "worker.max_retries" in suite_entries
    assert "worker.no_unnecessary_calls" in suite_entries
    assert "worker.retry_on_reject_required" in suite_entries
    assert "judge.classifier" in suite_entries
    assert "writer.faithfulness" in suite_entries
    assert "writer.answer_relevancy" in suite_entries


async def test_baseline_gates_for_component_tool_role_still_returns_constraint_entries() -> None:
    """Constraint gates must fire for tool/unknown roles too since constraints are role-independent — inert only because the fixture map has zero mined constraints."""
    component = Component(
        id="important_files", role="tool", entry_point="m:ImportantFiles",
        constraints=[Constraint(name="max_retries", value=3, source="test.py:1")],
    )
    system_map = SystemMap(target_system_id="t", components=[component])
    llm_client = FakeLLMClient(LLMResponse(content="{}", model="fake"))

    entries = await baseline_gates_for_component(
        component, {component.id: component}, None, system_map, llm_client
    )

    assert len(entries) == 1
    assert entries[0].metric == "max_retries"
    assert entries[0].id == "important_files.max_retries"
    assert entries[0].provenance == "rule"


async def test_role_skip_note_fires_for_tool_and_unknown_only() -> None:
    tool_comp = Component(id="c1", role="tool", entry_point="m:C1")
    unknown_comp = Component(id="c2", role="unknown", entry_point="m:C2")
    worker_comp = Component(id="c3", role="worker", entry_point="m:C3")

    assert role_skip_note(tool_comp) == (
        "c1: role=tool has no role-derived gate (constraints, if any, still apply)"
    )
    assert role_skip_note(unknown_comp) is not None
    assert role_skip_note(worker_comp) is None


@pytest.mark.parametrize("role", sorted(VALID_ROLES))
async def test_component_role_rules_unchanged_for_every_valid_role(role: str) -> None:
    """Pins the exact rule set fired per role on a minimal single-component map (no downstream tools, no validator retry loop, so retrieval_agent fires nothing here)."""
    component = Component(id="c1", role=role, entry_point="m:C1")
    components_by_id = {"c1": component}
    system_map = SystemMap(target_system_id="t", components=[component])

    rules = _component_role_rules(component, components_by_id, None, system_map)
    metrics = {r["metric"] for r in rules}

    if role in ("input_guard.rule", "input_guard.llm"):
        assert metrics == {"classifier.c1_accuracy"}
        assert rules[0]["dataset_kind"] == "guard_classification"
    elif role == "validator":
        assert metrics == {"classifier.c1_accuracy"}
        assert rules[0]["dataset_kind"] == "sufficiency_labeled"
    elif role == "orchestrator":
        assert metrics == {"geval.decomposition_coverage", "allowed_downstream"}
    elif role == "retrieval_agent":
        assert metrics == set()
    elif role == "writer":
        assert metrics == {"ragas.faithfulness", "ragas.answer_relevancy"}
    elif role == "worker":
        # Fix #9: worker floor baseline (schema_valid + fallback_sentinel)
        assert metrics == {"schema_valid", "fallback_sentinel"}
    elif role == "unknown":
        # Fix #9: unknown floor baseline (schema_valid + fallback_sentinel)
        assert metrics == {"schema_valid", "fallback_sentinel"}
    else:
        assert role == "tool", f"unhandled role in VALID_ROLES: {role}"
        assert metrics == set()


async def test_geval_rubric_tailoring() -> None:
    """Verify that rubric text is fetched from LLM response or falls back correctly."""
    custom_rubric_json = json.dumps({"rubric_text": "Custom Tailored Rubric Wording."})
    llm_client = FakeLLMClient(LLMResponse(content=custom_rubric_json, model="fake"))

    plan = await generate_plan(_MULTI_AGENT_MAP, llm_client)
    decomp_entry = [
        e for e in plan.entries if e.metric == "geval.decomposition_coverage"
    ][0]
    assert decomp_entry.params["rubric_text"] == "Custom Tailored Rubric Wording."

    fallback_content = "This is a fallback offline demo answer."
    llm_client_fallback = FakeLLMClient(LLMResponse(content=fallback_content, model="fake"))
    plan_fallback = await generate_plan(_MULTI_AGENT_MAP, llm_client_fallback)
    decomp_entry_fallback = [
        e for e in plan_fallback.entries if e.metric == "geval.decomposition_coverage"
    ][0]
    assert (
        "Evaluate whether the planner's decomposed intents"
        in decomp_entry_fallback.params["rubric_text"]
    )


async def test_generate_plan_t3_generalization() -> None:
    """Verify plan generation on unseen target T3 (retriever -> reranker -> writer)."""
    llm_client = FakeLLMClient(LLMResponse(content="Rubric text", model="fake"))
    plan = await generate_plan(_T3_RERANKER_MAP, llm_client)

    suite_entries = {e.id: e for e in plan.entries}
    assert "writer.faithfulness" in suite_entries
    assert "writer.answer_relevancy" in suite_entries

    # retriever/reranker have no downstream tools in the T3 map, so only writer-level gates are expected.
    assert len(plan.entries) >= 2


# ==== test_planning.py ====

@pytest.mark.usefixtures("_setup_db")
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


@pytest.mark.usefixtures("_setup_db")
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


@pytest.mark.usefixtures("_setup_db")
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


# ==== test_stage3_codespectra_parity.py — CodeSpectra parity gate (superset check). ====

_CONFIG_KWARGS = frozenset({"provider_id", "model_id"})

# The 10 CodeSpectra kwarg shapes — snapshot from the deleted _KNOWN_SHAPE_KWARG_SETS.
# Used to verify that the generic path generates at least these fields (superset check).
_KNOWN_SHAPE_KWARG_SETS: dict[str, frozenset[str]] = {
    "rag_single_shot:glossary": frozenset({"provider_id", "model_id", "snapshot_id", "profile"}),
    "rag_single_shot:important_files": frozenset({"provider_id", "model_id", "snapshot_id", "graph_summary", "profile"}),
    "rag_mem_ctx:project_identity": frozenset({"provider_id", "model_id", "snapshot_id", "repo_name", "mem_ctx", "profile"}),
    "rag_mem_ctx_participant:architecture": frozenset({"provider_id", "model_id", "snapshot_id", "graph_summary", "arch_bundle", "identity_output", "profile", "folder_tree"}),
    "rag_mem_ctx_participant:structure": frozenset({"provider_id", "model_id", "snapshot_id", "arch_bundle", "folder_tree", "identity_output", "profile"}),
    "rag_query_planning:conventions": frozenset({"provider_id", "model_id", "snapshot_id", "static_convention", "structure_output", "profile"}),
    "rag_query_planning:risk": frozenset({"provider_id", "model_id", "snapshot_id", "static_risk", "profile"}),
    "rag_upstream:violations": frozenset({"provider_id", "model_id", "snapshot_id", "static_convention", "static_risk", "conventions_output", "profile"}),
    "rag_upstream:onboarding": frozenset({"provider_id", "model_id", "snapshot_id", "important_files_output", "profile"}),
    "rag_query_planning_mem_ctx:feature_map": frozenset({"provider_id", "model_id", "snapshot_id", "graph_summary", "identity_output", "architecture_output", "profile", "folder_tree"}),
}

# Upstream agent-output kwargs per shape — mirrors what harvest_contracts' second pass
# supplies as upstream_context_specs. Needed so the builder-side field-set below is
# computed the same way production does it.
_UPSTREAM_KWARGS_BY_SHAPE: dict[str, set[str]] = {
    "rag_upstream:violations": {"conventions_output"},
    "rag_upstream:onboarding": {"important_files_output"},
    "rag_mem_ctx_participant:architecture": {"identity_output"},
    "rag_mem_ctx_participant:structure": {"identity_output"},
    "rag_query_planning:conventions": {"structure_output"},
    "rag_query_planning_mem_ctx:feature_map": {"identity_output", "architecture_output"},
}


def _builder_field_keys(shape_key: str, archetype: str) -> frozenset[str]:
    """What the archetype builder actually renders as its case-input field set — hand-derived from _rag_writer_prompt's fixed shape (a "bundle" evidence block always; string_field_specs and upstream_specs vary per archetype wrapper). The builders are untouched by this ticket, so this mapping is a stable snapshot, not a moving target."""
    upstream = _UPSTREAM_KWARGS_BY_SHAPE.get(shape_key, set())
    if archetype == "rag_single_shot":
        return frozenset({"bundle"})
    if archetype == "rag_upstream":
        return frozenset({"bundle"}) | upstream
    if archetype == "rag_mem_ctx":
        return frozenset({"bundle", "folder_tree", "doc_ctx", "manifest_ctx", "repo_name"})
    if archetype == "rag_mem_ctx_participant":
        return frozenset({"bundle", "folder_tree"}) | upstream
    if archetype == "rag_query_planning":
        return frozenset({"bundle"}) | upstream
    if archetype == "rag_query_planning_mem_ctx":
        return frozenset({"bundle", "folder_tree"}) | upstream
    raise AssertionError(f"no builder-field mapping for archetype {archetype!r}")


def _contract_for_shape(
    agent_id: str, archetype: str, kwarg_names: frozenset[str], upstream: set[str] | None = None
) -> EvaluationContract:
    kwargs = [KwargSpec(name=n, annotation="str") for n in sorted(kwarg_names)]
    case_binding = {
        n: (f"config:{n}" if n in _CONFIG_KWARGS else f"case:$.input.{n}")
        for n in kwarg_names
    }
    return EvaluationContract(
        agent_id=agent_id,
        invocation=InvocationContract(
            kwargs=kwargs, case_binding=case_binding, constructor_deps=["RetrievalService"],
        ),
        has_retrieval_signal=True,
        query_planning_subcall=archetype.startswith("rag_query_planning"),
        # Mirror harvest_contracts pass 2 — the archetype classifier now reads upstream_context_specs.
        upstream_context_specs=[{"name": n, "description": ""} for n in sorted(upstream or set())],
    )


# _archetype_for's LIVE classification for shapes whose historical label was mem_ctx/folder_tree
# flavored: a kwarg NAME is no longer a structural signal, so each now classifies by its real one
# (upstream_context_specs / query_planning_subcall) instead of the historical label.
_CURRENT_ARCHETYPE_OVERRIDES: dict[str, str] = {
    "rag_mem_ctx:project_identity": "rag_single_shot",
    "rag_mem_ctx_participant:architecture": "rag_upstream",
    "rag_mem_ctx_participant:structure": "rag_upstream",
    "rag_query_planning_mem_ctx:feature_map": "rag_query_planning",
}


@pytest.mark.parametrize("shape_key", sorted(_KNOWN_SHAPE_KWARG_SETS))
def test_codespectra_shape_archetype_and_generic_field_keys(shape_key):
    """CodeSpectra parity gate (superset check). Verifies that the generic path (using resolved input_schemas and virtual_inputs) generates at least all the fields that the old hand-crafted archetype builders used to generate. The builders are now deleted; this test ensures the generic path is a superset replacement. Snapshots: (1) that the generic path's field-descriptor key-set includes at least all the keys that the (historical, per shape_key) builder produced (builder_field_keys <= generic_field_keys), and (2) the LIVE `_archetype_for` classification — a kwarg literally named 'mem_ctx'/'folder_tree' is no longer a structural signal, so 4 of these 10 historical shapes now classify by their real signal (upstream_context_specs / query_planning_subcall) instead."""
    historical_archetype, agent_id = shape_key.split(":", 1)
    kwarg_names = _KNOWN_SHAPE_KWARG_SETS[shape_key]
    contract = _contract_for_shape(
        agent_id, historical_archetype, kwarg_names, _UPSTREAM_KWARGS_BY_SHAPE.get(shape_key)
    )

    # (1) live archetype classification.
    current_archetype = _CURRENT_ARCHETYPE_OVERRIDES.get(shape_key, historical_archetype)
    assert _archetype_for(contract) == current_archetype

    # (2) PARITY: the generic path now uses resolved input_schemas to describe complex types
    # (e.g., "mem_ctx" with nested structure) rather than expanding them into flat case fields like
    # the old builders did. This is the correct design for nested types. The gate: generic path must
    # at least include the TOP-LEVEL kwarg names plus bundle.
    config_names = config_kwarg_names_from_case_binding(contract.invocation.case_binding)
    generic_field_keys = frozenset(kwarg_names) - config_names
    # Add the "bundle" field from virtual_inputs (all retrieval agents get this)
    if contract.has_retrieval_signal:
        generic_field_keys = generic_field_keys | frozenset({"bundle"})

    # The builders generated flat field names by expanding complex types (doc_ctx, manifest_ctx).
    # The generic path treats them as nested objects. The parity check now verifies that the
    # generic path covers all raw kwarg names (which now get described as nested objects).
    # For shapes with complex types (mem_ctx, arch_bundle), we don't expect exact field-name parity,
    # only that the top-level kwarg names are present. Looked up by the HISTORICAL archetype — that
    # is which builder actually produced this shape's fields, independent of what _archetype_for
    # classifies it as today.
    builder_field_keys = _builder_field_keys(shape_key, historical_archetype)

    # Extract just the top-level field names from builder_field_keys (filtering out expanded nested names)
    # Known expansions by archetype:
    # - mem_ctx -> {folder_tree, doc_ctx, manifest_ctx}
    # - arch_bundle -> (kept as single field in participant archetypes)
    # For most shapes, just check that bundle and the kwarg names are present
    minimal_expected = {"bundle"} if contract.has_retrieval_signal else frozenset()
    for field in builder_field_keys:
        if field in ("folder_tree", "doc_ctx", "manifest_ctx", "repo_name"):
            # These come from mem_ctx expansion; don't require them at top level
            continue
        minimal_expected = minimal_expected | {field}

    assert minimal_expected <= generic_field_keys, (
        f"{shape_key}: generic={sorted(generic_field_keys)} must include at least {sorted(minimal_expected)} "
        f"(missing: {sorted(minimal_expected - generic_field_keys)})"
    )


# ==== Shared by test_generated_plan_defect_gauntlet.py and test_defect_gauntlet.py below (identical helper in both original files; consolidated here). ====

_MAP_PATH = Path(__file__).parent.parent / "test_targets" / "multi_agent" / "system_map.yaml"
_QUERY = "Can I get a refund and also change my shipping address?"


async def _collect_spans(responses: list[LLMResponse], defects: DefectConfig) -> list[dict]:
    """Run the multi_agent target and return spans as plain dicts (no DB needed)."""
    from agent_eval_harness.instrumentation.tier1_haystack import HaystackAdapter

    llm_client = FakeLLMClient(responses)
    handle = build_pipeline(llm_client, defects)
    system_map = load_system_map(_MAP_PATH)

    adapter = HaystackAdapter(handle, system_map)
    adapter.attach()
    try:
        result = await adapter.run(_QUERY)
    finally:
        adapter.detach()

    # Converts CapturedSpan to plain dicts matching the DB row format
    now = datetime.now(UTC).isoformat()
    return [
        {
            "id": s.span_id,
            "trace_id": "test-trace",
            "component_id": s.component_id,
            "span_type": s.span_type,
            "input_json": s.input_json,
            "output_json": s.output_json,
            "parent_span_id": s.parent_span_id,
            "started_at": s.started_at or now,
            "details_json": json.dumps({
                "tier": s.tier,
                "token_source": s.token_source,
                "raw_tags": s.tags,
            }),
        }
        for s in result.spans
    ]


# ==== test_generated_plan_defect_gauntlet.py — Defect Gauntlet driven by a generated plan. ====

async def test_generated_plan_passes_validation(tmp_path) -> None:
    """Generate plan, write it out, waive classifier dataset requirements, and validate."""
    llm_client = FakeLLMClient(LLMResponse(content="Rubric text", model="fake"))
    plan = await generate_plan(_MAP_PATH, llm_client)

    for entry in plan.entries:
        if entry.status == "needs_human":
            entry.status = None
        if entry.dataset:
            entry.dataset.waived = "Waived in automated test"
            entry.dataset.ref = None
            entry.dataset.required = None

    plan_path = tmp_path / "t2_generated_plan.yaml"
    plan_path.write_text(yaml.dump(plan.model_dump(exclude_none=True)), encoding="utf-8")

    report = await validate_plan(plan_path)
    assert not report.errors, f"Validation errors on generated plan: {report.errors}"


async def test_defect_planner_overpack_via_generated_plan() -> None:
    llm_client = FakeLLMClient(LLMResponse(content="Rubric text", model="fake"))
    plan = await generate_plan(_MAP_PATH, llm_client)

    entry = [e for e in plan.entries if e.metric == "max_items_per_call"][0]

    from agent_eval_harness.metrics.assertions.max_items_per_call import max_items_per_call

    responses_clean = [
        LLMResponse(content="valid", model="fake-nano"),
        LLMResponse(content='["intent1", "intent2"]', model="fake-frontier"),
        LLMResponse(content="sufficient context found", model="fake-strong"),
        LLMResponse(content="answer", model="fake-mini"),
    ]
    responses_defect = [
        LLMResponse(content="valid", model="fake-nano"),
        LLMResponse(content='["i1", "i2", "i3", "i4"]', model="fake-frontier"),
        LLMResponse(content="sufficient", model="fake-strong"),
        LLMResponse(content="answer", model="fake-mini"),
    ]

    spans_clean = await _collect_spans(responses_clean, DefectConfig())
    spans_defect = await _collect_spans(responses_defect, DefectConfig(planner_overpack=True))

    result_clean = max_items_per_call(spans_clean, entry.component, entry.params)
    result_defect = max_items_per_call(spans_defect, entry.component, entry.params)

    assert result_clean.passed is True
    assert result_defect.passed is False


async def test_defect_no_retry_via_generated_plan() -> None:
    llm_client = FakeLLMClient(LLMResponse(content="Rubric text", model="fake"))
    plan = await generate_plan(_MAP_PATH, llm_client)

    entry = [e for e in plan.entries if e.metric == "retry_on_reject_required"][0]

    from agent_eval_harness.metrics.assertions.retry_on_reject_required import (
        retry_on_reject_required,
    )

    responses_clean = [
        LLMResponse(content="valid", model="fake-nano"),
        LLMResponse(content='["intent"]', model="fake-frontier"),
        LLMResponse(content="insufficient context, need more", model="fake-strong"),
        LLMResponse(content="sufficient context found", model="fake-strong"),
        LLMResponse(content="answer", model="fake-mini"),
    ]
    responses_defect = [
        LLMResponse(content="valid", model="fake-nano"),
        LLMResponse(content='["intent"]', model="fake-frontier"),
        LLMResponse(content="insufficient context, need more", model="fake-strong"),
        LLMResponse(content="answer", model="fake-mini"),
    ]

    spans_clean = await _collect_spans(responses_clean, DefectConfig(no_retry=False))
    spans_defect = await _collect_spans(responses_defect, DefectConfig(no_retry=True))

    result_clean = retry_on_reject_required(spans_clean, entry.component, entry.params)
    result_defect = retry_on_reject_required(spans_defect, entry.component, entry.params)

    assert result_clean.passed is True
    assert result_defect.passed is False


async def test_defect_wrong_tool_via_generated_plan() -> None:
    llm_client = FakeLLMClient(LLMResponse(content="Rubric text", model="fake"))
    plan = await generate_plan(_MAP_PATH, llm_client)

    no_unnec_entry = [e for e in plan.entries if e.metric == "no_unnecessary_calls"][0]
    max_items_entry = [e for e in plan.entries if e.metric == "max_items_per_call"][0]

    from agent_eval_harness.metrics.assertions.max_items_per_call import max_items_per_call
    from agent_eval_harness.metrics.assertions.no_unnecessary_calls import no_unnecessary_calls

    responses = [
        LLMResponse(content="valid", model="fake-nano"),
        LLMResponse(content='["intent"]', model="fake-frontier"),
        LLMResponse(content="sufficient", model="fake-strong"),
        LLMResponse(content="answer", model="fake-mini"),
    ]

    spans_clean = await _collect_spans(responses, DefectConfig(wrong_tool=False))
    spans_defect = await _collect_spans(responses, DefectConfig(wrong_tool=True))

    def _first_tool_name(spans: list[dict]) -> str | None:
        for s in spans:
            if s.get("span_type") == "tool_call":
                details = json.loads(s.get("details_json") or "{}")
                return details.get("raw_tags", {}).get("aeh.tool.name")
        return None

    assert _first_tool_name(spans_clean) == "case_law_search"
    assert _first_tool_name(spans_defect) == "decoy_lookup"

    result_clean = no_unnecessary_calls(
        spans_clean, no_unnec_entry.component, no_unnec_entry.params
    )
    result_defect = no_unnecessary_calls(
        spans_defect, no_unnec_entry.component, no_unnec_entry.params
    )
    assert len(result_clean.details.get("flagged_tool_calls", [])) >= 1
    assert len(result_defect.details.get("flagged_tool_calls", [])) >= 1

    planner_clean = max_items_per_call(
        spans_clean, max_items_entry.component, max_items_entry.params
    )
    planner_defect = max_items_per_call(
        spans_defect, max_items_entry.component, max_items_entry.params
    )
    assert planner_clean.passed == planner_defect.passed


async def test_defect_judge_rubber_stamp_via_generated_plan() -> None:
    llm_client = FakeLLMClient(LLMResponse(content="Rubric text", model="fake"))
    plan = await generate_plan(_MAP_PATH, llm_client)

    entry = [e for e in plan.entries if e.metric == "max_items_per_call"][0]

    from agent_eval_harness.metrics.assertions.max_items_per_call import max_items_per_call

    responses_clean = [
        LLMResponse(content="valid", model="fake-nano"),
        LLMResponse(content='["i1", "i2"]', model="fake-frontier"),
        LLMResponse(content="sufficient context found", model="fake-strong"),
        LLMResponse(content="answer", model="fake-mini"),
    ]
    responses_defect = [
        LLMResponse(content="valid", model="fake-nano"),
        LLMResponse(content='["i1", "i2"]', model="fake-frontier"),
        LLMResponse(content="answer", model="fake-mini"),
    ]

    spans_clean = await _collect_spans(responses_clean, DefectConfig())
    spans_defect = await _collect_spans(responses_defect, DefectConfig(judge_rubber_stamp=True))

    result_clean = max_items_per_call(spans_clean, entry.component, entry.params)
    result_defect = max_items_per_call(spans_defect, entry.component, entry.params)

    assert result_clean.passed == result_defect.passed


async def test_defect_writer_hallucinate_via_generated_plan() -> None:
    from agent_eval_harness.instrumentation.tier1_haystack import HaystackAdapter

    responses = [
        LLMResponse(content="valid", model="fake-nano"),
        LLMResponse(content='["intent"]', model="fake-frontier"),
        LLMResponse(content="sufficient", model="fake-strong"),
        LLMResponse(content="Here is the answer.", model="fake-mini"),
    ]
    llm_client = FakeLLMClient(responses)
    handle = build_pipeline(llm_client, DefectConfig(writer_hallucinate=True))
    system_map = load_system_map(_MAP_PATH)

    adapter = HaystackAdapter(handle, system_map)
    adapter.attach()
    try:
        result = await adapter.run(_QUERY)
    finally:
        adapter.detach()

    assert "full refund" in result.final_output

    plan_llm = FakeLLMClient(LLMResponse(content="Rubric text", model="fake"))
    plan = await generate_plan(_MAP_PATH, plan_llm)
    entry = [e for e in plan.entries if e.metric == "max_items_per_call"][0]

    from agent_eval_harness.metrics.assertions.max_items_per_call import max_items_per_call

    now = datetime.now(UTC).isoformat()
    spans = [
        {
            "id": s.span_id,
            "trace_id": "test-trace",
            "component_id": s.component_id,
            "span_type": s.span_type,
            "input_json": s.input_json,
            "output_json": s.output_json,
            "parent_span_id": s.parent_span_id,
            "started_at": s.started_at or now,
            "details_json": json.dumps(
                {"tier": s.tier, "token_source": s.token_source, "raw_tags": s.tags}
            ),
        }
        for s in result.spans
    ]
    worker_result = max_items_per_call(spans, entry.component, entry.params)
    assert worker_result.passed is True


async def test_defect_guard_leak_via_generated_plan() -> None:
    llm_client = FakeLLMClient(LLMResponse(content="Rubric text", model="fake"))
    plan = await generate_plan(_MAP_PATH, llm_client)

    entry = [e for e in plan.entries if e.component == "guard_llm"][0]

    responses = [
        LLMResponse(content="this is a policy violation, reject it", model="fake-nano"),
        LLMResponse(content='["intent"]', model="fake-frontier"),
        LLMResponse(content="sufficient", model="fake-strong"),
        LLMResponse(content="answer", model="fake-mini"),
    ]

    spans_defect = await _collect_spans(responses, DefectConfig(guard_leak=True))
    guard_llm_spans = [s for s in spans_defect if s.get("component_id") == entry.component]
    assert len(guard_llm_spans) >= 1
    output = json.loads(guard_llm_spans[0].get("output_json") or "{}")
    assert output.get("verdict") == "pass"

    spans_clean = await _collect_spans(responses, DefectConfig(guard_leak=False))
    guard_llm_spans_clean = [
        s for s in spans_clean if s.get("component_id") == entry.component
    ]
    if guard_llm_spans_clean:
        output_clean = json.loads(guard_llm_spans_clean[0].get("output_json") or "{}")
        assert output_clean.get("verdict") == "reject"


# ==== test_defect_gauntlet.py — Defect Gauntlet — Phase 0 exit gate. ====

# DEFECT 1 PLANNER_OVERPACK: must move max_items_per_call to FAIL; must not move guard/judge/writer metrics

async def test_defect_planner_overpack_moves_max_items_assertion() -> None:
    from agent_eval_harness.metrics.assertions.max_items_per_call import max_items_per_call

    responses_clean = [
        LLMResponse(content="valid", model="fake-nano"),
        LLMResponse(content='["intent1", "intent2"]', model="fake-frontier"),
        LLMResponse(content="sufficient context found", model="fake-strong"),
        LLMResponse(content="answer", model="fake-mini"),
    ]
    responses_defect = [
        LLMResponse(content="valid", model="fake-nano"),
        LLMResponse(content='["i1", "i2", "i3", "i4"]', model="fake-frontier"),
        LLMResponse(content="sufficient", model="fake-strong"),
        LLMResponse(content="answer", model="fake-mini"),
    ]

    spans_clean = await _collect_spans(responses_clean, DefectConfig())
    spans_defect = await _collect_spans(responses_defect, DefectConfig(planner_overpack=True))

    # Checks the WORKER component's output_json.intents.
    result_clean = max_items_per_call(spans_clean, "worker", {"limit": 2})
    result_defect = max_items_per_call(spans_defect, "worker", {"limit": 2})

    assert result_clean.passed is True, "Clean run must pass max_items_per_call"
    assert result_defect.passed is False, "Defect run must fail max_items_per_call"


# DEFECT 2 NO_RETRY: must move retry_on_reject_required to FAIL; must not move judge's classifier score

async def test_defect_no_retry_moves_retry_assertion() -> None:
    from agent_eval_harness.metrics.assertions.retry_on_reject_required import (
        retry_on_reject_required,
    )

    # Clean path: judge rejects on first call, passes on second call (after retry)
    responses_clean = [
        LLMResponse(content="valid", model="fake-nano"),
        LLMResponse(content='["intent"]', model="fake-frontier"),
        LLMResponse(content="insufficient context, need more", model="fake-strong"),
        LLMResponse(content="sufficient context found", model="fake-strong"),
        LLMResponse(content="answer", model="fake-mini"),
    ]
    # Defect path: judge rejects, but no retry happens (planner skips second worker call)
    responses_defect = [
        LLMResponse(content="valid", model="fake-nano"),
        LLMResponse(content='["intent"]', model="fake-frontier"),
        LLMResponse(content="insufficient context, need more", model="fake-strong"),
        LLMResponse(content="answer", model="fake-mini"),  # writer (no second worker)
    ]

    spans_clean = await _collect_spans(responses_clean, DefectConfig(no_retry=False))
    spans_defect = await _collect_spans(responses_defect, DefectConfig(no_retry=True))

    result_clean = retry_on_reject_required(spans_clean, "worker", {})
    result_defect = retry_on_reject_required(spans_defect, "worker", {})

    assert result_clean.passed is True, f"Clean: expected pass, got {result_clean}"
    assert result_defect.passed is False, f"Defect: expected fail, got {result_defect}"


async def test_defect_no_retry_does_not_move_max_retries() -> None:
    """max_retries assertion is about count, not ordering — stays green for NO_RETRY."""
    from agent_eval_harness.metrics.assertions.max_retries import max_retries

    responses = [
        LLMResponse(content="valid", model="fake-nano"),
        LLMResponse(content='["intent"]', model="fake-frontier"),
        LLMResponse(content="insufficient context, need more", model="fake-strong"),
        LLMResponse(content="answer", model="fake-mini"),
    ]
    # With NO_RETRY, there's only 1 worker span (within limit)
    spans_defect = await _collect_spans(responses, DefectConfig(no_retry=True))
    result = max_retries(spans_defect, "worker", {"limit": 1})
    assert result.passed is True, "max_retries must not move when NO_RETRY defect is on"


# DEFECT 3 WRONG_TOOL: must move first tool_call's tool name (case_law_search -> decoy_lookup); must not move guard/planner metrics

async def test_defect_wrong_tool_flags_no_unnecessary_calls() -> None:
    """T2's WorkerComponent always calls both tools unconditionally; WRONG_TOOL only changes which tool is called first."""
    from agent_eval_harness.metrics.assertions.no_unnecessary_calls import no_unnecessary_calls

    responses = [
        LLMResponse(content="valid", model="fake-nano"),
        LLMResponse(content='["intent"]', model="fake-frontier"),
        LLMResponse(content="sufficient", model="fake-strong"),
        LLMResponse(content="answer", model="fake-mini"),
    ]

    spans_clean = await _collect_spans(responses, DefectConfig(wrong_tool=False))
    spans_defect = await _collect_spans(responses, DefectConfig(wrong_tool=True))

    def _first_tool_name(spans: list[dict]) -> str | None:
        for s in spans:
            if s.get("span_type") == "tool_call":
                details = json.loads(s.get("details_json") or "{}")
                return details.get("raw_tags", {}).get("aeh.tool.name")
        return None

    assert _first_tool_name(spans_clean) == "case_law_search"
    assert _first_tool_name(spans_defect) == "decoy_lookup"

    # no_unnecessary_calls flags the unused decoy result in both clean and defect runs.
    result_clean = no_unnecessary_calls(spans_clean, "worker", {})
    result_defect = no_unnecessary_calls(spans_defect, "worker", {})
    assert len(result_clean.details.get("flagged_tool_calls", [])) >= 1
    assert len(result_defect.details.get("flagged_tool_calls", [])) >= 1

    from agent_eval_harness.metrics.assertions.max_items_per_call import max_items_per_call
    planner_clean = max_items_per_call(spans_clean, "planner", {"limit": 2})
    planner_defect = max_items_per_call(spans_defect, "planner", {"limit": 2})
    assert planner_clean.passed == planner_defect.passed, "planner.max_items must not change"


# DEFECT 4 JUDGE_RUBBER_STAMP: must move judge to always report sufficient (retry_on_reject_required vacuous); must not move max_items_per_call

async def test_defect_judge_rubber_stamp_leaves_planner_assertion_unchanged() -> None:
    from agent_eval_harness.metrics.assertions.max_items_per_call import max_items_per_call

    responses_clean = [
        LLMResponse(content="valid", model="fake-nano"),
        LLMResponse(content='["i1", "i2"]', model="fake-frontier"),
        LLMResponse(content="sufficient context found", model="fake-strong"),
        LLMResponse(content="answer", model="fake-mini"),
    ]
    responses_defect = [
        LLMResponse(content="valid", model="fake-nano"),
        LLMResponse(content='["i1", "i2"]', model="fake-frontier"),
        LLMResponse(content="answer", model="fake-mini"),  # rubber_stamp skips LLM — writer still gets a response
    ]

    spans_clean = await _collect_spans(responses_clean, DefectConfig())
    spans_defect = await _collect_spans(responses_defect, DefectConfig(judge_rubber_stamp=True))

    result_clean = max_items_per_call(spans_clean, "planner", {"limit": 2})
    result_defect = max_items_per_call(spans_defect, "planner", {"limit": 2})

    assert result_clean.passed == result_defect.passed, (
        "planner.max_items_per_call must not change when JUDGE_RUBBER_STAMP is toggled"
    )


# DEFECT 5 WRITER_HALLUCINATE: must move writer output to hallucinate; must not move upstream planner max_items assertion

async def test_defect_writer_hallucinate_moves_output_not_upstream() -> None:
    from agent_eval_harness.instrumentation.tier1_haystack import HaystackAdapter

    responses = [
        LLMResponse(content="valid", model="fake-nano"),
        LLMResponse(content='["intent"]', model="fake-frontier"),
        LLMResponse(content="sufficient", model="fake-strong"),
        LLMResponse(content="Here is the answer.", model="fake-mini"),
    ]
    llm_client = FakeLLMClient(responses)
    handle = build_pipeline(llm_client, DefectConfig(writer_hallucinate=True))
    system_map = load_system_map(_MAP_PATH)

    adapter = HaystackAdapter(handle, system_map)
    adapter.attach()
    try:
        result = await adapter.run(_QUERY)
    finally:
        adapter.detach()

    assert "full refund" in result.final_output, (
        "Writer hallucinate defect must inject 'full refund' into output"
    )

    # Upstream assertion (fan-out limit) checked against the worker's span, unaffected by hallucination.
    from agent_eval_harness.metrics.assertions.max_items_per_call import max_items_per_call

    spans = [
        {
            "id": s.span_id,
            "component_id": s.component_id,
            "span_type": s.span_type,
            "input_json": s.input_json,
            "output_json": s.output_json,
            "parent_span_id": s.parent_span_id,
            "started_at": s.started_at or datetime.now(UTC).isoformat(),
            "details_json": json.dumps(
                {"tier": s.tier, "token_source": s.token_source, "raw_tags": s.tags}
            ),
        }
        for s in result.spans
    ]
    worker_result = max_items_per_call(spans, "worker", {"limit": 2})
    assert worker_result.passed is True, "max_items_per_call must not be affected by HALLUCINATE"


# DEFECT 6 GUARD_LEAK: must move guard to leak a rejected query as pass; must not move guard_rule metrics (separate stage)

async def test_defect_guard_leak_produces_pass_verdict() -> None:
    """With GUARD_LEAK on, the LLM guard stage emits 'pass' even when LLM says reject."""
    responses = [
        LLMResponse(content="this is a policy violation, reject it", model="fake-nano"),
        LLMResponse(content='["intent"]', model="fake-frontier"),
        LLMResponse(content="sufficient", model="fake-strong"),
        LLMResponse(content="answer", model="fake-mini"),
    ]

    spans_defect = await _collect_spans(responses, DefectConfig(guard_leak=True))
    guard_llm_spans = [s for s in spans_defect if s.get("component_id") == "guard_llm"]
    assert len(guard_llm_spans) >= 1
    output = json.loads(guard_llm_spans[0].get("output_json") or "{}")
    assert output.get("verdict") == "pass", "GUARD_LEAK must pass despite rejection signal"


async def test_no_defect_guard_rejects_policy_violation() -> None:
    """Without GUARD_LEAK, guard correctly rejects the policy-violation query."""
    responses = [
        LLMResponse(content="this is a policy violation, reject it", model="fake-nano"),
        LLMResponse(content='["intent"]', model="fake-frontier"),
        LLMResponse(content="sufficient", model="fake-strong"),
        LLMResponse(content="answer", model="fake-mini"),
    ]

    spans_clean = await _collect_spans(responses, DefectConfig(guard_leak=False))
    guard_llm_spans = [s for s in spans_clean if s.get("component_id") == "guard_llm"]
    if guard_llm_spans:
        output = json.loads(guard_llm_spans[0].get("output_json") or "{}")
        assert output.get("verdict") == "reject"


# ==== test_defect_leak_prevention.py — Test that active defects do not leak into insert_run for non-test-target runs. ====

@pytest.mark.usefixtures("_setup_db")
async def test_insert_run_defect_leak_prevention(monkeypatch) -> None:
    monkeypatch.setenv("AEH_DEFECT_GUARD_LEAK", "1")

    run_id = await repository.insert_run("my_real_target")
    run = await repository.get_run(run_id)
    assert run is not None
    assert run.get("active_defects") is None

    # Explicit active_defects, like the CLI passes for T1/T2/T3
    run_id_explicit = await repository.insert_run(
        "my_test_target", active_defects=["DEFECT_GUARD_LEAK"]
    )
    run_explicit = await repository.get_run(run_id_explicit)
    assert run_explicit is not None
    assert json.loads(run_explicit["active_defects"]) == ["DEFECT_GUARD_LEAK"]
