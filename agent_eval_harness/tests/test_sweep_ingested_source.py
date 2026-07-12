"""RunSource='ingested' seam — scores an already-ingested Stage 4 run's persisted spans
directly, instead of run_sweep's normal live in-process execute_run path."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from agent_eval_harness.ingest.spanlog_ingest import parse_spanlog, persist_spanlog
from agent_eval_harness.mapping.system_map import Component, SystemMap, save_system_map
from agent_eval_harness.metrics.sweep import run_sweep
from agent_eval_harness.store import repository
from agent_eval_harness.store.database import close_db, init_db

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
async def _setup_db(tmp_path, monkeypatch):
    monkeypatch.setenv("AEH_DATA_DIR", str(tmp_path))
    await init_db()
    yield
    await close_db()


def _write_map(tmp_path: Path) -> str:
    system_map = SystemMap(
        target_system_id="codespectra",
        components=[Component(id="project_identity", role="writer", entry_point="a:b")],
    )
    map_path = tmp_path / "map.yaml"
    save_system_map(system_map, map_path)
    return str(map_path)


def _write_suite(tmp_path: Path, schema: dict) -> str:
    suite_path = tmp_path / "plan.yaml"
    suite_path.write_text(
        "entries:\n" + textwrap.dedent(f"""\
              - id: project_identity.schema_valid
                component: project_identity
                agent_id: project_identity
                metric: schema_valid
                metric_class: assertion
                params:
                  schema: {json.dumps(schema)}
                rationale: r
                provenance: rule
              - id: project_identity.some_judge
                component: project_identity
                agent_id: project_identity
                metric: geval.quality
                metric_class: llm_judge
                rationale: r
                provenance: rule
        """),
        encoding="utf-8",
    )
    return str(suite_path)


async def _persist_run_with_output(tmp_path: Path, output: dict) -> str:
    log_path = tmp_path / "eval_log.1.jsonl"
    records = [
        {"record": "header", "schema": "aeh.spanlog/1", "tracer_version": "1", "plan_id": "p1", "run_id": "r1"},
        {"record": "case_start", "trace_id": "t1", "dataset_case_id": "case-1", "input": {}},
        {
            "record": "span", "trace_id": "t1", "span_id": "sp1", "parent_span_id": None,
            "component_id": "project_identity", "span_type": "agent", "operation": "haystack.component.run",
            "started_at": "2026-01-01T00:00:00.000Z", "latency_ms": 100,
            "input_json": "{}", "output_json": json.dumps(output),
        },
        {"record": "case_end", "trace_id": "t1", "status": "ok", "final_output_json": json.dumps(output)},
        {"record": "run_summary", "attempted": 1, "succeeded": 1},
    ]
    log_path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    parsed = parse_spanlog(log_path)
    return await persist_spanlog(parsed, target_system_id="codespectra", eval_plan_id="p1")


async def test_ingested_source_scores_assertion_against_persisted_spans(tmp_path: Path) -> None:
    schema = {"type": "object", "properties": {"domain": {"type": "string"}}, "required": ["domain"]}
    map_path = _write_map(tmp_path)
    suite_path = _write_suite(tmp_path, schema)
    run_id = await _persist_run_with_output(tmp_path, {"domain": "web app"})

    result = await run_sweep(
        target="codespectra", map_path=map_path, suite_path=suite_path,
        llm_client=None, run_id=run_id, source="ingested",
    )

    schema_results = [r for r in result.results if r.metric_name == "assertion.schema_valid"]
    assert len(schema_results) == 1
    assert schema_results[0].passed is True
    assert schema_results[0].trace_id is not None


async def test_ingested_source_flags_schema_violation(tmp_path: Path) -> None:
    schema = {"type": "object", "properties": {"domain": {"type": "string"}}, "required": ["domain"]}
    map_path = _write_map(tmp_path)
    suite_path = _write_suite(tmp_path, schema)
    run_id = await _persist_run_with_output(tmp_path, {"wrong_field": "oops"})

    result = await run_sweep(
        target="codespectra", map_path=map_path, suite_path=suite_path,
        llm_client=None, run_id=run_id, source="ingested",
    )

    schema_results = [r for r in result.results if r.metric_name == "assertion.schema_valid"]
    assert schema_results[0].passed is False


async def test_ingested_source_judge_gate_returns_not_supported_placeholder(tmp_path: Path) -> None:
    schema = {"type": "object"}
    map_path = _write_map(tmp_path)
    suite_path = _write_suite(tmp_path, schema)
    run_id = await _persist_run_with_output(tmp_path, {"domain": "web"})

    result = await run_sweep(
        target="codespectra", map_path=map_path, suite_path=suite_path,
        llm_client=None, run_id=run_id, source="ingested",
    )

    judge_results = [r for r in result.results if r.metric_class == "llm_judge"]
    assert len(judge_results) == 1
    assert judge_results[0].passed is None
    assert "not supported" in judge_results[0].details["reason"]


async def test_ingested_source_does_not_overwrite_run_status_set_at_ingest(tmp_path: Path) -> None:
    """The run's status reflects whether the INJECTED EXECUTION succeeded (set at ingest
    time) — a scoring pass on top of it must never overwrite that, even on success."""
    schema = {"type": "object"}
    map_path = _write_map(tmp_path)
    suite_path = _write_suite(tmp_path, schema)
    run_id = await _persist_run_with_output(tmp_path, {"domain": "web"})
    run_before = await repository.get_run(run_id)

    await run_sweep(
        target="codespectra", map_path=map_path, suite_path=suite_path,
        llm_client=None, run_id=run_id, source="ingested",
    )

    run_after = await repository.get_run(run_id)
    assert run_after["status"] == run_before["status"] == "completed"


async def test_ingested_source_requires_an_existing_run_id(tmp_path: Path) -> None:
    map_path = _write_map(tmp_path)
    suite_path = _write_suite(tmp_path, {"type": "object"})

    with pytest.raises(ValueError, match="source='ingested' requires an existing run_id"):
        await run_sweep(
            target="codespectra", map_path=map_path, suite_path=suite_path,
            llm_client=None, source="ingested",
        )
