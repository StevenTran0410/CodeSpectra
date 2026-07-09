"""Defect Gauntlet Driven by Generated Plan (CS-265 §5)."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent_eval_harness.llm.client import LLMResponse
from agent_eval_harness.llm.fake_client import FakeLLMClient
from agent_eval_harness.mapping.system_map import load_system_map
from agent_eval_harness.planning.planner import generate_plan
from agent_eval_harness.planning.validation import validate_plan
from test_targets._shared.defects import DefectConfig
from test_targets.multi_agent.pipeline import build_pipeline

pytestmark = pytest.mark.asyncio

_MAP_PATH = (
    Path(__file__).parent.parent
    / "test_targets"
    / "multi_agent"
    / "system_map.yaml"
)
_QUERY = "Can I get a refund and also change my shipping address?"


async def _collect_spans(responses: list[LLMResponse], defects: DefectConfig) -> list[dict]:
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
            "details_json": json.dumps(
                {"tier": s.tier, "token_source": s.token_source, "raw_tags": s.tags}
            ),
        }
        for s in result.spans
    ]


async def test_generated_plan_passes_validation(tmp_path) -> None:
    """Generate plan, write it out, waive classifier dataset requirements, and validate."""
    llm_client = FakeLLMClient(LLMResponse(content="Rubric text", model="fake"))
    plan = await generate_plan(_MAP_PATH, llm_client)

    # Approve/waive all needs_human and dataset requirements for validation
    for entry in plan.entries:
        if entry.status == "needs_human":
            entry.status = None
        if entry.dataset:
            entry.dataset.waived = "Waived in automated test"
            entry.dataset.ref = None
            entry.dataset.required = None

    import yaml
    plan_path = tmp_path / "t2_generated_plan.yaml"
    plan_path.write_text(yaml.dump(plan.model_dump(exclude_none=True)), encoding="utf-8")

    report = await validate_plan(plan_path)
    assert not report.errors, f"Validation errors on generated plan: {report.errors}"


async def test_defect_planner_overpack_via_generated_plan() -> None:
    llm_client = FakeLLMClient(LLMResponse(content="Rubric text", model="fake"))
    plan = await generate_plan(_MAP_PATH, llm_client)

    # Retrieve max_items_per_call entry from the generated plan
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

    # Retrieve retry_on_reject_required entry from the generated plan
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
