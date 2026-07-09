"""Defect Gauntlet — Phase 0 exit gate."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent_eval_harness.llm.client import LLMResponse
from agent_eval_harness.llm.fake_client import FakeLLMClient
from agent_eval_harness.mapping.system_map import load_system_map
from test_targets._shared.defects import DefectConfig
from test_targets.multi_agent.pipeline import build_pipeline

pytestmark = pytest.mark.asyncio

_MAP_PATH = Path(__file__).parent.parent / "test_targets" / "multi_agent" / "system_map.yaml"
_QUERY = "Can I get a refund and also change my shipping address?"


# ────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ────────────────────────────────────────────────────────────────────────────

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

    # Convert CapturedSpan to plain dicts matching the DB row format
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


# ────────────────────────────────────────────────────────────────────────────
# DEFECT 1: PLANNER_OVERPACK
# Must move: assertion.max_items_per_call → FAIL
# Must NOT move: guard/judge/writer metrics
# ────────────────────────────────────────────────────────────────────────────

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

    # The assertion checks the WORKER component's output_json.intents.
    result_clean = max_items_per_call(spans_clean, "worker", {"limit": 2})
    result_defect = max_items_per_call(spans_defect, "worker", {"limit": 2})

    assert result_clean.passed is True, "Clean run must pass max_items_per_call"
    assert result_defect.passed is False, "Defect run must fail max_items_per_call"


# ────────────────────────────────────────────────────────────────────────────
# DEFECT 2: NO_RETRY
# Must move: assertion.retry_on_reject_required → FAIL
# Must NOT move: judge's classifier score
# ────────────────────────────────────────────────────────────────────────────

async def test_defect_no_retry_moves_retry_assertion() -> None:
    from agent_eval_harness.metrics.assertions.retry_on_reject_required import (
        retry_on_reject_required,
    )

    # Clean path: judge rejects on first call, passes on second call (after retry)
    responses_clean = [
        LLMResponse(content="valid", model="fake-nano"),
        LLMResponse(content='["intent"]', model="fake-frontier"),
        # first judge -> reject
        LLMResponse(content="insufficient context, need more", model="fake-strong"),
        # second judge -> pass
        LLMResponse(content="sufficient context found", model="fake-strong"),
        LLMResponse(content="answer", model="fake-mini"),
    ]
    # Defect path: judge rejects, but no retry happens (planner skips second worker call)
    responses_defect = [
        LLMResponse(content="valid", model="fake-nano"),
        LLMResponse(content='["intent"]', model="fake-frontier"),
        # judge -> reject
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
    # With NO_RETRY, there's only 1 worker span (= within limit)
    spans_defect = await _collect_spans(responses, DefectConfig(no_retry=True))
    result = max_retries(spans_defect, "worker", {"limit": 1})
    assert result.passed is True, "max_retries must not move when NO_RETRY defect is on"


# ────────────────────────────────────────────────────────────────────────────
# DEFECT 3: WRONG_TOOL
# Must move: first tool_call span's tool name (case_law_search -> decoy_lookup)
# Must NOT move: guard/planner metrics; no_unnecessary_calls itself (see note below)
# ────────────────────────────────────────────────────────────────────────────

async def test_defect_wrong_tool_flags_no_unnecessary_calls() -> None:
    """T2's WorkerComponent always calls both tools unconditionally; WRONG_TOOL only
    changes which tool is called first."""
    from agent_eval_harness.metrics.assertions.no_unnecessary_calls import no_unnecessary_calls

    responses = [
        LLMResponse(content="valid", model="fake-nano"),
        LLMResponse(content='["intent"]', model="fake-frontier"),
        LLMResponse(content="sufficient", model="fake-strong"),
        LLMResponse(content="answer", model="fake-mini"),
    ]

    spans_clean = await _collect_spans(responses, DefectConfig(wrong_tool=False))
    spans_defect = await _collect_spans(responses, DefectConfig(wrong_tool=True))

    # Metric-level differentiator: first tool_call span's tool name.
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

    # And that planner's max_items assertion is unchanged
    from agent_eval_harness.metrics.assertions.max_items_per_call import max_items_per_call
    planner_clean = max_items_per_call(spans_clean, "planner", {"limit": 2})
    planner_defect = max_items_per_call(spans_defect, "planner", {"limit": 2})
    assert planner_clean.passed == planner_defect.passed, "planner.max_items must not change"


# ────────────────────────────────────────────────────────────────────────────
# DEFECT 4: JUDGE_RUBBER_STAMP
# Must move: judge always says sufficient=True → retry_on_reject_required vacuous
# Must NOT move: assertion.max_items_per_call
# ────────────────────────────────────────────────────────────────────────────

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
        # rubber_stamp skips LLM — writer still gets a response
        LLMResponse(content="answer", model="fake-mini"),
    ]

    spans_clean = await _collect_spans(responses_clean, DefectConfig())
    spans_defect = await _collect_spans(responses_defect, DefectConfig(judge_rubber_stamp=True))

    result_clean = max_items_per_call(spans_clean, "planner", {"limit": 2})
    result_defect = max_items_per_call(spans_defect, "planner", {"limit": 2})

    assert result_clean.passed == result_defect.passed, (
        "planner.max_items_per_call must not change when JUDGE_RUBBER_STAMP is toggled"
    )


# ────────────────────────────────────────────────────────────────────────────
# DEFECT 5: WRITER_HALLUCINATE
# Must move: writer produces hallucinated output (visible in final_output)
# Must NOT move: upstream assertion pass/fail (planner max_items)
# ────────────────────────────────────────────────────────────────────────────

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

    # DEFECT_WRITER_HALLUCINATE injects fabricated claims
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


# ────────────────────────────────────────────────────────────────────────────
# DEFECT 6: GUARD_LEAK
# Must move: guard leaks a rejected query through as pass
# Must NOT move: guard_rule metrics (rule stage is separate)
# ────────────────────────────────────────────────────────────────────────────

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
