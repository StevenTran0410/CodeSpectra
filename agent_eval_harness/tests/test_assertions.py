"""Tests for all 6 assertion types — synthetic span fixtures, no DB, no pipeline runs.

Each assertion type gets both a PASS and FAIL case (CS-263 §11).
"""
from __future__ import annotations

import json
from datetime import UTC, datetime


def _span(
    *,
    span_id: str,
    component_id: str | None = None,
    span_type: str = "llm",
    input_json: str | None = None,
    output_json: str | None = None,
    parent_span_id: str | None = None,
    started_at: str | None = None,
    details_json: str = "{}",
) -> dict:
    return {
        "id": span_id,
        "component_id": component_id,
        "span_type": span_type,
        "input_json": input_json,
        "output_json": output_json,
        "parent_span_id": parent_span_id,
        "started_at": started_at or datetime.now(UTC).isoformat(),
        "details_json": details_json,
    }


# ────────────────────────────────────────────────────────────────────────────
# max_items_per_call
# ────────────────────────────────────────────────────────────────────────────

def test_max_items_per_call_pass() -> None:
    from agent_eval_harness.metrics.assertions.max_items_per_call import max_items_per_call

    spans = [
        _span(
            span_id="s1",
            component_id="planner",
            output_json=json.dumps({"intents": ["a", "b"]}),
        )
    ]
    result = max_items_per_call(spans, "planner", {"limit": 2})
    assert result.passed is True
    assert result.details["limit"] == 2
    assert result.details["offending_span_ids"] == []


def test_max_items_per_call_fail() -> None:
    from agent_eval_harness.metrics.assertions.max_items_per_call import max_items_per_call

    spans = [
        _span(
            span_id="s1",
            component_id="planner",
            output_json=json.dumps({"intents": ["a", "b", "c", "d"]}),
        )
    ]
    result = max_items_per_call(spans, "planner", {"limit": 2})
    assert result.passed is False
    assert "s1" in result.details["offending_span_ids"]


# ────────────────────────────────────────────────────────────────────────────
# max_retries
# ────────────────────────────────────────────────────────────────────────────

def test_max_retries_pass() -> None:
    from agent_eval_harness.metrics.assertions.max_retries import max_retries

    spans = [
        _span(span_id="s1", component_id="worker"),
        _span(span_id="s2", component_id="worker"),  # 1 retry = allowed
    ]
    result = max_retries(spans, "worker", {"limit": 1})
    assert result.passed is True
    assert result.details["max_allowed"] == 2


def test_max_retries_fail() -> None:
    from agent_eval_harness.metrics.assertions.max_retries import max_retries

    spans = [
        _span(span_id="s1", component_id="worker"),
        _span(span_id="s2", component_id="worker"),
        _span(span_id="s3", component_id="worker"),  # 2 retries = over limit
    ]
    result = max_retries(spans, "worker", {"limit": 1})
    assert result.passed is False
    assert result.details["span_count"] == 3


# ────────────────────────────────────────────────────────────────────────────
# allowed_downstream
# ────────────────────────────────────────────────────────────────────────────

def test_allowed_downstream_pass() -> None:
    from agent_eval_harness.metrics.assertions.allowed_downstream import allowed_downstream

    spans = [
        _span(span_id="parent", component_id="planner"),
        _span(span_id="child1", component_id="worker", parent_span_id="parent"),
    ]
    result = allowed_downstream(spans, "planner", {"allowed": ["worker", "judge"]})
    assert result.passed is True


def test_allowed_downstream_fail() -> None:
    from agent_eval_harness.metrics.assertions.allowed_downstream import allowed_downstream

    spans = [
        _span(span_id="parent", component_id="planner"),
        _span(span_id="child1", component_id="bad_component", parent_span_id="parent"),
    ]
    result = allowed_downstream(spans, "planner", {"allowed": ["worker"]})
    assert result.passed is False
    assert any(v["child_component_id"] == "bad_component" for v in result.details["violations"])


def test_allowed_downstream_skips_unmatched() -> None:
    """Spans with component_id=None (unmatched) must not trigger violations."""
    from agent_eval_harness.metrics.assertions.allowed_downstream import allowed_downstream

    spans = [
        _span(span_id="parent", component_id="planner"),
        _span(span_id="child_unmatched", component_id=None, parent_span_id="parent"),
    ]
    result = allowed_downstream(spans, "planner", {"allowed": ["worker"]})
    assert result.passed is True


# ────────────────────────────────────────────────────────────────────────────
# retry_on_reject_required
# ────────────────────────────────────────────────────────────────────────────

def test_retry_on_reject_required_pass() -> None:
    """Judge rejects, then a second worker span follows → pass."""
    from agent_eval_harness.metrics.assertions.retry_on_reject_required import (
        retry_on_reject_required,
    )

    spans = [
        _span(span_id="w1", component_id="worker", started_at="2024-01-01T10:00:00"),
        _span(
            span_id="j1",
            component_id="judge",
            output_json=json.dumps({"sufficient": False}),
            started_at="2024-01-01T10:01:00",
        ),
        _span(span_id="w2", component_id="worker", started_at="2024-01-01T10:02:00"),
        _span(span_id="wr", component_id="writer", started_at="2024-01-01T10:03:00"),
    ]
    result = retry_on_reject_required(spans, "worker", {})
    assert result.passed is True


def test_retry_on_reject_required_fail_no_retry() -> None:
    """Judge rejects, NO second worker span → fail (DEFECT_NO_RETRY)."""
    from agent_eval_harness.metrics.assertions.retry_on_reject_required import (
        retry_on_reject_required,
    )

    spans = [
        _span(span_id="w1", component_id="worker", started_at="2024-01-01T10:00:00"),
        _span(
            span_id="j1",
            component_id="judge",
            output_json=json.dumps({"sufficient": False}),
            started_at="2024-01-01T10:01:00",
        ),
        _span(span_id="wr", component_id="writer", started_at="2024-01-01T10:02:00"),
    ]
    result = retry_on_reject_required(spans, "worker", {})
    assert result.passed is False
    assert "j1" in result.details["missing_retries_after_rejection_span_ids"]


def test_retry_on_reject_required_vacuous_pass_no_rejections() -> None:
    """No rejections at all → vacuously true."""
    from agent_eval_harness.metrics.assertions.retry_on_reject_required import (
        retry_on_reject_required,
    )

    spans = [
        _span(span_id="w1", component_id="worker", started_at="2024-01-01T10:00:00"),
        _span(
            span_id="j1",
            component_id="judge",
            output_json=json.dumps({"sufficient": True}),
            started_at="2024-01-01T10:01:00",
        ),
        _span(span_id="wr", component_id="writer", started_at="2024-01-01T10:02:00"),
    ]
    result = retry_on_reject_required(spans, "worker", {})
    assert result.passed is True
    assert result.details["rejections"] == 0


# ────────────────────────────────────────────────────────────────────────────
# arg_schema
# ────────────────────────────────────────────────────────────────────────────

def test_arg_schema_pass() -> None:
    from agent_eval_harness.metrics.assertions.arg_schema import arg_schema

    schema = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    spans = [
        _span(
            span_id="t1",
            component_id="worker",
            span_type="tool_call",
            output_json=json.dumps({"query": "test"}),
        )
    ]
    result = arg_schema(spans, "worker", {"schema": schema})
    assert result.passed is True


def test_arg_schema_fail() -> None:
    from agent_eval_harness.metrics.assertions.arg_schema import arg_schema

    schema = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    spans = [
        _span(
            span_id="t1",
            component_id="worker",
            span_type="tool_call",
            output_json=json.dumps({"wrong_field": 123}),
        )
    ]
    result = arg_schema(spans, "worker", {"schema": schema})
    assert result.passed is False
    assert any(v["span_id"] == "t1" for v in result.details["violations"])


def test_arg_schema_only_checks_tool_call_spans() -> None:
    """Non-tool_call spans must be ignored."""
    from agent_eval_harness.metrics.assertions.arg_schema import arg_schema

    schema = {"type": "object", "required": ["query"]}
    spans = [
        _span(
            span_id="s1",
            component_id="worker",
            span_type="llm",  # not tool_call
            output_json=json.dumps({"wrong": 1}),
        )
    ]
    result = arg_schema(spans, "worker", {"schema": schema})
    assert result.passed is True


# ────────────────────────────────────────────────────────────────────────────
# no_unnecessary_calls
# ────────────────────────────────────────────────────────────────────────────

def test_no_unnecessary_calls_pass() -> None:
    from agent_eval_harness.metrics.assertions.no_unnecessary_calls import no_unnecessary_calls

    tool_result = "Case law result: employee handbook section 4.2"
    spans = [
        _span(
            span_id="tool1",
            component_id="worker",
            span_type="tool_call",
            output_json=json.dumps({"result": tool_result}),
            started_at="2024-01-01T10:00:00",
        ),
        _span(
            span_id="later",
            component_id="writer",
            input_json=json.dumps({"context": tool_result}),
            started_at="2024-01-01T10:01:00",
        ),
    ]
    result = no_unnecessary_calls(spans, "worker", {})
    assert result.passed is True


def test_no_unnecessary_calls_fail_unused_result() -> None:
    """Tool result not referenced anywhere later → flagged."""
    from agent_eval_harness.metrics.assertions.no_unnecessary_calls import no_unnecessary_calls

    spans = [
        _span(
            span_id="tool1",
            component_id="worker",
            span_type="tool_call",
            output_json=json.dumps({"result": "decoy result nobody uses"}),
            started_at="2024-01-01T10:00:00",
        ),
        _span(
            span_id="later",
            component_id="writer",
            input_json=json.dumps({"context": "totally different content"}),
            started_at="2024-01-01T10:01:00",
        ),
    ]
    result = no_unnecessary_calls(spans, "worker", {})
    assert result.passed is False
    assert any(f["span_id"] == "tool1" for f in result.details["flagged_tool_calls"])
