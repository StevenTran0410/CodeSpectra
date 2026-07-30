"""Tier-1 instrumentation: a custom, lightweight Haystack Tracer."""
from __future__ import annotations

import contextlib
import contextvars
import time
import uuid
from collections.abc import Iterator
from typing import Any

from haystack.tracing import Span as HaystackSpan
from haystack.tracing import Tracer as HaystackTracer
from haystack.tracing import disable_tracing, enable_tracing

from agent_eval_harness.instrumentation._extract import (
    extract_model_and_tokens,
    safe_json,
    utc_now_iso,
)
from agent_eval_harness.instrumentation.base import (
    CapturedSpan,
    InstrumentationAdapter,
    PipelineHandle,
    SpanType,
    TraceResult,
)
from agent_eval_harness.mapping.engine import map_spans_to_components
from agent_eval_harness.mapping.system_map import SystemMap

# Operation names AEH's own code uses for manual inner spans, distinct from Haystack's automatic component spans
OP_LLM_CALL = "aeh.llm_call"
OP_TOOL_CALL = "aeh.tool_call"
OP_VALIDATOR_DECISION = "aeh.validator_decision"

_MANUAL_OP_SPAN_TYPES: dict[str, SpanType] = {
    OP_LLM_CALL: "llm_call",
    OP_TOOL_CALL: "tool_call",
    OP_VALIDATOR_DECISION: "validator_decision",
}
_PIPELINE_ROOT_OPS = {"haystack.pipeline.run", "haystack.async_pipeline.run"}


class HarnessSpan(HaystackSpan):
    def __init__(self, span_id: str, parent_span_id: str | None, operation_name: str) -> None:
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self.operation_name = operation_name
        self.tags: dict[str, Any] = {}
        self.started_at = utc_now_iso()
        self.latency_ms: int | None = None
        self._start_monotonic = time.monotonic()

    def set_tag(self, key: str, value: Any) -> None:
        self.tags[key] = value


class HarnessTracer(HaystackTracer):
    """Captures every span flowing through haystack.tracing.tracer — both Haystack's own
    component spans and AEH's manual aeh.llm_call/tool_call/validator_decision spans."""

    def __init__(self) -> None:
        self.captured: list[HarnessSpan] = []
        self._current: contextvars.ContextVar[HarnessSpan | None] = contextvars.ContextVar(
            "aeh_current_span", default=None
        )

    @contextlib.contextmanager
    def trace(
        self,
        operation_name: str,
        tags: dict[str, Any] | None = None,
        parent_span: HaystackSpan | None = None,
    ) -> Iterator[HarnessSpan]:
        parent_id = parent_span.span_id if isinstance(parent_span, HarnessSpan) else None
        span = HarnessSpan(str(uuid.uuid4()), parent_id, operation_name)
        if tags:
            span.set_tags(tags)
        token = self._current.set(span)
        try:
            yield span
        finally:
            self._current.reset(token)
            span.latency_ms = int((time.monotonic() - span._start_monotonic) * 1000)
            self.captured.append(span)

    def current_span(self) -> HaystackSpan | None:
        return self._current.get()

    def reset(self) -> None:
        self.captured.clear()


def _classify_span_type(span: HarnessSpan) -> SpanType:
    if span.operation_name in _MANUAL_OP_SPAN_TYPES:
        return _MANUAL_OP_SPAN_TYPES[span.operation_name]
    if span.tags.get("haystack.component.type") == "HarnessChatGenerator":
        return "llm_call"
    return "agent"


def _normalize(span: HarnessSpan) -> CapturedSpan:
    """Manual spans (aeh.llm_call/tool_call/validator_decision) that need to record
    output/input via `aeh.output`/`aeh.input` tags instead of Haystack's own component tags."""
    span_type = _classify_span_type(span)

    haystack_output = span.tags.get("haystack.component.output")
    output_value = haystack_output if haystack_output is not None else span.tags.get("aeh.output")
    model, tokens_in, tokens_out, token_source = extract_model_and_tokens(output_value)

    input_value = span.tags.get("haystack.component.input", span.tags.get("aeh.input"))
    component_name = span.tags.get("haystack.component.name") or span.tags.get("component_name")

    return CapturedSpan(
        span_id=span.span_id,
        parent_span_id=span.parent_span_id,
        operation_name=span.operation_name,
        span_type=span_type,
        component_name=component_name,
        component_type=span.tags.get("haystack.component.type"),
        tags=dict(span.tags),
        input_json=safe_json(input_value),
        output_json=safe_json(output_value),
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        token_source=token_source,
        started_at=span.started_at,
        latency_ms=span.latency_ms,
        tier="tier1",
        component_id=None,
    )


def _extract_final_output(result: dict[str, dict[str, Any]], exit_node: str) -> str:
    """Duck-typed: exit_node's output may be Haystack-ChatMessage-shaped (a replies list)
    or a plain dict with an "answer" key."""
    node_output = result.get(exit_node, {})
    replies = node_output.get("replies")
    if isinstance(replies, list) and replies:
        text = getattr(replies[0], "text", None)
        if text is not None:
            return text
    answer = node_output.get("answer")
    if isinstance(answer, str):
        return answer
    return ""


class HaystackAdapter(InstrumentationAdapter):
    """Tier-1: attaches a HarnessTracer to Haystack's global tracing hook, runs the
    pipeline, and normalizes captured spans into a TraceResult."""

    def __init__(self, handle: PipelineHandle, system_map: SystemMap) -> None:
        self._handle = handle
        self._map = system_map
        self._tracer = HarnessTracer()

    def attach(self) -> None:
        from haystack.tracing import tracer as haystack_global_tracer

        if not haystack_global_tracer.is_content_tracing_enabled:
            raise RuntimeError(
                "HAYSTACK_CONTENT_TRACING_ENABLED must be 'true' before any `haystack` import "
                "happens in this process (checked once at haystack.tracing's first import) — "
                "otherwise spans silently lose their input/output content. Set it as the literal "
                "first line of your entry point, before importing agent_eval_harness."
            )
        enable_tracing(self._tracer)

    def detach(self) -> None:
        disable_tracing()

    async def run(self, query: str) -> TraceResult:
        started = time.monotonic()
        data = {node: {key: query} for node, key in self._handle.entry_inputs.items()}
        result = await self._handle.pipeline.run_async(data=data)
        total_latency_ms = int((time.monotonic() - started) * 1000)

        raw_spans = [s for s in self._tracer.captured if s.operation_name not in _PIPELINE_ROOT_OPS]
        captured = [_normalize(s) for s in raw_spans]

        # Pipeline-root spans were dropped above; null out any parent ref that pointed at one so no span is left with a dangling parent
        captured_ids = {c.span_id for c in captured}
        for c in captured:
            if c.parent_span_id is not None and c.parent_span_id not in captured_ids:
                c.parent_span_id = None

        mapping_result = map_spans_to_components(captured, self._map)

        final_output = _extract_final_output(result, self._handle.exit_node)
        total_tokens = sum((c.tokens_in or 0) + (c.tokens_out or 0) for c in captured)

        self._tracer.reset()
        return TraceResult(
            root_input=query,
            final_output=final_output,
            spans=captured,
            total_tokens=total_tokens,
            total_latency_ms=total_latency_ms,
            unmatched=mapping_result.unmatched,
            ambiguous=mapping_result.ambiguous,
        )
