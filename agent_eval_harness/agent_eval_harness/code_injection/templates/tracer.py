"""AEH Stage 4 span logger: implements Haystack's own tracing.Tracer protocol (matching AEH's in-process HarnessTracer) so spans carry real component identity rather than tag-matching, and appends JSON lines to out/eval_log.{pid}.jsonl without a lock since asyncio only interleaves at await points."""
from __future__ import annotations

import contextlib
import contextvars
import json
import os
import time
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from haystack.tracing import Span as HaystackSpan
from haystack.tracing import Tracer as HaystackTracer

_OUT_DIR = Path(__file__).resolve().parent / "out"
_LOG_PATH = _OUT_DIR / f"eval_log.{os.getpid()}.jsonl"
_PIPELINE_ROOT_OPS = {"haystack.pipeline.run", "haystack.async_pipeline.run"}

_current_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "aeh_current_trace_id", default=None
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def write_log_line(record: dict[str, Any]) -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    with _LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str))
        f.write("\n")


def log_path() -> Path:
    return _LOG_PATH


@contextlib.contextmanager
def set_current_trace(trace_id: str) -> Iterator[None]:
    """Wraps a dataset case's pipeline invocation so spans emitted while active are stamped with this trace_id, giving each case its own root trace."""
    token = _current_trace_id.set(trace_id)
    try:
        yield
    finally:
        _current_trace_id.reset(token)


class AehSpan(HaystackSpan):
    def __init__(self, span_id: str, parent_span_id: str | None, operation_name: str) -> None:
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self.operation_name = operation_name
        self.tags: dict[str, Any] = {}
        self.started_at = _utc_now_iso()
        self.latency_ms: int | None = None
        self._start_monotonic = time.monotonic()

    def set_tag(self, key: str, value: Any) -> None:
        self.tags[key] = value


def _classify_span_type(operation_name: str, tags: dict[str, Any]) -> str:
    if tags.get("haystack.component.type"):
        return "agent"
    return "other"


def _extract_tokens(output: Any) -> tuple[str | None, int | None, int | None, str | None]:
    """Best-effort: looks for usage under common key names in the agent's dict output; anything not found is left None (never guessed), and token_source is "measured" only when a real usage block was found."""
    if not isinstance(output, dict):
        return None, None, None, None
    usage = output.get("usage") or output.get("token_usage")
    if isinstance(usage, dict):
        tokens_in = usage.get("prompt_tokens") or usage.get("input_tokens")
        tokens_out = usage.get("completion_tokens") or usage.get("output_tokens")
        model = usage.get("model") or output.get("model")
        if tokens_in is not None or tokens_out is not None:
            return model, tokens_in, tokens_out, "measured"
    return output.get("model"), None, None, None


def _safe_json(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return json.dumps(str(value))


class AehTracer(HaystackTracer):
    def __init__(self) -> None:
        self._current: contextvars.ContextVar[AehSpan | None] = contextvars.ContextVar(
            "aeh_current_span", default=None
        )

    @contextlib.contextmanager
    def trace(
        self,
        operation_name: str,
        tags: dict[str, Any] | None = None,
        parent_span: HaystackSpan | None = None,
    ) -> Iterator[AehSpan]:
        parent = parent_span if isinstance(parent_span, AehSpan) else self._current.get()
        # Pipeline-root spans are never written, so their direct children are top-level (never a dangling parent_span_id).
        has_written_parent = parent is not None and parent.operation_name not in _PIPELINE_ROOT_OPS
        parent_id = parent.span_id if has_written_parent else None
        span = AehSpan(str(uuid.uuid4()), parent_id, operation_name)
        if tags:
            span.set_tags(tags)
        token = self._current.set(span)
        error: BaseException | None = None
        try:
            yield span
        except BaseException as exc:  # noqa: BLE001 — re-raised below, only recorded here
            error = exc
            raise
        finally:
            self._current.reset(token)
            span.latency_ms = int((time.monotonic() - span._start_monotonic) * 1000)
            if operation_name not in _PIPELINE_ROOT_OPS:
                self._write_span(span, error)

    def current_span(self) -> HaystackSpan | None:
        return self._current.get()

    def _write_span(self, span: AehSpan, error: BaseException | None) -> None:
        output = span.tags.get("haystack.component.output")
        input_value = span.tags.get("haystack.component.input")
        model, tokens_in, tokens_out, token_source = _extract_tokens(output)
        write_log_line({
            "record": "span",
            "trace_id": _current_trace_id.get(),
            "span_id": span.span_id,
            "parent_span_id": span.parent_span_id,
            "component_id": span.tags.get("haystack.component.name"),
            "span_type": _classify_span_type(span.operation_name, span.tags),
            "operation": span.operation_name,
            "started_at": span.started_at,
            "latency_ms": span.latency_ms,
            "input_json": _safe_json(input_value),
            "output_json": _safe_json(output),
            "model": model,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "token_source": token_source,
            "error": str(error) if error is not None else None,
        })


def register_tracer() -> None:
    """Call once, before any other `haystack` import in the process (haystack.tracing checks HAYSTACK_CONTENT_TRACING_ENABLED only at its own first import)."""
    from haystack.tracing import tracer as haystack_global_tracer
    from haystack.tracing import enable_tracing

    if not haystack_global_tracer.is_content_tracing_enabled:
        raise RuntimeError(
            "HAYSTACK_CONTENT_TRACING_ENABLED must be 'true' before any `haystack` import in "
            "this process. Set it as the literal first statement of the entry point."
        )
    enable_tracing(AehTracer())
