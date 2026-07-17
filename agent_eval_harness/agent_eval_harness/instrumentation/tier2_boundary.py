"""Tier-2 boundary-wrapper adapter — fallback for targets with no native tracing."""
from __future__ import annotations

import importlib
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from agent_eval_harness.instrumentation._extract import (
    extract_model_and_tokens,
    safe_json,
    utc_now_iso,
)
from agent_eval_harness.instrumentation.base import (
    CapturedSpan,
    InstrumentationAdapter,
    TraceResult,
)
from agent_eval_harness.mapping.system_map import SystemMap

_HTTP_TIMEOUT_SECONDS = 60.0


def _resolve_entry_point(entry_point: str) -> Callable[..., Awaitable[Any]]:
    """entry_point is "module.path:function_name" (the Python-callable form)
    or HTTP entry points (starts with http:// or https://)."""
    if entry_point.startswith(("http://", "https://")):
        import httpx

        async def _http_callable(query: str, prior_output: Any = None) -> Any:
            async with httpx.AsyncClient() as client:
                payload = {"query": query}
                if prior_output is not None:
                    payload["prior_output"] = prior_output
                resp = await client.post(entry_point, json=payload, timeout=_HTTP_TIMEOUT_SECONDS)
                resp.raise_for_status()
                return resp.json()

        return _http_callable

    module_path, _, func_name = entry_point.partition(":")
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


def _extract_text(output: Any) -> str:
    if isinstance(output, dict):
        for key in ("answer", "text", "content"):
            value = output.get(key)
            if isinstance(value, str):
                return value
    return ""


class BoundaryWrapperAdapter(InstrumentationAdapter):
    def __init__(self, system_map: SystemMap, component_order: list[str]) -> None:
        self._map = system_map
        self._component_order = component_order
        self._entry_points: dict[str, Callable[..., Awaitable[Any]]] = {}

    def attach(self) -> None:
        for component_id in self._component_order:
            component = self._map.component_by_id(component_id)
            if component is None:
                raise ValueError(f"component '{component_id}' not found in system map")
            self._entry_points[component_id] = _resolve_entry_point(component.entry_point)

    def detach(self) -> None:
        self._entry_points.clear()

    async def run(self, query: str) -> TraceResult:
        spans: list[CapturedSpan] = []
        prior_output: Any = None
        final_text = ""
        total_latency_ms = 0
        total_tokens = 0

        for component_id in self._component_order:
            fn = self._entry_points[component_id]
            started_at = utc_now_iso()
            started = time.monotonic()

            from agent_eval_harness.llm.routing_client import current_component_id_var
            token = current_component_id_var.set(component_id)
            try:
                output = await (fn(query) if prior_output is None else fn(query, prior_output))
            finally:
                current_component_id_var.reset(token)

            latency_ms = int((time.monotonic() - started) * 1000)
            total_latency_ms += latency_ms

            model, tokens_in, tokens_out, token_source = extract_model_and_tokens(output)
            total_tokens += (tokens_in or 0) + (tokens_out or 0)

            spans.append(
                CapturedSpan(
                    span_id=str(uuid.uuid4()),
                    parent_span_id=None,
                    operation_name=f"aeh.tier2.{component_id}",
                    span_type="agent",
                    component_name=component_id,
                    component_type=None,
                    tags={},
                    input_json=safe_json(query if prior_output is None else prior_output),
                    output_json=safe_json(output),
                    model=model,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    token_source=token_source,
                    started_at=started_at,
                    latency_ms=latency_ms,
                    tier="tier2",
                    component_id=component_id,  # pre-assigned — bypasses the mapping engine
                )
            )
            prior_output = output
            final_text = _extract_text(output)

        return TraceResult(
            root_input=query,
            final_output=final_text,
            spans=spans,
            total_tokens=total_tokens,
            total_latency_ms=total_latency_ms,
            unmatched=[],
            ambiguous=[],
        )
