"""Manual inner-span helper for test-target component code.

Mirrors the exact pattern haystack.components.agents.agent.py uses internally
(tracer.trace(op, tags, parent_span=tracer.current_span())) — this is how ANY
Haystack component achieves nested spans, built-in or ours.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from haystack import tracing


@contextmanager
def manual_span(
    operation_name: str, component_name: str, extra_tags: dict[str, Any] | None = None
) -> Iterator[Any]:
    tags: dict[str, Any] = {"component_name": component_name}
    if extra_tags:
        tags.update(extra_tags)
    with tracing.tracer.trace(
        operation_name, tags=tags, parent_span=tracing.tracer.current_span()
    ) as span:
        yield span
