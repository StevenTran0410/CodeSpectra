"""Core metric result type used by all metric classes (CS-263)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

MetricClass = Literal["assertion", "classifier", "llm_judge"]


@dataclass
class MetricResult:
    metric_name: str
    metric_class: MetricClass
    score: float | None
    passed: bool | None
    details: dict[str, Any] = field(default_factory=dict)
    component_id: str | None = None
    span_id: str | None = None
    trace_id: str | None = None
    evaluator: str | None = None
    cost_tokens: int | None = None


def extract_cost_tokens(adapter: Any) -> int | None:
    """Extract total cost tokens from an LLM adapter with token tracking."""
    if not (hasattr(adapter, "_last_prompt_tokens") and adapter._last_prompt_tokens is not None):
        return None
    pt = adapter._last_prompt_tokens or 0
    ct = adapter._last_completion_tokens or 0
    return pt + ct
