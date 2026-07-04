"""InstrumentationAdapter interface + captured-span data shapes.

Both tiers (Haystack-native, boundary-wrapper) produce the SAME `CapturedSpan`/
`TraceResult` shapes so the mapping engine, store, and CLI never branch on tier.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

TokenSource = Literal["measured", "estimated"]
Tier = Literal["tier1", "tier2"]
SpanType = Literal["agent", "llm_call", "tool_call", "validator_decision"]


@dataclass
class CapturedSpan:
    """One normalized span, tier-agnostic, ready for the mapping engine + store."""

    span_id: str
    parent_span_id: str | None
    operation_name: str
    span_type: SpanType
    component_name: str | None = None
    component_type: str | None = None
    tags: dict[str, Any] = field(default_factory=dict)
    input_json: str | None = None
    output_json: str | None = None
    model: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    token_source: TokenSource | None = None
    started_at: str = ""
    latency_ms: int | None = None
    tier: Tier = "tier1"
    component_id: str | None = None  # pre-assigned for tier2; filled by mapping engine for tier1


@dataclass
class TraceResult:
    root_input: str
    final_output: str
    spans: list[CapturedSpan]
    total_tokens: int
    total_latency_ms: int
    unmatched: list[CapturedSpan] = field(default_factory=list)
    ambiguous: list[tuple[CapturedSpan, list[str]]] = field(default_factory=list)


class _HaystackPipelineLike(Protocol):
    async def run_async(self, data: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]: ...


@dataclass
class PipelineHandle:
    """What a test target's build_pipeline() hands back to the runner/adapter —
    a built Haystack AsyncPipeline plus just enough wiring info to feed a raw text
    query in and pull the final text answer out, without HaystackAdapter having to
    guess at arbitrary pipeline shapes (that guesswork is CS-264's job, not CS-261's).

    entry_inputs maps {component_name: input_key} for every component that should
    directly receive the raw query (e.g. T1 feeds "query" to both "retriever" and
    "writer"; T2 feeds it only to "guard" — everything downstream is wired via
    pipeline.connect()).
    """

    pipeline: _HaystackPipelineLike
    entry_inputs: dict[str, str]
    exit_node: str


class InstrumentationAdapter(ABC):
    """Lifecycle: attach() once, run(query) once per query, detach() once.

    Identical contract for both tiers — the runner never branches on which
    adapter it holds.
    """

    @abstractmethod
    def attach(self) -> None: ...

    @abstractmethod
    async def run(self, query: str) -> TraceResult: ...

    @abstractmethod
    def detach(self) -> None: ...
