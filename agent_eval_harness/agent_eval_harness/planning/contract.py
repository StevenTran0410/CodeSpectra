"""Per-agent EvaluationContract models, harvested statically (AST) from the target."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class KwargSpec(BaseModel):
    """One parameter of the entry method, as written in source."""

    name: str
    annotation: str | None = None
    default_repr: str | None = None
    required: bool = True


class InvocationContract(BaseModel):
    """How to drive this agent — callable, kwargs, and how a dataset case maps onto them."""

    callable: str = ""
    method: str = ""
    kwargs: list[KwargSpec] = Field(default_factory=list)
    constructor_deps: list[str] = Field(default_factory=list)
    invocation_mode: Literal["pipeline_entry", "per_agent_route", "in_harness", "unsupported"] = "unsupported"
    route: str | None = None
    case_binding: dict[str, str] = Field(default_factory=dict)
    source: Literal["ast", "llm", "human"] = "ast"
    citations: list[str] = Field(default_factory=list)


class OutputContract(BaseModel):
    """The agent's output shape, harvested from the target's own schemas/fallbacks."""

    json_schema: dict[str, Any] | None = None
    schema_source: str | None = None
    fallback_literal: dict[str, Any] | None = None
    fallback_source: str | None = None
    validated_in_target: bool = False
    # field -> allowed literal values, e.g. {"confidence": ["low","medium","high"]}; empty until harvested.
    schema_enum_values: dict[str, list[str]] = Field(default_factory=dict)


class ObservabilityContract(BaseModel):
    """What evaluation can actually observe for this agent; llm_fields lists LLM-inferred keys."""

    has_tools: bool = False
    # None = static harvest did not determine this (LLM may fill); False/True = static fact.
    has_separable_context: bool | None = None
    context_location: str | None = None
    input_kind: Literal["query", "structured", "unknown"] = "unknown"
    is_multi_turn: bool = False
    spans_have_usage: bool = True
    llm_call_budget: int | None = None
    llm_fields: list[str] = Field(default_factory=list)


class EvaluationContract(BaseModel):
    agent_id: str
    component_id: str = ""
    invocation: InvocationContract | None = None
    output: OutputContract | None = None
    observability: ObservabilityContract = Field(default_factory=ObservabilityContract)
    constants: dict[str, int] = Field(default_factory=dict)
    connect_edges: list[dict[str, str]] = Field(default_factory=list)
    needs_human: list[str] = Field(default_factory=list)
    # letter -> sorted field names this agent's entry method statically reads from that upstream section (fan-in agents only).
    field_downstream_consumers: dict[str, list[str]] = Field(default_factory=dict)
    # True when the entry method calls a module-level `plan_queries(...)` helper before retrieval.
    query_planning_subcall: bool = False
    # True when keyword-tier OR role-tier retrieval signal fires (D1).
    has_retrieval_signal: bool = False
    # Harvested upstream output specs — [{name, description}, ...] — for generic dataset builders (D3).
    upstream_context_specs: list[dict] = Field(default_factory=list)
