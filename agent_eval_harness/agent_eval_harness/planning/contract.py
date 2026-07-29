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
    # Field-level schema when annotation names an in-repo type (or was resolved via an unambiguous
    # upstream agent output); None = unresolved. Mirrors agent_knowledge.ContractArg's input_schemas entry.
    resolved_schema: dict[str, Any] | None = None
    # Provenance tag mirroring agent_knowledge.ContractArg.source_kind (config-static /
    # upstream-agent-output / runtime-state / internally-retrieved); None = unclassified.
    source_kind: str | None = None


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
    section_letter: str | None = None
    # field -> allowed literal values, e.g. {"confidence": ["low","medium","high"]}; empty until harvested.
    schema_enum_values: dict[str, list[str]] = Field(default_factory=dict)
    # Set when the AST-derived json_schema disagreed with a prompt-derived key-set (see contract_harvest cross-validation).
    schema_discrepancy: str | None = None
    # Which return branch produced json_schema; None when undeterminable.
    cardinality: Literal["object", "array"] | None = None
    # True when a yield/stream co-exists with the return dict this schema was harvested from (e.g. streamed markdown + a returned meta-JSON) — the json_schema stays the sole scorable gold.
    has_streamed_output: bool = False


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
    entry_is_async: bool = False


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
    retrieves_internally: bool = False
    # Harvested upstream output specs — [{name, description}, ...] — for generic dataset builders (D3).
    upstream_context_specs: list[dict] = Field(default_factory=list)
    # Deterministic archetype computed at Stage 2 after upstream_context_specs is filled (CS-311 relocate).
    archetype: str = ""
