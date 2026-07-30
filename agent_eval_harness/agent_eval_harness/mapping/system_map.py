"""System Map schema — pydantic models + YAML loader."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class SpanMatchBlock(BaseModel):
    """One AND-block within a component's span_match list (OR across the list)."""

    component_name: str | None = None
    component_type: str | None = None
    span_name_pattern: str | None = None
    tags: dict[str, Any] = Field(default_factory=dict)


class Constraint(BaseModel):
    name: str
    value: Any = None
    source: str  # file:line citation (required)


class CallSiteRecord(BaseModel):
    """One enumerated call site in a component's call closure — a receipt ledger a downstream
    verdict layer can cite as evidence."""
    site_id: str
    file: str
    lineno: int
    callee_text: str
    source_line: str = ""
    outcome: str = "unresolved"  # "resolved" | "out_of_scope" | "unresolved"
    reason: str | None = None
    external_module: str | None = None


class Component(BaseModel):
    id: str
    role: str
    role_confidence: float | None = None
    role_source: str | None = None
    is_tool: bool | None = None
    is_library_object: bool | None = None  # True for a framework/library link surfaced in a chain (LCEL): explicit degrade, never harvestable
    constructor_fanout: int | None = None
    constructor_downstream: list[str] | None = None  # WHICH siblings this constructs; fanout is its count
    model: str | None = None
    entry_point: str
    entry_kind: str | None = None  # "class" | "function" | "bound_method"; None on legacy maps => treated as class
    needs_human: bool | None = None  # True when a wiring step's defining file could not be resolved by any layer (import/MRO/RRF) — surfaced for review, never silently dropped
    out_of_scope: bool | None = None  # True when this component's file was deliberately refused (boundary/excluded) rather than widened into the accepted set
    out_of_scope_reason: str | None = None  # why (e.g. "file classified boundary during expansion") — set together with out_of_scope, never guessed
    file: str = ""
    span_match: list[SpanMatchBlock] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    upstream: list[str] = Field(default_factory=list)
    downstream: list[str] = Field(default_factory=list)
    # CS-319 (additive; empty except on a LangGraph map with an intra-node call graph): subset of
    # `downstream` reached by a gray "call" edge, and subset whose hard edge is a conditional router branch.
    call_downstream: list[str] = Field(default_factory=list)
    conditional_downstream: list[str] = Field(default_factory=list)
    # CS-321: motif classification (linear/branch/loop).
    motif: str | None = None
    # Did every call in this component's closure resolve, or did at least one stop at an unresolved
    # seam? None when the closure was never walked (no recognized entry method).
    closure_complete: bool | None = None
    call_sites: list[CallSiteRecord] = Field(default_factory=list)
    # Static tri-state verdict (None means undecided, never "no"): does this component itself make a
    # model call. model_call_source names which tier decided it; model_call_citation is its "file:line".
    makes_model_call: bool | None = None
    model_call_source: str | None = None
    model_call_citation: str | None = None
    model_call_evidence_hash: str | None = None


class SystemMap(BaseModel):
    target_system_id: str
    components: list[Component]
    framework: str | None = None  # the scanner that produced this map; None on legacy maps
    system_type: str | None = None  # detected agentic system type (pipeline, orchestrator, etc.)
    discrepancies: list[str] = Field(default_factory=list)

    def component_by_id(self, component_id: str) -> Component | None:
        for c in self.components:
            if c.id == component_id:
                return c
        return None


_SYSTEM_MAP_CACHE: dict[tuple[str, float], SystemMap] = {}


def load_system_map(path: str | Path) -> SystemMap:
    """Schema validation is a hard gate — invalid shape fails loudly. Cached by (path, mtime)."""
    p = Path(path)
    try:
        mtime = p.stat().st_mtime
    except OSError:
        mtime = 0.0
    cache_key = (str(p.resolve()), mtime)
    if cache_key in _SYSTEM_MAP_CACHE:
        return _SYSTEM_MAP_CACHE[cache_key]

    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    sys_map = SystemMap.model_validate(data)
    _SYSTEM_MAP_CACHE[cache_key] = sys_map
    return sys_map



def save_system_map(system_map: SystemMap, path: str | Path) -> None:
    """Serialize system map to yaml."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.dump(system_map.model_dump(), f, default_flow_style=False, sort_keys=False)

