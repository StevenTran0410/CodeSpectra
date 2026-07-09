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


class Component(BaseModel):
    id: str
    role: str
    model: str | None = None
    entry_point: str
    file: str = ""
    span_match: list[SpanMatchBlock] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    upstream: list[str] = Field(default_factory=list)
    downstream: list[str] = Field(default_factory=list)


class SystemMap(BaseModel):
    target_system_id: str
    components: list[Component]
    discrepancies: list[str] = Field(default_factory=list)

    def component_by_id(self, component_id: str) -> Component | None:
        for c in self.components:
            if c.id == component_id:
                return c
        return None


def load_system_map(path: str | Path) -> SystemMap:
    """Schema validation is a hard gate — invalid shape fails loudly."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return SystemMap.model_validate(data)


def save_system_map(system_map: SystemMap, path: str | Path) -> None:
    """Serialize system map to yaml."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.dump(system_map.model_dump(), f, default_flow_style=False, sort_keys=False)

