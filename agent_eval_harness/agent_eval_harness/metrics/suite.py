"""Suite config schema — designed to match the Evaluation Plan format exactly."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class DatasetRef(BaseModel):
    """Inline dataset reference — either a named ref or a required-spec dict."""

    ref: str | None = None
    required: dict[str, Any] | None = None
    waived: str | None = None


class SuiteEntry(BaseModel):
    """One evaluation entry in the suite — maps to one evaluations row per case/trace."""

    id: str
    component: str
    metric: str
    metric_class: Literal["assertion", "classifier", "llm_judge"]
    # Representability only — nothing consumes level at run-time yet.
    level: Literal["component", "trace", "session"] = "component"
    # codegen keys off execution="entrypoint" for in-process classifier injection.
    execution: Literal["harness", "entrypoint"] = "harness"
    dataset: DatasetRef | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""
    provenance: Literal["rule", "human_added", "llm_suggested"] = "rule"
    status: str | None = None
    agent_id: str | None = None



class Suite(BaseModel):
    """Evaluation suite — a list of SuiteEntry items loaded from a YAML file."""

    entries: list[SuiteEntry]


def load_suite(path: str | Path) -> Suite:
    """Load and validate a suite YAML file."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        data = {"entries": data}
    return Suite.model_validate(data)
