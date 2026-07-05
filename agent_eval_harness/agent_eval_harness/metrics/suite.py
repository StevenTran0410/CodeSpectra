"""Suite config schema — designed to match the CS-265 Evaluation Plan format exactly.

CS-263 §7: CS-265's ticket explicitly specifies the future schema; CS-263 must
adopt it now so CS-265 only needs to write a *generator*, not change the consuming
code (the sweep runner). All fields from the CS-265 example are present here.
"""
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
    dataset: DatasetRef | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""
    provenance: Literal["rule", "human_added", "llm_suggested"] = "rule"
    status: str | None = None


class Suite(BaseModel):
    """Evaluation suite — a list of SuiteEntry items loaded from a YAML file."""

    entries: list[SuiteEntry]


def load_suite(path: str | Path) -> Suite:
    """Load and validate a suite YAML file.

    Accepts either a top-level list (shorthand) or ``{entries: [...]}`` dict.
    Raises ``pydantic.ValidationError`` on invalid schema — hard gate per epic
    convention.
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        data = {"entries": data}
    return Suite.model_validate(data)
