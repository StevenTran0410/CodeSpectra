"""Stage 3 agentic planner's per-agent report — the auditable "why" behind the plan."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from agent_eval_harness.planning.contract import EvaluationContract
from agent_eval_harness.metrics.suite import DatasetRef

GateLocation = Literal["input", "output", "handoff", "internal_tool"]
GateToolkit = Literal["assertion", "classifier", "ragas", "deepeval"]


class AgentDataProfile(BaseModel):
    """Data-Flow Analyst node output — read from the agent's real code, not guessed from its role."""

    agent_id: str
    purpose: str | None = None
    input_data: str = ""
    output_data: str = ""
    internal_tools: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    consistency_notes: list[str] = Field(default_factory=list)
    # Structured observability mirrors — None = analyst didn't answer.
    input_kind: str | None = None
    has_separable_context: bool | None = None
    context_location: str | None = None


class EvaluationGate(BaseModel):
    """One proposed evaluation point; compiles 1:1 into a metrics/suite.py SuiteEntry."""

    id: str
    agent_id: str
    component: str
    location: GateLocation
    property: str = ""
    metric: str
    metric_class: Literal["assertion", "classifier", "llm_judge"]
    toolkit: GateToolkit
    params: dict[str, Any] = Field(default_factory=dict)
    dataset: DatasetRef | None = None
    dataset_kind: str | None = None
    level: str = "component"
    rationale: str = ""
    provenance: Literal["rule", "human_added", "llm_suggested"] = "rule"
    status: str | None = None


class AgentPlanReport(BaseModel):
    agent_id: str
    role: str = "unknown"
    label: str = ""
    data_profile: AgentDataProfile | None = None
    contract: EvaluationContract | None = None
    gates: list[EvaluationGate] = Field(default_factory=list)
    needs_human: list[str] = Field(default_factory=list)
    # Per-agent opt-out for the synthetic_agent_io workflow-eval dataset; Stage3Screen toggle controls cost.
    eval_enabled: bool = True


class EvaluationPlanReport(BaseModel):
    target_system_id: str
    agents: list[AgentPlanReport] = Field(default_factory=list)
    advisory_notes: list[str] = Field(default_factory=list)
    schema_version: int | None = None


def load_plan_report(path: str | Path) -> EvaluationPlanReport:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return EvaluationPlanReport.model_validate(data)


def save_plan_report(report: EvaluationPlanReport, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.dump(report.model_dump(), f, default_flow_style=False, sort_keys=False)
