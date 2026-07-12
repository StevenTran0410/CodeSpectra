"""Unified metric registry: validate <=> dispatchable.

Single source of truth replacing three divergent copies (planning/validation.py's inline
valid_judges set, planning/agentic_planner.py's _validate_metric, and metrics/sweep.py's
_dispatch_judge if/elif chain). A metric is only representable if registered here;
ragas.context_precision is intentionally absent — it has no dispatch handler.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class MetricSpec:
    """Complete description of one metric family."""

    metric_class: Literal["assertion", "classifier", "llm_judge"]
    # Parameters that MUST be non-empty for this gate to be runnable.
    required_params: list[str] = field(default_factory=list)
    # Importable handler name used by sweep._dispatch_judge; None = assertion (uses ASSERTIONS dict).
    dispatch: str | None = None
    # ObservabilityContract boolean fields that make this metric meaningless when False/None.
    # Evaluated as: getattr(obs, slot) is False or getattr(obs, slot) is None  → meaningless.
    meaningless_when: list[str] = field(default_factory=list)
    # Required slots from the contract that must be non-None at plan time.
    required_slots: list[str] = field(default_factory=list)
    # "free" = deterministic/no LLM call; "llm_per_case" = one LLM call per case.
    cost_class: Literal["free", "llm_per_case"] = "free"
    # Human-readable description for GATE_DESIGNER_SYSTEM prompt injection.
    description: str = ""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Prefix sentinel: entries whose key ends with ".*" are prefix families.
# validate_metric() and get_dispatch() handle prefix matching.

METRIC_REGISTRY: dict[str, MetricSpec] = {
    # ── Assertions (cost_class: free; dispatch=None → ASSERTIONS dict) ────────
    "allowed_downstream": MetricSpec(
        metric_class="assertion",
        required_params=["allowed"],
        description="Downstream component set matches the declared allowed list.",
        cost_class="free",
    ),
    "arg_schema": MetricSpec(
        metric_class="assertion",
        required_params=["schema"],
        description="Agent output matches a JSON schema. Prefer schema_valid (contract-seeded).",
        cost_class="free",
    ),
    "max_items_per_call": MetricSpec(
        metric_class="assertion",
        description="Agent emits ≤ N items per call.",
        cost_class="free",
    ),
    "max_retries": MetricSpec(
        metric_class="assertion",
        description="Agent retries ≤ N times.",
        cost_class="free",
    ),
    "no_unnecessary_calls": MetricSpec(
        metric_class="assertion",
        meaningless_when=["has_tools"],  # meaningful ONLY when has_tools=True
        description="Agent makes no tool calls beyond what is needed.",
        cost_class="free",
    ),
    "retry_on_reject_required": MetricSpec(
        metric_class="assertion",
        description="Agent retries when a downstream component rejects its output.",
        cost_class="free",
    ),
    # ── Deterministic metrics ──────────────────────────────────────────────────
    "schema_valid": MetricSpec(
        metric_class="assertion",
        required_params=["schema"],
        required_slots=["output.json_schema"],
        description="Agent output validates against the contract's harvested JSON schema.",
        cost_class="free",
    ),
    "fallback_sentinel": MetricSpec(
        metric_class="assertion",
        required_slots=["output.fallback_literal"],
        description="Output is not equal to the contract's fallback literal (fallbacks always pass schema — this gate catches them).",
        cost_class="free",
    ),
    "referential_integrity": MetricSpec(
        metric_class="assertion",
        description="Evidence/file-path fields in output are a subset of snapshot manifest ∪ retrieved-chunk rel_paths.",
        cost_class="free",
    ),
    "field_match": MetricSpec(
        metric_class="assertion",
        required_params=["fields"],
        description="Per-field exact|set|tolerance comparison vs gold.",
        cost_class="free",
    ),
    "list_field_prf1": MetricSpec(
        metric_class="assertion",
        required_params=["field"],
        description="Precision/recall/F1 on a list-valued output field vs gold.",
        cost_class="free",
    ),
    "llm_call_budget": MetricSpec(
        metric_class="assertion",
        required_slots=["observability.llm_call_budget"],
        description="LLM span count for this component ≤ contract's llm_call_budget.",
        cost_class="free",
    ),
    # ── LLM judges ────────────────────────────────────────────────────────────
    "ragas.faithfulness": MetricSpec(
        metric_class="llm_judge",
        dispatch="ragas_faithfulness",
        meaningless_when=["has_separable_context"],  # meaningless when False/None
        description="Ragas faithfulness: answer is grounded in retrieved context. Requires separable context.",
        cost_class="llm_per_case",
    ),
    "ragas.answer_relevancy": MetricSpec(
        metric_class="llm_judge",
        dispatch="ragas_answer_relevancy",
        meaningless_when=["input_kind_is_query"],  # meaningless when input is not a natural-language query
        description="Ragas answer relevancy: answer addresses the user query. Only meaningful for query-shaped inputs.",
        cost_class="llm_per_case",
    ),
    # ragas.context_precision intentionally ABSENT — no dispatch handler exists.
    "tool_correctness": MetricSpec(
        metric_class="llm_judge",
        dispatch="tool_correctness",
        required_params=["expected_tools"],
        meaningless_when=["has_tools"],  # meaningless when has_tools=False
        description="DeepEval tool correctness: agent calls the expected tools.",
        cost_class="llm_per_case",
    ),
    "grounding_judge_span_prompt": MetricSpec(
        metric_class="llm_judge",
        dispatch="geval",
        required_params=["rubric_text"],
        description=(
            "GEval variant that binds context to the span's full input prompt — "
            "for inlined-evidence agents where context is not separable. Sampled."
        ),
        cost_class="llm_per_case",
    ),
    "geval.decomposition_coverage": MetricSpec(
        metric_class="llm_judge",
        dispatch="geval",
        required_params=["rubric_text"],
        meaningless_when=["input_kind_is_query"],  # no decomposition artifact on a fixed-fan-out step
        description=(
            "GEval judge: decomposed intents cover the input query. Only meaningful for a genuine "
            "dynamic planner/router whose input is a free-text query it decomposes at runtime — "
            "never for a fixed-fan-out orchestrator, regardless of its role label."
        ),
        cost_class="llm_per_case",
    ),
    # geval.* prefix family (all other geval.<name> metrics) — handled by prefix matching below.
    # classifier.* prefix family — handled by prefix matching below.
}

# Closed vocabulary for dataset.required.kind.
DATASET_KINDS: frozenset[str] = frozenset(
    {
        "qa_testset",
        "guard_classification",
        "decomposition_gold",
        "sufficiency_labeled",
        "field_match_gold",
        "retrieval_grounded",
        "multi_turn_session",
        # Repo-input snapshot kinds — added to the registry that already shipped the 3
        # kinds above; not a rename, these didn't exist before.
        "snapshot_fixture",
        "snapshot_regression_baseline",
        "recorded_report_replay",  # mines real analysis_reports rows for K/L, no new ingestion needed
        "synthetic_agent_io",  # CS-289: LLM-synthesized input+gold at an agent's real LLM boundary
    }
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_GEVAL_PREFIX = "geval."
_CLASSIFIER_PREFIX = "classifier."


def _spec_for(metric: str) -> MetricSpec | None:
    """Return the MetricSpec for a metric name, handling prefix families."""
    if metric in METRIC_REGISTRY:
        return METRIC_REGISTRY[metric]
    if metric.startswith(_GEVAL_PREFIX):
        return MetricSpec(
            metric_class="llm_judge",
            dispatch="geval",
            required_params=["rubric_text"],
            description="GEval LLM judge with a custom rubric.",
            cost_class="llm_per_case",
        )
    if metric.startswith(_CLASSIFIER_PREFIX):
        return MetricSpec(
            metric_class="classifier",
            description="Classifier gate; requires dataset.ref and params.entry_point.",
            cost_class="free",
        )
    return None


def validate_metric(metric: str, metric_class: str | None = None) -> bool:
    """Return True if the metric is registered and (optionally) matches metric_class."""
    spec = _spec_for(metric)
    if spec is None:
        return False
    if metric_class is not None and spec.metric_class != metric_class:
        return False
    return True


def get_spec(metric: str) -> MetricSpec | None:
    """Return the full MetricSpec or None if unregistered."""
    return _spec_for(metric)


def get_dispatch(metric: str) -> str | None:
    """Return the dispatch handler name, or None for assertions (use ASSERTIONS dict)."""
    spec = _spec_for(metric)
    if spec is None:
        return None
    return spec.dispatch


def iter_metrics() -> list[tuple[str, MetricSpec]]:
    """All explicitly registered metrics (does not enumerate prefix families)."""
    return list(METRIC_REGISTRY.items())


def dataset_kind_valid(kind: str) -> bool:
    return kind in DATASET_KINDS
