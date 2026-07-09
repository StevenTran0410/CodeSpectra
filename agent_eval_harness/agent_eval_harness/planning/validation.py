"""CS-281 — Evaluation Plan Validation Engine.

validate_plan() now returns PlanValidationReport {errors, readiness} instead of
a bare list[str].  The errors list is backward-compatible (hard gate for callers
in cli.py and existing tests); readiness is the new per-gate sidecar.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from agent_eval_harness.metrics.registry import (
    DATASET_KINDS,
    get_spec,
    validate_metric,
)
from agent_eval_harness.metrics.suite import load_suite
from agent_eval_harness.store import repository

logger = logging.getLogger("agent_eval_harness.planning.validation")

# Stable reason-code strings — CS-282 wires dataset_unreviewed; §6b wires bind_unresolvable.
_REASON_NO_QUERIES = "no_queries"
_REASON_DATASET_UNFULFILLED = "dataset_unfulfilled"
_REASON_DATASET_UNREVIEWED = "dataset_unreviewed"  # reserved for CS-282
_REASON_MISSING_PARAMS = "missing_params"
_REASON_NOT_DISPATCHABLE = "metric_not_dispatchable"
_REASON_CLASSIFIER_UNBOUND = "classifier_unbound"
_REASON_INVALID_DATASET_KIND = "invalid_dataset_kind"
_REASON_BIND_UNRESOLVABLE = "bind_unresolvable"  # reserved for §6b / post CS-284

# Prefix, not "<TODO>" — the real placeholder is "<TODO: add a representative query
# for this target>" (planner.py:304,385), which never contains a literal "<TODO>".
_TODO_PATTERN = "<TODO"


class GateReadiness(BaseModel):
    """Per-gate executability verdict with reasons."""

    status: Literal["runnable", "blocked"]
    reasons: list[str]


class PlanValidationReport(BaseModel):
    """validate_plan() result — errors is the hard gate; readiness is the per-gate sidecar."""

    errors: list[str]
    readiness: dict[str, GateReadiness]


async def validate_plan(plan_path: str | Path) -> PlanValidationReport:
    """Validate an evaluation plan; return hard errors + per-gate readiness."""
    errors: list[str] = []
    readiness: dict[str, GateReadiness] = {}

    # 1. Schema parse
    try:
        suite = load_suite(plan_path)
    except Exception as e:
        return PlanValidationReport(
            errors=[f"Schema validation failed: {e}"], readiness={}
        )

    # Fetch DB datasets for reference checks
    try:
        db_datasets = await repository.list_dataset_ids()
        existing_ds_ids = {ds["dataset_id"] for ds in db_datasets}
    except Exception as e:
        logger.warning("Could not fetch dataset IDs from DB: %s", e)
        existing_ds_ids = set()

    for entry in suite.entries:
        blocked_reasons: list[str] = []

        # ── needs_human / unknown placeholder ─────────────────────────────────
        if entry.status == "needs_human":
            errors.append(
                f"Entry '{entry.id}': carries leftover 'needs_human' status marker. "
                "Must be reviewed and approved by human."
            )
        if entry.metric == "unknown":
            errors.append(
                f"Entry '{entry.id}': has 'unknown' metric placeholder. "
                "Human must define a valid metric."
            )
            readiness[entry.id] = GateReadiness(
                status="blocked", reasons=[_REASON_NOT_DISPATCHABLE]
            )
            continue

        # ── §2 Metric registry check ───────────────────────────────────────────
        spec = get_spec(entry.metric)
        if spec is None:
            errors.append(
                f"Entry '{entry.id}': metric '{entry.metric}' is not registered "
                "in METRIC_REGISTRY — unrepresentable."
            )
            blocked_reasons.append(_REASON_NOT_DISPATCHABLE)
        else:
            # metric_class consistency
            if spec.metric_class != entry.metric_class:
                errors.append(
                    f"Entry '{entry.id}': metric_class '{entry.metric_class}' "
                    f"disagrees with registry (expected '{spec.metric_class}')."
                )
            # Classifier prefix enforcement
            if entry.metric_class == "classifier" and not entry.metric.startswith("classifier."):
                errors.append(
                    f"Entry '{entry.id}': classifier metric '{entry.metric}' "
                    "must start with 'classifier.' prefix."
                )
            # Missing required_params → blocked
            for req_param in spec.required_params:
                val = entry.params.get(req_param)
                if not val or (isinstance(val, str) and (_TODO_PATTERN in val or not val.strip())):
                    blocked_reasons.append(_REASON_MISSING_PARAMS)
                    break

        # ── params.repeats validation ──────────────────────────────────────────
        repeats = entry.params.get("repeats")
        if repeats is not None:
            if not isinstance(repeats, int) or repeats < 1:
                errors.append(
                    f"Entry '{entry.id}': params.repeats must be an integer ≥ 1."
                )

        # ── Dataset checks ────────────────────────────────────────────────────
        if entry.dataset:
            ref = entry.dataset.ref
            required = entry.dataset.required
            waived = entry.dataset.waived

            if not ref and not required and not waived:
                errors.append(
                    f"Entry '{entry.id}': dataset reference is empty. "
                    "Must specify a 'ref', 'required' block, or 'waived' reason."
                )
            elif required and not ref and not waived:
                kind = required.get("kind", "")
                errors.append(
                    f"Entry '{entry.id}': dataset requirement of kind "
                    f"'{kind}' is unfulfilled. "
                    "Must resolve by specifying a 'ref' or 'waived' reason."
                )
                blocked_reasons.append(_REASON_DATASET_UNFULFILLED)
                # Validate kind vocabulary
                if kind and kind not in DATASET_KINDS:
                    errors.append(
                        f"Entry '{entry.id}': dataset kind '{kind}' is not in "
                        f"DATASET_KINDS vocabulary: {sorted(DATASET_KINDS)}"
                    )
                    blocked_reasons.append(_REASON_INVALID_DATASET_KIND)
            elif ref and ref not in existing_ds_ids:
                errors.append(
                    f"Entry '{entry.id}': dataset reference '{ref}' "
                    "does not exist in the database. "
                    "Must create the dataset or waive/mark as required."
                )
        elif entry.metric_class == "classifier":
            errors.append(
                f"Entry '{entry.id}': classifier metrics require a dataset reference."
            )
            blocked_reasons.append(_REASON_CLASSIFIER_UNBOUND)

        # ── Classifier: entry_point required ──────────────────────────────────
        if entry.metric_class == "classifier":
            if not entry.params.get("entry_point") and _REASON_CLASSIFIER_UNBOUND not in blocked_reasons:
                blocked_reasons.append(_REASON_CLASSIFIER_UNBOUND)

        # ── Queries / no_queries ───────────────────────────────────────────────
        if entry.metric_class in ("assertion", "llm_judge"):
            has_ref = bool(entry.dataset and entry.dataset.ref)
            queries = entry.params.get("queries")
            has_good_queries = bool(queries) and all(
                _TODO_PATTERN not in str(q) for q in (queries if isinstance(queries, list) else [])
            )
            if not has_ref and not has_good_queries:
                blocked_reasons.append(_REASON_NO_QUERIES)

        readiness[entry.id] = GateReadiness(
            status="blocked" if blocked_reasons else "runnable",
            reasons=blocked_reasons,
        )

    return PlanValidationReport(errors=errors, readiness=readiness)
