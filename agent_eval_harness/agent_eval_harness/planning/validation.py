"""validation.py — Evaluation Plan Validation Engine (CS-265 §4).

Validates:
1. Schema-validity (using load_suite).
2. Metric name existence in assertions, judges, or classifiers.
3. Dataset reference resolution (exists in DB OR is required/waived).
4. Zero needs_human markers or unknown metric entries.
"""
from __future__ import annotations

import logging
from pathlib import Path

from agent_eval_harness.metrics.assertions.registry import get_assertion
from agent_eval_harness.metrics.suite import load_suite
from agent_eval_harness.store import repository

logger = logging.getLogger("agent_eval_harness.planning.validation")


async def validate_plan(plan_path: str | Path) -> list[str]:
    """Validate an evaluation plan and return a list of error messages.

    If the list is empty, validation has passed.
    """
    errors: list[str] = []

    # 1. Schema validation
    try:
        suite = load_suite(plan_path)
    except Exception as e:
        return [f"Schema validation failed: {e}"]

    # Fetch DB datasets for reference checks
    try:
        db_datasets = await repository.list_dataset_ids()
        existing_ds_ids = {ds["dataset_id"] for ds in db_datasets}
    except Exception as e:
        logger.warning(f"Could not fetch dataset IDs from DB: {e}")
        existing_ds_ids = set()

    for entry in suite.entries:
        # 2. Check needs_human markers
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
            continue

        # 3. Check metric name registry
        metric_class = entry.metric_class
        metric = entry.metric

        if metric_class == "assertion":
            try:
                get_assertion(metric)
            except KeyError:
                errors.append(
                    f"Entry '{entry.id}': assertion metric '{metric}' does not exist in registry."
                )
        elif metric_class == "llm_judge":
            valid_judges = {
                "ragas.faithfulness",
                "ragas.answer_relevancy",
                "ragas.context_precision",
                "tool_correctness",
            }
            if metric not in valid_judges and not metric.startswith("geval."):
                errors.append(
                    f"Entry '{entry.id}': LLM judge metric '{metric}' is not registered."
                )
        elif metric_class == "classifier":
            if not metric.startswith("classifier."):
                errors.append(
                    f"Entry '{entry.id}': classifier metric '{metric}' "
                    "must start with 'classifier.' prefix."
                )
        else:
            errors.append(
                f"Entry '{entry.id}': unknown metric_class '{metric_class}'."
            )

        # 4. Check dataset references
        if entry.dataset:
            ref = entry.dataset.ref
            required = entry.dataset.required
            waived = entry.dataset.waived

            # Check if completely empty
            if not ref and not required and not waived:
                errors.append(
                    f"Entry '{entry.id}': dataset reference is empty. "
                    "Must specify a 'ref', 'required' block, or 'waived' reason."
                )
            # If required but not resolved/waived
            elif required and not ref and not waived:
                errors.append(
                    f"Entry '{entry.id}': dataset requirement of kind "
                    f"'{required.get('kind')}' is unfulfilled. "
                    "Must resolve by specifying a 'ref' or 'waived' reason."
                )
            # If ref is specified, it must exist in the database
            elif ref and ref not in existing_ds_ids:
                errors.append(
                    f"Entry '{entry.id}': dataset reference '{ref}' "
                    "does not exist in the database. "
                    "Must create the dataset or waive/mark as required."
                )
        elif metric_class == "classifier":
            errors.append(
                f"Entry '{entry.id}': classifier metrics require a dataset reference."
            )

    return errors
