"""Classifier scorer — sklearn-based evaluation over labeled datasets."""
from __future__ import annotations

import json
from typing import Any

from agent_eval_harness.instrumentation.tier2_boundary import _resolve_entry_point
from agent_eval_harness.llm.client import LLMClient
from agent_eval_harness.metrics.types import MetricResult
from agent_eval_harness.store import repository

CLASSIFIER_PASS_THRESHOLD = 0.8


async def score_classifier(
    dataset_id: str,
    component_entry_point: str,
    llm_client: LLMClient,
    *,
    component_id: str | None = None,
    metric_name: str = "classifier.accuracy",
) -> MetricResult:
    """Run every case in ``dataset_id`` through the resolved component function, then score
    predictions against gold labels via sklearn."""
    try:
        from sklearn.metrics import classification_report, confusion_matrix
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is not installed. Run `pip install scikit-learn>=1.4.0`."
        ) from exc

    cases = await repository.get_dataset_cases(dataset_id)
    if not cases:
        return MetricResult(
            metric_name=metric_name,
            metric_class="classifier",
            score=None,
            passed=None,
            details={"error": f"No cases found in dataset '{dataset_id}'"},
            component_id=component_id,
            evaluator="sklearn",
        )

    component_cls = _resolve_entry_point(component_entry_point)
    component = component_cls(llm_client)

    y_true: list[str] = []
    y_pred: list[str] = []
    case_ids: list[str] = []
    errors: list[dict] = []

    for case in cases:
        case_id = case["id"]
        input_data = json.loads(case.get("input_json") or "{}")
        expected_data = json.loads(case.get("expected_json") or "{}")
        query = input_data.get("query", input_data.get("text", ""))
        true_label = str(expected_data.get("verdict", expected_data.get("label", "")))
        if not true_label:
            continue

        try:
            result: Any = await component.run_async(query)
            if isinstance(result, dict):
                pred_label = str(
                    result.get("verdict", result.get("label", result.get("decision", "")))
                )
            elif hasattr(result, "verdict"):
                pred_label = str(result.verdict)
            else:
                pred_label = str(result)
        except Exception as exc:  # noqa: BLE001
            errors.append({"case_id": case_id, "error": str(exc)})
            continue

        y_true.append(true_label)
        y_pred.append(pred_label)
        case_ids.append(case_id)

    if not y_true:
        return MetricResult(
            metric_name=metric_name,
            metric_class="classifier",
            score=None,
            passed=None,
            details={"errors": errors, "total_cases": len(cases)},
            component_id=component_id,
            evaluator="sklearn",
        )

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=sorted(set(y_true) | set(y_pred)))
    accuracy: float = report.get("accuracy", 0.0)  # type: ignore[assignment]

    # Decision-cost extras for validator role
    false_pass_ids: list[str] = []  # pred=true/pass but actual=false/reject
    false_reject_ids: list[str] = []  # pred=false/reject but actual=true/pass

    pass_tokens = {"pass", "true", "sufficient", "1"}
    reject_tokens = {"reject", "false", "insufficient", "0"}

    for cid, t, p in zip(case_ids, y_true, y_pred):
        t_lower, p_lower = t.lower(), p.lower()
        if t_lower in reject_tokens and p_lower in pass_tokens:
            false_pass_ids.append(cid)
        elif t_lower in pass_tokens and p_lower in reject_tokens:
            false_reject_ids.append(cid)

    return MetricResult(
        metric_name=metric_name,
        metric_class="classifier",
        score=accuracy,
        passed=accuracy >= CLASSIFIER_PASS_THRESHOLD,
        details={
            "total_cases": len(cases),
            "evaluated": len(y_true),
            "errors": errors,
            "report": report,
            "confusion_matrix": cm.tolist(),
            "false_pass_case_ids": false_pass_ids,
            "false_reject_case_ids": false_reject_ids,
        },
        component_id=component_id,
        evaluator="sklearn",
    )
