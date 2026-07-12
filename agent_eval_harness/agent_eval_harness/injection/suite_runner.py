"""Mirrors metrics/sweep.py::run_sweep's persistence shape (one run_id, one insert_evaluation call per MetricResult) so existing run-viewing UI/routes work unchanged."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from agent_eval_harness.injection.agent_invokers import invoke_agent
from agent_eval_harness.injection.proxy_provider_service import ProxyProviderService
from agent_eval_harness.injection.scoring import score_case
from agent_eval_harness.llm.client import LLMClient
from agent_eval_harness.metrics.types import MetricResult
from agent_eval_harness.planning.contract import EvaluationContract
from agent_eval_harness.store import repository

logger = logging.getLogger("agent_eval_harness.injection.suite_runner")


@dataclass
class SuiteRunResult:
    run_id: str
    results: list[MetricResult] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)


async def run_synthetic_agent_io_suite(
    agent_id: str,
    target_system_id: str,
    contract: EvaluationContract,
    dataset_cases: list[dict[str, Any]],
    provider_id: str,
    model_id: str,
    agent_llm_client: LLMClient,
    judge_llm_client: LLMClient,
    *,
    gold_dependent_fields: list[str] | None = None,
    run_id: str | None = None,
) -> SuiteRunResult:
    """Scores each case via a separate cross-family judge_llm_client — never the same provider that generated the gold."""
    if run_id is None:
        run_id = await repository.insert_run(target_system_id=target_system_id)

    result = SuiteRunResult(run_id=run_id)
    provider_service = ProxyProviderService(agent_llm_client)

    try:
        for row in dataset_cases:
            case_id = row["id"]
            case_input = json.loads(row["input_json"] or "{}")
            expected_raw = row.get("expected_json")
            gold = json.loads(expected_raw) if expected_raw else None

            try:
                actual_output = await invoke_agent(
                    agent_id, case_input, provider_service, provider_id, model_id,
                )
            except Exception as exc:
                logger.error(
                    "injection run: agent '%s' case %s raised: %s", agent_id, case_id, exc
                )
                result.errors.append({"case_id": case_id, "error": str(exc)})
                continue

            metrics = await score_case(
                agent_id, contract, case_input, gold, actual_output, judge_llm_client,
                gold_dependent_fields=gold_dependent_fields,
            )
            result.results.extend(metrics)
            for mr in metrics:
                await repository.insert_evaluation(
                    run_id=run_id,
                    metric_name=mr.metric_name,
                    metric_class=mr.metric_class,
                    component_id=mr.component_id or agent_id,
                    score=mr.score,
                    passed=mr.passed,
                    details={**mr.details, "dataset_case_id": case_id},
                    evaluator=mr.evaluator,
                    cost_tokens=mr.cost_tokens,
                    agent_id=agent_id,
                )
    except Exception:
        await repository.finish_run(run_id, "failed")
        raise

    await repository.finish_run(run_id, "completed")
    return result
