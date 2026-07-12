"""Scores an agent output against its contract in a fixed order — deterministic checks first, then gold-independent groundedness G-Eval, then gold-dependent G-Eval only if groundedness passed; do not reorder."""
from __future__ import annotations

import json
from typing import Any

from agent_eval_harness.llm.client import LLMClient
from agent_eval_harness.metrics.judges.deepeval_geval import run_geval
from agent_eval_harness.metrics.scoring_helpers import matches_fallback
from agent_eval_harness.metrics.types import MetricResult
from agent_eval_harness.planning.contract import EvaluationContract

_GROUNDEDNESS_RUBRIC = (
    "Determine whether every claim in ACTUAL_OUTPUT is directly supported by INPUT. "
    "Penalize any fact, score, name, or claim in ACTUAL_OUTPUT that is not present in or "
    "reasonably inferable from INPUT. Do not penalize omissions — only fabrications."
)

_GOLD_DEPENDENT_RUBRIC_TEMPLATE = (
    "Compare ACTUAL_OUTPUT against EXPECTED_OUTPUT — both are the value of the single field "
    "'{field}'. Score how semantically consistent ACTUAL_OUTPUT is with EXPECTED_OUTPUT for "
    "this field. Exact wording is not required, but the substance/judgment must match."
)


def normalize_for_judging(data: Any) -> str:
    """json.dumps sorted+compact — strips whitespace/ordering fingerprints before either side reaches the judge."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _schema_valid(actual_output: dict, json_schema: dict[str, Any] | None) -> MetricResult:
    if not json_schema:
        return MetricResult(
            metric_name="assertion.schema_valid",
            metric_class="assertion",
            score=None,
            passed=None,
            details={"note": "no json_schema harvested — nothing to check"},
        )
    import jsonschema

    try:
        jsonschema.validate(instance=actual_output, schema=json_schema)
        return MetricResult(
            metric_name="assertion.schema_valid", metric_class="assertion",
            score=None, passed=True, details={},
        )
    except jsonschema.ValidationError as exc:
        return MetricResult(
            metric_name="assertion.schema_valid", metric_class="assertion",
            score=None, passed=False, details={"error": exc.message},
        )


def _enum_membership(
    actual_output: dict, schema_enum_values: dict[str, list[str]] | None
) -> MetricResult:
    if not schema_enum_values:
        return MetricResult(
            metric_name="assertion.enum_membership",
            metric_class="assertion",
            score=None,
            passed=None,
            details={"note": "no schema_enum_values harvested — nothing to check"},
        )
    violations = []
    for field, allowed in schema_enum_values.items():
        if field not in actual_output:
            continue
        value = actual_output[field]
        if value not in allowed:
            violations.append({"field": field, "value": value, "allowed": allowed})
    return MetricResult(
        metric_name="assertion.enum_membership", metric_class="assertion",
        score=None, passed=len(violations) == 0, details={"violations": violations},
    )


def _fallback_sentinel(actual_output: dict, fallback_literal: dict[str, Any] | None) -> MetricResult:
    if not fallback_literal:
        return MetricResult(
            metric_name="assertion.fallback_sentinel",
            metric_class="assertion",
            score=None,
            passed=None,
            details={"note": "no fallback literal harvested — nothing to check"},
        )
    hit = matches_fallback(actual_output, fallback_literal)
    return MetricResult(
        metric_name="assertion.fallback_sentinel", metric_class="assertion",
        score=None, passed=not hit, details={"fallback_hit": hit},
    )


async def score_case(
    agent_id: str,
    contract: EvaluationContract,
    case_input: dict[str, Any],
    gold: dict[str, Any] | None,
    actual_output: dict[str, Any],
    judge_llm_client: LLMClient,
    *,
    gold_dependent_fields: list[str] | None = None,
) -> list[MetricResult]:
    output = contract.output
    json_schema = output.json_schema if output else None
    schema_enum_values = output.schema_enum_values if output else None
    fallback_literal = output.fallback_literal if output else None

    results: list[MetricResult] = [
        _schema_valid(actual_output, json_schema),
        _enum_membership(actual_output, schema_enum_values),
        _fallback_sentinel(actual_output, fallback_literal),
    ]

    groundedness = await run_geval(
        f"{agent_id}.groundedness",
        _GROUNDEDNESS_RUBRIC,
        normalize_for_judging(case_input),
        normalize_for_judging(actual_output),
        judge_llm_client,
        component_id=agent_id,
    )
    results.append(groundedness)

    if not groundedness.passed or not gold_dependent_fields or gold is None:
        return results

    for field in gold_dependent_fields:
        if field not in actual_output or field not in gold:
            continue
        judged = await run_geval(
            f"{agent_id}.gold_dependent.{field}",
            _GOLD_DEPENDENT_RUBRIC_TEMPLATE.format(field=field),
            normalize_for_judging(case_input),
            normalize_for_judging(actual_output[field]),
            judge_llm_client,
            expected_output=normalize_for_judging(gold[field]),
            component_id=agent_id,
        )
        results.append(judged)

    return results
