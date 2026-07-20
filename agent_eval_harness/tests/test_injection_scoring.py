"""Tests for the Stage-4 scoring pipeline (CS-289 Workstream D3/D4)."""
from __future__ import annotations

import json

import pytest

from agent_eval_harness.injection.judge_selection import pick_judge_provider
from agent_eval_harness.injection.scoring import normalize_for_judging, score_case
from agent_eval_harness.planning.contract import EvaluationContract, OutputContract

_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_confidence": {"type": "string"},
        "notes": {"type": "string"},
        "purpose": {"type": "string"},
    },
    "required": ["overall_confidence", "notes"],
}


def _contract(**output_kwargs) -> EvaluationContract:
    return EvaluationContract(
        agent_id="auditor",
        output=OutputContract(json_schema=_SCHEMA, **output_kwargs),
    )


def _geval_responses(score: float, reason: str = "ok"):
    from agent_eval_harness.llm.client import LLMResponse

    steps = LLMResponse(
        content=json.dumps({"steps": ["Check groundedness.", "Check no fabrication."]}),
        model="fake",
    )
    scored = LLMResponse(content=json.dumps({"score": score, "reason": reason}), model="fake")
    return [steps, scored]


def test_normalize_for_judging_strips_key_order_and_whitespace():
    a = normalize_for_judging({"b": 1, "a": 2})
    b = normalize_for_judging({"a": 2, "b": 1})
    assert a == b
    assert " " not in a


@pytest.mark.asyncio
async def test_score_case_deterministic_anchors_run_before_groundedness():
    pytest.importorskip("deepeval")
    from agent_eval_harness.llm.fake_client import FakeLLMClient

    contract = _contract(fallback_literal={"notes": "Audit could not be completed."})
    actual_output = {"overall_confidence": "high", "notes": "looks good"}
    llm_client = FakeLLMClient(_geval_responses(10))

    results = await score_case(
        "auditor", contract, {"A": {"confidence": "high"}}, None, actual_output, llm_client,
    )

    names = [r.metric_name for r in results]
    assert names[0] == "assertion.schema_valid"
    assert names[1] == "assertion.enum_membership"
    assert names[2] == "assertion.fallback_sentinel"
    assert "groundedness" in names[3]
    assert results[0].passed is True  # schema-valid
    assert results[1].passed is None  # no schema_enum_values harvested -> no-op
    assert results[2].passed is True  # not a match for the fallback literal
    assert results[3].passed is True  # groundedness 1.0 >= 0.5
    assert len(results) == 4  # no gold_dependent_fields requested


@pytest.mark.asyncio
async def test_score_case_flags_fallback_hit_and_schema_violation():
    pytest.importorskip("deepeval")
    from agent_eval_harness.llm.fake_client import FakeLLMClient

    contract = _contract(fallback_literal={"notes": "Audit could not be completed."})
    actual_output = {"notes": "Audit could not be completed."}  # missing required + fallback match
    llm_client = FakeLLMClient(_geval_responses(10))

    results = await score_case("auditor", contract, {}, None, actual_output, llm_client)

    assert results[0].passed is False  # missing overall_confidence
    assert results[2].passed is False  # matches the fallback literal


@pytest.mark.asyncio
async def test_score_case_order_short_circuits_gold_dependent_on_groundedness_failure():
    pytest.importorskip("deepeval")
    from agent_eval_harness.llm.fake_client import FakeLLMClient

    contract = _contract()
    actual_output = {"overall_confidence": "high", "notes": "invented fact not in input"}
    gold = {"purpose": "this repo does X"}
    llm_client = FakeLLMClient(_geval_responses(2, reason="fabricated"))

    results = await score_case(
        "auditor", contract, {}, gold, actual_output, llm_client,
        gold_dependent_fields=["purpose"],
    )

    assert len(results) == 4  # gold_dependent step skipped
    assert results[3].passed is False
    assert len(llm_client.calls) == 2  # only the groundedness GEval's 2 schema-constrained calls


@pytest.mark.asyncio
async def test_score_case_runs_gold_dependent_field_when_groundedness_passes():
    pytest.importorskip("deepeval")
    from agent_eval_harness.llm.fake_client import FakeLLMClient

    contract = _contract()
    actual_output = {"overall_confidence": "high", "notes": "fine", "purpose": "does X"}
    gold = {"purpose": "does X well"}
    llm_client = FakeLLMClient(
        _geval_responses(10) + _geval_responses(9, reason="close enough")
    )

    results = await score_case(
        "auditor", contract, {}, gold, actual_output, llm_client,
        gold_dependent_fields=["purpose"],
    )

    assert len(results) == 5
    assert "gold_dependent.purpose" in results[4].metric_name
    assert results[4].passed is True
    assert len(llm_client.calls) == 4


def test_pick_judge_provider_prefers_different_kind():
    class P:
        def __init__(self, provider_id, kind):
            self.provider_id = provider_id
            self.kind = kind

    generator = P("gen", "openai")
    available = [P("gen", "openai"), P("other", "anthropic")]

    chosen = pick_judge_provider(generator, available)
    assert chosen.provider_id == "other"


def test_pick_judge_provider_warns_and_falls_back_when_no_other_kind_configured():
    class P:
        def __init__(self, provider_id, kind):
            self.provider_id = provider_id
            self.kind = kind

    generator = P("gen", "openai")
    available = [P("gen", "openai")]

    chosen = pick_judge_provider(generator, available)
    assert chosen.provider_id == "gen"


def test_pick_judge_provider_none_when_nothing_configured():
    class P:
        provider_id = "gen"
        kind = "openai"

    assert pick_judge_provider(P(), []) is None
