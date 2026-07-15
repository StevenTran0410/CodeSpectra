from __future__ import annotations

import json

import pytest

from agent_eval_harness.judging.case_judge import (
    compute_field_precision_recall,
    judge_case_semantic_match,
    summarize_agent_judgments,
)
from agent_eval_harness.llm.fake_client import FakeLLMClient
from agent_eval_harness.llm.client import LLMResponse


@pytest.mark.asyncio
async def test_judge_case_semantic_match_parses_valid_response() -> None:
    client = FakeLLMClient(LLMResponse(
        content=json.dumps({"score": 0.85, "notes": "domain and purpose match; runtime_type differs"}),
        model="gpt-5.4-mini",
        prompt_tokens=200,
        completion_tokens=30,
    ))

    result = await judge_case_semantic_match(
        client, {"domain": "mobile app"}, {"domain": "workforce management"}
    )

    assert result["score"] == 0.85
    assert "runtime_type" in result["notes"]
    assert result["model"] == "gpt-5.4-mini"
    assert result["tokens"] == 230


@pytest.mark.asyncio
async def test_judge_case_semantic_match_clamps_out_of_range_score() -> None:
    client = FakeLLMClient(LLMResponse(
        content=json.dumps({"score": 1.7, "notes": "over-confident judge"}), model="m",
    ))

    result = await judge_case_semantic_match(client, {}, {})

    assert result["score"] == 1.0


@pytest.mark.asyncio
async def test_judge_case_semantic_match_handles_malformed_json() -> None:
    client = FakeLLMClient(LLMResponse(content="not json at all", model="m"))

    result = await judge_case_semantic_match(client, {"a": 1}, {"a": 1})

    assert result["score"] == 0.0
    assert "unparseable" not in result["notes"]  # exact wording isn't load-bearing, just no crash
    assert isinstance(result["notes"], str)


def test_precision_recall_on_matching_list_field() -> None:
    result = {"tech_stack": ["TypeScript", "react-native", "Expo"]}
    expected = {"tech_stack": ["TypeScript", "React Native", "Expo", "Jest"]}

    scores = compute_field_precision_recall(result, expected)

    assert scores["tech_stack"]["precision"] == pytest.approx(2 / 3)  # TypeScript, Expo matched (case-insensitive)
    assert scores["tech_stack"]["recall"] == pytest.approx(2 / 4)


def test_precision_recall_skips_non_list_fields() -> None:
    result = {"domain": "mobile app", "confidence": "high"}
    expected = {"domain": "mobile app", "confidence": "high"}

    scores = compute_field_precision_recall(result, expected)

    assert scores == {}


def test_precision_recall_skips_field_missing_on_result_side() -> None:
    result = {"tech_stack": ["TypeScript"]}
    expected = {"tech_stack": ["TypeScript"], "evidence_files": ["README.md"]}

    scores = compute_field_precision_recall(result, expected)

    assert "evidence_files" not in scores
    assert "tech_stack" in scores


def test_precision_recall_perfect_match() -> None:
    result = {"evidence_files": ["a.py", "b.py"]}
    expected = {"evidence_files": ["a.py", "b.py"]}

    scores = compute_field_precision_recall(result, expected)

    assert scores["evidence_files"]["precision"] == 1.0
    assert scores["evidence_files"]["recall"] == 1.0


@pytest.mark.asyncio
async def test_judge_uses_llm_index_pairs_to_catch_synonyms() -> None:
    # "Postgres" vs "PostgreSQL" would NOT match on exact string overlap — the judge's
    # index-pair matching is the whole point of this path.
    client = FakeLLMClient(LLMResponse(content=json.dumps({
        "score": 0.9,
        "notes": "tech stack matches modulo naming",
        "field_matches": {"tech_stack": [[0, 1], [1, 0]]},
    }), model="m"))

    result = await judge_case_semantic_match(
        client,
        {"tech_stack": ["Postgres", "React"]},
        {"tech_stack": ["React Native", "PostgreSQL"]},
    )

    assert result["field_matches"]["tech_stack"]["precision"] == pytest.approx(2 / 2)
    assert result["field_matches"]["tech_stack"]["recall"] == pytest.approx(2 / 2)
    assert result["field_matches"]["tech_stack"]["matched_via"] == "llm"


@pytest.mark.asyncio
async def test_judge_drops_out_of_range_and_duplicate_index_pairs() -> None:
    # index 5 is out of range, and result-index 0 is reused for two pairs — only the first
    # legitimate, non-reused pair should ever be counted. Never trust the model's own count.
    client = FakeLLMClient(LLMResponse(content=json.dumps({
        "score": 0.5,
        "notes": "partial",
        "field_matches": {"tech_stack": [[0, 0], [1, 5], [2, 0]]},
    }), model="m"))

    result = await judge_case_semantic_match(
        client,
        {"tech_stack": ["TypeScript"]},
        {"tech_stack": ["TypeScript", "React", "Jest"]},
    )

    pr = result["field_matches"]["tech_stack"]
    assert pr["precision"] == pytest.approx(1 / 1)  # only [0,0] survives validation
    assert pr["recall"] == pytest.approx(1 / 3)


@pytest.mark.asyncio
async def test_judge_falls_back_to_exact_match_when_field_omitted() -> None:
    # judge answers score/notes but forgets field_matches entirely for tech_stack.
    client = FakeLLMClient(LLMResponse(content=json.dumps({
        "score": 0.7, "notes": "ok", "field_matches": {},
    }), model="m"))

    result = await judge_case_semantic_match(
        client,
        {"tech_stack": ["TypeScript", "Jest"]},
        {"tech_stack": ["typescript", "Jest", "Expo"]},
    )

    pr = result["field_matches"]["tech_stack"]
    assert pr["matched_via"] == "exact"
    assert pr["precision"] == pytest.approx(2 / 2)  # typescript, jest matched case-insensitively
    assert pr["recall"] == pytest.approx(2 / 3)


@pytest.mark.asyncio
async def test_judge_falls_back_to_exact_match_on_malformed_json() -> None:
    client = FakeLLMClient(LLMResponse(content="not json at all", model="m"))

    result = await judge_case_semantic_match(
        client,
        {"tech_stack": ["TypeScript"]},
        {"tech_stack": ["TypeScript"]},
    )

    pr = result["field_matches"]["tech_stack"]
    assert pr["matched_via"] == "exact"
    assert pr["precision"] == 1.0
    assert pr["recall"] == 1.0


@pytest.mark.asyncio
async def test_summarize_agent_judgments_parses_valid_response() -> None:
    client = FakeLLMClient(LLMResponse(
        content=json.dumps({"insight": "Consistently strong on frameworks, weak on evidence_files."}),
        model="deepseek-v4-flash", prompt_tokens=300, completion_tokens=40,
    ))
    case_summaries = [
        {"score": 0.9, "notes": "close match"},
        {"score": 0.2, "notes": "wrong domain entirely"},
    ]

    result = await summarize_agent_judgments(client, case_summaries)

    assert result["insight"] == "Consistently strong on frameworks, weak on evidence_files."
    assert result["model"] == "deepseek-v4-flash"
    assert result["tokens"] == 340
    # both cases' score+notes must actually reach the model, not just the count
    sent_prompt = client.calls[0][1].content
    assert "0.90" in sent_prompt and "close match" in sent_prompt
    assert "0.20" in sent_prompt and "wrong domain entirely" in sent_prompt


@pytest.mark.asyncio
async def test_summarize_agent_judgments_handles_malformed_json() -> None:
    client = FakeLLMClient(LLMResponse(content="not json at all", model="m"))

    result = await summarize_agent_judgments(client, [{"score": 0.5, "notes": "n/a"}])

    assert "not valid JSON" in result["insight"]
