"""DeepEvalLLMAdapter's schema-constrained generation must make a REAL LLM call
(json_mode + the schema's JSON Schema in-prompt) and return an actual validated
schema instance — not a hardcoded mock value. This is the fix for a real bug: the
old adapter short-circuited every schema call with a constant canned value, which
silently faked every deepeval G-Eval score and every qa_testset synthesis case."""
from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from agent_eval_harness.llm.client import LLMResponse
from agent_eval_harness.llm.fake_client import FakeLLMClient


class _Verdict(BaseModel):
    label: str
    confidence: float


def _adapter(client: FakeLLMClient):
    pytest.importorskip("deepeval")
    from agent_eval_harness.llm.deepeval_adapter import make_deepeval_llm_adapter

    return make_deepeval_llm_adapter(client)


async def test_a_generate_without_schema_returns_raw_text() -> None:
    client = FakeLLMClient(LLMResponse(content="plain answer", model="fake"))
    adapter = _adapter(client)

    result = await adapter.a_generate("say something")

    assert result == "plain answer"
    assert client.calls[-1][-1].content == "say something"


async def test_a_generate_with_schema_calls_real_llm_and_validates() -> None:
    client = FakeLLMClient(
        LLMResponse(content=json.dumps({"label": "positive", "confidence": 0.87}), model="fake")
    )
    adapter = _adapter(client)

    result = await adapter.a_generate("classify this", schema=_Verdict)

    assert isinstance(result, _Verdict)
    assert result.label == "positive"
    assert result.confidence == 0.87
    # the schema's JSON Schema must actually be in the prompt sent to the LLM
    sent_prompt = client.calls[-1][-1].content
    assert "label" in sent_prompt and "confidence" in sent_prompt


async def test_a_generate_with_schema_retries_once_on_bad_json() -> None:
    client = FakeLLMClient(
        [
            LLMResponse(content="not json at all", model="fake"),
            LLMResponse(content=json.dumps({"label": "neutral", "confidence": 0.5}), model="fake"),
        ]
    )
    adapter = _adapter(client)

    result = await adapter.a_generate("classify this", schema=_Verdict)

    assert isinstance(result, _Verdict)
    assert result.label == "neutral"
    assert len(client.calls) == 2


async def test_a_generate_with_schema_raises_after_repeated_failure() -> None:
    client = FakeLLMClient(LLMResponse(content="still not json", model="fake"))
    adapter = _adapter(client)

    with pytest.raises(ValueError, match="_Verdict"):
        await adapter.a_generate("classify this", schema=_Verdict)


async def test_a_generate_with_schema_rejects_json_missing_required_fields() -> None:
    client = FakeLLMClient(
        [
            LLMResponse(content=json.dumps({"label": "positive"}), model="fake"),  # missing confidence
            LLMResponse(content=json.dumps({"label": "positive", "confidence": 0.9}), model="fake"),
        ]
    )
    adapter = _adapter(client)

    result = await adapter.a_generate("classify this", schema=_Verdict)

    assert result.confidence == 0.9
    assert len(client.calls) == 2


def test_generate_sync_delegates_to_a_generate_with_schema() -> None:
    client = FakeLLMClient(
        LLMResponse(content=json.dumps({"label": "negative", "confidence": 0.1}), model="fake")
    )
    adapter = _adapter(client)

    result = adapter.generate("classify this", schema=_Verdict)

    assert isinstance(result, _Verdict)
    assert result.label == "negative"
