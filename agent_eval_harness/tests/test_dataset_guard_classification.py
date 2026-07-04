import pytest

from agent_eval_harness.datasets.generators.guard_classification import generate
from agent_eval_harness.llm.client import LLMResponse
from agent_eval_harness.llm.fake_client import FakeLLMClient


@pytest.mark.asyncio
async def test_guard_classification_mechanical():
    # Only mechanical categories
    config = {
        "dataset_name": "test_mech_v1",
        "categories": [
            {"name": "too_short", "kind": "mechanical", "count": 25},
            {"name": "gibberish", "kind": "mechanical", "count": 25},
            # We must make sure valid cases make up >= 40% of the total
            # Wait, too_short (reject) + gibberish (reject) = 50 rejects.
            # To have >=40% valid, we need at least 34 valid cases (out of 84).
            # Let's add a mechanical/valid category if possible, or just valid:
            {"name": "valid", "kind": "mechanical", "count": 35}
        ]
    }
    # Mechanical generation needs no LLM client
    cases = await generate(config, llm_client=None, seed=42)
    assert len(cases) == 85
    
    too_short_cases = [c for c in cases if c.labels.get("category") == "too_short"]
    assert len(too_short_cases) == 25
    for c in too_short_cases:
        assert c.expected == {"verdict": "reject", "category": "too_short"}
        assert c.provenance == "synthetic"

    valid_cases = [c for c in cases if c.labels.get("category") == "valid"]
    assert len(valid_cases) == 35
    for c in valid_cases:
        assert c.expected == {"verdict": "pass"}
        assert c.provenance == "synthetic"

@pytest.mark.asyncio
async def test_guard_classification_validation_failure():
    # If category count < 25, should raise ValueError
    config = {
        "dataset_name": "test_fail_v1",
        "categories": [
            {"name": "too_short", "kind": "mechanical", "count": 24}
        ]
    }
    with pytest.raises(ValueError, match="less than the required minimum of 25"):
        await generate(config, llm_client=None)

    # If valid cases ratio < 40%, should raise ValueError
    config_ratio = {
        "dataset_name": "test_ratio_v1",
        "categories": [
            {"name": "too_short", "kind": "mechanical", "count": 30},
            {"name": "gibberish", "kind": "mechanical", "count": 30},
            {"name": "valid", "kind": "mechanical", "count": 25}  # 25 / 85 = 29.4% < 40%
        ]
    }
    with pytest.raises(ValueError, match="does not meet the requirement of having >= 40% valid"):
        await generate(config_ratio, llm_client=None)

@pytest.mark.asyncio
async def test_guard_classification_semantic():
    config = {
        "dataset_name": "test_sem_v1",
        "categories": [
            {
                "name": "off_topic",
                "kind": "semantic",
                "count": 30,
                "rubric": "not related to company",
            },
            {"name": "borderline_valid", "kind": "semantic", "count": 25}
        ]
    }
    
    # Fake LLM client to return custom queries
    mock_off_topic = '["where is the sun", "who is the President"]'
    mock_borderline = (
        '["is code review required for vacation requests", "how to submit sick leave"]'
    )
    
    fake_client = FakeLLMClient([
        LLMResponse(content=mock_off_topic, model="fake-model"),
        LLMResponse(content=mock_borderline, model="fake-model")
    ])
    
    cases = await generate(config, llm_client=fake_client)
    # We requested 30 + 25 = 55 cases.
    # Wait, valid ratio: borderline_valid count is 25. Total is 55. 25/55 = 45.4% >= 40%
    assert len(cases) == 55
    
    off_topic_cases = [c for c in cases if c.labels.get("category") == "off_topic"]
    # If the response list was smaller than count (which our mock lists of size 2 are),
    # the generator pads it, so we expect exactly 30 cases.
    assert len(off_topic_cases) == 30
    for c in off_topic_cases:
        assert c.expected == {"verdict": "reject", "category": "off_topic"}
        assert c.provenance == "synthetic"
