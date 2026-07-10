import pytest

from agent_eval_harness.datasets.generators.decomposition_gold import generate
from agent_eval_harness.llm.client import LLMResponse
from agent_eval_harness.llm.fake_client import FakeLLMClient


@pytest.mark.asyncio
async def test_decomposition_gold_splits(tmp_path):
    # Create two temporary system maps with different constraints
    map_path_2 = tmp_path / "system_map_2.yaml"
    map_path_2.write_text("""
target_system_id: test_system
discrepancies: []
components:
  - id: planner
    role: orchestrator
    entry_point: "dummy:entry"
    constraints:
      - name: max_items_per_call
        value: 2
        source: "test"
""", encoding="utf-8")

    map_path_3 = tmp_path / "system_map_3.yaml"
    map_path_3.write_text("""
target_system_id: test_system
discrepancies: []
components:
  - id: planner
    role: orchestrator
    entry_point: "dummy:entry"
    constraints:
      - name: max_items_per_call
        value: 3
        source: "test"
""", encoding="utf-8")

    config_2 = {
        "dataset_name": "test_decomp_2",
        "system_map_path": str(map_path_2),
        "component_id": "planner",
        "count": 3
    }
    
    config_3 = {
        "dataset_name": "test_decomp_3",
        "system_map_path": str(map_path_3),
        "component_id": "planner",
        "count": 3
    }

    # count=3 splits into 3 categories (clean, rambling, over_limit), 1 case each via 3 prompts.
    # over_limit's intent count (5) must exceed both maps' max_items_per_call limit to test the split.
    mock_clean = '[{"query": "Do X and Y", "intents": ["I1", "I2"]}]'
    mock_rambling = '[{"query": "Please do X", "intents": ["I1"]}]'
    mock_over_limit = '[{"query": "Do 1 2 3 4 5", "intents": ["I1", "I2", "I3", "I4", "I5"]}]'

    fake_client_2 = FakeLLMClient([
        LLMResponse(content=mock_clean, model="fake"),
        LLMResponse(content=mock_rambling, model="fake"),
        LLMResponse(content=mock_over_limit, model="fake")
    ])

    fake_client_3 = FakeLLMClient([
        LLMResponse(content=mock_clean, model="fake"),
        LLMResponse(content=mock_rambling, model="fake"),
        LLMResponse(content=mock_over_limit, model="fake")
    ])

    # Run for map with limit = 2
    cases_2 = await generate(config_2, llm_client=fake_client_2)
    assert len(cases_2) == 3
    
    # Over limit cases are tagged with category="over_limit" or similar
    over_limit_case_2 = [c for c in cases_2 if c.labels.get("category") == "over_limit"][0]
    assert len(over_limit_case_2.expected["intents"]) == 5
    # split size for limit 2 should be: [[I1, I2], [I3, I4], [I5]]
    assert over_limit_case_2.expected["call_split"] == [["I1", "I2"], ["I3", "I4"], ["I5"]]

    # Run for map with limit = 3
    cases_3 = await generate(config_3, llm_client=fake_client_3)
    assert len(cases_3) == 3
    over_limit_case_3 = [c for c in cases_3 if c.labels.get("category") == "over_limit"][0]
    assert len(over_limit_case_3.expected["intents"]) == 5
    # split size for limit 3 should be: [[I1, I2, I3], [I4, I5]]
    assert over_limit_case_3.expected["call_split"] == [["I1", "I2", "I3"], ["I4", "I5"]]


@pytest.mark.asyncio
async def test_decomposition_gold_missing_constraint_skips_over_limit(tmp_path):
    """Component with no max_items_per_call constraint: no raise, no over_limit category."""
    map_path = tmp_path / "system_map_no_limit.yaml"
    map_path.write_text("""
target_system_id: test_system
discrepancies: []
components:
  - id: planner
    role: orchestrator
    entry_point: "dummy:entry"
    constraints: []
""", encoding="utf-8")

    config = {
        "dataset_name": "test_decomp_no_limit",
        "system_map_path": str(map_path),
        "component_id": "planner",
        "count": 4,
    }

    mock_clean = '[{"query": "Do X and Y", "intents": ["I1", "I2"]}]'
    mock_rambling = '[{"query": "Please do X", "intents": ["I1"]}]'
    fake_client = FakeLLMClient([
        LLMResponse(content=mock_clean, model="fake"),
        LLMResponse(content=mock_rambling, model="fake"),
    ])

    cases = await generate(config, llm_client=fake_client)
    categories = {c.labels.get("category") for c in cases}
    assert categories == {"clean", "rambling"}
    assert "over_limit" not in categories
    assert all("call_split" not in c.expected for c in cases)
