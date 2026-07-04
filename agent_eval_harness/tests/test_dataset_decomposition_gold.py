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

    # Configs
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

    # Fake LLM client to return JSON lists of queries and their intents
    # Since count=3, it splits into 3 categories: clean, rambling, over_limit. So it does 3 prompts.
    # Each category gets count // 3 = 1 case.
    # For clean: returns 1 case with 2 intents: ["I1", "I2"]
    # For rambling: returns 1 case with 1 intent: ["I1"]
    # For over_limit: returns 1 case with 5 intents (limit 2+1=3 for map_2,
    # limit 3+1=4 for map_3. Let's return 5 intents to be safe).
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
