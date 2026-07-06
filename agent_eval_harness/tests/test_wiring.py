import pytest
from unittest.mock import AsyncMock
from agent_eval_harness.discovery.wiring import (
    _detect_haystack,
    _detect_langgraph,
    _detect_langchain_lcel,
    _detect_via_llm,
    detect_wiring_block,
)

def test_detect_haystack():
    haystack_code = """
pipeline = Pipeline()
pipeline.add_component("prompt_builder", PromptBuilder(template="Hello"))
pipeline.add_component("llm", OpenAIGenerator(model="gpt-4"))
pipeline.connect("prompt_builder.prompt", "llm.prompt")
"""
    file_contents = {"main.py": haystack_code}
    wiring = _detect_haystack(file_contents)
    assert wiring is not None
    assert wiring.framework == "haystack"
    assert len(wiring.nodes) == 2
    
    aliases = {n.alias for n in wiring.nodes}
    assert "prompt_builder" in aliases
    assert "llm" in aliases
    
    classes = {n.class_name for n in wiring.nodes}
    assert "PromptBuilder" in classes
    assert "OpenAIGenerator" in classes
    
    assert len(wiring.edges) == 1
    assert wiring.edges[0].src == "prompt_builder"
    assert wiring.edges[0].dst == "llm"


def test_detect_langgraph():
    langgraph_code = """
workflow = StateGraph(State)
workflow.add_node("agent", run_agent)
workflow.add_node("action", ToolExecutor())
workflow.add_edge("agent", "action")
workflow.add_edge("action", "agent")
"""
    file_contents = {"graph.py": langgraph_code}
    wiring = _detect_langgraph(file_contents)
    assert wiring is not None
    assert wiring.framework == "langgraph"
    assert len(wiring.nodes) == 2
    
    aliases = {n.alias for n in wiring.nodes}
    assert "agent" in aliases
    assert "action" in aliases
    
    assert len(wiring.edges) == 2
    assert {"src": wiring.edges[0].src, "dst": wiring.edges[0].dst} == {"src": "agent", "dst": "action"}
    assert {"src": wiring.edges[1].src, "dst": wiring.edges[1].dst} == {"src": "action", "dst": "agent"}


def test_detect_langchain_lcel():
    lcel_code = """
chain = prompt_template | chat_model | output_parser
"""
    file_contents = {"chain.py": lcel_code}
    wiring = _detect_langchain_lcel(file_contents)
    assert wiring is not None
    assert wiring.framework == "langchain"
    assert len(wiring.nodes) == 3
    
    aliases = [n.alias for n in wiring.nodes]
    # Expect prompt_template, chat_model, output_parser
    assert "prompt_template" in aliases
    assert "chat_model" in aliases
    assert "output_parser" in aliases
    
    assert len(wiring.edges) == 2
    assert wiring.edges[0].src == "prompt_template"
    assert wiring.edges[0].dst == "chat_model"
    assert wiring.edges[1].src == "chat_model"
    assert wiring.edges[1].dst == "output_parser"


@pytest.mark.anyio
async def test_detect_via_llm():
    custom_code = """
def run_my_custom_chain():
    x = fetch_prompt()
    y = call_llm(x)
    return parse(y)
"""
    file_contents = {"custom.py": custom_code}
    
    mock_response = AsyncMock()
    mock_response.content = """
{
  "framework": "llm_inferred",
  "nodes": [
    {"alias": "fetch", "class_name": "fetch_prompt", "source_hint_file": "custom.py"},
    {"alias": "llm", "class_name": "call_llm", "source_hint_file": "custom.py"}
  ],
  "edges": [
    {"src": "fetch", "dst": "llm"}
  ]
}
"""
    mock_llm_client = AsyncMock()
    mock_llm_client.complete.return_value = mock_response
    
    wiring = await _detect_via_llm(file_contents, mock_llm_client)
    assert wiring is not None
    assert wiring.framework == "llm_inferred"
    assert wiring.source == "llm_fallback"
    assert len(wiring.nodes) == 2
    assert wiring.nodes[0].alias == "fetch"
    assert wiring.nodes[1].alias == "llm"
    assert len(wiring.edges) == 1
    assert wiring.edges[0].src == "fetch"
    assert wiring.edges[0].dst == "llm"
