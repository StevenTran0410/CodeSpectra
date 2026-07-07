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


def test_detect_haystack_resolves_wrapped_component_to_its_real_file():
    """Factory/wrapper pattern: nested class must resolve to its own file, not the shared wrapper's."""
    orchestrator_code = """
from .agents.agent_conventions import ConventionsAgent
from .agents.agent_risk import RiskAgent

pipeline = Pipeline()
pipeline.add_component("conventions", _SectionAgentComponent(agent=ConventionsAgent()))
pipeline.add_component("risk", _SectionAgentComponent(agent=RiskAgent()))
pipeline.connect("conventions.output", "risk.input")
"""
    file_contents = {
        "backend/domain/analysis/agent_pipeline.py": orchestrator_code,
        "backend/domain/analysis/agents/agent_conventions.py": "class ConventionsAgent:\n    pass\n",
        "backend/domain/analysis/agents/agent_risk.py": "class RiskAgent:\n    pass\n",
    }
    wiring = _detect_haystack(file_contents)
    assert wiring is not None

    by_alias = {n.alias: n for n in wiring.nodes}
    assert by_alias["conventions"].class_name == "ConventionsAgent"
    assert by_alias["conventions"].source_hint_file == "backend/domain/analysis/agents/agent_conventions.py"
    assert by_alias["risk"].class_name == "RiskAgent"
    assert by_alias["risk"].source_hint_file == "backend/domain/analysis/agents/agent_risk.py"

    # The two components must resolve to DIFFERENT files — this is the actual bug:
    # both used to collapse onto the orchestrator file itself.
    assert by_alias["conventions"].source_hint_file != by_alias["risk"].source_hint_file


def test_detect_haystack_resolves_self_attr_closure_to_its_real_file():
    """Agent built in __init__, only referenced via closure/lambda at add_component() — must still resolve to its own file."""
    orchestrator_code = """
class Pipeline:
    def __init__(self):
        self._conventions = ConventionsAgent(provider, retrieval)
        self._risk = RiskAgent(provider, retrieval)

    def build(self):
        pipeline = AsyncPipeline()
        pipeline.add_component(
            "conventions",
            _SectionAgentComponent("A", lambda c, d: self._conventions.run(c), on_done),
        )
        pipeline.add_component(
            "risk",
            _SectionAgentComponent("B", lambda c, d: self._risk.run(c), on_done),
        )
        pipeline.connect("conventions.output", "risk.input")
"""
    file_contents = {
        "backend/domain/analysis/agent_pipeline.py": orchestrator_code,
        "backend/domain/analysis/agents/agent_conventions.py": "class ConventionsAgent:\n    pass\n",
        "backend/domain/analysis/agents/agent_risk.py": "class RiskAgent:\n    pass\n",
    }
    wiring = _detect_haystack(file_contents)
    assert wiring is not None

    by_alias = {n.alias: n for n in wiring.nodes}
    assert by_alias["conventions"].class_name == "ConventionsAgent"
    assert by_alias["conventions"].source_hint_file == "backend/domain/analysis/agents/agent_conventions.py"
    assert by_alias["risk"].class_name == "RiskAgent"
    assert by_alias["risk"].source_hint_file == "backend/domain/analysis/agents/agent_risk.py"
    assert by_alias["conventions"].source_hint_file != by_alias["risk"].source_hint_file


def test_detect_haystack_falls_back_to_current_file_when_class_unresolvable():
    """Unresolvable nested class must fall back to the calling file, never raise or leave source_hint_file empty."""
    code = """
pipeline.add_component("writer", _SectionAgentComponent(agent=UnknownAgent()))
"""
    file_contents = {"main.py": code}
    wiring = _detect_haystack(file_contents)
    assert wiring is not None
    node = wiring.nodes[0]
    assert node.class_name == "UnknownAgent"
    assert node.source_hint_file == "main.py"


def test_detect_haystack_ignores_plain_helper_call_as_constructor_argument():
    """A plain helper-call constructor arg must not be mistaken for a wrapped component (regression guard)."""
    code = """
pipeline.add_component("retriever", RetrieverComponent(_load_corpus()))
"""
    file_contents = {"pipeline.py": code}
    wiring = _detect_haystack(file_contents)
    assert wiring is not None
    node = wiring.nodes[0]
    assert node.class_name == "RetrieverComponent"
    assert node.source_hint_file == "pipeline.py"


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


def test_detect_langchain_lcel_ignores_pep604_type_unions():
    """`str | None` and similar type-annotation unions must not be mistaken for
    an LCEL pipe chain — same BinOp/BitOr AST shape, different meaning entirely."""
    type_union_code = """
from dataclasses import dataclass, field

@dataclass
class StaticRiskConfig:
    threshold: str | None = None
    weight: float | None = None
    flags: set | None = None
    enabled: bool | None = None
    extra: dict | None = None
"""
    file_contents = {"static_risk.py": type_union_code}
    wiring = _detect_langchain_lcel(file_contents)
    assert wiring is None, f"type-union annotations must not produce a fake wiring block, got: {wiring}"


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


@pytest.mark.anyio
async def test_detect_via_llm_propagates_rate_limit():
    from agent_eval_harness.llm.client import RateLimitExceeded
    mock_llm_client = AsyncMock()
    mock_llm_client.complete.side_effect = RateLimitExceeded("prov", "model")
    
    with pytest.raises(RateLimitExceeded):
        await _detect_via_llm({"custom.py": "pass"}, mock_llm_client)
