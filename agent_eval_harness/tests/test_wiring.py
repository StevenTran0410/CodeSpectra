import ast

import pytest
from unittest.mock import AsyncMock
from agent_eval_harness.discovery.wiring import (
    _build_enclosing_class_map,
    _build_stategraph_var_names,
    _const_str_or_sentinel,
    _decorator_effective_name,
    _detect_haystack,
    _detect_langgraph,
    _detect_langchain_lcel,
    _detect_via_llm,
    _imports_langgraph,
    detect_wiring_block,
)


# ---- CS-312 Slice 1: low-level helper truth tables -----------------------------------------

def test_build_enclosing_class_map_owner_and_free():
    tree = ast.parse("class Agent:\n    def node(self):\n        helper()\ndef free():\n    other()\n")
    helper = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "helper")
    other = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "other")
    m = _build_enclosing_class_map(tree)
    assert m[id(helper)] == "Agent"
    assert id(other) not in m


def test_build_stategraph_var_names_local_and_self():
    tree = ast.parse("g = StateGraph(S)\nself.flow = StateGraph(S)\nother = Foo()\n")
    assert _build_stategraph_var_names(tree) == {"g", "self.flow"}


def test_imports_langgraph_true_and_false():
    assert _imports_langgraph(ast.parse("from langgraph.graph import StateGraph\n")) is True
    assert _imports_langgraph(ast.parse("import langgraph.func\n")) is True
    assert _imports_langgraph(ast.parse("import os\n")) is False


def test_decorator_effective_name_bare_and_call_wrapped():
    tree = ast.parse("@task\ndef a(): ...\n@entrypoint(checkpointer=x)\ndef b(): ...\n")
    decs = {f.name: f.decorator_list[0] for f in tree.body}
    assert _decorator_effective_name(decs["a"]) == "task"
    assert _decorator_effective_name(decs["b"]) == "entrypoint"


def test_const_str_or_sentinel_maps_start_end():
    assert _const_str_or_sentinel(ast.parse("'x'", mode="eval").body) == "x"
    assert _const_str_or_sentinel(ast.parse("START", mode="eval").body) == "__start__"
    assert _const_str_or_sentinel(ast.parse("END", mode="eval").body) == "__end__"
    assert _const_str_or_sentinel(ast.parse("other", mode="eval").body) is None


# ---- CS-312 Slice 2: extended _detect_langgraph -------------------------------------------

def test_detect_langgraph_conditional_edges_and_sentinels():
    code = (
        "graph = StateGraph(S)\n"
        "graph.add_edge(START, 'a')\n"
        "graph.add_conditional_edges('a', route, {'b': 'b', 'c': 'c'})\n"
        "graph.add_edge('c', END)\n"
    )
    wiring = _detect_langgraph({"g.py": code})
    assert wiring is not None
    pairs = {(e.src, e.dst) for e in wiring.edges}
    assert {("__start__", "a"), ("a", "b"), ("a", "c"), ("c", "__end__")} <= pairs


def test_detect_langgraph_bound_method_owner():
    code = (
        "class Agent:\n"
        "    def build(self):\n"
        "        g = StateGraph(S)\n"
        "        g.add_node('x', self.foo)\n"
        "    def foo(self, state):\n        return state\n"
    )
    wiring = _detect_langgraph({"g.py": code})
    node = next(n for n in wiring.nodes if n.alias == "x")
    assert node.entry_kind == "bound_method"
    assert node.owner_class == "Agent"
    assert node.callee_name == "foo"


def test_detect_langgraph_receiver_guard_rejects_unrelated_add_node():
    code = (
        "class Foo:\n"
        "    def add_node(self, a, b):\n        pass\n"
        "f = Foo()\n"
        "f.add_node('x', 'y')\n"
    )
    assert _detect_langgraph({"g.py": code}) is None


def test_detect_langgraph_create_react_agent():
    code = (
        "from langgraph.prebuilt import create_react_agent\n"
        "agent = create_react_agent(model=m, tools=[tool_a, tool_b])\n"
    )
    wiring = _detect_langgraph({"g.py": code})
    aliases = {n.alias for n in wiring.nodes}
    assert "agent" in aliases
    pairs = {(e.src, e.dst) for e in wiring.edges}
    assert {("agent", "tool_a"), ("agent", "tool_b")} <= pairs


def test_detect_langgraph_task_entrypoint():
    code = (
        "from langgraph.func import task, entrypoint\n"
        "@task\ndef step(x):\n    return x\n"
        "@entrypoint()\ndef flow(x):\n    return step(x)\n"
    )
    wiring = _detect_langgraph({"g.py": code})
    aliases = {n.alias for n in wiring.nodes}
    assert {"step", "flow"} <= aliases
    pairs = {(e.src, e.dst) for e in wiring.edges}
    assert ("flow", "step") in pairs

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
    
    classes = {n.callee_name for n in wiring.nodes}
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
    assert by_alias["conventions"].callee_name == "ConventionsAgent"
    assert by_alias["conventions"].source_hint_file == "backend/domain/analysis/agents/agent_conventions.py"
    assert by_alias["risk"].callee_name == "RiskAgent"
    assert by_alias["risk"].source_hint_file == "backend/domain/analysis/agents/agent_risk.py"

    # The two components must resolve to DIFFERENT files (regression: both used to collapse onto the orchestrator file).
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
    assert by_alias["conventions"].callee_name == "ConventionsAgent"
    assert by_alias["conventions"].source_hint_file == "backend/domain/analysis/agents/agent_conventions.py"
    assert by_alias["risk"].callee_name == "RiskAgent"
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
    assert node.callee_name == "UnknownAgent"
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
    assert node.callee_name == "RetrieverComponent"
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
from langchain_core.runnables import RunnableSequence
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


def test_detect_langchain_lcel_ignores_pipe_without_langchain_import():
    """A plain `set(...) | set(...)` (or any bitwise-or) in a file that never imports
    langchain must NOT be mistaken for an LCEL pipe chain — the pipe idiom is only trusted
    when the file imports langchain (mirrors the factory-path gate)."""
    set_union_code = """
def merge(indeg, outdeg):
    seen = set(indeg.keys()) | set(outdeg.keys())
    return seen
"""
    assert _detect_langchain_lcel({"graph_service.py": set_union_code}) is None


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


def test_detect_langchain_lcel_ignores_numpy_dtype_union():
    """`np.float32 | np.int64` is a numpy dtype union, not an LCEL chain — Attribute operands whose
    attr is a known dtype must be rejected by the operand guard."""
    code = """
import numpy as np
mixed_dtype = np.float32 | np.int64
"""
    assert _detect_langchain_lcel({"dtypes.py": code}) is None


def test_detect_langchain_lcel_ignores_flag_enum_union():
    """`LogLevel.DEBUG | LogLevel.INFO` is a flag-enum OR, not an LCEL chain — ALL-CAPS Attribute
    operands must be rejected by the operand guard."""
    code = """
level = LogLevel.DEBUG | LogLevel.INFO
"""
    assert _detect_langchain_lcel({"flags.py": code}) is None


def test_detect_langchain_lcel_factory_idiom():
    """The dominant production LCEL idiom composes chains with factory functions (no `|`), inside a
    function body, linked by call arguments. The detector must map the full node+edge graph."""
    code = """
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

def get_rag_chain(model="gpt-4o-mini"):
    llm = ChatOpenAI(model=model)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain
"""
    wiring = _detect_langchain_lcel({"langchain_utils.py": code})
    assert wiring is not None
    assert wiring.framework == "langchain"

    callees = {n.callee_name for n in wiring.nodes}
    assert callees == {"retriever", "history_aware_retriever", "question_answer_chain", "rag_chain"}

    edges = {(e.src, e.dst) for e in wiring.edges}
    assert edges == {
        ("history_aware_retriever", "retriever"),
        ("rag_chain", "history_aware_retriever"),
        ("rag_chain", "question_answer_chain"),
    }


def test_detect_langchain_factory_idiom_import_gated():
    """The factory idiom is import-gated: the same call shapes without a langchain import must NOT
    produce a wiring block (keeps `.as_retriever()` from false-positiving on non-LangChain code)."""
    code = """
retriever = store.as_retriever(k=2)
chain = create_retrieval_chain(a, b)
"""
    assert _detect_langchain_lcel({"not_langchain.py": code}) is None


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


# ---- CS-317: mixed-framework union + per-node framework tag + back-compat -----------------

def test_detect_wiring_block_static_unions_haystack_and_langgraph():
    """A cluster mixing a Haystack pipeline and a LangGraph StateGraph must yield ONE union block
    containing BOTH frameworks' nodes, each node tagged with its producing framework — not one
    detector starving the other (first-hit-wins regression guard)."""
    from agent_eval_harness.discovery.wiring import detect_wiring_block_static
    haystack_file = (
        "from haystack import Pipeline\n"
        "p = Pipeline()\n"
        "p.add_component('embedder', TextEmbedder())\n"
        "p.add_component('retriever', Retriever())\n"
        "p.connect('embedder', 'retriever')\n"
    )
    langgraph_file = (
        "from langgraph.graph import StateGraph\n"
        "class Orchestrator:\n"
        "    def build(self):\n"
        "        g = StateGraph(dict)\n"
        "        g.add_node('plan', self._plan)\n"
        "        g.add_node('act', self._act)\n"
        "        g.add_edge('plan', 'act')\n"
        "    def _plan(self, s): ...\n"
        "    def _act(self, s): ...\n"
    )
    block = detect_wiring_block_static({"pipe.py": haystack_file, "graph.py": langgraph_file})
    assert block is not None
    frameworks = {n.framework for n in block.nodes}
    assert "haystack" in frameworks and "langgraph" in frameworks
    assert "haystack" in block.framework and "langgraph" in block.framework
    aliases = {n.alias for n in block.nodes}
    assert {"embedder", "retriever", "plan", "act"} <= aliases
    edge_pairs = {(e.src, e.dst) for e in block.edges}
    assert ("embedder", "retriever") in edge_pairs  # haystack edge survives
    assert ("plan", "act") in edge_pairs            # langgraph edge survives


def test_detect_wiring_block_static_single_framework_preserves_label():
    """A single-framework cluster returns its own framework label unchanged (no 'mixed'), so existing
    single-framework behavior and the Haystack golden invariant are untouched."""
    from agent_eval_harness.discovery.wiring import detect_wiring_block_static
    haystack_file = (
        "from haystack import Pipeline\n"
        "p = Pipeline()\n"
        "p.add_component('embedder', TextEmbedder())\n"
        "p.connect('embedder', 'embedder')\n"
    )
    block = detect_wiring_block_static({"pipe.py": haystack_file})
    assert block is not None
    assert block.framework == "haystack"
    assert all(n.framework == "haystack" for n in block.nodes)


def test_union_namespaces_cross_framework_alias_collision():
    """Two frameworks emitting the same alias must not collapse into one node; the later one is
    namespaced and its edges rewritten so topology still joins (union precision floor)."""
    from agent_eval_harness.discovery.wiring import union_wiring_blocks, WiringBlock, WiringNode, WiringEdge
    a = WiringBlock(
        nodes=[WiringNode(alias="agent", callee_name="A", source_hint_file="a.py"),
               WiringNode(alias="tail", callee_name="TailA", source_hint_file="a.py")],
        edges=[WiringEdge(src="agent", dst="tail")], framework="haystack", source="static",
    )
    b = WiringBlock(
        nodes=[WiringNode(alias="agent", callee_name="B", source_hint_file="b.py"),
               WiringNode(alias="head", callee_name="HeadB", source_hint_file="b.py")],
        edges=[WiringEdge(src="agent", dst="head")], framework="langgraph", source="static",
    )
    u = union_wiring_blocks([a, b])
    aliases = [n.alias for n in u.nodes]
    assert aliases.count("agent") == 1                 # haystack 'agent' kept
    assert "agent@langgraph" in aliases                # langgraph 'agent' namespaced
    edge_pairs = {(e.src, e.dst) for e in u.edges}
    assert ("agent", "tail") in edge_pairs             # haystack edge intact
    assert ("agent@langgraph", "head") in edge_pairs   # langgraph edge rewritten to namespaced alias


def test_wiringnode_backcompat_old_row_without_framework():
    """An old wiring_block_json row whose nodes predate the framework field must deserialize cleanly,
    defaulting framework to 'unknown' (no corruption / no KeyError)."""
    from agent_eval_harness.discovery.wiring import WiringBlock
    old_row = {
        "nodes": [{"alias": "n1", "callee_name": "C1", "source_hint_file": "x.py",
                   "entry_kind": "class", "owner_class": None}],  # no 'framework' key
        "edges": [{"src": "n1", "dst": "n1"}],
        "framework": "haystack", "source": "static",
    }
    block = WiringBlock.from_dict(old_row)
    assert block.nodes[0].framework == "unknown"
    assert block.to_dict()["nodes"][0]["framework"] == "unknown"  # round-trips forward


# ---- CS-317 revision: partition into per-system candidates ----


def test_split_block_components_single_chain_is_one_system():
    """A connected chain of nodes is ONE system (one component)."""
    from agent_eval_harness.discovery.wiring import _split_block_components, WiringBlock, WiringNode, WiringEdge
    block = WiringBlock(
        nodes=[WiringNode(alias=a, callee_name=a, source_hint_file="f.py") for a in ("a", "b", "c")],
        edges=[WiringEdge(src="a", dst="b"), WiringEdge(src="b", dst="c")],
        framework="haystack", source="static",
    )
    comps = _split_block_components(block)
    assert len(comps) == 1
    assert {n.alias for n in comps[0].nodes} == {"a", "b", "c"}


def test_split_block_components_two_disjoint_same_framework_pipelines_split():
    """Two INDEPENDENT same-framework pipelines in one block are two systems — partition is by wiring
    connectivity, not hardcoded one-per-framework."""
    from agent_eval_harness.discovery.wiring import _split_block_components, WiringBlock, WiringNode, WiringEdge
    block = WiringBlock(
        nodes=[WiringNode(alias=a, callee_name=a, source_hint_file="f.py")
               for a in ("a1", "a2", "b1", "b2")],
        edges=[WiringEdge(src="a1", dst="a2"), WiringEdge(src="b1", dst="b2")],
        framework="haystack", source="static",
    )
    comps = _split_block_components(block)
    assert len(comps) == 2
    alias_sets = sorted((frozenset(n.alias for n in c.nodes) for c in comps), key=sorted)
    assert alias_sets == [frozenset({"a1", "a2"}), frozenset({"b1", "b2"})]
    assert all(c.framework == "haystack" for c in comps)


def test_split_block_components_isolated_node_is_own_singleton():
    """A wiring node touched by no edge is its own single-node system."""
    from agent_eval_harness.discovery.wiring import _split_block_components, WiringBlock, WiringNode, WiringEdge
    block = WiringBlock(
        nodes=[WiringNode(alias="a", callee_name="a", source_hint_file="f.py"),
               WiringNode(alias="b", callee_name="b", source_hint_file="f.py"),
               WiringNode(alias="lonely", callee_name="lonely", source_hint_file="f.py")],
        edges=[WiringEdge(src="a", dst="b")],
        framework="langgraph", source="static",
    )
    comps = _split_block_components(block)
    assert len(comps) == 2
    assert any({n.alias for n in c.nodes} == {"lonely"} for c in comps)


def test_split_block_components_start_end_sentinels_keep_graph_together():
    """START/END sentinel edges bridge real nodes of the SAME graph (one component), and sentinel-only
    edges never leak into another component."""
    from agent_eval_harness.discovery.wiring import _split_block_components, WiringBlock, WiringNode, WiringEdge
    block = WiringBlock(
        nodes=[WiringNode(alias=a, callee_name=a, source_hint_file="g.py") for a in ("plan", "act")],
        edges=[WiringEdge(src="__start__", dst="plan"), WiringEdge(src="plan", dst="act"),
               WiringEdge(src="act", dst="__end__")],
        framework="langgraph", source="static",
    )
    comps = _split_block_components(block)
    assert len(comps) == 1
    assert {n.alias for n in comps[0].nodes} == {"plan", "act"}


def test_detect_wiring_systems_splits_mixed_into_pure_blocks():
    """A mixed Haystack + LangGraph file set yields TWO PURE blocks (no '+'), one per framework — the
    partition primitive that replaces union-into-one-candidate."""
    from agent_eval_harness.discovery.wiring import detect_wiring_systems
    haystack_file = (
        "from haystack import Pipeline\n"
        "p = Pipeline()\n"
        "p.add_component('embedder', TextEmbedder())\n"
        "p.add_component('retriever', Retriever())\n"
        "p.connect('embedder', 'retriever')\n"
    )
    langgraph_file = (
        "from langgraph.graph import StateGraph\n"
        "class Orchestrator:\n"
        "    def build(self):\n"
        "        g = StateGraph(dict)\n"
        "        g.add_node('plan', self._plan)\n"
        "        g.add_node('act', self._act)\n"
        "        g.add_edge('plan', 'act')\n"
        "    def _plan(self, s): ...\n"
        "    def _act(self, s): ...\n"
    )
    systems = detect_wiring_systems({"pipe.py": haystack_file, "graph.py": langgraph_file})
    frameworks = sorted(b.framework for b in systems)
    assert frameworks == ["haystack", "langgraph"]
    assert all("+" not in b.framework for b in systems)  # each block is PURE
    hay = next(b for b in systems if b.framework == "haystack")
    lg = next(b for b in systems if b.framework == "langgraph")
    assert {n.alias for n in hay.nodes} == {"embedder", "retriever"}
    assert {n.alias for n in lg.nodes} == {"plan", "act"}
    assert lg.edges  # StateGraph edges preserved, not orphaned


def test_detect_wiring_systems_empty_when_no_static_wiring():
    """No static wiring => [] so the engine falls back to its single-candidate / LLM path unchanged."""
    from agent_eval_harness.discovery.wiring import detect_wiring_systems
    assert detect_wiring_systems({"plain.py": "x = 1\ndef f():\n    return x\n"}) == []
