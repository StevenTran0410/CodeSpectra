"""T-LG langgraph_agent: 2 function nodes + 1 class with 2 bound-method nodes, a router with
add_conditional_edges, and sentinel START/END edges. Scanner-only fixture — Stage 1-3 harvest is
pure AST, the graph is never actually compiled/run by any test."""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class AgentState(TypedDict, total=False):
    query: str
    plan: list[str]
    step_cursor: int
    findings: list[str]
    summary: str


async def load_context(state: AgentState) -> AgentState:
    """Function node."""
    state["plan"] = ["investigate", "investigate", "synthesize"]
    state["step_cursor"] = 0
    return state


async def plan_step(state: AgentState, max_steps: int = 3) -> AgentState:
    """Function node — a real typed kwarg (max_steps) beyond state, so the case has rich fields."""
    state["step_cursor"] = min(state.get("step_cursor", 0), max_steps)
    return state


class ResearchAgent:
    """Owner class of the 2 bound-method nodes."""

    def __init__(self, llm_client, defects=None) -> None:
        self._llm = llm_client
        self._defects = defects

    def build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)

        graph.add_edge(START, "load_context")
        graph.add_edge("load_context", "plan_step")

        graph.add_node("load_context", load_context)
        graph.add_node("plan_step", plan_step)
        graph.add_node("investigate", self._node_investigate)
        graph.add_node("synthesize", self._node_synthesize)

        graph.add_conditional_edges(
            "plan_step", self._route,
            {"investigate": "investigate", "synthesize": "synthesize"},
        )
        graph.add_edge("investigate", "plan_step")
        graph.add_edge("synthesize", END)
        return graph

    async def _node_investigate(self, state: AgentState, top_k: int = 5) -> AgentState:
        findings = state.get("findings", [])
        findings.append(f"finding from step {state.get('step_cursor', 0)}")
        state["findings"] = findings[:top_k]
        state["step_cursor"] = state.get("step_cursor", 0) + 1
        return state

    async def _node_synthesize(self, state: AgentState) -> AgentState:
        state["summary"] = "; ".join(state.get("findings", []))
        return state

    def _route(self, state: AgentState) -> str:
        if state.get("step_cursor", 0) >= len(state.get("plan", [])):
            return "synthesize"
        return "investigate"


def compile_agent(agent: ResearchAgent):
    return agent.build_graph().compile()
