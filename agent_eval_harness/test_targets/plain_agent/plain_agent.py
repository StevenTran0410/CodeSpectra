"""test_targets/plain_agent — hand-rolled, framework-free agents (no Haystack/LangGraph/LangChain).

Two shapes on purpose, so the scanner is proven against more than one vendor's format:
  - `ResearchAgent` mirrors the PROVIDER-ABSTRACTION shape (the real ask-mode agent's shape): a base
    class exposing an async `_call(...)` method, a `*Agent` subclass whose `run(...)` invokes the LLM
    through `self._call(...)` (NOT a raw SDK), and a module-level `*_SYSTEM` prompt constant.
  - `ToolAgent` mirrors the RAW-SDK shape: a `tools=[...]` list of module-level functions and a manual
    tool-call dispatch loop over `.tool_calls`. Lower-confidence / secondary heuristic.

Nothing here is imported by AEH — the passes are pure ast.parse.
"""
from __future__ import annotations

from dataclasses import dataclass, field

RESEARCH_SYSTEM = (
    "You are a research assistant. Answer the user's question using retrieved sources; "
    "never fabricate a citation."
)

DEFAULT_MODEL = "gpt-4o-mini"


class _LLMBase:
    """Provider-abstraction base — the concrete transport is irrelevant to the shape."""

    def __init__(self, provider: object | None = None) -> None:
        self._provider = provider

    async def _call(self, system_prompt: str, user_prompt: str, max_tokens: int = 1000) -> str:
        raise NotImplementedError("provider-abstraction stand-in — not called by AEH's static passes")


class ResearchAgent(_LLMBase):
    """Provider-abstraction agent: entry `run()` calls the inherited `self._call(...)`, no raw SDK."""

    async def run(self, provider_id: str, model_id: str, question: str) -> dict:
        answer = await self._call(RESEARCH_SYSTEM, question, max_tokens=800)
        return {"answer": answer}


@dataclass
class ToolCall:
    name: str
    arguments: str


@dataclass
class _ChatResponse:
    tool_calls: list[ToolCall] = field(default_factory=list)
    content: str = ""


def tool_search(query: str) -> str:
    """A tool the agent can call: looks up sources for `query`."""
    return f"search results for: {query}"


def tool_summarize(text: str) -> str:
    """A second tool the agent can call: condenses `text`."""
    return f"summary of: {text}"


class ToolAgent:
    """Raw-SDK-shaped agent: a tools list of module-level functions + a manual dispatch loop."""

    def __init__(self, client: object | None = None) -> None:
        self._client = client
        self.tools = [tool_search, tool_summarize]

    def run(self, response: _ChatResponse) -> str:
        result = ""
        for tool_call in response.tool_calls:
            for fn in self.tools:
                if tool_call.name == fn.__name__:
                    result = fn(tool_call.arguments)
                    break
        return result


# A bare `*Agent` class with NO corroborating signal — must NOT be emitted (precision guard).
class DisabledAgent:
    version = 1
