"""T2 multi_agent components: guard(rule+llm) -> planner -> worker(+2 tools) ->
judge -> writer.

PlannerComponent is the only top-level Haystack-scheduled orchestration node
besides guard — it internally composes worker/judge/writer (plain method calls,
each wrapped in its own manual span), since Haystack's static DAG model doesn't
cleanly express the judge-reject -> retry loop. This matches the epic's own
role-taxonomy semantics: "orchestrator: decomposes intent, routes to sub-agents,
manages retries" (CS-260 §3) — retry ownership belongs to the orchestrator.
"""
from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

from haystack import component

from agent_eval_harness.instrumentation.tier1_haystack import (
    OP_LLM_CALL,
    OP_TOOL_CALL,
    OP_VALIDATOR_DECISION,
)
from agent_eval_harness.llm.client import LLMClient, LLMMessage
from test_targets._shared.defects import DefectConfig
from test_targets._shared.tracing_helpers import manual_span
from test_targets.linear_rag.components import WriterComponent

__all__ = [
    "GuardComponent",
    "PlannerComponent",
    "WorkerComponent",
    "JudgeComponent",
    "WriterComponent",
]


def _usage_tags(response) -> dict:
    return {
        "model": response.model,
        "tokens_in": response.prompt_tokens,
        "tokens_out": response.completion_tokens,
        "token_source": response.token_source,
    }


@component
class GuardComponent:
    """rule stage (deterministic, no LLM) + llm stage — two nested manual spans
    sharing one Haystack pipeline node (component_name='guard'), disambiguated by
    the aeh.check.kind tag, matching CS-261's 'guard (2 sub-spans)' requirement.
    """

    __haystack_supports_async__ = True
    MIN_QUERY_LENGTH = 5

    def __init__(self, llm_client: LLMClient, defects: DefectConfig | None = None) -> None:
        self._llm_client = llm_client
        self._defects = defects or DefectConfig.from_env()

    @component.output_types(query=str, passed=bool)
    def run(self, query: str) -> dict:
        raise NotImplementedError("GuardComponent is async-only — use run_async().")

    @component.output_types(query=str, passed=bool)
    async def run_async(self, query: str) -> dict:
        with manual_span(OP_VALIDATOR_DECISION, "guard", {"aeh.check.kind": "rule"}) as span:
            rule_pass = len(query.strip()) >= self.MIN_QUERY_LENGTH
            span.set_content_tag("aeh.output", {"verdict": "pass" if rule_pass else "reject"})

        with manual_span(OP_VALIDATOR_DECISION, "guard", {"aeh.check.kind": "llm"}) as span:
            response = await self._llm_client.complete(
                [LLMMessage(role="user", content=f"Classify policy violation for: {query}")]
            )
            llm_verdict = "reject" if "reject" in response.content.lower() else "pass"
            if self._defects.guard_leak:
                llm_verdict = "pass"
            span.set_content_tag("aeh.output", {"verdict": llm_verdict, **_usage_tags(response)})

        return {"query": query, "passed": rule_pass and llm_verdict == "pass"}


class WorkerComponent:
    """Composed (called from planner), not a scheduled Haystack pipeline node —
    Haystack cannot statically express "worker runs 1-2 times depending on
    intent count", so this is a plain class whose run_async wraps itself in a
    manual span each call, giving "worker span(s)" plurality for free.
    """

    def __init__(
        self,
        tools: dict[str, Callable[[str], Awaitable[str]]],
        defects: DefectConfig | None = None,
    ) -> None:
        self._tools = tools
        self._defects = defects or DefectConfig.from_env()

    async def run_async(self, intents: list[str]) -> dict:
        tool_order = (
            ["decoy_lookup", "case_law_search"]
            if self._defects.wrong_tool
            else ["case_law_search", "decoy_lookup"]
        )
        with manual_span("aeh.worker_step", "worker", {"aeh.intents": intents}) as span:
            context: list[str] = []
            joined_query = " ".join(intents)
            for tool_name in tool_order:
                with manual_span(OP_TOOL_CALL, "worker", {"aeh.tool.name": tool_name}) as tool_span:
                    result = await self._tools[tool_name](joined_query)
                    tool_span.set_content_tag("aeh.output", {"result": result})
                if tool_name == "case_law_search":
                    context.append(result)
            span.set_content_tag("aeh.output", {"context": context, "intents": intents})
        return {"context": context}


class JudgeComponent:
    """Composed (called from planner) — sufficiency verdict, can trigger one retry."""

    def __init__(self, llm_client: LLMClient, defects: DefectConfig | None = None) -> None:
        self._llm_client = llm_client
        self._defects = defects or DefectConfig.from_env()

    async def run_async(self, query: str, worker_output: dict) -> dict:
        with manual_span(OP_VALIDATOR_DECISION, "judge") as span:
            if self._defects.judge_rubber_stamp:
                sufficient = True
                span.set_content_tag(
                    "aeh.output", {"sufficient": sufficient, "rubber_stamped": True}
                )
            else:
                context = worker_output.get("context", [])
                prompt = f"Is this context sufficient to answer '{query}'? Context: {context}"
                response = await self._llm_client.complete(
                    [LLMMessage(role="user", content=prompt)]
                )
                content_lower = response.content.lower()
                # "insufficient" contains "sufficient" as a substring — check the
                # negative form first so a rejection is never misread as a pass.
                sufficient = (
                    "insufficient" not in content_lower and "sufficient" in content_lower
                )
                span.set_content_tag(
                    "aeh.output", {"sufficient": sufficient, **_usage_tags(response)}
                )
        return {"sufficient": sufficient, "context": worker_output.get("context", [])}


@component
class PlannerComponent:
    __haystack_supports_async__ = True
    MAX_ITEMS_PER_CALL = 2

    def __init__(
        self,
        llm_client: LLMClient,
        worker: WorkerComponent,
        judge: JudgeComponent,
        writer: WriterComponent,
        defects: DefectConfig | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._worker = worker
        self._judge = judge
        self._writer = writer
        self._defects = defects or DefectConfig.from_env()

    @component.output_types(answer=str)
    def run(self, query: str, passed: bool = True) -> dict:
        raise NotImplementedError("PlannerComponent is async-only — use run_async().")

    @component.output_types(answer=str)
    async def run_async(self, query: str, passed: bool = True) -> dict:
        if not passed:
            return {"answer": "Request blocked by guard."}

        intents = await self._decompose(query)
        batch_size = len(intents) if self._defects.planner_overpack else self.MAX_ITEMS_PER_CALL
        chunked = [intents[i : i + batch_size] for i in range(0, len(intents), batch_size)]
        batches = chunked or [intents]

        worker_output: dict = {"context": []}
        for batch in batches:
            worker_output = await self._worker.run_async(batch)

        judge_verdict = await self._judge.run_async(query, worker_output)
        if not judge_verdict["sufficient"] and not self._defects.no_retry:
            worker_output = await self._worker.run_async(intents)
            judge_verdict = await self._judge.run_async(query, worker_output)

        writer_output = await self._writer.run_async(query, judge_verdict.get("context", []))
        return {"answer": writer_output["answer"]}

    async def _decompose(self, query: str) -> list[str]:
        with manual_span(OP_LLM_CALL, "planner") as span:
            response = await self._llm_client.complete(
                [LLMMessage(role="user", content=f"Decompose into intents as a JSON list: {query}")]
            )
            span.set_content_tag("aeh.output", _usage_tags(response))
        try:
            intents = json.loads(response.content)
            if isinstance(intents, list) and intents:
                return [str(item) for item in intents]
        except (json.JSONDecodeError, TypeError):
            pass
        return [query]
