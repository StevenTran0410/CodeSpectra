"""T1 linear_rag components: RetrieverComponent (pure Python, no LLM) -> WriterComponent."""
from __future__ import annotations

import re

from haystack import component
from haystack.dataclasses import ChatMessage

from agent_eval_harness.instrumentation.tier1_haystack import OP_LLM_CALL
from agent_eval_harness.llm.client import LLMClient
from agent_eval_harness.llm.generator import HarnessChatGenerator
from test_targets._shared.defects import DefectConfig, maybe_hallucinate
from test_targets._shared.tracing_helpers import manual_span

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _keyword_overlap(query: str, doc: str) -> int:
    return len(_tokenize(query) & _tokenize(doc))


@component
class RetrieverComponent:
    """Pure Python keyword-overlap ranker — genuinely offline, no LLM involved."""

    __haystack_supports_async__ = True

    def __init__(self, corpus: list[str], top_k: int = 3) -> None:
        self._corpus = corpus
        self._top_k = top_k

    @component.output_types(documents=list[str])
    def run(self, query: str) -> dict:
        raise NotImplementedError("RetrieverComponent is async-only — use run_async().")

    @component.output_types(documents=list[str])
    async def run_async(self, query: str) -> dict:
        ranked = sorted(self._corpus, key=lambda doc: -_keyword_overlap(query, doc))
        return {"documents": ranked[: self._top_k]}


@component
class WriterComponent:
    """Answers strictly from the retrieved context. Holds a HarnessChatGenerator
    internally (composition, not a scheduled pipeline node) and wraps its call in
    a manual aeh.llm_call span, so DEFECT_WRITER_HALLUCINATE can post-process the
    reply after the LLM call completes."""

    __haystack_supports_async__ = True

    def __init__(self, llm_client: LLMClient, defects: DefectConfig | None = None) -> None:
        self._generator = HarnessChatGenerator(
            llm_client,
            system_prompt=(
                "Answer the question using ONLY the provided context. "
                "Do not add any information that is not present in the context."
            ),
        )
        self._defects = defects or DefectConfig.from_env()

    @component.output_types(answer=str)
    def run(self, query: str, documents: list[str]) -> dict:
        raise NotImplementedError("WriterComponent is async-only — use run_async().")

    @component.output_types(answer=str)
    async def run_async(self, query: str, documents: list[str]) -> dict:
        context = "\n".join(f"- {d}" for d in documents)
        messages = [ChatMessage.from_user(f"Context:\n{context}\n\nQuestion: {query}")]

        with manual_span(OP_LLM_CALL, "writer") as span:
            gen_result = await self._generator.run_async(messages=messages)
            reply = gen_result["replies"][0]
            usage = (reply.meta or {}).get("usage", {})
            span.set_content_tag(
                "aeh.output",
                {
                    "model": (reply.meta or {}).get("model"),
                    "tokens_in": usage.get("prompt_tokens"),
                    "tokens_out": usage.get("completion_tokens"),
                    "token_source": (reply.meta or {}).get("aeh_token_source"),
                },
            )

        final_text = maybe_hallucinate(reply.text or "", self._defects)
        return {"answer": final_text}
