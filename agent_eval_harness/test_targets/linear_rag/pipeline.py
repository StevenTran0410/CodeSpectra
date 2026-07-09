"""T1 linear_rag: retriever -> writer. build_pipeline() + tier-2 demo entry points."""
from __future__ import annotations

from pathlib import Path

from haystack import AsyncPipeline

from agent_eval_harness.instrumentation.base import PipelineHandle
from agent_eval_harness.llm.client import LLMClient, LLMResponse
from agent_eval_harness.llm.fake_client import FakeLLMClient
from test_targets._shared.defects import DefectConfig
from test_targets.linear_rag.components import RetrieverComponent, WriterComponent

_CORPUS_DIR = Path(__file__).parent / "corpus"


def _load_corpus() -> list[str]:
    return [p.read_text(encoding="utf-8").strip() for p in sorted(_CORPUS_DIR.glob("*.txt"))]


def build_pipeline(llm_client: LLMClient, defects: DefectConfig | None = None) -> PipelineHandle:
    defects = defects or DefectConfig.from_env()
    pipeline = AsyncPipeline()
    pipeline.add_component("retriever", RetrieverComponent(_load_corpus()))
    pipeline.add_component("writer", WriterComponent(llm_client, defects))
    pipeline.connect("retriever.documents", "writer.documents")
    return PipelineHandle(
        pipeline=pipeline,
        entry_inputs={"retriever": "query", "writer": "query"},
        exit_node="writer",
    )


# Tier-2 boundary-wrapper demo entry points — plain async functions sharing the same
# underlying component logic, so tier-1 vs tier-2 differ only in span granularity.
_default_llm_client: LLMClient | None = None


def set_default_llm_client(client: LLMClient) -> None:
    global _default_llm_client
    _default_llm_client = client


def _get_default_llm_client() -> LLMClient:
    global _default_llm_client
    if _default_llm_client is None:
        _default_llm_client = FakeLLMClient(
            LLMResponse(content="This is a fallback tier-2 demo answer.", model="fake-tier2")
        )
    return _default_llm_client


async def run_retrieve(query: str) -> dict:
    retriever = RetrieverComponent(_load_corpus())
    return await retriever.run_async(query)


async def run_write(query: str, prior_output: dict) -> dict:
    writer = WriterComponent(_get_default_llm_client(), DefectConfig.from_env())
    return await writer.run_async(query, prior_output.get("documents", []))
