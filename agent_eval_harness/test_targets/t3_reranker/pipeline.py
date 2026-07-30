"""T3 pipeline: retriever -> reranker -> writer."""
from __future__ import annotations

from pathlib import Path

from haystack import AsyncPipeline

from agent_eval_harness.instrumentation.base import PipelineHandle
from agent_eval_harness.llm.client import LLMClient
from test_targets._shared.defects import DefectConfig
from test_targets.t3_reranker.components import (
    RerankerComponent,
    RetrieverComponent,
    WriterComponent,
)

_CORPUS_DIR = Path(__file__).parent.parent / "linear_rag" / "corpus"


def _load_corpus() -> list[str]:
    return [p.read_text(encoding="utf-8").strip() for p in sorted(_CORPUS_DIR.glob("*.txt"))]


def build_pipeline(llm_client: LLMClient, defects: DefectConfig | None = None) -> PipelineHandle:
    defects = defects or DefectConfig.from_env()
    pipeline = AsyncPipeline()
    pipeline.add_component("retriever", RetrieverComponent(_load_corpus()))
    pipeline.add_component("reranker", RerankerComponent())
    pipeline.add_component("writer", WriterComponent(llm_client, defects))
    
    pipeline.connect("retriever.documents", "reranker.documents")
    pipeline.connect("reranker.reranked_docs", "writer.documents")
    
    return PipelineHandle(
        pipeline=pipeline,
        entry_inputs={"retriever": "query", "reranker": "query", "writer": "query"},
        exit_node="writer",
    )
