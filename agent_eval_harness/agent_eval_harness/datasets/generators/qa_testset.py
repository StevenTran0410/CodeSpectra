import glob
import random
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from agent_eval_harness.datasets.generator_utils import build_qa_testset_case
from agent_eval_harness.datasets.types import DatasetCase
from agent_eval_harness.llm.client import LLMClient
from agent_eval_harness.llm.embedding_client import EmbeddingClient

_MAX_GOLDENS_PER_CONTEXT = 2
# Ragas rejects short source docs outright; pad below this floor rather than fail the run.
_RAGAS_MIN_WORD_COUNT = 150
_RAGAS_PADDING_WORD_COUNT = 200


class QATestsetConfig(BaseModel):
    dataset_name: str
    corpus_paths: list[str]
    count: int
    backend: str = "deepeval"  # "deepeval" or "ragas"

@runtime_checkable
class QATestsetBackend(Protocol):
    async def generate_cases(
        self,
        dataset_name: str,
        corpus_paths: list[Path],
        count: int,
        llm_client: LLMClient,
        embedding_client: EmbeddingClient,
    ) -> list[DatasetCase]:
        ...

class DeepEvalQATestsetBackend:
    async def generate_cases(
        self,
        dataset_name: str,
        corpus_paths: list[Path],
        count: int,
        llm_client: LLMClient,
        embedding_client: EmbeddingClient,
    ) -> list[DatasetCase]:
        try:
            from deepeval.synthesizer import Synthesizer
        except ImportError:
            raise ImportError(
                "deepeval is not installed. Please install it using `pip install deepeval` "
                "or install the [datasets] optional dependency."
            )

        from deepeval.synthesizer.config import ContextConstructionConfig

        from agent_eval_harness.llm.deepeval_adapter import (
            make_deepeval_embedding_model,
            make_deepeval_llm_adapter,
        )

        llm_adapter = make_deepeval_llm_adapter(llm_client)
        real_embedder = make_deepeval_embedding_model(embedding_client)
        synthesizer = Synthesizer(model=llm_adapter)

        # DeepEval requires file paths as list of strings
        paths_str = [str(p.resolve()) for p in corpus_paths]

        context_config = ContextConstructionConfig(
            embedder=real_embedder,
            critic_model=llm_adapter
        )
        goldens = synthesizer.generate_goldens_from_docs(
            document_paths=paths_str,
            max_goldens_per_context=_MAX_GOLDENS_PER_CONTEXT,
            include_expected_output=True,
            context_construction_config=context_config
        )

        # If deepeval returned fewer or more goldens, we select/pad them
        cases = []
        for g in goldens[:count]:
            cases.append(build_qa_testset_case(
                dataset_name=dataset_name,
                query=g.input,
                answer=g.expected_output,
                contexts=g.context
            ))

        return cases

class RagasQATestsetBackend:
    async def generate_cases(
        self,
        dataset_name: str,
        corpus_paths: list[Path],
        count: int,
        llm_client: LLMClient,
        embedding_client: EmbeddingClient,
    ) -> list[DatasetCase]:
        from agent_eval_harness.llm.ragas_adapter import (
            make_ragas_embeddings,
            make_ragas_llm_adapter,
            stub_missing_langchain_community_vertexai,
        )

        try:
            # Must run before the `from ragas.testset import ...` below — see
            stub_missing_langchain_community_vertexai()

            from langchain_core.documents import Document
            from ragas.testset import TestsetGenerator
        except ImportError:
            raise ImportError(
                "ragas or langchain_core is not installed. Please install ragas "
                "or install the [datasets] optional dependency."
            )

        llm = make_ragas_llm_adapter(llm_client)
        embeddings = make_ragas_embeddings(embedding_client)

        docs = []
        for path in corpus_paths:
            with open(path, encoding="utf-8") as f:
                content = f.read()
            if len(content.split()) < _RAGAS_MIN_WORD_COUNT:
                content = content + "\n" + (" ".join(["placeholder"] * _RAGAS_PADDING_WORD_COUNT))
            docs.append(Document(page_content=content, metadata={"source": str(path.name)}))

        generator = TestsetGenerator.from_langchain(llm, embeddings)
        testset = generator.generate_with_langchain_docs(docs, testset_size=count)
        df = testset.to_pandas()

        cases = []
        for _, row in df.iterrows():
            query = row.get("user_input") or row.get("question") or ""
            answer = row.get("reference") or row.get("ground_truth") or ""
            contexts = row.get("reference_contexts") or row.get("contexts") or []
            cases.append(build_qa_testset_case(
                dataset_name=dataset_name,
                query=query,
                answer=answer,
                contexts=list(contexts)
            ))
        return cases

# Global backend instance registry
_BACKENDS = {
    "deepeval": DeepEvalQATestsetBackend(),
    "ragas": RagasQATestsetBackend(),
}

async def generate(
    config: dict,
    llm_client: LLMClient | None,
    seed: int | None = None,
    embedding_client: EmbeddingClient | None = None,
) -> list[DatasetCase]:
    parsed_config = QATestsetConfig.model_validate(config)
    
    if seed is not None:
        random.seed(seed)
        
    dataset_name = parsed_config.dataset_name
    
    # Expand globs/paths for corpus
    corpus_files = []
    for path_pattern in parsed_config.corpus_paths:
        expanded = glob.glob(path_pattern)
        if not expanded:
            # try literal path relative to workspace or absolute
            expanded = [path_pattern]
        for path_str in expanded:
            p = Path(path_str)
            if p.is_file():
                corpus_files.append(p)
            elif p.is_dir():
                corpus_files.extend(list(p.glob("*.txt")) + list(p.glob("*.md")))
                
    if not corpus_files:
        raise ValueError(f"No corpus files found for paths: {parsed_config.corpus_paths}")

    backend_name = parsed_config.backend.lower()
    if backend_name not in _BACKENDS:
        raise ValueError(
            f"Unknown QA testset backend: {backend_name}. Choose 'deepeval' or 'ragas'"
        )

    if not llm_client:
        raise ValueError("LLM client is required for qa_testset generation")

    if not embedding_client:
        raise ValueError("Embedding client is required for qa_testset generation")

    backend = _BACKENDS[backend_name]
    return await backend.generate_cases(
        dataset_name=dataset_name,
        corpus_paths=corpus_files,
        count=parsed_config.count,
        llm_client=llm_client,
        embedding_client=embedding_client,
    )
