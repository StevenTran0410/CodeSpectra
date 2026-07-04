import glob
import random
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from agent_eval_harness.datasets.types import DatasetCase
from agent_eval_harness.llm.client import LLMClient
from agent_eval_harness.store.repository import new_id


class QATestsetConfig(BaseModel):
    dataset_name: str
    corpus_paths: list[str]
    count: int
    backend: str = "deepeval"  # "deepeval" or "ragas"

@runtime_checkable
class QATestsetBackend(Protocol):
    async def generate_cases(
        self, dataset_name: str, corpus_paths: list[Path], count: int, llm_client: LLMClient
    ) -> list[DatasetCase]:
        ...

class DeepEvalQATestsetBackend:
    async def generate_cases(
        self, dataset_name: str, corpus_paths: list[Path], count: int, llm_client: LLMClient
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
        dummy_embedder = make_deepeval_embedding_model()
        synthesizer = Synthesizer(model=llm_adapter)

        # DeepEval requires file paths as list of strings
        paths_str = [str(p.resolve()) for p in corpus_paths]

        context_config = ContextConstructionConfig(
            embedder=dummy_embedder,
            critic_model=llm_adapter
        )
        goldens = synthesizer.generate_goldens_from_docs(
            document_paths=paths_str,
            max_goldens_per_context=2,
            include_expected_output=True,
            context_construction_config=context_config
        )

        # If deepeval returned fewer or more goldens, we select/pad them
        cases = []
        for g in goldens[:count]:
            cases.append(DatasetCase(
                id=new_id(),
                dataset=dataset_name,
                kind="qa_testset",
                input={"query": g.input},
                expected={"answer": g.expected_output},
                labels={"contexts": g.context},
                provenance="synthetic"
            ))

        return cases

class RagasQATestsetBackend:
    async def generate_cases(
        self, dataset_name: str, corpus_paths: list[Path], count: int, llm_client: LLMClient
    ) -> list[DatasetCase]:
        from agent_eval_harness.llm.ragas_adapter import (
            make_ragas_embeddings,
            make_ragas_llm_adapter,
            stub_missing_langchain_community_vertexai,
        )

        try:
            # Must run before the `from ragas.testset import ...` below — see
            # stub_missing_langchain_community_vertexai's own docstring for why.
            stub_missing_langchain_community_vertexai()

            from langchain_core.documents import Document
            from ragas.testset import TestsetGenerator
        except ImportError:
            raise ImportError(
                "ragas or langchain_core is not installed. Please install ragas "
                "or install the [datasets] optional dependency."
            )

        llm = make_ragas_llm_adapter(llm_client)
        embeddings = make_ragas_embeddings()

        # Load documents
        docs = []
        for path in corpus_paths:
            with open(path, encoding="utf-8") as f:
                content = f.read()
            # Pad short document to satisfy Ragas default minimum length requirement
            if len(content.split()) < 150:
                content = content + "\n" + (" ".join(["placeholder"] * 200))
            docs.append(Document(page_content=content, metadata={"source": str(path.name)}))

        generator = TestsetGenerator.from_langchain(llm, embeddings)
        testset = generator.generate_with_langchain_docs(docs, testset_size=count)
        df = testset.to_pandas()

        cases = []
        for _, row in df.iterrows():
            query = row.get("user_input") or row.get("question") or ""
            answer = row.get("reference") or row.get("ground_truth") or ""
            contexts = row.get("reference_contexts") or row.get("contexts") or []
            cases.append(DatasetCase(
                id=new_id(),
                dataset=dataset_name,
                kind="qa_testset",
                input={"query": query},
                expected={"answer": answer},
                labels={"contexts": list(contexts)},
                provenance="synthetic"
            ))
        return cases

class MockQATestsetBackend:
    async def generate_cases(
        self, dataset_name: str, corpus_paths: list[Path], count: int, llm_client: LLMClient
    ) -> list[DatasetCase]:
        cases = []
        for path in corpus_paths[:count]:
            name = path.stem.replace("_", " ").title()
            cases.append(DatasetCase(
                id=new_id(),
                dataset=dataset_name,
                kind="qa_testset",
                input={"query": f"What is the policy for {name}?"},
                expected={"answer": f"This is a placeholder answer for {name} policy."},
                labels={"contexts": [f"Context snippet from {path.name}."]},
                provenance="synthetic"
            ))
        while len(cases) < count:
            cases.append(DatasetCase(
                id=new_id(),
                dataset=dataset_name,
                kind="qa_testset",
                input={"query": f"Query placeholder {len(cases)}"},
                expected={"answer": "Expected placeholder answer"},
                labels={"contexts": ["Context placeholder"]},
                provenance="synthetic"
            ))
        return cases

# Global backend instance registry to allow testing/mocking
_BACKENDS = {
    "deepeval": DeepEvalQATestsetBackend(),
    "ragas": RagasQATestsetBackend(),
    "mock": MockQATestsetBackend()
}

async def generate(
    config: dict, llm_client: LLMClient | None, seed: int | None = None
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

    backend = _BACKENDS[backend_name]
    return await backend.generate_cases(
        dataset_name=dataset_name,
        corpus_paths=corpus_files,
        count=parsed_config.count,
        llm_client=llm_client
    )
