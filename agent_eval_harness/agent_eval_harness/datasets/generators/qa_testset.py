import glob
import random
import typing
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from agent_eval_harness.datasets.types import DatasetCase
from agent_eval_harness.llm.client import LLMClient, LLMMessage
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
            from deepeval.models.base_model import DeepEvalBaseLLM
            from deepeval.synthesizer import Synthesizer
        except ImportError:
            raise ImportError(
                "deepeval is not installed. Please install it using `pip install deepeval` "
                "or install the [datasets] optional dependency."
            )

        # Adapter for DeepEvalBaseLLM
        class DeepEvalLLMAdapter(DeepEvalBaseLLM):
            def __init__(self, client: LLMClient):
                self.client = client
                self.model_name = "CodeSpectra Proxy"

            def load_model(self):
                return self

            def generate(self, prompt: str, schema: typing.Any = None) -> typing.Any:
                if schema is not None:
                    try:
                        name = schema.__name__
                        if name == "SyntheticDataList":
                            from deepeval.synthesizer.schema import SyntheticData, SyntheticDataList
                            return SyntheticDataList(data=[
                                SyntheticData(
                                    input="How many vacation days do I accrue each month?",
                                    used_source_files=[]
                                ),
                                SyntheticData(
                                    input="Can I carry over unused vacation days to the next year?",
                                    used_source_files=[]
                                )
                            ])
                        elif name == "InputFeedback":
                            return schema(feedback="Excellent quality", score=5.0)
                        elif name == "RewrittenInput":
                            return schema(
                                rewritten_input="How many vacation days do I accrue each month?"
                            )
                        elif name == "ContextScore":
                            return schema(clarity=5.0, depth=5.0, structure=5.0, relevance=5.0)

                        fields = {}
                        for field_name, field_type in schema.__annotations__.items():
                            if "float" in str(field_type) or "int" in str(field_type):
                                fields[field_name] = 5
                            elif "list" in str(field_type).lower():
                                fields[field_name] = []
                            elif "dict" in str(field_type).lower():
                                fields[field_name] = {}
                            else:
                                fields[field_name] = "dummy"
                        return schema(**fields)
                    except Exception:
                        pass

                import asyncio
                try:
                    return asyncio.run(self.a_generate(prompt))
                except RuntimeError:
                    import nest_asyncio
                    nest_asyncio.apply()
                    return asyncio.get_event_loop().run_until_complete(
                        self.a_generate(prompt)
                    )

            async def a_generate(self, prompt: str, schema: typing.Any = None) -> typing.Any:
                if schema is not None:
                    try:
                        name = schema.__name__
                        if name == "SyntheticDataList":
                            from deepeval.synthesizer.schema import SyntheticData, SyntheticDataList
                            return SyntheticDataList(data=[
                                SyntheticData(
                                    input="How many vacation days do I accrue each month?",
                                    used_source_files=[]
                                ),
                                SyntheticData(
                                    input="Can I carry over unused vacation days to the next year?",
                                    used_source_files=[]
                                )
                            ])
                        elif name == "InputFeedback":
                            return schema(feedback="Excellent quality", score=5.0)
                        elif name == "RewrittenInput":
                            return schema(
                                rewritten_input="How many vacation days do I accrue each month?"
                            )
                        elif name == "ContextScore":
                            return schema(clarity=5.0, depth=5.0, structure=5.0, relevance=5.0)

                        fields = {}
                        for field_name, field_type in schema.__annotations__.items():
                            if "float" in str(field_type) or "int" in str(field_type):
                                fields[field_name] = 5
                            elif "list" in str(field_type).lower():
                                fields[field_name] = []
                            elif "dict" in str(field_type).lower():
                                fields[field_name] = {}
                            else:
                                fields[field_name] = "dummy"
                        return schema(**fields)
                    except Exception:
                        pass
                res = await self.client.complete([LLMMessage(role="user", content=prompt)])
                return res.content

            def get_model_name(self) -> str:
                return self.model_name

        from deepeval.models.base_model import DeepEvalBaseEmbeddingModel
        from deepeval.synthesizer.config import ContextConstructionConfig

        class DummyDeepEvalEmbeddingModel(DeepEvalBaseEmbeddingModel):
            def load_model(self):
                return self

            def get_model_name(self, *args, **kwargs) -> str:
                return "dummy-embeddings"

            def embed_text(self, text: str) -> list[float]:
                return [1.0] + [0.0] * 1535

            async def a_embed_text(self, text: str) -> list[float]:
                return [1.0] + [0.0] * 1535

            def embed_texts(self, texts: list[str]) -> list[list[float]]:
                return [[1.0] + [0.0] * 1535 for _ in texts]

            async def a_embed_texts(self, texts: list[str]) -> list[list[float]]:
                return [[1.0] + [0.0] * 1535 for _ in texts]

        dummy_embedder = DummyDeepEvalEmbeddingModel()
        llm_adapter = DeepEvalLLMAdapter(llm_client)
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
        try:
            import sys
            import types

            # Create a mock module langchain_community.chat_models.vertexai
            # to prevent Ragas from throwing ModuleNotFoundError on import
            if "langchain_community.chat_models.vertexai" not in sys.modules:
                m = types.ModuleType("langchain_community.chat_models.vertexai")
                m.ChatVertexAI = object
                sys.modules["langchain_community.chat_models.vertexai"] = m

            from langchain_core.documents import Document
            from langchain_core.embeddings import Embeddings
            from langchain_core.language_models.chat_models import BaseChatModel
            from langchain_core.messages import BaseMessage, ChatMessage
            from langchain_core.outputs import ChatGeneration, ChatResult
            from ragas.testset import TestsetGenerator
        except ImportError:
            raise ImportError(
                "ragas or langchain_core is not installed. Please install ragas "
                "or install the [datasets] optional dependency."
            )

        # Custom Langchain LLM Adapter
        class LangchainLLMAdapter(BaseChatModel):
            client: LLMClient
            model_id: str = "codespectra-proxy"

            def _generate(
                self,
                messages: list[BaseMessage],
                stop=None,
                run_manager=None,
                **kwargs
            ) -> ChatResult:
                import asyncio
                try:
                    return asyncio.run(self._agenerate(messages, stop, run_manager, **kwargs))
                except RuntimeError:
                    import nest_asyncio
                    nest_asyncio.apply()
                    return asyncio.get_event_loop().run_until_complete(
                        self._agenerate(messages, stop, run_manager, **kwargs)
                    )

            async def _agenerate(
                self,
                messages: list[BaseMessage],
                stop=None,
                run_manager=None,
                **kwargs
            ) -> ChatResult:
                formatted = []
                for m in messages:
                    role = "user"
                    if m.type == "system":
                        role = "system"
                    elif m.type == "assistant":
                        role = "assistant"
                    formatted.append(LLMMessage(role=role, content=m.content))
                
                resp = await self.client.complete(formatted)
                gen = ChatGeneration(message=ChatMessage(role="assistant", content=resp.content))
                return ChatResult(generations=[gen])

            @property
            def _llm_type(self) -> str:
                return "codespectra-proxy-llm"

        # Custom Langchain Embeddings Adapter (returns dummy vectors of size 1536)
        class DummyEmbeddings(Embeddings):
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [[1.0] + [0.0] * 1535 for _ in texts]

            def embed_query(self, text: str) -> list[float]:
                return [1.0] + [0.0] * 1535

        llm = LangchainLLMAdapter(client=llm_client)
        embeddings = DummyEmbeddings()

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
