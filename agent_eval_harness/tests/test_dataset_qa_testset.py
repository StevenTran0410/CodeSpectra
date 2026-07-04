from pathlib import Path

import pytest

from agent_eval_harness.datasets.generators.qa_testset import (
    MockQATestsetBackend,
    QATestsetBackend,
    generate,
)
from agent_eval_harness.llm.fake_client import FakeLLMClient


@pytest.mark.asyncio
async def test_qa_testset_mock_backend():
    # Use mock backend which runs offline without dependencies
    config = {
        "dataset_name": "t1_qa_test_v1",
        "corpus_paths": ["test_targets/linear_rag/corpus/*.txt"],
        "count": 5,
        "backend": "mock"
    }
    fake_client = FakeLLMClient([])
    cases = await generate(config, llm_client=fake_client, seed=42)
    assert len(cases) == 5
    for case in cases:
        assert case.dataset == "t1_qa_test_v1"
        assert case.kind == "qa_testset"
        assert "query" in case.input
        assert "answer" in case.expected
        assert "contexts" in case.labels
        assert case.provenance == "synthetic"


@pytest.mark.asyncio
async def test_qa_testset_validation_failures():
    fake_client = FakeLLMClient([])

    # llm_client=None raises ValueError
    config = {
        "dataset_name": "test_v1",
        "corpus_paths": ["test_targets/linear_rag/corpus/*.txt"],
        "count": 5,
        "backend": "mock"
    }
    with pytest.raises(ValueError, match="LLM client is required for qa_testset generation"):
        await generate(config, llm_client=None)

    # Unknown backend name raises ValueError
    config_unknown = {
        "dataset_name": "test_v1",
        "corpus_paths": ["test_targets/linear_rag/corpus/*.txt"],
        "count": 5,
        "backend": "unknown"
    }
    with pytest.raises(ValueError, match="Unknown QA testset backend: unknown"):
        await generate(config_unknown, llm_client=fake_client)

    # Empty/missing corpus_paths raises ValueError
    config_empty_corpus = {
        "dataset_name": "test_v1",
        "corpus_paths": [],
        "count": 5,
        "backend": "mock"
    }
    with pytest.raises(ValueError, match="No corpus files found"):
        await generate(config_empty_corpus, llm_client=fake_client)

    # Non-existent corpus path raises ValueError
    config_non_existent = {
        "dataset_name": "test_v1",
        "corpus_paths": ["non_existent_folder/*.txt"],
        "count": 5,
        "backend": "mock"
    }
    with pytest.raises(ValueError, match="No corpus files found"):
        await generate(config_non_existent, llm_client=fake_client)


def test_qa_testset_backend_protocol():
    # Assert that MockQATestsetBackend conforms to QATestsetBackend protocol
    backend = MockQATestsetBackend()
    assert isinstance(backend, QATestsetBackend)

    # Assert that a custom fake backend class also conforms
    class CustomFakeBackend:
        async def generate_cases(
            self,
            dataset_name: str,
            corpus_paths: list[Path],
            count: int,
            llm_client,
            seed: int = 42
        ):
            return []

    custom_backend = CustomFakeBackend()
    assert isinstance(custom_backend, QATestsetBackend)
