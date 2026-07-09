import pytest

from agent_eval_harness.datasets.generators.qa_testset import (
    _BACKENDS,
    QATestsetBackend,
    generate,
)
from agent_eval_harness.llm.fake_client import FakeLLMClient


@pytest.mark.asyncio
async def test_qa_testset_validation_failures():
    fake_client = FakeLLMClient([])

    # llm_client=None raises ValueError
    config = {
        "dataset_name": "test_v1",
        "corpus_paths": ["test_targets/linear_rag/corpus/*.txt"],
        "count": 5,
        "backend": "deepeval"
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
        "backend": "deepeval"
    }
    with pytest.raises(ValueError, match="No corpus files found"):
        await generate(config_empty_corpus, llm_client=fake_client)

    # Non-existent corpus path raises ValueError
    config_non_existent = {
        "dataset_name": "test_v1",
        "corpus_paths": ["non_existent_folder/*.txt"],
        "count": 5,
        "backend": "deepeval"
    }
    with pytest.raises(ValueError, match="No corpus files found"):
        await generate(config_non_existent, llm_client=fake_client)


def test_qa_testset_backends_registry_has_no_mock():
    """Mock backend was removed — qa_testset must always exercise a real LLM-backed
    synthesis library (deepeval or ragas), never hardcoded placeholder text."""
    assert set(_BACKENDS.keys()) == {"deepeval", "ragas"}
    for backend in _BACKENDS.values():
        assert isinstance(backend, QATestsetBackend)
