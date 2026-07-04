import pytest

from agent_eval_harness.datasets.generators.sufficiency_labeled import generate
from agent_eval_harness.datasets.types import DatasetCase
from agent_eval_harness.store import repository


@pytest.mark.asyncio
async def test_sufficiency_labeled_ablation():
    # 1. Insert a source qa_testset case with multiple contexts
    source_case = DatasetCase(
        id="qa_1",
        dataset="t1_qa_v1",
        kind="qa_testset",
        input={"query": "What is the policy?"},
        expected={"answer": "The policy is X."},
        labels={"contexts": ["context 1", "context 2"]},
        provenance="synthetic"
    )
    await repository.insert_dataset_cases_bulk("t1_qa_v1", [source_case])

    # 2. Run the sufficiency generator config
    config = {
        "dataset_name": "t1_sufficiency_v1",
        "source_dataset_id": "t1_qa_v1"
    }

    cases = await generate(config, llm_client=None)
    # We expect 3 cases generated: sufficient, insufficient, and partial_context
    assert len(cases) == 3

    # Check sufficient case (full context)
    suff = [c for c in cases if c.labels.get("variant") == "full_context"][0]
    assert suff.input["query"] == "What is the policy?"
    assert suff.input["context"] == ["context 1", "context 2"]
    assert suff.expected == {"sufficient": True}
    assert suff.provenance == "synthetic"

    # Check insufficient case (no context)
    insuff = [c for c in cases if c.labels.get("variant") == "no_context"][0]
    assert insuff.input["query"] == "What is the policy?"
    assert insuff.input["context"] == []
    assert insuff.expected == {"sufficient": False}
    assert insuff.provenance == "synthetic"

    # Check partial context case (1 context out of 2)
    partial = [c for c in cases if c.labels.get("variant") == "partial_context"][0]
    assert partial.input["query"] == "What is the policy?"
    assert partial.input["context"] == ["context 1"]
    assert partial.expected is None
    assert partial.provenance == "synthetic"
