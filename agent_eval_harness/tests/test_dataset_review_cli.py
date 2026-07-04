import pytest

from agent_eval_harness.datasets.review import run_review_loop
from agent_eval_harness.datasets.types import DatasetCase
from agent_eval_harness.store import repository


@pytest.mark.asyncio
async def test_dataset_review_cli_loop(monkeypatch):
    # 1. Insert 3 cases under t2_review_v1
    case1 = DatasetCase(
        id="rev_c1",
        dataset="t2_review_v1",
        kind="guard_classification",
        input={"query": "q1"},
        expected={"verdict": "reject", "category": "off_topic"},
        labels={"category": "off_topic"},
        provenance="synthetic"
    )
    case2 = DatasetCase(
        id="rev_c2",
        dataset="t2_review_v1",
        kind="guard_classification",
        input={"query": "q2"},
        expected={"verdict": "reject", "category": "jailbreak"},
        labels={"category": "jailbreak"},
        provenance="synthetic"
    )
    case3 = DatasetCase(
        id="rev_c3",
        dataset="t2_review_v1",
        kind="guard_classification",
        input={"query": "q3"},
        expected={"verdict": "pass"},
        labels={"category": "valid"},
        provenance="synthetic"
    )
    await repository.insert_dataset_cases_bulk("t2_review_v1", [case1, case2, case3])

    # 2. Mock input() responses for review loop:
    # First case: accept -> "a"
    # Second case: edit -> "e", then enters corrected expected JSON:
    # '{"verdict": "reject", "category": "jailbreak_custom"}'
    # Third case: reject -> "r"
    inputs = [
        "a",  # Accept first case
        "e",  # Edit second case
        '{"verdict": "reject", "category": "jailbreak_custom"}',  # Enter new expected
        "r"  # Reject (delete) third case
    ]
    
    input_iter = iter(inputs)
    monkeypatch.setattr("builtins.input", lambda *args, **kwargs: next(input_iter))

    # Run review loop
    await run_review_loop("t2_review_v1")

    # 3. Verify final DB state
    cases = await repository.get_dataset_cases("t2_review_v1")
    # Case 3 was deleted, so we should only have 2 cases left
    assert len(cases) == 2

    # Verify Case 1: accepted
    db_c1 = [c for c in cases if c["id"] == "rev_c1"][0]
    assert db_c1["provenance"] == "generated+reviewed"
    assert "off_topic" in db_c1["expected_json"]

    # Verify Case 2: edited
    db_c2 = [c for c in cases if c["id"] == "rev_c2"][0]
    assert db_c2["provenance"] == "generated+reviewed"
    assert "jailbreak_custom" in db_c2["expected_json"]
