import pytest

from agent_eval_harness.datasets.types import DatasetCase
from agent_eval_harness.datasets.versioning import next_version
from agent_eval_harness.store import repository


@pytest.mark.asyncio
async def test_dataset_versioning_and_immutability():
    # 1. Base name is new
    v1 = await next_version("t2_guard")
    assert v1 == "t2_guard_v1"

    # 2. Insert some cases under t2_guard_v1
    case1 = DatasetCase(
        id="c1",
        dataset="t2_guard_v1",
        kind="guard_classification",
        input={"query": "test query"},
        expected={"verdict": "pass"},
        labels={"category": "valid"},
        provenance="synthetic"
    )
    await repository.insert_dataset_cases_bulk("t2_guard_v1", [case1])

    # 3. Request next version, should see v2
    v2 = await next_version("t2_guard")
    assert v2 == "t2_guard_v2"

    # 4. Check immutability: insert to v2 and make sure v1 remains unchanged
    case2 = DatasetCase(
        id="c2",
        dataset="t2_guard_v2",
        kind="guard_classification",
        input={"query": "another test query"},
        expected={"verdict": "reject"},
        labels={"category": "gibberish"},
        provenance="synthetic"
    )
    await repository.insert_dataset_cases_bulk("t2_guard_v2", [case2])

    cases_v1 = await repository.get_dataset_cases("t2_guard_v1")
    assert len(cases_v1) == 1
    assert cases_v1[0]["id"] == "c1"

    cases_v2 = await repository.get_dataset_cases("t2_guard_v2")
    assert len(cases_v2) == 1
    assert cases_v2[0]["id"] == "c2"
