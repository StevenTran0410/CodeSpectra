import textwrap

import pytest

from agent_eval_harness.datasets.fulfillment import export_dataset, fulfill_plan
from agent_eval_harness.datasets.types import DatasetCase
from agent_eval_harness.llm.client import LLMResponse
from agent_eval_harness.llm.fake_client import FakeLLMClient
from agent_eval_harness.store import repository

pytestmark = pytest.mark.asyncio


def _write_plan(path, entries_yaml: str) -> None:
    path.write_text(f"entries:\n{entries_yaml}", encoding="utf-8")


async def test_snapshot_fixture_group_fulfilled_and_written_back(tmp_path):
    plan_path = tmp_path / "plan.yaml"
    _write_plan(plan_path, textwrap.dedent("""\
          - id: agent1.snap
            component: c1
            agent_id: agent1
            metric: schema_valid
            metric_class: assertion
            dataset:
              required: {kind: snapshot_fixture, min_cases: 1}
            rationale: r
            provenance: rule
    """))
    llm_client = FakeLLMClient(LLMResponse(content="{}", model="fake"))

    report = await fulfill_plan(
        plan_path, str(tmp_path / "map.yaml"), "snap-1", str(tmp_path),
        "prov-1", "model-1", llm_client,
    )

    key = "snapshot_fixture/agent1"
    assert report[key]["status"] == "fulfilled"
    dataset_id = report[key]["dataset_id"]

    cases = await repository.get_dataset_cases(dataset_id)
    assert len(cases) == 1

    from agent_eval_harness.metrics.suite import load_suite
    suite = load_suite(plan_path)
    assert suite.entries[0].dataset.ref == dataset_id
    assert suite.entries[0].dataset.required is None


async def test_field_match_gold_derives_tech_stack_from_disk(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.x]\n", encoding="utf-8")
    plan_path = tmp_path / "plan.yaml"
    _write_plan(plan_path, textwrap.dedent("""\
          - id: project_identity.field_match
            component: project_identity
            agent_id: project_identity
            metric: field_match
            metric_class: assertion
            params: {fields: ["repo_name"]}
            dataset:
              required: {kind: field_match_gold, min_cases: 1}
            rationale: r
            provenance: rule
    """))
    llm_client = FakeLLMClient(LLMResponse(content="{}", model="fake"))

    report = await fulfill_plan(
        plan_path, str(tmp_path / "map.yaml"), "snap-1", str(tmp_path),
        "prov-1", "model-1", llm_client,
    )

    key = "field_match_gold/project_identity"
    assert report[key]["status"] == "fulfilled"
    cases = await repository.get_dataset_cases(report[key]["dataset_id"])
    assert len(cases) == 1
    import json
    expected = json.loads(cases[0]["expected_json"])
    assert expected["field_paths"]["repo_name"] == tmp_path.name
    assert "python" in expected["field_paths"]["tech_stack"]


async def test_snapshot_regression_baseline_always_needs_human(tmp_path):
    plan_path = tmp_path / "plan.yaml"
    _write_plan(plan_path, textwrap.dedent("""\
          - id: agent1.baseline
            component: c1
            agent_id: agent1
            metric: field_match
            metric_class: assertion
            params: {fields: ["x"]}
            dataset:
              required: {kind: snapshot_regression_baseline, min_cases: 1}
            rationale: r
            provenance: rule
    """))
    llm_client = FakeLLMClient(LLMResponse(content="{}", model="fake"))

    report = await fulfill_plan(
        plan_path, str(tmp_path / "map.yaml"), "snap-1", str(tmp_path),
        "prov-1", "model-1", llm_client,
    )

    key = "snapshot_regression_baseline/agent1"
    assert report[key]["status"] == "needs_human"
    assert "CS-283" in report[key]["reason"] or "live target-execution" in report[key]["reason"]


async def test_guard_classification_underivable_categories_needs_human(tmp_path):
    plan_path = tmp_path / "plan.yaml"
    _write_plan(plan_path, textwrap.dedent("""\
          - id: guard.classifier
            component: guard
            agent_id: guard_agent
            metric: classifier.guard_check
            metric_class: classifier
            params: {entry_point: "guard.run"}
            dataset:
              required: {kind: guard_classification, min_cases: 40}
            rationale: r
            provenance: llm_suggested
    """))
    llm_client = FakeLLMClient(LLMResponse(content="{}", model="fake"))

    report = await fulfill_plan(
        plan_path, str(tmp_path / "map.yaml"), "snap-1", str(tmp_path),
        "prov-1", "model-1", llm_client,
    )

    key = "guard_classification/guard_agent"
    assert report[key]["status"] == "needs_human"


async def test_seed_cases_land_as_handwritten_and_skip_min_cases_shortfall(tmp_path):
    plan_path = tmp_path / "plan.yaml"
    _write_plan(plan_path, textwrap.dedent("""\
          - id: agent1.snap
            component: c1
            agent_id: agent1
            metric: schema_valid
            metric_class: assertion
            dataset:
              required: {kind: snapshot_fixture, min_cases: 1}
            rationale: r
            provenance: rule
    """))
    llm_client = FakeLLMClient(LLMResponse(content="{}", model="fake"))
    instructions = {
        "snapshot_fixture/agent1": {
            "seed_cases": [{"input": {"shape": "kwargs", "kwargs": {"snapshot_id": "manual-1"}}}]
        }
    }

    report = await fulfill_plan(
        plan_path, str(tmp_path / "map.yaml"), "snap-1", str(tmp_path),
        "prov-1", "model-1", llm_client, instructions=instructions,
    )

    key = "snapshot_fixture/agent1"
    assert report[key]["status"] == "fulfilled"
    cases = await repository.get_dataset_cases(report[key]["dataset_id"])
    # 1 handwritten seed + 1 generated (snapshot_id "snap-1")
    assert len(cases) == 2
    provenances = {c["provenance"] for c in cases}
    assert provenances == {"handwritten", "synthetic"}


async def test_export_dataset_excludes_synthetic_never_leaves_aeh():
    dataset_id = "t_export_v1"
    await repository.insert_dataset_cases_bulk(dataset_id, [
        DatasetCase(id="export_c1", dataset=dataset_id, kind="qa_testset",
                    input={"query": "q"}, expected=None, labels=None, provenance="synthetic"),
        DatasetCase(id="export_c2", dataset=dataset_id, kind="qa_testset",
                    input={"query": "q2"}, expected=None, labels=None, provenance="generated+reviewed"),
        DatasetCase(id="export_c3", dataset=dataset_id, kind="qa_testset",
                    input={"query": "q3"}, expected=None, labels=None, provenance="handwritten"),
    ])
    await repository.insert_dataset_metadata(dataset_id, "qa_testset", min_cases=1)

    exported = await export_dataset(dataset_id)

    assert {c.id for c in exported} == {"export_c2", "export_c3"}
    assert all("kind" not in c.input for c in exported)
