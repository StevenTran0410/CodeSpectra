import json
import textwrap

import pytest

from agent_eval_harness.datasets.generators.recorded_report_replay import generate
from agent_eval_harness.datasets.fulfillment import fulfill_plan
from agent_eval_harness.datasets.types import DatasetCase
from agent_eval_harness.llm.client import LLMResponse
from agent_eval_harness.llm.fake_client import FakeLLMClient
from agent_eval_harness.planning.contract import EvaluationContract
from agent_eval_harness.planning.report import AgentPlanReport, EvaluationPlanReport, save_plan_report
from agent_eval_harness.store import repository
from tests._stubs import FakeCodeSpectraClient


def _write_plan(path, entries_yaml: str) -> None:
    path.write_text(f"entries:\n{entries_yaml}", encoding="utf-8")


async def test_generator_happy_path_auditor():
    reports = [
        {
            "report_id": "rep-1",
            "snapshot_id": "snap-1",
            "sections": {
                "A": {"content": "A1"},
                "J": {"content": "J1"},
                "K": {"content": "K1"},
            },
        }
    ]
    # For auditor group, K was sliced out during _derive_config
    config = {
        "dataset_name": "test_ds_auditor",
        "reports": reports,
    }
    cases = await generate(config, None)
    assert len(cases) == 1
    assert cases[0].kind == "recorded_report_replay"
    assert cases[0].input["shape"] == "all_sections"
    sections = cases[0].input["all_sections"]
    assert "A" in sections
    assert "J" in sections
    assert "K" in sections  # generate() keeps whatever sections are passed in config
    assert cases[0].labels["report_id"] == "rep-1"


async def test_generator_empty_reports():
    config = {
        "dataset_name": "test_ds_empty",
        "reports": [],
    }
    cases = await generate(config, None)
    assert cases == []


async def test_fulfill_plan_integration_recorded_report_replay(tmp_path):
    map_path = tmp_path / "map.yaml"
    # Letters come from each agent's own harvested field_downstream_consumers, not an
    # "agent_id == synthesizer" literal — auditor reads A/J (no K), synthesizer reads A/B/K.
    plan_report = EvaluationPlanReport(
        target_system_id="t",
        agents=[
            AgentPlanReport(
                agent_id="auditor",
                contract=EvaluationContract(
                    agent_id="auditor", field_downstream_consumers={"A": ["x"], "J": ["y"]},
                ),
            ),
            AgentPlanReport(
                agent_id="synthesizer",
                contract=EvaluationContract(
                    agent_id="synthesizer",
                    field_downstream_consumers={"A": ["x"], "B": ["y"], "K": ["z"]},
                ),
            ),
        ],
    )
    save_plan_report(plan_report, map_path.with_name(map_path.stem + "_plan_report.yaml"))

    plan_path = tmp_path / "plan.yaml"
    _write_plan(plan_path, textwrap.dedent("""\
          - id: auditor.recorded
            component: auditor
            agent_id: auditor
            metric: schema_valid
            metric_class: assertion
            dataset:
              required: {kind: recorded_report_replay, min_cases: 2}
            rationale: r
            provenance: rule
          - id: synthesizer.recorded
            component: synthesizer
            agent_id: synthesizer
            metric: schema_valid
            metric_class: assertion
            dataset:
              required: {kind: recorded_report_replay, min_cases: 2}
            rationale: r
            provenance: rule
    """))

    reports = [
        {"id": "rep-1", "repo_id": "test_repo", "snapshot_id": "snap-1"},
        {"id": "rep-2", "repo_id": "test_repo", "snapshot_id": "snap-1"},
    ]
    full_reports = {
        "rep-1": {
            "id": "rep-1",
            "report": {
                "sections": {
                    "A": {"content": "A1"},
                    "K": {"content": "K1"},
                }
            }
        },
        "rep-2": {
            "id": "rep-2",
            "report": {
                "sections": {
                    "B": {"content": "B2"},
                    "K": {"content": "K2"},
                }
            }
        }
    }
    stub_client = FakeCodeSpectraClient(reports=reports, full_reports=full_reports)
    llm_client = FakeLLMClient(LLMResponse(content="{}", model="fake"))

    report = await fulfill_plan(
        plan_path, str(tmp_path / "map.yaml"), "snap-1", str(tmp_path),
        "prov-1", "model-1", llm_client,
        codespectra_client=stub_client,
    )

    auditor_key = "recorded_report_replay/auditor"
    synthesizer_key = "recorded_report_replay/synthesizer"

    assert report[auditor_key]["status"] == "fulfilled"
    assert report[synthesizer_key]["status"] == "fulfilled"

    auditor_cases = await repository.get_dataset_cases(report[auditor_key]["dataset_id"])
    assert len(auditor_cases) == 2
    sec_case_1 = json.loads(auditor_cases[0]["input_json"])["all_sections"]
    assert "K" not in sec_case_1

    synthesizer_cases = await repository.get_dataset_cases(report[synthesizer_key]["dataset_id"])
    assert len(synthesizer_cases) == 2
    sec_case_syn_1 = json.loads(synthesizer_cases[0]["input_json"])["all_sections"]
    assert "K" in sec_case_syn_1


async def test_fulfill_plan_without_client_regression(tmp_path):
    plan_path = tmp_path / "plan.yaml"
    _write_plan(plan_path, textwrap.dedent("""\
          - id: auditor.recorded
            component: auditor
            agent_id: auditor
            metric: schema_valid
            metric_class: assertion
            dataset:
              required: {kind: recorded_report_replay, min_cases: 1}
            rationale: r
            provenance: rule
    """))

    llm_client = FakeLLMClient(LLMResponse(content="{}", model="fake"))

    report = await fulfill_plan(
        plan_path, str(tmp_path / "map.yaml"), "snap-1", str(tmp_path),
        "prov-1", "model-1", llm_client,
        codespectra_client=None,
    )

    auditor_key = "recorded_report_replay/auditor"
    assert report[auditor_key]["status"] == "needs_human"
