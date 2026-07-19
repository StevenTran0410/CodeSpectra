import json
import textwrap

import pytest

from agent_eval_harness.datasets.fulfillment import (
    _archetype_for,
    _derive_config,
    _qa_testset_backend,
    _TOPO_ORDER,
    export_dataset,
    fulfill_plan,
)
from agent_eval_harness.datasets.types import DatasetCase
from agent_eval_harness.llm.client import LLMResponse
from agent_eval_harness.llm.fake_client import FakeLLMClient
from agent_eval_harness.metrics.suite import SuiteEntry
from agent_eval_harness.planning.contract import (
    EvaluationContract,
    InvocationContract,
    KwargSpec,
    OutputContract,
)
from agent_eval_harness.planning.report import AgentPlanReport, EvaluationPlanReport, save_plan_report
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


async def test_prune_prior_versions_removes_old_keeps_latest_and_other_bases():
    """Re-fulfilling must not accumulate v1/v2/v3: _prune_prior_versions drops older versions of
    the same base_name (keeping the just-written one) and never touches a different base_name."""
    from agent_eval_harness.datasets.fulfillment import _prune_prior_versions

    base = "synthetic_agent_io_agentX"
    for v in ("v1", "v2", "v3"):
        did = f"{base}_{v}"
        await repository.insert_dataset_cases_bulk(did, [
            DatasetCase(id=f"{did}_c1", dataset=did, kind="synthetic_agent_io",
                        input={"x": 1}, expected={}, labels={"agent_id": "agentX"}, provenance="synthetic"),
        ])
        await repository.insert_dataset_metadata(did, "synthetic_agent_io", min_cases=1)
    other = "synthetic_agent_io_other_v1"
    await repository.insert_dataset_cases_bulk(other, [
        DatasetCase(id="other_c1", dataset=other, kind="synthetic_agent_io",
                    input={"x": 1}, expected={}, labels={"agent_id": "other"}, provenance="synthetic"),
    ])
    await repository.insert_dataset_metadata(other, "synthetic_agent_io", min_cases=1)

    deleted = await _prune_prior_versions(base, keep=f"{base}_v3")

    assert deleted == 2
    ids = {d["dataset_id"] for d in await repository.list_dataset_ids()}
    assert f"{base}_v3" in ids
    assert f"{base}_v1" not in ids and f"{base}_v2" not in ids
    assert other in ids  # different base_name untouched


async def test_export_dataset_fail_closed_on_unknown_provenance():
    # A case with an unrecognised provenance must never reach export_dataset's output.
    dataset_id = "t_export_bogus_v1"
    db = repository.get_db()
    await db.execute(
        "INSERT INTO dataset_cases (id, dataset_id, input_json, expected_json, labels_json, provenance) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("bogus_c1", dataset_id, '{"x": 1}', None, None, "bogus"),
    )
    await db.commit()
    await repository.insert_dataset_metadata(dataset_id, "qa_testset", min_cases=1)

    exported = await export_dataset(dataset_id)

    assert not any(c.id == "bogus_c1" for c in exported), (
        "export_dataset must be fail-closed: 'bogus' provenance must not export"
    )


def _entry(metric: str, metric_class: str = "llm_judge") -> SuiteEntry:
    return SuiteEntry(
        id=f"c.{metric}.llm0", component="c", metric=metric, metric_class=metric_class,
        dataset={"required": {"kind": "qa_testset", "min_cases": 20}},
    )


def test_qa_testset_backend_picks_ragas_when_a_consumer_uses_it():
    entries = [_entry("geval.decomposition_coverage"), _entry("ragas.faithfulness")]
    assert _qa_testset_backend(entries) == "ragas"


def test_qa_testset_backend_defaults_to_deepeval_for_geval_consumers():
    entries = [_entry("geval.decomposition_coverage"), _entry("geval.audit_output_completeness")]
    assert _qa_testset_backend(entries) == "deepeval"


def test_qa_testset_backend_defaults_to_deepeval_when_no_llm_judge_consumer():
    entries = [_entry("classifier.guard_rule_accuracy", metric_class="classifier")]
    assert _qa_testset_backend(entries) == "deepeval"


async def test_extra_snapshot_ids_field_match_gold(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.x]\n", encoding="utf-8")
    snap2_path = tmp_path / "snap2"
    snap2_path.mkdir()
    (snap2_path / "package.json").write_text("{}\n", encoding="utf-8")

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

    from tests._stubs import FakeCodeSpectraClient
    stub_client = FakeCodeSpectraClient(files={
        "snapshot_local_path_snap-2": str(snap2_path),
    })

    llm_client = FakeLLMClient(LLMResponse(content="{}", model="fake"))
    instructions = {
        "field_match_gold/project_identity": {
            "extra_snapshot_ids": ["snap-2"]
        }
    }

    report = await fulfill_plan(
        plan_path, str(tmp_path / "map.yaml"), "snap-1", str(tmp_path),
        "prov-1", "model-1", llm_client,
        instructions=instructions,
        codespectra_client=stub_client,
    )

    key = "field_match_gold/project_identity"
    assert report[key]["status"] == "fulfilled"
    cases = await repository.get_dataset_cases(report[key]["dataset_id"])
    assert len(cases) == 2

    c1 = [c for c in cases if json.loads(c["input_json"])["kwargs"]["snapshot_id"] == "snap-1"][0]
    expected1 = json.loads(c1["expected_json"])
    assert expected1["field_paths"]["repo_name"] == tmp_path.name
    assert "python" in expected1["field_paths"]["tech_stack"]

    c2 = [c for c in cases if json.loads(c["input_json"])["kwargs"]["snapshot_id"] == "snap-2"][0]
    expected2 = json.loads(c2["expected_json"])
    assert expected2["field_paths"]["repo_name"] == "snap2"
    assert "node" in expected2["field_paths"]["tech_stack"]


async def test_extra_snapshot_ids_snapshot_fixture(tmp_path):
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

    from tests._stubs import FakeCodeSpectraClient
    stub_client = FakeCodeSpectraClient()
    llm_client = FakeLLMClient(LLMResponse(content="{}", model="fake"))

    instructions = {
        "snapshot_fixture/agent1": {
            "extra_snapshot_ids": ["snap-2", "snap-3"]
        }
    }

    report = await fulfill_plan(
        plan_path, str(tmp_path / "map.yaml"), "snap-1", str(tmp_path),
        "prov-1", "model-1", llm_client,
        instructions=instructions,
        codespectra_client=stub_client,
    )

    key = "snapshot_fixture/agent1"
    assert report[key]["status"] == "fulfilled"
    cases = await repository.get_dataset_cases(report[key]["dataset_id"])

    assert len(cases) == 3
    snapshot_ids_generated = {json.loads(c["input_json"])["kwargs"]["snapshot_id"] for c in cases}
    assert snapshot_ids_generated == {"snap-1", "snap-2", "snap-3"}


_SYNTH_SCHEMA = {
    "type": "object",
    "properties": {"overall_confidence": {"type": "string"}, "notes": {"type": "string"}},
    "required": ["overall_confidence", "notes"],
}


def _write_plan_report(map_path, *, agent_id: str, eval_enabled: bool, with_contract: bool = True):
    contract = None
    if with_contract:
        contract = EvaluationContract(
            agent_id=agent_id,
            output=OutputContract(json_schema=_SYNTH_SCHEMA),
            field_downstream_consumers={"A": ["confidence"]},
        )
    report = EvaluationPlanReport(
        target_system_id="t",
        agents=[AgentPlanReport(agent_id=agent_id, contract=contract, eval_enabled=eval_enabled)],
    )
    plan_report_path = map_path.with_name(map_path.stem + "_plan_report.yaml")
    save_plan_report(report, plan_report_path)
    return plan_report_path


def test_topo_order_includes_synthetic_agent_io():
    assert "synthetic_agent_io" in _TOPO_ORDER


def test_archetype_for_fan_in_judge_when_field_downstream_consumers_present():
    contract = EvaluationContract(agent_id="auditor", field_downstream_consumers={"A": ["confidence"]})
    assert _archetype_for(contract) == "fan_in_judge"


def test_archetype_for_unimplemented_when_no_field_downstream_consumers():
    contract = EvaluationContract(agent_id="project_identity")
    assert _archetype_for(contract) == "unimplemented"


def _contract_with_kwargs(
    agent_id: str, kwarg_names: list[str], *, query_planning: bool = False
) -> EvaluationContract:
    return EvaluationContract(
        agent_id=agent_id,
        invocation=InvocationContract(
            kwargs=[KwargSpec(name=n) for n in kwarg_names],
            constructor_deps=["ProviderConfigService", "RetrievalService"],
        ),
        query_planning_subcall=query_planning,
        has_retrieval_signal=True,  # D1: signal set by harvest; required for non-unimplemented archetypes
    )


def test_archetype_for_unimplemented_without_retrieval_signal():
    contract = EvaluationContract(
        agent_id="x",
        invocation=InvocationContract(kwargs=[], constructor_deps=["ProviderConfigService"]),
        has_retrieval_signal=False,
    )
    assert _archetype_for(contract) == "unimplemented"


def test_archetype_for_real_archetype_when_has_retrieval_signal_and_empty_constructor_deps():
    # A6/D1: empty constructor_deps is no longer a blocker when has_retrieval_signal is True (e.g. role-tier fired)
    contract = EvaluationContract(
        agent_id="retriever",
        invocation=InvocationContract(kwargs=[], constructor_deps=[]),
        has_retrieval_signal=True,
    )
    assert _archetype_for(contract) != "unimplemented"
    assert _archetype_for(contract) == "rag_single_shot"


def test_archetype_for_rag_single_shot_matches_glossary_and_important_files():
    # I (glossary): provider_id, model_id, snapshot_id, profile
    assert _archetype_for(
        _contract_with_kwargs("glossary", ["provider_id", "model_id", "snapshot_id", "profile"])
    ) == "rag_single_shot"
    # G (important_files): + graph_summary (excluded from the upstream-signal check)
    assert _archetype_for(
        _contract_with_kwargs(
            "important_files", ["provider_id", "model_id", "snapshot_id", "graph_summary", "profile"]
        )
    ) == "rag_single_shot"


def test_archetype_for_rag_mem_ctx_matches_project_identity():
    contract = _contract_with_kwargs(
        "project_identity",
        ["provider_id", "model_id", "snapshot_id", "repo_name", "mem_ctx", "profile"],
    )
    assert _archetype_for(contract) == "rag_mem_ctx"


def test_archetype_for_rag_upstream_matches_violations_and_onboarding():
    # E (violations): static_convention/static_risk/conventions_output are upstream signals
    assert _archetype_for(
        _contract_with_kwargs(
            "violations",
            ["provider_id", "model_id", "snapshot_id", "static_convention", "static_risk",
             "conventions_output", "profile"],
        )
    ) == "rag_upstream"
    # H (onboarding): important_files_output
    assert _archetype_for(
        _contract_with_kwargs(
            "onboarding",
            ["provider_id", "model_id", "snapshot_id", "important_files_output", "profile"],
        )
    ) == "rag_upstream"


def test_archetype_for_rag_mem_ctx_participant_matches_architecture_and_structure():
    assert _archetype_for(
        _contract_with_kwargs(
            "architecture",
            ["provider_id", "model_id", "snapshot_id", "graph_summary", "arch_bundle",
             "identity_output", "profile", "folder_tree"],
        )
    ) == "rag_mem_ctx_participant"
    assert _archetype_for(
        _contract_with_kwargs(
            "structure",
            ["provider_id", "model_id", "snapshot_id", "arch_bundle", "folder_tree",
             "identity_output", "profile"],
        )
    ) == "rag_mem_ctx_participant"


def test_archetype_for_rag_query_planning_matches_conventions_and_risk():
    assert _archetype_for(
        _contract_with_kwargs(
            "conventions",
            ["provider_id", "model_id", "snapshot_id", "static_convention", "structure_output", "profile"],
            query_planning=True,
        )
    ) == "rag_query_planning"
    assert _archetype_for(
        _contract_with_kwargs(
            "risk", ["provider_id", "model_id", "snapshot_id", "static_risk", "profile"],
            query_planning=True,
        )
    ) == "rag_query_planning"


def test_archetype_for_rag_query_planning_mem_ctx_matches_feature_map():
    contract = _contract_with_kwargs(
        "feature_map",
        ["provider_id", "model_id", "snapshot_id", "graph_summary", "identity_output",
         "architecture_output", "profile", "folder_tree"],
        query_planning=True,
    )
    assert _archetype_for(contract) == "rag_query_planning_mem_ctx"


async def test_derive_config_synthetic_agent_io_returns_config_when_enabled_and_contract_present():
    contract = EvaluationContract(
        agent_id="auditor", output=OutputContract(json_schema=_SYNTH_SCHEMA),
        field_downstream_consumers={"A": ["confidence"]},
    )
    entries = [SuiteEntry(id="e", component="auditor", agent_id="auditor", metric="schema_valid", metric_class="assertion")]

    config = await _derive_config(
        "synthetic_agent_io", "ds-1", "synthetic_agent_io/auditor", entries,
        map_path="map.yaml", snapshot_id="s1", snapshot_local_path=".",
        provider_id="p", model_id="m", dataset_id_by_group={}, painpoint=None, min_cases=20,
        eval_enabled_by_agent={"auditor": True},
        contract_by_agent={"auditor": contract},
        profile_by_agent={},
    )

    assert config is not None
    assert config["agent_id"] == "auditor"
    assert config["archetype"] == "fan_in_judge"
    assert config["contract"]["field_downstream_consumers"] == {"A": ["confidence"]}


async def test_derive_config_synthetic_agent_io_none_when_eval_disabled():
    entries = [SuiteEntry(id="e", component="auditor", agent_id="auditor", metric="schema_valid", metric_class="assertion")]
    config = await _derive_config(
        "synthetic_agent_io", "ds-1", "synthetic_agent_io/auditor", entries,
        map_path="map.yaml", snapshot_id="s1", snapshot_local_path=".",
        provider_id="p", model_id="m", dataset_id_by_group={}, painpoint=None, min_cases=20,
        eval_enabled_by_agent={"auditor": False},
        contract_by_agent={"auditor": EvaluationContract(agent_id="auditor")},
        profile_by_agent={},
    )
    assert config is None


async def test_derive_config_synthetic_agent_io_none_when_contract_missing():
    entries = [SuiteEntry(id="e", component="auditor", agent_id="auditor", metric="schema_valid", metric_class="assertion")]
    config = await _derive_config(
        "synthetic_agent_io", "ds-1", "synthetic_agent_io/auditor", entries,
        map_path="map.yaml", snapshot_id="s1", snapshot_local_path=".",
        provider_id="p", model_id="m", dataset_id_by_group={}, painpoint=None, min_cases=20,
        eval_enabled_by_agent={"auditor": True},
        contract_by_agent={},
        profile_by_agent={},
    )
    assert config is None


async def test_fulfill_plan_synthetic_agent_io_skipped_when_eval_disabled(tmp_path):
    map_path = tmp_path / "map.yaml"
    _write_plan_report(map_path, agent_id="auditor", eval_enabled=False)
    plan_path = tmp_path / "plan.yaml"
    _write_plan(plan_path, textwrap.dedent("""\
          - id: auditor.synthetic
            component: auditor
            agent_id: auditor
            metric: schema_valid
            metric_class: assertion
            dataset:
              required: {kind: synthetic_agent_io, min_cases: 2}
            rationale: r
            provenance: rule
    """))
    llm_client = FakeLLMClient(LLMResponse(content="[]", model="fake"))

    report = await fulfill_plan(
        plan_path, str(map_path), "snap-1", str(tmp_path), "prov-1", "model-1", llm_client,
    )

    key = "synthetic_agent_io/auditor"
    assert report[key]["status"] == "skipped"
    assert len(llm_client.calls) == 0  # never even attempted generation


async def test_fulfill_plan_synthetic_agent_io_fulfilled_end_to_end(tmp_path):
    map_path = tmp_path / "map.yaml"
    _write_plan_report(map_path, agent_id="auditor", eval_enabled=True)
    plan_path = tmp_path / "plan.yaml"
    _write_plan(plan_path, textwrap.dedent("""\
          - id: auditor.synthetic
            component: auditor
            agent_id: auditor
            metric: schema_valid
            metric_class: assertion
            dataset:
              required: {kind: synthetic_agent_io, min_cases: 2}
            rationale: r
            provenance: rule
    """))

    case_payload = json.dumps([
        {"input": {"A": {"confidence": "high"}}, "gold": {"overall_confidence": "high", "notes": "n1"}},
        {"input": {"A": {"confidence": "low"}}, "gold": {"overall_confidence": "low", "notes": "n2"}},
    ])
    llm_client = FakeLLMClient(LLMResponse(content=case_payload, model="fake"))

    report = await fulfill_plan(
        plan_path, str(map_path), "snap-1", str(tmp_path), "prov-1", "model-1", llm_client,
    )

    key = "synthetic_agent_io/auditor"
    assert report[key]["status"] == "fulfilled"
    cases = await repository.get_dataset_cases(report[key]["dataset_id"])
    assert len(cases) == 2
    first_input = json.loads(cases[0]["input_json"])
    assert first_input["shape"] == "all_sections"


async def test_fulfill_plan_force_agent_ids_regenerates_already_fulfilled_dataset(tmp_path):
    """The "Regenerate" action (DatasetReviewScreen) — a synthetic_agent_io dataset that's
    already fulfilled is normally skipped entirely (see the carry-forward test above); force_agent_ids
    must delete the stale dataset and its cases, then let the walk re-fulfill it as if new."""
    map_path = tmp_path / "map.yaml"
    _write_plan_report(map_path, agent_id="auditor", eval_enabled=True)
    plan_path = tmp_path / "plan.yaml"
    _write_plan(plan_path, textwrap.dedent("""\
          - id: auditor.synthetic_agent_io
            component: auditor
            agent_id: auditor
            metric: schema_valid
            metric_class: assertion
            dataset:
              required: {kind: synthetic_agent_io, min_cases: 2}
            rationale: r
            provenance: rule
    """))

    first_payload = json.dumps([
        {"input": {"A": {"confidence": "high"}}, "gold": {"overall_confidence": "high", "notes": "bad"}},
        {"input": {"A": {"confidence": "low"}}, "gold": {"overall_confidence": "low", "notes": "bad"}},
    ])
    first_report = await fulfill_plan(
        plan_path, str(map_path), "snap-1", str(tmp_path), "prov-1", "model-1",
        FakeLLMClient(LLMResponse(content=first_payload, model="fake")),
    )
    stale_dataset_id = first_report["synthetic_agent_io/auditor"]["dataset_id"]

    # Second call without force: already fulfilled, so it's skipped and absent from the report.
    rerun_report = await fulfill_plan(
        plan_path, str(map_path), "snap-1", str(tmp_path), "prov-1", "model-1",
        FakeLLMClient(LLMResponse(content="[]", model="fake")),
    )
    assert "synthetic_agent_io/auditor" not in rerun_report

    # force_agent_ids resets required.min_cases to the global knob (DEFAULT_CASES_PER_AGENT, 5 by
    # default) — it can't recover the stale entry's original 2, that's gone once fulfilled. 3
    # items/round covers 5 in 2 rounds, well under the round ceiling, with no top-up needed.
    fresh_payload = json.dumps([
        {"input": {"A": {"confidence": "high"}}, "gold": {"overall_confidence": "high", "notes": "fresh1"}},
        {"input": {"A": {"confidence": "low"}}, "gold": {"overall_confidence": "low", "notes": "fresh2"}},
        {"input": {"A": {"confidence": "medium"}}, "gold": {"overall_confidence": "medium", "notes": "fresh3"}},
    ])
    forced_report = await fulfill_plan(
        plan_path, str(map_path), "snap-1", str(tmp_path), "prov-1", "model-1",
        FakeLLMClient(LLMResponse(content=fresh_payload, model="fake")),
        force_agent_ids=["auditor"],
    )

    assert forced_report["synthetic_agent_io/auditor"]["status"] == "fulfilled"
    new_dataset_id = forced_report["synthetic_agent_io/auditor"]["dataset_id"]

    # next_version() reclaims the freed "v1" slot once the stale row is gone — same id is fine,
    # what matters is the OLD cases are gone and the NEW ones (from the forced round) are there.
    from agent_eval_harness.config import DEFAULT_CASES_PER_AGENT
    new_cases = await repository.get_dataset_cases(new_dataset_id)
    assert len(new_cases) == DEFAULT_CASES_PER_AGENT
    notes = {json.loads(c["expected_json"])["notes"] for c in new_cases}
    assert notes == {"fresh1", "fresh2", "fresh3"}  # only fresh content, "bad" is gone


async def test_fulfill_plan_only_agent_ids_scopes_to_requested_agents_only(tmp_path):
    """Stage3Screen's per-agent 'Fulfill Data' button — only_agent_ids must process exactly
    the requested agents' groups and leave every other group entirely absent from the
    report (not fulfilled, not failed, not needs_human — just never touched)."""
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
          - id: agent2.snap
            component: c2
            agent_id: agent2
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
        only_agent_ids=["agent1"],
    )

    assert "snapshot_fixture/agent1" in report
    assert report["snapshot_fixture/agent1"]["status"] == "fulfilled"
    assert "snapshot_fixture/agent2" not in report

    from agent_eval_harness.metrics.suite import load_suite
    suite = load_suite(plan_path)
    by_id = {e.id: e for e in suite.entries}
    assert by_id["agent1.snap"].dataset.ref is not None  # written back
    assert by_id["agent2.snap"].dataset.required is not None  # left untouched, still unfulfilled

