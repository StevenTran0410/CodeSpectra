"""Consolidated misc dataset tests: CLI, decomposition_gold, guard_classification, perturbation, qa_testset, sufficiency_labeled, versioning, recorded_report_replay."""
import asyncio
import json
import pathlib
import textwrap

import pytest

from agent_eval_harness import cli
from agent_eval_harness.datasets.fulfillment import fulfill_plan
from agent_eval_harness.datasets.generators.decomposition_gold import generate as generate_decomposition_gold
from agent_eval_harness.datasets.generators.guard_classification import generate as generate_guard_classification
from agent_eval_harness.datasets.generators.qa_testset import (
    _BACKENDS,
    QATestsetBackend,
    generate as generate_qa_testset,
)
from agent_eval_harness.datasets.generators.recorded_report_replay import generate as generate_recorded_report_replay
from agent_eval_harness.datasets.generators.sufficiency_labeled import generate as generate_sufficiency_labeled
from agent_eval_harness.datasets.perturbation import (
    degrade_section,
    drop_section,
    inject_conflict,
    oversize_section,
)
from agent_eval_harness.datasets.types import DatasetCase
from agent_eval_harness.datasets.versioning import next_version
from agent_eval_harness.llm.client import LLMResponse
from agent_eval_harness.llm.fake_client import FakeLLMClient
from agent_eval_harness.planning.contract import EvaluationContract
from agent_eval_harness.planning.report import AgentPlanReport, EvaluationPlanReport, save_plan_report
from agent_eval_harness.store import repository
from agent_eval_harness.store.database import init_db
from tests._stubs import FakeCodeSpectraClient


# --- cli: dataset ls / generate ---

@pytest.fixture
def _restore_shared_db_after_cli_closes_it():
    yield
    asyncio.run(init_db())


def test_cli_dataset_commands(tmp_path, capsys, _restore_shared_db_after_cli_closes_it):
    config_file = tmp_path / "guard_config.yaml"
    config_file.write_text("""
dataset_name: t2_cli_guard
categories:
  - name: too_short
    kind: mechanical
    count: 25
  - name: gibberish
    kind: mechanical
    count: 25
  - name: valid
    kind: mechanical
    count: 35
""", encoding="utf-8")

    exit_code_gen = cli.main([
        "dataset", "generate",
        "--kind", "guard_classification",
        "--config", str(config_file),
        "--seed", "42"
    ])
    assert exit_code_gen == 0

    out_gen = capsys.readouterr().out
    assert "Generated 85 cases" in out_gen
    assert "t2_cli_guard_v1" in out_gen

    exit_code_ls = cli.main([
        "dataset", "ls"
    ])
    assert exit_code_ls == 0

    out_ls = capsys.readouterr().out
    assert "t2_cli_guard_v1" in out_ls
    assert "Total Cases: 85" in out_ls
    assert "Status:      pending review" in out_ls


def test_cli_dataset_generate_qa_testset_missing_options(tmp_path, _restore_shared_db_after_cli_closes_it):
    config_file = tmp_path / "qa_config.yaml"
    config_file.write_text("""
dataset_name: t2_cli_qa
corpus_paths: ["non_existent_path"]
count: 5
backend: deepeval
""", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        cli.main([
            "dataset", "generate",
            "--kind", "qa_testset",
            "--config", str(config_file)
        ])
    assert "requires either --use-local-embedding or --embedding-provider-id" in str(exc_info.value)


def test_cli_dataset_generate_qa_testset_with_local_mocked(tmp_path, monkeypatch, _restore_shared_db_after_cli_closes_it):
    config_file = tmp_path / "qa_config.yaml"
    config_file.write_text("""
dataset_name: t2_cli_qa
corpus_paths: ["non_existent_path"]
count: 5
backend: deepeval
""", encoding="utf-8")

    # Mock next_version to return static version
    # Mock get_generator to return a mock generator function
    # Mock init_db and close_db to be no-ops
    async def mock_next_version(base):
        return f"{base}_v1"

    async def mock_generator(config, llm_client, seed, embedding_client=None):
        assert embedding_client is not None
        assert embedding_client._use_local is True
        return []

    monkeypatch.setattr("agent_eval_harness.datasets.versioning.next_version", mock_next_version)
    monkeypatch.setattr("agent_eval_harness.datasets.registry.get_generator", lambda kind: mock_generator)

    exit_code = cli.main([
        "dataset", "generate",
        "--kind", "qa_testset",
        "--config", str(config_file),
        "--use-local-embedding",
        "--backend-url", "http://localhost:8000",
        "--backend-token", "some-token"
    ])
    assert exit_code == 0


# --- decomposition_gold ---

@pytest.mark.asyncio
async def test_decomposition_gold_splits(tmp_path):
    map_path_2 = tmp_path / "system_map_2.yaml"
    map_path_2.write_text("""
target_system_id: test_system
discrepancies: []
components:
  - id: planner
    role: orchestrator
    entry_point: "dummy:entry"
    constraints:
      - name: max_items_per_call
        value: 2
        source: "test"
""", encoding="utf-8")

    map_path_3 = tmp_path / "system_map_3.yaml"
    map_path_3.write_text("""
target_system_id: test_system
discrepancies: []
components:
  - id: planner
    role: orchestrator
    entry_point: "dummy:entry"
    constraints:
      - name: max_items_per_call
        value: 3
        source: "test"
""", encoding="utf-8")

    config_2 = {
        "dataset_name": "test_decomp_2",
        "system_map_path": str(map_path_2),
        "component_id": "planner",
        "count": 3,
        "max_items_constraint_name": "max_items_per_call",
    }

    config_3 = {
        "dataset_name": "test_decomp_3",
        "system_map_path": str(map_path_3),
        "component_id": "planner",
        "count": 3,
        "max_items_constraint_name": "max_items_per_call",
    }

    # over_limit's intent count (5) must exceed both maps' max_items_per_call limit to test the split.
    mock_clean = '[{"query": "Do X and Y", "intents": ["I1", "I2"]}]'
    mock_rambling = '[{"query": "Please do X", "intents": ["I1"]}]'
    mock_over_limit = '[{"query": "Do 1 2 3 4 5", "intents": ["I1", "I2", "I3", "I4", "I5"]}]'

    fake_client_2 = FakeLLMClient([
        LLMResponse(content=mock_clean, model="fake"),
        LLMResponse(content=mock_rambling, model="fake"),
        LLMResponse(content=mock_over_limit, model="fake")
    ])

    fake_client_3 = FakeLLMClient([
        LLMResponse(content=mock_clean, model="fake"),
        LLMResponse(content=mock_rambling, model="fake"),
        LLMResponse(content=mock_over_limit, model="fake")
    ])

    cases_2 = await generate_decomposition_gold(config_2, llm_client=fake_client_2)
    assert len(cases_2) == 3

    # Over limit cases are tagged with category="over_limit" or similar
    over_limit_case_2 = [c for c in cases_2 if c.labels.get("category") == "over_limit"][0]
    assert len(over_limit_case_2.expected["intents"]) == 5
    assert over_limit_case_2.expected["call_split"] == [["I1", "I2"], ["I3", "I4"], ["I5"]]

    cases_3 = await generate_decomposition_gold(config_3, llm_client=fake_client_3)
    assert len(cases_3) == 3
    over_limit_case_3 = [c for c in cases_3 if c.labels.get("category") == "over_limit"][0]
    assert len(over_limit_case_3.expected["intents"]) == 5
    assert over_limit_case_3.expected["call_split"] == [["I1", "I2", "I3"], ["I4", "I5"]]


@pytest.mark.asyncio
async def test_decomposition_gold_missing_constraint_skips_over_limit(tmp_path):
    """Component with no max_items_per_call constraint: no raise, no over_limit category."""
    map_path = tmp_path / "system_map_no_limit.yaml"
    map_path.write_text("""
target_system_id: test_system
discrepancies: []
components:
  - id: planner
    role: orchestrator
    entry_point: "dummy:entry"
    constraints: []
""", encoding="utf-8")

    config = {
        "dataset_name": "test_decomp_no_limit",
        "system_map_path": str(map_path),
        "component_id": "planner",
        "count": 4,
    }

    mock_clean = '[{"query": "Do X and Y", "intents": ["I1", "I2"]}]'
    mock_rambling = '[{"query": "Please do X", "intents": ["I1"]}]'
    fake_client = FakeLLMClient([
        LLMResponse(content=mock_clean, model="fake"),
        LLMResponse(content=mock_rambling, model="fake"),
    ])

    cases = await generate_decomposition_gold(config, llm_client=fake_client)
    categories = {c.labels.get("category") for c in cases}
    assert categories == {"clean", "rambling"}
    assert "over_limit" not in categories
    assert all("call_split" not in c.expected for c in cases)


# --- guard_classification ---

@pytest.mark.asyncio
async def test_guard_classification_mechanical():
    # Only mechanical categories
    config = {
        "dataset_name": "test_mech_v1",
        "categories": [
            {"name": "too_short", "kind": "mechanical", "count": 25},
            {"name": "gibberish", "kind": "mechanical", "count": 25},
            # We must make sure valid cases make up >= 40% of the total
            {"name": "valid", "kind": "mechanical", "count": 35}
        ]
    }
    # Mechanical generation needs no LLM client
    cases = await generate_guard_classification(config, llm_client=None, seed=42)
    assert len(cases) == 85

    too_short_cases = [c for c in cases if c.labels.get("category") == "too_short"]
    assert len(too_short_cases) == 25
    for c in too_short_cases:
        assert c.expected == {"verdict": "reject", "category": "too_short"}
        assert c.provenance == "synthetic"

    valid_cases = [c for c in cases if c.labels.get("category") == "valid"]
    assert len(valid_cases) == 35
    for c in valid_cases:
        assert c.expected == {"verdict": "pass"}
        assert c.provenance == "synthetic"

@pytest.mark.asyncio
async def test_guard_classification_validation_failure():
    # If category count < 25, should raise ValueError
    config = {
        "dataset_name": "test_fail_v1",
        "categories": [
            {"name": "too_short", "kind": "mechanical", "count": 24}
        ]
    }
    with pytest.raises(ValueError, match="less than the required minimum of 25"):
        await generate_guard_classification(config, llm_client=None)

    # If valid cases ratio < 40%, should raise ValueError
    config_ratio = {
        "dataset_name": "test_ratio_v1",
        "categories": [
            {"name": "too_short", "kind": "mechanical", "count": 30},
            {"name": "gibberish", "kind": "mechanical", "count": 30},
            {"name": "valid", "kind": "mechanical", "count": 25}  # 25 / 85 = 29.4% < 40%
        ]
    }
    with pytest.raises(ValueError, match="does not meet the requirement of having >= 40% valid"):
        await generate_guard_classification(config_ratio, llm_client=None)

@pytest.mark.asyncio
async def test_guard_classification_semantic():
    config = {
        "dataset_name": "test_sem_v1",
        "categories": [
            {
                "name": "off_topic",
                "kind": "semantic",
                "count": 30,
                "rubric": "not related to company",
            },
            {"name": "borderline_valid", "kind": "semantic", "count": 25}
        ]
    }

    # Fake LLM client to return custom queries
    mock_off_topic = '["where is the sun", "who is the President"]'
    mock_borderline = (
        '["is code review required for vacation requests", "how to submit sick leave"]'
    )

    fake_client = FakeLLMClient([
        LLMResponse(content=mock_off_topic, model="fake-model"),
        LLMResponse(content=mock_borderline, model="fake-model")
    ])

    cases = await generate_guard_classification(config, llm_client=fake_client)
    # We requested 30 + 25 = 55 cases.
    assert len(cases) == 55

    off_topic_cases = [c for c in cases if c.labels.get("category") == "off_topic"]
    # The mock response list has only 2 items; the generator must cycle it to reach the full count.
    assert len(off_topic_cases) == 30
    for c in off_topic_cases:
        assert c.expected == {"verdict": "reject", "category": "off_topic"}
        assert c.provenance == "synthetic"


# --- perturbation ---

def test_drop_section():
    all_sections = {"foo": {"repo_name": "test", "content": "hello"}, "bar": {"content": "world"}}
    res = drop_section(all_sections, "foo")
    assert "foo" not in res
    assert "bar" in res
    assert "foo" in all_sections  # Verify caller dict is not mutated
    assert json.loads(json.dumps(res)) == res  # Verify serialization


def test_degrade_section_dict_field_generic():
    """Generic dict-shaped field: every leaf degrades to its own typed empty — no field-name
    special-casing (repo_name/confidence/blind_spots/domain/runtime_type do not exist here)."""
    all_sections = {
        "analysis": {
            "tags": ["a", "b"],
            "nested": {"x": 1},
            "note": "hello",
            "score": 42,
        }
    }
    res = degrade_section(all_sections, "analysis")
    assert res["analysis"]["tags"] == []
    assert res["analysis"]["nested"] == {}
    assert res["analysis"]["note"] == ""
    assert res["analysis"]["score"] is None

    assert all_sections["analysis"]["note"] == "hello"  # caller dict not mutated
    assert json.loads(json.dumps(res)) == res  # serializable


def test_degrade_section_missing_field_is_noop():
    all_sections = {"analysis": {"note": "hello"}}
    res = degrade_section(all_sections, "not_present")
    assert res == all_sections


def test_degrade_section_list_typed_field_yields_empty_list_not_dict():
    """multi_agent's WorkerComponent/JudgeComponent output {"context": list[str]} — a
    list-typed top-level field must degrade to [], never a {} stub (the :23-24 shape bug)."""
    case_like = {"query": "What is the vacation policy?", "context": ["doc one", "doc two"]}
    res = degrade_section(case_like, "context")
    assert res["context"] == []
    assert isinstance(res["context"], list)
    assert case_like["context"] == ["doc one", "doc two"]  # caller dict not mutated


def test_degrade_section_str_typed_field_yields_empty_string():
    case_like = {"query": "hello", "answer": "some text"}
    res = degrade_section(case_like, "answer")
    assert res["answer"] == ""


def test_degrade_section_scalar_typed_field_yields_none():
    case_like = {"query": "hello", "count": 3}
    res = degrade_section(case_like, "count")
    assert res["count"] is None


def test_inject_conflict_flat_string_fields():
    """Token injected directly into flat top-level string fields (no nested dict required)."""
    input_data = {"purpose": "Original", "description": "Also original"}
    res = inject_conflict(input_data, ["purpose", "description"], "__mr_conflict__")
    assert res["purpose"] == "Original __mr_conflict__"
    assert res["description"] == "Also original __mr_conflict__"


def test_inject_conflict_missing_field_gets_token_set_directly():
    input_data = {"purpose": "Original"}
    res = inject_conflict(input_data, ["purpose", "new_field"], "__mr_conflict__")
    assert res["new_field"] == "__mr_conflict__"


def test_oversize_section():
    all_sections = {
        "a": {"repo_name": "test", "purpose": "Short purpose"},
    }
    res = oversize_section(all_sections, "a", target_len=1000)
    assert len(res["a"]["purpose"]) == 1000
    assert "Short purpose" in res["a"]["purpose"]

    assert len(all_sections["a"]["purpose"]) < 100  # Verify caller dict is not mutated
    assert json.loads(json.dumps(res)) == res  # Verify serialization


def test_grep_gate_no_codespectra_literals_in_perturbation() -> None:
    """CS-302 AC4 / Nguyen tac so 0 grep gate: zero CodeSpectra report literals remain in
    perturbation.py — the generic isinstance fallback is the whole degrade mechanism, and the
    oversize skip-list is caller/config-supplied. Ban the LITERAL, not just the `if k ==` form:
    re-adding `confidence`/`repo_name` anywhere must turn this red."""
    forbidden = [
        "blind_spots",
        "domain",
        "runtime_type",
        "section_scores",
        "weakest_sections",
        "confidence",
        "repo_name",
    ]
    source_path = (
        pathlib.Path(__file__).parent.parent
        / "agent_eval_harness"
        / "datasets"
        / "perturbation.py"
    )
    content = source_path.read_text(encoding="utf-8")

    found = [literal for literal in forbidden if literal in content]
    assert not found, f"Forbidden CodeSpectra-report literal(s) found in perturbation.py: {found}"

    # No single-letter section keys (A-L) used as dict literals/params anywhere.
    import re

    single_letter_keys = re.findall(r'["\']([A-L])["\']\s*[:,)]', content)
    assert not single_letter_keys, f"Single-letter section keys found: {single_letter_keys}"


def test_oversize_section_skip_fields_from_caller_not_code_literal() -> None:
    """CS-302 Slice 4: the fields NOT to pad come from the caller (config/contract), never a
    code constant. A caller-supplied skip-list must protect exactly those fields."""
    all_sections = {"a": {"tag": "keep-me", "body": "short"}}
    # Without a skip-list, the first string field ('tag') is the fallback padding target.
    res_default = oversize_section(all_sections, "a", target_len=1000)
    assert len(res_default["a"]["tag"]) == 1000

    # With 'tag' skipped, padding moves to the next string field ('body') instead.
    res_skip = oversize_section(all_sections, "a", target_len=1000, skip_fields=("tag",))
    assert res_skip["a"]["tag"] == "keep-me"
    assert len(res_skip["a"]["body"]) == 1000


# --- qa_testset ---

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
        await generate_qa_testset(config, llm_client=None)

    # Unknown backend name raises ValueError
    config_unknown = {
        "dataset_name": "test_v1",
        "corpus_paths": ["test_targets/linear_rag/corpus/*.txt"],
        "count": 5,
        "backend": "unknown"
    }
    with pytest.raises(ValueError, match="Unknown QA testset backend: unknown"):
        await generate_qa_testset(config_unknown, llm_client=fake_client)

    # Empty/missing corpus_paths raises ValueError
    config_empty_corpus = {
        "dataset_name": "test_v1",
        "corpus_paths": [],
        "count": 5,
        "backend": "deepeval"
    }
    with pytest.raises(ValueError, match="No corpus files found"):
        await generate_qa_testset(config_empty_corpus, llm_client=fake_client)

    # Non-existent corpus path raises ValueError
    config_non_existent = {
        "dataset_name": "test_v1",
        "corpus_paths": ["non_existent_folder/*.txt"],
        "count": 5,
        "backend": "deepeval"
    }
    with pytest.raises(ValueError, match="No corpus files found"):
        await generate_qa_testset(config_non_existent, llm_client=fake_client)


def test_qa_testset_backends_registry_has_no_mock():
    """Mock backend was removed — qa_testset must always exercise a real LLM-backed
    synthesis library (deepeval or ragas), never hardcoded placeholder text."""
    assert set(_BACKENDS.keys()) == {"deepeval", "ragas"}
    for backend in _BACKENDS.values():
        assert isinstance(backend, QATestsetBackend)


# --- sufficiency_labeled ---

@pytest.mark.asyncio
async def test_sufficiency_labeled_ablation():
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

    config = {
        "dataset_name": "t1_sufficiency_v1",
        "source_dataset_id": "t1_qa_v1"
    }

    cases = await generate_sufficiency_labeled(config, llm_client=None)
    # We expect 3 cases generated: sufficient, insufficient, and partial_context
    assert len(cases) == 3

    suff = [c for c in cases if c.labels.get("variant") == "full_context"][0]
    assert suff.input["query"] == "What is the policy?"
    assert suff.input["context"] == ["context 1", "context 2"]
    assert suff.expected == {"sufficient": True}
    assert suff.provenance == "synthetic"

    insuff = [c for c in cases if c.labels.get("variant") == "no_context"][0]
    assert insuff.input["query"] == "What is the policy?"
    assert insuff.input["context"] == []
    assert insuff.expected == {"sufficient": False}
    assert insuff.provenance == "synthetic"

    partial = [c for c in cases if c.labels.get("variant") == "partial_context"][0]
    assert partial.input["query"] == "What is the policy?"
    assert partial.input["context"] == ["context 1"]
    assert partial.expected is None
    assert partial.provenance == "synthetic"


# --- versioning ---

@pytest.mark.asyncio
async def test_dataset_versioning_and_immutability():
    v1 = await next_version("t2_guard")
    assert v1 == "t2_guard_v1"

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

    v2 = await next_version("t2_guard")
    assert v2 == "t2_guard_v2"

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


# --- recorded_report_replay ---

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
    cases = await generate_recorded_report_replay(config, None)
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
    cases = await generate_recorded_report_replay(config, None)
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
