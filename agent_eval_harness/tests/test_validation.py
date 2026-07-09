"""Tests for the evaluation plan validation engine."""
from __future__ import annotations

import textwrap

import pytest

from agent_eval_harness.planning.validation import PlanValidationReport, validate_plan
from agent_eval_harness.store.database import close_db, init_db

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
async def _init_db(tmp_path, monkeypatch):
    monkeypatch.setenv("AEH_DATA_DIR", str(tmp_path))
    await init_db()
    yield
    await close_db()


def _errors(report: PlanValidationReport) -> list[str]:
    """Backward-compat helper: extract the errors list."""
    return report.errors


async def test_validation_clean_plan(tmp_path) -> None:
    """A valid plan with no errors should return an empty errors list and all runnable readiness."""
    plan_content = textwrap.dedent("""
        entries:
          - id: writer.faithfulness
            component: writer
            metric: ragas.faithfulness
            metric_class: llm_judge
            rationale: "valid"
            provenance: rule
            params:
              queries: ["What is the capital of France?"]
    """)
    plan_path = tmp_path / "valid_plan.yaml"
    plan_path.write_text(plan_content, encoding="utf-8")

    report = await validate_plan(plan_path)
    assert isinstance(report, PlanValidationReport)
    assert not _errors(report)
    assert report.readiness["writer.faithfulness"].status == "runnable"


async def test_validation_invalid_schema(tmp_path) -> None:
    """Invalid schema should fail load_suite and return error."""
    plan_content = textwrap.dedent("""
        entries:
          - id: writer.faithfulness
            component: writer
            metric: ragas.faithfulness
            metric_class: invalid_class_name
    """)
    plan_path = tmp_path / "invalid_schema.yaml"
    plan_path.write_text(plan_content, encoding="utf-8")

    report = await validate_plan(plan_path)
    errors = _errors(report)
    assert len(errors) == 1
    assert "Schema validation failed" in errors[0]


async def test_validation_invalid_metric_name(tmp_path) -> None:
    """Unregistered metrics produce errors + metric_not_dispatchable readiness."""
    plan_content = textwrap.dedent("""
        entries:
          - id: writer.unknown_metric
            component: writer
            metric: ragas.unknown_metric_typo
            metric_class: llm_judge
            rationale: "invalid"
            provenance: rule
          - id: planner.bad_assertion
            component: planner
            metric: non_existent_assertion_name
            metric_class: assertion
            rationale: "invalid"
            provenance: rule
    """)
    plan_path = tmp_path / "invalid_metrics.yaml"
    plan_path.write_text(plan_content, encoding="utf-8")

    report = await validate_plan(plan_path)
    errors = _errors(report)
    assert len(errors) == 2
    assert any("not registered in METRIC_REGISTRY" in e for e in errors)
    # Both gates are blocked with metric_not_dispatchable
    assert report.readiness["writer.unknown_metric"].status == "blocked"
    assert "metric_not_dispatchable" in report.readiness["writer.unknown_metric"].reasons
    assert report.readiness["planner.bad_assertion"].status == "blocked"
    assert "metric_not_dispatchable" in report.readiness["planner.bad_assertion"].reasons


async def test_validation_dataset_missing_and_waived(tmp_path) -> None:
    """Test validation of dataset ref exists, or is required/waived."""
    plan_content = textwrap.dedent("""
        entries:
          - id: guard.classifier
            component: guard
            metric: classifier.guard_accuracy
            metric_class: classifier
            dataset:
              ref: nonexistent_dataset_ref
            rationale: "invalid dataset ref"
            provenance: rule
          - id: guard2.classifier
            component: guard
            metric: classifier.guard_accuracy
            metric_class: classifier
            dataset:
              waived: "Skip this for local tests"
            rationale: "waived is valid"
            provenance: rule
          - id: guard3.classifier
            component: guard
            metric: classifier.guard_accuracy
            metric_class: classifier
            dataset:
              required: {kind: guard_classification, min_cases: 40}
            rationale: "required is valid"
            provenance: rule
    """)
    plan_path = tmp_path / "dataset_validation.yaml"
    plan_path.write_text(plan_content, encoding="utf-8")

    report = await validate_plan(plan_path)
    errors = _errors(report)
    # nonexistent ref error + dataset_unfulfilled (guard3) + both missing entry_point
    assert any("nonexistent_dataset_ref" in e for e in errors)
    assert any("guard_classification' is unfulfilled" in e for e in errors)
    # guard3 has unfulfilled dataset → blocked
    assert report.readiness["guard3.classifier"].status == "blocked"
    assert "dataset_unfulfilled" in report.readiness["guard3.classifier"].reasons


async def test_validation_needs_human_and_unknown(tmp_path) -> None:
    """Leftover needs_human markers or unknown metric types must trigger validation failure."""
    plan_content = textwrap.dedent("""
        entries:
          - id: unknown_comp.unknown
            component: unknown_comp
            metric: unknown
            metric_class: assertion
            rationale: "needs review"
            provenance: llm_suggested
            status: needs_human
    """)
    plan_path = tmp_path / "needs_human.yaml"
    plan_path.write_text(plan_content, encoding="utf-8")

    report = await validate_plan(plan_path)
    errors = _errors(report)
    assert len(errors) == 2
    assert any("carries leftover 'needs_human' status marker" in e for e in errors)
    assert any("has 'unknown' metric placeholder" in e for e in errors)


async def test_validation_no_queries_blocked(tmp_path) -> None:
    """Gates with no queries and no dataset.ref are blocked with no_queries reason."""
    plan_content = textwrap.dedent("""
        entries:
          - id: writer.faithfulness
            component: writer
            metric: ragas.faithfulness
            metric_class: llm_judge
            rationale: "no queries"
            provenance: llm_suggested
    """)
    plan_path = tmp_path / "no_queries.yaml"
    plan_path.write_text(plan_content, encoding="utf-8")

    report = await validate_plan(plan_path)
    assert report.readiness["writer.faithfulness"].status == "blocked"
    assert "no_queries" in report.readiness["writer.faithfulness"].reasons


async def test_validation_real_todo_placeholder_blocks_no_queries(tmp_path) -> None:
    """A real generated query placeholder (not just the bare '<TODO>') must still be
    recognized as a bad query, so it doesn't silently pass readiness."""
    plan_content = textwrap.dedent("""
        entries:
          - id: architecture.allowed_downstream
            component: architecture
            metric: allowed_downstream
            metric_class: assertion
            rationale: "orchestrator fan-out"
            provenance: rule
            params:
              allowed: ["auditor"]
              queries: ["<TODO: add a representative query for this target>"]
    """)
    plan_path = tmp_path / "real_todo.yaml"
    plan_path.write_text(plan_content, encoding="utf-8")

    report = await validate_plan(plan_path)
    assert report.readiness["architecture.allowed_downstream"].status == "blocked"
    assert "no_queries" in report.readiness["architecture.allowed_downstream"].reasons


async def test_validation_dataset_unreviewed_blocks_ref(tmp_path) -> None:
    """A ref pointing at a real dataset that still has only synthetic cases left is
    blocked with dataset_unreviewed."""
    from agent_eval_harness.datasets.types import DatasetCase
    from agent_eval_harness.store import repository

    await repository.insert_dataset_cases_bulk("qa_unreviewed_v1", [
        DatasetCase(id="c1", dataset="qa_unreviewed_v1", kind="qa_testset",
                    input={"query": "q"}, expected=None, labels=None, provenance="synthetic"),
    ])
    await repository.insert_dataset_metadata("qa_unreviewed_v1", "qa_testset", min_cases=1)

    plan_content = textwrap.dedent("""
        entries:
          - id: writer.faithfulness
            component: writer
            metric: ragas.faithfulness
            metric_class: llm_judge
            dataset:
              ref: qa_unreviewed_v1
            rationale: "valid"
            provenance: rule
    """)
    plan_path = tmp_path / "unreviewed.yaml"
    plan_path.write_text(plan_content, encoding="utf-8")

    report = await validate_plan(plan_path)
    assert report.readiness["writer.faithfulness"].status == "blocked"
    assert "dataset_unreviewed" in report.readiness["writer.faithfulness"].reasons


async def test_validation_invalid_dataset_kind(tmp_path) -> None:
    """Invented dataset kinds are flagged as invalid_dataset_kind."""
    plan_content = textwrap.dedent("""
        entries:
          - id: ret.faithfulness
            component: ret
            metric: ragas.faithfulness
            metric_class: llm_judge
            dataset:
              required: {kind: retrieval_grounded_outputs, min_cases: 20}
            rationale: "LLM invented kind"
            provenance: llm_suggested
            params:
              queries: ["q"]
    """)
    plan_path = tmp_path / "bad_kind.yaml"
    plan_path.write_text(plan_content, encoding="utf-8")

    report = await validate_plan(plan_path)
    # retrieval_grounded_outputs is not in DATASET_KINDS
    reasons = report.readiness["ret.faithfulness"].reasons
    assert "invalid_dataset_kind" in reasons or "dataset_unfulfilled" in reasons
    assert any("retrieval_grounded_outputs" in e for e in _errors(report))
