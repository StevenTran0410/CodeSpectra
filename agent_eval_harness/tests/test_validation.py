"""Tests for Evaluation Plan Validation Engine (CS-265)."""
from __future__ import annotations

import textwrap

import pytest

from agent_eval_harness.planning.validation import validate_plan
from agent_eval_harness.store.database import close_db, init_db

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
async def _init_db(tmp_path, monkeypatch):
    monkeypatch.setenv("AEH_DATA_DIR", str(tmp_path))
    await init_db()
    yield
    await close_db()


async def test_validation_clean_plan(tmp_path) -> None:
    """A valid plan with no errors should return an empty list of errors."""
    plan_content = textwrap.dedent("""
        entries:
          - id: writer.faithfulness
            component: writer
            metric: ragas.faithfulness
            metric_class: llm_judge
            rationale: "valid"
            provenance: rule
    """)
    plan_path = tmp_path / "valid_plan.yaml"
    plan_path.write_text(plan_content, encoding="utf-8")

    errors = await validate_plan(plan_path)
    assert not errors


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

    errors = await validate_plan(plan_path)
    assert len(errors) == 1
    assert "Schema validation failed" in errors[0]


async def test_validation_invalid_metric_name(tmp_path) -> None:
    """Assert error returned for unregistered metric names."""
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

    errors = await validate_plan(plan_path)
    assert len(errors) == 2
    assert any("LLM judge metric" in err and "is not registered" in err for err in errors)
    assert any("assertion metric" in err and "does not exist in registry" in err for err in errors)


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

    errors = await validate_plan(plan_path)
    assert len(errors) == 2
    assert any(
        "dataset reference 'nonexistent_dataset_ref' does not exist" in err
        for err in errors
    )
    assert any(
        "dataset requirement of kind 'guard_classification' is unfulfilled" in err
        for err in errors
    )


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

    errors = await validate_plan(plan_path)
    assert len(errors) == 2
    assert any("carries leftover 'needs_human' status marker" in err for err in errors)
    assert any("has 'unknown' metric placeholder" in err for err in errors)
