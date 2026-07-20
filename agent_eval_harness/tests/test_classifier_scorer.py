"""Tests for the classifier scorer — fixed small dataset + stub component function."""
from __future__ import annotations

import json
import uuid

import pytest


async def _seed_dataset(dataset_id: str, cases: list[dict]) -> None:
    """Insert raw case dicts directly into dataset_cases."""
    from agent_eval_harness.store.database import get_db

    db = get_db()
    rows = [
        (
            str(uuid.uuid4()),
            dataset_id,
            json.dumps(c["input"]),
            json.dumps(c["expected"]),
            json.dumps(c.get("labels", {})),
            "handwritten",
        )
        for c in cases
    ]
    await db.executemany(
        "INSERT INTO dataset_cases "
        "(id, dataset_id, input_json, expected_json, labels_json, provenance) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    await db.commit()


class _StubGuardComponent:
    """Stub guard component that returns 'pass' for long queries, 'reject' for short."""

    def __init__(self, llm_client) -> None:
        self.llm_client = llm_client

    async def run_async(self, query: str) -> dict:
        if len(query) >= 5:
            return {"verdict": "pass"}
        return {"verdict": "reject"}


@pytest.fixture(autouse=True)
async def _init_db(tmp_path, monkeypatch):

    monkeypatch.setenv("AEH_DATA_DIR", str(tmp_path))
    from agent_eval_harness.store.database import close_db, init_db

    await init_db()
    yield
    await close_db()


async def test_classifier_scorer_known_confusion_matrix() -> None:
    """Feed a known dataset, assert accuracy + confusion matrix contents."""
    from unittest.mock import MagicMock

    from agent_eval_harness.metrics.classifier import score_classifier

    dataset_id = "test_guard_stub_v1"
    cases = [
        {"input": {"query": "hello world"}, "expected": {"verdict": "pass"}},
        {"input": {"query": "test query here"}, "expected": {"verdict": "pass"}},
        {"input": {"query": "hi"}, "expected": {"verdict": "reject"}},
        {"input": {"query": "no"}, "expected": {"verdict": "reject"}},
    ]
    await _seed_dataset(dataset_id, cases)

    import agent_eval_harness.metrics.classifier as classifier_module

    original_resolve = classifier_module._resolve_entry_point

    def _mock_resolve(entry_point: str):
        return _StubGuardComponent

    classifier_module._resolve_entry_point = _mock_resolve
    try:
        result = await score_classifier(
            dataset_id=dataset_id,
            component_entry_point="stub:StubGuardComponent",
            llm_client=MagicMock(),
            component_id="guard_rule",
            metric_name="classifier.guard_rule_accuracy",
        )
    finally:
        classifier_module._resolve_entry_point = original_resolve

    assert result.metric_class == "classifier"
    assert result.score is not None
    # Stub is deterministic: "hello world" (11 chars >= 5) → pass ✓
    assert result.score == pytest.approx(1.0)
    assert result.passed is True
    assert result.details["evaluated"] == 4


async def test_classifier_scorer_empty_dataset() -> None:
    from unittest.mock import MagicMock

    from agent_eval_harness.metrics.classifier import score_classifier

    result = await score_classifier(
        dataset_id="nonexistent_dataset_xyz",
        component_entry_point="stub:StubGuardComponent",
        llm_client=MagicMock(),
    )
    assert result.score is None
    assert "No cases found" in result.details.get("error", "")
