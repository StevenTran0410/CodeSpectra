"""CLI smoke tests — invoke cli.main() directly (no subprocess), capture stdout."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent_eval_harness import cli
from agent_eval_harness.store.database import init_db

_T1_MAP = str(Path(__file__).parent.parent / "test_targets" / "linear_rag" / "system_map.yaml")


@pytest.fixture(autouse=True)
def _restore_shared_db_after_cli_closes_it():
    """cli.main() runs its own init_db()/close_db() lifecycle (correct for a
    real standalone process) — but that close_db() call tears down the SAME
    module-level connection the session-scoped test fixture opened, breaking
    every test that runs afterward. Reopen it once this test's process-lifecycle
    simulation is done.
    """
    yield
    asyncio.run(init_db())


def test_cli_run_linear_rag(capsys) -> None:
    exit_code = cli.main(
        [
            "run",
            "--target",
            "test_targets.linear_rag.pipeline:build_pipeline",
            "--map",
            _T1_MAP,
            "--query",
            "What is the vacation policy?",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "retriever" in out
    assert "writer" in out
    assert "mapped" in out
    assert "run completed" in out


def test_cli_run_prints_unmatched_warning_for_broken_map(tmp_path, capsys) -> None:
    broken_map = tmp_path / "broken_map.yaml"
    broken_map.write_text(
        """
target_system_id: linear_rag
discrepancies: []
components:
  - id: retriever
    role: retrieval_agent
    entry_point: "test_targets.linear_rag.pipeline:run_retrieve"
    span_match: [{component_name: "nonexistent_1"}]
    constraints: []
    upstream: []
    downstream: [writer]
  - id: writer
    role: writer
    entry_point: "test_targets.linear_rag.pipeline:run_write"
    span_match: [{component_name: "nonexistent_2"}]
    constraints: []
    upstream: [retriever]
    downstream: []
""",
        encoding="utf-8",
    )

    exit_code = cli.main(
        [
            "run",
            "--target",
            "test_targets.linear_rag.pipeline:build_pipeline",
            "--map",
            str(broken_map),
            "--query",
            "What is the vacation policy?",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "unmatched" in out.lower()
    assert "mapped 0/" in out
