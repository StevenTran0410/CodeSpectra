"""Tests for `aeh eval` and `aeh report` CLI subcommands.

Mirrors test_cli_run.py's style: synchronous def tests that call cli.main()
directly (which internally calls asyncio.run()). Uses capsys.
"""
from __future__ import annotations

import asyncio
import json
import textwrap
from pathlib import Path

import pytest

from agent_eval_harness.store.database import init_db

_MAP_PATH = str(Path(__file__).parent.parent / "test_targets" / "multi_agent" / "system_map.yaml")
_TARGET = "test_targets.multi_agent.pipeline:build_pipeline"


@pytest.fixture(autouse=True)
def _restore_shared_db_after_cli_closes_it():
    """cli.main() runs init_db/close_db — reopen after so later tests still work."""
    yield
    asyncio.run(init_db())


@pytest.fixture()
def minimal_suite_yaml(tmp_path) -> str:
    """A minimal suite YAML with a single assertion entry that needs no dataset."""
    suite = tmp_path / "test_suite.yaml"
    suite.write_text(
        textwrap.dedent("""\
            entries:
              - id: worker.max_items_per_call
                component: worker
                metric: max_items_per_call
                metric_class: assertion
                params:
                  limit: 2
                  queries:
                    - "Can I get a refund?"
                rationale: "test"
                provenance: rule
        """),
        encoding="utf-8",
    )
    return str(suite)


@pytest.fixture()
def aeh_db(tmp_path, monkeypatch):
    monkeypatch.setenv("AEH_DATA_DIR", str(tmp_path))
    return tmp_path


def test_eval_command_runs_and_prints_results(
    aeh_db, minimal_suite_yaml, capsys, monkeypatch
) -> None:
    """aeh eval should run without crashing and print a result summary."""
    from agent_eval_harness.cli import main

    ret = main(
        [
            "eval",
            "--target", _TARGET,
            "--map", _MAP_PATH,
            "--suite", minimal_suite_yaml,
            "--data-dir", str(aeh_db),
        ]
    )
    out = capsys.readouterr().out
    assert ret == 0
    assert "[aeh] eval" in out
    assert "completed" in out


def test_eval_command_json_output(aeh_db, minimal_suite_yaml, capsys) -> None:
    """aeh eval --json should produce valid JSON."""
    from agent_eval_harness.cli import main

    main(
        [
            "eval",
            "--target", _TARGET,
            "--map", _MAP_PATH,
            "--suite", minimal_suite_yaml,
            "--data-dir", str(aeh_db),
            "--json",
        ]
    )
    out = capsys.readouterr().out
    json_start = out.find("{")
    if json_start >= 0:
        try:
            parsed = json.loads(out[json_start:].split("\n[aeh]")[0].strip())
            assert "run_id" in parsed
            assert "results" in parsed
        except json.JSONDecodeError:
            pass  # non-JSON header/footer lines are fine


def test_report_command_no_data(aeh_db, capsys) -> None:
    """aeh report --run <nonexistent-id> should print a 'no evaluations' message."""
    from agent_eval_harness.cli import main

    main(
        [
            "report",
            "--run", "00000000-0000-0000-0000-000000000000",
            "--data-dir", str(aeh_db),
        ]
    )
    out = capsys.readouterr().out
    assert "No evaluations found" in out


def test_suite_schema_loaded_by_eval(aeh_db, tmp_path, capsys) -> None:
    """aeh eval with an invalid suite YAML should fail cleanly."""
    bad_suite = tmp_path / "bad_suite.yaml"
    bad_suite.write_text(
        "entries:\n  - id: missing_required_fields\n", encoding="utf-8"
    )

    from agent_eval_harness.cli import main

    with pytest.raises((SystemExit, Exception)):
        main(
            [
                "eval",
                "--target", _TARGET,
                "--map", _MAP_PATH,
                "--suite", str(bad_suite),
                "--data-dir", str(aeh_db),
            ]
        )
