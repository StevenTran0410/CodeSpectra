import asyncio

import pytest

from agent_eval_harness import cli
from agent_eval_harness.store.database import init_db


@pytest.fixture(autouse=True)
def _restore_shared_db_after_cli_closes_it():
    yield
    asyncio.run(init_db())

def test_cli_dataset_commands(tmp_path, capsys):
    # 1. Create a config file for mechanical guard classification
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

    # 2. Run generate CLI command
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

    # 3. Run ls CLI command
    exit_code_ls = cli.main([
        "dataset", "ls"
    ])
    assert exit_code_ls == 0

    out_ls = capsys.readouterr().out
    assert "t2_cli_guard_v1" in out_ls
    assert "Total Cases: 85" in out_ls
    assert "Status:      pending review" in out_ls
