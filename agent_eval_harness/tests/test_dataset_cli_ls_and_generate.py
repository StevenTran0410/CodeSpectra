import asyncio

import pytest

from agent_eval_harness import cli
from agent_eval_harness.store.database import init_db


@pytest.fixture(autouse=True)
def _restore_shared_db_after_cli_closes_it():
    yield
    asyncio.run(init_db())

def test_cli_dataset_commands(tmp_path, capsys):
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


def test_cli_dataset_generate_qa_testset_missing_options(tmp_path):
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


def test_cli_dataset_generate_qa_testset_with_local_mocked(tmp_path, monkeypatch):
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

