"""Test that active defects do not leak into insert_run for non-test-target runs (CS-266)."""
from __future__ import annotations

import json

import pytest

from agent_eval_harness.store import repository
from agent_eval_harness.store.database import close_db, init_db

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
async def _setup_db(tmp_path, monkeypatch):
    monkeypatch.setenv("AEH_DATA_DIR", str(tmp_path))
    await init_db()
    yield
    await close_db()


async def test_insert_run_defect_leak_prevention(monkeypatch) -> None:
    # Set a defect environment variable
    monkeypatch.setenv("AEH_DEFECT_GUARD_LEAK", "1")

    # 1. Insert a run targeting a generic system, passing no active_defects.
    # It should not have active defects populated (active_defects should be None).
    run_id = await repository.insert_run("my_real_target")
    run = await repository.get_run(run_id)
    assert run is not None
    assert run.get("active_defects") is None

    # 2. Insert a run explicitly passing active_defects (like what the CLI does for T1/T2/T3)
    run_id_explicit = await repository.insert_run(
        "my_test_target", active_defects=["DEFECT_GUARD_LEAK"]
    )
    run_explicit = await repository.get_run(run_id_explicit)
    assert run_explicit is not None
    assert json.loads(run_explicit["active_defects"]) == ["DEFECT_GUARD_LEAK"]
