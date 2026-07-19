"""CS-302 AC1: a derived metamorphic_relation case goes the whole way — derive → sweep → a real
pass/fail in the report — not "the case exists in the DB". Uses the self-consistency relation
(transform=null) so it yields a determinate verdict off the reviewed source gold, no agent run."""
from __future__ import annotations

from pathlib import Path

import pytest

from agent_eval_harness.datasets.metamorphic_derive import (
    approve_relation,
    derive_dataset,
    preview_relation,
)
from agent_eval_harness.datasets.types import DatasetCase
from agent_eval_harness.mapping.system_map import Component, SystemMap, save_system_map
from agent_eval_harness.metrics.sweep import run_sweep
from agent_eval_harness.store import repository
from agent_eval_harness.store.database import close_db, init_db

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
async def _setup_db(tmp_path, monkeypatch):
    monkeypatch.setenv("AEH_DATA_DIR", str(tmp_path))
    await init_db()
    yield
    await close_db()


def _write_map(tmp_path: Path) -> str:
    system_map = SystemMap(
        target_system_id="acme",
        components=[Component(id="judge", role="validator", entry_point="a:Judge")],
    )
    map_path = tmp_path / "map.yaml"
    save_system_map(system_map, map_path)
    return str(map_path)


def _write_suite(tmp_path: Path, dataset_ref: str) -> str:
    suite_path = tmp_path / "plan.yaml"
    suite_path.write_text(
        "entries:\n"
        "  - id: judge.metamorphic\n"
        "    component: judge\n"
        "    agent_id: judge\n"
        "    metric: metamorphic_relation\n"
        "    metric_class: assertion\n"
        f"    dataset: {{ref: {dataset_ref}}}\n"
        "    rationale: r\n"
        "    provenance: rule\n",
        encoding="utf-8",
    )
    return str(suite_path)


async def _seed_reviewed_source(dataset_id: str) -> None:
    """A reviewed fan_in_judge cohort whose gold carries section_scores + weakest_sections —
    one case whose weakest list is a valid argmin-3, one that wrongly omits a lower-scoring key."""
    cases = [
        DatasetCase(
            id=repository.new_id(), dataset=dataset_id, kind="synthetic_agent_io",
            input={"payload": "valid"},
            expected={"section_scores": {"A": 1, "B": 2, "C": 3, "D": 4},
                      "weakest_sections": ["A", "B", "C"]},
            labels={"archetype": "fan_in_judge"}, provenance="generated+reviewed",
        ),
        DatasetCase(
            id=repository.new_id(), dataset=dataset_id, kind="synthetic_agent_io",
            input={"payload": "invalid"},
            expected={"section_scores": {"D": 49, "E": 58, "J": 61, "G": 63},
                      "weakest_sections": ["D", "E", "G"]},  # omits J(61) < G(63)
            labels={"archetype": "fan_in_judge"}, provenance="generated+reviewed",
        ),
    ]
    await repository.insert_dataset_cases_bulk(dataset_id, cases)
    await repository.insert_dataset_metadata(dataset_id, "synthetic_agent_io", min_cases=1)


async def test_metamorphic_relation_case_scored_end_to_end_yields_pass_and_fail(tmp_path: Path):
    source_id = "mr_ac1_source_v1"
    await _seed_reviewed_source(source_id)

    preview = await preview_relation("weakest_is_argmin", source_id)
    await approve_relation("weakest_is_argmin", source_id, samples=preview["samples"])
    derived = await derive_dataset(
        "weakest_is_argmin", source_id, dataset_name="mr_ac1_derived_v1", spot_audit_pct=0.0
    )
    assert derived["minted_count"] == 2

    map_path = _write_map(tmp_path)
    suite_path = _write_suite(tmp_path, derived["dataset_id"])

    result = await run_sweep(
        target="acme", map_path=map_path, suite_path=suite_path,
        llm_client=None, source="live",
    )

    scored = [r for r in result.results if r.metric_name == "assertion.metamorphic_relation"]
    assert len(scored) == 2
    verdicts = sorted(r.passed for r in scored)  # one True, one False — a real verdict, not None
    assert verdicts == [False, True]
    for r in scored:
        assert r.details.get("checked_against") == "source_expected_output"
