"""CI guard: the SQL provenance whitelist in run_eval.py must stay in sync with TRUSTED_PROVENANCE — run_eval.py can't import agent_eval_harness by design, so this duplicate check is the strongest enforcement available."""
import importlib.util
import pathlib
import re
import sqlite3
import sys

import pytest

from agent_eval_harness.datasets.types import TRUSTED_PROVENANCE
from agent_eval_harness.store.database import close_db, init_db

# Own DB lifecycle: sibling test modules run their own init_db()/close_db() cycle, else the
# shared DB is left closed for whichever module runs next alphabetically.


@pytest.fixture(autouse=True)
async def _setup_db(tmp_path, monkeypatch):
    monkeypatch.setenv("AEH_DATA_DIR", str(tmp_path))
    await init_db()
    yield
    await close_db()

_TEMPLATES_DIR = (
    pathlib.Path(__file__).parent.parent
    / "agent_eval_harness"
    / "code_injection"
    / "templates"
)
_TEMPLATE_PATH = _TEMPLATES_DIR / "run_eval.py"


def test_run_eval_sql_provenance_list_matches_trusted_provenance():
    source = _TEMPLATE_PATH.read_text(encoding="utf-8")
    match = re.search(r"provenance\s+IN\s+\(([^)]+)\)", source)
    assert match, (
        f"Could not find 'provenance IN (...)' clause in {_TEMPLATE_PATH}. "
        "Update the regex if the SQL was restructured."
    )
    raw = match.group(1)
    # Extract quoted string literals: 'value' or "value"
    sql_values = set(re.findall(r"['\"]([^'\"]+)['\"]", raw))
    expected = set(TRUSTED_PROVENANCE)
    assert sql_values == expected, (
        f"Whitelist drift detected!\n"
        f"  run_eval.py SQL IN-list : {sorted(sql_values)}\n"
        f"  TRUSTED_PROVENANCE      : {sorted(expected)}\n"
        "Update the master template (run_eval.py), never a deployed copy."
    )


def _load_run_eval_module():
    """Imports the master run_eval.py template as a real module (its sibling tracer.py stub satisfies its only non-stdlib import) so its actual SQL, not a regex proxy, runs."""
    if str(_TEMPLATES_DIR) not in sys.path:
        sys.path.insert(0, str(_TEMPLATES_DIR))
    spec = importlib.util.spec_from_file_location("run_eval_template_under_test", _TEMPLATE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ac3_run_eval_load_cases_includes_derived_reviewed(tmp_path):
    """A 'derived+reviewed' case reaches run_eval._load_cases (not silently filtered to 0 by the SQL whitelist), while a still-unreviewed 'synthetic' case is excluded."""
    # Own subdir, distinct from the autouse _setup_db fixture's DB — a raw sqlite file built by hand.
    raw_db_dir = tmp_path / "raw_run_eval_fixture"
    raw_db_dir.mkdir()
    db_path = raw_db_dir / "aeh.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE dataset_cases (id TEXT PRIMARY KEY, dataset_id TEXT, input_json TEXT, "
        "expected_json TEXT, labels_json TEXT, provenance TEXT)"
    )
    conn.executemany(
        "INSERT INTO dataset_cases (id, dataset_id, input_json, expected_json, labels_json, "
        "provenance) VALUES (?, ?, ?, ?, ?, ?)",
        [
            # labels carry agent_id — that selects the dispatch module, so a case without it is skipped before provenance is even considered.
            ("c1", "ds1", '{"kind":"x","a":1}', "{}", '{"agent_id":"a1"}', "generated+reviewed"),
            ("c2", "ds1", '{"kind":"x","a":2}', "{}", '{"agent_id":"a1"}', "derived+reviewed"),
            ("c3", "ds1", '{"kind":"x","a":3}', "{}", '{"agent_id":"a1"}', "synthetic"),
        ],
    )
    conn.commit()
    conn.close()

    module = _load_run_eval_module()
    cases = module._load_cases({"aeh_db_path": str(db_path), "dataset_ids": ["ds1"]})

    assert {c["id"] for c in cases} == {"c1", "c2"}


async def test_ac3_derived_reviewed_case_survives_export_dataset():
    """export_dataset keeps a 'derived+reviewed' case, same trust tier as 'generated+reviewed'/'handwritten' — never silently dropped."""
    from agent_eval_harness.datasets.fulfillment import export_dataset
    from agent_eval_harness.datasets.types import DatasetCase
    from agent_eval_harness.store import repository

    dataset_id = "ac3_derived_reviewed_export_v1"
    case = DatasetCase(
        id=repository.new_id(),
        dataset=dataset_id,
        kind="metamorphic_relation",
        input={"query": "q"},
        expected={"invariant_op": "field_equals", "invariant_params": {"target_field": "sufficient", "expect": False}},
        labels={"relation_id": "r1", "source_case_id": "src1"},
        provenance="derived+reviewed",
    )
    await repository.insert_dataset_cases_bulk(dataset_id, [case])
    await repository.insert_dataset_metadata(dataset_id, "metamorphic_relation", min_cases=1)

    exported = await export_dataset(dataset_id)
    assert len(exported) == 1
    assert exported[0].provenance == "derived+reviewed"
