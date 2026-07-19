"""Behavioral tests for the post-review derive route (agent_eval_harness/datasets/metamorphic_derive.py).

Covers the 3 remaining mandatory invariants (argmin_k is covered in test_metamorphic_ops.py):
1. Negative derive test — an UNREVIEWED source cohort must 400 (MetamorphicPreconditionError)
   and mint zero derived+reviewed rows (AC4 / Q1-b guard).
4. End-to-end amortization proof on a non-CodeSpectra target (multi_agent-shaped JudgeComponent
   I/O): derived cases land as derived+reviewed, and synthetic_count for that dataset is
   exactly 0 with spot_audit_pct=0 (AC2).
"""
import pytest

from agent_eval_harness.datasets.metamorphic_derive import (
    MetamorphicPreconditionError,
    approve_relation,
    derive_dataset,
    preview_relation,
)
from agent_eval_harness.datasets.types import DatasetCase
from agent_eval_harness.store import repository
from agent_eval_harness.store.database import close_db, init_db

# Own DB lifecycle (module convention, e.g. test_eval_run_route.py) — several sibling test
# modules run their own init_db()/close_db() cycle, which otherwise leaves the shared
# session-scoped DB closed for whatever alphabetically runs next.
pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
async def _setup_db(tmp_path, monkeypatch):
    monkeypatch.setenv("AEH_DATA_DIR", str(tmp_path))
    await init_db()
    yield
    await close_db()


async def _seed_reviewed_multi_agent_source(dataset_id: str, n: int = 4) -> None:
    """multi_agent JudgeComponent-shaped cases: input {query, context}, expected {sufficient, context} —
    flat, section-letter-free, exactly the acceptance target's real I/O shape."""
    cases = [
        DatasetCase(
            id=repository.new_id(),
            dataset=dataset_id,
            kind="synthetic_agent_io",
            input={"query": f"What is policy {i}?", "context": [f"doc {i} para 1", f"doc {i} para 2"]},
            expected={"sufficient": True, "context": [f"doc {i} para 1", f"doc {i} para 2"]},
            labels=None,
            provenance="generated+reviewed",
        )
        for i in range(n)
    ]
    await repository.insert_dataset_cases_bulk(dataset_id, cases)
    await repository.insert_dataset_metadata(dataset_id, "synthetic_agent_io", min_cases=1)


async def _get_dataset_row(dataset_id: str) -> dict | None:
    rows = await repository.list_dataset_ids()
    return next((r for r in rows if r["dataset_id"] == dataset_id), None)


# ---------------------------------------------------------------------------
# Invariant 1 — negative derive test (AC4 / Q1-b guard)
# ---------------------------------------------------------------------------

async def test_derive_rejects_unreviewed_source_and_mints_zero_rows():
    source_id = "mderive_unreviewed_source_v1"
    unreviewed_case = DatasetCase(
        id=repository.new_id(),
        dataset=source_id,
        kind="synthetic_agent_io",
        input={"query": "q", "context": ["c"]},
        expected={"sufficient": True, "context": ["c"]},
        labels=None,
        provenance="synthetic",  # NOT yet reviewed
    )
    await repository.insert_dataset_cases_bulk(source_id, [unreviewed_case])
    await repository.insert_dataset_metadata(source_id, "synthetic_agent_io", min_cases=1)

    row = await _get_dataset_row(source_id)
    assert row is not None and row["review_complete"] is False

    target_dataset_id = "mderive_unreviewed_target_v1"

    try:
        await derive_dataset(
            "empty_context_is_insufficient", source_id,
            dataset_name=target_dataset_id, spot_audit_pct=0.0,
        )
        assert False, "expected MetamorphicPreconditionError for an unreviewed source cohort"
    except MetamorphicPreconditionError as e:
        assert "not fully reviewed" in str(e)

    minted = await repository.get_dataset_cases(target_dataset_id)
    assert minted == []  # zero derived+reviewed rows minted


async def test_preview_also_rejects_unreviewed_source():
    source_id = "mderive_unreviewed_preview_source_v1"
    unreviewed_case = DatasetCase(
        id=repository.new_id(), dataset=source_id, kind="synthetic_agent_io",
        input={"query": "q", "context": []}, expected={"sufficient": False, "context": []},
        labels=None, provenance="synthetic",
    )
    await repository.insert_dataset_cases_bulk(source_id, [unreviewed_case])
    await repository.insert_dataset_metadata(source_id, "synthetic_agent_io", min_cases=1)

    try:
        await preview_relation("empty_context_is_insufficient", source_id)
        assert False, "expected MetamorphicPreconditionError"
    except MetamorphicPreconditionError:
        pass


async def test_derive_rejects_when_relation_not_yet_approved():
    """Source IS fully reviewed but the relation has never been preview-approved for it —
    still a 400, still zero rows minted (the second half of the double precondition)."""
    source_id = "mderive_unapproved_relation_source_v1"
    await _seed_reviewed_multi_agent_source(source_id, n=2)

    target_dataset_id = "mderive_unapproved_relation_target_v1"
    try:
        await derive_dataset(
            "empty_context_is_insufficient", source_id,
            dataset_name=target_dataset_id, spot_audit_pct=0.0,
        )
        assert False, "expected MetamorphicPreconditionError: relation not approved"
    except MetamorphicPreconditionError as e:
        assert "not been preview-approved" in str(e)

    minted = await repository.get_dataset_cases(target_dataset_id)
    assert minted == []


# ---------------------------------------------------------------------------
# Invariant 4 — end-to-end amortization proof on multi_agent (AC2)
# ---------------------------------------------------------------------------

async def test_ac2_derive_on_multi_agent_shaped_source_yields_zero_synthetic_count():
    source_id = "mderive_ac2_source_v1"
    await _seed_reviewed_multi_agent_source(source_id, n=4)

    preview = await preview_relation("empty_context_is_insufficient", source_id, sample_size=10)
    assert preview["sample_count"] == 4
    # The :23-24 fix in action: a list-typed field degrades to [], not {}.
    for sample in preview["samples"]:
        assert sample["derived_input"]["context"] == []

    await approve_relation("empty_context_is_insufficient", source_id, samples=preview["samples"])

    target_dataset_id = "mderive_ac2_target_v1"
    result = await derive_dataset(
        "empty_context_is_insufficient", source_id,
        dataset_name=target_dataset_id, spot_audit_pct=0.0,
    )
    assert result["minted_count"] == 4
    assert result["spot_audit_count"] == 0

    minted_cases = await repository.get_dataset_cases(target_dataset_id)
    assert len(minted_cases) == 4
    assert all(c["provenance"] == "derived+reviewed" for c in minted_cases)

    row = await _get_dataset_row(target_dataset_id)
    assert row is not None
    assert row["synthetic_count"] == 0  # AC2: amortization proof via repository.py:247's own formula
    assert row["reviewed_count"] == 0  # trusted via derived+reviewed, not generated+reviewed
    assert row["review_complete"] is True


async def test_derive_is_idempotent_on_rerun():
    source_id = "mderive_idempotent_source_v1"
    await _seed_reviewed_multi_agent_source(source_id, n=3)

    target_dataset_id = "mderive_idempotent_target_v1"
    preview = await preview_relation("empty_context_is_insufficient", source_id)
    await approve_relation("empty_context_is_insufficient", source_id, samples=preview["samples"])

    first = await derive_dataset(
        "empty_context_is_insufficient", source_id,
        dataset_name=target_dataset_id, spot_audit_pct=0.0,
    )
    assert first["minted_count"] == 3

    second = await derive_dataset(
        "empty_context_is_insufficient", source_id,
        dataset_name=target_dataset_id, spot_audit_pct=0.0,
    )
    assert second["minted_count"] == 0
    assert second["skipped_existing_count"] == 3

    minted_cases = await repository.get_dataset_cases(target_dataset_id)
    assert len(minted_cases) == 3  # no duplicate rows from the re-run


async def test_derive_spot_audit_pct_routes_sample_to_review_queue():
    source_id = "mderive_spotaudit_source_v1"
    await _seed_reviewed_multi_agent_source(source_id, n=10)

    target_dataset_id = "mderive_spotaudit_target_v1"
    preview = await preview_relation("empty_context_is_insufficient", source_id)
    await approve_relation("empty_context_is_insufficient", source_id, samples=preview["samples"])

    result = await derive_dataset(
        "empty_context_is_insufficient", source_id,
        dataset_name=target_dataset_id, spot_audit_pct=0.10,
    )
    assert result["spot_audit_count"] == 1  # floor(1) on 10 * 0.10
    assert result["minted_count"] == 9

    row = await _get_dataset_row(target_dataset_id)
    assert row["synthetic_count"] == 1


# ---------------------------------------------------------------------------
# CS-302 Slice 1.3 — applies_to is a HARD gate, not decoration
# ---------------------------------------------------------------------------

async def _seed_reviewed_source_with_archetype(dataset_id: str, archetype: str, n: int = 2) -> None:
    cases = [
        DatasetCase(
            id=repository.new_id(), dataset=dataset_id, kind="synthetic_agent_io",
            input={"query": f"q{i}", "context": [f"doc {i}"]},
            expected={"sufficient": True, "context": [f"doc {i}"]},
            labels={"archetype": archetype}, provenance="generated+reviewed",
        )
        for i in range(n)
    ]
    await repository.insert_dataset_cases_bulk(dataset_id, cases)
    await repository.insert_dataset_metadata(dataset_id, "synthetic_agent_io", min_cases=1)


async def test_preview_refuses_when_cohort_archetype_mismatches_applies_to():
    """empty_context_is_insufficient applies_to fan_in_judge; a rag_single_shot cohort (fields
    present, so the field check passes) must still be refused on the archetype gate."""
    source_id = "mderive_archetype_mismatch_v1"
    await _seed_reviewed_source_with_archetype(source_id, "rag_single_shot")

    try:
        await preview_relation("empty_context_is_insufficient", source_id)
        assert False, "expected MetamorphicPreconditionError on archetype mismatch"
    except MetamorphicPreconditionError as e:
        assert "different archetype" in str(e)


async def test_derive_succeeds_when_cohort_archetype_matches_applies_to():
    """The positive path: a fan_in_judge cohort matches the relation's applies_to and derives."""
    source_id = "mderive_archetype_match_v1"
    await _seed_reviewed_source_with_archetype(source_id, "fan_in_judge")

    preview = await preview_relation("empty_context_is_insufficient", source_id)
    await approve_relation("empty_context_is_insufficient", source_id, samples=preview["samples"])
    result = await derive_dataset(
        "empty_context_is_insufficient", source_id,
        dataset_name="mderive_archetype_match_target_v1", spot_audit_pct=0.0,
    )
    assert result["minted_count"] == 2


async def test_derive_unknown_field_name_is_rejected_loudly():
    """A relation referencing a field absent from the source's real shape must fail loud
    (needs_human), never a vacuous pass."""
    source_id = "mderive_unknown_field_source_v1"
    cases = [
        DatasetCase(
            id=repository.new_id(), dataset=source_id, kind="synthetic_agent_io",
            input={"totally_different_field": 1}, expected={"also_different": True},
            labels=None, provenance="generated+reviewed",
        )
    ]
    await repository.insert_dataset_cases_bulk(source_id, cases)
    await repository.insert_dataset_metadata(source_id, "synthetic_agent_io", min_cases=1)

    try:
        await preview_relation("empty_context_is_insufficient", source_id)
        assert False, "expected MetamorphicPreconditionError for unknown field names"
    except MetamorphicPreconditionError as e:
        assert "needs_human" in str(e)
