"""Metamorphic relations (derive/ops/YAML), sweep scoring (live/ingested/metamorphic), CLI, and Tier-2 instrumentation."""
from __future__ import annotations

import asyncio
import json
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_eval_harness.datasets.metamorphic_derive import (
    MetamorphicPreconditionError,
    approve_relation,
    derive_dataset,
    preview_relation,
)
from agent_eval_harness.datasets.metamorphic_ops import (
    apply_transform,
    argmin_k,
    contains_injected_token,
    evaluate_invariant,
    field_equals,
    non_empty,
    subset_eq,
)
from agent_eval_harness.datasets.metamorphic_relations import get_relation, load_relations
from agent_eval_harness.datasets.types import DatasetCase
from agent_eval_harness.ingest.spanlog_ingest import parse_spanlog, persist_spanlog
from agent_eval_harness.instrumentation.tier2_boundary import BoundaryWrapperAdapter, _resolve_entry_point
from agent_eval_harness.llm.client import LLMResponse
from agent_eval_harness.llm.fake_client import FakeLLMClient
from agent_eval_harness.mapping.builder.pipeline import SystemMapBuilder
from agent_eval_harness.mapping.system_map import Component, SystemMap, load_system_map, save_system_map
from agent_eval_harness.metrics.sweep import run_sweep
from agent_eval_harness.store import repository
from agent_eval_harness.store.database import close_db, init_db
from test_targets.linear_rag.pipeline import set_default_llm_client

# Shared by the metamorphic-derive / sweep-ingested / sweep-metamorphic-relation sections below:
# each test needs its own isolated DB (not the conftest session-shared one), so it must be
# requested explicitly rather than autoused module-wide (ops/YAML/CLI/tier2 tests never touch it).
@pytest.fixture
async def _setup_db(tmp_path, monkeypatch):
    monkeypatch.setenv("AEH_DATA_DIR", str(tmp_path))
    await init_db()
    yield
    await close_db()


# ===== metamorphic_derive: post-review derive route =====
# Covers: UNREVIEWED source cohort must 400 (MetamorphicPreconditionError) and mint zero rows;
# end-to-end amortization proof on a non-CodeSpectra target (multi_agent-shaped JudgeComponent
# I/O) where synthetic_count lands at 0 with spot_audit_pct=0; and the applies_to archetype gate.

async def _seed_reviewed_multi_agent_source(dataset_id: str, n: int = 4) -> None:
    """multi_agent JudgeComponent-shaped cases: input {query, context}, expected {sufficient, context}."""
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


# Invariant 1: negative derive test — an unreviewed source cohort must be rejected.

async def test_derive_rejects_unreviewed_source_and_mints_zero_rows(_setup_db):
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
    assert minted == []


async def test_preview_also_rejects_unreviewed_source(_setup_db):
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


async def test_derive_rejects_when_relation_not_yet_approved(_setup_db):
    """Source IS fully reviewed but the relation was never preview-approved: still a 400, zero rows minted."""
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


# Invariant 4: end-to-end amortization proof on multi_agent — synthetic_count lands at 0.

async def test_ac2_derive_on_multi_agent_shaped_source_yields_zero_synthetic_count(_setup_db):
    source_id = "mderive_ac2_source_v1"
    await _seed_reviewed_multi_agent_source(source_id, n=4)

    preview = await preview_relation("empty_context_is_insufficient", source_id, sample_size=10)
    assert preview["sample_count"] == 4
    # A list-typed field degrades to [], not {}.
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
    assert row["synthetic_count"] == 0  # amortization proof via repository's own formula
    assert row["reviewed_count"] == 0  # trusted via derived+reviewed, not generated+reviewed
    assert row["review_complete"] is True


async def test_derive_is_idempotent_on_rerun(_setup_db):
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


async def test_derive_spot_audit_pct_routes_sample_to_review_queue(_setup_db):
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


# applies_to is a HARD gate, not decoration.

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


async def test_preview_refuses_when_cohort_archetype_mismatches_applies_to(_setup_db):
    """A rag_single_shot cohort (fields present, so the field check passes) must still be refused: archetype gate."""
    source_id = "mderive_archetype_mismatch_v1"
    await _seed_reviewed_source_with_archetype(source_id, "rag_single_shot")

    try:
        await preview_relation("empty_context_is_insufficient", source_id)
        assert False, "expected MetamorphicPreconditionError on archetype mismatch"
    except MetamorphicPreconditionError as e:
        assert "different archetype" in str(e)


async def test_derive_succeeds_when_cohort_archetype_matches_applies_to(_setup_db):
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


async def test_derive_unknown_field_name_is_rejected_loudly(_setup_db):
    """A relation referencing a field absent from the source's real shape must fail loud (needs_human)."""
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


# ===== metamorphic_ops: transform/invariant primitives =====

# argmin_k — the mandatory behavioral invariant test: it must actually FIRE.

def test_argmin_k_flags_omitted_lower_score():
    """Auditor case: scores D=49,E=58,J=61,G=63; weakest_sections=[D,E,G] wrongly omits J(61) < G(63)."""
    output = {
        "section_scores": {"D": 49, "E": 58, "J": 61, "G": 63},
        "weakest_sections": ["D", "E", "G"],
    }
    holds = argmin_k(
        output, scores_field="section_scores", report_field="weakest_sections", k=3, allow_ties=True
    )
    assert holds is False


def test_argmin_k_allows_valid_tie_at_boundary():
    """95/95 tie for the 3rd (last) weakest slot — a legitimate tie-break must NOT be flagged."""
    output = {
        "section_scores": {"A": 80, "B": 90, "C": 95, "D": 95, "E": 99},
        "weakest_sections": ["A", "B", "C"],  # picks C over D at the tied boundary score
    }
    holds = argmin_k(
        output, scores_field="section_scores", report_field="weakest_sections", k=3, allow_ties=True
    )
    assert holds is True

    # The other valid tie-break choice (D instead of C) must also not be flagged.
    output2 = {**output, "weakest_sections": ["A", "B", "D"]}
    assert argmin_k(
        output2, scores_field="section_scores", report_field="weakest_sections", k=3, allow_ties=True
    ) is True


def test_argmin_k_rejects_boundary_omission_even_with_ties_allowed():
    """A tie is not a license to omit a strictly-lower key outright."""
    output = {
        "section_scores": {"A": 80, "B": 90, "C": 95, "D": 95, "E": 99},
        "weakest_sections": ["C", "D", "E"],  # drops A/B, includes the non-tied E — invalid
    }
    holds = argmin_k(
        output, scores_field="section_scores", report_field="weakest_sections", k=3, allow_ties=True
    )
    assert holds is False


def test_argmin_k_no_ties_requires_exact_match():
    output = {"section_scores": {"A": 1, "B": 2, "C": 3}, "weakest_sections": ["A", "B"]}
    assert argmin_k(
        output, scores_field="section_scores", report_field="weakest_sections", k=2, allow_ties=False
    ) is True
    output_wrong = {**output, "weakest_sections": ["A", "C"]}
    assert argmin_k(
        output_wrong, scores_field="section_scores", report_field="weakest_sections", k=2, allow_ties=False
    ) is False


def test_argmin_k_missing_fields_returns_false():
    assert argmin_k({}, scores_field="scores", report_field="weakest", k=2) is False
    assert argmin_k(
        {"scores": {"a": 1}}, scores_field="scores", report_field="weakest", k=2
    ) is False  # fewer keys than k


# subset_eq

def test_subset_eq_true_and_false():
    assert subset_eq({"a": [1, 2], "b": [1, 2, 3]}, sub_field="a", super_field="b") is True
    assert subset_eq({"a": [1, 4], "b": [1, 2, 3]}, sub_field="a", super_field="b") is False
    assert subset_eq({"a": "not-a-list", "b": [1]}, sub_field="a", super_field="b") is False


# contains_injected_token

def test_contains_injected_token_in_list_field():
    output = {"violations_found": ["contradiction: __mr_conflict__ present", "other"]}
    assert contains_injected_token(
        output, output_field="violations_found", token="__mr_conflict__", min_count=1
    ) is True


def test_contains_injected_token_absent():
    output = {"violations_found": []}
    assert contains_injected_token(
        output, output_field="violations_found", token="__mr_conflict__", min_count=1
    ) is False


def test_contains_injected_token_string_field_min_count():
    output = {"summary": "__mr_conflict__ appears twice: __mr_conflict__"}
    assert contains_injected_token(output, output_field="summary", token="__mr_conflict__", min_count=2) is True
    assert contains_injected_token(output, output_field="summary", token="__mr_conflict__", min_count=3) is False


# non_empty

def test_non_empty_variants():
    assert non_empty({"context": ["x"]}, target_field="context") is True
    assert non_empty({"context": []}, target_field="context") is False
    assert non_empty({"context": None}, target_field="context") is False
    assert non_empty({}, target_field="missing") is False
    assert non_empty({"count": 0}, target_field="count") is True  # scalar zero is not "empty"


# field_equals

def test_field_equals():
    assert field_equals({"sufficient": False}, target_field="sufficient", expect=False) is True
    assert field_equals({"sufficient": True}, target_field="sufficient", expect=False) is False
    assert field_equals({}, target_field="sufficient", expect=False) is False


# apply_transform / evaluate_invariant dispatch (non-CodeSpectra fixture, generic field names)

def test_apply_transform_degrade_section_on_flat_list_field():
    """multi_agent-shaped input: a list-typed top-level field degrades to []."""
    input_data = {"query": "What is the vacation policy?", "context": ["doc a", "doc b"]}
    result = apply_transform("degrade_section", input_data, {"target_field": "context"})
    assert result["context"] == []
    assert input_data["context"] == ["doc a", "doc b"]  # caller dict untouched


def test_apply_transform_inject_conflict():
    input_data = {"purpose": "original", "description": "also original"}
    result = apply_transform(
        "inject_conflict", input_data, {"target_fields": ["purpose", "description"], "token": "__mr_conflict__"}
    )
    assert "__mr_conflict__" in result["purpose"]
    assert "__mr_conflict__" in result["description"]


def test_apply_transform_unknown_op_raises():
    with pytest.raises(ValueError):
        apply_transform("not_a_real_op", {}, {})


def test_evaluate_invariant_dispatch():
    assert evaluate_invariant("field_equals", {"sufficient": False}, {"target_field": "sufficient", "expect": False}) is True


def test_evaluate_invariant_unknown_op_raises():
    with pytest.raises(ValueError):
        evaluate_invariant("not_a_real_op", {}, {})


# ===== metamorphic_relations_yaml: relation catalog loading =====

def test_load_relations_parses_all_three():
    relations = load_relations()
    assert set(relations.keys()) == {
        "weakest_is_argmin",
        "injected_conflict_is_flagged",
        "empty_context_is_insufficient",
    }


def test_weakest_is_argmin_has_no_transform():
    relation = get_relation("weakest_is_argmin")
    assert relation is not None
    assert relation.transform is None
    assert relation.invariant.op == "argmin_k"
    assert relation.invariant.params["k"] == 3
    assert relation.invariant.params["allow_ties"] is True


def test_empty_context_is_insufficient_shape():
    """The non-CodeSpectra acceptance-target relation: transform field name is generic ('context'), never a section letter."""
    relation = get_relation("empty_context_is_insufficient")
    assert relation is not None
    assert relation.transform.op == "degrade_section"
    assert relation.transform.params["target_field"] == "context"
    assert relation.invariant.op == "field_equals"
    assert relation.invariant.params["target_field"] == "sufficient"
    assert relation.invariant.params["expect"] is False


def test_get_relation_unknown_id_returns_none():
    assert get_relation("does_not_exist") is None


# ===== sweep_ingested_source: RunSource='ingested' seam =====
# Scores an already-ingested Stage 4 run's persisted spans directly, instead of run_sweep's
# normal live in-process execute_run path.

def _write_map_ingested(tmp_path: Path) -> str:
    system_map = SystemMap(
        target_system_id="codespectra",
        components=[Component(id="project_identity", role="writer", entry_point="a:b")],
    )
    map_path = tmp_path / "map.yaml"
    save_system_map(system_map, map_path)
    return str(map_path)


def _write_suite_ingested(tmp_path: Path, schema: dict) -> str:
    suite_path = tmp_path / "plan.yaml"
    suite_path.write_text(
        "entries:\n" + textwrap.dedent(f"""\
              - id: project_identity.schema_valid
                component: project_identity
                agent_id: project_identity
                metric: schema_valid
                metric_class: assertion
                params:
                  schema: {json.dumps(schema)}
                rationale: r
                provenance: rule
              - id: project_identity.some_judge
                component: project_identity
                agent_id: project_identity
                metric: geval.quality
                metric_class: llm_judge
                rationale: r
                provenance: rule
        """),
        encoding="utf-8",
    )
    return str(suite_path)


async def _persist_run_with_output(tmp_path: Path, output: dict) -> str:
    log_path = tmp_path / "eval_log.1.jsonl"
    records = [
        {"record": "header", "schema": "aeh.spanlog/1", "tracer_version": "1", "plan_id": "p1", "run_id": "r1"},
        {"record": "case_start", "trace_id": "t1", "dataset_case_id": "case-1", "input": {}},
        {
            "record": "span", "trace_id": "t1", "span_id": "sp1", "parent_span_id": None,
            "component_id": "project_identity", "span_type": "agent", "operation": "haystack.component.run",
            "started_at": "2026-01-01T00:00:00.000Z", "latency_ms": 100,
            "input_json": "{}", "output_json": json.dumps(output),
        },
        {"record": "case_end", "trace_id": "t1", "status": "ok", "final_output_json": json.dumps(output)},
        {"record": "run_summary", "attempted": 1, "succeeded": 1},
    ]
    log_path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    parsed = parse_spanlog(log_path)
    return await persist_spanlog(parsed, target_system_id="codespectra", eval_plan_id="p1")


async def test_ingested_source_scores_assertion_against_persisted_spans(tmp_path: Path, _setup_db) -> None:
    schema = {"type": "object", "properties": {"domain": {"type": "string"}}, "required": ["domain"]}
    map_path = _write_map_ingested(tmp_path)
    suite_path = _write_suite_ingested(tmp_path, schema)
    run_id = await _persist_run_with_output(tmp_path, {"domain": "web app"})

    result = await run_sweep(
        target="codespectra", map_path=map_path, suite_path=suite_path,
        llm_client=None, run_id=run_id, source="ingested",
    )

    schema_results = [r for r in result.results if r.metric_name == "assertion.schema_valid"]
    assert len(schema_results) == 1
    assert schema_results[0].passed is True
    assert schema_results[0].trace_id is not None


async def test_ingested_source_flags_schema_violation(tmp_path: Path, _setup_db) -> None:
    schema = {"type": "object", "properties": {"domain": {"type": "string"}}, "required": ["domain"]}
    map_path = _write_map_ingested(tmp_path)
    suite_path = _write_suite_ingested(tmp_path, schema)
    run_id = await _persist_run_with_output(tmp_path, {"wrong_field": "oops"})

    result = await run_sweep(
        target="codespectra", map_path=map_path, suite_path=suite_path,
        llm_client=None, run_id=run_id, source="ingested",
    )

    schema_results = [r for r in result.results if r.metric_name == "assertion.schema_valid"]
    assert schema_results[0].passed is False


async def test_ingested_source_judge_gate_returns_not_supported_placeholder(tmp_path: Path, _setup_db) -> None:
    schema = {"type": "object"}
    map_path = _write_map_ingested(tmp_path)
    suite_path = _write_suite_ingested(tmp_path, schema)
    run_id = await _persist_run_with_output(tmp_path, {"domain": "web"})

    result = await run_sweep(
        target="codespectra", map_path=map_path, suite_path=suite_path,
        llm_client=None, run_id=run_id, source="ingested",
    )

    judge_results = [r for r in result.results if r.metric_class == "llm_judge"]
    assert len(judge_results) == 1
    assert judge_results[0].passed is None
    assert "not supported" in judge_results[0].details["reason"]


async def test_ingested_source_does_not_overwrite_run_status_set_at_ingest(tmp_path: Path, _setup_db) -> None:
    """The run's status (set at ingest) must never be overwritten by a scoring pass, even on success."""
    schema = {"type": "object"}
    map_path = _write_map_ingested(tmp_path)
    suite_path = _write_suite_ingested(tmp_path, schema)
    run_id = await _persist_run_with_output(tmp_path, {"domain": "web"})
    run_before = await repository.get_run(run_id)

    await run_sweep(
        target="codespectra", map_path=map_path, suite_path=suite_path,
        llm_client=None, run_id=run_id, source="ingested",
    )

    run_after = await repository.get_run(run_id)
    assert run_after["status"] == run_before["status"] == "completed"


async def test_ingested_source_requires_an_existing_run_id(tmp_path: Path, _setup_db) -> None:
    map_path = _write_map_ingested(tmp_path)
    suite_path = _write_suite_ingested(tmp_path, {"type": "object"})

    with pytest.raises(ValueError, match="source='ingested' requires an existing run_id"):
        await run_sweep(
            target="codespectra", map_path=map_path, suite_path=suite_path,
            llm_client=None, source="ingested",
        )


# ===== sweep_metamorphic_relation: derived case scored end-to-end =====
# Derive, sweep, a real pass/fail in the report — not just "the case exists in the DB" — using
# the self-consistency relation (transform=null) for a determinate verdict off the reviewed
# source gold, no agent run.

def _write_map_mr(tmp_path: Path) -> str:
    system_map = SystemMap(
        target_system_id="acme",
        components=[Component(id="judge", role="validator", entry_point="a:Judge")],
    )
    map_path = tmp_path / "map.yaml"
    save_system_map(system_map, map_path)
    return str(map_path)


def _write_suite_mr(tmp_path: Path, dataset_ref: str) -> str:
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
    """A reviewed fan_in_judge cohort: one valid argmin-3, one wrongly omits a lower-scoring key."""
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


async def test_metamorphic_relation_case_scored_end_to_end_yields_pass_and_fail(tmp_path: Path, _setup_db):
    source_id = "mr_ac1_source_v1"
    await _seed_reviewed_source(source_id)

    preview = await preview_relation("weakest_is_argmin", source_id)
    await approve_relation("weakest_is_argmin", source_id, samples=preview["samples"])
    derived = await derive_dataset(
        "weakest_is_argmin", source_id, dataset_name="mr_ac1_derived_v1", spot_audit_pct=0.0
    )
    assert derived["minted_count"] == 2

    map_path = _write_map_mr(tmp_path)
    suite_path = _write_suite_mr(tmp_path, derived["dataset_id"])

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


# ===== sweep_runner_cli: `aeh eval` / `aeh report` CLI subcommands (via cli.main()) =====

_CLI_MAP_PATH = str(Path(__file__).parent.parent / "test_targets" / "multi_agent" / "system_map.yaml")
_CLI_TARGET = "test_targets.multi_agent.pipeline:build_pipeline"


class TestSweepRunnerCli:
    # autouse (not module-wide) so its teardown reliably runs after monkeypatch reverts AEH_DATA_DIR,
    # restoring the shared DB cli.main()'s own init_db/close_db cycle closed, for whatever runs next.
    @pytest.fixture(autouse=True)
    def _restore_shared_db_after_cli_closes_it(self):
        """cli.main() runs init_db/close_db — reopen after so later tests still work."""
        yield
        asyncio.run(init_db())

    @pytest.fixture()
    def minimal_suite_yaml(self, tmp_path) -> str:
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
    def aeh_db(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AEH_DATA_DIR", str(tmp_path))
        return tmp_path

    def test_eval_command_runs_and_prints_results(
        self, aeh_db, minimal_suite_yaml, capsys, monkeypatch
    ) -> None:
        """aeh eval should run without crashing and print a result summary."""
        from agent_eval_harness.cli import main

        ret = main(
            [
                "eval",
                "--target", _CLI_TARGET,
                "--map", _CLI_MAP_PATH,
                "--suite", minimal_suite_yaml,
                "--data-dir", str(aeh_db),
            ]
        )
        out = capsys.readouterr().out
        assert ret == 0
        assert "[aeh] eval" in out
        assert "completed" in out

    def test_eval_command_json_output(self, aeh_db, minimal_suite_yaml, capsys) -> None:
        """aeh eval --json should produce valid JSON."""
        from agent_eval_harness.cli import main

        main(
            [
                "eval",
                "--target", _CLI_TARGET,
                "--map", _CLI_MAP_PATH,
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

    def test_report_command_no_data(self, aeh_db, capsys) -> None:
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

    def test_suite_schema_loaded_by_eval(self, aeh_db, tmp_path, capsys) -> None:
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
                    "--target", _CLI_TARGET,
                    "--map", _CLI_MAP_PATH,
                    "--suite", str(bad_suite),
                    "--data-dir", str(aeh_db),
                ]
            )


# ===== tier2_fallback: Tier-2 boundary-wrapper fallback demo (T1 only) =====

_TIER2_MAP_PATH = Path(__file__).parent.parent / "test_targets" / "linear_rag" / "system_map.yaml"


async def test_tier2_fallback_coarser_no_nested_spans() -> None:
    set_default_llm_client(
        FakeLLMClient(LLMResponse(content="Vacation is 15 days per year.", model="fake-mini"))
    )
    system_map = load_system_map(_TIER2_MAP_PATH)
    adapter = BoundaryWrapperAdapter(system_map, ["retriever", "writer"])

    adapter.attach()
    try:
        result = await adapter.run("What is the vacation policy?")
    finally:
        adapter.detach()

    assert len(result.spans) == 2  # coarse: exactly one span per component
    assert {s.component_id for s in result.spans} == {"retriever", "writer"}
    for span in result.spans:
        assert span.tier == "tier2"
        assert span.parent_span_id is None  # no nesting at this granularity
    assert "vacation" in result.final_output.lower()


# ===== tier2_http_entry_and_docs_reconcile: Tier-2 HTTP entry points and doc-reconciliation =====

@pytest.mark.asyncio
async def test_resolve_http_entry_point() -> None:
    """Verify that resolving an HTTP entry point returns an HTTP post wrapper client."""
    fn = _resolve_entry_point("http://localhost:9999/api/agent")

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"answer": "HTTP Success", "model": "mock-llm", "tokens_in": 10, "tokens_out": 20}
    mock_resp.status_code = 200

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp

        out = await fn("hello query", {"some": "prior"})

        mock_post.assert_called_once_with(
            "http://localhost:9999/api/agent",
            json={"query": "hello query", "prior_output": {"some": "prior"}},
            timeout=60.0,
        )
        assert out == {"answer": "HTTP Success", "model": "mock-llm", "tokens_in": 10, "tokens_out": 20}


@pytest.mark.asyncio
async def test_reconcile_docs_matches_components(tmp_path) -> None:
    """Verify that _reconcile_docs identifies components missing from hand-written docs."""
    builder = SystemMapBuilder(tmp_path)

    doc_file = tmp_path / "arch.md"
    doc_file.write_text("# System Architecture\n\nThis mentions component: retriever.", encoding="utf-8")

    components = [
        Component(id="retriever", role="agent", entry_point="main:ret"),
        Component(id="writer", role="agent", entry_point="main:writer"),
    ]

    discrepancies = await builder._reconcile_docs(doc_file, components)

    # retriever is mentioned in doc, but writer is not mentioned
    assert len(discrepancies) == 1
    assert "writer" in discrepancies[0]
    assert "retriever" not in discrepancies[0]
