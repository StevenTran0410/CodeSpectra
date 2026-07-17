"""Post-review metamorphic derivation pass — an explicit route, not a fulfill_plan phase
(fulfill_plan's min_cases floor runs PRE-review over freshly-synthesized cases in the same
pass; this pass runs AFTER a human has finished reviewing the source cohort, so it cannot
share fulfill_plan's loop). Enforces the double precondition at runtime: (1) the source
cohort is fully reviewed (repository.list_dataset_ids()'s review_complete), and (2) the relation
itself has been preview-approved for THIS source cohort. Either unmet -> MetamorphicPreconditionError
with an explicit reason, zero derived+reviewed rows minted."""
from __future__ import annotations

import json
import logging
import math
import random
from typing import Any

from agent_eval_harness.datasets.metamorphic_ops import apply_transform
from agent_eval_harness.datasets.metamorphic_relations import RelationSpec, get_relation
from agent_eval_harness.datasets.registry import get_generator
from agent_eval_harness.store import repository

logger = logging.getLogger("agent_eval_harness.datasets.metamorphic_derive")

_FIELD_PARAM_SUFFIXES = ("_field", "_fields")


class MetamorphicPreconditionError(ValueError):
    """Raised with an explicit, user-facing reason when a derive/preview precondition is
    unmet — callers (ui/server.py) surface this as HTTP 400."""


def _approval_key(relation_id: str, source_dataset_id: str) -> str:
    return f"__metamorphic_approval__/{relation_id}/{source_dataset_id}"


async def _source_review_complete(source_dataset_id: str) -> bool:
    rows = await repository.list_dataset_ids()
    row = next((r for r in rows if r["dataset_id"] == source_dataset_id), None)
    return bool(row and row["review_complete"])


def _relation_field_names(relation: RelationSpec) -> set[str]:
    """Extracts every *_field/*_fields parameter value — the field names this relation
    references — from both its transform and invariant op specs."""
    names: set[str] = set()
    for op_spec in (relation.transform, relation.invariant):
        if op_spec is None:
            continue
        for key, val in op_spec.params.items():
            if not key.endswith(_FIELD_PARAM_SUFFIXES):
                continue
            if isinstance(val, str):
                names.add(val)
            elif isinstance(val, list):
                names.update(v for v in val if isinstance(v, str))
    return names


def _known_field_names(source_cases: list[dict]) -> set[str]:
    """Union of top-level keys across a sample of source cases' input/expected/labels —
    a lightweight, generic stand-in for the target's harvested output.json_schema."""
    names: set[str] = set()
    for db_case in source_cases[:20]:
        for col in ("input_json", "expected_json", "labels_json"):
            raw = db_case.get(col)
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except Exception as e:
                logger.debug(f"_known_field_names: skipping unparseable {col} on case sample: {e}")
                continue
            if isinstance(data, dict):
                names.update(data.keys())
    return names


async def _load_relation_and_source(
    relation_id: str, source_dataset_id: str
) -> tuple[RelationSpec, list[dict]]:
    relation = get_relation(relation_id)
    if relation is None:
        raise MetamorphicPreconditionError(f"Unknown metamorphic relation id: {relation_id!r}")
    if not await _source_review_complete(source_dataset_id):
        raise MetamorphicPreconditionError(
            f"Source dataset {source_dataset_id!r} is not fully reviewed yet (has unreviewed "
            "'synthetic' cases) — cannot preview or derive from an unreviewed cohort."
        )
    source_cases = await repository.get_dataset_cases(source_dataset_id)
    unknown_fields = _relation_field_names(relation) - _known_field_names(source_cases)
    if unknown_fields:
        raise MetamorphicPreconditionError(
            f"Relation {relation_id!r} references field(s) {sorted(unknown_fields)} not found "
            f"in any case of source dataset {source_dataset_id!r} — needs_human: verify field "
            "names before deriving; refusing a vacuous pass."
        )
    return relation, source_cases


async def preview_relation(
    relation_id: str, source_dataset_id: str, sample_size: int = 5
) -> dict[str, Any]:
    """Renders up to sample_size sample derivations for human approval — this preview IS the
    runtime relation-review precondition (the only check that catches a relation that is
    unit-test-clean but conceptually wrong)."""
    relation, source_cases = await _load_relation_and_source(relation_id, source_dataset_id)

    samples = []
    for db_case in source_cases[:sample_size]:
        input_data = json.loads(db_case["input_json"])
        input_data.pop("kind", None)
        transformed = input_data
        if relation.transform is not None:
            transformed = apply_transform(relation.transform.op, input_data, relation.transform.params)
        samples.append(
            {
                "source_case_id": db_case["id"],
                "original_input": input_data,
                "derived_input": transformed,
                "invariant": {"op": relation.invariant.op, "params": relation.invariant.params},
            }
        )
    return {
        "relation_id": relation_id,
        "source_dataset_id": source_dataset_id,
        "applies_to": relation.applies_to,
        "sample_count": len(samples),
        "samples": samples,
    }


async def approve_relation(
    relation_id: str, source_dataset_id: str, samples: list[dict] | None = None
) -> None:
    """Persists a runtime approval marker for (relation, source cohort) — zero migration, reuses
    the existing `datasets` metadata table (fulfillment.py's insert_dataset_metadata idiom)."""
    if get_relation(relation_id) is None:
        raise MetamorphicPreconditionError(f"Unknown metamorphic relation id: {relation_id!r}")
    await repository.insert_dataset_metadata(
        _approval_key(relation_id, source_dataset_id),
        "_metamorphic_approval",
        instructions={"approved": True, "preview_samples": samples or []},
        min_cases=0,
    )


async def _relation_approved(relation_id: str, source_dataset_id: str) -> bool:
    row = await repository.get_dataset_metadata(_approval_key(relation_id, source_dataset_id))
    if not row:
        return False
    stored = json.loads(row["instructions_json"]) if row.get("instructions_json") else {}
    return bool(stored.get("approved"))


async def derive_dataset(
    relation_id: str,
    source_dataset_id: str,
    *,
    dataset_name: str | None = None,
    spot_audit_pct: float = 0.10,
) -> dict[str, Any]:
    """Mints derived+reviewed cases for (relation, source cohort). Double precondition enforced
    here (source reviewed AND relation preview-approved) — unmet either => zero rows minted.
    Idempotent: re-running skips source cases already derived (keyed by their labels.source_case_id).
    spot_audit_pct routes a bounded sample back to the existing review UI as 'synthetic' instead
    of trusting them outright (correlated-error mitigation) — spot_audit_pct=0 mints purely
    mechanical derived+reviewed rows."""
    _relation, _source_cases = await _load_relation_and_source(relation_id, source_dataset_id)
    if not await _relation_approved(relation_id, source_dataset_id):
        raise MetamorphicPreconditionError(
            f"Relation {relation_id!r} has not been preview-approved for source dataset "
            f"{source_dataset_id!r} yet — call preview then approve first."
        )

    dataset_id = dataset_name or f"metamorphic_relation__{relation_id}__{source_dataset_id}"

    existing_cases = await repository.get_dataset_cases(dataset_id)
    existing_source_case_ids: set[str] = set()
    for c in existing_cases:
        labels = json.loads(c["labels_json"]) if c.get("labels_json") else {}
        source_case_id = labels.get("source_case_id")
        if source_case_id:
            existing_source_case_ids.add(source_case_id)

    generator_fn = get_generator("metamorphic_relation")
    all_derived = await generator_fn(
        {"dataset_name": dataset_id, "relation_id": relation_id, "source_dataset_id": source_dataset_id},
        None,
    )
    new_cases = [
        c for c in all_derived
        if (c.labels or {}).get("source_case_id") not in existing_source_case_ids
    ]

    spot_audit_count = 0
    if new_cases and spot_audit_pct > 0:
        n_audit = min(max(1, math.ceil(len(new_cases) * spot_audit_pct)), len(new_cases))
        for i in random.sample(range(len(new_cases)), n_audit):
            new_cases[i].provenance = "synthetic"
        spot_audit_count = n_audit

    if new_cases:
        await repository.insert_dataset_cases_bulk(dataset_id, new_cases)
        if not existing_cases:
            await repository.insert_dataset_metadata(dataset_id, "metamorphic_relation", min_cases=1)

    return {
        "dataset_id": dataset_id,
        "minted_count": len(new_cases) - spot_audit_count,
        "spot_audit_count": spot_audit_count,
        "skipped_existing_count": len(all_derived) - len(new_cases),
    }
