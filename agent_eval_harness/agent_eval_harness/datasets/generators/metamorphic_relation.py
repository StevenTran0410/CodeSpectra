"""Metamorphic-relation dataset generator: mechanically derives gold from an already-reviewed
source dataset via a git-reviewed relation (transform + invariant) — no LLM call, no per-case
review. Callers (metamorphic_derive.py) enforce the double precondition (source reviewed AND
relation runtime-approved) BEFORE invoking this; this generator only re-checks that each
individual source case is itself trusted provenance."""
from __future__ import annotations

import json
import logging

from pydantic import BaseModel

from agent_eval_harness.datasets.metamorphic_ops import apply_transform
from agent_eval_harness.datasets.metamorphic_relations import RelationSpec, get_relation
from agent_eval_harness.datasets.types import TRUSTED_PROVENANCE, DatasetCase
from agent_eval_harness.llm.client import LLMClient
from agent_eval_harness.store import repository
from agent_eval_harness.store.repository import new_id

logger = logging.getLogger("agent_eval_harness.datasets.generators.metamorphic_relation")


class MetamorphicRelationConfig(BaseModel):
    dataset_name: str
    relation_id: str
    source_dataset_id: str


def load_case_input(db_case: dict) -> dict:
    """Parses a source case's input, stripping the kind-sniffing artifact from an older storage shape."""
    input_data = json.loads(db_case["input_json"])
    input_data.pop("kind", None)
    return input_data


def apply_relation_transform(input_data: dict, relation: RelationSpec) -> dict:
    """Applies relation.transform to input_data, or returns it unchanged when there is none."""
    if relation.transform is None:
        return input_data
    return apply_transform(relation.transform.op, input_data, relation.transform.params)


async def generate(
    config: dict, llm_client: LLMClient | None = None, **_kwargs
) -> list[DatasetCase]:
    parsed = MetamorphicRelationConfig.model_validate(config)
    relation = get_relation(parsed.relation_id)
    if relation is None:
        raise ValueError(f"Unknown metamorphic relation id: {parsed.relation_id!r}")

    source_db_cases = await repository.get_dataset_cases(parsed.source_dataset_id)
    cases: list[DatasetCase] = []
    for db_case in source_db_cases:
        if db_case["provenance"] not in TRUSTED_PROVENANCE:
            continue  # never derive from an unreviewed source case

        input_data = apply_relation_transform(load_case_input(db_case), relation)

        expected: dict = {
            "invariant_op": relation.invariant.op,
            "invariant_params": relation.invariant.params,
            "source_case_id": db_case["id"],
        }
        if relation.transform is None:
            source_expected_raw = db_case.get("expected_json")
            if source_expected_raw:
                try:
                    expected["source_expected_output"] = json.loads(source_expected_raw)
                except json.JSONDecodeError as e:
                    logger.debug(f"generate: source case {db_case['id']} has unparseable expected_json: {e}")

        cases.append(
            DatasetCase(
                id=new_id(),
                dataset=parsed.dataset_name,
                kind="metamorphic_relation",
                input=input_data,
                expected=expected,
                labels={"relation_id": relation.id, "source_case_id": db_case["id"]},
                provenance="derived+reviewed",
            )
        )
    return cases
