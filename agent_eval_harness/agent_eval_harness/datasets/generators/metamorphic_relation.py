"""Metamorphic-relation dataset generator: mechanically derives gold from an already-reviewed
source dataset via a git-reviewed relation (transform + invariant) — no LLM call, no per-case
human review. The relation's transform+invariant ARE the gold; review the relation once,
amortize across every source case. Callers (metamorphic_derive.py) must enforce the double
precondition (source reviewed AND relation runtime-approved) BEFORE invoking this — this
generator only defends against an individually-unreviewed source case slipping through."""
from __future__ import annotations

import json

from pydantic import BaseModel

from agent_eval_harness.datasets.metamorphic_ops import apply_transform
from agent_eval_harness.datasets.metamorphic_relations import get_relation
from agent_eval_harness.datasets.types import TRUSTED_PROVENANCE, DatasetCase
from agent_eval_harness.llm.client import LLMClient
from agent_eval_harness.store import repository
from agent_eval_harness.store.repository import new_id


class MetamorphicRelationConfig(BaseModel):
    dataset_name: str
    relation_id: str
    source_dataset_id: str


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

        input_data = json.loads(db_case["input_json"])
        input_data.pop("kind", None)  # kind-sniffing artifact from an older storage shape
        if relation.transform is not None:
            input_data = apply_transform(relation.transform.op, input_data, relation.transform.params)

        cases.append(
            DatasetCase(
                id=new_id(),
                dataset=parsed.dataset_name,
                kind="metamorphic_relation",
                input=input_data,
                expected={
                    "invariant_op": relation.invariant.op,
                    "invariant_params": relation.invariant.params,
                },
                labels={"relation_id": relation.id, "source_case_id": db_case["id"]},
                provenance="derived+reviewed",
            )
        )
    return cases
