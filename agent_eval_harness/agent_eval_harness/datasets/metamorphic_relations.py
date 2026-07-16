"""Loads metamorphic relation definitions from discovery/metamorphic_relations.yaml — a
git-reviewed config artifact (contract_harvest.py's D1 keyword-signal load pattern: degrade to
empty on any parse failure, never raise)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

logger = logging.getLogger("agent_eval_harness.datasets.metamorphic_relations")

_RELATIONS_PATH = Path(__file__).parent.parent / "discovery" / "metamorphic_relations.yaml"


class OpSpec(BaseModel):
    op: str
    params: dict[str, Any] = {}


class RelationSpec(BaseModel):
    id: str
    applies_to: str
    transform: OpSpec | None = None
    invariant: OpSpec


def load_relations() -> dict[str, RelationSpec]:
    """Returns {relation_id: RelationSpec}; {} on any missing/malformed file (degrade, don't break)."""
    if not _RELATIONS_PATH.exists():
        return {}
    try:
        data = yaml.safe_load(_RELATIONS_PATH.read_text(encoding="utf-8")) or {}
        raw_relations = data.get("relations") or []
        return {r["id"]: RelationSpec.model_validate(r) for r in raw_relations}
    except Exception as e:
        logger.warning(f"metamorphic_relations: failed to load {_RELATIONS_PATH}: {e}")
        return {}


def get_relation(relation_id: str) -> RelationSpec | None:
    return load_relations().get(relation_id)
