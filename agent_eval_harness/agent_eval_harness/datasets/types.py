from typing import Any, Literal

from pydantic import BaseModel

from agent_eval_harness.metrics.registry import DATASET_KINDS

Provenance = Literal["synthetic", "handwritten", "generated+reviewed", "derived+reviewed"]

# Single source of truth for export-eligible provenance values.
TRUSTED_PROVENANCE: tuple[str, ...] = ("generated+reviewed", "handwritten", "derived+reviewed")

__all__ = ["DATASET_KINDS", "DatasetCase", "Provenance", "TRUSTED_PROVENANCE"]

class DatasetCase(BaseModel):
    id: str
    dataset: str            # dataset_id, e.g. "t2_guard_v1"
    kind: str               # one of DATASET_KINDS
    input: dict[str, Any]
    expected: dict[str, Any] | None = None
    labels: dict[str, Any] | None = None
    provenance: Provenance
