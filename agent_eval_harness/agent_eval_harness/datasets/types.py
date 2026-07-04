from typing import Any, Literal

from pydantic import BaseModel

Provenance = Literal["synthetic", "handwritten", "generated+reviewed"]

DATASET_KINDS = ("guard_classification", "qa_testset", "decomposition_gold", "sufficiency_labeled")

class DatasetCase(BaseModel):
    id: str
    dataset: str            # dataset_id, e.g. "t2_guard_v1"
    kind: str               # one of DATASET_KINDS
    input: dict[str, Any]
    expected: dict[str, Any] | None = None
    labels: dict[str, Any] | None = None
    provenance: Provenance
