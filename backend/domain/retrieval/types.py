"""Retrieval types (RPA-034)."""
from enum import Enum

from pydantic import BaseModel


class RetrievalMode(str, Enum):
    HYBRID = "hybrid"
    VECTORLESS = "vectorless"


class RetrievalSection(str, Enum):
    ARCHITECTURE = "architecture"
    CONVENTIONS = "conventions"
    FEATURE_MAP = "feature_map"
    IMPORTANT_FILES = "important_files"
    GLOSSARY = "glossary"


class BuildRetrievalIndexRequest(BaseModel):
    snapshot_id: str
    force_rebuild: bool = True


class BuildRetrievalIndexResponse(BaseModel):
    snapshot_id: str
    chunk_count: int
    files_indexed: int
    generated_at: str


class RetrievalEvidence(BaseModel):
    chunk_id: str
    rel_path: str
    chunk_index: int
    reason_codes: list[str]
    score: float
    token_estimate: int
    excerpt: str


class RetrievalBundle(BaseModel):
    snapshot_id: str
    mode: RetrievalMode
    section: RetrievalSection
    query: str
    budget_tokens: int
    used_tokens: int
    evidences: list[RetrievalEvidence]


class RetrieveRequest(BaseModel):
    snapshot_id: str
    query: str
    section: RetrievalSection
    mode: RetrievalMode = RetrievalMode.HYBRID
    max_results: int = 20


class RetrievalCompareResponse(BaseModel):
    snapshot_id: str
    section: RetrievalSection
    query: str
    baseline: RetrievalBundle
    vectorless: RetrievalBundle
    precision_at_5_delta: float
    evidence_hit_rate_delta: float
    token_cost_delta: int
