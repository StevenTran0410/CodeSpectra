"""Retrieval types (RPA-034)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class RetrievalMode(StrEnum):
    HYBRID = "hybrid"
    VECTORLESS = "vectorless"


class RetrievalSection(StrEnum):
    ARCHITECTURE = "architecture"
    CONVENTIONS = "conventions"
    FEATURE_MAP = "feature_map"
    IMPORTANT_FILES = "important_files"
    GLOSSARY = "glossary"
    QA = "qa"


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
    quality: RetrievalQuality | None = None


class RetrieveRequest(BaseModel):
    snapshot_id: str
    query: str
    section: RetrievalSection
    mode: RetrievalMode = RetrievalMode.HYBRID
    max_results: int = 20
    min_confidence: float | None = None


class RetrievalCompareResponse(BaseModel):
    snapshot_id: str
    section: RetrievalSection
    query: str
    baseline: RetrievalBundle
    vectorless: RetrievalBundle
    precision_at_5_delta: float
    evidence_hit_rate_delta: float
    token_cost_delta: int


class TwoStageRequest(BaseModel):
    snapshot_id: str
    query: str
    section: RetrievalSection
    budget: int | None = None
    min_confidence: float | None = None


class StageCandidate(BaseModel):
    chunk_id: str
    rel_path: str
    chunk_index: int
    bm25_score: float
    token_estimate: int
    excerpt: str


class StageExpansion(BaseModel):
    seed_path: str
    symbol_refs: list[str]
    community_members: list[str]
    neighbor_files: list[str]
    net_new_count: int


class RankedChunk(BaseModel):
    chunk_id: str
    rel_path: str
    chunk_index: int
    score: float
    chunk_type: str = "block"
    bm25_component: float
    symbol_bonus: float
    module_bonus: float
    centrality_bonus: float
    token_estimate: int
    excerpt: str


@dataclass(frozen=True)
class RetrievalQuality:
    score_gap_ok: bool
    coverage: float
    path_entropy: float
    has_definition: bool
    flags: list[str]
    quality_label: str


class TwoStageStage3Result(BaseModel):
    ranked: list[RankedChunk]
    used_tokens: int
    budget_tokens: int
    used_cpp_ranker: bool


class TwoStageBundle(BaseModel):
    snapshot_id: str
    query: str
    section: RetrievalSection
    stage1: dict  # {"candidates": list[StageCandidate]}
    stage2: dict  # {"expansions": list[StageExpansion]}
    stage3: TwoStageStage3Result


# ── Impact retrieval types (RPA-060) ─────────────────────────────────────────


class ImpactRankedChunk(BaseModel):
    chunk_id: str
    rel_path: str
    chunk_index: int
    hop_distance: int | None  # 0=seed, 1-4=cone, None=bm25-only
    score: float
    hop_weight: float
    centrality_bonus: float
    community_bonus: float
    symbol_bonus: float
    bm25_boost: float
    token_estimate: int
    excerpt: str


class ImpactRetrievalRequest(BaseModel):
    snapshot_id: str
    seed_files: list[str]
    query: str | None = None
    max_hops: int = 3
    budget: int = 8000


class CommunityImpact(BaseModel):
    community_id: int
    member_count: int
    hub_paths: list[str]


class ImpactRetrievalBundle(BaseModel):
    seed_files: list[str]
    query: str | None
    impact_cone: dict[str, int]  # file -> hop_distance
    affected_communities: list[CommunityImpact]
    call_chains: list[dict]  # TraceStep serialized
    ranked_chunks: list[ImpactRankedChunk]
    risk_summary: dict  # high_risk_files, total_affected, affected_community_count


# ── RRF Multi-Signal Fusion Types (CS-252) ──────────────────────────────────


class SignalRankEntry(BaseModel):
    chunk_id: str
    rel_path: str
    rank: int
    raw_score: float
    signal_name: str
    excerpt: str = ""
    token_estimate: int = 0


class FusedRankEntry(BaseModel):
    chunk_id: str
    rel_path: str
    fused_score: float
    per_signal_ranks: dict[str, int]
    excerpt: str
    token_estimate: int = 0


class RerankedEntry(BaseModel):
    """Result from cross-encoder reranking of fused entries (CS-254)."""

    chunk_id: str
    rel_path: str
    fused_score: float
    rerank_score: float
    fused_rank: int
    excerpt: str
    token_estimate: int = 0


class RrfFusionBundle(BaseModel):
    snapshot_id: str
    query: str
    section: RetrievalSection
    bm25_signal: list[SignalRankEntry]
    graph_signal: list[SignalRankEntry]
    module_signal: list[SignalRankEntry]
    category_signal: list[SignalRankEntry]
    fused: list[FusedRankEntry]
    reranked: list[RerankedEntry] = Field(default_factory=list)
    reranker_status: Literal["ok", "no_gpu", "model_load_failed", "disabled"] = "ok"
    final: list[FusedRankEntry] = Field(default_factory=list)


class RrfFusionRequest(BaseModel):
    snapshot_id: str
    query: str
    section: RetrievalSection
    budget: int | None = None
    min_confidence: float | None = None
