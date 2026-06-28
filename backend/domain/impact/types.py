"""Service-layer Pydantic models for impact analysis (RPA-060)."""
from __future__ import annotations

from pydantic import BaseModel


class BlastRadiusRequest(BaseModel):
    snapshot_id: str
    changed_files: list[str]
    report_id: str | None = None
    max_hops: int = 3
    include_call_chains: bool = True


class RiskFile(BaseModel):
    rel_path: str
    hop_distance: int
    centrality_rank: int | None = None


class BlastRadiusResponse(BaseModel):
    changed_files: list[str]
    # total_affected, by_hop, affected_communities, high_risk_files, call_chains,
    # cochange_hints (git-history co-change, CS-246), convention_violations (CS-246)
    blast_radius: dict
    subgraph: dict      # nodes, edges, seed_files, hop_colors
    context_chunks: list[dict]  # ImpactRankedChunk serialized


class PlanRequest(BaseModel):
    snapshot_id: str
    task_description: str
    report_id: str
    provider_id: str | None = None
    model_id: str | None = None


class PlanResponse(BaseModel):
    task: str
    suggested_files: list[dict]
    patterns_to_follow: list[dict]
    violations_to_avoid: list[dict]
    related_features: list[dict]
    important_files_affected: list[dict]
    tests_to_add: list[dict]
    blast_radius_preview: dict | None = None
