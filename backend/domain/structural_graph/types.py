"""Structural graph types (RPA-033)."""
from pydantic import BaseModel


class BuildGraphRequest(BaseModel):
    snapshot_id: str
    force_rebuild: bool = True


class GraphNodeScore(BaseModel):
    rel_path: str
    indegree: int
    outdegree: int
    score: int


class StructuralGraphSummary(BaseModel):
    snapshot_id: str
    total_nodes: int
    total_edges: int
    external_edges: int
    entrypoints: list[str]
    top_central_files: list[GraphNodeScore]
    generated_at: str
    native_toolchain: str | None = None


class BuildGraphResponse(BaseModel):
    summary: StructuralGraphSummary


class GraphEdge(BaseModel):
    snapshot_id: str
    src_path: str
    dst_path: str
    edge_type: str
    is_external: bool


class GraphEdgesResponse(BaseModel):
    snapshot_id: str
    edges: list[GraphEdge]


class GraphNeighborsResponse(BaseModel):
    snapshot_id: str
    seed_path: str
    hops: int
    nodes: list[str]
    edges: list[GraphEdge]
