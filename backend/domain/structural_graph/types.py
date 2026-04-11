"""Structural graph types (RPA-033, CS-102)."""
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


# ── CS-102: community detection types ────────────────────────────────────────

class CommunityInfo(BaseModel):
    community_id: int
    member_count: int
    hub_paths: list[str]
    modularity_contribution: float
    # IDs of communities that share at least one import edge with this community.
    # Populated by detect_communities(); empty until that runs.
    neighbor_community_ids: list[int] = []
    # True when this community has exactly 1 node AND no internal neighbours
    # (i.e. the singleton absorption pass could not find a community to absorb it into).
    is_singleton: bool = False
    llm_summary: str | None = None
    generated_at: str


class GraphCommunitiesResponse(BaseModel):
    snapshot_id: str
    total_communities: int
    communities: list[CommunityInfo]
    # Flat index: node_path -> community_id for all known nodes (built from DB rows)
    node_index: dict[str, int]


class NodeCommunityResponse(BaseModel):
    snapshot_id: str
    node_path: str
    community_id: int
    members: list[str]


class CyclesResponse(BaseModel):
    snapshot_id: str
    cycles: list[list[str]]
