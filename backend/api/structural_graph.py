"""Structural graph endpoints (RPA-033, CS-102)."""
from fastapi import APIRouter, HTTPException, Query

from domain.structural_graph.service import StructuralGraphService
from domain.structural_graph.types import (
    BuildGraphRequest,
    BuildGraphResponse,
    CyclesResponse,
    GraphCommunitiesResponse,
    GraphEdgesResponse,
    GraphNeighborsResponse,
    NodeCommunityResponse,
    StructuralGraphSummary,
)

router = APIRouter(tags=["structural-graph"])
_service = StructuralGraphService()


@router.post("/build", response_model=BuildGraphResponse)
async def build_graph(body: BuildGraphRequest) -> BuildGraphResponse:
    try:
        return await _service.build(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/summary/{snapshot_id}", response_model=StructuralGraphSummary)
async def get_graph_summary(snapshot_id: str) -> StructuralGraphSummary:
    return await _service.summary(snapshot_id)


@router.get("/edges/{snapshot_id}", response_model=GraphEdgesResponse)
async def list_graph_edges(
    snapshot_id: str, limit: int = 2000, internal_only: bool = False
) -> GraphEdgesResponse:
    return await _service.edges(snapshot_id, limit=limit, internal_only=internal_only)


@router.get("/neighbors/{snapshot_id}", response_model=GraphNeighborsResponse)
async def graph_neighbors(
    snapshot_id: str,
    path: str = Query(..., description="Seed file path"),
    hops: int = Query(1, ge=1, le=4),
    limit: int = Query(300, ge=10, le=2000),
) -> GraphNeighborsResponse:
    try:
        return await _service.neighbors(snapshot_id, seed_path=path, hops=hops, limit=limit)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


# ── CS-102: community detection endpoints ─────────────────────────────────────

@router.get("/communities/{snapshot_id}", response_model=GraphCommunitiesResponse)
async def list_communities(snapshot_id: str) -> GraphCommunitiesResponse:
    """Return cached community partition from the last Louvain run."""
    try:
        return await _service.list_communities(snapshot_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/community/{snapshot_id}", response_model=NodeCommunityResponse)
async def community_for_node(
    snapshot_id: str,
    path: str = Query(..., description="Node file path"),
) -> NodeCommunityResponse:
    """Return community ID and all members for a given node."""
    try:
        return await _service.community_for_node(snapshot_id, path)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/cycles/{snapshot_id}", response_model=CyclesResponse)
async def graph_cycles(snapshot_id: str) -> CyclesResponse:
    """Return circular import cycles (strongly connected components)."""
    try:
        return await _service.cycles(snapshot_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/{snapshot_id}")
async def export_graph_json(snapshot_id: str) -> dict:
    """Export full graph (nodes, edges, communities, cycles) as a single JSON blob.

    Use for debugging: share the output to diagnose import resolution or
    community clustering issues.
    """
    try:
        return await _service.export_graph_json(snapshot_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
