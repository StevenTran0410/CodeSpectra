"""Structural graph endpoints."""
from fastapi import APIRouter, HTTPException, Query

from domain.structural_graph.service import StructuralGraphService
from domain.structural_graph.types import (
    BuildGraphRequest,
    BuildGraphResponse,
    GraphEdgesResponse,
    GraphNeighborsResponse,
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
async def list_graph_edges(snapshot_id: str, limit: int = 2000) -> GraphEdgesResponse:
    return await _service.edges(snapshot_id, limit=limit)


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
