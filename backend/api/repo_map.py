"""Repo map endpoints."""
from fastapi import APIRouter, HTTPException

from domain.repo_map.service import RepoMapService
from domain.repo_map.types import (
    BuildRepoMapRequest,
    BuildRepoMapResponse,
    RepoMapCsvResponse,
    RepoMapSummary,
    SymbolsResponse,
)

router = APIRouter(tags=["repo-map"])
_service = RepoMapService()


@router.post("/build", response_model=BuildRepoMapResponse)
async def build_repo_map(body: BuildRepoMapRequest) -> BuildRepoMapResponse:
    try:
        return await _service.build(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/summary/{snapshot_id}", response_model=RepoMapSummary)
async def get_repo_map_summary(snapshot_id: str) -> RepoMapSummary:
    return await _service.summary(snapshot_id)


@router.get("/symbols/{snapshot_id}", response_model=SymbolsResponse)
async def list_symbols(snapshot_id: str, limit: int = 500, path_prefix: str | None = None) -> SymbolsResponse:
    return await _service.symbols(snapshot_id, limit=limit, path_prefix=path_prefix)


@router.get("/search/{snapshot_id}", response_model=SymbolsResponse)
async def search_symbols(snapshot_id: str, q: str, limit: int = 120) -> SymbolsResponse:
    return await _service.search(snapshot_id, q=q, limit=limit)


@router.get("/export/{snapshot_id}", response_model=RepoMapCsvResponse)
async def export_symbols_csv(snapshot_id: str, exclude_tests: bool = True) -> RepoMapCsvResponse:
    return await _service.export_csv(snapshot_id, exclude_tests=exclude_tests)
