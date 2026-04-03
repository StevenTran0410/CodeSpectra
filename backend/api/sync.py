"""Sync engine endpoints."""
from fastapi import APIRouter, HTTPException

from domain.sync_engine.service import SyncEngineService
from domain.sync_engine.types import PrepareSnapshotRequest, RepoSnapshot

router = APIRouter(tags=["sync"])
_service = SyncEngineService()


@router.post("/prepare", response_model=RepoSnapshot, status_code=201)
async def prepare_snapshot(body: PrepareSnapshotRequest) -> RepoSnapshot:
    try:
        return await _service.prepare_snapshot(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/snapshot/{snapshot_id}", response_model=RepoSnapshot)
async def get_snapshot(snapshot_id: str) -> RepoSnapshot:
    return await _service.get_snapshot(snapshot_id)


@router.get("/repo/{repo_id}", response_model=list[RepoSnapshot])
async def list_snapshots(repo_id: str, limit: int = 20) -> list[RepoSnapshot]:
    return await _service.list_for_repo(repo_id, limit)
