"""Local folder repository endpoints."""
from fastapi import APIRouter, HTTPException

from domain.local_repo.service import LocalRepoService
from domain.local_repo.types import (
    AddLocalRepoRequest,
    CloneFromUrlRequest,
    EstimateFileCountResponse,
    LocalRepo,
    SetActiveSnapshotRequest,
    SetBranchRequest,
    UpdateRepoSettingsRequest,
    ValidateFolderRequest,
    ValidateFolderResponse,
)

router = APIRouter(tags=["local-repo"])
_service = LocalRepoService()


@router.post("/validate", response_model=ValidateFolderResponse)
async def validate_folder(body: ValidateFolderRequest) -> ValidateFolderResponse:
    return await _service.validate(body)


@router.get("/", response_model=list[LocalRepo])
async def list_repos() -> list[LocalRepo]:
    return await _service.list_all()


@router.post("/", response_model=LocalRepo, status_code=201)
async def add_repo(body: AddLocalRepoRequest) -> LocalRepo:
    return await _service.add(body)


@router.post("/clone", response_model=LocalRepo, status_code=201)
async def clone_repo(body: CloneFromUrlRequest) -> LocalRepo:
    try:
        return await _service.clone_from_url(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{repo_id}", status_code=204)
async def remove_repo(repo_id: str) -> None:
    try:
        await _service.remove(repo_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{repo_id}/revalidate", response_model=LocalRepo)
async def revalidate_repo(repo_id: str) -> LocalRepo:
    return await _service.revalidate(repo_id)


@router.get("/{repo_id}/branches", response_model=list[str])
async def list_branches(repo_id: str, refresh: bool = False) -> list[str]:
    return await _service.list_branches(repo_id, refresh=refresh)


@router.post("/{repo_id}/branch", response_model=LocalRepo)
async def set_branch(repo_id: str, body: SetBranchRequest) -> LocalRepo:
    return await _service.set_branch(repo_id, body)


@router.post("/{repo_id}/active-snapshot", response_model=LocalRepo)
async def set_active_snapshot(repo_id: str, body: SetActiveSnapshotRequest) -> LocalRepo:
    return await _service.set_active_snapshot(repo_id, body.snapshot_id)


@router.post("/{repo_id}/settings", response_model=LocalRepo)
async def update_repo_settings(repo_id: str, body: UpdateRepoSettingsRequest) -> LocalRepo:
    return await _service.update_settings(repo_id, body)


@router.get("/{repo_id}/estimate-file-count", response_model=EstimateFileCountResponse)
async def estimate_file_count(repo_id: str) -> EstimateFileCountResponse:
    return await _service.estimate_file_count(repo_id)
