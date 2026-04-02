"""Local folder repository endpoints."""
from fastapi import APIRouter

from domain.local_repo.service import LocalRepoService
from domain.local_repo.types import (
    AddLocalRepoRequest,
    LocalRepo,
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


@router.delete("/{repo_id}", status_code=204)
async def remove_repo(repo_id: str) -> None:
    await _service.remove(repo_id)


@router.post("/{repo_id}/revalidate", response_model=LocalRepo)
async def revalidate_repo(repo_id: str) -> LocalRepo:
    return await _service.revalidate(repo_id)
