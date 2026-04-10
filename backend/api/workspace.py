from fastapi import APIRouter

from domain.workspace.service import WorkspaceService
from domain.workspace.types import CreateWorkspaceRequest, RenameWorkspaceRequest, Workspace

router = APIRouter(tags=["workspace"])
_service = WorkspaceService()


@router.get("/", response_model=list[Workspace])
async def list_workspaces() -> list[Workspace]:
    return await _service.list_all()


@router.post("/", response_model=Workspace, status_code=201)
async def create_workspace(body: CreateWorkspaceRequest) -> Workspace:
    return await _service.create(body.name, body.description)


@router.get("/{workspace_id}", response_model=Workspace)
async def get_workspace(workspace_id: str) -> Workspace:
    return await _service.get_by_id(workspace_id)


@router.put("/{workspace_id}", response_model=Workspace)
async def rename_workspace(workspace_id: str, body: RenameWorkspaceRequest) -> Workspace:
    return await _service.rename(workspace_id, body.name)


@router.delete("/{workspace_id}", status_code=204)
async def delete_workspace(workspace_id: str) -> None:
    await _service.delete(workspace_id)
