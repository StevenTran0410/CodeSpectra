import importlib.metadata

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["app"])


class VersionResponse(BaseModel):
    status: str
    version: str


@router.get("/health", response_model=VersionResponse)
async def health() -> VersionResponse:
    try:
        version = importlib.metadata.version("codespectra-backend")
    except importlib.metadata.PackageNotFoundError:
        version = "dev"
    return VersionResponse(status="ok", version=version)
