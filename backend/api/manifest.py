"""Manifest endpoints."""
from fastapi import APIRouter, HTTPException

from domain.manifest.service import ManifestService
from domain.manifest.types import (
    BuildManifestRequest,
    BuildManifestResponse,
    ManifestFileContentResponse,
    ManifestPreviewResponse,
    ManifestTreeResponse,
)

router = APIRouter(tags=["manifest"])
_service = ManifestService()


@router.post("/build", response_model=BuildManifestResponse)
async def build_manifest(body: BuildManifestRequest) -> BuildManifestResponse:
    try:
        return await _service.build(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/preview/{snapshot_id}", response_model=ManifestPreviewResponse)
async def preview_manifest(snapshot_id: str, limit: int = 200) -> ManifestPreviewResponse:
    return await _service.preview(snapshot_id, limit)


@router.get("/tree/{snapshot_id}", response_model=ManifestTreeResponse)
async def tree_manifest(snapshot_id: str, limit: int = 5000) -> ManifestTreeResponse:
    return await _service.tree(snapshot_id, limit)


@router.get("/file/{snapshot_id}", response_model=ManifestFileContentResponse)
async def read_manifest_file(snapshot_id: str, path: str, max_bytes: int = 200000) -> ManifestFileContentResponse:
    try:
        return await _service.read_file(snapshot_id, path, max_bytes=max_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
