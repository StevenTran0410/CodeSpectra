"""Manifest endpoints."""
from fastapi import APIRouter

from domain.manifest.service import ManifestService
from shared.http_utils import handle_value_error
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
@handle_value_error
async def build_manifest(body: BuildManifestRequest) -> BuildManifestResponse:
    return await _service.build(body)


@router.get("/preview/{snapshot_id}", response_model=ManifestPreviewResponse)
async def preview_manifest(snapshot_id: str, limit: int = 200) -> ManifestPreviewResponse:
    return await _service.preview(snapshot_id, limit)


@router.get("/tree/{snapshot_id}", response_model=ManifestTreeResponse)
async def tree_manifest(snapshot_id: str, limit: int = 5000) -> ManifestTreeResponse:
    return await _service.tree(snapshot_id, limit)


@router.get("/file/{snapshot_id}", response_model=ManifestFileContentResponse)
@handle_value_error
async def read_manifest_file(snapshot_id: str, path: str, max_bytes: int = 200000) -> ManifestFileContentResponse:
    return await _service.read_file(snapshot_id, path, max_bytes=max_bytes)
