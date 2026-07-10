"""Local embedding model toggle — stored as a flag in app_metadata.

Mirrors api/gpu_reranker.py byte-for-byte in structure: a single global
boolean flag in app_metadata (not workspace-scoped, since this controls a
machine-wide GPU capability), GET + POST routes for status and toggle.

GPU detection lives in domain/shared/gpu.py; the embedding model singleton
lives in domain/embeddings/local_model.py.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from domain.embeddings.local_model import (
    _HF_MODEL_URL,
    _LOCAL_EMBEDDING_ENABLED_KEY,
    _MIN_VRAM_GB,
    _MODEL_ID,
    _MODEL_KEY,
    embedding_weights_dir,
    embedding_weights_ready,
    local_embedding_available,
)
from domain.shared.gpu import detect_gpu
from domain.shared.model_weights import download_error, download_status, start_download
from infrastructure.db.database import get_db

router = APIRouter(tags=["local-embedding"])

_MIN_VRAM_GB_THRESHOLD = _MIN_VRAM_GB  # forward from local_model for error messages


class LocalEmbeddingStatus(BaseModel):
    enabled: bool
    gpu_available: bool
    vram_gb: float | None = None
    model_id: str = "Qwen/Qwen3-Embedding-0.6B"
    weights_ready: bool
    weights_dir: str
    download_url: str
    download_status: str
    download_error: str | None = None


class SetLocalEmbeddingRequest(BaseModel):
    enabled: bool


def _status(enabled: bool, gpu_ok: bool, vram_gb: float | None) -> LocalEmbeddingStatus:
    return LocalEmbeddingStatus(
        enabled=enabled,
        gpu_available=gpu_ok,
        vram_gb=vram_gb,
        weights_ready=embedding_weights_ready(),
        weights_dir=embedding_weights_dir(),
        download_url=_HF_MODEL_URL,
        download_status=download_status(_MODEL_KEY),
        download_error=download_error(_MODEL_KEY),
    )


@router.get("/status", response_model=LocalEmbeddingStatus)
async def get_local_embedding_status() -> LocalEmbeddingStatus:
    gpu_ok, vram_gb = detect_gpu()
    enabled = await local_embedding_available()
    return _status(enabled, gpu_ok, vram_gb)


@router.post("/download", response_model=LocalEmbeddingStatus)
async def download_embedding_weights() -> LocalEmbeddingStatus:
    gpu_ok, vram_gb = detect_gpu()
    if not gpu_ok:
        raise HTTPException(
            status_code=400,
            detail=f"No GPU with >= {_MIN_VRAM_GB_THRESHOLD}GB VRAM detected; cannot download.",
        )
    start_download(_MODEL_KEY, _MODEL_ID)
    enabled = await local_embedding_available()
    return _status(enabled, gpu_ok, vram_gb)


@router.post("/status", response_model=LocalEmbeddingStatus)
async def set_local_embedding_status(body: SetLocalEmbeddingRequest) -> LocalEmbeddingStatus:
    gpu_ok, vram_gb = detect_gpu()
    if body.enabled and not gpu_ok:
        raise HTTPException(
            status_code=400,
            detail=f"No GPU with >= {_MIN_VRAM_GB_THRESHOLD}GB VRAM detected; cannot enable local embedding model.",
        )
    if body.enabled and not embedding_weights_ready():
        raise HTTPException(
            status_code=400,
            detail=f"Model weights not found at {embedding_weights_dir()}. Download from "
            f"{_HF_MODEL_URL} and place the files there before enabling.",
        )

    db = get_db()
    value = "true" if body.enabled else "false"
    await db.execute(
        "INSERT OR REPLACE INTO app_metadata (key, value) VALUES (?, ?)",
        (_LOCAL_EMBEDDING_ENABLED_KEY, value),
    )
    await db.commit()
    return _status(body.enabled, gpu_ok, vram_gb)
