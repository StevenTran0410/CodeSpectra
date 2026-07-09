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
    _LOCAL_EMBEDDING_ENABLED_KEY,
    _MIN_VRAM_GB,
    local_embedding_available,
)
from domain.shared.gpu import detect_gpu
from infrastructure.db.database import get_db

router = APIRouter(tags=["local-embedding"])

_MIN_VRAM_GB_THRESHOLD = _MIN_VRAM_GB  # forward from local_model for error messages


class LocalEmbeddingStatus(BaseModel):
    enabled: bool
    gpu_available: bool
    vram_gb: float | None = None
    model_id: str = "Qwen/Qwen3-Embedding-0.6B"


class SetLocalEmbeddingRequest(BaseModel):
    enabled: bool


@router.get("/status", response_model=LocalEmbeddingStatus)
async def get_local_embedding_status() -> LocalEmbeddingStatus:
    gpu_ok, vram_gb = detect_gpu()
    enabled = await local_embedding_available()
    return LocalEmbeddingStatus(enabled=enabled, gpu_available=gpu_ok, vram_gb=vram_gb)


@router.post("/status", response_model=LocalEmbeddingStatus)
async def set_local_embedding_status(body: SetLocalEmbeddingRequest) -> LocalEmbeddingStatus:
    gpu_ok, vram_gb = detect_gpu()
    if body.enabled and not gpu_ok:
        raise HTTPException(
            status_code=400,
            detail=f"No GPU with >= {_MIN_VRAM_GB_THRESHOLD}GB VRAM detected; cannot enable local embedding model.",
        )

    db = get_db()
    value = "true" if body.enabled else "false"
    await db.execute(
        "INSERT OR REPLACE INTO app_metadata (key, value) VALUES (?, ?)",
        (_LOCAL_EMBEDDING_ENABLED_KEY, value),
    )
    await db.commit()
    return LocalEmbeddingStatus(enabled=body.enabled, gpu_available=gpu_ok, vram_gb=vram_gb)
