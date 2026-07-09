"""GPU reranker global toggle — stored as a flag in app_metadata.

Mirrors the cloud-consent pattern (api/consent.py): a single global boolean
flag in app_metadata, not workspace-scoped, since this controls a machine-wide
capability (GPU availability), not a per-project preference.

GPU detection itself lives in domain/retrieval/cross_encoder_rerank.py (the
domain layer never imports from api/, so the single source of truth for
"is there a usable GPU" stays in domain and this module imports from there).
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from domain.retrieval.cross_encoder_rerank import _GPU_RERANKER_ENABLED_KEY, _MIN_VRAM_GB, detect_gpu
from infrastructure.db.database import get_db

router = APIRouter(tags=["gpu-reranker"])


class GpuRerankerStatus(BaseModel):
    enabled: bool
    gpu_available: bool
    vram_gb: float | None = None


class SetGpuRerankerRequest(BaseModel):
    enabled: bool


async def _get_enabled_flag() -> bool:
    db = get_db()
    async with db.execute(
        "SELECT value FROM app_metadata WHERE key = ?", (_GPU_RERANKER_ENABLED_KEY,)
    ) as cur:
        row = await cur.fetchone()
    return row is not None and row["value"] == "true"


@router.get("/status", response_model=GpuRerankerStatus)
async def get_gpu_reranker_status() -> GpuRerankerStatus:
    gpu_ok, vram_gb = detect_gpu()
    enabled = await _get_enabled_flag() if gpu_ok else False
    return GpuRerankerStatus(enabled=enabled, gpu_available=gpu_ok, vram_gb=vram_gb)


@router.post("/status", response_model=GpuRerankerStatus)
async def set_gpu_reranker_status(body: SetGpuRerankerRequest) -> GpuRerankerStatus:
    gpu_ok, vram_gb = detect_gpu()
    if body.enabled and not gpu_ok:
        raise HTTPException(
            status_code=400,
            detail=f"No GPU with >= {_MIN_VRAM_GB}GB VRAM detected; cannot enable GPU reranker.",
        )

    db = get_db()
    value = "true" if body.enabled else "false"
    await db.execute(
        "INSERT OR REPLACE INTO app_metadata (key, value) VALUES (?, ?)",
        (_GPU_RERANKER_ENABLED_KEY, value),
    )
    await db.commit()
    return GpuRerankerStatus(enabled=body.enabled, gpu_available=gpu_ok, vram_gb=vram_gb)
