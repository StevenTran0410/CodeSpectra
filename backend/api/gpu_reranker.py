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

from domain.retrieval.cross_encoder_rerank import (
    _GPU_RERANKER_ENABLED_KEY,
    _HF_MODEL_URL,
    _MIN_VRAM_GB,
    _MODEL_ID,
    _MODEL_KEY,
    detect_gpu,
    reranker_weights_dir,
    reranker_weights_ready,
)
from domain.shared.model_weights import download_error, download_status, start_download
from infrastructure.db.database import get_db

router = APIRouter(tags=["gpu-reranker"])


class GpuRerankerStatus(BaseModel):
    enabled: bool
    gpu_available: bool
    vram_gb: float | None = None
    weights_ready: bool
    weights_dir: str
    download_url: str
    download_status: str
    download_error: str | None = None


class SetGpuRerankerRequest(BaseModel):
    enabled: bool


async def _get_enabled_flag() -> bool:
    db = get_db()
    async with db.execute(
        "SELECT value FROM app_metadata WHERE key = ?", (_GPU_RERANKER_ENABLED_KEY,)
    ) as cur:
        row = await cur.fetchone()
    return row is not None and row["value"] == "true"


def _status(enabled: bool, gpu_ok: bool, vram_gb: float | None) -> GpuRerankerStatus:
    return GpuRerankerStatus(
        enabled=enabled,
        gpu_available=gpu_ok,
        vram_gb=vram_gb,
        weights_ready=reranker_weights_ready(),
        weights_dir=reranker_weights_dir(),
        download_url=_HF_MODEL_URL,
        download_status=download_status(_MODEL_KEY),
        download_error=download_error(_MODEL_KEY),
    )


@router.get("/status", response_model=GpuRerankerStatus)
async def get_gpu_reranker_status() -> GpuRerankerStatus:
    gpu_ok, vram_gb = detect_gpu()
    enabled = await _get_enabled_flag() if gpu_ok and reranker_weights_ready() else False
    return _status(enabled, gpu_ok, vram_gb)


@router.post("/download", response_model=GpuRerankerStatus)
async def download_reranker_weights() -> GpuRerankerStatus:
    gpu_ok, vram_gb = detect_gpu()
    if not gpu_ok:
        raise HTTPException(
            status_code=400,
            detail=f"No GPU with >= {_MIN_VRAM_GB}GB VRAM detected; cannot download.",
        )
    start_download(_MODEL_KEY, _MODEL_ID)
    enabled = await _get_enabled_flag() if reranker_weights_ready() else False
    return _status(enabled, gpu_ok, vram_gb)


@router.post("/status", response_model=GpuRerankerStatus)
async def set_gpu_reranker_status(body: SetGpuRerankerRequest) -> GpuRerankerStatus:
    gpu_ok, vram_gb = detect_gpu()
    if body.enabled and not gpu_ok:
        raise HTTPException(
            status_code=400,
            detail=f"No GPU with >= {_MIN_VRAM_GB}GB VRAM detected; cannot enable GPU reranker.",
        )
    if body.enabled and not reranker_weights_ready():
        raise HTTPException(
            status_code=400,
            detail=f"Model weights not found at {reranker_weights_dir()}. Download from "
            f"{_HF_MODEL_URL} and place the files there before enabling.",
        )

    db = get_db()
    value = "true" if body.enabled else "false"
    await db.execute(
        "INSERT OR REPLACE INTO app_metadata (key, value) VALUES (?, ?)",
        (_GPU_RERANKER_ENABLED_KEY, value),
    )
    await db.commit()
    return _status(body.enabled, gpu_ok, vram_gb)
