"""Shared GPU detection utilities for domain-layer models.

Hoisted from domain/retrieval/cross_encoder_rerank.py so both the
cross-encoder reranker and the local embedding model can share the same
detect_gpu / release_gpu_cache implementations without importing from
each other (which would create a circular dependency).
"""
from __future__ import annotations

# Minimum VRAM (GB) required for GPU-gated domain models.
GPU_MIN_VRAM_GB = 2.0

_gpu_cache: dict[float, tuple[bool, float | None]] = {}


def detect_gpu(min_vram_gb: float = GPU_MIN_VRAM_GB) -> tuple[bool, float | None]:
    """Returns (meets_min_vram, vram_gb).

    Never raises — any torch/CUDA import or probing error is treated as
    'no usable GPU', not a crash. Results are cached after the first call.
    """
    if min_vram_gb in _gpu_cache:
        return _gpu_cache[min_vram_gb]
    try:
        import torch

        if not torch.cuda.is_available():
            gpu_available, vram_gb = False, None
        else:
            total_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_gb = round(total_bytes / (1024**3), 2)
            gpu_available = vram_gb >= min_vram_gb
    except Exception:
        gpu_available, vram_gb = False, None
    _gpu_cache[min_vram_gb] = (gpu_available, vram_gb)
    return _gpu_cache[min_vram_gb]


def release_gpu_cache() -> None:
    """Return PyTorch's CUDA caching-allocator pool to the driver.

    Call between batches, not within a single inference call.
    Never raises — a no-op if torch/CUDA aren't available.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
