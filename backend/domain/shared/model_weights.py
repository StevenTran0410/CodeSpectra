"""Local model-weights directory conventions for GPU-only domain models.

Weights are never downloaded silently/automatically — every loader in
domain/embeddings/ and domain/retrieval/cross_encoder_rerank.py passes an
explicit local path + local_files_only=True, so the underlying HF libraries
never reach the network on their own. A download only starts when the user
explicitly clicks "Download" in Settings, which calls start_download() below;
that runs as a background asyncio task so the request that triggered it
returns immediately, and progress is polled via download_status().
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path


def weights_dir(model_key: str) -> Path:
    """Absolute path to the folder a model's weights are downloaded into / must
    be manually placed in."""
    base = Path(os.getenv("CODESPECTRA_DATA_DIR", ".")).resolve()
    return base / "model_weights" / model_key


def weights_ready(model_key: str) -> bool:
    """True if the weights folder looks populated.

    Presence of config.json is used as a lightweight, format-agnostic signal —
    every Hugging Face model repo has one regardless of weight file format
    (safetensors vs. bin, sharded vs. single file).
    """
    return (weights_dir(model_key) / "config.json").is_file()


# model_key -> in-flight download task. A finished (or never-started) task is
# absent or done(); download_status() treats that as 'idle' unless weights_ready().
_download_tasks: dict[str, asyncio.Task] = {}
_download_errors: dict[str, str] = {}


def download_status(model_key: str) -> str:
    """One of 'idle' | 'downloading' | 'done' | 'failed'."""
    task = _download_tasks.get(model_key)
    if task is not None and not task.done():
        return "downloading"
    if weights_ready(model_key):
        return "done"
    if model_key in _download_errors:
        return "failed"
    return "idle"


def download_error(model_key: str) -> str | None:
    return _download_errors.get(model_key)


def start_download(model_key: str, repo_id: str) -> None:
    """Fire-and-forget background download. Idempotent — a no-op if a download
    for this model_key is already in flight."""
    task = _download_tasks.get(model_key)
    if task is not None and not task.done():
        return
    _download_errors.pop(model_key, None)
    _download_tasks[model_key] = asyncio.get_event_loop().create_task(
        _run_download(model_key, repo_id)
    )


async def _run_download(model_key: str, repo_id: str) -> None:
    try:
        await asyncio.to_thread(_download_sync, model_key, repo_id)
    except Exception as exc:
        _download_errors[model_key] = str(exc)


def _download_sync(model_key: str, repo_id: str) -> None:
    from huggingface_hub import snapshot_download  # noqa: PLC0415 — optional extra

    target = weights_dir(model_key)
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=str(target))
