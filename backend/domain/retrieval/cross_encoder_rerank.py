"""Cross-encoder reranking using jinaai/jina-reranker-v3: a GPU-only reranker that rescores fused entries using a local cross-encoder model. Lazy-loaded singleton, 4-bit quantization via bitsandbytes, uses trust_remote_code=True. Weights only arrive via an explicit user-triggered download (Settings "Download" button) or manual placement in weights_dir(_MODEL_KEY) — this module itself never reaches the network."""

from __future__ import annotations

import time

from domain.shared.gpu import GPU_MIN_VRAM_GB, detect_gpu, release_gpu_cache
from domain.shared.model_weights import weights_dir, weights_ready
from shared.logger import logger

from .types import FusedRankEntry, RerankedEntry

# Lazy-loaded singleton model
_model = None

# Minimum VRAM (GB) required to consider the GPU usable for this model
_MIN_VRAM_GB = GPU_MIN_VRAM_GB

_MODEL_KEY = "jina-reranker-v3"
_MODEL_ID = "jinaai/jina-reranker-v3"
_HF_MODEL_URL = "https://huggingface.co/jinaai/jina-reranker-v3"


def reranker_weights_ready() -> bool:
    return weights_ready(_MODEL_KEY)


def reranker_weights_dir() -> str:
    return str(weights_dir(_MODEL_KEY))

# Jina-reranker-v3 max sequence length (verified against model card)
# This is the maximum tokens the model can accept for both query + passages combined
_MAX_SEQ_LENGTH = 8192

# torch/transformers are an optional extra ([reranker] in pyproject.toml); most installs
# don't have them, so every import of torch/transformers here must be deferred into
# function bodies to keep this module (and the whole retrieval pipeline) importable
# without it.


def _check_gpu_available() -> bool:
    """Check if CUDA GPU is available (cached). Backward-compat wrapper."""
    available, _ = detect_gpu()
    return available


def release_gpu_cache() -> None:
    """Return PyTorch's CUDA caching-allocator pool to the driver. PyTorch holds
    freed tensors in a pool for fast reuse rather than releasing VRAM immediately.
    Call between batches, not within a single rerank call (where reuse is wanted).
    Never raises — a no-op if torch/CUDA aren't available."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# Reserved VRAM (GB) for model weights + CUDA context + typical desktop GPU usage
# (compositor, browser GPU process, etc.) before any budget is left for reranking.
# Conservative estimate for model load overhead.
_RESERVED_VRAM_GB = 3.5

# Worst case: every passage gets truncated up to _MAX_SEQ_LENGTH chars, ~4 chars/token.
_TOKENS_PER_CANDIDATE_WORST_CASE = _MAX_SEQ_LENGTH / 4

# Quadratic-attention-memory constant (GB per token^2)
_QUADRATIC_MEM_CONSTANT_GB_PER_TOKEN2 = 50.0 / (100 * _TOKENS_PER_CANDIDATE_WORST_CASE) ** 2

_MIN_RERANK_TOP_N = 3
_MAX_RERANK_TOP_N = 30  # per-batch ceiling regardless of VRAM
_SAFETY_FACTOR = 0.85  

def recommended_rerank_batch_size(vram_gb: float | None) -> int:
    if not vram_gb or vram_gb <= 0:
        return _MIN_RERANK_TOP_N

    usable_gb = max(0.0, vram_gb - _RESERVED_VRAM_GB) * _SAFETY_FACTOR
    if usable_gb <= 0:
        return _MIN_RERANK_TOP_N

    max_total_tokens = (usable_gb / _QUADRATIC_MEM_CONSTANT_GB_PER_TOKEN2) ** 0.5
    top_n = int(max_total_tokens / _TOKENS_PER_CANDIDATE_WORST_CASE)
    return max(_MIN_RERANK_TOP_N, min(_MAX_RERANK_TOP_N, top_n))


def load_reranker() -> bool:
    """Load jina-reranker-v3 model with 4-bit quantization.

    Returns:
        True if model loaded successfully, False if GPU unavailable or load failed.
    """
    global _model

    if _model is not None:
        return True  # Already loaded

    if not _check_gpu_available():
        logger.warning("[reranker] GPU not available; cross-encoder reranking unavailable")
        return False

    if not reranker_weights_ready():
        logger.warning(
            f"[reranker] weights not found at {reranker_weights_dir()} — "
            f"download from {_HF_MODEL_URL} and place the files there"
        )
        return False

    try:
        import torch
        from transformers import AutoModel, BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        _model = AutoModel.from_pretrained(
            reranker_weights_dir(),
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map="cuda",
            local_files_only=True,
        )

        logger.debug(f"[reranker] Model loaded: {_MODEL_ID} (4-bit quantization) from {reranker_weights_dir()}")
        return True

    except Exception as e:
        logger.error("[reranker] Failed to load model: %s: %s", type(e).__name__, str(e))
        _model = None
        return False


def rerank_fused_entries(
    query: str,
    fused: list[FusedRankEntry],
    chunk_content_by_id: dict[str, str],
    top_n: int | None = None,
    rank_offset: int = 0,
) -> tuple[list[RerankedEntry], str]:
    """Rerank fused entries using cross-encoder model.

    For each entry, constructs the passage from chunk_content_by_id (full content,
    falling back to excerpt if content missing), truncates to max sequence length,
    and scores via the model's .rerank() method.

    Args:
        query: Search query
        fused: List of fused rank entries to rerank
        chunk_content_by_id: dict[chunk_id -> full_content_text]
        top_n: If specified, only return top_n results
        rank_offset: Added to each entry's position-derived fused_rank. Callers
            batching across multiple calls (see rrf_fusion._rerank_in_batches)
            must pass the batch's starting offset in the original fused list,
            otherwise fused_rank would restart from 1 in every batch.

    Returns:
        Tuple of (reranked_entries, status_code):
        - status_code: 'ok', 'no_gpu', or 'model_load_failed'
        - reranked_entries: sorted by rerank_score descending (or empty if status != 'ok')
    """
    if not fused:
        return [], "ok"

    if not _check_gpu_available():
        return [], "no_gpu"

    if not load_reranker():
        return [], "model_load_failed"

    # Prepare passages: resolve full content from chunk_content_by_id, falling back to excerpt
    passages = []

    for i, entry in enumerate(fused):
        # Try to get full content; fallback to excerpt (200-char)
        content = chunk_content_by_id.get(entry.chunk_id)
        passage = content if content else entry.excerpt

        # Truncate to max sequence length, keeping the start (most identifying content)
        if len(passage) > _MAX_SEQ_LENGTH:
            truncated = passage[:_MAX_SEQ_LENGTH]
            logger.debug(
                "[reranker] Truncated passage for chunk_id=%s from %d to %d chars",
                entry.chunk_id,
                len(passage),
                len(truncated),
            )
            passage = truncated

        passages.append(passage)

    total_chars = sum(len(p) for p in passages)
    logger.debug(
        "[reranker] Reranking %d candidates (~%d chars, ~%d tokens est.) for query: %r",
        len(passages),
        total_chars,
        total_chars // 4,
        query[:80],
    )
    t0 = time.monotonic()

    # Call the model's .rerank() method
    # Returns list[dict] with keys: 'document', 'relevance_score', 'index', 'embedding'
    try:
        scores = _model.rerank(query, passages, top_n=top_n)
    except Exception as e:
        elapsed = time.monotonic() - t0
        logger.error(
            "[reranker] Rerank call failed after %.1fs: %s: %s", elapsed, type(e).__name__, str(e)
        )
        return [], "model_load_failed"

    logger.debug("[reranker] Reranked %d candidates in %.1fs", len(passages), time.monotonic() - t0)

    # Build RerankedEntry list, preserving original fused metadata
    reranked = []
    for score_obj in scores:
        # score_obj is a dict with 'index' and 'relevance_score' keys
        idx = int(score_obj.get("index", -1))
        rerank_score = float(score_obj.get("relevance_score", 0.0))

        if idx < 0 or idx >= len(fused):
            logger.warning("[reranker] Invalid score index %d for fused list length %d", idx, len(fused))
            continue

        original_entry = fused[idx]

        reranked.append(
            RerankedEntry(
                chunk_id=original_entry.chunk_id,
                rel_path=original_entry.rel_path,
                fused_score=original_entry.fused_score,
                rerank_score=rerank_score,
                fused_rank=idx + 1 + rank_offset,  # 1-indexed position in the original (pre-batching) fused list
                excerpt=original_entry.excerpt,
                token_estimate=original_entry.token_estimate,
            )
        )

    # Sort by rerank_score descending (EXPLICIT FINAL SORT per claim #2)
    reranked.sort(key=lambda e: e.rerank_score, reverse=True)

    return reranked, "ok"


_GPU_RERANKER_ENABLED_KEY = "gpu_reranker_enabled"


async def is_gpu_reranker_enabled() -> bool:
    """Global on/off switch (app_metadata flag, mirrors api/consent.py's pattern),
    read by retrieve_rrf_fusion() to decide whether to run the rerank stage at all.

    Re-checks GPU availability and weights presence defensively — a stale 'true'
    flag carried over from a different machine/session (or after weights were
    deleted) must never cause a crash, only a graceful no-op."""
    available, _ = detect_gpu()
    if not available or not reranker_weights_ready():
        return False

    from infrastructure.db.database import get_db

    db = get_db()
    async with db.execute(
        "SELECT value FROM app_metadata WHERE key = ?", (_GPU_RERANKER_ENABLED_KEY,)
    ) as cur:
        row = await cur.fetchone()
    return row is not None and row["value"] == "true"
