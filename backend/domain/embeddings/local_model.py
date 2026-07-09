"""Local GPU embedding model — lazy singleton wrapping Qwen3-Embedding-0.6B via
sentence-transformers.

Mirrors domain/retrieval/cross_encoder_rerank.py's pattern exactly:
  - Lazy-loaded singleton, VRAM-gated, never imports torch/sentence-transformers
    at module load time (optional [embeddings] extra in pyproject.toml).
  - Global on/off toggle in app_metadata (_LOCAL_EMBEDDING_ENABLED_KEY).
  - detect_gpu() / release_gpu_cache() from domain.shared.gpu.
"""
from __future__ import annotations

from shared.logger import logger
from domain.shared.gpu import detect_gpu, release_gpu_cache

# app_metadata key for the on/off toggle (mirrors _GPU_RERANKER_ENABLED_KEY)
_LOCAL_EMBEDDING_ENABLED_KEY = "local_embedding_enabled"

# Minimum VRAM required (inherited from shared GPU detection threshold)
from domain.shared.gpu import GPU_MIN_VRAM_GB
_MIN_VRAM_GB = GPU_MIN_VRAM_GB

# Model to load — Apache 2.0 licensed, strong MTEB multilingual performance
_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"

# Lazy singleton
_embedder = None


def get_embedder():
    """Return the loaded SentenceTransformer singleton, or None if unavailable.

    First call triggers a model load (may take several seconds). Subsequent
    calls return immediately from the cached singleton.
    Never raises — returns None on any load failure.
    """
    global _embedder
    if _embedder is not None:
        return _embedder
    gpu_ok, _ = detect_gpu()
    if not gpu_ok:
        logger.warning("local_embedding: no usable GPU detected — cannot load model")
        return None
    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        logger.info(f"local_embedding: loading {_MODEL_ID}")
        _embedder = SentenceTransformer(_MODEL_ID, device="cuda")
        logger.info("local_embedding: model loaded")
    except Exception as exc:
        logger.warning(f"local_embedding: model load failed: {exc}")
        _embedder = None
    return _embedder


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings and return float vectors.

    Raises RuntimeError if no GPU or model load failed — callers must gate
    on get_embedder() / local_embedding_available() first.
    """
    model = get_embedder()
    if model is None:
        raise RuntimeError(
            "Local embedding model is unavailable — no GPU or model load failed"
        )
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return [v.tolist() for v in vectors]


async def local_embedding_available() -> bool:
    """True if the local model is both GPU-usable AND enabled via the Settings toggle."""
    gpu_ok, _ = detect_gpu()
    if not gpu_ok:
        return False

    from infrastructure.db.database import get_db

    db = get_db()
    async with db.execute(
        "SELECT value FROM app_metadata WHERE key = ?", (_LOCAL_EMBEDDING_ENABLED_KEY,)
    ) as cur:
        row = await cur.fetchone()
    return row is not None and row["value"] == "true"
