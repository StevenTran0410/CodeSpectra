"""Local GPU embedding model — lazy singleton wrapping Qwen3-Embedding-0.6B via
sentence-transformers.

Mirrors domain/retrieval/cross_encoder_rerank.py's pattern exactly:
  - Lazy-loaded singleton, VRAM-gated, never imports torch/sentence-transformers
    at module load time (optional [embeddings] extra in pyproject.toml).
  - Global on/off toggle in app_metadata (_LOCAL_EMBEDDING_ENABLED_KEY).
  - detect_gpu() from domain.shared.gpu.
  - Weights only arrive via an explicit user-triggered download (Settings
    "Download" button, see domain.shared.model_weights.start_download) or by
    being placed manually in weights_dir(_MODEL_KEY); every load call passes
    local_files_only=True, so this module itself never reaches the network.
"""
from __future__ import annotations

from shared.logger import logger
from domain.shared.gpu import detect_gpu
from domain.shared.model_weights import weights_dir, weights_ready

# app_metadata key for the on/off toggle (mirrors _GPU_RERANKER_ENABLED_KEY)
_LOCAL_EMBEDDING_ENABLED_KEY = "local_embedding_enabled"

# Minimum VRAM required (inherited from shared GPU detection threshold)
from domain.shared.gpu import GPU_MIN_VRAM_GB
_MIN_VRAM_GB = GPU_MIN_VRAM_GB

# Model to load — Apache 2.0 licensed, strong MTEB multilingual performance
_MODEL_KEY = "qwen3-embedding-0.6b"
_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
_HF_MODEL_URL = "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B"

# Lazy singleton
_embedder = None


def embedding_weights_ready() -> bool:
    return weights_ready(_MODEL_KEY)


def embedding_weights_dir() -> str:
    return str(weights_dir(_MODEL_KEY))


def get_embedder():
    """Return the loaded SentenceTransformer singleton, or None if unavailable.

    First call triggers a model load (may take several seconds). Subsequent
    calls return immediately from the cached singleton.
    Never raises — returns None on any load failure, including missing weights.
    """
    global _embedder
    if _embedder is not None:
        return _embedder
    gpu_ok, _ = detect_gpu()
    if not gpu_ok:
        logger.warning("local_embedding: no usable GPU detected — cannot load model")
        return None
    if not embedding_weights_ready():
        logger.warning(
            f"local_embedding: weights not found at {embedding_weights_dir()} — "
            f"download from {_HF_MODEL_URL} and place the files there"
        )
        return None
    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        logger.info(f"local_embedding: loading {_MODEL_ID} from {embedding_weights_dir()}")
        _embedder = SentenceTransformer(
            embedding_weights_dir(), device="cuda", local_files_only=True
        )
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
    """True if the local model is GPU-usable, has weights on disk, AND is
    enabled via the Settings toggle."""
    gpu_ok, _ = detect_gpu()
    if not gpu_ok or not embedding_weights_ready():
        return False

    from infrastructure.db.database import get_db

    db = get_db()
    async with db.execute(
        "SELECT value FROM app_metadata WHERE key = ?", (_LOCAL_EMBEDDING_ENABLED_KEY,)
    ) as cur:
        row = await cur.fetchone()
    return row is not None and row["value"] == "true"
