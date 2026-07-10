"""
CodeSpectra analysis engine — FastAPI/uvicorn entry point.

Electron spawns this process and reads stdout for the BACKEND_READY signal.
Usage: python main.py --port PORT
"""
import argparse
import socket
import sys

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.app import router as app_router
from api.analysis import router as analysis_router
from api.consent import router as consent_router
from api.external import router as external_router
from api.gpu_reranker import router as gpu_reranker_router
from api.local_embedding import router as local_embedding_router
from api.impact import router as impact_router
from api.job import router as job_router
from api.local_repo import router as local_repo_router
from api.manifest import router as manifest_router
from api.provider import router as provider_router
from api.qa import router as qa_router
from api.retrieval import router as retrieval_router
from api.repo_map import router as repo_map_router
from api.structural_graph import router as structural_graph_router
from api.sync import router as sync_router
from api.workspace import router as workspace_router
from domain.model_connector.errors import ProviderError, ProviderErrorCode
from infrastructure.db.database import close_db, init_db
from shared.logger import logger

# HTTP status codes for each ProviderErrorCode
_PROVIDER_ERROR_STATUS: dict[ProviderErrorCode, int] = {
    ProviderErrorCode.CONNECTION_REFUSED: 503,
    ProviderErrorCode.TIMEOUT: 503,
    ProviderErrorCode.AUTH_FAILED: 401,
    ProviderErrorCode.MODEL_NOT_FOUND: 404,
    ProviderErrorCode.CONTEXT_LIMIT_EXCEEDED: 422,
    ProviderErrorCode.RATE_LIMITED: 429,
    ProviderErrorCode.UNKNOWN: 502,
}


# ──────────────────────────────────────────────────────────────────────────────
# Application factory
# ──────────────────────────────────────────────────────────────────────────────
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from domain.qa.classifier_service import ClassifierService as _ClassifierService


async def _warm_up_local_gpu_models() -> None:
    """Eagerly load the GPU reranker and/or local embedding model if the user
    already has them enabled (GPU present, weights on disk) — avoids the first
    real request after startup paying the multi-second model-load cost."""
    try:
        from domain.retrieval.cross_encoder_rerank import is_gpu_reranker_enabled, load_reranker

        if await is_gpu_reranker_enabled():
            await asyncio.to_thread(load_reranker)
    except Exception as e:
        logger.warning(f"GPU reranker warm-up failed: {e}")

    try:
        from domain.embeddings.local_model import get_embedder, local_embedding_available

        if await local_embedding_available():
            await asyncio.to_thread(get_embedder)
    except Exception as e:
        logger.warning(f"Local embedding model warm-up failed: {e}")


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    await init_db()
    # Warm up the intent classifier in background — loads DB examples + trains sklearn model.
    # Non-blocking: a slow startup does not delay BACKEND_READY.
    asyncio.get_event_loop().create_task(
        _ClassifierService().warm_up_with_db_examples(),
        name="classifier-warmup",
    )
    asyncio.get_event_loop().create_task(
        _warm_up_local_gpu_models(),
        name="local-gpu-models-warmup",
    )
    logger.info("Backend startup complete")
    yield
    await close_db()
    logger.info("Backend shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="CodeSpectra Backend",
        version="0.1.0",
        docs_url="/docs",  # only reachable from localhost
        openapi_url="/openapi.json",
        lifespan=_lifespan,
    )

    # Only allow requests from the Electron renderer (localhost/127.0.0.1)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:*", "http://127.0.0.1:*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(app_router, prefix="/api/app")
    app.include_router(impact_router, prefix="/api/impact")
    app.include_router(workspace_router, prefix="/api/workspace")
    app.include_router(provider_router, prefix="/api/provider")
    app.include_router(consent_router, prefix="/api/consent")
    app.include_router(gpu_reranker_router, prefix="/api/gpu-reranker")
    app.include_router(local_embedding_router, prefix="/api/local-embedding")
    app.include_router(local_repo_router, prefix="/api/local-repo")
    app.include_router(sync_router, prefix="/api/sync")
    app.include_router(manifest_router, prefix="/api/manifest")
    app.include_router(repo_map_router, prefix="/api/repo-map")
    app.include_router(structural_graph_router, prefix="/api/graph")
    app.include_router(retrieval_router, prefix="/api/retrieval")
    app.include_router(analysis_router, prefix="/api/analysis")
    app.include_router(qa_router, prefix="/api/qa")
    app.include_router(job_router, prefix="/api/job")
    app.include_router(external_router, prefix="/api/external")

    @app.exception_handler(ProviderError)
    async def provider_error_handler(_req: Request, exc: ProviderError) -> JSONResponse:
        status = _PROVIDER_ERROR_STATUS.get(exc.code, 502)
        return JSONResponse(
            status_code=status,
            content={"error": exc.code.value, "message": exc.message, "retryable": exc.retryable},
        )

    return app


# ──────────────────────────────────────────────────────────────────────────────
# Custom uvicorn server that prints the BACKEND_READY signal after binding
# ──────────────────────────────────────────────────────────────────────────────
class _ReadySignalServer(uvicorn.Server):
    def __init__(self, config: uvicorn.Config, port: int) -> None:
        super().__init__(config)
        self._port = port

    async def startup(self, sockets=None) -> None:  # type: ignore[override]
        await super().startup(sockets=sockets)
        # Print after the socket is bound — Electron reads this line
        print(f"BACKEND_READY:{self._port}", flush=True)
        logger.info(f"Listening on 127.0.0.1:{self._port}")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="CodeSpectra backend server")
    parser.add_argument("--port", type=int, default=0, help="Port to bind (0 = auto)")
    args = parser.parse_args()

    port = args.port if args.port > 0 else _find_free_port()

    app = create_app()
    config = uvicorn.Config(
        app=app,
        host="127.0.0.1",
        port=port,
        log_level="warning",  # uvicorn's own logs suppressed; we use our logger
        access_log=False,
    )
    server = _ReadySignalServer(config, port)

    try:
        import asyncio  # noqa: PLC0415

        asyncio.run(server.serve())
    except KeyboardInterrupt:
        logger.info("Backend shutting down (KeyboardInterrupt)")
        sys.exit(0)


if __name__ == "__main__":
    main()
