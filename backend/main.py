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
from api.consent import router as consent_router
from api.job import router as job_router
from api.local_repo import router as local_repo_router
from api.provider import router as provider_router
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
from contextlib import asynccontextmanager
from typing import AsyncGenerator


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    await init_db()
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
    app.include_router(workspace_router, prefix="/api/workspace")
    app.include_router(provider_router, prefix="/api/provider")
    app.include_router(consent_router, prefix="/api/consent")
    app.include_router(local_repo_router, prefix="/api/local-repo")
    app.include_router(sync_router, prefix="/api/sync")
    app.include_router(job_router, prefix="/api/job")

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
