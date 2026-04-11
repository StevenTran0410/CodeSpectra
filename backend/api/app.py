import importlib
import importlib.metadata
import sys

from fastapi import APIRouter
from pydantic import BaseModel

from infrastructure.db.database import get_db

router = APIRouter(tags=["app"])


class VersionResponse(BaseModel):
    status: str
    version: str


class GitConfig(BaseModel):
    ssh_key_path: str | None = None


@router.get("/health", response_model=VersionResponse)
async def health() -> VersionResponse:
    try:
        version = importlib.metadata.version("codespectra-backend")
    except importlib.metadata.PackageNotFoundError:
        version = "dev"
    return VersionResponse(status="ok", version=version)


@router.get("/git-config", response_model=GitConfig)
async def get_git_config() -> GitConfig:
    db = get_db()
    async with db.execute(
        "SELECT value FROM app_metadata WHERE key='git_ssh_key_path'"
    ) as cur:
        row = await cur.fetchone()
    return GitConfig(ssh_key_path=row["value"] if row else None)


class NativeFunction(BaseModel):
    name: str
    available: bool
    description: str


class DiagnosticsResponse(BaseModel):
    python_version: str
    native_module_loaded: bool
    native_functions: list[NativeFunction]


@router.get("/diagnostics", response_model=DiagnosticsResponse)
async def get_diagnostics() -> DiagnosticsResponse:
    _NATIVE_FUNCTIONS = [
        ("compute_scores",     "Graph centrality scoring (in-degree × 3 + out-degree sort)"),
        ("expand_neighbors",   "BFS graph neighborhood expansion for graph viewer"),
        ("compute_scc",        "Tarjan SCC — circular import cycle detection"),
        ("compute_louvain",    "Louvain community detection (falls back to Python if absent)"),
        ("scan_keywords_bulk", "Bulk word-boundary keyword scan (TODO/FIXME hotspots)"),
    ]
    try:
        mod = importlib.import_module("domain.structural_graph._native_graph")
        loaded = True
    except Exception:
        mod = None
        loaded = False

    funcs = [
        NativeFunction(
            name=name,
            available=loaded and hasattr(mod, name),
            description=desc,
        )
        for name, desc in _NATIVE_FUNCTIONS
    ]
    return DiagnosticsResponse(
        python_version=sys.version.split()[0],
        native_module_loaded=loaded,
        native_functions=funcs,
    )


@router.put("/git-config", response_model=GitConfig)
async def set_git_config(body: GitConfig) -> GitConfig:
    db = get_db()
    if body.ssh_key_path:
        await db.execute(
            "INSERT OR REPLACE INTO app_metadata (key, value) VALUES ('git_ssh_key_path', ?)",
            (body.ssh_key_path,),
        )
    else:
        await db.execute("DELETE FROM app_metadata WHERE key='git_ssh_key_path'")
    await db.commit()
    return body
