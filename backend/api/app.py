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
    module: str


class DiagnosticsResponse(BaseModel):
    python_version: str
    native_module_loaded: bool
    native_functions: list[NativeFunction]


# (function_name, module_path, description)
_NATIVE_REGISTRY: list[tuple[str, str, str]] = [
    # _native_graph module
    ("compute_scores",     "domain.structural_graph._native_graph", "Graph centrality scoring (in-degree * 3 + out-degree sort)"),
    ("expand_neighbors",   "domain.structural_graph._native_graph", "BFS graph neighborhood expansion for graph viewer"),
    ("compute_scc",        "domain.structural_graph._native_graph", "Tarjan SCC — circular import cycle detection"),
    ("compute_louvain",    "domain.structural_graph._native_graph", "Louvain community detection"),
    ("scan_keywords_bulk", "domain.structural_graph._native_graph", "Bulk word-boundary keyword scan (TODO/FIXME hotspots)"),
    ("rank_and_budget",    "domain.structural_graph._native_graph", "Sort + token-budget truncation for retrieval re-ranking"),
    # _native_bm25 module
    ("tokenize",           "domain.retrieval._native_bm25", "BM25 text tokenizer ([A-Za-z0-9_]+ regex)"),
    ("batch_score",        "domain.retrieval._native_bm25", "Batch BM25 scoring for retrieval chunks"),
    ("batch_impact_score", "domain.retrieval._native_bm25", "Batch impact scoring (hop + centrality + community + symbol)"),
]


@router.get("/diagnostics", response_model=DiagnosticsResponse)
async def get_diagnostics() -> DiagnosticsResponse:
    # Cache loaded modules to avoid redundant imports
    _loaded_modules: dict[str, object | None] = {}

    def _get_mod(module_path: str) -> object | None:
        if module_path not in _loaded_modules:
            try:
                _loaded_modules[module_path] = importlib.import_module(module_path)
            except Exception:
                _loaded_modules[module_path] = None
        return _loaded_modules[module_path]

    funcs = []
    for name, module_path, desc in _NATIVE_REGISTRY:
        mod = _get_mod(module_path)
        funcs.append(NativeFunction(
            name=name,
            available=mod is not None and hasattr(mod, name),
            description=desc,
            module=module_path.rsplit(".", 1)[-1],  # e.g. "_native_graph"
        ))

    any_loaded = any(f.available for f in funcs)
    return DiagnosticsResponse(
        python_version=sys.version.split()[0],
        native_module_loaded=any_loaded,
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
