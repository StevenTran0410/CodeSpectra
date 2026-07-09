"""External-agent-facing API (narrow slice, for AEH's CodeSpectraProxyClient). Not a full implementation: no external_call_log table, no generalized multi-endpoint auth framework — just a bearer-token-gated LLM passthrough so an external harness (starting with AEH) can reuse whatever provider the user already configured here, without AEH ever holding a provider API key of its own."""
from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel

from domain.model_connector.service import ProviderConfigService
from domain.model_connector.types import ChatMessage, ChatRequest, ChatResponse
from infrastructure.db.database import get_db

from domain.retrieval.service import RetrievalService
from domain.retrieval.types import RrfFusionRequest, RrfFusionBundle
from domain.structural_graph.service import StructuralGraphService
from domain.structural_graph.types import GraphNeighborsResponse, GraphCommunitiesResponse, FileSymbolEdgesResponse
from domain.repo_map.service import RepoMapService
from domain.repo_map.types import SymbolsResponse
from domain.manifest.service import ManifestService
from domain.manifest.types import ManifestFileContentResponse
from domain.analysis.service import AnalysisService
from domain.analysis.types import AnalysisReportSummary, AnalysisReport
from domain.sync_engine.service import SyncEngineService
from domain.sync_engine.types import RepoSnapshot
from domain.local_repo.service import LocalRepoService
from domain.local_repo.types import LocalRepo

router = APIRouter(tags=["external"])
_service = ProviderConfigService()
_retrieval_service = RetrievalService()
_graph_service = StructuralGraphService()
_repo_map_service = RepoMapService()
_manifest_service = ManifestService()
_analysis_service = AnalysisService()
_sync_service = SyncEngineService()
_local_repo_service = LocalRepoService()


class LLMCompleteRequest(BaseModel):
    provider_id: str
    model_id: str | None = None
    messages: list[ChatMessage]
    max_completion_tokens: int = 2048
    temperature: float | None = 0.2
    json_mode: bool = False


class ProviderSummary(BaseModel):
    provider_id: str
    display_name: str
    model_id: str


async def _get_external_token() -> str | None:
    import os

    env_token = os.getenv("CODESPECTRA_EXTERNAL_TOKEN")
    if env_token:
        return env_token
    db = get_db()
    async with db.execute(
        "SELECT value FROM app_metadata WHERE key='external_api_token'"
    ) as cur:
        row = await cur.fetchone()
    return row["value"] if row else None


async def require_external_token(authorization: str = Header(default="")) -> None:
    expected = await _get_external_token()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="external API token not configured — set CODESPECTRA_EXTERNAL_TOKEN "
            "or an 'external_api_token' app_metadata row",
        )
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")


@router.post(
    "/llm/complete",
    response_model=ChatResponse,
    dependencies=[Depends(require_external_token)],
)
async def llm_complete(body: LLMCompleteRequest) -> ChatResponse:
    return await _service.chat(
        ChatRequest(
            provider_id=body.provider_id,
            model_id=body.model_id,
            messages=body.messages,
            max_completion_tokens=body.max_completion_tokens,
            temperature=body.temperature,
            json_mode=body.json_mode,
            stream=False,
        )
    )


@router.get(
    "/llm/providers",
    response_model=list[ProviderSummary],
    dependencies=[Depends(require_external_token)],
)
async def list_llm_providers() -> list[ProviderSummary]:
    configs = await _service.list_all()
    return [
        ProviderSummary(provider_id=c.id, display_name=c.display_name, model_id=c.model_id)
        for c in configs
    ]


@router.post(
    "/retrieval/search",
    response_model=RrfFusionBundle,
    dependencies=[Depends(require_external_token)],
)
async def search_retrieval(body: RrfFusionRequest) -> RrfFusionBundle:
    """Wraps retrieve_rrf_fusion (debug/comparison path), not the plain budget-capped retrieve() — deliberately, for AEH's discovery fingerprinting, which needs "does this term appear anywhere in the repo" over a bare keyword query. retrieve()'s section-budget cap can silently drop a chunk containing an exact match below a differently-scored chunk for the same query (confirmed empirically against this repo's own agent_pipeline.py — a real haystack import). retrieve_rrf_fusion's fused list is unbounded and BM25-weighted, reliably surfacing exact-term hits."""
    return await _retrieval_service.retrieve_rrf_fusion(body)


@router.get(
    "/graph/{snapshot_id}/neighbors",
    response_model=GraphNeighborsResponse,
    dependencies=[Depends(require_external_token)],
)
async def get_graph_neighbors(
    snapshot_id: str,
    seed_path: str,
    hops: int = 1,
    limit: int = 300,
) -> GraphNeighborsResponse:
    return await _graph_service.neighbors(snapshot_id, seed_path, hops, limit)


@router.get(
    "/graph/{snapshot_id}/communities",
    response_model=GraphCommunitiesResponse,
    dependencies=[Depends(require_external_token)],
)
async def get_graph_communities(snapshot_id: str) -> GraphCommunitiesResponse:
    return await _graph_service.list_communities(snapshot_id)


@router.get(
    "/graph/{snapshot_id}/symbol-edges",
    response_model=FileSymbolEdgesResponse,
    dependencies=[Depends(require_external_token)],
)
async def get_graph_symbol_edges(
    snapshot_id: str,
    file_path: str,
) -> FileSymbolEdgesResponse:
    return await _graph_service.symbol_edges_for_file(snapshot_id, file_path)


@router.get(
    "/repo-map/{snapshot_id}/search",
    response_model=SymbolsResponse,
    dependencies=[Depends(require_external_token)],
)
async def search_repo_map_symbols(
    snapshot_id: str,
    q: str,
    limit: int = 120,
) -> SymbolsResponse:
    return await _repo_map_service.search(snapshot_id, q, limit)


@router.get(
    "/manifest/{snapshot_id}/file",
    response_model=ManifestFileContentResponse,
    dependencies=[Depends(require_external_token)],
)
async def read_manifest_file(
    snapshot_id: str,
    rel_path: str,
    max_bytes: int = 200_000,
) -> ManifestFileContentResponse:
    return await _manifest_service.read_file(snapshot_id, rel_path, max_bytes)


@router.get(
    "/analysis/reports",
    response_model=list[AnalysisReportSummary],
    dependencies=[Depends(require_external_token)],
)
async def list_analysis_reports(
    repo_id: str | None = None,
    workspace_id: str | None = None,
    limit: int = 30,
) -> list[AnalysisReportSummary]:
    return await _analysis_service.list_reports(repo_id, workspace_id, limit)


@router.get(
    "/analysis/reports/{report_id}",
    response_model=AnalysisReport,
    dependencies=[Depends(require_external_token)],
)
async def get_analysis_report(report_id: str) -> AnalysisReport:
    return await _analysis_service.get_report(report_id)


@router.get(
    "/snapshots/{snapshot_id}",
    response_model=RepoSnapshot,
    dependencies=[Depends(require_external_token)],
)
async def get_repo_snapshot(snapshot_id: str) -> RepoSnapshot:
    return await _sync_service.get_snapshot(snapshot_id)


@router.get(
    "/repos/{repo_id}",
    response_model=LocalRepo,
    dependencies=[Depends(require_external_token)],
)
async def get_local_repo(repo_id: str) -> LocalRepo:
    return await _local_repo_service.get_by_id(repo_id)


@router.get(
    "/repos",
    response_model=list[LocalRepo],
    dependencies=[Depends(require_external_token)],
)
async def list_local_repos(
    workspace_id: str | None = None,
    mode: str | None = None,
) -> list[LocalRepo]:
    return await _local_repo_service.list_all(workspace_id, mode)

