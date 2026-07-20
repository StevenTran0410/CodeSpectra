"""FastAPI UI Server for AEH per-agent drill-down dashboard."""
from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Protocol, TypeVar

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent_eval_harness.code_injection.injection_target import BranchInjectionTarget, InjectionTargetError
from agent_eval_harness.code_injection.plan_renderer import (
    PRESERVED_PLAN_FILES,
    render_eval_plan_files,
)
from agent_eval_harness.code_injection.wiring import build_wiring
from agent_eval_harness.config import AEHConfig
from agent_eval_harness.datasets.fulfillment import export_dataset
from agent_eval_harness.discovery.analysis_context import load_project_context
from agent_eval_harness.discovery.client import CodeSpectraClient
from agent_eval_harness.discovery.engine import run_discovery_background
from agent_eval_harness.discovery.expansion import expand_candidate
from agent_eval_harness.judging.case_judge import judge_case_semantic_match, summarize_agent_judgments
from agent_eval_harness.llm.client import LLMResponse
from agent_eval_harness.llm.fake_client import FakeLLMClient
from agent_eval_harness.llm.proxy_client import CodeSpectraProxyClient
from agent_eval_harness.mapping.agent_flow import (
    build_source_by_component,
    load_agent_flow_map,
    save_agent_flow_map,
    separate_agent_flows,
)
from agent_eval_harness.mapping.builder.pipeline import SystemMapBuilder
from agent_eval_harness.mapping.system_map import load_system_map, save_system_map, SystemMap
from agent_eval_harness.metrics.suite import load_suite
from agent_eval_harness.planning.agentic_planner import generate_plan_agentic
from agent_eval_harness.planning.report import load_plan_report, save_plan_report
from agent_eval_harness.store import repository
from agent_eval_harness.store.database import DB_FILENAME, close_db, get_db, init_db

logger = logging.getLogger("agent_eval_harness.ui.server")

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.getenv("AEH_DATA_DIR"):
        os.environ["AEH_DATA_DIR"] = os.getcwd()
    await init_db()
    await repository.cancel_orphaned_running_sessions()
    yield
    await close_db()


app = FastAPI(
    title="AEH Local Evaluation Dashboard",
    description="Browse, compare, and debug agent evaluation runs.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunListItem(BaseModel):
    id: str
    target_system_id: str
    eval_plan_id: str | None
    started_at: str
    finished_at: str | None
    status: str
    map_path: str | None
    active_defects: list[str]
    pass_rate: float
    judge_cost: int


class ComponentAggregate(BaseModel):
    total: int
    passed: int


class RunDetailResponse(BaseModel):
    id: str
    target_system_id: str
    eval_plan_id: str | None
    started_at: str
    finished_at: str | None
    status: str
    map_path: str | None
    active_defects: list[str]
    system_map: dict[str, Any]
    component_aggregates: dict[str, ComponentAggregate]
    overall_pass_rate: float
    target: str | None = None
    suite_path: str | None = None
    parent_run_id: str | None = None
    model_overrides: dict[str, str] = {}


class EvaluationDetailItem(BaseModel):
    id: str
    metric_name: str
    metric_class: str
    score: float | None
    passed: bool | None
    details: dict[str, Any]
    evaluator: str | None
    cost_tokens: int | None
    trace_id: str | None
    span_id: str | None
    root_input: str | None
    final_output: str | None
    trace_tokens: int | None
    trace_latency: int | None


class TraceDetailResponse(BaseModel):
    trace: dict[str, Any] | None
    spans: list[dict[str, Any]]


_F = TypeVar("_F", bound=Callable[..., Any])


def _log_and_500(action: str) -> Callable[[_F], _F]:
    """Route decorator: log an unexpected exception and convert it to HTTP 500; HTTPException passes through unchanged."""
    def decorator(func: _F) -> _F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"{action}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e)) from e
        return wrapper  # type: ignore[return-value]
    return decorator


async def _get_run_or_404(run_id: str) -> dict:
    run = await repository.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.get("/api/runs", response_model=list[RunListItem])
@_log_and_500("Failed to list runs")
async def get_runs(target_system_id: str | None = None):
    db_runs = await repository.list_runs(target_system_id)
    results = []
    for r in db_runs:
        defects = []
        if r.get("active_defects"):
            try:
                defects = json.loads(r["active_defects"])
            except Exception as e:
                logger.warning(f"Could not parse active_defects for run {r['id']}: {e}")
        results.append(
            RunListItem(
                id=r["id"],
                target_system_id=r["target_system_id"],
                eval_plan_id=r["eval_plan_id"],
                started_at=r["started_at"],
                finished_at=r["finished_at"],
                status=r["status"],
                map_path=r["map_path"],
                active_defects=defects,
                pass_rate=r["pass_rate"],
                judge_cost=r["judge_cost"],
            )
        )
    return results


@app.get("/api/runs/{run_id}", response_model=RunDetailResponse)
async def get_run_detail(run_id: str):
    run = await _get_run_or_404(run_id)

    system_map_dict = {
        "target_system_id": run["target_system_id"],
        "discrepancies": [],
        "components": [],
    }
    if run.get("map_path"):
        try:
            m_path = Path(run["map_path"])
            if m_path.exists():
                sys_map = load_system_map(str(m_path))
                system_map_dict = sys_map.model_dump()
        except Exception as e:
            logger.warning(f"Could not load system map from {run['map_path']}: {e}")

    db = get_db()
    async with db.execute(
        "SELECT component_id, COUNT(*) as total, "
        "SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END) as passed "
        "FROM evaluations WHERE run_id = ? AND component_id IS NOT NULL "
        "GROUP BY component_id",
        (run_id,),
    ) as cur:
        rows = await cur.fetchall()

    aggregates = {
        row["component_id"]: ComponentAggregate(
            total=row["total"], passed=row["passed"]
        )
        for row in rows
    }

    defects = []
    if run.get("active_defects"):
        try:
            defects = json.loads(run["active_defects"])
        except Exception as e:
            logger.warning(f"Could not parse active_defects for run {run_id}: {e}")

    async with db.execute(
        "SELECT COALESCE("
        "CAST(SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END) AS REAL) / "
        "NULLIF(COUNT(CASE WHEN passed IS NOT NULL THEN 1 END), 0), "
        "0.0) as pass_rate "
        "FROM evaluations WHERE run_id = ?",
        (run_id,),
    ) as cur:
        rate_row = await cur.fetchone()
    overall_pass_rate = rate_row["pass_rate"] if rate_row else 0.0

    model_overrides = {}
    if run.get("model_overrides"):
        try:
            model_overrides = json.loads(run["model_overrides"])
        except Exception as e:
            logger.warning(f"Could not parse model_overrides for run {run_id}: {e}")

    return RunDetailResponse(
        id=run["id"],
        target_system_id=run["target_system_id"],
        eval_plan_id=run["eval_plan_id"],
        started_at=run["started_at"],
        finished_at=run["finished_at"],
        status=run["status"],
        map_path=run["map_path"],
        active_defects=defects,
        system_map=system_map_dict,
        component_aggregates=aggregates,
        overall_pass_rate=overall_pass_rate,
        target=run.get("target"),
        suite_path=run.get("suite_path"),
        parent_run_id=run.get("parent_run_id"),
        model_overrides=model_overrides,
    )


@app.get(
    "/api/runs/{run_id}/components/{component_id}",
    response_model=list[EvaluationDetailItem],
)
async def get_component_evaluations(run_id: str, component_id: str):
    db = get_db()
    query = """
        SELECT
            e.id, e.metric_name, e.metric_class, e.score, e.passed, e.details_json,
            e.evaluator, e.cost_tokens, e.trace_id, e.span_id,
            t.root_input, t.final_output,
            t.total_tokens as trace_tokens,
            t.total_latency_ms as trace_latency
        FROM evaluations e
        LEFT JOIN traces t ON e.trace_id = t.id
        WHERE e.run_id = ? AND e.component_id = ?
        ORDER BY e.rowid
    """
    async with db.execute(query, (run_id, component_id)) as cur:
        rows = await cur.fetchall()

    results = []
    for r in rows:
        details = {}
        if r["details_json"]:
            try:
                details = json.loads(r["details_json"])
            except Exception as e:
                logger.warning(f"Could not parse details_json for evaluation {r['id']}: {e}")
        results.append(
            EvaluationDetailItem(
                id=r["id"],
                metric_name=r["metric_name"],
                metric_class=r["metric_class"],
                score=r["score"],
                passed=r["passed"] if r["passed"] is None else bool(r["passed"]),
                details=details,
                evaluator=r["evaluator"],
                cost_tokens=r["cost_tokens"],
                trace_id=r["trace_id"],
                span_id=r["span_id"],
                root_input=r["root_input"],
                final_output=r["final_output"],
                trace_tokens=r["trace_tokens"],
                trace_latency=r["trace_latency"],
            )
        )
    return results


@app.get("/api/traces/{trace_id}", response_model=TraceDetailResponse)
async def get_trace_detail(trace_id: str):
    db = get_db()
    async with db.execute("SELECT * FROM traces WHERE id = ?", (trace_id,)) as cur:
        row = await cur.fetchone()
    trace_dict = dict(row) if row else None

    spans = await repository.get_span_tree(trace_id)
    return TraceDetailResponse(trace=trace_dict, spans=spans)


@app.get("/api/datasets/{dataset_id}/cases")
async def get_dataset_cases(dataset_id: str):
    cases = await repository.get_dataset_cases(dataset_id)
    return cases


@app.get("/api/datasets")
async def list_datasets_route():
    return await repository.list_dataset_ids()


class CaseVerdictRequest(BaseModel):
    verdict: str  # accept | edit | reject
    input_json: dict | None = None
    expected_json: dict | None = None
    labels_json: dict | None = None


@app.post("/api/datasets/cases/{case_id}/verdict")
async def case_verdict_route(case_id: str, body: CaseVerdictRequest):
    case = await repository.get_dataset_case_by_id(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found.")

    if body.verdict == "reject":
        await repository.delete_dataset_case(case_id)
        remaining = await repository.get_dataset_cases(case["dataset_id"])
        metadata = await repository.get_dataset_metadata(case["dataset_id"])
        min_cases = metadata["min_cases"] if metadata else 1
        return {"success": True, "remaining": len(remaining), "shortfall": max(0, min_cases - len(remaining))}

    if body.verdict in ("accept", "edit"):
        metadata = await repository.get_dataset_metadata(case["dataset_id"])
        if metadata and metadata["kind"] == "snapshot_regression_baseline":
            labels = json.loads(case["labels_json"]) if case["labels_json"] else {}
            if not (labels.get("schema_valid_passed") and labels.get("fallback_sentinel_passed")):
                raise HTTPException(
                    status_code=400,
                    detail="schema_valid and fallback_sentinel must both pass before approving a baseline case.",
                )
        await repository.update_case_provenance(
            case_id,
            "generated+reviewed",
            expected_json=json.dumps(body.expected_json) if body.expected_json is not None else None,
            input_json=json.dumps(body.input_json) if body.input_json is not None else None,
            labels_json=json.dumps(body.labels_json) if body.labels_json is not None else None,
        )
        return {"success": True}

    raise HTTPException(status_code=400, detail=f"Unknown verdict: {body.verdict}")


@app.get("/api/metamorphic-relations")
async def list_metamorphic_relations_route():
    from agent_eval_harness.datasets.metamorphic_relations import load_relations

    return [r.model_dump() for r in load_relations().values()]


class MetamorphicPreviewRequest(BaseModel):
    relation_id: str
    sample_size: int = 5


class MetamorphicApproveRequest(BaseModel):
    relation_id: str
    samples: list[dict] | None = None


class MetamorphicDeriveRequest(BaseModel):
    relation_id: str
    dataset_name: str | None = None
    spot_audit_pct: float = 0.10


@app.post("/api/datasets/{dataset_id}/metamorphic/preview")
async def metamorphic_preview_route(dataset_id: str, body: MetamorphicPreviewRequest):
    from agent_eval_harness.datasets.metamorphic_derive import (
        MetamorphicPreconditionError,
        preview_relation,
    )

    try:
        return await preview_relation(body.relation_id, dataset_id, sample_size=body.sample_size)
    except MetamorphicPreconditionError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/datasets/{dataset_id}/metamorphic/approve")
async def metamorphic_approve_route(dataset_id: str, body: MetamorphicApproveRequest):
    from agent_eval_harness.datasets.metamorphic_derive import (
        MetamorphicPreconditionError,
        approve_relation,
    )

    try:
        await approve_relation(body.relation_id, dataset_id, samples=body.samples)
        return {"success": True}
    except MetamorphicPreconditionError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/datasets/{dataset_id}/metamorphic/derive")
async def metamorphic_derive_route(dataset_id: str, body: MetamorphicDeriveRequest):
    from agent_eval_harness.datasets.metamorphic_derive import (
        MetamorphicPreconditionError,
        derive_dataset,
    )

    try:
        return await derive_dataset(
            body.relation_id,
            dataset_id,
            dataset_name=body.dataset_name,
            spot_audit_pct=body.spot_audit_pct,
        )
    except MetamorphicPreconditionError as e:
        raise HTTPException(status_code=400, detail=str(e))


class ProviderSummary(BaseModel):
    provider_id: str
    display_name: str
    model_id: str | None = None
    kind: str = ""


@app.get("/api/providers", response_model=list[ProviderSummary])
async def get_providers(backend_url: str | None = None, backend_token: str | None = None):
    import httpx

    config = AEHConfig.load()
    backend_url = backend_url or config.backend_url
    backend_token = backend_token or config.backend_token
    if not backend_url or not backend_token:
        return []

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{backend_url.rstrip('/')}/api/external/llm/providers",
                headers={"Authorization": f"Bearer {backend_token}"},
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.warning(f"Failed to fetch providers from CodeSpectra backend: {e}")
        return []


class RerunRequest(BaseModel):
    model_overrides: dict[str, str] = {}  # {component_id: model_id}
    active_defects: list[str] | None = None


_rerun_in_flight: set[str] = set()


@app.post("/api/runs/{run_id}/rerun")
async def rerun_run(run_id: str, body: RerunRequest):
    from agent_eval_harness.llm.routing_client import RoutingLLMClient
    from agent_eval_harness.metrics.sweep import run_sweep

    parent = await _get_run_or_404(run_id)

    if not parent.get("target") or not parent.get("suite_path"):
        raise HTTPException(
            status_code=409,
            detail=(
                "This run predates rerun support (missing target/suite_path) "
                "— cannot re-execute"
            ),
        )

    if run_id in _rerun_in_flight:
        raise HTTPException(
            status_code=409,
            detail="A rerun is already in progress for this run",
        )

    _rerun_in_flight.add(run_id)

    active_defects = body.active_defects
    if active_defects is None:
        try:
            active_defects = (
                json.loads(parent["active_defects"]) if parent.get("active_defects") else []
            )
        except Exception as e:
            logger.warning(f"Could not parse parent active_defects for rerun of {run_id}: {e}")
            active_defects = []

    new_run_id = await repository.insert_run(
        target_system_id=parent["target_system_id"],
        eval_plan_id=parent["eval_plan_id"],
        map_path=parent["map_path"],
        active_defects=active_defects,
        target=parent["target"],
        suite_path=parent["suite_path"],
        parent_run_id=run_id,
        model_overrides=body.model_overrides,
    )

    async def _execute_rerun_task():
        try:
            config = AEHConfig.load()

            if config.provider_id and config.backend_url and config.backend_token:
                default_client = CodeSpectraProxyClient(
                    config.backend_url,
                    config.backend_token,
                    config.provider_id,
                    config.model_id,
                    reasoning_effort=config.reasoning_effort,
                    thinking_budget=config.thinking_budget,
                )
            else:
                default_client = FakeLLMClient(
                    LLMResponse(
                        content="This is a fallback offline demo answer.", model="fake-default"
                    )
                )

            overrides = {}
            system_map = None
            if parent.get("map_path"):
                try:
                    system_map = load_system_map(parent["map_path"])
                except Exception as e:
                    logger.warning(f"Could not load system map for rerun: {e}")

            for comp_id, override_val in body.model_overrides.items():
                if not override_val:
                    continue
                if ":" in override_val:
                    p_id, m_id = override_val.split(":", 1)
                else:
                    p_id = config.provider_id or "fake-provider"
                    m_id = override_val

                if config.backend_url and config.backend_token and p_id != "fake-provider":
                    client = CodeSpectraProxyClient(
                        config.backend_url,
                        config.backend_token,
                        provider_id=p_id,
                        model_id=m_id,
                        reasoning_effort=config.reasoning_effort,
                        thinking_budget=config.thinking_budget,
                    )
                else:
                    client = FakeLLMClient(
                        LLMResponse(content=f"Canned override answer for {comp_id}", model=m_id)
                    )

                # Route BOTH comp_id (for Tier-2 contextvar) and component_name (for Tier-1 Haystack tags)
                overrides[comp_id] = client
                if system_map:
                    comp = system_map.component_by_id(comp_id)
                    if comp:
                        for sm in comp.span_match:
                            if sm.component_name:
                                overrides[sm.component_name] = client

            routing_client = RoutingLLMClient(default=default_client, overrides=overrides)

            try:
                # new_run_id reuses the row inserted above instead of creating a second one.
                await run_sweep(
                    target=parent["target"],
                    map_path=parent["map_path"],
                    suite_path=parent["suite_path"],
                    llm_client=routing_client,
                    active_defects=active_defects,
                    run_id=new_run_id,
                )
            finally:
                await routing_client.aclose()
        except Exception as e:
            logger.error(f"Background rerun failed: {e}", exc_info=True)
            try:
                await repository.finish_run(new_run_id, "failed")
            except Exception as finish_err:
                logger.error(f"Also failed to mark run {new_run_id} as failed: {finish_err}")
        finally:
            _rerun_in_flight.discard(run_id)

    asyncio.create_task(_execute_rerun_task())
    return {"run_id": new_run_id}


class IngestEvalRunRequest(BaseModel):
    manifest_path: str


@app.post("/api/eval-runs/ingest")
async def ingest_eval_run_route(body: IngestEvalRunRequest):
    """Stage 4's offline half: reads a manifest + spanlog an injected target produced and persists it as an 'ingested' run."""
    from agent_eval_harness.ingest.spanlog_ingest import IngestError, parse_spanlog, persist_spanlog

    manifest_path = Path(body.manifest_path)
    if not manifest_path.exists():
        raise HTTPException(status_code=400, detail=f"manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"could not read manifest: {e}")

    log_path_str = manifest.get("log_path")
    if not log_path_str or not Path(log_path_str).exists():
        raise HTTPException(status_code=400, detail="manifest is missing a valid log_path")

    try:
        parsed = parse_spanlog(Path(log_path_str))
    except IngestError as e:
        raise HTTPException(status_code=422, detail=str(e))

    run_id = await persist_spanlog(
        parsed,
        target_system_id=manifest.get("plan_id", "unknown"),
        eval_plan_id=manifest.get("plan_id"),
    )
    run = await repository.get_run(run_id)
    return {"run_id": run_id, "status": run["status"] if run else "unknown"}



class StartDiscoveryRequest(BaseModel):
    repo_ref: str
    snapshot_id: str
    provider_id: str | None = None
    model_id: str | None = None
    backend_url: str | None = None
    backend_token: str | None = None
    reasoning_effort: str | None = None
    thinking_budget: int | None = None


class DiscoverySessionResponse(BaseModel):
    id: str
    repo_ref: str
    snapshot_id: str
    status: str
    error: str | None
    created_at: str
    finished_at: str | None
    pause_info: dict | None = None
    pipeline_stage: str


class DiscoveryCandidateResponse(BaseModel):
    id: str
    session_id: str
    name: str
    frameworks: list[str]
    entry_points: list[str]
    evidence: list[dict]
    confidence: str
    needs_human: bool = False
    verdict: str
    community_id: str | None = None
    cluster_files: list[str] = []
    hub_paths: list[str] = []
    wiring_block: dict | None = None
    excluded_files: list[str] = []
    matched_files: list[str] = []
    file_provenance: dict[str, str] = {}


class UpdateVerdictRequest(BaseModel):
    verdict: str


class UpdateExcludedFilesRequest(BaseModel):
    excluded_files: list[str]


async def _get_discovery_session_or_404(session_id: str) -> dict:
    session = await repository.get_discovery_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Discovery session not found")
    return session


async def _get_expansion_session_or_404(session_id: str) -> dict:
    sess = await repository.get_expansion_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Expansion session not found.")
    return sess


def _get_repo_root() -> Path:
    """Repo root is parent of the target source folder; loaded from AEHConfig."""
    config = AEHConfig.load()
    if not config.backend_source_path:
        raise HTTPException(
            status_code=400,
            detail="backend_source_path not configured — set it in .aeh/config.yaml",
        )
    backend_path = Path(config.backend_source_path)
    if not backend_path.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"backend_source_path does not exist or is not a directory: {backend_path}",
        )
    repo_root = backend_path.parent
    if not (repo_root / ".git").exists():
        raise HTTPException(
            status_code=400,
            detail=f"parent of backend_source_path is not a git repo root: {repo_root}",
        )
    return repo_root


class _LLMConfigRequest(Protocol):
    """Structural type for any request body carrying the six provider/backend override fields."""
    provider_id: str | None
    model_id: str | None
    backend_url: str | None
    backend_token: str | None
    reasoning_effort: str | None
    thinking_budget: int | None


def _resolve_synthesis_config(
    body: _LLMConfigRequest, config: AEHConfig
) -> tuple[str, str, str, str | None, str | None, int | None]:
    """Resolve provider/backend config for an LLM-calling endpoint from request overrides + AEHConfig, or raise 400."""
    provider_id = body.provider_id or config.provider_id
    backend_url = body.backend_url or config.backend_url
    backend_token = body.backend_token or config.backend_token

    if not backend_url or not backend_token:
        raise HTTPException(
            status_code=400,
            detail="Missing CodeSpectra backend connection config (url/token)."
        )
    if not provider_id:
        raise HTTPException(
            status_code=400,
            detail="No LLM provider configured. Set up a provider in Settings, then select it.",
        )
    return (
        provider_id,
        backend_url,
        backend_token,
        body.model_id or config.model_id,
        body.reasoning_effort or config.reasoning_effort,
        body.thinking_budget if body.thinking_budget is not None else config.thinking_budget,
    )


def _build_synthesis_clients(
    body: _LLMConfigRequest, config: AEHConfig
) -> tuple[CodeSpectraProxyClient, CodeSpectraClient]:
    """Resolve synthesis config and build the paired LLM proxy + CodeSpectra API clients most routes need."""
    provider_id, backend_url, backend_token, model_id, reasoning_effort, thinking_budget = (
        _resolve_synthesis_config(body, config)
    )
    llm_client = CodeSpectraProxyClient(
        backend_url, backend_token, provider_id, model_id,
        reasoning_effort=reasoning_effort, thinking_budget=thinking_budget,
    )
    client = CodeSpectraClient(backend_url, backend_token)
    return llm_client, client


@app.post("/api/discovery/sessions")
@_log_and_500("Failed to start discovery session")
async def start_discovery(body: StartDiscoveryRequest):
    config = AEHConfig.load()
    llm_client, client = _build_synthesis_clients(body, config)

    session_id = await repository.insert_discovery_session(body.repo_ref, body.snapshot_id)

    asyncio.create_task(
        run_discovery_background(session_id, body.snapshot_id, body.repo_ref, client, llm_client)
    )

    return {"session_id": session_id}


@app.get("/api/discovery/sessions", response_model=list[DiscoverySessionResponse])
@_log_and_500("Failed to list discovery sessions")
async def list_discovery_sessions(repo_ref: str | None = None, snapshot_id: str | None = None):
    sessions = await repository.list_discovery_sessions(repo_ref, snapshot_id)
    return [
        DiscoverySessionResponse(
            id=s["id"],
            repo_ref=s["repo_ref"],
            snapshot_id=s["snapshot_id"],
            status=s["status"],
            error=s["error"],
            created_at=s["created_at"],
            finished_at=s["finished_at"],
            pause_info=s.get("pause_info"),
            pipeline_stage=s["pipeline_stage"],
        )
        for s in sessions
    ]


@app.get("/api/discovery/sessions/{session_id}", response_model=DiscoverySessionResponse)
async def get_discovery_session(session_id: str):
    session = await _get_discovery_session_or_404(session_id)
    return DiscoverySessionResponse(
        id=session["id"],
        repo_ref=session["repo_ref"],
        snapshot_id=session["snapshot_id"],
        status=session["status"],
        error=session["error"],
        created_at=session["created_at"],
        finished_at=session["finished_at"],
        pause_info=session.get("pause_info"),
        pipeline_stage=session["pipeline_stage"],
    )


@app.get("/api/discovery/sessions/{session_id}/candidates", response_model=list[DiscoveryCandidateResponse])
@_log_and_500("Failed to list discovery candidates")
async def list_discovery_candidates(session_id: str):
    candidates = await repository.get_discovery_candidates(session_id)
    return [
        DiscoveryCandidateResponse(
            id=c["id"],
            session_id=c["session_id"],
            name=c["name"],
            frameworks=c["frameworks"],
            entry_points=c["entry_points"],
            evidence=c["evidence"],
            confidence=c["confidence"],
            needs_human=c.get("needs_human", False),
            verdict=c["verdict"],
            community_id=c.get("community_id"),
            cluster_files=c.get("cluster_files", []),
            hub_paths=c.get("hub_paths", []),
            wiring_block=c.get("wiring_block"),
            excluded_files=c.get("excluded_files", []),
            matched_files=c.get("matched_files", []),
            file_provenance=c.get("file_provenance", {}),
        )
        for c in candidates
    ]


@app.post("/api/discovery/candidates/{candidate_id}/verdict")
@_log_and_500("Failed to update candidate verdict")
async def update_candidate_verdict(candidate_id: str, body: UpdateVerdictRequest):
    if body.verdict not in ["proposed", "confirmed", "rejected"]:
        raise HTTPException(status_code=400, detail="Invalid verdict")
    await repository.update_candidate_verdict(candidate_id, body.verdict)
    return {"success": True}


@app.post("/api/discovery/candidates/{candidate_id}/excluded-files")
@_log_and_500("Failed to update excluded files")
async def update_candidate_excluded_files_route(candidate_id: str, body: UpdateExcludedFilesRequest):
    await repository.update_candidate_excluded_files(candidate_id, body.excluded_files)
    return {"success": True}


class ResumeDiscoveryRequest(BaseModel):
    provider_id: str | None = None
    model_id: str | None = None
    backend_url: str | None = None
    backend_token: str | None = None
    reasoning_effort: str | None = None
    thinking_budget: int | None = None


@app.post("/api/discovery/sessions/{session_id}/resume")
async def resume_discovery(session_id: str, body: ResumeDiscoveryRequest):
    session = await _get_discovery_session_or_404(session_id)
    if session["status"] != "paused_rate_limit":
        raise HTTPException(
            status_code=400, detail=f"Session is not paused (status={session['status']})"
        )

    config = AEHConfig.load()
    backend_url = body.backend_url or config.backend_url
    backend_token = body.backend_token or config.backend_token
    pause_info = session.get("pause_info") or {}
    provider_id = body.provider_id or pause_info.get("provider_id") or config.provider_id
    model_id = body.model_id if body.model_id is not None else pause_info.get("model_id")
    reasoning_effort = (
        body.reasoning_effort or pause_info.get("reasoning_effort") or config.reasoning_effort
    )
    thinking_budget = (
        body.thinking_budget
        if body.thinking_budget is not None
        else pause_info.get("thinking_budget", config.thinking_budget)
    )

    llm_client = CodeSpectraProxyClient(
        backend_url, backend_token, provider_id, model_id,
        reasoning_effort=reasoning_effort, thinking_budget=thinking_budget,
    )
    client = CodeSpectraClient(backend_url, backend_token)

    existing = await repository.get_discovery_candidates(session_id)
    already_named = {c["community_id"]: c for c in existing if c.get("community_id")}

    await repository.resume_discovery_session(session_id)
    asyncio.create_task(
        run_discovery_background(
            session_id, session["snapshot_id"], session["repo_ref"], client, llm_client,
            already_named=already_named,
        )
    )
    return {"success": True}


@app.get("/health")
async def health_check():
    return {"status": "ok"}



class ExpandCandidateRequest(BaseModel):
    provider_id: str | None = None
    model_id: str | None = None
    backend_url: str | None = None
    backend_token: str | None = None
    reasoning_effort: str | None = None
    thinking_budget: int | None = None
    node_budget: int = 100
    hop_cap: int = 3
    classify_provider_id: str | None = None
    classify_model_id: str | None = None
    classify_reasoning_effort: str | None = None
    classify_thinking_budget: int | None = None


async def run_expansion_background(
    session_id: str,
    snapshot_id: str,
    candidate: dict,
    client: CodeSpectraClient,
    llm_client: CodeSpectraProxyClient,
    classify_llm_client: CodeSpectraProxyClient,
    node_budget: int,
    hop_cap: int,
):
    try:
        res = await expand_candidate(
            snapshot_id,
            candidate,
            client,
            classify_llm_client,
            node_budget=node_budget,
            hop_cap=hop_cap,
        )

        # Log scanner summary so an empty map is diagnosable
        _reasons = res.get("boundary_reasons", [])
        from collections import Counter
        top_skip = [r for r, _ in Counter(r for r in _reasons if r).most_common(3)]
        logger.info(
            "Scanner summary: name=%s accepted=%d boundary=%d top_skip_reasons=%s",
            candidate["name"],
            len(res["accepted"]),
            len(res["boundary"]),
            top_skip,
        )

        snapshot = await client.get_snapshot(snapshot_id)
        local_path_str = snapshot.get("local_path")
        if not local_path_str:
            raise RuntimeError(f"Snapshot {snapshot_id} is missing local_path context.")

        local_path = Path(local_path_str)
        # accepted entries are dicts with file, role_hint, key_symbols, follow fields
        abs_files = [local_path / (p["file"] if isinstance(p, dict) else p) for p in res["accepted"]]

        builder = SystemMapBuilder(llm_client)
        system_map, _summary = await builder.build_from_files(
            abs_files,
            package_root=local_path,
            target_system_id=candidate["name"],
        )

        data_dir = os.getenv("AEH_DATA_DIR", ".")
        map_dir = Path(data_dir) / "maps"
        map_dir.mkdir(parents=True, exist_ok=True)
        map_path = map_dir / f"{session_id}.yaml"
        save_system_map(system_map, map_path)

        await repository.finish_expansion_session(
            session_id,
            "completed",
            map_path=str(map_path),
            accepted=res["accepted"],
            boundary=res["boundary"],
            stop_reason=res["stop_reason"],
            accepted_edges=res.get("accepted_edges", []),
        )
    except Exception as e:
        logger.error(f"Expansion runner failed: {e}", exc_info=True)
        await repository.finish_expansion_session(session_id, "failed", error=str(e))
    finally:
        # both must be closed or this leaks a connection pool per run
        await client.aclose()
        if hasattr(llm_client, "aclose"):
            await llm_client.aclose()
        if classify_llm_client is not llm_client and hasattr(classify_llm_client, "aclose"):
            await classify_llm_client.aclose()


@app.post("/api/discovery/candidates/{candidate_id}/expand")
@_log_and_500("Failed to start expansion session")
async def start_expansion(candidate_id: str, body: ExpandCandidateRequest):
    candidate = await repository.get_discovery_candidate(candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate component not found.")

    # discovery_candidates has no snapshot_id column — resolve it via the parent session
    session = await repository.get_discovery_session(candidate["session_id"])
    if not session:
        raise HTTPException(status_code=404, detail="Parent discovery session not found.")
    snapshot_id = session["snapshot_id"]

    config = AEHConfig.load()
    provider_id, backend_url, backend_token, model_id, reasoning_effort, thinking_budget = (
        _resolve_synthesis_config(body, config)
    )
    llm_client = CodeSpectraProxyClient(
        backend_url, backend_token, provider_id, model_id,
        reasoning_effort=reasoning_effort, thinking_budget=thinking_budget,
    )

    classify_provider_id = body.classify_provider_id or provider_id
    classify_model_id = body.classify_model_id or model_id
    classify_reasoning_effort = body.classify_reasoning_effort or reasoning_effort
    classify_thinking_budget = (
        body.classify_thinking_budget
        if body.classify_thinking_budget is not None
        else thinking_budget
    )
    if (
        classify_provider_id == provider_id
        and classify_model_id == model_id
        and classify_reasoning_effort == reasoning_effort
        and classify_thinking_budget == thinking_budget
    ):
        classify_llm_client = llm_client
    else:
        classify_llm_client = CodeSpectraProxyClient(
            backend_url, backend_token, classify_provider_id, classify_model_id,
            reasoning_effort=classify_reasoning_effort, thinking_budget=classify_thinking_budget,
        )

    client = CodeSpectraClient(backend_url, backend_token)

    session_id = repository.new_id()
    await repository.insert_expansion_session(session_id, candidate_id, snapshot_id)

    asyncio.create_task(
        run_expansion_background(
            session_id,
            snapshot_id,
            candidate,
            client,
            llm_client,
            classify_llm_client,
            body.node_budget,
            body.hop_cap,
        )
    )

    return {"session_id": session_id}


@app.get("/api/discovery/expansion-sessions/{session_id}")
async def get_expansion_session(session_id: str):
    return await _get_expansion_session_or_404(session_id)


@app.get("/api/discovery/candidates/{candidate_id}/expansion-sessions")
async def list_expansion_sessions(candidate_id: str):
    return await repository.list_expansion_sessions_for_candidate(candidate_id)


@app.get("/api/discovery/expansion-sessions/{session_id}/map")
async def get_expansion_map(session_id: str):
    sess = await _get_expansion_session_or_404(session_id)
    if sess["status"] != "completed":
        raise HTTPException(status_code=400, detail="Expansion session has not completed.")
    if not sess["map_path"]:
        raise HTTPException(status_code=400, detail="Expansion session does not have map output.")

    try:
        system_map = load_system_map(sess["map_path"])
        return system_map.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load map file: {e}")


@app.put("/api/discovery/expansion-sessions/{session_id}/map")
async def update_expansion_map(session_id: str, body: SystemMap):
    sess = await _get_expansion_session_or_404(session_id)
    try:
        save_system_map(body, sess["map_path"])
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update map file: {e}")


class GeneratePlanRequest(BaseModel):
    provider_id: str | None = None
    model_id: str | None = None
    backend_url: str | None = None
    backend_token: str | None = None
    reasoning_effort: str | None = None
    thinking_budget: int | None = None


class JudgeAgentCasesRequest(BaseModel):
    agent_id: str
    # Re-judges cases already scored, replacing their evaluations — needed whenever the scoring rules change, since stored scores are otherwise frozen.
    force: bool = False
    provider_id: str | None = None
    model_id: str | None = None
    backend_url: str | None = None
    backend_token: str | None = None
    reasoning_effort: str | None = None
    thinking_budget: int | None = None


class UpdatePlanRequest(BaseModel):
    entries: list[dict]


class AdvanceSessionRequest(BaseModel):
    confirmed_candidates: list[str] | None = None      # for awaiting_candidate_review
    confirmed_map_session_id: str | None = None        # for awaiting_map_review
    confirmed_plan: bool | None = None                 # for awaiting_plan_review
    provider_id: str | None = None
    model_id: str | None = None
    backend_url: str | None = None
    backend_token: str | None = None


class EnrichAgentsRequest(BaseModel):
    provider_id: str | None = None
    model_id: str | None = None
    backend_url: str | None = None
    backend_token: str | None = None
    reasoning_effort: str | None = None
    thinking_budget: int | None = None
    depth: str = 'normal'
    agent_ids: list[str] | None = None
    force_agent_ids: list[str] | None = None


@app.post("/api/discovery/expansion-sessions/{session_id}/plan")
async def generate_plan_route(session_id: str, body: GeneratePlanRequest):
    """Stage 3 plans per agent — hard-gated on Stage 2's agent-flow separation, no flat-component fallback."""
    sess = await _get_expansion_session_or_404(session_id)
    if sess["status"] != "completed":
        raise HTTPException(status_code=400, detail="Expansion has not completed.")
    if not sess["map_path"]:
        raise HTTPException(status_code=400, detail="No system map for this session.")
    agent_flows_path_str = sess.get("agent_flows_path")
    if not agent_flows_path_str or not Path(agent_flows_path_str).exists():
        raise HTTPException(
            status_code=400,
            detail="Run Stage 2 agent-flow separation first — Stage 3 plans per agent.",
        )
    if not await repository.list_agent_knowledge(session_id):
        raise HTTPException(
            status_code=400,
            detail="Run Stage 2.5 enrichment first — component roles come from enrichment.",
        )

    config = AEHConfig.load()
    llm_client, client = _build_synthesis_clients(body, config)

    try:
        import yaml

        system_map = load_system_map(sess["map_path"])
        agent_flow_map = load_agent_flow_map(agent_flows_path_str)

        snapshot = await client.get_snapshot(sess["snapshot_id"])
        local_path_str = snapshot.get("local_path")
        if not local_path_str:
            raise HTTPException(status_code=400, detail="Snapshot is missing local_path context.")
        local_path = Path(local_path_str)
        # accepted entries are dicts with file, role_hint, key_symbols, follow fields
        abs_files = [local_path / (p["file"] if isinstance(p, dict) else p) for p in sess["accepted"]]
        source_by_component = build_source_by_component(abs_files, system_map)

        harvest_files = abs_files + [local_path / p for p in sess.get("boundary", [])]

        map_path = Path(sess["map_path"])
        plan_path = map_path.with_name(map_path.stem + "_plan.yaml")
        plan_report_path = map_path.with_name(map_path.stem + "_plan_report.yaml")
        # Regenerating the plan deletes the previous plan's datasets — they'd go stale otherwise.
        if plan_path.exists():
            try:
                old_suite = load_suite(plan_path)
                for ref in {e.dataset.ref for e in old_suite.entries if e.dataset and e.dataset.ref}:
                    await repository.delete_dataset(ref)
            except Exception as e:
                logger.warning(f"generate_plan: could not delete previous plan's datasets: {e}")
        previous_report = None
        if plan_report_path.exists():
            try:
                previous_report = load_plan_report(plan_report_path)
            except Exception as e:
                logger.warning(f"generate_plan: could not load previous plan report for incremental reuse: {e}")

        # Load project context once, shared across all analyst prompts
        _plan_project_ctx = None
        try:
            _plan_project_ctx = await load_project_context(client, sess["snapshot_id"])
        except Exception as _pce:
            logger.warning(f"generate_plan: could not load project context: {_pce}")

        suite, plan_report = await generate_plan_agentic(
            system_map, agent_flow_map, source_by_component, sess["accepted_edges"], llm_client,
            files=harvest_files, files_root=local_path,
            previous_report=previous_report,
            project_context=_plan_project_ctx,
            conventions=config.contract_conventions,
        )

        plan_path.write_text(
            yaml.dump(suite.model_dump(), allow_unicode=True),
            encoding="utf-8"
        )
        save_plan_report(plan_report, plan_report_path)

        await repository.update_expansion_session_plan_path(session_id, str(plan_path))
        await repository.update_expansion_session_plan_report_path(session_id, str(plan_report_path))

        candidate = await repository.get_discovery_candidate(sess["candidate_id"])
        if candidate:
            await repository.update_discovery_session_pipeline_stage(candidate["session_id"], "awaiting_plan_review")

        from agent_eval_harness.planning.validation import validate_plan
        report = await validate_plan(plan_path)
        result = suite.model_dump()
        result["readiness"] = {eid: r.model_dump() for eid, r in report.readiness.items()}
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"generate_plan failed: {e}", exc_info=True)
        try:
            candidate = await repository.get_discovery_candidate(sess["candidate_id"])
            if candidate:
                await repository.update_discovery_session_pipeline_stage(
                    candidate["session_id"], "awaiting_map_review"
                )
        except Exception as reset_err:
            logger.warning(f"could not reset discovery pipeline_stage after plan failure: {reset_err}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await client.aclose()
        if hasattr(llm_client, "aclose"):
            await llm_client.aclose()


@app.get("/api/discovery/expansion-sessions/{session_id}/plan-report")
async def get_plan_report_route(session_id: str):
    sess = await _get_expansion_session_or_404(session_id)
    plan_report_path_str = sess.get("plan_report_path")
    if not plan_report_path_str or not Path(plan_report_path_str).exists():
        raise HTTPException(status_code=404, detail="No plan report for this session.")
    try:
        return load_plan_report(plan_report_path_str).model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load plan report: {e}")


@app.put("/api/discovery/expansion-sessions/{session_id}/plan-report")
async def update_plan_report_route(session_id: str, body: dict):
    """Round-trips the per-agent AgentPlanReport (unlike PUT .../plan, which round-trips the Suite gate list) — what Stage3Screen's per-agent eval_enabled toggle writes to."""
    sess = await _get_expansion_session_or_404(session_id)
    plan_report_path_str = sess.get("plan_report_path")
    if not plan_report_path_str or not Path(plan_report_path_str).exists():
        raise HTTPException(status_code=400, detail="No plan report to update.")

    from agent_eval_harness.planning.report import EvaluationPlanReport

    try:
        new_report = EvaluationPlanReport.model_validate(body)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid plan report: {e}")

    save_plan_report(new_report, plan_report_path_str)
    return {"success": True}


class EvalRunRequest(BaseModel):
    """provider_id/model_id drive the agent-under-test's call; judge_provider_id/judge_model_id optionally override the judge."""
    provider_id: str | None = None
    model_id: str | None = None
    judge_provider_id: str | None = None
    judge_model_id: str | None = None
    backend_url: str | None = None
    backend_token: str | None = None


@app.post("/api/discovery/expansion-sessions/{session_id}/eval-run")
async def run_eval_route(session_id: str, body: EvalRunRequest):
    """Runs the synthetic_agent_io suite for every eval_enabled agent, one real LLM call per case, scored by a separate cross-family judge provider where possible."""
    sess = await _get_expansion_session_or_404(session_id)
    plan_report_path_str = sess.get("plan_report_path")
    plan_path_str = sess.get("plan_path")
    if not plan_report_path_str or not Path(plan_report_path_str).exists():
        raise HTTPException(status_code=400, detail="No plan report for this session.")
    if not plan_path_str or not Path(plan_path_str).exists():
        raise HTTPException(status_code=400, detail="No plan for this session.")

    config = AEHConfig.load()
    provider_id = body.provider_id or config.provider_id
    backend_url = body.backend_url or config.backend_url
    backend_token = body.backend_token or config.backend_token
    if not backend_url or not backend_token:
        raise HTTPException(status_code=400, detail="Missing backend connection config.")
    if not provider_id:
        raise HTTPException(status_code=400, detail="No LLM provider configured.")
    model_id = body.model_id or config.model_id

    from agent_eval_harness.injection.judge_selection import pick_judge_provider
    from agent_eval_harness.injection.suite_runner import run_synthetic_agent_io_suite

    plan_report = load_plan_report(plan_report_path_str)
    suite = load_suite(plan_path_str)
    system_map = load_system_map(sess["map_path"]) if sess.get("map_path") else None
    target_system_id = system_map.target_system_id if system_map else session_id

    enabled_agents = [a for a in plan_report.agents if a.eval_enabled and a.contract is not None]
    if not enabled_agents:
        return {"agents": {}, "note": "no agent has eval_enabled=True"}

    providers_raw = await get_providers(backend_url, backend_token)
    available = [ProviderSummary.model_validate(p) for p in providers_raw]
    generator_summary = next(
        (p for p in available if p.provider_id == provider_id),
        ProviderSummary(provider_id=provider_id, display_name="", model_id=model_id, kind=""),
    )
    judge_pid = body.judge_provider_id
    judge_mid = body.judge_model_id
    if not judge_pid and available:
        judge = pick_judge_provider(generator_summary, available)
        if judge is not None:
            judge_pid = judge.provider_id
            judge_mid = judge.model_id or model_id
    if not judge_pid:
        judge_pid, judge_mid = provider_id, model_id

    agent_llm_client = CodeSpectraProxyClient(backend_url, backend_token, provider_id, model_id)
    judge_llm_client = CodeSpectraProxyClient(backend_url, backend_token, judge_pid, judge_mid or model_id)

    results: dict[str, dict] = {}
    try:
        for agent in enabled_agents:
            entry = next(
                (
                    e for e in suite.entries
                    if e.agent_id == agent.agent_id
                    and e.dataset
                    and e.dataset.required
                    and e.dataset.required.get("kind") == "synthetic_agent_io"
                    and e.dataset.ref
                ),
                None,
            )
            if entry is None:
                results[agent.agent_id] = {
                    "status": "needs_human",
                    "reason": "no fulfilled synthetic_agent_io dataset for this agent",
                }
                continue
            dataset_cases = await repository.get_dataset_cases(entry.dataset.ref)
            if not dataset_cases:
                results[agent.agent_id] = {"status": "needs_human", "reason": "dataset has zero cases"}
                continue

            from agent_eval_harness.datasets.generators.synthetic_agent_io import is_dataset_stale

            current_schema = (agent.contract.output.json_schema if agent.contract.output else None)
            stale = is_dataset_stale(dataset_cases, current_schema)

            run_result = await run_synthetic_agent_io_suite(
                agent.agent_id, target_system_id, agent.contract, dataset_cases,
                provider_id, model_id, agent_llm_client, judge_llm_client,
            )
            results[agent.agent_id] = {
                "status": "completed",
                "run_id": run_result.run_id,
                "metric_count": len(run_result.results),
                "error_count": len(run_result.errors),
                # True if the dataset's gold predates the agent's current contract schema — never blocks scoring, just a UI hint.
                "stale": stale,
            }
    finally:
        if hasattr(agent_llm_client, "aclose"):
            await agent_llm_client.aclose()
        if hasattr(judge_llm_client, "aclose"):
            await judge_llm_client.aclose()

    return {"agents": results}


@app.delete("/api/discovery/expansion-sessions/{session_id}/stage3")
async def reset_stage3_route(session_id: str):
    """Deletes every dataset fulfilled from this session's plan, the plan/plan-report files and their DB pointers, then resets the session to 'awaiting_map_review'. Stage 2 artifacts are untouched."""
    sess = await _get_expansion_session_or_404(session_id)

    deleted_dataset_ids: list[str] = []
    plan_path_str = sess.get("plan_path")
    if plan_path_str and Path(plan_path_str).exists():
        try:
            suite = load_suite(plan_path_str)
            dataset_ids = {e.dataset.ref for e in suite.entries if e.dataset and e.dataset.ref}
            for dataset_id in dataset_ids:
                await repository.delete_dataset(dataset_id)
                deleted_dataset_ids.append(dataset_id)
        except Exception as e:
            logger.warning(f"reset_stage3: failed to read/delete datasets from plan: {e}")

    for path_str in (sess.get("plan_path"), sess.get("plan_report_path")):
        if not path_str:
            continue
        try:
            Path(path_str).unlink(missing_ok=True)
        except OSError as e:
            logger.warning(f"reset_stage3: failed to delete {path_str}: {e}")

    await repository.update_expansion_session_plan_path(session_id, None)
    await repository.update_expansion_session_plan_report_path(session_id, None)

    candidate = await repository.get_discovery_candidate(sess["candidate_id"])
    if candidate:
        await repository.update_discovery_session_pipeline_stage(
            candidate["session_id"], "awaiting_map_review"
        )

    return {"success": True, "deleted_dataset_ids": deleted_dataset_ids}


@app.get("/api/discovery/expansion-sessions/{session_id}/plan")
async def get_plan_route(session_id: str):
    sess = await _get_expansion_session_or_404(session_id)
    plan_path_str = sess.get("plan_path")
    if not plan_path_str or not Path(plan_path_str).exists():
        raise HTTPException(status_code=404, detail="No plan file for this session.")
    try:
        from agent_eval_harness.planning.validation import validate_plan
        suite = load_suite(plan_path_str)
        report = await validate_plan(plan_path_str)
        result = suite.model_dump()
        result["readiness"] = {eid: r.model_dump() for eid, r in report.readiness.items()}
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load plan: {e}")


class FulfillDatasetsRequest(BaseModel):
    provider_id: str | None = None
    model_id: str | None = None
    backend_url: str | None = None
    backend_token: str | None = None
    reasoning_effort: str | None = None
    thinking_budget: int | None = None
    instructions: dict[str, dict] | None = None
    embedding_provider_id: str | None = None
    embedding_model_id: str | None = None
    use_local_embedding: bool = False
    # Scopes fulfillment to just these agents' groups; None fulfills every unfulfilled group.
    agent_ids: list[str] | None = None
    # Deletes and re-fulfills these agents' synthetic_agent_io dataset even if already fulfilled.
    force_agent_ids: list[str] | None = None


@app.post("/api/discovery/expansion-sessions/{session_id}/datasets/fulfill")
@_log_and_500("fulfill_datasets failed")
async def fulfill_datasets_route(session_id: str, body: FulfillDatasetsRequest):
    sess = await _get_expansion_session_or_404(session_id)
    if not sess.get("plan_path") or not Path(sess["plan_path"]).exists():
        raise HTTPException(status_code=400, detail="No plan for this session — generate a plan first.")

    config = AEHConfig.load()
    provider_id, backend_url, backend_token, model_id, reasoning_effort, thinking_budget = (
        _resolve_synthesis_config(body, config)
    )

    llm_client = CodeSpectraProxyClient(
        backend_url, backend_token, provider_id, model_id,
        reasoning_effort=reasoning_effort, thinking_budget=thinking_budget,
    )

    embedding_client = None
    if body.use_local_embedding or body.embedding_provider_id:
        from agent_eval_harness.llm.embedding_client import CodeSpectraEmbeddingProxyClient
        embedding_client = CodeSpectraEmbeddingProxyClient(
            backend_url,
            backend_token,
            provider_id=body.embedding_provider_id,
            model_id=body.embedding_model_id,
            use_local=body.use_local_embedding,
        )

    client = CodeSpectraClient(backend_url, backend_token)
    try:
        from agent_eval_harness.datasets.fulfillment import fulfill_plan

        snapshot = await client.get_snapshot(sess["snapshot_id"])
        local_path_str = snapshot.get("local_path")
        if not local_path_str:
            raise HTTPException(status_code=400, detail="Snapshot is missing local_path context.")

        report = await fulfill_plan(
            sess["plan_path"], sess["map_path"], sess["snapshot_id"], local_path_str,
            provider_id, model_id, llm_client, instructions=body.instructions,
            embedding_client=embedding_client, codespectra_client=client,
            only_agent_ids=body.agent_ids,
            only_kind="synthetic_agent_io" if body.agent_ids else None,
            force_agent_ids=body.force_agent_ids,
            session_id=session_id,
        )
        return report
    finally:
        await client.aclose()
        if hasattr(llm_client, "aclose"):
            await llm_client.aclose()
        if embedding_client and hasattr(embedding_client, "aclose"):
            await embedding_client.aclose()


@app.put("/api/discovery/expansion-sessions/{session_id}/plan")
async def update_plan_route(session_id: str, body: UpdatePlanRequest):
    sess = await _get_expansion_session_or_404(session_id)
    plan_path_str = sess.get("plan_path")
    if not plan_path_str or not Path(plan_path_str).exists():
        raise HTTPException(status_code=400, detail="No plan to update.")

    from agent_eval_harness.metrics.suite import Suite, load_suite
    import yaml

    old_suite = load_suite(plan_path_str)
    old_by_id = {e.id: e for e in old_suite.entries}

    try:
        new_suite = Suite.model_validate({"entries": body.entries})
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid plan entries: {e}")

    updated_entries = []
    for entry in new_suite.entries:
        old = old_by_id.get(entry.id)
        if old is None:
            updated = entry.model_copy(update={"provenance": "human_added"})
        elif (
            entry.metric != old.metric
            or entry.metric_class != old.metric_class
            or entry.rationale != old.rationale
            or entry.params != old.params
        ):
            updated = entry.model_copy(update={"provenance": "human_added"})
        else:
            updated = entry
        updated_entries.append(updated)

    final_suite = Suite(entries=updated_entries)
    Path(plan_path_str).write_text(
        yaml.dump(final_suite.model_dump(), allow_unicode=True),
        encoding="utf-8"
    )
    return {"success": True}


class AgentFlowRequest(BaseModel):
    """provider_id/model_id here select LLM 2, independent of the expansion/classify model."""
    provider_id: str | None = None
    model_id: str | None = None
    backend_url: str | None = None
    backend_token: str | None = None
    reasoning_effort: str | None = None
    thinking_budget: int | None = None


@app.post("/api/discovery/expansion-sessions/{session_id}/agent-flows")
@_log_and_500("generate_agent_flows failed")
async def generate_agent_flows_route(session_id: str, body: AgentFlowRequest):
    sess = await _get_expansion_session_or_404(session_id)
    if sess["status"] != "completed":
        raise HTTPException(status_code=400, detail="Expansion has not completed.")
    if not sess["map_path"]:
        raise HTTPException(status_code=400, detail="No system map for this session.")

    config = AEHConfig.load()
    llm_client, client = _build_synthesis_clients(body, config)

    try:
        system_map = load_system_map(sess["map_path"])

        snapshot = await client.get_snapshot(sess["snapshot_id"])
        local_path_str = snapshot.get("local_path")
        if not local_path_str:
            raise HTTPException(
                status_code=400, detail="Snapshot is missing local_path context."
            )
        local_path = Path(local_path_str)
        # accepted entries are dicts with file, role_hint, key_symbols, follow fields
        abs_files = [local_path / (p["file"] if isinstance(p, dict) else p) for p in sess["accepted"]]

        source_by_component = build_source_by_component(abs_files, system_map)

        project_ctx = await load_project_context(client, sess["snapshot_id"])
        feature_map_hint = (
            project_ctx.feature_map.as_context_block()
            if project_ctx and project_ctx.feature_map
            else None
        )
        agent_flow_map = await separate_agent_flows(
            system_map, source_by_component, llm_client, feature_map_hint=feature_map_hint
        )

        map_path = Path(sess["map_path"])
        agent_flows_path = map_path.with_name(map_path.stem + "_agentflows.yaml")
        save_agent_flow_map(agent_flow_map, agent_flows_path)

        await repository.update_expansion_session_agentflows_path(
            session_id, str(agent_flows_path)
        )

        return agent_flow_map.model_dump()
    finally:
        await client.aclose()
        if hasattr(llm_client, "aclose"):
            await llm_client.aclose()


@app.get("/api/discovery/expansion-sessions/{session_id}/agent-flows")
async def get_agent_flows_route(session_id: str):
    sess = await _get_expansion_session_or_404(session_id)
    agent_flows_path_str = sess.get("agent_flows_path")
    if not agent_flows_path_str or not Path(agent_flows_path_str).exists():
        raise HTTPException(status_code=404, detail="No agent-flow map for this session.")
    try:
        agent_flow_map = load_agent_flow_map(agent_flows_path_str)
        return agent_flow_map.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load agent-flow map: {e}")


@app.post("/api/discovery/expansion-sessions/{session_id}/enrich-agents")
@_log_and_500("enrich_agents failed")
async def enrich_agents_route(session_id: str, body: EnrichAgentsRequest):
    """Stage 2.5 enrichment: concurrent LLM analysis of discovered agents."""
    sess = await _get_expansion_session_or_404(session_id)
    if sess["status"] != "completed":
        raise HTTPException(status_code=400, detail="Expansion has not completed.")
    if not sess["map_path"]:
        raise HTTPException(status_code=400, detail="No system map for this session.")
    agent_flows_path_str = sess.get("agent_flows_path")
    if not agent_flows_path_str or not Path(agent_flows_path_str).exists():
        raise HTTPException(
            status_code=400,
            detail="Run Stage 2 agent-flow separation first.",
        )

    config = AEHConfig.load()
    llm_client, client = _build_synthesis_clients(body, config)

    try:
        system_map = load_system_map(sess["map_path"])
        agent_flow_map = load_agent_flow_map(agent_flows_path_str)

        snapshot = await client.get_snapshot(sess["snapshot_id"])
        local_path_str = snapshot.get("local_path")
        if not local_path_str:
            raise HTTPException(status_code=400, detail="Snapshot is missing local_path context.")

        from agent_eval_harness.discovery.enrichment import enrich_agents

        knowledge_list = await enrich_agents(
            session_id=session_id,
            agent_flow_map=agent_flow_map,
            system_map=system_map,
            accepted_with_annotations=sess.get("accepted", []),
            accepted_edges=sess.get("accepted_edges", []),
            client=client,
            llm_client=llm_client,
            snapshot_id=sess["snapshot_id"],
            depth=body.depth or 'normal',
            agent_ids=body.agent_ids,
            force_agent_ids=body.force_agent_ids,
            map_path=sess["map_path"],
            agent_flows_path=agent_flows_path_str,
            repo_root=Path(local_path_str),
        )

        enriched_count = sum(1 for k in knowledge_list if not k.degraded)
        degraded_count = sum(1 for k in knowledge_list if k.degraded)
        skipped_count = len(agent_flow_map.agents) - len(knowledge_list)

        return {
            "enriched_count": enriched_count,
            "degraded_count": degraded_count,
            "skipped_count": skipped_count,
        }
    finally:
        await client.aclose()
        if hasattr(llm_client, "aclose"):
            await llm_client.aclose()


@app.get("/api/discovery/expansion-sessions/{session_id}/agent-knowledge")
async def get_agent_knowledge_route(session_id: str):
    """Agent knowledge records with the parsed sidecar JSON content included."""
    await _get_expansion_session_or_404(session_id)
    knowledge_list = await repository.list_agent_knowledge(session_id)
    for record in knowledge_list:
        record["content"] = None
        json_path_str = record.get("json_path")
        if not json_path_str:
            continue
        try:
            record["content"] = json.loads(Path(json_path_str).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Could not read agent knowledge sidecar {json_path_str}: {e}")
    return knowledge_list


@app.post("/api/discovery/expansion-sessions/{session_id}/eval-branch")
async def create_eval_branch(session_id: str, base_ref: str = "main"):
    await _get_expansion_session_or_404(session_id)
    repo_root = _get_repo_root()
    try:
        info = BranchInjectionTarget.prepare(repo_root, session_id, base_ref=base_ref)
    except InjectionTargetError as e:
        raise HTTPException(status_code=409, detail=str(e))
    await repository.update_expansion_session_eval_branch(
        session_id, info.branch_name, info.current_branch
    )
    return {"branch_name": info.branch_name, "current_branch": info.current_branch}


@app.post("/api/discovery/expansion-sessions/{session_id}/eval-branch/restore")
async def restore_eval_branch(session_id: str):
    sess = await _get_expansion_session_or_404(session_id)
    original_branch = sess.get("eval_original_branch")
    if not original_branch:
        raise HTTPException(
            status_code=400,
            detail="No eval branch recorded for this session.",
        )
    repo_root = _get_repo_root()
    try:
        BranchInjectionTarget.restore(repo_root, original_branch)
    except InjectionTargetError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"restored_branch": original_branch}


def _plan_dir_for(session_id: str) -> Path:
    return Path(os.getenv("AEH_DATA_DIR", ".")) / "aeh" / session_id / "plans"


def _reharvest_contracts(sess: dict, system_map) -> dict:
    """Re-run the contract harvest at render time (pure AST, costs nothing) — the plan_report's frozen copy goes stale as soon as the target changes, sending the coding agent at a call shape that no longer exists.
    Returns {} on any problem so the caller keeps the stored copy."""
    try:
        flows_path = sess.get("agent_flows_path")
        if not flows_path or not Path(flows_path).exists():
            return {}
        root = _get_repo_root()
        rel = [(p["file"] if isinstance(p, dict) else p) for p in (sess.get("accepted") or [])]
        rel += list(sess.get("boundary") or [])
        files = [f for f in (root / r for r in rel) if f.exists()]
        if not files:
            return {}
        from agent_eval_harness.mapping.builder.contract_harvest import harvest_contracts

        return harvest_contracts(
            system_map, load_agent_flow_map(flows_path), files, root,
            AEHConfig.load().contract_conventions,
        )
    except Exception as e:
        logger.warning(f"eval-plan: contract re-harvest failed, using stored contracts: {e}")
        return {}


@app.get("/api/discovery/expansion-sessions/{session_id}/eval-plan")
async def get_eval_plan(session_id: str):
    """Existing handoff plan for this session, or plan_path=None if none was rendered."""
    sess = await _get_expansion_session_or_404(session_id)
    stored = sess.get("eval_plan_md_path")
    plan_dir = Path(stored).parent if stored else _plan_dir_for(session_id)
    files = sorted(p.name for p in plan_dir.glob("*.md")) if plan_dir.exists() else []
    if not files:
        return {"plan_path": None, "plan_dir": None, "files": []}
    entry = plan_dir / "AGENTS.md"
    return {
        "plan_path": str(entry if entry.exists() else plan_dir / files[0]),
        "plan_dir": str(plan_dir),
        "files": files,
    }


@app.delete("/api/discovery/expansion-sessions/{session_id}/eval-plan")
async def delete_eval_plan(session_id: str):
    """Delete this session's rendered plan files and clear the idempotency pointer. Files the coding agent authors are kept — Regenerate routes through here, so deleting them would defeat the point of preserving them on re-render."""
    await _get_expansion_session_or_404(session_id)
    plan_dir = _plan_dir_for(session_id)
    removed = 0
    kept: list[str] = []
    if plan_dir.exists():
        for p in plan_dir.glob("*.md"):
            if p.name in PRESERVED_PLAN_FILES:
                kept.append(p.name)
                continue
            p.unlink()
            removed += 1
    await repository.update_expansion_session_eval_plan_md_path(session_id, None)
    return {"deleted": removed, "kept": kept}


@app.post("/api/discovery/expansion-sessions/{session_id}/eval-plan")
async def create_eval_plan(session_id: str, base_ref: str = "main"):
    sess = await _get_expansion_session_or_404(session_id)
    repo_root = _get_repo_root()
    eval_branch = sess.get("eval_branch_name") or f"aeh/eval-{session_id}"

    # Idempotent on the stored plan path; nothing is checked in the target repo.
    if sess.get("eval_plan_md_path"):
        existing = Path(sess["eval_plan_md_path"])
        existing_dir = existing.parent
        return {
            "plan_path": str(existing),
            "plan_dir": str(existing_dir),
            "files": sorted(p.name for p in existing_dir.glob("*.md")) if existing_dir.exists() else [],
        }

    if not sess.get("map_path"):
        raise HTTPException(status_code=400, detail="No system map for this session.")
    if not sess.get("plan_path"):
        raise HTTPException(
            status_code=400,
            detail="No plan for this session — generate a plan first.",
        )
    plan_report_path_str = sess.get("plan_report_path")
    if not plan_report_path_str or not Path(plan_report_path_str).exists():
        raise HTTPException(
            status_code=400,
            detail="No plan report for this session — regenerate the plan in Stage 3.",
        )
    system_map = load_system_map(sess["map_path"])
    suite = load_suite(sess["plan_path"])
    plan_report = load_plan_report(plan_report_path_str)
    # In-memory only: the plan on disk keeps whatever Stage 3 recorded.
    fresh_contracts = _reharvest_contracts(sess, system_map)
    for agent_report in plan_report.agents:
        if agent_report.agent_id in fresh_contracts:
            agent_report.contract = fresh_contracts[agent_report.agent_id]
    datasets_to_gate_ids: dict[str, list[str]] = {}
    for entry in suite.entries:
        if entry.dataset and entry.dataset.ref:
            datasets_to_gate_ids.setdefault(entry.dataset.ref, []).append(entry.id)
    # Descriptive only — run_eval.py reads cases live from AEH's sqlite DB, not this export.
    dataset_summaries: list[dict] = []
    for dataset_id, gate_ids in datasets_to_gate_ids.items():
        cases = await export_dataset(dataset_id)
        # Metadata knows the kind even when every case is filtered out by provenance.
        meta = await repository.get_dataset_metadata(dataset_id)
        kind = (meta or {}).get("kind") or (cases[0].kind if cases else "")
        dataset_summaries.append({
            "dataset_id": dataset_id,
            "kind": kind,
            "case_count": len(cases),
            "gate_ids": gate_ids,
            "example_case": cases[0].model_dump() if cases else None,
        })
    # Derive agent_invocations from harvested contracts (no CodeSpectra hardcodes)
    agent_invocations = {}
    if plan_report and plan_report.agents:
        for agent_report in plan_report.agents:
            if agent_report.contract and agent_report.contract.invocation:
                invocation = agent_report.contract.invocation
                agent_invocations[agent_report.agent_id] = {
                    "invocation_mode": invocation.invocation_mode,
                    "case_binding": invocation.case_binding,
                    "route": invocation.route,
                    "constructor_deps": invocation.constructor_deps,
                }

    wiring = build_wiring(system_map, session_id, dataset_kinds=None, agent_invocations=agent_invocations)
    aeh_data_dir = Path(os.getenv("AEH_DATA_DIR", "."))
    wiring["aeh_db_path"] = str((aeh_data_dir / DB_FILENAME).resolve())
    # Spanlogs and manifests land here, never inside the target repo.
    wiring["aeh_out_dir"] = str((aeh_data_dir / "aeh" / session_id / "runs").resolve())
    wiring["dataset_ids"] = sorted(datasets_to_gate_ids.keys())
    from agent_eval_harness.discovery.enrichment import resolve_agent_knowledge_dir
    files = render_eval_plan_files(
        system_map, wiring, dataset_summaries, session_id, eval_branch,
        plan_report=plan_report, repo_root=repo_root, base_ref=base_ref,
        knowledge_dir=resolve_agent_knowledge_dir(session_id),
    )

    # The 4 handoff files live in AppData, never in the target repo.
    plan_dir = aeh_data_dir / "aeh" / session_id / "plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in files.items():
        target = plan_dir / filename
        if filename in PRESERVED_PLAN_FILES and target.exists():
            continue  # the coding agent's own notes — a re-render must not erase them
        target.write_text(content, encoding="utf-8")

    # Idempotency + UI anchor points at the entrypoint file (AGENTS.md) in the plan dir.
    entry_path = plan_dir / "AGENTS.md"
    await repository.update_expansion_session_eval_plan_md_path(
        session_id, str(entry_path)
    )
    return {"plan_path": str(entry_path), "plan_dir": str(plan_dir),
            "files": sorted(files.keys())}


@app.post("/api/discovery/expansion-sessions/{session_id}/eval-results")
async def load_eval_results(session_id: str, manifest_path: str | None = None):
    """Load eval results, delegating to the generic ingest endpoint; if manifest_path is None, attempts a legacy path for backward compat."""
    sess = await _get_expansion_session_or_404(session_id)
    from agent_eval_harness.ingest.spanlog_ingest import IngestError, parse_spanlog, persist_spanlog

    # Runs write their manifest to this session's own runs dir; the pre-AppData location is only a fallback for plans installed before that move.
    if not manifest_path:
        candidates = [
            Path(os.getenv("AEH_DATA_DIR", ".")) / "aeh" / session_id / "runs" / "manifest.json",
            _get_repo_root() / "backend" / ".aeh" / "manifest.json",
        ]
        manifest_path = str(next((c for c in candidates if c.exists()), candidates[0]))

    manifest_path_obj = Path(manifest_path)
    if not manifest_path_obj.exists():
        raise HTTPException(
            status_code=404,
            detail=f"manifest not found at {manifest_path}",
        )

    try:
        manifest = json.loads(manifest_path_obj.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"could not read manifest: {e}")

    # Idempotent on the manifest's run_id — re-ingesting an unchanged manifest returns the existing run
    manifest_run_id = manifest.get("run_id")
    if (
        manifest_run_id
        and sess.get("eval_manifest_run_id") == manifest_run_id
        and sess.get("eval_run_id")
    ):
        existing = await repository.get_run(sess["eval_run_id"])
        if existing:
            return {"run_id": sess["eval_run_id"], "status": existing["status"]}

    log_path_str = manifest.get("log_path")
    if not log_path_str or not Path(log_path_str).exists():
        raise HTTPException(
            status_code=400,
            detail="manifest is missing a valid log_path",
        )

    try:
        parsed = parse_spanlog(Path(log_path_str))
    except IngestError as e:
        raise HTTPException(status_code=422, detail=str(e))

    run_id = await persist_spanlog(
        parsed,
        target_system_id=manifest.get("plan_id", "unknown"),
        eval_plan_id=manifest.get("plan_id"),
    )
    await repository.update_expansion_session_eval_run(session_id, run_id, manifest_run_id)
    run = await repository.get_run(run_id)
    return {"run_id": run_id, "status": run["status"] if run else "unknown"}


@app.get("/api/eval-runs/{run_id}/cases")
async def get_eval_run_cases(run_id: str):
    """Per-agent input/result/expected for an ingested run; agent_id comes from the dataset case's own labels_json."""
    run = await repository.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
    traces = await repository.get_traces_for_run(run_id)
    evaluations = await repository.get_evaluations_for_run(run_id)
    evals_by_trace: dict[str, list[dict]] = {}
    agent_summaries: dict[str, dict] = {}
    for ev in evaluations:
        if ev.get("trace_id"):
            evals_by_trace.setdefault(ev["trace_id"], []).append({
                "metric_name": ev["metric_name"],
                "metric_class": ev["metric_class"],
                "score": ev["score"],
                "details": json.loads(ev["details_json"]) if ev.get("details_json") else {},
            })
        elif ev["metric_name"] == "agent_summary" and ev.get("component_id"):
            details = json.loads(ev["details_json"]) if ev.get("details_json") else {}
            agent_summaries[ev["component_id"]] = {
                "insight": details.get("insight", ""),
                "case_count": details.get("case_count", 0),
                "avg_score": ev["score"],
            }

    case_ids = [t["dataset_case_id"] for t in traces if t.get("dataset_case_id")]
    cases_by_id = await repository.get_dataset_cases_by_ids(case_ids)

    agents: dict[str, list[dict]] = {}
    for trace in traces:
        case_id = trace.get("dataset_case_id")
        dataset_case = cases_by_id.get(case_id) if case_id else None
        agent_id = "unknown"
        expected = None
        if dataset_case:
            try:
                labels = json.loads(dataset_case["labels_json"]) if dataset_case.get("labels_json") else {}
            except json.JSONDecodeError:
                labels = {}
            agent_id = labels.get("agent_id", "unknown")
            if dataset_case.get("expected_json"):
                try:
                    expected = json.loads(dataset_case["expected_json"])
                except json.JSONDecodeError:
                    expected = dataset_case["expected_json"]
        agents.setdefault(agent_id, []).append({
            "case_id": case_id,
            "trace_id": trace["id"],
            "input": trace.get("root_input"),
            "result": trace.get("final_output"),
            "expected": expected,
            "evaluations": evals_by_trace.get(trace["id"], []),
        })
    return {"run_id": run_id, "status": run["status"], "agents": agents, "agent_summaries": agent_summaries}


_JUDGE_BATCH_SIZE = 4  # concurrent LLM calls per batch — this is a dev-only tool, favor speed


@app.post("/api/eval-runs/{run_id}/judge")
async def judge_eval_run_agent_cases(run_id: str, body: JudgeAgentCasesRequest):
    """Ad-hoc per-agent semantic-match + field precision/recall scoring; idempotent per case
    (existing 'semantic_match' eval is skipped) unless `force`, which re-scores every case."""
    run = await repository.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")

    if body.provider_id and body.backend_url and body.backend_token:
        llm_client = CodeSpectraProxyClient(
            body.backend_url, body.backend_token, body.provider_id, body.model_id,
            reasoning_effort=body.reasoning_effort, thinking_budget=body.thinking_budget,
        )
    else:
        llm_client = FakeLLMClient(
            LLMResponse(content='{"score": 0.0, "notes": "no provider configured"}', model="fake")
        )

    traces = await repository.get_traces_for_run(run_id)
    if body.force:
        # Wipe first, so a re-judge replaces this agent's scores rather than stacking a second set of rows the UI would read arbitrarily.
        await repository.delete_evaluations_for_component(run_id, body.agent_id)
        already_scored_trace_ids: set[str] = set()
    else:
        existing = await repository.get_evaluations_for_component(run_id, body.agent_id)
        already_scored_trace_ids = {
            e["trace_id"] for e in existing if e["metric_name"] == "semantic_match"
        }

    case_ids = [t["dataset_case_id"] for t in traces if t.get("dataset_case_id")]
    cases_by_id = await repository.get_dataset_cases_by_ids(case_ids)

    pending: list[tuple[dict, Any, Any]] = []
    skipped = 0
    for trace in traces:
        case_id = trace.get("dataset_case_id")
        if not case_id:
            continue
        dataset_case = cases_by_id.get(case_id)
        if not dataset_case:
            continue
        try:
            labels = json.loads(dataset_case["labels_json"]) if dataset_case.get("labels_json") else {}
        except json.JSONDecodeError:
            labels = {}
        if labels.get("agent_id") != body.agent_id:
            continue
        if trace["id"] in already_scored_trace_ids:
            skipped += 1
            continue

        try:
            result = json.loads(trace["final_output"]) if trace.get("final_output") else None
        except json.JSONDecodeError:
            result = trace.get("final_output")
        expected = None
        if dataset_case.get("expected_json"):
            try:
                expected = json.loads(dataset_case["expected_json"])
            except json.JSONDecodeError:
                expected = dataset_case["expected_json"]
        pending.append((trace, result, expected))

    scored = 0
    for i in range(0, len(pending), _JUDGE_BATCH_SIZE):
        batch = pending[i:i + _JUDGE_BATCH_SIZE]
        judged_batch = await asyncio.gather(
            *[judge_case_semantic_match(llm_client, result, expected) for _trace, result, expected in batch]
        )
        for (trace, _result, _expected), judged in zip(batch, judged_batch):
            await repository.insert_evaluation(
                run_id, "semantic_match", "llm_judge",
                trace_id=trace["id"], component_id=body.agent_id,
                score=judged["score"], details={"notes": judged["notes"]},
                evaluator=judged["model"], cost_tokens=judged["tokens"],
            )
            for field_name, pr in judged["field_matches"].items():
                # F1, not precision — precision alone rewards omitting gold items and punishes correct extra detail; None when gold was empty means unscorable, not zero.
                await repository.insert_evaluation(
                    run_id, f"precision_recall.{field_name}", "llm_judge",
                    trace_id=trace["id"], component_id=body.agent_id,
                    score=pr["f1"], details=pr,
                )
            for field_name, sm in judged["scalar_matches"].items():
                await repository.insert_evaluation(
                    run_id, f"field_exact.{field_name}", "llm_judge",
                    trace_id=trace["id"], component_id=body.agent_id,
                    score=sm["score"], details=sm,
                )
            scored += 1

    return {"scored": scored, "skipped": skipped}


@app.post("/api/eval-runs/{run_id}/agent-summary")
async def summarize_eval_run_agent(run_id: str, body: JudgeAgentCasesRequest):
    """Synthesizes one holistic insight from an agent's already-judged per-case scores; idempotent per run."""
    run = await repository.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")

    existing = await repository.get_evaluations_for_component(run_id, body.agent_id)
    prior = next((e for e in existing if e["metric_name"] == "agent_summary"), None)
    if prior:
        details = json.loads(prior["details_json"]) if prior.get("details_json") else {}
        return {"insight": details.get("insight", ""), "case_count": details.get("case_count", 0), "cached": True}

    case_summaries = [
        {"score": e["score"], "notes": json.loads(e["details_json"]).get("notes", "") if e.get("details_json") else ""}
        for e in existing
        if e["metric_name"] == "semantic_match" and e["score"] is not None
    ]
    if not case_summaries:
        raise HTTPException(
            status_code=400,
            detail="No scored cases yet for this agent — run /judge for it first.",
        )

    if body.provider_id and body.backend_url and body.backend_token:
        llm_client = CodeSpectraProxyClient(
            body.backend_url, body.backend_token, body.provider_id, body.model_id,
            reasoning_effort=body.reasoning_effort, thinking_budget=body.thinking_budget,
        )
    else:
        llm_client = FakeLLMClient(
            LLMResponse(content='{"insight": "no provider configured"}', model="fake")
        )

    summary = await summarize_agent_judgments(llm_client, case_summaries)
    avg_score = sum(c["score"] for c in case_summaries) / len(case_summaries)
    await repository.insert_evaluation(
        run_id, "agent_summary", "llm_judge",
        component_id=body.agent_id, score=avg_score,
        details={"insight": summary["insight"], "case_count": len(case_summaries)},
        evaluator=summary["model"], cost_tokens=summary["tokens"],
    )
    return {"insight": summary["insight"], "case_count": len(case_summaries), "cached": False}


@app.post("/api/discovery/sessions/{session_id}/advance")
async def advance_session(session_id: str, body: AdvanceSessionRequest):
    sess = await repository.get_discovery_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found.")

    stage = sess.get("pipeline_stage", "fingerprinting")

    if stage == "awaiting_candidate_review":
        if not body.confirmed_candidates:
            raise HTTPException(
                status_code=400,
                detail="Must provide confirmed_candidates to advance past candidate review."
            )
        await repository.update_discovery_session_pipeline_stage(session_id, "expanding")

    elif stage == "awaiting_map_review":
        if not body.confirmed_map_session_id:
            raise HTTPException(
                status_code=400,
                detail="Must provide confirmed_map_session_id to advance past map review."
            )
        await repository.update_discovery_session_pipeline_stage(session_id, "planning")

    elif stage == "awaiting_plan_review":
        if not body.confirmed_plan:
            raise HTTPException(
                status_code=400,
                detail="Must confirm the plan (confirmed_plan=true) to advance to done."
            )
        await repository.update_discovery_session_pipeline_stage(session_id, "done")

    elif stage in ("fingerprinting", "expanding", "planning"):
        return {"pipeline_stage": stage, "status": sess["status"]}

    else:
        raise HTTPException(status_code=400, detail=f"Unknown pipeline stage: {stage}")

    updated = await repository.get_discovery_session(session_id)
    return {"pipeline_stage": updated.get("pipeline_stage")}


current_dir = Path(__file__).parent
dist_dir = current_dir / "dist"

if dist_dir.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=str(dist_dir / "assets")),
        name="assets",
    )


@app.get("/{catchall:path}")
async def read_index(catchall: str):
    index_file = dist_dir / "index.html"
    if index_file.exists():
        if catchall.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")
        return FileResponse(str(index_file))
    return {
        "message": (
            "AEH UI server is running on localhost. "
            "Please build the frontend (npm run build) to serve the UI."
        )
    }
