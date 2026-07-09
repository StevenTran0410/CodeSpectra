"""FastAPI UI Server for AEH per-agent drill-down dashboard."""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import asyncio
from agent_eval_harness.config import AEHConfig
from agent_eval_harness.llm.proxy_client import CodeSpectraProxyClient
from agent_eval_harness.discovery.client import CodeSpectraClient
from agent_eval_harness.discovery.engine import run_discovery_background
from agent_eval_harness.discovery.expansion import expand_candidate
from agent_eval_harness.mapping.agent_flow import (
    build_source_by_component,
    load_agent_flow_map,
    save_agent_flow_map,
    separate_agent_flows,
)
from agent_eval_harness.mapping.builder.pipeline import SystemMapBuilder
from agent_eval_harness.mapping.system_map import load_system_map, save_system_map, SystemMap
from agent_eval_harness.planning.agentic_planner import generate_plan_agentic
from agent_eval_harness.planning.report import load_plan_report, save_plan_report
from agent_eval_harness.store import repository
from agent_eval_harness.store.database import close_db, get_db, init_db

logger = logging.getLogger("agent_eval_harness.ui.server")

# Life-span manager for db initialization
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Set the data directory if env is not already set
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

# Enable CORS for local development (e.g. Vite on port 5173 talking to FastAPI on 8321)
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


# --- API Routes ---

async def _get_run_or_404(run_id: str) -> dict:
    run = await repository.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.get("/api/runs", response_model=list[RunListItem])
async def get_runs(target_system_id: str | None = None):
    try:
        db_runs = await repository.list_runs(target_system_id)
        results = []
        for r in db_runs:
            defects = []
            if r.get("active_defects"):
                try:
                    defects = json.loads(r["active_defects"])
                except Exception:
                    pass
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
    except Exception as e:
        logger.error(f"Failed to list runs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/runs/{run_id}", response_model=RunDetailResponse)
async def get_run_detail(run_id: str):
    run = await _get_run_or_404(run_id)

    # Load system map from map_path if possible
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

    # Fetch component aggregates
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
        except Exception:
            pass

    # Fetch overall pass rate from database
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
        except Exception:
            pass

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
            except Exception:
                pass
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


class ProviderSummary(BaseModel):
    provider_id: str
    display_name: str
    model_id: str | None = None


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
    from agent_eval_harness.llm.client import LLMResponse
    from agent_eval_harness.llm.fake_client import FakeLLMClient
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

    # 1. Determine active defects (fallback to parent's active defects if not provided)
    active_defects = body.active_defects
    if active_defects is None:
        try:
            active_defects = (
                json.loads(parent["active_defects"]) if parent.get("active_defects") else []
            )
        except Exception:
            active_defects = []

    # 2. Insert the new run row in "running" state
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

    # 3. Define the background executor
    async def _execute_rerun_task():
        try:
            config = AEHConfig.load()

            # Build the base/default client
            if config.provider_id and config.backend_url and config.backend_token:
                default_client = CodeSpectraProxyClient(
                    config.backend_url,
                    config.backend_token,
                    config.provider_id,
                    config.model_id,
                )
            else:
                default_client = FakeLLMClient(
                    LLMResponse(
                        content="This is a fallback offline demo answer.", model="fake-default"
                    )
                )

            # Build override clients
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
                # Determine provider_id and model_id
                if ":" in override_val:
                    p_id, m_id = override_val.split(":", 1)
                else:
                    p_id = config.provider_id or "fake-provider"
                    m_id = override_val

                # Build client for this override
                if config.backend_url and config.backend_token and p_id != "fake-provider":
                    client = CodeSpectraProxyClient(
                        config.backend_url,
                        config.backend_token,
                        provider_id=p_id,
                        model_id=m_id,
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
                # Run the sweep using the routing client, passing new_run_id to reuse the row
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
            except Exception:
                pass
        finally:
            _rerun_in_flight.discard(run_id)

    asyncio.create_task(_execute_rerun_task())
    return {"run_id": new_run_id}


# --- Discovery Models & Endpoints ---

class StartDiscoveryRequest(BaseModel):
    repo_ref: str
    snapshot_id: str
    provider_id: str | None = None
    model_id: str | None = None
    backend_url: str | None = None
    backend_token: str | None = None


class DiscoverySessionResponse(BaseModel):
    id: str
    repo_ref: str
    snapshot_id: str
    status: str
    error: str | None
    created_at: str
    finished_at: str | None
    pause_info: dict | None = None


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


def _resolve_synthesis_config(
    body: StartDiscoveryRequest | ExpandCandidateRequest, config: AEHConfig
) -> tuple[str, str, str, str | None]:
    """Resolve provider/backend config for Pass C synthesis LLM calls, or raise 400."""
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
            detail="No LLM provider configured for Pass C synthesis. Set up a provider in "
            "Settings, then select it before running Discovery.",
        )
    return provider_id, backend_url, backend_token, body.model_id or config.model_id


@app.post("/api/discovery/sessions")
async def start_discovery(body: StartDiscoveryRequest):
    try:
        config = AEHConfig.load()
        provider_id, backend_url, backend_token, model_id = _resolve_synthesis_config(body, config)
        llm_client = CodeSpectraProxyClient(backend_url, backend_token, provider_id, model_id)

        client = CodeSpectraClient(backend_url, backend_token)

        # Start session
        session_id = await repository.insert_discovery_session(body.repo_ref, body.snapshot_id)

        # Execute background scan
        asyncio.create_task(
            run_discovery_background(session_id, body.snapshot_id, body.repo_ref, client, llm_client)
        )

        return {"session_id": session_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start discovery session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/discovery/sessions", response_model=list[DiscoverySessionResponse])
async def list_discovery_sessions(repo_ref: str | None = None, snapshot_id: str | None = None):
    try:
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
            )
            for s in sessions
        ]
    except Exception as e:
        logger.error(f"Failed to list discovery sessions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


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
    )


@app.get("/api/discovery/sessions/{session_id}/candidates", response_model=list[DiscoveryCandidateResponse])
async def list_discovery_candidates(session_id: str):
    try:
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
    except Exception as e:
        logger.error(f"Failed to list discovery candidates: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/discovery/candidates/{candidate_id}/verdict")
async def update_candidate_verdict(candidate_id: str, body: UpdateVerdictRequest):
    try:
        if body.verdict not in ["proposed", "confirmed", "rejected"]:
            raise HTTPException(status_code=400, detail="Invalid verdict")
        await repository.update_candidate_verdict(candidate_id, body.verdict)
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update candidate verdict: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/discovery/candidates/{candidate_id}/excluded-files")
async def update_candidate_excluded_files_route(candidate_id: str, body: UpdateExcludedFilesRequest):
    try:
        await repository.update_candidate_excluded_files(candidate_id, body.excluded_files)
        return {"success": True}
    except Exception as e:
        logger.error(f"Failed to update excluded files: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class ResumeDiscoveryRequest(BaseModel):
    provider_id: str | None = None
    model_id: str | None = None
    backend_url: str | None = None
    backend_token: str | None = None


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

    llm_client = CodeSpectraProxyClient(backend_url, backend_token, provider_id, model_id)
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


# --- Expansion & Planning Endpoints ---

class ExpandCandidateRequest(BaseModel):
    provider_id: str | None = None
    model_id: str | None = None
    backend_url: str | None = None
    backend_token: str | None = None
    node_budget: int = 100
    hop_cap: int = 3
    classify_provider_id: str | None = None
    classify_model_id: str | None = None


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

        snapshot = await client.get_snapshot(snapshot_id)
        local_path_str = snapshot.get("local_path")
        if not local_path_str:
            raise RuntimeError(f"Snapshot {snapshot_id} is missing local_path context.")

        local_path = Path(local_path_str)
        abs_files = [local_path / p for p in res["accepted"]]

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
async def start_expansion(candidate_id: str, body: ExpandCandidateRequest):
    try:
        candidate = await repository.get_discovery_candidate(candidate_id)
        if not candidate:
            raise HTTPException(status_code=404, detail="Candidate component not found.")

        # discovery_candidates has no snapshot_id column — resolve it via the parent session
        session = await repository.get_discovery_session(candidate["session_id"])
        if not session:
            raise HTTPException(status_code=404, detail="Parent discovery session not found.")
        snapshot_id = session["snapshot_id"]

        config = AEHConfig.load()
        provider_id, backend_url, backend_token, model_id = _resolve_synthesis_config(body, config)
        llm_client = CodeSpectraProxyClient(backend_url, backend_token, provider_id, model_id)

        classify_provider_id = body.classify_provider_id or provider_id
        classify_model_id = body.classify_model_id or model_id
        if classify_provider_id == provider_id and classify_model_id == model_id:
            classify_llm_client = llm_client
        else:
            classify_llm_client = CodeSpectraProxyClient(
                backend_url, backend_token, classify_provider_id, classify_model_id
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
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Failed to start expansion session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


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


@app.post("/api/discovery/expansion-sessions/{session_id}/plan")
async def generate_plan_route(session_id: str, body: GeneratePlanRequest):
    """Stage 3's own DAG orchestrator (CS-265 redesign). Hard-gated on Stage 2's
    agent-flow separation having already run — the agent is the planning unit, so
    there is no flat-component fallback (see repo_atlas_plan/
    aeh_stage3_agentic_eval_planner_plan.md §2)."""
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

    config = AEHConfig.load()
    provider_id = body.provider_id or config.provider_id
    backend_url = body.backend_url or config.backend_url
    backend_token = body.backend_token or config.backend_token
    if not backend_url or not backend_token:
        raise HTTPException(status_code=400, detail="Missing backend connection config.")
    if not provider_id:
        raise HTTPException(status_code=400, detail="No LLM provider configured.")

    model_id = body.model_id or config.model_id
    llm_client = CodeSpectraProxyClient(backend_url, backend_token, provider_id, model_id)
    client = CodeSpectraClient(backend_url, backend_token)

    try:
        import yaml

        system_map = load_system_map(sess["map_path"])
        agent_flow_map = load_agent_flow_map(agent_flows_path_str)

        snapshot = await client.get_snapshot(sess["snapshot_id"])
        local_path_str = snapshot.get("local_path")
        if not local_path_str:
            raise HTTPException(status_code=400, detail="Snapshot is missing local_path context.")
        local_path = Path(local_path_str)
        abs_files = [local_path / p for p in sess["accepted"]]
        source_by_component = build_source_by_component(abs_files, system_map)

        suite, plan_report = await generate_plan_agentic(
            system_map, agent_flow_map, source_by_component, sess["accepted_edges"], llm_client,
            files=abs_files, files_root=local_path,
        )

        map_path = Path(sess["map_path"])
        plan_path = map_path.with_name(map_path.stem + "_plan.yaml")
        plan_path.write_text(
            yaml.dump(suite.model_dump(), allow_unicode=True),
            encoding="utf-8"
        )
        plan_report_path = map_path.with_name(map_path.stem + "_plan_report.yaml")
        save_plan_report(plan_report, plan_report_path)

        await repository.update_expansion_session_plan_path(session_id, str(plan_path))
        await repository.update_expansion_session_plan_report_path(session_id, str(plan_report_path))

        # Advance parent discovery session pipeline stage
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


@app.get("/api/discovery/expansion-sessions/{session_id}/plan")
async def get_plan_route(session_id: str):
    sess = await _get_expansion_session_or_404(session_id)
    plan_path_str = sess.get("plan_path")
    if not plan_path_str or not Path(plan_path_str).exists():
        raise HTTPException(status_code=404, detail="No plan file for this session.")
    try:
        from agent_eval_harness.metrics.suite import load_suite
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
    instructions: dict[str, dict] | None = None


@app.post("/api/discovery/expansion-sessions/{session_id}/datasets/fulfill")
async def fulfill_datasets_route(session_id: str, body: FulfillDatasetsRequest):
    sess = await _get_expansion_session_or_404(session_id)
    if not sess.get("plan_path") or not Path(sess["plan_path"]).exists():
        raise HTTPException(status_code=400, detail="No plan for this session — generate a plan first.")

    config = AEHConfig.load()
    provider_id = body.provider_id or config.provider_id
    backend_url = body.backend_url or config.backend_url
    backend_token = body.backend_token or config.backend_token
    if not backend_url or not backend_token:
        raise HTTPException(status_code=400, detail="Missing backend connection config.")
    if not provider_id:
        raise HTTPException(status_code=400, detail="No LLM provider configured.")
    model_id = body.model_id or config.model_id

    llm_client = CodeSpectraProxyClient(backend_url, backend_token, provider_id, model_id)
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
        )
        return report
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"fulfill_datasets failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await client.aclose()
        if hasattr(llm_client, "aclose"):
            await llm_client.aclose()


@app.put("/api/discovery/expansion-sessions/{session_id}/plan")
async def update_plan_route(session_id: str, body: UpdatePlanRequest):
    sess = await _get_expansion_session_or_404(session_id)
    plan_path_str = sess.get("plan_path")
    if not plan_path_str or not Path(plan_path_str).exists():
        raise HTTPException(status_code=400, detail="No plan to update.")

    from agent_eval_harness.metrics.suite import Suite, load_suite
    import yaml

    # Load existing plan to diff provenance
    old_suite = load_suite(plan_path_str)
    old_by_id = {e.id: e for e in old_suite.entries}

    # Validate via Suite.model_validate - hard gate per CS-273 spec
    try:
        new_suite = Suite.model_validate({"entries": body.entries})
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid plan entries: {e}")

    # Flip provenance to human_added for materially changed entries
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
            updated = entry  # unchanged - keep original provenance
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


@app.post("/api/discovery/expansion-sessions/{session_id}/agent-flows")
async def generate_agent_flows_route(session_id: str, body: AgentFlowRequest):
    sess = await _get_expansion_session_or_404(session_id)
    if sess["status"] != "completed":
        raise HTTPException(status_code=400, detail="Expansion has not completed.")
    if not sess["map_path"]:
        raise HTTPException(status_code=400, detail="No system map for this session.")

    config = AEHConfig.load()
    provider_id = body.provider_id or config.provider_id
    backend_url = body.backend_url or config.backend_url
    backend_token = body.backend_token or config.backend_token
    if not backend_url or not backend_token:
        raise HTTPException(status_code=400, detail="Missing backend connection config.")
    if not provider_id:
        raise HTTPException(status_code=400, detail="No LLM provider configured.")
    model_id = body.model_id or config.model_id

    llm_client = CodeSpectraProxyClient(backend_url, backend_token, provider_id, model_id)
    client = CodeSpectraClient(backend_url, backend_token)

    try:
        system_map = load_system_map(sess["map_path"])

        snapshot = await client.get_snapshot(sess["snapshot_id"])
        local_path_str = snapshot.get("local_path")
        if not local_path_str:
            raise HTTPException(
                status_code=400, detail="Snapshot is missing local_path context."
            )
        local_path = Path(local_path_str)
        abs_files = [local_path / p for p in sess["accepted"]]

        source_by_component = build_source_by_component(abs_files, system_map)
        agent_flow_map = await separate_agent_flows(system_map, source_by_component, llm_client)

        map_path = Path(sess["map_path"])
        agent_flows_path = map_path.with_name(map_path.stem + "_agentflows.yaml")
        save_agent_flow_map(agent_flow_map, agent_flows_path)

        await repository.update_expansion_session_agentflows_path(
            session_id, str(agent_flows_path)
        )

        return agent_flow_map.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"generate_agent_flows failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
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
        # Running - return current status (poll-friendly)
        return {"pipeline_stage": stage, "status": sess["status"]}

    else:
        raise HTTPException(status_code=400, detail=f"Unknown pipeline stage: {stage}")

    updated = await repository.get_discovery_session(session_id)
    return {"pipeline_stage": updated.get("pipeline_stage")}


current_dir = Path(__file__).parent
dist_dir = current_dir / "dist"

# If the UI build output directory exists, serve it
if dist_dir.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=str(dist_dir / "assets")),
        name="assets",
    )


@app.get("/{catchall:path}")
async def read_index(catchall: str):
    # Only serve the index if index.html exists, otherwise return a message
    index_file = dist_dir / "index.html"
    if index_file.exists():
        # Do not serve index.html for API requests that fall through
        if catchall.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")
        return FileResponse(str(index_file))
    return {
        "message": (
            "AEH UI server is running on localhost. "
            "Please build the frontend (npm run build) to serve the UI."
        )
    }
