"""Analysis runtime service (RPA-035)."""
from __future__ import annotations

import asyncio
import json

from domain.job.service import JobService
from domain.job.types import CreateJobRequest, StepName
from domain.manifest.service import ManifestService
from domain.manifest.types import BuildManifestRequest
from domain.repo_map.service import RepoMapService
from domain.repo_map.types import BuildRepoMapRequest
from domain.retrieval.service import RetrievalService
from domain.retrieval.types import (
    BuildRetrievalIndexRequest,
    RetrievalMode,
    RetrievalSection,
    RetrieveRequest,
)
from domain.structural_graph.service import StructuralGraphService
from domain.structural_graph.types import BuildGraphRequest
from infrastructure.db.database import get_db
from shared.errors import NotFoundError
from shared.logger import logger
from shared.utils import new_id, utc_now_iso

from .agent_pipeline import AnalysisAgentPipeline
from .types import (
    AnalysisEstimateResponse,
    AnalysisReport,
    AnalysisReportSummary,
    PrivacyMode,
    ScanMode,
    StartAnalysisRequest,
)

_bg_tasks: set[asyncio.Task] = set()


class AnalysisService:
    def __init__(self) -> None:
        self._jobs = JobService()
        self._manifest = ManifestService()
        self._repo_map = RepoMapService()
        self._graph = StructuralGraphService()
        self._retrieval = RetrievalService()
        self._agents = AnalysisAgentPipeline()

    async def estimate(self, repo_id: str, snapshot_id: str) -> AnalysisEstimateResponse:
        async with get_db().execute(
            "SELECT COUNT(*) as c FROM manifest_files WHERE snapshot_id=?",
            (snapshot_id,),
        ) as cur:
            row = await cur.fetchone()
        file_count = int(row["c"] if row else 0)
        # rough estimate for expectation management
        est_tokens = max(1200, file_count * 220)
        return AnalysisEstimateResponse(
            repo_id=repo_id,
            snapshot_id=snapshot_id,
            file_count=file_count,
            estimated_tokens=est_tokens,
        )

    async def list_reports(self, repo_id: str | None = None, limit: int = 30) -> list[AnalysisReportSummary]:
        db = get_db()
        safe_limit = max(1, min(limit, 200))
        if repo_id:
            async with db.execute(
                """
                SELECT id, job_id, repo_id, snapshot_id, provider_id, model_id, scan_mode, privacy_mode, created_at
                FROM analysis_reports
                WHERE repo_id=?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (repo_id, safe_limit),
            ) as cur:
                rows = await cur.fetchall()
        else:
            async with db.execute(
                """
                SELECT id, job_id, repo_id, snapshot_id, provider_id, model_id, scan_mode, privacy_mode, created_at
                FROM analysis_reports
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (safe_limit,),
            ) as cur:
                rows = await cur.fetchall()
        return [
            AnalysisReportSummary(
                id=r["id"],
                job_id=r["job_id"],
                repo_id=r["repo_id"],
                snapshot_id=r["snapshot_id"],
                provider_id=r["provider_id"],
                model_id=r["model_id"],
                scan_mode=ScanMode(r["scan_mode"]),
                privacy_mode=PrivacyMode(r["privacy_mode"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]

    async def get_report(self, report_id: str) -> AnalysisReport:
        async with get_db().execute(
            """
            SELECT id, job_id, repo_id, snapshot_id, provider_id, model_id, scan_mode, privacy_mode, report_json, created_at
            FROM analysis_reports
            WHERE id=?
            """,
            (report_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            raise NotFoundError("AnalysisReport", report_id)
        return AnalysisReport(
            summary=AnalysisReportSummary(
                id=row["id"],
                job_id=row["job_id"],
                repo_id=row["repo_id"],
                snapshot_id=row["snapshot_id"],
                provider_id=row["provider_id"],
                model_id=row["model_id"],
                scan_mode=ScanMode(row["scan_mode"]),
                privacy_mode=PrivacyMode(row["privacy_mode"]),
                created_at=row["created_at"],
            ),
            report=json.loads(row["report_json"] or "{}"),
        )

    async def get_report_by_job(self, job_id: str) -> AnalysisReport:
        async with get_db().execute(
            """
            SELECT id
            FROM analysis_reports
            WHERE job_id=?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (job_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            raise NotFoundError("AnalysisReportByJob", job_id)
        return await self.get_report(row["id"])

    async def start(self, req: StartAnalysisRequest):
        async with get_db().execute("SELECT 1 FROM local_repos WHERE id=?", (req.repo_id,)) as cur:
            repo = await cur.fetchone()
        if repo is None:
            raise NotFoundError("LocalRepo", req.repo_id)
        async with get_db().execute("SELECT 1 FROM repo_snapshots WHERE id=?", (req.snapshot_id,)) as cur:
            snap = await cur.fetchone()
        if snap is None:
            raise NotFoundError("RepoSnapshot", req.snapshot_id)

        if req.privacy_mode == PrivacyMode.BYOK_CLOUD:
            async with get_db().execute(
                "SELECT value FROM app_metadata WHERE key='cloud_consent_given'"
            ) as cur:
                c = await cur.fetchone()
            consent = c is not None and c["value"] == "true"
            if not consent:
                raise ValueError("Cloud consent is required before BYOK cloud analysis")

        steps = [
            StepName.MANIFEST.value,
            StepName.PARSE.value,
            StepName.GRAPH.value,
            StepName.GENERATE.value,
            StepName.EXPORT.value,
        ]
        if req.scan_mode == ScanMode.FULL:
            steps.insert(3, StepName.EMBED.value)

        job = await self._jobs.create(
            CreateJobRequest(type="analysis", repo_id=req.repo_id, steps=steps)
        )

        task = asyncio.create_task(self._run(job.id, req))
        _bg_tasks.add(task)
        task.add_done_callback(lambda t: _bg_tasks.discard(t))
        return job

    async def _run(self, job_id: str, req: StartAnalysisRequest) -> None:
        try:
            await self._jobs.start(job_id)

            async def _cancelled() -> bool:
                if self._jobs.is_cancelled(job_id):
                    logger.info(f"analysis job {job_id} cancelled signal received")
                    return True
                return False

            await self._jobs.update_step(job_id, StepName.MANIFEST.value, 5, "Building manifest")
            await self._manifest.build(BuildManifestRequest(snapshot_id=req.snapshot_id))
            if await _cancelled():
                return
            await self._jobs.update_step(job_id, StepName.MANIFEST.value, 100, "Manifest ready")

            await self._jobs.update_step(job_id, StepName.PARSE.value, 10, "Extracting symbols")
            await self._repo_map.build(BuildRepoMapRequest(snapshot_id=req.snapshot_id, force_rebuild=True))
            if await _cancelled():
                return
            await self._jobs.update_step(job_id, StepName.PARSE.value, 100, "Repo map ready")

            await self._jobs.update_step(job_id, StepName.GRAPH.value, 15, "Building structural graph")
            await self._graph.build(BuildGraphRequest(snapshot_id=req.snapshot_id, force_rebuild=True))
            if await _cancelled():
                return
            await self._jobs.update_step(job_id, StepName.GRAPH.value, 100, "Graph ready")

            if req.scan_mode == ScanMode.FULL:
                await self._jobs.update_step(job_id, StepName.EMBED.value, 20, "Building retrieval index")
                await self._retrieval.build_index(
                    BuildRetrievalIndexRequest(snapshot_id=req.snapshot_id, force_rebuild=True)
                )
                if await _cancelled():
                    return
                await self._jobs.update_step(job_id, StepName.EMBED.value, 100, "Retrieval index ready")

            if req.scan_mode != ScanMode.FULL:
                await self._jobs.update_step(job_id, StepName.GENERATE.value, 20, "Preparing retrieval index")
                await self._retrieval.build_index(
                    BuildRetrievalIndexRequest(snapshot_id=req.snapshot_id, force_rebuild=False)
                )
                if await _cancelled():
                    return

            await self._jobs.update_step(job_id, StepName.GENERATE.value, 30, "Retrieving section contexts")
            architecture = await self._retrieval.retrieve(
                RetrieveRequest(
                    snapshot_id=req.snapshot_id,
                    query="system architecture layers entrypoints modules integrations",
                    section=RetrievalSection.ARCHITECTURE,
                    mode=RetrievalMode.HYBRID,
                    max_results=20,
                )
            )
            conventions = await self._retrieval.retrieve(
                RetrieveRequest(
                    snapshot_id=req.snapshot_id,
                    query="coding conventions naming error handling dependency style",
                    section=RetrievalSection.CONVENTIONS,
                    mode=RetrievalMode.HYBRID,
                    max_results=20,
                )
            )
            feature_map = await self._retrieval.retrieve(
                RetrieveRequest(
                    snapshot_id=req.snapshot_id,
                    query="feature map functionality modules services data flow",
                    section=RetrievalSection.FEATURE_MAP,
                    mode=RetrievalMode.HYBRID,
                    max_results=20,
                )
            )
            important = await self._retrieval.retrieve(
                RetrieveRequest(
                    snapshot_id=req.snapshot_id,
                    query="important files entrypoint backbone central files",
                    section=RetrievalSection.IMPORTANT_FILES,
                    mode=RetrievalMode.VECTORLESS,
                    max_results=20,
                )
            )
            risk = await self._retrieval.retrieve(
                RetrieveRequest(
                    snapshot_id=req.snapshot_id,
                    query="risk complexity hotspot TODO FIXME circular import",
                    section=RetrievalSection.IMPORTANT_FILES,
                    mode=RetrievalMode.VECTORLESS,
                    max_results=20,
                )
            )
            if await _cancelled():
                return
            await self._jobs.update_step(job_id, StepName.GENERATE.value, 70, "Running Haystack agent pipeline")
            report = self._agents.run(
                architecture=architecture,
                important=important,
                conventions=conventions,
                feature_map=feature_map,
                risk=risk,
            )
            if await _cancelled():
                return
            await self._jobs.update_step(job_id, StepName.GENERATE.value, 100, "Sections generated")

            await self._jobs.update_step(job_id, StepName.EXPORT.value, 60, "Assembling report artifact")
            logger.info(f"[analysis:{job_id}] report_sections={len(report.get('sections', []))}")
            await get_db().execute(
                """
                INSERT INTO analysis_reports
                (id, job_id, repo_id, snapshot_id, provider_id, model_id, scan_mode, privacy_mode, report_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_id(),
                    job_id,
                    req.repo_id,
                    req.snapshot_id,
                    req.provider_id,
                    req.model_id,
                    req.scan_mode.value,
                    req.privacy_mode.value,
                    json.dumps(report),
                    utc_now_iso(),
                ),
            )
            await get_db().commit()
            await asyncio.sleep(0.08)
            await self._jobs.update_step(job_id, StepName.EXPORT.value, 100, "Report assembled")

            await self._jobs.finish(job_id)
        except Exception as e:
            await self._jobs.fail(job_id, str(e))

