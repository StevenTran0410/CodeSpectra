"""Analysis runtime service (RPA-035)."""
from __future__ import annotations

import asyncio
import json
import re

from domain.job.service import JobService
from domain.job.types import CreateJobRequest, StepName
from domain.manifest.service import ManifestService
from domain.manifest.types import BuildManifestRequest
from domain.model_connector.service import ProviderConfigService
from domain.repo_map.service import RepoMapService
from domain.repo_map.types import BuildRepoMapRequest
from domain.retrieval.service import RetrievalService
from domain.retrieval.types import BuildRetrievalIndexRequest
from domain.structural_graph.service import StructuralGraphService
from domain.structural_graph.types import BuildGraphRequest
from infrastructure.db.database import get_db
from shared.errors import NotFoundError
from shared.logger import logger
from shared.utils import new_id, utc_now_iso

from .agent_pipeline import AnalysisAgentPipeline
from .orchestrator import RunDirectorAgent
from .types import (
    AnalysisEstimateResponse,
    AnalysisReport,
    AnalysisReportMarkdownResponse,
    AnalysisReportSummary,
    PrivacyMode,
    ScanMode,
    StartAnalysisRequest,
)

_bg_tasks: set[asyncio.Task] = set()


def _slug(s: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9._-]+", "-", (s or "").strip())
    out = out.strip("-._")
    return out or "report"


def _render_report_markdown(report: AnalysisReport) -> str:
    summary = report.summary
    sections = report.report.get("sections", []) if isinstance(report.report, dict) else []
    conf = report.report.get("confidence_summary", {}) if isinstance(report.report, dict) else {}

    lines: list[str] = [
        f"# CodeSpectra Analysis Report",
        "",
        f"- Report ID: `{summary.id}`",
        f"- Job ID: `{summary.job_id}`",
        f"- Repo: `{summary.repo_name or summary.repo_id}`",
        f"- Branch: `{summary.branch or 'unknown'}`",
        f"- Model: `{summary.model_id}`",
        f"- Scan mode: `{summary.scan_mode.value}`",
        f"- Privacy mode: `{summary.privacy_mode.value}`",
        f"- Created at: `{summary.created_at}`",
        "",
        "## Confidence Summary",
        "",
        f"- High: {int(conf.get('high', 0))}",
        f"- Medium: {int(conf.get('medium', 0))}",
        f"- Low: {int(conf.get('low', 0))}",
        "",
    ]

    for i, s in enumerate(sections, start=1):
        if not isinstance(s, dict):
            continue
        section_id = str(s.get("section", "")).strip() or f"section-{i}"
        confidence = str(s.get("confidence", "unknown")).strip()
        content = str(s.get("content", "")).strip()
        lines.extend(
            [
                f"## {i}. {section_id}",
                "",
                f"- Confidence: `{confidence}`",
                "",
                content if content else "_No content_",
                "",
            ]
        )

        details = s.get("details")
        if isinstance(details, dict) and details:
            lines.extend(["### Structured Details", ""])
            for k, v in details.items():
                lines.append(f"- **{k}**:")
                if isinstance(v, list):
                    for item in v[:20]:
                        lines.append(f"  - {item}")
                else:
                    lines.append(f"  - {v}")
            lines.append("")

        evidences = s.get("evidence_files", [])
        if isinstance(evidences, list) and evidences:
            lines.extend(["### Evidence Files", ""])
            for f in evidences[:30]:
                lines.append(f"- `{f}`")
            lines.append("")

        blind = s.get("blind_spots", [])
        if isinstance(blind, list) and blind:
            lines.extend(["### Blind Spots", ""])
            for b in blind[:20]:
                lines.append(f"- {b}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


class AnalysisService:
    def __init__(self) -> None:
        self._jobs = JobService()
        self._manifest = ManifestService()
        self._repo_map = RepoMapService()
        self._graph = StructuralGraphService()
        self._retrieval = RetrievalService()
        self._provider = ProviderConfigService()
        self._agents = AnalysisAgentPipeline(self._provider, self._retrieval)
        self._director = RunDirectorAgent(
            self._provider, self._retrieval, self._agents, self._graph
        )

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
                SELECT
                    ar.id, ar.job_id, ar.repo_id, ar.snapshot_id, ar.provider_id, ar.model_id,
                    ar.scan_mode, ar.privacy_mode, ar.created_at,
                    COALESCE(lr.name, ar.repo_id) as repo_name,
                    COALESCE(rs.branch, lr.selected_branch, lr.git_branch, 'unknown') as branch
                FROM analysis_reports ar
                LEFT JOIN local_repos lr ON lr.id = ar.repo_id
                LEFT JOIN repo_snapshots rs ON rs.id = ar.snapshot_id
                WHERE ar.repo_id=?
                ORDER BY ar.created_at DESC
                LIMIT ?
                """,
                (repo_id, safe_limit),
            ) as cur:
                rows = await cur.fetchall()
        else:
            async with db.execute(
                """
                SELECT
                    ar.id, ar.job_id, ar.repo_id, ar.snapshot_id, ar.provider_id, ar.model_id,
                    ar.scan_mode, ar.privacy_mode, ar.created_at,
                    COALESCE(lr.name, ar.repo_id) as repo_name,
                    COALESCE(rs.branch, lr.selected_branch, lr.git_branch, 'unknown') as branch
                FROM analysis_reports ar
                LEFT JOIN local_repos lr ON lr.id = ar.repo_id
                LEFT JOIN repo_snapshots rs ON rs.id = ar.snapshot_id
                ORDER BY ar.created_at DESC
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
                repo_name=r["repo_name"],
                snapshot_id=r["snapshot_id"],
                branch=r["branch"],
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
            SELECT
                ar.id, ar.job_id, ar.repo_id, ar.snapshot_id, ar.provider_id, ar.model_id,
                ar.scan_mode, ar.privacy_mode, ar.report_json, ar.created_at,
                COALESCE(lr.name, ar.repo_id) as repo_name,
                COALESCE(rs.branch, lr.selected_branch, lr.git_branch, 'unknown') as branch
            FROM analysis_reports ar
            LEFT JOIN local_repos lr ON lr.id = ar.repo_id
            LEFT JOIN repo_snapshots rs ON rs.id = ar.snapshot_id
            WHERE ar.id=?
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
                repo_name=row["repo_name"],
                snapshot_id=row["snapshot_id"],
                branch=row["branch"],
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

    async def delete_report(self, report_id: str) -> None:
        db = get_db()
        async with db.execute("DELETE FROM analysis_reports WHERE id=?", (report_id,)) as cur:
            if cur.rowcount == 0:
                raise NotFoundError("AnalysisReport", report_id)
        await db.commit()

    async def export_report_markdown(self, report_id: str) -> AnalysisReportMarkdownResponse:
        rep = await self.get_report(report_id)
        default_name = f"codespectra-report-{_slug(rep.summary.repo_name or rep.summary.repo_id)}-{report_id[:8]}.md"
        return AnalysisReportMarkdownResponse(
            report_id=report_id,
            default_name=default_name,
            markdown=_render_report_markdown(rep),
        )

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

            await self._jobs.update_step(job_id, StepName.GENERATE.value, 30, "Director planning + broker retrieval")
            if await _cancelled():
                return
            await self._jobs.update_step(job_id, StepName.GENERATE.value, 70, "Running LLM power agents")
            report = await self._director.run(
                provider_id=req.provider_id,
                model_id=req.model_id,
                snapshot_id=req.snapshot_id,
                scan_mode=req.scan_mode.value,
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

