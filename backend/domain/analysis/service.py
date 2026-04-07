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


def _section_footer(s: dict, lines: list[str]) -> None:
    """Append evidence files + blind spots footer shared by all sections."""
    evidences = s.get("evidence_files", [])
    if isinstance(evidences, list) and evidences:
        lines.append("")
        lines.append("**Evidence files:** " + ", ".join(f"`{f}`" for f in evidences[:20]))
    blind = s.get("blind_spots", [])
    if isinstance(blind, list) and blind:
        lines.append("")
        lines.append("**Blind spots:** " + "; ".join(str(b) for b in blind[:10]))
    conf = s.get("confidence", "")
    if conf:
        lines.append(f"\n_Confidence: {conf}_")


def _render_section_a(s: dict, lines: list[str]) -> None:
    lines += [
        "## A — Project Identity Card", "",
        f"**Repo:** {s.get('repo_name', '')}  ",
        f"**Domain:** {s.get('domain', '')}  ",
        f"**Runtime type:** `{s.get('runtime_type', '')}`  ",
        "",
        f"**Purpose:** {s.get('purpose', '')}",
        "",
        f"**Business context:** {s.get('business_context', '')}",
        "",
    ]
    stack = s.get("tech_stack", [])
    if stack:
        lines.append("**Tech stack:** " + ", ".join(f"`{t}`" for t in stack))
    _section_footer(s, lines)
    lines.append("")


def _render_section_g(s: dict, lines: list[str]) -> None:
    lines += ["## G — Important Files Radar", ""]
    slots = ["entrypoint", "backbone", "critical_config", "highest_centrality",
             "most_dangerous_to_touch", "read_first"]
    labels = {
        "entrypoint": "Entrypoint",
        "backbone": "Backbone",
        "critical_config": "Critical config",
        "highest_centrality": "Highest centrality",
        "most_dangerous_to_touch": "Most dangerous to touch",
        "read_first": "Read first",
    }
    for slot in slots:
        v = s.get(slot)
        if isinstance(v, dict):
            lines.append(f"- **{labels[slot]}:** `{v.get('file', '')}` — {v.get('reason', '')}")
    other = s.get("other_important", [])
    if isinstance(other, list) and other:
        lines.append("")
        lines.append("**Other important files:**")
        for item in other[:10]:
            if isinstance(item, dict):
                lines.append(f"  - `{item.get('file', '')}` — {item.get('reason', '')}")
    _section_footer(s, lines)
    lines.append("")


def _render_section_i(s: dict, lines: list[str]) -> None:
    lines += ["## I — Glossary / Domain Terms", ""]
    terms = s.get("terms", [])
    if isinstance(terms, list) and terms:
        lines.append("| Term | Definition | Evidence |")
        lines.append("|---|---|---|")
        for t in terms[:40]:
            if not isinstance(t, dict):
                continue
            evs = t.get("evidence_files", [])
            ev_str = ", ".join(f"`{e}`" for e in evs[:2]) if isinstance(evs, list) else ""
            lines.append(f"| **{t.get('term', '')}** | {t.get('definition', '')} | {ev_str} |")
    _section_footer(s, lines)
    lines.append("")


def _render_section_j(s: dict, lines: list[str]) -> None:
    lines += ["## J — Risk / Complexity / Unknowns", ""]
    summary = s.get("summary", "")
    if summary:
        lines += [summary, ""]
    findings = s.get("findings", [])
    if isinstance(findings, list) and findings:
        for sev in ("high", "medium", "low"):
            group = [f for f in findings if isinstance(f, dict) and f.get("severity") == sev]
            if not group:
                continue
            lines.append(f"### {sev.upper()} severity ({len(group)})")
            for f in group:
                lines.append(f"- **[{f.get('category', '')}]** {f.get('title', '')}")
                rationale = f.get("rationale", "")
                if rationale:
                    lines.append(f"  {rationale}")
                ev = f.get("evidence", [])
                if isinstance(ev, list) and ev:
                    lines.append("  Evidence: " + ", ".join(f"`{e}`" for e in ev[:3]))
            lines.append("")
    _section_footer(s, lines)
    lines.append("")


_SECTION_RENDERERS = {
    "A": _render_section_a,
    "G": _render_section_g,
    "I": _render_section_i,
    "J": _render_section_j,
}

_SECTION_ORDER = ["A", "G", "I", "J", "B", "C", "D", "E", "F", "H", "K"]


def _render_report_markdown(report: AnalysisReport) -> str:
    summary = report.summary
    raw = report.report if isinstance(report.report, dict) else {}

    lines: list[str] = [
        "# CodeSpectra Analysis Report",
        "",
        f"- **Report ID:** `{summary.id}`",
        f"- **Repo:** {summary.repo_name or summary.repo_id}",
        f"- **Branch:** {summary.branch or 'unknown'}",
        f"- **Model:** `{summary.model_id}`",
        f"- **Scan mode:** {summary.scan_mode.value}",
        f"- **Created at:** {summary.created_at}",
        "",
    ]

    sections_v2 = raw.get("sections_v2")
    if isinstance(sections_v2, dict) and sections_v2:
        for letter in _SECTION_ORDER:
            sec = sections_v2.get(letter)
            if not isinstance(sec, dict) or "error" in sec:
                continue
            renderer = _SECTION_RENDERERS.get(letter)
            if renderer:
                renderer(sec, lines)
            else:
                # Generic fallback for sections without a dedicated renderer (B–K added later)
                lines.append(f"## {letter} — (section)")
                lines.append("")
                for k, v in sec.items():
                    if k in ("confidence", "evidence_files", "blind_spots"):
                        continue
                    if isinstance(v, list):
                        lines.append(f"**{k}:** " + ", ".join(str(i) for i in v[:20]))
                    else:
                        lines.append(f"**{k}:** {v}")
                _section_footer(sec, lines)
                lines.append("")
    else:
        lines.append("_No analysis section data found. Re-run analysis to generate sections._")
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
        async with get_db().execute("SELECT name FROM local_repos WHERE id=?", (req.repo_id,)) as cur:
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

        repo_name = str(repo["name"] or "") if repo else ""
        task = asyncio.create_task(self._run(job.id, req, repo_name))
        _bg_tasks.add(task)
        task.add_done_callback(lambda t: _bg_tasks.discard(t))
        return job

    async def _run(self, job_id: str, req: StartAnalysisRequest, repo_name: str = "") -> None:
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

            await self._jobs.update_step(job_id, StepName.GENERATE.value, 30, "Running analysis agents")
            if await _cancelled():
                return
            await self._jobs.update_step(job_id, StepName.GENERATE.value, 70, "Running LLM power agents")
            report = await self._director.run(
                provider_id=req.provider_id,
                model_id=req.model_id,
                snapshot_id=req.snapshot_id,
                scan_mode=req.scan_mode.value,
                repo_name=repo_name,
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

