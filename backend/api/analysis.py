"""Analysis run endpoints."""
from fastapi import APIRouter, Query

from domain.analysis.service import AnalysisService
from shared.http_utils import handle_value_error
from domain.analysis.types import (
    AnalysisEstimateResponse,
    AnalysisReport,
    AnalysisReportMarkdownResponse,
    AnalysisReportSummary,
    AuditExportResponse,
    CompareRequest,
    ReportDiffResponse,
    RerunSectionRequest,
    RerunSectionResponse,
    SectionSourcesResponse,
    StalenessResult,
    StartAnalysisRequest,
    StartAnalysisResponse,
)

router = APIRouter(tags=["analysis"])
_service = AnalysisService()


@router.get("/estimate/{repo_id}/{snapshot_id}", response_model=AnalysisEstimateResponse)
async def estimate_scope(repo_id: str, snapshot_id: str) -> AnalysisEstimateResponse:
    return await _service.estimate(repo_id=repo_id, snapshot_id=snapshot_id)


@router.post("/start", response_model=StartAnalysisResponse, status_code=201)
@handle_value_error
async def start_analysis(body: StartAnalysisRequest) -> StartAnalysisResponse:
    return await _service.start(body)


@router.get("/events/{job_id}")
async def get_section_events(
    job_id: str,
    from_idx: int = Query(default=0, ge=0),
) -> dict[str, object]:
    events, job_done = _service.get_section_events(job_id, from_idx)
    return {"events": events, "job_done": job_done}


@router.get("/reports", response_model=list[AnalysisReportSummary])
async def list_reports(
    repo_id: str | None = None,
    workspace_id: str | None = None,
    limit: int = 30,
) -> list[AnalysisReportSummary]:
    return await _service.list_reports(repo_id=repo_id, workspace_id=workspace_id, limit=limit)


@router.get("/reports/{report_id}", response_model=AnalysisReport)
async def get_report(report_id: str) -> AnalysisReport:
    return await _service.get_report(report_id)


@router.get("/report-by-job/{job_id}", response_model=AnalysisReport)
async def get_report_by_job(job_id: str) -> AnalysisReport:
    return await _service.get_report_by_job(job_id)


@router.delete("/reports/{report_id}", status_code=204)
async def delete_report(report_id: str) -> None:
    return await _service.delete_report(report_id)


@router.get("/reports/{report_id}/markdown", response_model=AnalysisReportMarkdownResponse)
async def export_report_markdown(report_id: str) -> AnalysisReportMarkdownResponse:
    return await _service.export_report_markdown(report_id)


@router.get("/reports/{report_id}/export/audit", response_model=AuditExportResponse)
@handle_value_error
async def export_audit_section(report_id: str) -> AuditExportResponse:
    return await _service.export_audit_section(report_id)


@router.post("/rerun_section", response_model=RerunSectionResponse)
@handle_value_error
async def rerun_section(body: RerunSectionRequest) -> RerunSectionResponse:
    return await _service.rerun_section(
        body.report_id,
        body.section,
        body.provider_id,
        body.model_id,
    )


@router.post("/compare", response_model=ReportDiffResponse)
@handle_value_error
async def compare_reports_endpoint(body: CompareRequest) -> ReportDiffResponse:
    return await _service.compare_reports(body.report_id_a, body.report_id_b)


@router.get("/reports/{report_id}/sections/{section_id}/sources", response_model=SectionSourcesResponse)
async def get_section_sources(report_id: str, section_id: str) -> SectionSourcesResponse:
    return await _service.get_section_sources(report_id, section_id)


@router.get("/staleness", response_model=StalenessResult)
async def get_staleness(report_id: str = Query(...)) -> StalenessResult:
    return await _service.check_staleness(report_id)
