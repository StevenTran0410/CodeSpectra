"""Analysis run endpoints (RPA-035)."""
from fastapi import APIRouter, HTTPException

from domain.analysis.service import AnalysisService
from domain.analysis.types import (
    AnalysisEstimateResponse,
    AnalysisReport,
    AnalysisReportSummary,
    StartAnalysisRequest,
)
from domain.job.types import Job

router = APIRouter(tags=["analysis"])
_service = AnalysisService()


@router.get("/estimate/{repo_id}/{snapshot_id}", response_model=AnalysisEstimateResponse)
async def estimate_scope(repo_id: str, snapshot_id: str) -> AnalysisEstimateResponse:
    return await _service.estimate(repo_id=repo_id, snapshot_id=snapshot_id)


@router.post("/start", response_model=Job, status_code=201)
async def start_analysis(body: StartAnalysisRequest) -> Job:
    try:
        return await _service.start(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/reports", response_model=list[AnalysisReportSummary])
async def list_reports(repo_id: str | None = None, limit: int = 30) -> list[AnalysisReportSummary]:
    return await _service.list_reports(repo_id=repo_id, limit=limit)


@router.get("/reports/{report_id}", response_model=AnalysisReport)
async def get_report(report_id: str) -> AnalysisReport:
    return await _service.get_report(report_id)


@router.get("/report-by-job/{job_id}", response_model=AnalysisReport)
async def get_report_by_job(job_id: str) -> AnalysisReport:
    return await _service.get_report_by_job(job_id)
