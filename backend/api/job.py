"""Job endpoints — create, query, cancel."""
from fastapi import APIRouter, HTTPException

from domain.job.service import JobService
from domain.job.types import CreateJobRequest, Job

router = APIRouter(tags=["job"])
_service = JobService()


@router.post("/", response_model=Job, status_code=201)
async def create_job(body: CreateJobRequest) -> Job:
    return await _service.create(body)


@router.get("/{job_id}", response_model=Job)
async def get_job(job_id: str) -> Job:
    return await _service.get(job_id)


@router.post("/{job_id}/cancel", response_model=Job)
async def cancel_job(job_id: str) -> Job:
    try:
        return await _service.cancel(job_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/repo/{repo_id}", response_model=list[Job])
async def list_jobs_for_repo(repo_id: str, limit: int = 20) -> list[Job]:
    return await _service.list_for_repo(repo_id, limit)


@router.get("/", response_model=list[Job])
async def list_recent_jobs(limit: int = 20) -> list[Job]:
    return await _service.list_recent(limit)
