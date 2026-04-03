"""JobService — create, update, cancel and query analysis jobs."""
import asyncio
import json

from infrastructure.db.database import get_db
from shared.errors import NotFoundError
from shared.logger import logger
from shared.utils import new_id, utc_now_iso

from .types import (
    CreateJobRequest,
    Job,
    JobStatus,
    StepState,
    StepStatus,
)

# in-process cancel signals keyed by job_id
_cancel_events: dict[str, asyncio.Event] = {}


def _row_to_job(row) -> Job:
    return Job(
        id=row["id"],
        type=row["type"],
        repo_id=row["repo_id"],
        status=JobStatus(row["status"]),
        steps=_decode_steps(row["steps"]),
        current_step=row["current_step"],
        error=row["error"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
    )


def _decode_steps(raw: str) -> dict[str, StepState]:
    data = json.loads(raw or "{}")
    return {k: StepState(**v) for k, v in data.items()}


def _encode_steps(steps: dict[str, StepState]) -> str:
    return json.dumps({k: v.model_dump() for k, v in steps.items()})


class JobService:
    async def create(self, req: CreateJobRequest) -> Job:
        job_id = new_id()
        now = utc_now_iso()
        initial_steps = {s: StepState() for s in req.steps}

        await get_db().execute(
            """INSERT INTO jobs (id, type, repo_id, status, steps, current_step,
               error, started_at, finished_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (job_id, req.type, req.repo_id, JobStatus.PENDING.value,
             _encode_steps(initial_steps), None, None, now, None),
        )
        await get_db().commit()
        _cancel_events[job_id] = asyncio.Event()
        logger.info(f"Job {job_id} created ({req.type})")
        return await self.get(job_id)

    async def start(self, job_id: str) -> Job:
        await get_db().execute(
            "UPDATE jobs SET status=? WHERE id=?",
            (JobStatus.RUNNING.value, job_id),
        )
        await get_db().commit()
        return await self.get(job_id)

    async def update_step(
        self,
        job_id: str,
        step: str,
        progress: int,
        message: str | None = None,
    ) -> None:
        """Update a single step's progress. Marks it running automatically."""
        job = await self.get(job_id)
        steps = job.steps
        steps[step] = StepState(
            status=StepStatus.RUNNING if progress < 100 else StepStatus.DONE,
            progress=progress,
            message=message,
        )
        await get_db().execute(
            "UPDATE jobs SET steps=?, current_step=? WHERE id=?",
            (_encode_steps(steps), step, job_id),
        )
        await get_db().commit()

    async def finish(self, job_id: str) -> Job:
        now = utc_now_iso()
        await get_db().execute(
            "UPDATE jobs SET status=?, finished_at=?, current_step=? WHERE id=?",
            (JobStatus.DONE.value, now, None, job_id),
        )
        await get_db().commit()
        _cancel_events.pop(job_id, None)
        logger.info(f"Job {job_id} done")
        return await self.get(job_id)

    async def fail(self, job_id: str, error: str) -> Job:
        now = utc_now_iso()
        await get_db().execute(
            "UPDATE jobs SET status=?, error=?, finished_at=? WHERE id=?",
            (JobStatus.FAILED.value, error, now, job_id),
        )
        await get_db().commit()
        _cancel_events.pop(job_id, None)
        logger.error(f"Job {job_id} failed: {error}")
        return await self.get(job_id)

    async def cancel(self, job_id: str) -> Job:
        event = _cancel_events.get(job_id)
        if event:
            event.set()   # signal the running task to stop
        now = utc_now_iso()
        await get_db().execute(
            "UPDATE jobs SET status=?, finished_at=? WHERE id=?",
            (JobStatus.CANCELLED.value, now, job_id),
        )
        await get_db().commit()
        _cancel_events.pop(job_id, None)
        logger.info(f"Job {job_id} cancelled")
        return await self.get(job_id)

    def is_cancelled(self, job_id: str) -> bool:
        """Check from inside a running task whether cancellation was requested."""
        event = _cancel_events.get(job_id)
        return event.is_set() if event else False

    async def get(self, job_id: str) -> Job:
        async with get_db().execute(
            "SELECT * FROM jobs WHERE id=?", (job_id,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            raise NotFoundError("Job", job_id)
        return _row_to_job(row)

    async def list_for_repo(self, repo_id: str, limit: int = 20) -> list[Job]:
        async with get_db().execute(
            "SELECT * FROM jobs WHERE repo_id=? ORDER BY started_at DESC LIMIT ?",
            (repo_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [_row_to_job(r) for r in rows]

    async def list_recent(self, limit: int = 20) -> list[Job]:
        async with get_db().execute(
            "SELECT * FROM jobs ORDER BY started_at DESC LIMIT ?", (limit,)
        ) as cur:
            rows = await cur.fetchall()
        return [_row_to_job(r) for r in rows]
