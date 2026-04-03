"""Job domain types — shared across the entire analysis pipeline."""
from enum import Enum

from pydantic import BaseModel


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class StepName(str, Enum):
    """Canonical step names used across all analysis pipeline tickets."""
    CLONE = "clone"
    SYNC = "sync"
    MANIFEST = "manifest"
    PARSE = "parse"
    GRAPH = "graph"
    EMBED = "embed"
    GENERATE = "generate"
    EXPORT = "export"


class StepState(BaseModel):
    status: StepStatus = StepStatus.PENDING
    progress: int = 0        # 0-100
    message: str | None = None


class Job(BaseModel):
    id: str
    type: str                # e.g. "analysis"
    repo_id: str | None
    status: JobStatus
    steps: dict[str, StepState]   # StepName → StepState
    current_step: str | None
    error: str | None
    started_at: str
    finished_at: str | None


class CreateJobRequest(BaseModel):
    type: str
    repo_id: str | None = None
    steps: list[str]         # ordered list of StepName values
