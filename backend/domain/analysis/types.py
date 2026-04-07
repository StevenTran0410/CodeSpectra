"""Analysis runtime types (RPA-035)."""

from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any

from pydantic import BaseModel

# Callback: section_label, status ("done"|"error"), duration_ms,
# data (None on error), error (None on success).
SectionDoneCallback = Callable[
    [str, str, int, dict[str, Any] | None, str | None],
    Awaitable[None],
]


class ScanMode(str, Enum):
    QUICK = "quick"
    FULL = "full"


class PrivacyMode(str, Enum):
    STRICT_LOCAL = "strict_local"
    BYOK_CLOUD = "byok_cloud"


class StartAnalysisRequest(BaseModel):
    repo_id: str
    snapshot_id: str
    scan_mode: ScanMode = ScanMode.QUICK
    privacy_mode: PrivacyMode = PrivacyMode.STRICT_LOCAL
    provider_id: str
    model_id: str


class AnalysisEstimateResponse(BaseModel):
    repo_id: str
    snapshot_id: str
    file_count: int
    estimated_tokens: int


class AnalysisReportSummary(BaseModel):
    id: str
    job_id: str
    repo_id: str
    repo_name: str | None = None
    snapshot_id: str
    branch: str | None = None
    provider_id: str
    model_id: str
    scan_mode: ScanMode
    privacy_mode: PrivacyMode
    created_at: str


class AnalysisReport(BaseModel):
    summary: AnalysisReportSummary
    report: dict


class AnalysisReportMarkdownResponse(BaseModel):
    report_id: str
    default_name: str
    markdown: str
