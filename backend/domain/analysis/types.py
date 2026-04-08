"""Analysis runtime types (RPA-035)."""

from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

# Callback: section_label, status ("done"|"error"), duration_ms,
# data (None on error), error (None on success).
SectionDoneCallback = Callable[
    [str, str, int, dict[str, Any] | None, str | None],
    Awaitable[None],
]


class ScanMode(StrEnum):
    QUICK = "quick"
    FULL = "full"


class PrivacyMode(StrEnum):
    STRICT_LOCAL = "strict_local"
    BYOK_CLOUD = "byok_cloud"


class StartAnalysisRequest(BaseModel):
    repo_id: str
    snapshot_id: str
    scan_mode: ScanMode = ScanMode.QUICK
    privacy_mode: PrivacyMode = PrivacyMode.STRICT_LOCAL
    provider_id: str
    model_id: str


class ModelWarning(BaseModel):
    code: str
    message: str
    severity: str


class StartAnalysisResponse(BaseModel):
    job_id: str
    warning: ModelWarning | None = None


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


class AuditExportResponse(BaseModel):
    report_id: str
    default_name: str
    markdown: str


class RerunSectionRequest(BaseModel):
    report_id: str
    section: str
    provider_id: str
    model_id: str


class RerunSectionResponse(BaseModel):
    section: str
    data: dict[str, Any]
    duration_ms: int


class CompareRequest(BaseModel):
    report_id_a: str
    report_id_b: str


class ReportDiffResponse(BaseModel):
    report_id_a: str
    report_id_b: str
    quality_trend: str
    sections_changed: int
    identical: bool
    section_diffs: dict[str, dict[str, Any]]
