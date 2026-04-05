"""Analysis runtime types (RPA-035)."""
from enum import Enum

from pydantic import BaseModel


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
    snapshot_id: str
    provider_id: str
    model_id: str
    scan_mode: ScanMode
    privacy_mode: PrivacyMode
    created_at: str


class AnalysisReport(BaseModel):
    summary: AnalysisReportSummary
    report: dict

