"""Types for local folder repositories."""
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class RepoSourceType(str, Enum):
    GITHUB = "github"
    BITBUCKET = "bitbucket"
    LOCAL_FOLDER = "local_folder"


class SyncMode(str, Enum):
    LATEST = "latest"   # always pull latest on selected branch
    PINNED = "pinned"   # lock to pinned ref / commit SHA


class LocalRepo(BaseModel):
    """A local folder tracked as a repository source."""

    id: str
    workspace_id: str | None = None
    path: str
    name: str
    source_type: RepoSourceType = RepoSourceType.LOCAL_FOLDER
    is_git_repo: bool
    git_branch: str | None        # actual HEAD branch at last validation
    git_head_hash: str | None
    git_remote_url: str | None
    has_size_warning: bool
    selected_branch: str | None   # user-chosen branch to analyze (None = use HEAD)
    active_snapshot_id: str | None = None
    sync_mode: SyncMode = SyncMode.LATEST
    pinned_ref: str | None
    ignore_overrides: list[str] = Field(default_factory=list)
    detect_submodules: bool = True
    added_at: str
    last_validated_at: str


class SetBranchRequest(BaseModel):
    branch: str

    @field_validator("branch")
    @classmethod
    def branch_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("branch cannot be empty")
        return v


class SetActiveSnapshotRequest(BaseModel):
    snapshot_id: str | None = None


class UpdateRepoSettingsRequest(BaseModel):
    sync_mode: SyncMode = SyncMode.LATEST
    pinned_ref: str | None = None
    ignore_overrides: list[str] = Field(default_factory=list)
    detect_submodules: bool = True

    @field_validator("pinned_ref")
    @classmethod
    def normalize_pinned_ref(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        return v or None

    @field_validator("ignore_overrides")
    @classmethod
    def normalize_ignore_overrides(cls, v: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in v:
            p = item.strip()
            if not p or p in seen:
                continue
            out.append(p)
            seen.add(p)
        return out


class ValidateFolderRequest(BaseModel):
    path: str

    @field_validator("path")
    @classmethod
    def path_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("path cannot be empty")
        return v


class ValidateFolderResponse(BaseModel):
    """Result of validating a folder path — not yet persisted."""

    path: str
    name: str
    exists: bool
    is_directory: bool
    is_git_repo: bool
    git_branch: str | None
    git_head_hash: str | None
    git_remote_url: str | None
    has_size_warning: bool
    size_warning_reason: str | None


class AddLocalRepoRequest(BaseModel):
    path: str
    workspace_id: str | None = None

    @field_validator("path")
    @classmethod
    def path_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("path cannot be empty")
        return v


class CloneFromUrlRequest(BaseModel):
    url: str
    dest_path: str  # absolute path where the repo should be cloned
    workspace_id: str | None = None

    @field_validator("url")
    @classmethod
    def url_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("url cannot be empty")
        return v

    @field_validator("dest_path")
    @classmethod
    def dest_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("dest_path cannot be empty")
        return v


class EstimateFileCountResponse(BaseModel):
    estimated_file_count: int
    workspace_default_ignores: list[str]
    repo_ignore_overrides: list[str]
    effective_ignores: list[str]
