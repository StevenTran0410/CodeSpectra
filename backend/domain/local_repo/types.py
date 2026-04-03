"""Types for local folder repositories."""
from enum import Enum

from pydantic import BaseModel, field_validator


class RepoSourceType(str, Enum):
    GITHUB = "github"
    BITBUCKET = "bitbucket"
    LOCAL_FOLDER = "local_folder"


class LocalRepo(BaseModel):
    """A local folder tracked as a repository source."""

    id: str
    path: str
    name: str
    source_type: RepoSourceType = RepoSourceType.LOCAL_FOLDER
    is_git_repo: bool
    git_branch: str | None        # actual HEAD branch at last validation
    git_head_hash: str | None
    git_remote_url: str | None
    has_size_warning: bool
    selected_branch: str | None   # user-chosen branch to analyze (None = use HEAD)
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
