"""Pydantic models for the GitHub integration domain."""
from pydantic import BaseModel


class DeviceFlowStart(BaseModel):
    device_code: str
    user_code: str
    verification_uri: str
    expires_in: int
    interval: int


class DeviceFlowPollResult(BaseModel):
    # "pending" | "success" | "expired" | "denied" | "slow_down" | "error"
    status: str
    # returned only when status == "success"
    account: "GitHubAccount | None" = None


class GitHubAccount(BaseModel):
    id: str
    login: str
    display_name: str | None
    avatar_url: str | None
    created_at: str
    updated_at: str


class GitHubRepo(BaseModel):
    id: int
    full_name: str
    name: str
    owner_login: str
    is_private: bool
    description: str | None
    default_branch: str
    html_url: str
    ssh_url: str
    clone_url: str
    updated_at: str


class GitHubRepoListResponse(BaseModel):
    repos: list[GitHubRepo]
    page: int
    has_more: bool
