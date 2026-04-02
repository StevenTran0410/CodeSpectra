"""FastAPI router for GitHub OAuth device flow and repository discovery."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from domain.github_host.service import GitHubService
from domain.github_host.types import (
    DeviceFlowPollResult,
    DeviceFlowStart,
    GitHubAccount,
    GitHubRepoListResponse,
)

router = APIRouter(tags=["github"])
_svc = GitHubService()


# ──────────────────────────────────────────────────────────────────────────────
# Request models
# ──────────────────────────────────────────────────────────────────────────────

class PollRequest(BaseModel):
    device_code: str


class ListReposRequest(BaseModel):
    query: str | None = None
    page: int = 1


# ──────────────────────────────────────────────────────────────────────────────
# OAuth Device Flow
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/device-flow/start", response_model=DeviceFlowStart)
async def start_device_flow() -> DeviceFlowStart:
    """Begin GitHub OAuth device flow. Returns user_code + verification_uri to show the user."""
    try:
        return await _svc.start_device_flow()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/device-flow/poll", response_model=DeviceFlowPollResult)
async def poll_device_flow(body: PollRequest) -> DeviceFlowPollResult:
    """Poll GitHub to check if the user has authorised.

    Call this once per polling interval (returned by start_device_flow).
    - status "pending"   → user has not yet authorized; poll again
    - status "slow_down" → increase polling interval by 5 s and retry
    - status "success"   → token stored; account field is populated
    - status "expired"   → device code expired; restart the flow
    - status "denied"    → user cancelled; restart the flow
    """
    try:
        return await _svc.poll_device_flow(body.device_code)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ──────────────────────────────────────────────────────────────────────────────
# Account management
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/account", response_model=GitHubAccount | None)
async def get_account():
    """Return the connected GitHub account, or null if not connected."""
    return await _svc.get_account()


@router.delete("/account", status_code=204)
async def disconnect_account() -> None:
    """Remove the stored GitHub token and disconnect the account."""
    await _svc.disconnect()


# ──────────────────────────────────────────────────────────────────────────────
# Repository discovery
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/repos", response_model=GitHubRepoListResponse)
async def list_repos(query: str | None = None, page: int = 1) -> GitHubRepoListResponse:
    """List repositories the connected user has access to, with optional search."""
    from shared.errors import NotFoundError
    try:
        return await _svc.list_repos(query=query, page=page)
    except NotFoundError:
        raise HTTPException(status_code=401, detail="No GitHub account connected")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"GitHub API error: {e}")
