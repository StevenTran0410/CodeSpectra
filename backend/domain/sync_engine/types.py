"""Sync engine types — RepoSnapshot and related models."""
from enum import Enum

from pydantic import BaseModel


class SnapshotStatus(str, Enum):
    PENDING  = "pending"
    SYNCING  = "syncing"
    READY    = "ready"
    FAILED   = "failed"


class ClonePolicy(str, Enum):
    FULL    = "full"     # standard git clone
    SHALLOW = "shallow"  # --depth=1  (faster, less history)
    PARTIAL = "partial"  # --filter=blob:none  (blobless, metadata only)


class RepoSnapshot(BaseModel):
    id: str
    local_repo_id: str
    branch: str | None
    commit_hash: str | None
    local_path: str
    status: SnapshotStatus
    error: str | None
    clone_policy: ClonePolicy
    manual_ignores: list[str] = []
    synced_at: str
    created_at: str


class PrepareSnapshotRequest(BaseModel):
    local_repo_id: str
    branch: str | None = None                          # None = use repo's selected_branch or HEAD
    clone_policy: ClonePolicy = ClonePolicy.FULL
