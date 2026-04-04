"""SyncEngineService — clone, sync, and snapshot management."""
import asyncio
import json
import os
import subprocess
from pathlib import Path

from infrastructure.db.database import get_db
from shared.errors import NotFoundError
from shared.git_utils import is_ssh_url, read_git_info, run_git_sync
from shared.logger import logger
from shared.utils import new_id, utc_now_iso

from . import lock as path_lock
from .types import ClonePolicy, PrepareSnapshotRequest, RepoSnapshot, SnapshotStatus

_BACKGROUND_SYNC_TASKS: set[asyncio.Task] = set()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

async def _get_ssh_env() -> dict:
    """Build env dict with GIT_SSH_COMMAND if an SSH key is configured."""
    env = os.environ.copy()
    async with get_db().execute(
        "SELECT value FROM app_metadata WHERE key='git_ssh_key_path'"
    ) as cur:
        row = await cur.fetchone()
    if row and row["value"]:
        env["GIT_SSH_COMMAND"] = (
            f'ssh -i "{row["value"]}" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new'
        )
    return env


def _run_clone(url: str, dest: str, policy: ClonePolicy, env: dict) -> subprocess.CompletedProcess:
    args = ["git", "clone"]
    if policy == ClonePolicy.SHALLOW:
        args += ["--depth=1"]
    elif policy == ClonePolicy.PARTIAL:
        args += ["--filter=blob:none", "--no-checkout"]
    args += [url, dest]
    return subprocess.run(args, capture_output=True, text=True, timeout=600, env=env)


def _run_sync(path: str, branch: str, env: dict) -> tuple[bool, str]:
    """Fetch + checkout + pull. Returns (ok, error_message)."""
    if run_git_sync(path, ["fetch", "origin"], env=env, timeout=120) is None:
        # fetch failed but may still work offline — just warn
        logger.warning(f"git fetch failed for {path}, trying local checkout")

    current_branch = run_git_sync(path, ["rev-parse", "--abbrev-ref", "HEAD"], env=env, timeout=10)
    # Skip checkout if already on target branch
    if current_branch != branch:
        result = subprocess.run(
            ["git", "checkout", branch],
            cwd=path, capture_output=True, text=True, timeout=30, env=env
        )
        if result.returncode != 0:
            return False, result.stderr.strip()

    result = subprocess.run(
        ["git", "pull", "origin", branch],
        cwd=path, capture_output=True, text=True, timeout=120, env=env
    )
    if result.returncode != 0:
        return False, result.stderr.strip()

    return True, ""


# ──────────────────────────────────────────────────────────────────────────────
# Preflight
# ──────────────────────────────────────────────────────────────────────────────

def _dirty_files(path: str) -> list[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=path, capture_output=True, text=True, timeout=10
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []
    files: list[str] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if len(line) > 3:
            files.append(line[3:])
    return files


async def _preflight(path: str, target_branch: str | None) -> None:
    """Raise ValueError with a user-friendly message if preconditions fail."""
    p = Path(path)

    # 1. Parent must be writable (for clone) or path must exist and be writable (for sync)
    check_dir = p if p.exists() else p.parent
    if not os.access(str(check_dir), os.W_OK):
        raise ValueError(f"No write permission to '{check_dir}'")

    if not p.exists():
        return  # nothing more to check — clone will create it

    # 2. Dirty working tree check
    dirty = _dirty_files(str(p))
    if dirty:
        current_branch = run_git_sync(str(p), ["rev-parse", "--abbrev-ref", "HEAD"], timeout=10)
        # Block only when branch switch is needed; same-branch sync can proceed
        if target_branch and current_branch and target_branch != current_branch:
            preview = "\n".join(dirty[:5])
            more = f"\n...and {len(dirty)-5} more file(s)" if len(dirty) > 5 else ""
            raise ValueError(
                "Cannot switch branch because the repository has uncommitted changes.\n"
                "Commit or stash these files first:\n"
                f"{preview}{more}"
            )
        logger.warning(f"Repo at {path} has uncommitted changes — syncing current branch only")

    # 3. Submodule detection (warn only)
    if (p / ".gitmodules").exists():
        logger.warning(f"Repo at {path} has submodules — they won't be synced automatically")


# ──────────────────────────────────────────────────────────────────────────────
# DB helpers
# ──────────────────────────────────────────────────────────────────────────────

def _row_to_snapshot(row) -> RepoSnapshot:
    try:
        manual_ignores = json.loads(row["manual_ignores"] or "[]")
    except Exception:
        manual_ignores = []
    return RepoSnapshot(
        id=row["id"],
        local_repo_id=row["local_repo_id"],
        branch=row["branch"],
        commit_hash=row["commit_hash"],
        local_path=row["local_path"],
        status=SnapshotStatus(row["status"]),
        error=row["error"],
        clone_policy=ClonePolicy(row["clone_policy"]),
        manual_ignores=manual_ignores,
        synced_at=row["synced_at"],
        created_at=row["created_at"],
    )


async def _create_record(
    local_repo_id: str, branch: str | None, local_path: str, policy: ClonePolicy
) -> RepoSnapshot:
    snap_id = new_id()
    now = utc_now_iso()
    await get_db().execute(
        """INSERT INTO repo_snapshots
           (id, local_repo_id, branch, commit_hash, local_path, status, error,
            clone_policy, manual_ignores, synced_at, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (snap_id, local_repo_id, branch, None, local_path,
         SnapshotStatus.PENDING.value, None, policy.value, "[]", now, now),
    )
    await get_db().commit()
    return await SyncEngineService().get_snapshot(snap_id)


async def _update_status(snap_id: str, status: SnapshotStatus, commit_hash: str | None = None, error: str | None = None) -> None:
    now = utc_now_iso()
    await get_db().execute(
        "UPDATE repo_snapshots SET status=?, commit_hash=?, error=?, synced_at=? WHERE id=?",
        (status.value, commit_hash, error, now, snap_id),
    )
    await get_db().commit()


# ──────────────────────────────────────────────────────────────────────────────
# Service
# ──────────────────────────────────────────────────────────────────────────────

class SyncEngineService:
    async def _run_prepare_task(
        self,
        snapshot_id: str,
        path: str,
        remote_url: str | None,
        branch: str | None,
        clone_policy: ClonePolicy,
    ) -> None:
        try:
            async with path_lock.acquire(path):
                await _update_status(snapshot_id, SnapshotStatus.SYNCING)

                if not (Path(path) / ".git").exists():
                    if not remote_url:
                        raise ValueError("No remote URL - cannot clone a folder with no git remote")
                    env = await _get_ssh_env() if is_ssh_url(remote_url) else os.environ.copy()

                    def _do_clone():
                        return _run_clone(remote_url, path, clone_policy, env)

                    result = await asyncio.to_thread(_do_clone)
                    if result.returncode != 0:
                        lines = [
                            ln for ln in result.stderr.splitlines()
                            if ln.strip() and not ln.startswith("Cloning into")
                        ]
                        raise ValueError("\n".join(lines) or "git clone failed")
                    logger.info(f"Cloned '{remote_url}' -> {path}")
                else:
                    if branch:
                        env = await _get_ssh_env() if is_ssh_url(remote_url or "") else os.environ.copy()
                        ok, err = await asyncio.to_thread(_run_sync, path, branch, env)
                        if not ok:
                            raise ValueError(f"git sync failed: {err}")
                    logger.info(f"Synced repo at {path} to branch '{branch}'")

                git_info = await read_git_info(path)
                commit_hash = git_info.get("head_hash")
                await _update_status(snapshot_id, SnapshotStatus.READY, commit_hash=commit_hash)
                logger.info(f"Snapshot {snapshot_id} ready at {commit_hash}")
        except Exception as e:
            await _update_status(snapshot_id, SnapshotStatus.FAILED, error=str(e))
            logger.error(f"Snapshot {snapshot_id} failed: {e}")

    async def prepare_snapshot(self, req: PrepareSnapshotRequest) -> RepoSnapshot:
        """
        Kick off clone/sync in background and return snapshot immediately.
        """
        # 1. Load local repo record
        async with get_db().execute(
            "SELECT * FROM local_repos WHERE id=?", (req.local_repo_id,)
        ) as cur:
            repo_row = await cur.fetchone()
        if repo_row is None:
            raise NotFoundError("LocalRepo", req.local_repo_id)

        path = repo_row["path"]
        remote_url = repo_row["git_remote_url"]
        branch = req.branch or repo_row["selected_branch"] or repo_row["git_branch"]

        # 2. Guard against parallel prepare on same repo
        async with get_db().execute(
            """
            SELECT id FROM repo_snapshots
            WHERE local_repo_id=? AND status IN (?, ?)
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (req.local_repo_id, SnapshotStatus.PENDING.value, SnapshotStatus.SYNCING.value),
        ) as cur:
            running = await cur.fetchone()
        if running is not None:
            raise ValueError("A snapshot prepare is already running for this repository")

        # 3. Preflight and create pending snapshot
        await _preflight(path, branch)
        snapshot = await _create_record(req.local_repo_id, branch, path, req.clone_policy)

        # 4. Run actual sync in background
        task = asyncio.create_task(
            self._run_prepare_task(
                snapshot_id=snapshot.id,
                path=path,
                remote_url=remote_url,
                branch=branch,
                clone_policy=req.clone_policy,
            )
        )
        _BACKGROUND_SYNC_TASKS.add(task)
        task.add_done_callback(lambda t: _BACKGROUND_SYNC_TASKS.discard(t))
        return snapshot

    async def get_snapshot(self, snapshot_id: str) -> RepoSnapshot:
        async with get_db().execute(
            "SELECT * FROM repo_snapshots WHERE id=?", (snapshot_id,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            raise NotFoundError("RepoSnapshot", snapshot_id)
        return _row_to_snapshot(row)

    async def list_for_repo(self, local_repo_id: str, limit: int = 20) -> list[RepoSnapshot]:
        async with get_db().execute(
            "SELECT * FROM repo_snapshots WHERE local_repo_id=? ORDER BY created_at DESC LIMIT ?",
            (local_repo_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [_row_to_snapshot(r) for r in rows]

    async def delete_snapshot(self, snapshot_id: str) -> None:
        db = get_db()
        async with db.execute("SELECT * FROM repo_snapshots WHERE id=?", (snapshot_id,)) as cur:
            row = await cur.fetchone()
        if row is None:
            raise NotFoundError("RepoSnapshot", snapshot_id)

        repo_id = row["local_repo_id"]
        path = row["local_path"]

        # Guard concurrent sync activity on same working tree path.
        async with path_lock.acquire(path):
            # If this snapshot is active for its repo, clear active pointer first.
            await db.execute(
                """
                UPDATE local_repos
                SET active_snapshot_id=NULL
                WHERE id=? AND active_snapshot_id=?
                """,
                (repo_id, snapshot_id),
            )

            # Remove all index artifacts bound to this snapshot.
            await db.execute("DELETE FROM manifest_files WHERE snapshot_id=?", (snapshot_id,))
            await db.execute("DELETE FROM code_symbols WHERE snapshot_id=?", (snapshot_id,))
            await db.execute("DELETE FROM repo_maps WHERE snapshot_id=?", (snapshot_id,))
            await db.execute("DELETE FROM structural_graph_edges WHERE snapshot_id=?", (snapshot_id,))
            await db.execute("DELETE FROM structural_graph_summaries WHERE snapshot_id=?", (snapshot_id,))

            # Finally remove snapshot record.
            await db.execute("DELETE FROM repo_snapshots WHERE id=?", (snapshot_id,))
            await db.commit()

        logger.info(f"Deleted snapshot {snapshot_id} and related artifacts")
