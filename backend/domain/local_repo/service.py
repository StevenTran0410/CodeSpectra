"""LocalRepoService — path validation, git metadata reading, and CRUD."""
import asyncio
import os
import subprocess  # still used by clone_from_url
from pathlib import Path

from infrastructure.db.database import get_db
from shared.errors import ConflictError, NotFoundError
from shared.git_utils import is_ssh_url, list_branches, list_branches_sync, read_git_info, read_git_info_sync
from shared.logger import logger
from shared.utils import new_id, utc_now_iso

from .types import (
    AddLocalRepoRequest,
    CloneFromUrlRequest,
    LocalRepo,
    RepoSourceType,
    SetBranchRequest,
    ValidateFolderRequest,
    ValidateFolderResponse,
)


# Directories that indicate the repo may be heavy to scan
_SIZE_WARNING_DIRS = frozenset({"node_modules", ".venv", "venv", "env", "target", "build", "dist"})
# If the root contains this many immediate entries, warn
_ROOT_ENTRY_THRESHOLD = 200


def _check_size_warning_sync(path: str) -> tuple[bool, str | None]:
    """Quick check for large-repo indicators (blocking — run in thread)."""
    p = Path(path)
    try:
        entries = list(p.iterdir())
    except PermissionError:
        return False, None

    if len(entries) > _ROOT_ENTRY_THRESHOLD:
        return True, f"Root directory has {len(entries)} entries — scan may be slow"

    heavy = [e.name for e in entries if e.name in _SIZE_WARNING_DIRS]
    if heavy:
        return True, f"Contains heavy directories: {', '.join(heavy[:3])}"

    return False, None


async def _check_size_warning(path: str) -> tuple[bool, str | None]:
    return await asyncio.to_thread(_check_size_warning_sync, path)


def _row_to_model(row) -> LocalRepo:
    return LocalRepo(
        id=row["id"],
        path=row["path"],
        name=row["name"],
        source_type=RepoSourceType(row["source_type"]),
        is_git_repo=bool(row["is_git_repo"]),
        git_branch=row["git_branch"],
        git_head_hash=row["git_head_hash"],
        git_remote_url=row["git_remote_url"],
        has_size_warning=bool(row["has_size_warning"]),
        selected_branch=row["selected_branch"],
        added_at=row["added_at"],
        last_validated_at=row["last_validated_at"],
    )


class LocalRepoService:
    async def validate(self, req: ValidateFolderRequest) -> ValidateFolderResponse:
        p = Path(req.path)
        exists = p.exists()
        is_dir = p.is_dir() if exists else False
        name = p.name or req.path

        if not exists or not is_dir:
            return ValidateFolderResponse(
                path=req.path,
                name=name,
                exists=exists,
                is_directory=is_dir,
                is_git_repo=False,
                git_branch=None,
                git_head_hash=None,
                git_remote_url=None,
                has_size_warning=False,
                size_warning_reason=None,
            )

        git_info, (size_warn, size_reason) = await asyncio.gather(
            read_git_info(req.path),
            _check_size_warning(req.path),
        )

        return ValidateFolderResponse(
            path=req.path,
            name=name,
            exists=True,
            is_directory=True,
            is_git_repo=git_info["is_git_repo"],
            git_branch=git_info["branch"],
            git_head_hash=git_info["head_hash"],
            git_remote_url=git_info["remote_url"],
            has_size_warning=size_warn,
            size_warning_reason=size_reason,
        )

    async def add(self, req: AddLocalRepoRequest) -> LocalRepo:
        validation = await self.validate(ValidateFolderRequest(path=req.path))
        if not validation.exists or not validation.is_directory:
            raise ValueError(f"Path '{req.path}' is not a valid directory")

        db = get_db()
        async with db.execute("SELECT 1 FROM local_repos WHERE path = ?", (req.path,)) as cur:
            if await cur.fetchone():
                raise ConflictError(f"Folder '{req.path}' is already added")

        repo_id = new_id()
        now = utc_now_iso()

        await db.execute(
            """INSERT INTO local_repos
               (id, path, name, source_type, is_git_repo, git_branch,
                git_head_hash, git_remote_url, has_size_warning, selected_branch,
                added_at, last_validated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                repo_id,
                req.path,
                validation.name,
                RepoSourceType.LOCAL_FOLDER.value,
                int(validation.is_git_repo),
                validation.git_branch,
                validation.git_head_hash,
                validation.git_remote_url,
                int(validation.has_size_warning),
                None,   # selected_branch defaults to None (use HEAD)
                now,
                now,
            ),
        )
        await db.commit()
        logger.info(f"Added local repo '{validation.name}' at {req.path}")

        row = await self._fetch_row(repo_id)
        return _row_to_model(row)

    async def list_all(self) -> list[LocalRepo]:
        db = get_db()
        async with db.execute("SELECT * FROM local_repos ORDER BY added_at ASC") as cur:
            rows = await cur.fetchall()
        return [_row_to_model(r) for r in rows]

    async def get_by_id(self, repo_id: str) -> LocalRepo:
        row = await self._fetch_row(repo_id)
        return _row_to_model(row)

    async def remove(self, repo_id: str) -> None:
        db = get_db()
        async with db.execute("DELETE FROM local_repos WHERE id = ?", (repo_id,)) as cur:
            if cur.rowcount == 0:
                raise NotFoundError("LocalRepo", repo_id)
        await db.commit()
        logger.info(f"Removed local repo {repo_id}")

    async def list_branches(self, repo_id: str) -> list[str]:
        """Return all local git branches for this repo, or [] if not a git repo."""
        repo = await self.get_by_id(repo_id)
        if not repo.is_git_repo:
            return []
        return await list_branches(repo.path)

    async def set_branch(self, repo_id: str, req: SetBranchRequest) -> LocalRepo:
        """Persist the user's chosen analysis branch."""
        existing = await self.get_by_id(repo_id)
        if not existing.is_git_repo:
            raise ValueError("Cannot set branch on a non-git folder")

        # Verify the branch actually exists locally
        branches = await list_branches(existing.path)
        if branches and req.branch not in branches:
            raise ValueError(
                f"Branch '{req.branch}' not found in local repo. "
                f"Available: {', '.join(branches[:10])}"
            )

        db = get_db()
        await db.execute(
            "UPDATE local_repos SET selected_branch=? WHERE id=?",
            (req.branch, repo_id),
        )
        await db.commit()
        logger.info(f"Set analysis branch to '{req.branch}' for repo {repo_id}")
        return await self.get_by_id(repo_id)

    async def revalidate(self, repo_id: str) -> LocalRepo:
        """Refresh git metadata for an existing local repo."""
        existing = await self.get_by_id(repo_id)
        validation = await self.validate(ValidateFolderRequest(path=existing.path))
        now = utc_now_iso()
        db = get_db()
        await db.execute(
            """UPDATE local_repos
               SET is_git_repo=?, git_branch=?, git_head_hash=?, git_remote_url=?,
                   has_size_warning=?, last_validated_at=?
               WHERE id=?""",
            (
                int(validation.is_git_repo),
                validation.git_branch,
                validation.git_head_hash,
                validation.git_remote_url,
                int(validation.has_size_warning),
                now,
                repo_id,
            ),
        )
        await db.commit()
        row = await self._fetch_row(repo_id)
        return _row_to_model(row)

    async def clone_from_url(self, req: CloneFromUrlRequest) -> LocalRepo:
        """Clone a remote git URL to dest_path, then register it as a local repo."""
        dest = Path(req.dest_path)

        if dest.exists():
            raise ConflictError(f"Destination '{req.dest_path}' already exists")

        dest.parent.mkdir(parents=True, exist_ok=True)

        # Build environment — inject GIT_SSH_COMMAND for SSH URLs if a key is configured
        env = os.environ.copy()
        if is_ssh_url(req.url):
            db = get_db()
            async with db.execute(
                "SELECT value FROM app_metadata WHERE key='git_ssh_key_path'"
            ) as cur:
                row = await cur.fetchone()
            ssh_key = row["value"] if row else None
            if ssh_key:
                env["GIT_SSH_COMMAND"] = (
                    f'ssh -i "{ssh_key}" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new'
                )

        def _do_clone() -> subprocess.CompletedProcess:
            return subprocess.run(
                ["git", "clone", req.url, str(dest)],
                capture_output=True,
                text=True,
                timeout=300,  # 5 min max
                env=env,
            )

        result = await asyncio.to_thread(_do_clone)
        if result.returncode != 0:
            # Filter out the informational "Cloning into '...'" line — only keep actual errors
            lines = [
                ln for ln in (result.stderr or "").splitlines()
                if ln.strip() and not ln.strip().startswith("Cloning into ")
            ]
            msg = "\n".join(lines).strip() or "git clone failed (unknown error)"
            raise ValueError(msg)

        logger.info(f"Cloned '{req.url}' → {req.dest_path}")
        return await self.add(AddLocalRepoRequest(path=req.dest_path))

    async def _fetch_row(self, repo_id: str):
        db = get_db()
        async with db.execute("SELECT * FROM local_repos WHERE id = ?", (repo_id,)) as cur:
            row = await cur.fetchone()
        if row is None:
            raise NotFoundError("LocalRepo", repo_id)
        return row
