"""LocalRepoService — path validation, git metadata reading, and CRUD."""
import asyncio
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from infrastructure.db.database import get_db
from shared.errors import ConflictError, NotFoundError
from shared.logger import logger

from .types import (
    AddLocalRepoRequest,
    LocalRepo,
    RepoSourceType,
    ValidateFolderRequest,
    ValidateFolderResponse,
)

# Directories that indicate the repo may be heavy to scan
_SIZE_WARNING_DIRS = frozenset({"node_modules", ".venv", "venv", "env", "target", "build", "dist"})
# If the root contains this many immediate entries, warn
_ROOT_ENTRY_THRESHOLD = 200


def _run_git(cwd: str, args: list[str]) -> str | None:
    """Run a git sub-command synchronously (called inside asyncio.to_thread)."""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or None if result.returncode == 0 else None
    except Exception:
        return None


def _read_git_info_sync(path: str) -> dict:
    """Read branch, HEAD hash, remote URL (blocking — run in thread)."""
    git_dir = Path(path) / ".git"
    if not git_dir.exists():
        return {"is_git_repo": False, "branch": None, "head_hash": None, "remote_url": None}

    branch = _run_git(path, ["rev-parse", "--abbrev-ref", "HEAD"])
    head = _run_git(path, ["rev-parse", "HEAD"])
    remote = _run_git(path, ["remote", "get-url", "origin"])

    return {
        "is_git_repo": True,
        "branch": branch,
        "head_hash": head[:12] if head else None,
        "remote_url": remote,
    }


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


async def _read_git_info(path: str) -> dict:
    return await asyncio.to_thread(_read_git_info_sync, path)


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
            _read_git_info(req.path),
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

        repo_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        await db.execute(
            """INSERT INTO local_repos
               (id, path, name, source_type, is_git_repo, git_branch,
                git_head_hash, git_remote_url, has_size_warning, added_at, last_validated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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

    async def revalidate(self, repo_id: str) -> LocalRepo:
        """Refresh git metadata for an existing local repo."""
        existing = await self.get_by_id(repo_id)
        validation = await self.validate(ValidateFolderRequest(path=existing.path))
        now = datetime.now(timezone.utc).isoformat()
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

    async def _fetch_row(self, repo_id: str):
        db = get_db()
        async with db.execute("SELECT * FROM local_repos WHERE id = ?", (repo_id,)) as cur:
            row = await cur.fetchone()
        if row is None:
            raise NotFoundError("LocalRepo", repo_id)
        return row
