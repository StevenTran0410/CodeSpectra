"""LocalRepoService — path validation, git metadata reading, and CRUD."""
import asyncio
import fnmatch
import json
import os
import shutil
import stat
import subprocess  # still used by clone_from_url
from pathlib import Path

from infrastructure.db.database import get_db
from shared.errors import ConflictError, NotFoundError
from shared.git_utils import is_ssh_url, list_branches, read_git_info, run_git
from shared.logger import logger
from shared.utils import new_id, utc_now_iso

from domain.snapshot_cleanup import delete_repo_artifacts

from .types import (
    AddLocalRepoRequest,
    CloneFromUrlRequest,
    EstimateFileCountResponse,
    LocalRepo,
    RepoSourceType,
    SetBranchRequest,
    UpdateRepoSettingsRequest,
    ValidateFolderRequest,
    ValidateFolderResponse,
)


# Directories that indicate the repo may be heavy to scan
_SIZE_WARNING_DIRS = frozenset({"node_modules", ".venv", "venv", "env", "target", "build", "dist"})
# Workspace-level ignore defaults (read-only at repository setup screen)
_WORKSPACE_DEFAULT_IGNORES = [".git/**", "node_modules/**", ".venv/**", "venv/**", "dist/**", "build/**", "target/**"]
# If the root contains this many immediate entries, warn
_ROOT_ENTRY_THRESHOLD = 200


def _normalize_repo_path(path: str) -> str:
    """Canonicalize user path so add/remove checks are stable on Windows."""
    try:
        return str(Path(path).resolve())
    except Exception:
        return str(Path(path).absolute())


def _detect_source_type(url: str) -> RepoSourceType:
    """Infer repo host from a git URL (HTTPS or SSH) for display purposes."""
    u = url.lower()
    if "github.com" in u:
        return RepoSourceType.GITHUB
    if "bitbucket.org" in u:
        return RepoSourceType.BITBUCKET
    return RepoSourceType.LOCAL_FOLDER


def _is_under_path(path: Path, root: Path) -> bool:
    """Robust Windows-safe containment check."""
    try:
        p = str(path.resolve())
        r = str(root.resolve())
        return os.path.commonpath([p, r]) == r
    except Exception:
        return False


def _remove_tree_strict(path: Path) -> None:
    """Delete folder/file strictly (handles read-only files on Windows)."""
    if not path.exists():
        return
    if path.is_file() or path.is_symlink():
        path.unlink(missing_ok=True)
        return

    def _onerror(func, target, _exc_info):
        try:
            os.chmod(target, stat.S_IWRITE)
            func(target)
        except Exception:
            pass

    shutil.rmtree(path, onerror=_onerror)
    if path.exists():
        raise ValueError(f"Cannot delete managed clone folder: {path}")


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


def _count_files_with_ignores_sync(root: Path, ignore_patterns: list[str]) -> int:
    dir_prefixes = [p[:-3] for p in ignore_patterns if p.endswith("/**")]
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root).replace("\\", "/")
        if rel_dir == ".":
            rel_dir = ""

        # prune ignored directories early
        kept_dirs: list[str] = []
        for d in dirnames:
            rel = f"{rel_dir}/{d}" if rel_dir else d
            if any(rel == prefix or rel.startswith(f"{prefix}/") for prefix in dir_prefixes):
                continue
            kept_dirs.append(d)
        dirnames[:] = kept_dirs

        for name in filenames:
            rel = f"{rel_dir}/{name}" if rel_dir else name
            if any(fnmatch.fnmatch(rel, p) for p in ignore_patterns):
                continue
            count += 1
    return count


def _row_to_model(row) -> LocalRepo:
    try:
        ignore_overrides = json.loads(row["ignore_overrides"] or "[]")
    except Exception:
        ignore_overrides = []
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
        active_snapshot_id=row["active_snapshot_id"],
        sync_mode=row["sync_mode"],
        pinned_ref=row["pinned_ref"],
        ignore_overrides=ignore_overrides,
        detect_submodules=bool(row["detect_submodules"]),
        added_at=row["added_at"],
        last_validated_at=row["last_validated_at"],
    )


class LocalRepoService:
    async def _ssh_env_if_needed(self, url: str | None) -> dict | None:
        if not url or not is_ssh_url(url):
            return None
        db = get_db()
        async with db.execute(
            "SELECT value FROM app_metadata WHERE key='git_ssh_key_path'"
        ) as cur:
            row = await cur.fetchone()
        ssh_key = row["value"] if row else None
        if not ssh_key:
            return None
        env = os.environ.copy()
        env["GIT_SSH_COMMAND"] = (
            f'ssh -i "{ssh_key}" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new'
        )
        return env

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

    async def add(
        self,
        req: AddLocalRepoRequest,
        source_type: RepoSourceType = RepoSourceType.LOCAL_FOLDER,
    ) -> LocalRepo:
        normalized_path = _normalize_repo_path(req.path)
        validation = await self.validate(ValidateFolderRequest(path=normalized_path))
        if not validation.exists or not validation.is_directory:
            raise ValueError(f"Path '{normalized_path}' is not a valid directory")

        db = get_db()
        async with db.execute("SELECT 1 FROM local_repos WHERE path = ?", (normalized_path,)) as cur:
            if await cur.fetchone():
                raise ConflictError(f"Folder '{normalized_path}' is already added")

        repo_id = new_id()
        now = utc_now_iso()

        await db.execute(
            """INSERT INTO local_repos
               (id, path, name, source_type, is_git_repo, git_branch,
                git_head_hash, git_remote_url, has_size_warning, selected_branch,
                sync_mode, pinned_ref, ignore_overrides, detect_submodules,
                added_at, last_validated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                repo_id,
                normalized_path,
                validation.name,
                source_type.value,
                int(validation.is_git_repo),
                validation.git_branch,
                validation.git_head_hash,
                validation.git_remote_url,
                int(validation.has_size_warning),
                None,   # selected_branch defaults to None (use HEAD)
                "latest",
                None,
                "[]",
                1,
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
        repo = await self.get_by_id(repo_id)
        await delete_repo_artifacts(repo_id)

        # If this path is inside CodeSpectra-managed clone root, delete it from disk first.
        # For managed clones, deletion is strict to avoid silent leftovers causing clone conflicts.
        managed_root = Path.home() / "CodeSpectra" / "repos"
        repo_path = Path(repo.path)
        if _is_under_path(repo_path, managed_root):
            _remove_tree_strict(repo_path)
            logger.info(f"Deleted managed clone folder: {repo_path}")

        async with db.execute("DELETE FROM local_repos WHERE id = ?", (repo_id,)) as cur:
            if cur.rowcount == 0:
                raise NotFoundError("LocalRepo", repo_id)
        await db.commit()

        logger.info(f"Removed local repo {repo_id}")

    async def list_branches(self, repo_id: str, refresh: bool = False) -> list[str]:
        """Return branches. Refresh remote refs only when explicitly requested."""
        repo = await self.get_by_id(repo_id)
        if not repo.is_git_repo:
            return []

        if refresh:
            # Optional on-demand refresh to avoid slow branch dropdown open.
            env = await self._ssh_env_if_needed(repo.git_remote_url)
            await run_git(repo.path, ["fetch", "--all", "--prune"], env=env, timeout=15)

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

    async def set_active_snapshot(self, repo_id: str, snapshot_id: str | None) -> LocalRepo:
        repo = await self.get_by_id(repo_id)
        db = get_db()
        if snapshot_id:
            async with db.execute(
                "SELECT 1 FROM repo_snapshots WHERE id=? AND local_repo_id=?",
                (snapshot_id, repo_id),
            ) as cur:
                row = await cur.fetchone()
            if not row:
                raise ValueError("Snapshot does not belong to this repository")
        await db.execute(
            "UPDATE local_repos SET active_snapshot_id=? WHERE id=?",
            (snapshot_id, repo.id),
        )
        await db.commit()
        logger.info(f"Set active_snapshot_id={snapshot_id} for repo {repo_id}")
        return await self.get_by_id(repo_id)

    async def update_settings(self, repo_id: str, req: UpdateRepoSettingsRequest) -> LocalRepo:
        await self.get_by_id(repo_id)  # ensure exists
        db = get_db()
        await db.execute(
            """UPDATE local_repos
               SET sync_mode=?, pinned_ref=?, ignore_overrides=?, detect_submodules=?
               WHERE id=?""",
            (
                req.sync_mode.value,
                req.pinned_ref,
                json.dumps(req.ignore_overrides),
                int(req.detect_submodules),
                repo_id,
            ),
        )
        await db.commit()
        logger.info(f"Updated repository settings for {repo_id}")
        return await self.get_by_id(repo_id)

    async def estimate_file_count(self, repo_id: str) -> EstimateFileCountResponse:
        repo = await self.get_by_id(repo_id)
        root = Path(repo.path)
        if not root.exists() or not root.is_dir():
            raise ValueError("Repository path does not exist")

        effective = [*_WORKSPACE_DEFAULT_IGNORES, *repo.ignore_overrides]
        count = await asyncio.to_thread(_count_files_with_ignores_sync, root, effective)
        return EstimateFileCountResponse(
            estimated_file_count=count,
            workspace_default_ignores=_WORKSPACE_DEFAULT_IGNORES,
            repo_ignore_overrides=repo.ignore_overrides,
            effective_ignores=effective,
        )

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
        normalized_dest_path = _normalize_repo_path(req.dest_path)
        dest = Path(normalized_dest_path)

        if dest.exists():
            if dest.is_dir() and not any(dest.iterdir()):
                # stale empty folder from previous failed delete/clone
                dest.rmdir()

        if dest.exists():
            # If destination is already a git repo with same remote, reuse it.
            git_info = await read_git_info(str(dest))
            if git_info.get("is_git_repo"):
                remote = await run_git(str(dest), ["remote", "get-url", "origin"], timeout=10)
                if remote and remote.rstrip("/") == req.url.rstrip("/"):
                    logger.info(f"Destination already contains same repo, reusing: {normalized_dest_path}")
                    try:
                        return await self.add(AddLocalRepoRequest(path=normalized_dest_path))
                    except ConflictError:
                        # Already registered in DB, return existing row by path
                        db = get_db()
                        async with db.execute(
                            "SELECT id FROM local_repos WHERE path = ?",
                            (normalized_dest_path,),
                        ) as cur:
                            row = await cur.fetchone()
                        if row:
                            return await self.get_by_id(row["id"])
            raise ConflictError(
                f"Destination '{normalized_dest_path}' already exists. Remove it first or choose another repo URL."
            )

        dest.parent.mkdir(parents=True, exist_ok=True)

        # Build environment — inject GIT_SSH_COMMAND for SSH URLs if a key is configured
        env = os.environ.copy()
        ssh_env = await self._ssh_env_if_needed(req.url)
        if ssh_env:
            env = ssh_env

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

        logger.info(f"Cloned '{req.url}' → {normalized_dest_path}")
        return await self.add(
            AddLocalRepoRequest(path=normalized_dest_path),
            source_type=_detect_source_type(req.url),
        )

    async def _fetch_row(self, repo_id: str):
        db = get_db()
        async with db.execute("SELECT * FROM local_repos WHERE id = ?", (repo_id,)) as cur:
            row = await cur.fetchone()
        if row is None:
            raise NotFoundError("LocalRepo", repo_id)
        return row
