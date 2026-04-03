"""Shared git subprocess helpers used across local_repo and sync_engine."""
import asyncio
import subprocess
from pathlib import Path


def is_ssh_url(url: str) -> bool:
    return url.startswith("git@") or url.startswith("ssh://")


def run_git_sync(cwd: str, args: list[str], env: dict | None = None, timeout: int = 10) -> str | None:
    """Run a git sub-command synchronously. Returns stdout or None on failure."""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        return result.stdout.strip() or None if result.returncode == 0 else None
    except Exception:
        return None


def read_git_info_sync(path: str) -> dict:
    """Read branch, HEAD hash, remote URL. Returns dict with is_git_repo key."""
    git_dir = Path(path) / ".git"
    if not git_dir.exists():
        return {"is_git_repo": False, "branch": None, "head_hash": None, "remote_url": None}

    branch = run_git_sync(path, ["rev-parse", "--abbrev-ref", "HEAD"])
    head   = run_git_sync(path, ["rev-parse", "HEAD"])
    remote = run_git_sync(path, ["remote", "get-url", "origin"])

    return {
        "is_git_repo": True,
        "branch": branch,
        "head_hash": head[:12] if head else None,
        "remote_url": remote,
    }


def list_branches_sync(path: str) -> list[str]:
    raw = run_git_sync(path, ["branch", "--format=%(refname:short)"])
    if not raw:
        return []
    return [b.strip() for b in raw.splitlines() if b.strip()]


def is_dirty_sync(path: str) -> bool:
    """True if the working tree has uncommitted changes."""
    result = run_git_sync(path, ["status", "--porcelain"])
    return bool(result)


def has_submodules_sync(path: str) -> bool:
    return (Path(path) / ".gitmodules").exists()


async def run_git(cwd: str, args: list[str], env: dict | None = None, timeout: int = 10) -> str | None:
    return await asyncio.to_thread(run_git_sync, cwd, args, env, timeout)


async def read_git_info(path: str) -> dict:
    return await asyncio.to_thread(read_git_info_sync, path)


async def list_branches(path: str) -> list[str]:
    return await asyncio.to_thread(list_branches_sync, path)
