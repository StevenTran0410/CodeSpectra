"""Stage 4 branch management: checkout -b an isolated eval branch in the caller's
working directory. The worktree approach (WorktreeInjectionTarget) was removed in CS-284 —
the running backend must be restarted after a branch switch regardless, so worktree
isolation gave no benefit over in-place checkout with a dirty-tree guard."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


class InjectionTargetError(Exception):
    pass


def _run_git(args: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False,
    )
    if check and result.returncode != 0:
        raise InjectionTargetError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result


@dataclass
class BranchInfo:
    branch_name: str
    original_branch: str


class BranchInjectionTarget:
    @staticmethod
    def prepare(repo_root: Path, session_id: str, base_ref: str = "main") -> BranchInfo:
        # Dirty-check: refuse to proceed if working tree has uncommitted changes
        result = _run_git(["status", "--porcelain"], repo_root)
        if result.stdout.strip():
            raise InjectionTargetError(
                f"Working tree is dirty — commit or discard changes first:\n{result.stdout.strip()}"
            )
        # Capture the branch we are currently on before switching
        original_branch = _run_git(
            ["rev-parse", "--abbrev-ref", "HEAD"], repo_root
        ).stdout.strip()
        # Determine the eval branch name and switch to it (idempotent)
        branch_name = f"aeh/eval-{session_id}"
        exists = _run_git(
            ["rev-parse", "--verify", f"refs/heads/{branch_name}"], repo_root, check=False
        )
        if exists.returncode == 0:
            _run_git(["checkout", branch_name], repo_root)
        else:
            base_ok = _run_git(
                ["rev-parse", "--verify", base_ref], repo_root, check=False
            )
            if base_ok.returncode != 0:
                raise InjectionTargetError(
                    f"base_ref '{base_ref}' not found locally — fetch it or specify an existing ref"
                )
            _run_git(["checkout", "-b", branch_name, base_ref], repo_root)
        return BranchInfo(branch_name=branch_name, original_branch=original_branch)

    @staticmethod
    def restore(repo_root: Path, original_branch: str) -> None:
        result = _run_git(["status", "--porcelain"], repo_root)
        if result.stdout.strip():
            raise InjectionTargetError(
                f"Eval branch has uncommitted changes — commit or discard them first:\n{result.stdout.strip()}"
            )
        _run_git(["checkout", original_branch], repo_root)

    @staticmethod
    def current_branch(repo_root: Path) -> str:
        return _run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root).stdout.strip()
