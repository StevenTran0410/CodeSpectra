"""Stage 4 branch management: creates the eval branch WITHOUT checking it out — the user
switches to it themselves (their own git tooling / editor) whenever they're ready, since an
automatic checkout was switching their working directory out from under them mid-session.
`git branch <name> <base_ref>` only writes a new ref; it never touches the working tree or
HEAD, so there's no dirty-tree precondition to enforce here."""
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
    current_branch: str  # whatever HEAD was at call time — informational only, never switched


class BranchInjectionTarget:
    @staticmethod
    def prepare(repo_root: Path, session_id: str, base_ref: str = "main") -> BranchInfo:
        current_branch = _run_git(
            ["rev-parse", "--abbrev-ref", "HEAD"], repo_root
        ).stdout.strip()
        branch_name = f"aeh/eval-{session_id}"
        exists = _run_git(
            ["rev-parse", "--verify", f"refs/heads/{branch_name}"], repo_root, check=False
        )
        if exists.returncode != 0:
            base_ok = _run_git(
                ["rev-parse", "--verify", base_ref], repo_root, check=False
            )
            if base_ok.returncode != 0:
                raise InjectionTargetError(
                    f"base_ref '{base_ref}' not found locally — fetch it or specify an existing ref"
                )
            _run_git(["branch", branch_name, base_ref], repo_root)
        return BranchInfo(branch_name=branch_name, current_branch=current_branch)

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
