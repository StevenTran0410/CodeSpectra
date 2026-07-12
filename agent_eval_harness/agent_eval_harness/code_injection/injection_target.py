"""Where Stage 4 writes eval artifacts: always an isolated directory backed by a dedicated
git branch, never the caller's own working directory. `git checkout -b` in place would mutate
whatever the user (or a currently-running backend process) has open on disk right now;
`git worktree add` gives a fully separate directory sharing the same .git object store — no
clone, no risk, and no dirty-worktree check needed (a new worktree is always a clean checkout
of a specific commit, independent of whatever is uncommitted in the main working directory)."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


class InjectionTargetError(Exception):
    pass


def _run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise InjectionTargetError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result


@dataclass
class WorktreeInjectionTarget:
    repo_root: Path
    plan_id: str

    @property
    def branch_name(self) -> str:
        return f"aeh/eval-{self.plan_id}"

    @property
    def worktree_path(self) -> Path:
        return self.repo_root.parent / ".aeh-worktrees" / self.repo_root.name / self.plan_id

    def prepare(self) -> Path:
        """Creates the worktree on first call; returns the existing one unchanged on repeat
        calls for the same plan_id (idempotent — re-running the injector doesn't error)."""
        path = self.worktree_path
        if path.exists():
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        _run_git(["worktree", "add", "-b", self.branch_name, str(path), "HEAD"], cwd=self.repo_root)
        return path

    def cleanup(self) -> None:
        """Removes the worktree and its branch. Never called automatically by the injector —
        the user reviews the branch with their own git tooling first (locked decision)."""
        path = self.worktree_path
        if path.exists():
            _run_git(["worktree", "remove", "--force", str(path)], cwd=self.repo_root)
        try:
            _run_git(["branch", "-D", self.branch_name], cwd=self.repo_root)
        except InjectionTargetError:
            pass
