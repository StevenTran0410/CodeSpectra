import subprocess
from pathlib import Path

import pytest

from agent_eval_harness.code_injection.injection_target import WorktreeInjectionTarget

pytestmark = pytest.mark.skipif(
    subprocess.run(["git", "--version"], capture_output=True).returncode != 0,
    reason="git not available",
)


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "target_repo"
    root.mkdir()
    _git(["init"], root)
    _git(["config", "user.email", "test@example.com"], root)
    _git(["config", "user.name", "test"], root)
    (root / "app.py").write_text("print('hello')\n", encoding="utf-8")
    _git(["add", "."], root)
    _git(["commit", "-m", "initial"], root)
    return root


def test_prepare_creates_isolated_worktree_with_committed_content(repo: Path) -> None:
    target = WorktreeInjectionTarget(repo_root=repo, plan_id="plan-1")

    worktree = target.prepare()

    assert worktree.exists()
    assert (worktree / "app.py").read_text(encoding="utf-8") == "print('hello')\n"
    assert worktree != repo


def test_prepare_never_mutates_the_main_checkout(repo: Path) -> None:
    (repo / "app.py").write_text("print('uncommitted local edit')\n", encoding="utf-8")
    target = WorktreeInjectionTarget(repo_root=repo, plan_id="plan-2")

    worktree = target.prepare()
    (worktree / "app.py").write_text("print('injected code')\n", encoding="utf-8")

    assert (repo / "app.py").read_text(encoding="utf-8") == "print('uncommitted local edit')\n"


def test_prepare_is_idempotent_for_the_same_plan_id(repo: Path) -> None:
    target = WorktreeInjectionTarget(repo_root=repo, plan_id="plan-3")

    first = target.prepare()
    second = target.prepare()

    assert first == second
    assert first.exists()


def test_cleanup_removes_worktree_and_branch(repo: Path) -> None:
    target = WorktreeInjectionTarget(repo_root=repo, plan_id="plan-4")
    worktree = target.prepare()
    assert worktree.exists()

    target.cleanup()

    assert not worktree.exists()
    branches = subprocess.run(
        ["git", "branch", "--list", target.branch_name], cwd=repo, capture_output=True, text=True,
    ).stdout
    assert target.branch_name not in branches
