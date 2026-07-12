import subprocess
from pathlib import Path

import pytest

from agent_eval_harness.code_injection.injection_target import (
    BranchInjectionTarget,
    BranchInfo,
    InjectionTargetError,
)

pytestmark = pytest.mark.skipif(
    subprocess.run(["git", "--version"], capture_output=True).returncode != 0,
    reason="git not available",
)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    (tmp_path / "README.md").write_text("init", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "branch", "-M", "main"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    return tmp_path


def _current_branch(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo, capture_output=True, text=True, check=True,
    ).stdout.strip()


def test_prepare_refuses_dirty_tree_with_zero_writes(git_repo: Path) -> None:
    (git_repo / "dirty.txt").write_text("uncommitted", encoding="utf-8")
    with pytest.raises(InjectionTargetError):
        BranchInjectionTarget.prepare(git_repo, "sess-123")
    branches = subprocess.run(
        ["git", "branch", "--list", "aeh/eval-sess-123"],
        cwd=git_repo, capture_output=True, text=True,
    ).stdout
    assert "aeh/eval-sess-123" not in branches


def test_prepare_idempotent_already_on_eval_branch(git_repo: Path) -> None:
    original = _current_branch(git_repo)  # capture BEFORE first prepare
    info1 = BranchInjectionTarget.prepare(git_repo, "sess-abc")
    assert info1.original_branch == original  # first call captures the real original branch
    info2 = BranchInjectionTarget.prepare(git_repo, "sess-abc")
    assert info1.branch_name == info2.branch_name == "aeh/eval-sess-abc"
    # info2.original_branch is intentionally not asserted — the route's DB guard
    # prevents double-prepare in practice; on a second call HEAD is already the eval branch.


def test_restore_refuses_dirty_eval_branch(git_repo: Path) -> None:
    original = _current_branch(git_repo)
    BranchInjectionTarget.prepare(git_repo, "sess-xyz")
    (git_repo / "dirty.txt").write_text("uncommitted on eval branch", encoding="utf-8")
    with pytest.raises(InjectionTargetError):
        BranchInjectionTarget.restore(git_repo, original)


def test_restore_returns_to_original_branch(git_repo: Path) -> None:
    original = _current_branch(git_repo)
    BranchInjectionTarget.prepare(git_repo, "sess-ret")
    assert _current_branch(git_repo) == "aeh/eval-sess-ret"
    BranchInjectionTarget.restore(git_repo, original)
    assert _current_branch(git_repo) == original


def test_prepare_raises_on_missing_base_ref(git_repo: Path) -> None:
    with pytest.raises(InjectionTargetError, match="nonexistent-ref"):
        BranchInjectionTarget.prepare(git_repo, "sess-noref", base_ref="nonexistent-ref")
    branches = subprocess.run(
        ["git", "branch", "--list", "aeh/eval-sess-noref"],
        cwd=git_repo, capture_output=True, text=True,
    ).stdout
    assert "aeh/eval-sess-noref" not in branches
