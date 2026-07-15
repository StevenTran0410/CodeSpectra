import subprocess
from pathlib import Path

import pytest

from agent_eval_harness.code_injection.injection_target import (
    BranchInjectionTarget,
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


def _branch_exists(repo: Path, name: str) -> bool:
    branches = subprocess.run(
        ["git", "branch", "--list", name],
        cwd=repo, capture_output=True, text=True,
    ).stdout
    return name in branches


def test_prepare_creates_branch_without_switching(git_repo: Path) -> None:
    original = _current_branch(git_repo)
    info = BranchInjectionTarget.prepare(git_repo, "sess-abc")
    assert info.branch_name == "aeh/eval-sess-abc"
    assert info.current_branch == original
    assert _branch_exists(git_repo, "aeh/eval-sess-abc")
    # HEAD never moved — that's the entire point of this design.
    assert _current_branch(git_repo) == original


def test_prepare_does_not_require_a_clean_tree(git_repo: Path) -> None:
    (git_repo / "dirty.txt").write_text("uncommitted", encoding="utf-8")
    # Creating a branch ref never touches the working tree, so a dirty tree is fine.
    info = BranchInjectionTarget.prepare(git_repo, "sess-123")
    assert _branch_exists(git_repo, "aeh/eval-sess-123")
    assert info.current_branch == "main"


def test_prepare_idempotent_on_repeat_calls(git_repo: Path) -> None:
    info1 = BranchInjectionTarget.prepare(git_repo, "sess-abc")
    info2 = BranchInjectionTarget.prepare(git_repo, "sess-abc")
    assert info1.branch_name == info2.branch_name == "aeh/eval-sess-abc"
    assert info1.current_branch == info2.current_branch


def test_restore_refuses_dirty_eval_branch(git_repo: Path) -> None:
    original = _current_branch(git_repo)
    BranchInjectionTarget.prepare(git_repo, "sess-xyz")
    subprocess.run(["git", "checkout", "aeh/eval-sess-xyz"], cwd=git_repo, check=True, capture_output=True)
    (git_repo / "dirty.txt").write_text("uncommitted on eval branch", encoding="utf-8")
    with pytest.raises(InjectionTargetError):
        BranchInjectionTarget.restore(git_repo, original)


def test_restore_returns_to_original_branch(git_repo: Path) -> None:
    original = _current_branch(git_repo)
    BranchInjectionTarget.prepare(git_repo, "sess-ret")
    subprocess.run(["git", "checkout", "aeh/eval-sess-ret"], cwd=git_repo, check=True, capture_output=True)
    assert _current_branch(git_repo) == "aeh/eval-sess-ret"
    BranchInjectionTarget.restore(git_repo, original)
    assert _current_branch(git_repo) == original


def test_prepare_raises_on_missing_base_ref(git_repo: Path) -> None:
    with pytest.raises(InjectionTargetError, match="nonexistent-ref"):
        BranchInjectionTarget.prepare(git_repo, "sess-noref", base_ref="nonexistent-ref")
    assert not _branch_exists(git_repo, "aeh/eval-sess-noref")
