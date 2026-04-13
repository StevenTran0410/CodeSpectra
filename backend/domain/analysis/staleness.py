"""Staleness detection and incremental re-analysis support (CS-222)."""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path

import aiosqlite

from shared.git_utils import run_git_sync
from shared.logger import logger


@dataclass
class StalenessResult:
    """Result of staleness check for a report."""

    stale: bool
    old_commit: str | None = None
    current_commit: str | None = None
    changed_files_count: int = 0
    insertions: int = 0
    deletions: int = 0
    sections_affected: list[str] = None
    recommend_new_snapshot: bool = False

    def __post_init__(self) -> None:
        if self.sections_affected is None:
            self.sections_affected = []


SECTION_PATTERNS = {
    "A": ["README*", "readme*", "package.json", "pyproject.toml", "Cargo.toml", "go.mod"],
    "B": [
        "docker*",
        "Docker*",
        "*.yaml",
        "*.yml",
        "*.toml",
        "*.json",
        "Makefile",
        "*.config.*",
        "*.conf",
        "requirements*.txt",
    ],
    "C": [],
    "D": [
        "*.test.*",
        "*.spec.*",
        ".eslint*",
        ".prettier*",
        "tox.ini",
        "setup.cfg",
        "jest.config*",
        "vitest.config*",
    ],
    "E": ["*.test.*", "*.spec.*"],
    "F": ["src/**", "lib/**", "app/**", "pkg/**"],
    "G": ["main.*", "index.*", "app.*", "server.*", "cli.*", "__main__.*"],
    "H": ["README*", "CONTRIBUTING*", "docs/**", "doc/**"],
    "J": [],
}


def _matches_any_pattern(rel_path: str, patterns: list[str]) -> bool:
    """Check if rel_path matches any fnmatch pattern."""
    if not patterns:
        return False
    for pattern in patterns:
        if fnmatch.fnmatch(rel_path, pattern):
            return True
    return False


def _map_file_to_sections(rel_path: str) -> list[str]:
    """Map a file path to sections that might be affected."""
    affected = []

    top_level_dir = rel_path.split("/")[0] if "/" in rel_path else rel_path

    for section_key, patterns in SECTION_PATTERNS.items():
        if section_key == "C":
            if rel_path == top_level_dir and rel_path not in [
                "src",
                "lib",
                "app",
                "pkg",
                "docs",
            ]:
                affected.append("C")
        elif _matches_any_pattern(rel_path, patterns):
            affected.append(section_key)

    return affected


def _parse_git_diff_stat(output: str | None) -> tuple[int, int, int]:
    """Parse git diff --stat output. Returns (file_count, insertions, deletions)."""
    if not output:
        return 0, 0, 0

    lines = output.strip().split("\n")
    file_count = 0
    insertions = 0
    deletions = 0

    for line in lines:
        line = line.strip()
        if "files changed" in line or "file changed" in line:
            m = re.search(r"(\d+)\s+files? changed", line)
            if m:
                file_count = int(m.group(1))
            m = re.search(r"(\d+)\s+insertions?\(\+\)", line)
            if m:
                insertions = int(m.group(1))
            m = re.search(r"(\d+)\s+deletions?\(-\)", line)
            if m:
                deletions = int(m.group(1))
            break

    return file_count, insertions, deletions


async def check_staleness(report_id: str, db: aiosqlite.Connection) -> StalenessResult:
    """Check if an analysis report is stale."""
    try:
        async with db.execute(
            """
            SELECT ar.snapshot_id
            FROM analysis_reports ar
            WHERE ar.id = ?
            """,
            (report_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return StalenessResult(stale=False)
            snapshot_id = row[0]

        async with db.execute(
            """
            SELECT rs.commit_hash, rs.branch,
                   lr.path, lr.source_type, lr.selected_branch, lr.git_branch
            FROM repo_snapshots rs
            JOIN local_repos lr ON lr.id = rs.local_repo_id
            WHERE rs.id = ?
            """,
            (snapshot_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return StalenessResult(stale=False)
            old_commit_hash, snap_branch, repo_path, source_type, selected_branch, git_branch = row

        if not old_commit_hash or not repo_path:
            return StalenessResult(stale=False)

        if not Path(repo_path).exists():
            return StalenessResult(stale=False, old_commit=old_commit_hash)

        # Check if repo has a remote origin
        has_remote = run_git_sync(repo_path, ["remote", "get-url", "origin"]) is not None

        logger.info(
            "[staleness] report=%s source=%s path=%s old=%s has_remote=%s branch=%s",
            report_id[:8], source_type, repo_path, old_commit_hash[:8] if old_commit_hash else "?",
            has_remote, snap_branch or selected_branch or git_branch,
        )

        # Fetch latest from remote if available
        if has_remote:
            fetch_result = run_git_sync(repo_path, ["fetch", "origin", "--quiet"])
            if fetch_result is None:
                logger.warning("[staleness] git fetch failed for %s", repo_path)

        if has_remote:
            # Compare against remote tracking branch
            branch = snap_branch or selected_branch or git_branch or "main"
            current_commit = run_git_sync(repo_path, ["rev-parse", f"origin/{branch}"])
        else:
            # No remote — compare local HEAD
            current_commit = run_git_sync(repo_path, ["rev-parse", "HEAD"])
        if not current_commit:
            logger.warning("[staleness] rev-parse returned None for %s", repo_path)
            return StalenessResult(stale=False, old_commit=old_commit_hash)

        current_commit = current_commit.strip()
        logger.info("[staleness] old=%s current=%s match=%s", old_commit_hash[:8], current_commit[:8], old_commit_hash == current_commit)

        if current_commit == old_commit_hash:
            return StalenessResult(stale=False, old_commit=old_commit_hash, current_commit=current_commit)

        diff_stat = run_git_sync(repo_path, ["diff", "--stat", f"{old_commit_hash}..{current_commit}"])
        logger.info("[staleness] diff_stat raw: %s", repr(diff_stat[:200]) if diff_stat else "None")
        changed_files_count, insertions, deletions = _parse_git_diff_stat(diff_stat)
        logger.info("[staleness] files=%d ins=%d del=%d", changed_files_count, insertions, deletions)

        if changed_files_count == 0 and not diff_stat:
            # diff command itself failed (None) — commits might not exist in this clone
            return StalenessResult(stale=True, old_commit=old_commit_hash, current_commit=current_commit,
                                   recommend_new_snapshot=True)

        if changed_files_count == 0:
            return StalenessResult(stale=False, old_commit=old_commit_hash, current_commit=current_commit)

        # Count commits between old and current
        rev_list = run_git_sync(
            repo_path, ["rev-list", "--count", f"{old_commit_hash}..{current_commit}"]
        )
        commit_count = int(rev_list.strip()) if rev_list and rev_list.strip().isdigit() else 0

        is_ancestor = run_git_sync(
            repo_path, ["merge-base", "--is-ancestor", old_commit_hash, current_commit]
        )
        non_linear = is_ancestor is None

        # Recommend rebuild only for substantial changes:
        # non-linear history (force push/rebase), OR 5+ commits with significant LOC churn
        total_loc_change = insertions + deletions
        recommend_force_rebuild = non_linear or (commit_count >= 5 and total_loc_change >= 500)

        changed_files = []
        if diff_stat:
            for line in diff_stat.strip().split("\n"):
                line = line.strip()
                if not line or line.endswith("changed"):
                    continue
                parts = line.split("|")
                if parts:
                    rel_path = parts[0].strip()
                    changed_files.append(rel_path)

        sections_affected = set()
        for file_path in changed_files:
            affected = _map_file_to_sections(file_path)
            sections_affected.update(affected)

        return StalenessResult(
            stale=True,
            old_commit=old_commit_hash,
            current_commit=current_commit,
            changed_files_count=changed_files_count,
            insertions=insertions,
            deletions=deletions,
            sections_affected=sorted(list(sections_affected)),
            recommend_new_snapshot=recommend_force_rebuild,
        )

    except Exception:
        return StalenessResult(stale=False)
