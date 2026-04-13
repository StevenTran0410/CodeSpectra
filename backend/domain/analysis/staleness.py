"""Staleness detection and incremental re-analysis support (CS-222)."""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path

import aiosqlite

from shared.git_utils import run_git_sync


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
        if not line:
            continue

        if line.endswith("changed"):
            match = re.search(r"(\d+)\s+files? changed", line)
            if match:
                file_count = int(match.group(1))

            match = re.search(r"(\d+)\s+insertions?\(\+\)", line)
            if match:
                insertions = int(match.group(1))

            match = re.search(r"(\d+)\s+deletions?\(-\)", line)
            if match:
                deletions = int(match.group(1))
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
            SELECT rs.local_repo_id, rs.commit_hash, rs.local_path
            FROM repo_snapshots rs
            WHERE rs.id = ?
            """,
            (snapshot_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return StalenessResult(stale=False)
            local_repo_id, old_commit_hash, local_path = row

        if not old_commit_hash or not local_path:
            return StalenessResult(stale=False)

        if not Path(local_path).exists():
            return StalenessResult(stale=False, old_commit=old_commit_hash)

        current_commit = run_git_sync(local_path, ["rev-parse", "HEAD"])
        if not current_commit:
            return StalenessResult(stale=False, old_commit=old_commit_hash)

        current_commit = current_commit.strip()

        if current_commit == old_commit_hash:
            return StalenessResult(stale=False, old_commit=old_commit_hash, current_commit=current_commit)

        diff_stat = run_git_sync(local_path, ["diff", "--stat", f"{old_commit_hash}...{current_commit}"])
        changed_files_count, insertions, deletions = _parse_git_diff_stat(diff_stat)

        if changed_files_count == 0:
            return StalenessResult(stale=False, old_commit=old_commit_hash, current_commit=current_commit)

        # Count commits between old and current
        rev_list = run_git_sync(
            local_path, ["rev-list", "--count", f"{old_commit_hash}..{current_commit}"]
        )
        commit_count = int(rev_list.strip()) if rev_list and rev_list.strip().isdigit() else 0

        is_ancestor = run_git_sync(
            local_path, ["merge-base", "--is-ancestor", old_commit_hash, current_commit]
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
