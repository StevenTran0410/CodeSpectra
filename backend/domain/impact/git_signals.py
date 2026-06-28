"""Git-history co-change signal for blast-radius (CS-246).

Confirmed gap vs Greptile: ImpactService's blast_radius/plan relied solely on
the structural graph (cone/BFS, centrality, communities, call chains) with no
git-log integration. This module adds an additive "files that historically
changed together with the seed files" hint — it does not replace or alter the
existing graph-based impact cone.
"""
from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from shared.logger import logger

_COMMIT_MARKER = "__CS246_COMMIT__"
_MAX_COMMITS = 200


def _run_git_log(local_path: str, max_commits: int) -> str:
    result = subprocess.run(
        [
            "git",
            "log",
            f"-n{max_commits}",
            "--name-only",
            f"--pretty=format:{_COMMIT_MARKER}",
        ],
        cwd=local_path,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git log failed")
    return result.stdout


def _parse_commits(raw: str) -> list[list[str]]:
    """Split raw `git log --name-only` output into per-commit file lists."""
    commits: list[list[str]] = []
    current: list[str] = []
    for line in raw.splitlines():
        if line == _COMMIT_MARKER:
            if current:
                commits.append(current)
            current = []
        elif line.strip():
            current.append(line.strip())
    if current:
        commits.append(current)
    return commits


def _compute_cochange(
    commits: list[list[str]],
    seed_files: list[str],
    limit: int,
) -> dict[str, list[dict]]:
    """For each seed file, count how often other files appear in the same commit."""
    seed_set = set(seed_files)
    counts: dict[str, dict[str, int]] = {s: {} for s in seed_files}

    for files in commits:
        present_seeds = [s for s in seed_set if s in files]
        if not present_seeds:
            continue
        file_set = set(files)
        for s in present_seeds:
            bucket = counts[s]
            for f in file_set:
                if f == s:
                    continue
                bucket[f] = bucket.get(f, 0) + 1

    out: dict[str, list[dict]] = {}
    for s, bucket in counts.items():
        if not bucket:
            continue
        ranked = sorted(bucket.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
        out[s] = [{"rel_path": path, "cochange_count": count} for path, count in ranked]
    return out


async def get_cochange_files(
    local_path: str,
    seed_files: list[str],
    limit: int = 10,
    max_commits: int = _MAX_COMMITS,
) -> dict[str, list[dict]]:
    """Returns seed_file -> list of {rel_path, cochange_count}, sorted by count desc.

    Best-effort: returns {} on any git failure (missing repo, not a git repo,
    git not installed, timeout) rather than raising — this is an additive hint,
    not a required signal.
    """
    if not seed_files:
        return {}
    root = Path(local_path)
    if not root.exists() or not (root / ".git").exists():
        return {}
    try:
        raw = await asyncio.to_thread(_run_git_log, local_path, max_commits)
    except Exception as e:
        logger.warning("[git_signals] git log failed for %s: %s", local_path, e)
        return {}

    commits = _parse_commits(raw)
    return _compute_cochange(commits, seed_files, limit)
