"""ProjectContext dataclasses and async loader for code-analysis integration (CS-290 §B1)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from agent_eval_harness.discovery.client import CodeSpectraClient
from agent_eval_harness.discovery.analysis_context_extractors import (
    _extract_important_files,
    _filter_risk_findings,
    _extract_identity,
    _extract_synthesis,
)

logger = logging.getLogger("agent_eval_harness.discovery.analysis_context")

# Re-export extractors at module level for consumer convenience
__all__ = [
    "ReportSection",
    "ProjectContext",
    "load_project_context",
    "_extract_important_files",
    "_filter_risk_findings",
    "_extract_identity",
    "_extract_synthesis",
]


@dataclass
class ReportSection:
    """A single analysis report section with optional staleness flag."""

    content: dict[str, Any]
    stale: bool = False

    def as_context_block(self) -> str:
        """Format section as plain text context block for LLM injection.

        Follows _encode_hits_toon style: f-string joins, ~5 lines.
        """
        prefix = "[NOTE: analysis may be stale]\n" if self.stale else ""
        lines = [f"- {k}: {v}" for k, v in self.content.items() if v]
        return prefix + "\n".join(lines) + "\n"


@dataclass
class ProjectContext:
    """Typed structure for code-analysis report sections and metadata.

    Each section is optional (defaulting to None if not found in the report).
    Fields are populated via extractors that handle defensive parsing.
    """

    identity: ReportSection | None = None
    synthesis: ReportSection | None = None
    architecture: ReportSection | None = None
    structure: ReportSection | None = None
    feature_map: ReportSection | None = None
    important_files: list[str] | None = None
    risk_findings: list[dict[str, Any]] | None = None
    report_snapshot_id: str | None = None
    report_commit_hash: str | None = None
    is_stale: bool = False


async def load_project_context(
    client: CodeSpectraClient, snapshot_id: str
) -> ProjectContext | None:
    """Load code-analysis report for a given snapshot and return typed ProjectContext.

    Implements the 5-call chain: get_snapshot → get_repo → list_repos (match by path)
    → list_reports → get_report. Gracefully degrades to None on any error.

    Args:
        client: CodeSpectraClient instance with bearer token.
        snapshot_id: Snapshot ID to load analysis for.

    Returns:
        ProjectContext with extracted sections, or None if any step fails.
    """
    try:
        # Step 1: Get snapshot
        snap = await client.get_snapshot(snapshot_id)
        local_repo_id = snap.get("local_repo_id")
        if not local_repo_id:
            logger.debug(f"No local_repo_id in snapshot {snapshot_id}")
            return None

        # Step 2: Get repo
        repo = await client.get_repo(local_repo_id)
        repo_path = repo.get("path")
        workspace_id = repo.get("workspace_id")
        if not repo_path or not workspace_id:
            logger.debug(f"Missing repo path or workspace in {local_repo_id}")
            return None

        # Step 3: Find sibling code-analysis repo by path
        all_repos = await client.list_repos(workspace_id=workspace_id, mode="code_analysis")
        sibling_repo = None
        for r in all_repos:
            if r.get("path") == repo_path:
                sibling_repo = r
                break

        if not sibling_repo:
            logger.debug(f"No sibling code-analysis repo found for path {repo_path}")
            return None

        # Step 4: Get latest report for sibling repo
        reports = await client.list_reports(repo_id=sibling_repo["id"])
        if not reports:
            logger.debug(f"No analysis reports found for repo {sibling_repo['id']}")
            return None

        latest_report_id = reports[0]["id"]

        # Step 5: Get full report and extract sections
        full_report = await client.get_report(latest_report_id)
        report_data = full_report.get("report", {})
        sections = report_data.get("sections") or {}

        # Build ReportSection objects from extractors
        # Check staleness by comparing commit hashes
        snapshot_commit_hash = (
            snap.get("commit_hash")
            or snap.get("commit")
            or snap.get("head_commit")
        )
        report_commit_hash = report_data.get("commit_hash")
        is_stale = (
            snapshot_commit_hash is not None
            and report_commit_hash is not None
            and snapshot_commit_hash != report_commit_hash
        )

        # Extract sections A-L keyed by letter
        identity_section = sections.get("A")
        synthesis_section = sections.get("L")
        architecture_section = sections.get("B")
        structure_section = sections.get("C")
        feature_map_section = sections.get("F")
        important_files_section = sections.get("G")
        risk_section = sections.get("J")

        # Build context object
        return ProjectContext(
            identity=(
                ReportSection(_extract_identity(identity_section), is_stale)
                if identity_section
                else None
            ),
            synthesis=(
                ReportSection(_extract_synthesis(synthesis_section), is_stale)
                if synthesis_section
                else None
            ),
            architecture=(
                ReportSection(architecture_section, is_stale)
                if architecture_section
                else None
            ),
            structure=(
                ReportSection(structure_section, is_stale)
                if structure_section
                else None
            ),
            feature_map=(
                ReportSection(feature_map_section, is_stale)
                if feature_map_section
                else None
            ),
            important_files=_extract_important_files(important_files_section),
            risk_findings=_filter_risk_findings(risk_section),
            report_snapshot_id=latest_report_id,
            report_commit_hash=report_commit_hash,
            is_stale=is_stale,
        )

    except Exception as exc:
        logger.debug(f"Failed to load project context for {snapshot_id}: {exc}")
        return None
