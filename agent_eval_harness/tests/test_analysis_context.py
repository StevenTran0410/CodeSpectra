"""Async tests for ProjectContext loading from code-analysis reports."""
from __future__ import annotations

from typing import Any

import pytest

from agent_eval_harness.discovery.analysis_context import (
    ProjectContext,
    ReportSection,
    load_project_context,
)


class _StubClientForAnalysis:
    """Duck-typed CodeSpectraClient stub for testing load_project_context."""

    def __init__(
        self,
        snapshot: dict[str, Any],
        repo: dict[str, Any],
        sibling_repos: list[dict[str, Any]],
        reports: list[dict[str, Any]],
        full_report: dict[str, Any],
    ):
        self._snapshot = snapshot
        self._repo = repo
        self._sibling_repos = sibling_repos
        self._reports = reports
        self._full_report = full_report

    async def get_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        return self._snapshot

    async def get_repo(self, repo_id: str) -> dict[str, Any]:
        return self._repo

    async def list_repos(
        self, workspace_id: str | None = None, mode: str | None = None
    ) -> list[dict[str, Any]]:
        return self._sibling_repos

    async def list_reports(
        self, repo_id: str | None = None, workspace_id: str | None = None, limit: int = 30
    ) -> list[dict[str, Any]]:
        return self._reports

    async def get_report(self, report_id: str) -> dict[str, Any]:
        return self._full_report

    async def aclose(self) -> None:
        pass


def _make_report_section_dict(
    identity: dict[str, Any] | None = None,
    synthesis: dict[str, Any] | None = None,
    architecture: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Helper to create a full report dict with sections keyed by letter."""
    sections = {}
    if identity:
        sections["A"] = identity
    if synthesis:
        sections["L"] = synthesis
    if architecture:
        sections["B"] = architecture
    return {
        "report": {
            "sections": sections,
            "commit_hash": "abc123def456",
        }
    }


@pytest.mark.asyncio
async def test_load_project_context_with_valid_sections() -> None:
    """Load context when all API calls succeed and sections exist."""
    snapshot = {"local_repo_id": "repo-1", "commit_hash": "abc123def456"}
    repo = {"path": "/app/src", "workspace_id": "ws-1", "id": "repo-1"}
    sibling_repos = [
        {
            "id": "repo-2",
            "path": "/app/src",
            "mode": "code_analysis",
        }
    ]
    reports = [
        {
            "id": "report-1",
        }
    ]
    full_report = _make_report_section_dict(
        identity={
            "purpose": "Data pipeline",
            "domain": "ML",
            "runtime_type": "Batch",
            "tech_stack": "Python",
            "key_frameworks": ["Haystack"],
        },
        synthesis={
            "executive_summary": "Overview",
            "architecture_narrative": "Details",
            "reading_path": "Start here",
        },
    )

    client = _StubClientForAnalysis(snapshot, repo, sibling_repos, reports, full_report)
    ctx = await load_project_context(client, "snap-1")

    assert ctx is not None
    assert ctx.identity is not None
    assert ctx.identity.content["purpose"] == "Data pipeline"
    assert ctx.identity.stale is False
    assert ctx.synthesis is not None
    assert ctx.synthesis.content["executive_summary"] == "Overview"
    assert ctx.report_snapshot_id == "report-1"


@pytest.mark.asyncio
async def test_load_project_context_returns_none_on_get_snapshot_error() -> None:
    """Return None if get_snapshot raises an exception."""

    class _ErrorClient:
        async def get_snapshot(self, snapshot_id: str) -> dict:
            raise RuntimeError("API error")

        async def aclose(self) -> None:
            pass

    client = _ErrorClient()
    ctx = await load_project_context(client, "snap-1")
    assert ctx is None


@pytest.mark.asyncio
async def test_load_project_context_returns_none_when_no_sibling_repo() -> None:
    """Return None if no sibling repo is found."""
    snapshot = {"local_repo_id": "repo-1", "commit_hash": "abc123def456"}
    repo = {"path": "/app/src", "workspace_id": "ws-1", "id": "repo-1"}
    sibling_repos = [
        {
            "id": "repo-2",
            "path": "/other/path",  # Different path, no match
            "mode": "code_analysis",
        }
    ]

    client = _StubClientForAnalysis(
        snapshot, repo, sibling_repos, [], {"report": {}}
    )
    ctx = await load_project_context(client, "snap-1")
    assert ctx is None


@pytest.mark.asyncio
async def test_load_project_context_returns_none_when_no_reports() -> None:
    """Return None if no reports are found for the sibling repo."""
    snapshot = {"local_repo_id": "repo-1", "commit_hash": "abc123def456"}
    repo = {"path": "/app/src", "workspace_id": "ws-1", "id": "repo-1"}
    sibling_repos = [
        {
            "id": "repo-2",
            "path": "/app/src",
            "mode": "code_analysis",
        }
    ]
    reports: list[dict[str, Any]] = []  # Empty list

    client = _StubClientForAnalysis(snapshot, repo, sibling_repos, reports, {})
    ctx = await load_project_context(client, "snap-1")
    assert ctx is None


@pytest.mark.asyncio
async def test_load_project_context_staleness_detection() -> None:
    """Mark context as stale when snapshot commit differs from report commit."""
    snapshot = {
        "local_repo_id": "repo-1",
        "commit_hash": "snapshot_commit_hash",
    }
    repo = {"path": "/app/src", "workspace_id": "ws-1", "id": "repo-1"}
    sibling_repos = [{"id": "repo-2", "path": "/app/src", "mode": "code_analysis"}]
    reports = [{"id": "report-1"}]
    full_report = {
        "report": {
            "sections": {
                "A": {"purpose": "Test"},
            },
            "commit_hash": "different_commit_hash",
        }
    }

    client = _StubClientForAnalysis(snapshot, repo, sibling_repos, reports, full_report)
    ctx = await load_project_context(client, "snap-1")

    assert ctx is not None
    assert ctx.is_stale is True
    assert ctx.identity is not None
    assert ctx.identity.stale is True


@pytest.mark.asyncio
async def test_load_project_context_missing_commit_hash_not_stale() -> None:
    """Don't mark as stale if either commit hash is missing."""
    snapshot = {"local_repo_id": "repo-1"}  # No commit_hash
    repo = {"path": "/app/src", "workspace_id": "ws-1", "id": "repo-1"}
    sibling_repos = [{"id": "repo-2", "path": "/app/src", "mode": "code_analysis"}]
    reports = [{"id": "report-1"}]
    full_report = {
        "report": {
            "sections": {"A": {"purpose": "Test"}},
            "commit_hash": "some_hash",
        }
    }

    client = _StubClientForAnalysis(snapshot, repo, sibling_repos, reports, full_report)
    ctx = await load_project_context(client, "snap-1")

    assert ctx is not None
    assert ctx.is_stale is False


@pytest.mark.asyncio
async def test_load_project_context_partial_sections() -> None:
    """Load context when only some sections are present."""
    snapshot = {"local_repo_id": "repo-1", "commit_hash": "abc123"}
    repo = {"path": "/app/src", "workspace_id": "ws-1", "id": "repo-1"}
    sibling_repos = [{"id": "repo-2", "path": "/app/src", "mode": "code_analysis"}]
    reports = [{"id": "report-1"}]
    full_report = {
        "report": {
            "sections": {
                "A": {"purpose": "Test"},
                # No L (synthesis), no B (architecture)
            },
            "commit_hash": "abc123",
        }
    }

    client = _StubClientForAnalysis(snapshot, repo, sibling_repos, reports, full_report)
    ctx = await load_project_context(client, "snap-1")

    assert ctx is not None
    assert ctx.identity is not None
    assert ctx.synthesis is None
    assert ctx.architecture is None


@pytest.mark.asyncio
async def test_report_section_as_context_block_format() -> None:
    """Test ReportSection.as_context_block formats correctly."""
    section = ReportSection(
        content={
            "key1": "value1",
            "key2": "value2",
            "key3": None,  # Falsy values should be skipped
        },
        stale=False,
    )
    block = section.as_context_block()
    assert "- key1: value1" in block
    assert "- key2: value2" in block
    assert "key3" not in block
    assert "stale" not in block.lower()


@pytest.mark.asyncio
async def test_report_section_as_context_block_stale_marker() -> None:
    """Test ReportSection.as_context_block includes stale marker."""
    section = ReportSection(
        content={"key": "value"},
        stale=True,
    )
    block = section.as_context_block()
    assert "[NOTE: analysis may be stale]" in block
    assert "- key: value" in block
