"""CS-270 — thin-delegation tests for the new /api/external/* read routes.

These routes exist purely so AEH's discovery engine can query CodeSpectra's
already-built indexes over HTTP (CS-260 §4 REST boundary). Each handler must
do nothing but call the corresponding domain service method and return its
result unchanged — this test proves exactly that (no route reimplements any
logic), matching this codebase's existing convention of testing at the
service-call boundary via mocks rather than spinning up a live HTTP server
(no existing backend test does the latter).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

import api.external as external


@pytest.mark.asyncio
async def test_search_retrieval_delegates_to_retrieval_service() -> None:
    from domain.retrieval.types import RetrieveRequest, RetrievalSection

    sentinel = object()
    with patch.object(external, "_retrieval_service") as mock_service:
        mock_service.retrieve = AsyncMock(return_value=sentinel)
        body = RetrieveRequest(snapshot_id="snap-1", query="q", section=RetrievalSection.QA)
        result = await external.search_retrieval(body)

    mock_service.retrieve.assert_awaited_once_with(body)
    assert result is sentinel


@pytest.mark.asyncio
async def test_get_graph_neighbors_delegates_with_correct_args() -> None:
    sentinel = object()
    with patch.object(external, "_graph_service") as mock_service:
        mock_service.neighbors = AsyncMock(return_value=sentinel)
        result = await external.get_graph_neighbors("snap-1", "some/path.py", hops=2, limit=50)

    mock_service.neighbors.assert_awaited_once_with("snap-1", "some/path.py", 2, 50)
    assert result is sentinel


@pytest.mark.asyncio
async def test_get_graph_communities_delegates() -> None:
    sentinel = object()
    with patch.object(external, "_graph_service") as mock_service:
        mock_service.list_communities = AsyncMock(return_value=sentinel)
        result = await external.get_graph_communities("snap-1")

    mock_service.list_communities.assert_awaited_once_with("snap-1")
    assert result is sentinel


@pytest.mark.asyncio
async def test_get_graph_symbol_edges_delegates() -> None:
    sentinel = object()
    with patch.object(external, "_graph_service") as mock_service:
        mock_service.symbol_edges_for_file = AsyncMock(return_value=sentinel)
        result = await external.get_graph_symbol_edges("snap-1", "some/path.py")

    mock_service.symbol_edges_for_file.assert_awaited_once_with("snap-1", "some/path.py")
    assert result is sentinel


@pytest.mark.asyncio
async def test_search_repo_map_symbols_delegates() -> None:
    sentinel = object()
    with patch.object(external, "_repo_map_service") as mock_service:
        mock_service.search = AsyncMock(return_value=sentinel)
        result = await external.search_repo_map_symbols("snap-1", "MyClass", limit=10)

    mock_service.search.assert_awaited_once_with("snap-1", "MyClass", 10)
    assert result is sentinel


@pytest.mark.asyncio
async def test_read_manifest_file_delegates() -> None:
    sentinel = object()
    with patch.object(external, "_manifest_service") as mock_service:
        mock_service.read_file = AsyncMock(return_value=sentinel)
        result = await external.read_manifest_file("snap-1", "src/foo.py", max_bytes=1000)

    mock_service.read_file.assert_awaited_once_with("snap-1", "src/foo.py", 1000)
    assert result is sentinel


@pytest.mark.asyncio
async def test_list_analysis_reports_delegates() -> None:
    sentinel = object()
    with patch.object(external, "_analysis_service") as mock_service:
        mock_service.list_reports = AsyncMock(return_value=sentinel)
        result = await external.list_analysis_reports(repo_id="repo-1", workspace_id="ws-1", limit=5)

    mock_service.list_reports.assert_awaited_once_with("repo-1", "ws-1", 5)
    assert result is sentinel


@pytest.mark.asyncio
async def test_get_analysis_report_delegates() -> None:
    sentinel = object()
    with patch.object(external, "_analysis_service") as mock_service:
        mock_service.get_report = AsyncMock(return_value=sentinel)
        result = await external.get_analysis_report("report-1")

    mock_service.get_report.assert_awaited_once_with("report-1")
    assert result is sentinel


@pytest.mark.asyncio
async def test_get_repo_snapshot_delegates() -> None:
    sentinel = object()
    with patch.object(external, "_sync_service") as mock_service:
        mock_service.get_snapshot = AsyncMock(return_value=sentinel)
        result = await external.get_repo_snapshot("snap-1")

    mock_service.get_snapshot.assert_awaited_once_with("snap-1")
    assert result is sentinel


@pytest.mark.asyncio
async def test_get_local_repo_delegates() -> None:
    sentinel = object()
    with patch.object(external, "_local_repo_service") as mock_service:
        mock_service.get_by_id = AsyncMock(return_value=sentinel)
        result = await external.get_local_repo("repo-1")

    mock_service.get_by_id.assert_awaited_once_with("repo-1")
    assert result is sentinel


@pytest.mark.asyncio
async def test_list_local_repos_delegates() -> None:
    sentinel = object()
    with patch.object(external, "_local_repo_service") as mock_service:
        mock_service.list_all = AsyncMock(return_value=sentinel)
        result = await external.list_local_repos(workspace_id="ws-1", mode="aeh")

    mock_service.list_all.assert_awaited_once_with("ws-1", "aeh")
    assert result is sentinel


@pytest.mark.asyncio
async def test_all_new_routes_are_token_gated() -> None:
    """Every new route must depend on require_external_token — a route that
    forgets this dependency would leak CodeSpectra's index to any local process
    without the bearer token, defeating the entire narrow-slice design."""
    gated_paths = {
        "/retrieval/search",
        "/graph/{snapshot_id}/neighbors",
        "/graph/{snapshot_id}/communities",
        "/graph/{snapshot_id}/symbol-edges",
        "/repo-map/{snapshot_id}/search",
        "/manifest/{snapshot_id}/file",
        "/analysis/reports",
        "/analysis/reports/{report_id}",
        "/snapshots/{snapshot_id}",
        "/repos/{repo_id}",
        "/repos",
    }
    found = {route.path for route in external.router.routes if route.path in gated_paths}
    assert found == gated_paths, f"missing routes: {gated_paths - found}"

    for route in external.router.routes:
        if route.path in gated_paths:
            dependant_calls = [d.call for d in route.dependant.dependencies]
            assert external.require_external_token in dependant_calls, (
                f"route {route.path} is missing the require_external_token dependency"
            )
