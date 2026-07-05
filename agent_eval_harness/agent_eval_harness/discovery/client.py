"""CodeSpectra API external client for AEH.

Enables query of CodeSpectra's backend services over the gated /api/external endpoints.
"""
from __future__ import annotations

from typing import Any
import httpx


class CodeSpectraClient:
    """REST client wrapper around CodeSpectra's external token-gated API endpoints."""

    def __init__(
        self,
        base_url: str,
        bearer_token: str,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = bearer_token
        self._http = http_client or httpx.AsyncClient(timeout=60.0)

    async def _request(self, method: str, path: str, **kwargs) -> Any:
        headers = kwargs.pop("headers", {})
        headers["Authorization"] = f"Bearer {self._token}"
        resp = await self._http.request(
            method, f"{self._base_url}/api/external{path}", headers=headers, **kwargs
        )
        resp.raise_for_status()
        if resp.status_code == 204:
            return None
        return resp.json()

    async def search_retrieval(self, snapshot_id: str, query: str, section: str = "qa") -> dict:
        """Backed by retrieve_rrf_fusion (unbounded, BM25-weighted fused results
        under the "fused" key) — not the budget-capped retrieve(). Pass a bare
        keyword (e.g. "haystack"), not a natural-language phrase: a verbose
        query dilutes BM25's exact-term signal against unrelated chunks that
        happen to share other words."""
        return await self._request(
            "POST",
            "/retrieval/search",
            json={
                "snapshot_id": snapshot_id,
                "query": query,
                "section": section,
            },
        )

    async def get_neighbors(
        self, snapshot_id: str, seed_path: str, hops: int = 1, limit: int = 300
    ) -> dict:
        return await self._request(
            "GET",
            f"/graph/{snapshot_id}/neighbors",
            params={
                "seed_path": seed_path,
                "hops": hops,
                "limit": limit,
            },
        )

    async def get_communities(self, snapshot_id: str) -> dict:
        return await self._request("GET", f"/graph/{snapshot_id}/communities")

    async def get_symbol_edges(self, snapshot_id: str, file_path: str) -> dict:
        return await self._request(
            "GET",
            f"/graph/{snapshot_id}/symbol-edges",
            params={"file_path": file_path},
        )

    async def search_repo_map(self, snapshot_id: str, q: str, limit: int = 120) -> dict:
        return await self._request(
            "GET",
            f"/repo-map/{snapshot_id}/search",
            params={"q": q, "limit": limit},
        )

    async def read_file(
        self, snapshot_id: str, rel_path: str, max_bytes: int = 200_000
    ) -> dict:
        return await self._request(
            "GET",
            f"/manifest/{snapshot_id}/file",
            params={"rel_path": rel_path, "max_bytes": max_bytes},
        )

    async def list_reports(
        self,
        repo_id: str | None = None,
        workspace_id: str | None = None,
        limit: int = 30,
    ) -> list[dict]:
        params: dict[str, Any] = {"limit": limit}
        if repo_id:
            params["repo_id"] = repo_id
        if workspace_id:
            params["workspace_id"] = workspace_id
        return await self._request("GET", "/analysis/reports", params=params)

    async def get_report(self, report_id: str) -> dict:
        return await self._request("GET", f"/analysis/reports/{report_id}")

    async def get_snapshot(self, snapshot_id: str) -> dict:
        return await self._request("GET", f"/snapshots/{snapshot_id}")

    async def get_repo(self, repo_id: str) -> dict:
        return await self._request("GET", f"/repos/{repo_id}")

    async def list_repos(
        self, workspace_id: str | None = None, mode: str | None = None
    ) -> list[dict]:
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if mode:
            params["mode"] = mode
        return await self._request("GET", "/repos", params=params)

    async def aclose(self) -> None:
        await self._http.aclose()
