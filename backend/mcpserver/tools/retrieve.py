"""retrieve_context MCP tool."""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from ..project_index import get_snapshot_id


def register(mcp: FastMCP) -> None:

    @mcp.tool()
    async def retrieve_context(
        project_path: str,
        query: str,
        top_k: int = 10,
    ) -> list[dict]:
        """Hybrid semantic + BM25 search for relevant code chunks
        in an indexed project."""
        sid = get_snapshot_id(project_path)

        from domain.retrieval.service import RetrievalService
        from domain.retrieval.types import RetrieveRequest, RetrievalSection

        result = await RetrievalService().retrieve(
            RetrieveRequest(
                snapshot_id=sid,
                query=query,
                max_results=top_k,
                section=RetrievalSection.QA,
            )
        )
        return [
            {
                "file_path": e.rel_path,
                "excerpt": e.excerpt,
                "score": e.score,
                "token_estimate": e.token_estimate,
            }
            for e in result.evidences
        ]
