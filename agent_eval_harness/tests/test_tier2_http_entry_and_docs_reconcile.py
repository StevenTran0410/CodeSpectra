"""Tests for tier-2 HTTP entry points and doc-reconciliation."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_eval_harness.instrumentation.tier2_boundary import _resolve_entry_point
from agent_eval_harness.mapping.builder.pipeline import SystemMapBuilder
from agent_eval_harness.mapping.system_map import Component


@pytest.mark.asyncio
async def test_resolve_http_entry_point() -> None:
    """Verify that resolving an HTTP entry point returns an HTTP post wrapper client."""
    fn = _resolve_entry_point("http://localhost:9999/api/agent")
    
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"answer": "HTTP Success", "model": "mock-llm", "tokens_in": 10, "tokens_out": 20}
    mock_resp.status_code = 200
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        
        out = await fn("hello query", {"some": "prior"})
        
        mock_post.assert_called_once_with(
            "http://localhost:9999/api/agent",
            json={"query": "hello query", "prior_output": {"some": "prior"}},
            timeout=60.0,
        )
        assert out == {"answer": "HTTP Success", "model": "mock-llm", "tokens_in": 10, "tokens_out": 20}


@pytest.mark.asyncio
async def test_reconcile_docs_matches_components(tmp_path) -> None:
    """Verify that _reconcile_docs identifies components missing from hand-written docs."""
    builder = SystemMapBuilder(tmp_path)
    
    doc_file = tmp_path / "arch.md"
    doc_file.write_text("# System Architecture\n\nThis mentions component: retriever.", encoding="utf-8")
    
    components = [
        Component(id="retriever", role="agent", entry_point="main:ret"),
        Component(id="writer", role="agent", entry_point="main:writer"),
    ]
    
    discrepancies = await builder._reconcile_docs(doc_file, components)
    
    # retriever is mentioned in doc, but writer is not mentioned
    assert len(discrepancies) == 1
    assert "writer" in discrepancies[0]
    assert "retriever" not in discrepancies[0]
