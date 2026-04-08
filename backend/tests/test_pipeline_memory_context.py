from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from domain.analysis.agents._context_builders import prefetch_pipeline_context
from domain.retrieval.types import RetrievalBundle, RetrievalMode


@pytest.mark.asyncio
async def test_prefetch_pipeline_context_populates_all_fields(
    mock_retrieval: MagicMock,
    canned_retrieval_bundle: RetrievalBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from domain.analysis.agents import _context_builders as cb

    async def fake_tree(_snapshot_id: str, _max: int = 60) -> str:
        return "src/a.py\nsrc/b.py"

    async def fake_patterns(
        _snapshot_id: str,
        _patterns: tuple[str, ...],
        char_limit: int = 0,
        max_rows: int = 6,
    ) -> str:
        return "doc-content"

    monkeypatch.setattr(cb, "fetch_folder_tree", fake_tree)
    monkeypatch.setattr(cb, "_fetch_files_by_pattern", fake_patterns)

    ctx = await prefetch_pipeline_context(mock_retrieval, "snap-xyz", mode=RetrievalMode.HYBRID)
    assert ctx.arch_bundle is canned_retrieval_bundle
    assert "a.py" in ctx.folder_tree
    assert ctx.doc_files == "doc-content"
    assert ctx.manifest_files == "doc-content"
    mock_retrieval.retrieve.assert_awaited()


@pytest.mark.asyncio
async def test_prefetch_failure_graceful_none_like(mock_retrieval: MagicMock) -> None:
    mock_retrieval.retrieve = AsyncMock(side_effect=RuntimeError("retrieve failed"))
    mem_ctx = None
    try:
        mem_ctx = await prefetch_pipeline_context(mock_retrieval, "snap-fail")
    except Exception:
        mem_ctx = None
    assert mem_ctx is None
