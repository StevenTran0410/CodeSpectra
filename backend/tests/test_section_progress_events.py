"""Backend progress updates tests."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from domain.analysis.agent_pipeline import _SectionAgentComponent


@pytest.mark.asyncio
async def test_section_agent_emits_running_status() -> None:
    """Verify that _SectionAgentComponent emits running and done status sequentially."""
    calls = []

    async def mock_on_section_done(section, status, duration_ms, data, error) -> None:
        calls.append((section, status, data))

    async def mock_runner(ctx, deps):
        return {"output_key": "some_value"}

    def mock_fallback(ctx, deps):
        return {}

    comp = _SectionAgentComponent(
        section="A",
        runner=mock_runner,
        fallback=mock_fallback,
        on_section_done=mock_on_section_done,
    )

    ctx = {"provider_id": "p", "model_id": "m", "snapshot_id": "s", "repo_name": "r"}
    await comp.run_async(ctx)

    # Should have two calls: 'running' first, then 'done'
    assert len(calls) == 2
    assert calls[0] == ("A", "running", None)
    assert calls[1] == ("A", "done", {"output_key": "some_value"})
