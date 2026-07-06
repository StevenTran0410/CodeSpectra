from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from agent_eval_harness.llm.client import LLMMessage, LLMResponse
from agent_eval_harness.llm.fake_client import FakeLLMClient
from agent_eval_harness.llm.routing_client import RoutingLLMClient, current_component_id_var


@pytest.mark.asyncio
async def test_routing_llm_client_contextvar() -> None:
    default_client = FakeLLMClient(LLMResponse(content="default", model="default-model"))
    override_client = FakeLLMClient(LLMResponse(content="override", model="override-model"))

    routing_client = RoutingLLMClient(default=default_client, overrides={"writer": override_client})

    # 1. No contextvar -> should use default client
    res = await routing_client.complete([LLMMessage(role="user", content="hello")])
    assert res.content == "default"
    assert len(default_client.calls) == 1
    assert len(override_client.calls) == 0

    # 2. Contextvar set to writer -> should use override client
    token = current_component_id_var.set("writer")
    try:
        res = await routing_client.complete([LLMMessage(role="user", content="hello")])
        assert res.content == "override"
        assert len(default_client.calls) == 1
        assert len(override_client.calls) == 1
    finally:
        current_component_id_var.reset(token)

    # 3. Contextvar set to retriever (no override) -> should use default client
    token = current_component_id_var.set("retriever")
    try:
        res = await routing_client.complete([LLMMessage(role="user", content="hello")])
        assert res.content == "default"
        assert len(default_client.calls) == 2
        assert len(override_client.calls) == 1
    finally:
        current_component_id_var.reset(token)


class _MockSpan:
    def __init__(self, tags):
        self.tags = tags


@contextmanager
def _patched_current_span(mock_span):
    """Temporarily replace haystack's global tracer.current_span() return value."""
    import haystack.tracing

    original = getattr(haystack.tracing.tracer, "current_span", None)
    haystack.tracing.tracer.current_span = MagicMock(return_value=mock_span)
    try:
        yield
    finally:
        if original is not None:
            haystack.tracing.tracer.current_span = original
        else:
            del haystack.tracing.tracer.current_span


@pytest.mark.asyncio
async def test_routing_llm_client_haystack_automatic_component_span() -> None:
    """Haystack's OWN automatic per-component spans (base.py's"""
    default_client = FakeLLMClient(LLMResponse(content="default", model="default-model"))
    override_client = FakeLLMClient(LLMResponse(content="override", model="override-model"))
    routing_client = RoutingLLMClient(default=default_client, overrides={"writer": override_client})

    with _patched_current_span(_MockSpan(tags={"haystack.component.name": "writer"})):
        res = await routing_client.complete([LLMMessage(role="user", content="hello")])
        assert res.content == "override"
        assert len(override_client.calls) == 1
        assert len(default_client.calls) == 0


@pytest.mark.asyncio
async def test_routing_llm_client_manual_span_tag() -> None:
    """Every real T1/T2 test-target component wraps its LLM call in"""
    default_client = FakeLLMClient(LLMResponse(content="default", model="default-model"))
    override_client = FakeLLMClient(LLMResponse(content="override", model="override-model"))
    routing_client = RoutingLLMClient(default=default_client, overrides={"writer": override_client})

    with _patched_current_span(_MockSpan(tags={"component_name": "writer"})):
        res = await routing_client.complete([LLMMessage(role="user", content="hello")])
        assert res.content == "override"
        assert len(override_client.calls) == 1
        assert len(default_client.calls) == 0


@pytest.mark.asyncio
async def test_routing_llm_client_real_target_tier1_e2e(tmp_path, monkeypatch) -> None:
    """End-to-end regression guard: run T1's REAL Haystack pipeline (not a mock"""
    import os

    from agent_eval_harness.metrics.sweep import run_sweep
    from agent_eval_harness.store import repository
    from agent_eval_harness.store.database import close_db, get_db, init_db

    monkeypatch.setenv("AEH_DATA_DIR", str(tmp_path))
    await init_db()
    try:
        override_client = FakeLLMClient(
            LLMResponse(content="override answer", model="override-fake-model-xyz")
        )
        routing_client = RoutingLLMClient(
            default=default_client_for_e2e(), overrides={"writer": override_client}
        )

        result = await run_sweep(
            target="test_targets.linear_rag.pipeline:build_pipeline",
            map_path="test_targets/linear_rag/system_map.yaml",
            suite_path="test_targets/linear_rag/suite.yaml",
            llm_client=routing_client,
            tier="1",
        )
        assert result.errors == []

        db = get_db()
        async with db.execute(
            "SELECT s.model FROM spans s JOIN traces t ON s.trace_id = t.id "
            "WHERE t.run_id = ? AND s.component_id = 'writer' AND s.span_type = 'llm_call'",
            (result.run_id,),
        ) as cur:
            rows = await cur.fetchall()

        assert rows, "expected at least one writer llm_call span"
        assert all(r["model"] == "override-fake-model-xyz" for r in rows), (
            "writer's LLM span did not record the overridden model — the override "
            "was silently ignored"
        )
    finally:
        await close_db()


def default_client_for_e2e() -> FakeLLMClient:
    return FakeLLMClient(LLMResponse(content="default answer", model="fake-default"))
