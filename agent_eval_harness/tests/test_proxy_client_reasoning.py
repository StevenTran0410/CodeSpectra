"""CodeSpectraProxyClient attaches its constructor-level reasoning_effort/thinking_budget to every outgoing request body without any AEH call site needing to know about them."""
from __future__ import annotations

import json

import httpx
import pytest

from agent_eval_harness.llm.client import LLMMessage
from agent_eval_harness.llm.proxy_client import CodeSpectraProxyClient


class _CapturingTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.last_body: dict | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.last_body = json.loads(request.content)
        return httpx.Response(
            200,
            json={"content": "ok", "model_id": "m", "prompt_tokens": 1, "completion_tokens": 1},
            request=request,
        )


async def test_reasoning_fields_attached_when_configured() -> None:
    transport = _CapturingTransport()
    http_client = httpx.AsyncClient(transport=transport)
    client = CodeSpectraProxyClient(
        "http://test", "tok", "provider-x", "model-y",
        http_client=http_client,
        reasoning_effort="high",
        thinking_budget=4096,
    )

    await client.complete([LLMMessage(role="user", content="hi")])

    assert transport.last_body is not None
    assert transport.last_body["reasoning_effort"] == "high"
    assert transport.last_body["thinking_budget"] == 4096
    await client.aclose()


async def test_reasoning_fields_absent_by_default() -> None:
    transport = _CapturingTransport()
    http_client = httpx.AsyncClient(transport=transport)
    client = CodeSpectraProxyClient(
        "http://test", "tok", "provider-x", "model-y", http_client=http_client,
    )

    await client.complete([LLMMessage(role="user", content="hi")])

    assert transport.last_body is not None
    assert transport.last_body["reasoning_effort"] is None
    assert transport.last_body["thinking_budget"] is None
    await client.aclose()


async def test_per_call_reasoning_effort_overrides_constructor_default() -> None:
    """A caller doing bounded structured JSON extraction can force a low effort for just its own call, regardless of the client's configured default."""
    transport = _CapturingTransport()
    http_client = httpx.AsyncClient(transport=transport)
    client = CodeSpectraProxyClient(
        "http://test", "tok", "provider-x", "model-y",
        http_client=http_client,
        reasoning_effort="high",
    )

    await client.complete([LLMMessage(role="user", content="hi")], reasoning_effort="low")

    assert transport.last_body is not None
    assert transport.last_body["reasoning_effort"] == "low"
    await client.aclose()
