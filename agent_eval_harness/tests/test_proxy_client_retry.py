"""Tests for CodeSpectraProxyClient's rate-limit retry/backoff (CS-263 §8)."""
from __future__ import annotations

import httpx
import pytest

from agent_eval_harness.llm.client import LLMMessage
from agent_eval_harness.llm.proxy_client import CodeSpectraProxyClient

pytestmark = pytest.mark.asyncio


def _response(status_code: int, json_body: dict) -> httpx.Response:
    return httpx.Response(
        status_code, json=json_body, request=httpx.Request("POST", "http://test/complete")
    )


class _ScriptedTransport(httpx.AsyncBaseTransport):
    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = list(responses)
        self.call_count = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.call_count += 1
        return self._responses.pop(0)


async def test_retries_429_then_succeeds(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(
        "asyncio.sleep", lambda seconds: sleeps.append(seconds) or _noop_coro()
    )

    transport = _ScriptedTransport(
        [
            _response(429, {"detail": "rate limited"}),
            _response(429, {"detail": "rate limited"}),
            _response(
                200,
                {"content": "ok", "model_id": "m", "prompt_tokens": 1, "completion_tokens": 1},
            ),
        ]
    )
    http_client = httpx.AsyncClient(transport=transport)
    client = CodeSpectraProxyClient("http://test", "tok", "provider-x", http_client=http_client)

    result = await client.complete([LLMMessage(role="user", content="hi")])

    assert result.content == "ok"
    assert transport.call_count == 3
    assert len(sleeps) == 2  # backed off twice before the third (successful) attempt
    await client.aclose()


async def test_exhausts_retries_and_raises(monkeypatch) -> None:
    monkeypatch.setattr("asyncio.sleep", lambda seconds: _noop_coro())

    # 1 initial attempt + 3 retries = 4 total 429 responses needed to exhaust retries
    transport = _ScriptedTransport([_response(429, {"detail": "rate limited"}) for _ in range(4)])
    http_client = httpx.AsyncClient(transport=transport)
    client = CodeSpectraProxyClient("http://test", "tok", "provider-x", http_client=http_client)

    with pytest.raises(httpx.HTTPStatusError):
        await client.complete([LLMMessage(role="user", content="hi")])

    assert transport.call_count == 4
    await client.aclose()


async def test_non_429_error_fails_immediately_no_retry(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(
        "asyncio.sleep", lambda seconds: sleeps.append(seconds) or _noop_coro()
    )

    transport = _ScriptedTransport([_response(500, {"detail": "server error"})])
    http_client = httpx.AsyncClient(transport=transport)
    client = CodeSpectraProxyClient("http://test", "tok", "provider-x", http_client=http_client)

    with pytest.raises(httpx.HTTPStatusError):
        await client.complete([LLMMessage(role="user", content="hi")])

    assert transport.call_count == 1
    assert sleeps == []
    await client.aclose()


async def _noop_coro() -> None:
    return None
