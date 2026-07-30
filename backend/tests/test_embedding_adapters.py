from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

from domain.model_connector.openai.adapter import OpenAIAdapter
from domain.model_connector.gemini.adapter import GeminiAdapter
from domain.model_connector.anthropic.adapter import AnthropicAdapter
from domain.model_connector.deepseek.adapter import DeepSeekAdapter
from domain.model_connector.errors import ProviderError
from domain.model_connector.types import EmbedRequest, ProviderConfig, ProviderKind


def _config(kind: ProviderKind, model_id: str) -> ProviderConfig:
    return ProviderConfig(
        id="p1",
        kind=kind,
        display_name="test",
        base_url="http://test",
        model_id=model_id,
        extra={"api_key": "key"},
    )


def _fake_response(body: dict) -> httpx.Response:
    return httpx.Response(200, json=body, request=httpx.Request("POST", "http://test/x"))


# ── OpenAI Embeddings ────────────────────────────────────────────────────────


async def test_openai_embedding_success() -> None:
    adapter = OpenAIAdapter(_config(ProviderKind.OPENAI, "text-embedding-3-small"))
    adapter._client.post = AsyncMock(
        return_value=_fake_response(
            {
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 8, "total_tokens": 8},
            }
        )
    )

    req = EmbedRequest(
        provider_id="p1", model_id="text-embedding-3-small", texts=["hello world"]
    )
    resp = await adapter.embed(req)

    assert resp.provider_id == "p1"
    assert resp.model_id == "text-embedding-3-small"
    assert resp.embeddings == [[0.1, 0.2, 0.3]]
    assert resp.dimensions == 3

    # Check payload structure
    sent = adapter._client.post.call_args.kwargs["json"]
    assert sent["model"] == "text-embedding-3-small"
    assert sent["input"] == ["hello world"]


# ── Gemini Embeddings ────────────────────────────────────────────────────────


async def test_gemini_embedding_success() -> None:
    adapter = GeminiAdapter(_config(ProviderKind.GEMINI, "gemini-embedding-001"))
    adapter._client.post = AsyncMock(
        return_value=_fake_response(
            {"embeddings": [{"values": [0.4, 0.5, 0.6]}]}
        )
    )

    req = EmbedRequest(
        provider_id="p1",
        model_id="gemini-embedding-001",
        texts=["hello document"],
        task_type="retrieval_document",
    )
    resp = await adapter.embed(req)

    assert resp.provider_id == "p1"
    assert resp.model_id == "gemini-embedding-001"
    assert resp.embeddings == [[0.4, 0.5, 0.6]]
    assert resp.dimensions == 3

    # Check payload structure & taskType mapping
    sent = adapter._client.post.call_args.kwargs["json"]
    assert len(sent["requests"]) == 1
    assert sent["requests"][0]["model"] == "models/gemini-embedding-001"
    assert sent["requests"][0]["content"]["parts"][0]["text"] == "hello document"
    assert sent["requests"][0]["taskType"] == "RETRIEVAL_DOCUMENT"


async def test_gemini_embedding_query_task() -> None:
    adapter = GeminiAdapter(_config(ProviderKind.GEMINI, "gemini-embedding-001"))
    adapter._client.post = AsyncMock(
        return_value=_fake_response(
            {"embeddings": [{"values": [0.7, 0.8]}]}
        )
    )

    req = EmbedRequest(
        provider_id="p1",
        model_id="gemini-embedding-001",
        texts=["hello query"],
        task_type="retrieval_query",
    )
    await adapter.embed(req)

    sent = adapter._client.post.call_args.kwargs["json"]
    assert sent["requests"][0]["taskType"] == "RETRIEVAL_QUERY"


# ── Unsupported Cloud Providers ──────────────────────────────────────────────


async def test_anthropic_embedding_raises() -> None:
    adapter = AnthropicAdapter(_config(ProviderKind.ANTHROPIC, "claude-3-5-sonnet"))
    req = EmbedRequest(provider_id="p1", texts=["should fail"])
    with pytest.raises(ProviderError) as exc_info:
        await adapter.embed(req)
    assert "Anthropic has no embeddings API" in str(exc_info.value)


async def test_deepseek_embedding_raises() -> None:
    adapter = DeepSeekAdapter(_config(ProviderKind.DEEPSEEK, "deepseek-chat"))
    req = EmbedRequest(provider_id="p1", texts=["should fail"])
    with pytest.raises(ProviderError) as exc_info:
        await adapter.embed(req)
    assert "DeepSeek has no embeddings API" in str(exc_info.value)


# ── Local Embedding & Gating ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_local_embedding_available_disabled(monkeypatch) -> None:
    from domain.embeddings.local_model import local_embedding_available
    from infrastructure.db.database import get_db

    # Force GPU to appear available
    monkeypatch.setattr("domain.embeddings.local_model.detect_gpu", lambda: (True, 8.0))

    # Mock DB select to return None or "false"
    class FakeCursor:
        async def fetchone(self):
            return {"value": "false"}
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class FakeDB:
        def execute(self, query, params=None):
            return FakeCursor()

    monkeypatch.setattr("infrastructure.db.database.get_db", lambda: FakeDB())

    res = await local_embedding_available()
    assert res is False


@pytest.mark.asyncio
async def test_local_embedding_available_enabled(monkeypatch) -> None:
    from domain.embeddings.local_model import local_embedding_available
    monkeypatch.setattr("domain.embeddings.local_model.detect_gpu", lambda: (True, 8.0))
    monkeypatch.setattr("domain.embeddings.local_model.embedding_weights_ready", lambda: True)

    class FakeCursor:
        async def fetchone(self):
            return {"value": "true"}
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class FakeDB:
        def execute(self, query, params=None):
            return FakeCursor()

    monkeypatch.setattr("infrastructure.db.database.get_db", lambda: FakeDB())

    res = await local_embedding_available()
    assert res is True


@pytest.mark.asyncio
async def test_local_embedding_available_disabled_when_weights_missing(monkeypatch) -> None:
    """GPU present + toggle on is not enough — weights must actually be on disk;
    no falling back to an auto-download attempt."""
    from domain.embeddings.local_model import local_embedding_available
    monkeypatch.setattr("domain.embeddings.local_model.detect_gpu", lambda: (True, 8.0))
    monkeypatch.setattr("domain.embeddings.local_model.embedding_weights_ready", lambda: False)

    res = await local_embedding_available()
    assert res is False


@pytest.mark.asyncio
async def test_llm_embed_offloads_local_to_thread(monkeypatch) -> None:
    import api.external as external
    from domain.model_connector.types import EmbedResponse
    import asyncio

    # Mock local_embedding_available to return True
    async def mock_available():
        return True

    called_with = []
    def mock_embed_texts(texts):
        called_with.append(texts)
        return [[0.1, 0.2]]

    monkeypatch.setattr("domain.embeddings.local_model.local_embedding_available", mock_available)
    monkeypatch.setattr("domain.embeddings.local_model.embed_texts", mock_embed_texts)

    # Wrap to verify that it uses asyncio.to_thread
    orig_to_thread = asyncio.to_thread
    to_thread_calls = []
    async def mock_to_thread(func, *args, **kwargs):
        to_thread_calls.append(func)
        return await orig_to_thread(func, *args, **kwargs)

    monkeypatch.setattr("asyncio.to_thread", mock_to_thread)

    body = external.LLMEmbedRequest(
        texts=["hello"],
        use_local=True,
    )
    res = await external.llm_embed(body)

    assert res.provider_id == "local"
    assert res.embeddings == [[0.1, 0.2]]
    assert mock_embed_texts in to_thread_calls
    assert called_with == [["hello"]]

