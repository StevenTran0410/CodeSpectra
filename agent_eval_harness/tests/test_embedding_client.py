from __future__ import annotations

import json
import httpx
import pytest

from agent_eval_harness.llm.embedding_client import CodeSpectraEmbeddingProxyClient
from agent_eval_harness.llm.deepeval_adapter import make_deepeval_embedding_model
from agent_eval_harness.llm.ragas_adapter import make_ragas_embeddings


class _CapturingTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.last_body: dict | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.last_body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "provider_id": self.last_body.get("provider_id", "local"),
                "model_id": self.last_body.get("model_id", "m"),
                "embeddings": [[0.11, 0.22] for _ in self.last_body.get("texts", [])],
                "dimensions": 2,
            },
            request=request,
        )


@pytest.mark.asyncio
async def test_embedding_proxy_client_payload() -> None:
    transport = _CapturingTransport()
    http_client = httpx.AsyncClient(transport=transport)
    client = CodeSpectraEmbeddingProxyClient(
        "http://test",
        "tok",
        provider_id="provider-x",
        model_id="model-y",
        use_local=False,
        task_type="retrieval_document",
        http_client=http_client,
    )

    embeddings = await client.embed_texts(["hello", "world"])

    assert embeddings == [[0.11, 0.22], [0.11, 0.22]]
    assert transport.last_body is not None
    assert transport.last_body["provider_id"] == "provider-x"
    assert transport.last_body["model_id"] == "model-y"
    assert transport.last_body["use_local"] is False
    assert transport.last_body["task_type"] == "retrieval_document"
    assert transport.last_body["texts"] == ["hello", "world"]
    await client.aclose()


@pytest.mark.asyncio
async def test_embedding_proxy_client_local() -> None:
    transport = _CapturingTransport()
    http_client = httpx.AsyncClient(transport=transport)
    client = CodeSpectraEmbeddingProxyClient(
        "http://test",
        "tok",
        use_local=True,
        http_client=http_client,
    )

    await client.embed_texts(["test local"])

    assert transport.last_body is not None
    assert "provider_id" not in transport.last_body
    assert transport.last_body["use_local"] is True
    assert transport.last_body["texts"] == ["test local"]
    await client.aclose()


class FakeEmbeddingClient:
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 0.5] for t in texts]


def test_deepeval_adapter_delegation() -> None:
    fake_client = FakeEmbeddingClient()
    adapter = make_deepeval_embedding_model(fake_client)

    vec = adapter.embed_text("cat")
    assert vec == [3.0, 0.5]

    vecs = adapter.embed_texts(["dog", "elephant"])
    assert vecs == [[3.0, 0.5], [8.0, 0.5]]


def test_ragas_adapter_delegation() -> None:
    fake_client = FakeEmbeddingClient()
    adapter = make_ragas_embeddings(fake_client)

    vecs = adapter.embed_documents(["apple", "banana"])
    assert vecs == [[5.0, 0.5], [6.0, 0.5]]

    vec = adapter.embed_query("kiwi")
    assert vec == [4.0, 0.5]
