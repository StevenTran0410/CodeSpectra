"""AEH embedding client — Protocol definition and CodeSpectra proxy implementation.

EmbeddingClient is a runtime-checkable Protocol so callers can type-hint
against it without any concrete dependency. CodeSpectraEmbeddingProxyClient
mirrors CodeSpectraProxyClient's constructor signature and POSTs to the new
/api/external/llm/embed endpoint.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import httpx


@runtime_checkable
class EmbeddingClient(Protocol):
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts and return one float vector per input."""
        ...


class CodeSpectraEmbeddingProxyClient:
    """POSTs to CodeSpectra's /api/external/llm/embed endpoint.

    Constructor parameters mirror CodeSpectraProxyClient for consistency.
    ``provider_id`` and ``model_id`` are ignored when ``use_local=True``.
    """

    def __init__(
        self,
        base_url: str,
        bearer_token: str,
        provider_id: str | None = None,
        model_id: str | None = None,
        use_local: bool = False,
        task_type: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = bearer_token
        self._provider_id = provider_id
        self._model_id = model_id
        self._use_local = use_local
        self._task_type = task_type
        self._http = http_client or httpx.AsyncClient(timeout=120.0)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        body: dict = {"texts": texts, "use_local": self._use_local}
        if not self._use_local and self._provider_id:
            body["provider_id"] = self._provider_id
        if self._model_id:
            body["model_id"] = self._model_id
        if self._task_type:
            body["task_type"] = self._task_type

        resp = await self._http.post(
            f"{self._base_url}/api/external/llm/embed",
            headers={"Authorization": f"Bearer {self._token}"},
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["embeddings"]

    async def aclose(self) -> None:
        await self._http.aclose()
