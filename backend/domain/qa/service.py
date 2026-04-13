"""QA service — orchestrates question answering."""

from __future__ import annotations

from domain.model_connector.service import ProviderConfigService
from domain.retrieval.service import RetrievalService

from .agent import QAAgent
from .types import QARequest, QAResponse


class QAService:
    def __init__(
        self,
        provider_service: ProviderConfigService,
        retrieval_service: RetrievalService,
    ) -> None:
        self._provider = provider_service
        self._retrieval = retrieval_service
        self._agent = QAAgent(provider_service, retrieval_service)

    async def ask(self, req: QARequest) -> QAResponse:
        raw = await self._agent.run(
            provider_id=req.provider_id,
            model_id=req.model_id,
            snapshot_id=req.snapshot_id,
            question=req.question,
            include_debug=req.include_debug,
        )
        return QAResponse.model_validate(raw)
