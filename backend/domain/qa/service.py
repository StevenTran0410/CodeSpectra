"""QA service — orchestrates question answering."""

from __future__ import annotations

from domain.model_connector.service import ProviderConfigService
from domain.retrieval.service import RetrievalService

from .agent import QAAgent
from .types import QARequest, QAResponse
from .deep_research import DeepResearchAgent, DeepResearchRequest, DeepResearchResult


class QAService:
    def __init__(
        self,
        provider_service: ProviderConfigService,
        retrieval_service: RetrievalService,
    ) -> None:
        self._provider = provider_service
        self._retrieval = retrieval_service
        self._agent = QAAgent(provider_service, retrieval_service)
        self._deep_research_agent = DeepResearchAgent(provider_service, retrieval_service)

    async def ask(self, req: QARequest) -> QAResponse:
        raw = await self._agent.run(
            provider_id=req.provider_id,
            model_id=req.model_id,
            snapshot_id=req.snapshot_id,
            question=req.question,
            include_debug=req.include_debug,
        )
        return QAResponse.model_validate(raw)

    async def deep_research(self, req: DeepResearchRequest) -> DeepResearchResult:
        raw = await self._deep_research_agent.research(
            question=req.question,
            snapshot_id=req.snapshot_id,
            provider_id=req.provider_id,
            model_id=req.model_id,
            max_hops=req.max_hops,
            include_debug=req.include_debug,
        )
        return DeepResearchResult.model_validate(raw)
