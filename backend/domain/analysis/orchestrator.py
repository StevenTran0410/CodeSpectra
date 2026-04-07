"""Analysis orchestration — wires static analysis, graph summary, and agent pipeline."""
from __future__ import annotations

from typing import Any

from domain.model_connector.service import ProviderConfigService
from domain.retrieval.service import RetrievalService
from domain.structural_graph.service import StructuralGraphService
from domain.structural_graph.types import StructuralGraphSummary
from infrastructure.db.database import get_db
from shared.logger import logger

from .agent_pipeline import AnalysisAgentPipeline
from .static_convention import run_convention_analysis
from .static_risk import run_risk_analysis


class RunDirectorAgent:
    def __init__(
        self,
        providers: ProviderConfigService,
        retrieval: RetrievalService,
        pipeline: AnalysisAgentPipeline,
        graph: StructuralGraphService,
    ) -> None:
        self._providers = providers
        self._pipeline = pipeline
        self._graph = graph

    async def run(
        self,
        provider_id: str,
        model_id: str,
        snapshot_id: str,
        scan_mode: str,
        repo_name: str = "",
    ) -> dict[str, Any]:
        db = get_db()
        static_risk = None
        static_conv = None
        try:
            static_risk = await run_risk_analysis(snapshot_id, db)
            logger.info(
                "Static risk: %d findings for snapshot %s",
                len(static_risk.findings),
                snapshot_id,
            )
        except Exception as e:
            logger.warning("Static risk analysis failed: %s", e)

        try:
            static_conv = await run_convention_analysis(snapshot_id, db)
            logger.info(
                "Static convention: %d signals for snapshot %s",
                len(static_conv.signals),
                snapshot_id,
            )
        except Exception as e:
            logger.warning("Static convention analysis failed: %s", e)

        graph_summary: StructuralGraphSummary | None = None
        try:
            graph_summary = await self._graph.summary(snapshot_id)
            logger.info(
                "Graph summary: %d central files",
                len(graph_summary.top_central_files) if graph_summary else 0,
            )
        except Exception as e:
            logger.warning("Graph summary failed: %s", e)

        return await self._pipeline.run(
            provider_id=provider_id,
            model_id=model_id,
            snapshot_id=snapshot_id,
            repo_name=repo_name,
            graph_summary=graph_summary,
            static_risk=static_risk,
            static_convention=static_conv,
        )
