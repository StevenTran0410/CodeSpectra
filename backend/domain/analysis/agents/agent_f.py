"""Feature map agent (section F)."""

from __future__ import annotations

import time
from typing import Any

from domain.model_connector.service import ProviderConfigService
from domain.retrieval.service import RetrievalService
from domain.retrieval.types import RetrievalMode, RetrievalSection, RetrieveRequest
from domain.structural_graph.types import StructuralGraphSummary
from shared.logger import logger

from ..agent_pipeline import _normalize_conf
from ..prompts import AGENT_F_SCHEMA_STR, AGENT_F_SYSTEM, render_bundle
from ..schemas import validate_section
from .base import BaseTypedAgent

_COMBINED_QUERY = (
    "route definition controller use_case handler service public method "
    "routes.py router.ts views.py use_case feature flag entrypoint "
    "API endpoint handler request response workflow"
)


class AgentF(BaseTypedAgent):
    def __init__(
        self,
        provider_service: ProviderConfigService,
        retrieval_service: RetrievalService,
    ) -> None:
        super().__init__(provider_service)
        self._retrieval = retrieval_service

    def _fallback(self, reason: str) -> dict[str, Any]:
        return {
            "features": [],
            "confidence": "low",
            "evidence_files": [],
            "blind_spots": [f"Agent failed: {reason}"],
        }

    async def run(
        self,
        provider_id: str,
        model_id: str,
        snapshot_id: str,
        graph_summary: StructuralGraphSummary | None = None,
    ) -> dict[str, Any]:
        t0 = time.monotonic()
        n_chunks = 0
        try:
            graph_block = ""
            if graph_summary and graph_summary.top_central_files:
                lines = ["Graph centrality (top files by import score):"]
                for n in graph_summary.top_central_files[:10]:
                    lines.append(
                        f"  {n.rel_path} (score={n.score}, indegree={n.indegree})"
                    )
                graph_block = "\n".join(lines)
            prefix = f"{graph_block}\n\n" if graph_block else ""

            bundle = await self._retrieval.retrieve(
                RetrieveRequest(
                    snapshot_id=snapshot_id,
                    query=_COMBINED_QUERY,
                    section=RetrievalSection.FEATURE_MAP,
                    mode=RetrievalMode.HYBRID,
                    max_results=20,
                )
            )
            n_chunks = len(bundle.evidences)
            user_prompt = (
                f"{prefix}snapshot_id={snapshot_id}\n\nEvidence:\n{render_bundle(bundle)}"
            )
            data = await self._chat_json_typed(
                provider_id,
                model_id,
                AGENT_F_SYSTEM,
                user_prompt,
                schema_hint=AGENT_F_SCHEMA_STR,
                max_completion_tokens=16000,
            )

            raw_feat = data.get("features")
            feats: list[dict[str, Any]] = []
            if isinstance(raw_feat, list):
                for i in raw_feat:
                    if not isinstance(i, dict):
                        continue
                    kf = i.get("key_files")
                    ts = i.get("tests")
                    ro = i.get("reading_order")
                    feats.append(
                        {
                            "name": str(i.get("name", "") or ""),
                            "description": str(i.get("description", "") or ""),
                            "entrypoint": str(i.get("entrypoint", "") or ""),
                            "key_files": (
                                [str(x) for x in kf if x is not None]
                                if isinstance(kf, list)
                                else []
                            ),
                            "data_path": str(i.get("data_path", "") or ""),
                            "tests": (
                                [str(x) for x in ts if x is not None]
                                if isinstance(ts, list)
                                else []
                            ),
                            "reading_order": (
                                [str(x) for x in ro if x is not None]
                                if isinstance(ro, list)
                                else []
                            ),
                        }
                    )
            data["features"] = feats

            for key in ("evidence_files", "blind_spots"):
                raw = data.get(key)
                if isinstance(raw, list):
                    data[key] = [str(x) for x in raw if x is not None]
                else:
                    data[key] = []
            data["confidence"] = _normalize_conf(str(data.get("confidence", "medium")))
            validate_section("F", data)
            ms = int((time.monotonic() - t0) * 1000)
            logger.info("[AgentF] %d chunks retrieved, completed in %dms", n_chunks, ms)
            return data
        except Exception as e:
            logger.warning("[AgentF] failed: %s", e)
            ms = int((time.monotonic() - t0) * 1000)
            logger.info("[AgentF] %d chunks retrieved, completed in %dms", n_chunks, ms)
            return self._fallback(str(e))
