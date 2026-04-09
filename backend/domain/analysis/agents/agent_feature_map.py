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
from ..profiles import NORMAL_PROFILE, AnalysisProfile
from ..prompts import AGENT_F_SCHEMA_STR, AGENT_F_SYSTEM, render_bundle
from ..schemas import validate_section
from ._context_builders import extract_a_identity_context, extract_b_arch_context
from .base import BaseTypedAgent

_COMBINED_QUERY = (
    "route definition controller use_case handler service public method "
    "routes.py router.ts views.py use_case feature flag entrypoint "
    "API endpoint handler request response workflow"
)


class FeatureMapAgent(BaseTypedAgent):
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
        identity_output: dict | None = None,
        architecture_output: dict | None = None,
        profile: AnalysisProfile | None = None,
    ) -> dict[str, Any]:
        t0 = time.monotonic()
        n_chunks = 0
        _profile = profile or NORMAL_PROFILE
        try:
            graph_block = ""
            if graph_summary and graph_summary.top_central_files:
                lines = ["Graph centrality (top files by import score):"]
                for n in graph_summary.top_central_files[:10]:
                    lines.append(f"  {n.rel_path} (score={n.score}, indegree={n.indegree})")
                graph_block = "\n".join(lines)
            identity_block = extract_a_identity_context(identity_output)
            arch_block = extract_b_arch_context(architecture_output)
            prefix_parts = []
            if identity_block:
                prefix_parts.append(identity_block)
            if arch_block:
                prefix_parts.append(arch_block)
            if graph_block:
                prefix_parts.append(graph_block)
            prefix = "\n\n".join(prefix_parts) + ("\n\n" if prefix_parts else "")

            bundle = await self._retrieval.retrieve(
                RetrieveRequest(
                    snapshot_id=snapshot_id,
                    query=_COMBINED_QUERY,
                    section=RetrievalSection.FEATURE_MAP,
                    mode=RetrievalMode.HYBRID,
                    max_results=_profile.retrieval_max_results,
                )
            )
            n_chunks = len(bundle.evidences)
            user_prompt = f"{prefix}snapshot_id={snapshot_id}\n\nEvidence:\n{render_bundle(bundle)}"
            data = await self._chat_json_typed(
                provider_id,
                model_id,
                AGENT_F_SYSTEM,
                user_prompt,
                schema_hint=AGENT_F_SCHEMA_STR,
                max_completion_tokens=_profile.tokens_feature_map,
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
            logger.info("[FeatureMapAgent] %d chunks retrieved, completed in %dms", n_chunks, ms)
            return data
        except Exception as e:
            ms = int((time.monotonic() - t0) * 1000)
            logger.warning("[FeatureMapAgent] failed in %dms: %s", ms, e)
            return self._fallback(str(e))
