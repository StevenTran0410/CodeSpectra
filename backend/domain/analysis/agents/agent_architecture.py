"""Architecture overview agent (section B)."""

from __future__ import annotations

import time
from typing import Any

from domain.model_connector.service import ProviderConfigService
from domain.retrieval.service import RetrievalService
from domain.retrieval.types import RetrievalBundle, RetrievalMode, RetrievalSection, RetrieveRequest
from domain.structural_graph.types import StructuralGraphSummary
from shared.logger import logger

from ..agent_pipeline import _normalize_conf
from ..prompts import AGENT_B_SCHEMA_STR, AGENT_B_SYSTEM, render_bundle
from ..schemas import validate_section
from ._context_builders import extract_a_identity_context
from .base import BaseTypedAgent


class ArchitectureAgent(BaseTypedAgent):
    def __init__(
        self,
        provider_service: ProviderConfigService,
        retrieval_service: RetrievalService,
    ) -> None:
        super().__init__(provider_service)
        self._retrieval = retrieval_service

    def _fallback(self, reason: str) -> dict[str, Any]:
        return {
            "main_layers": [],
            "frameworks": [],
            "entrypoints": [],
            "main_services": [],
            "external_integrations": [],
            "config_sources": [],
            "database_hints": [],
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
        arch_bundle: RetrievalBundle | None = None,
        identity_output: dict | None = None,
    ) -> dict[str, Any]:
        t0 = time.monotonic()
        n_chunks = 0
        try:
            graph_block = ""
            if graph_summary and graph_summary.top_central_files:
                lines = ["Graph centrality (top files by import score):"]
                for n in graph_summary.top_central_files[:10]:
                    lines.append(f"  {n.rel_path} (score={n.score}, indegree={n.indegree})")
                graph_block = "\n".join(lines)
            if arch_bundle is not None:
                bundle = arch_bundle
                logger.info(
                    "[ArchitectureAgent] using arch_bundle from mem_ctx: %d chunks",
                    len(arch_bundle.evidences),
                )
            else:
                logger.info("[ArchitectureAgent] arch_bundle not in mem_ctx, using own retrieval")
                bundle = await self._retrieval.retrieve(
                    RetrieveRequest(
                        snapshot_id=snapshot_id,
                        query=(
                            "framework entrypoint bootstrap layer service router handler middleware"
                        ),
                        section=RetrievalSection.ARCHITECTURE,
                        mode=RetrievalMode.HYBRID,
                        max_results=30,
                    )
                )
            n_chunks = len(bundle.evidences)
            identity_block = extract_a_identity_context(identity_output)
            prefix_parts = []
            if identity_block:
                prefix_parts.append(identity_block)
            if graph_block:
                prefix_parts.append(graph_block)
            prefix = "\n\n".join(prefix_parts) + ("\n\n" if prefix_parts else "")
            user_prompt = f"{prefix}snapshot_id={snapshot_id}\n\nEvidence:\n{render_bundle(bundle)}"
            data = await self._chat_json_typed(
                provider_id,
                model_id,
                AGENT_B_SYSTEM,
                user_prompt,
                AGENT_B_SCHEMA_STR,
                max_completion_tokens=2500,
            )
            for key in (
                "main_layers",
                "frameworks",
                "entrypoints",
                "external_integrations",
                "config_sources",
                "database_hints",
                "evidence_files",
                "blind_spots",
            ):
                raw = data.get(key)
                if isinstance(raw, list):
                    data[key] = [str(x) for x in raw if x is not None]
                else:
                    data[key] = []
            raw_ms = data.get("main_services")
            if not isinstance(raw_ms, list):
                data["main_services"] = []
            else:
                data["main_services"] = [
                    {
                        "name": str(i.get("name", "") or ""),
                        "path": str(i.get("path", "") or ""),
                        "role": str(i.get("role", "") or ""),
                    }
                    for i in raw_ms
                    if isinstance(i, dict)
                ]
            data["confidence"] = _normalize_conf(str(data.get("confidence", "medium")))
            validate_section("B", data)
            ms = int((time.monotonic() - t0) * 1000)
            logger.info("[ArchitectureAgent] %d chunks retrieved, completed in %dms", n_chunks, ms)
            return data
        except Exception as e:
            ms = int((time.monotonic() - t0) * 1000)
            logger.warning("[ArchitectureAgent] failed in %dms: %s", ms, e)
            return self._fallback(str(e))
