"""Important files radar agent (section G)."""

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
from ..prompts import AGENT_G_SCHEMA_STR, AGENT_G_SYSTEM, render_bundle
from ..schemas import validate_section
from .base import BaseTypedAgent


def _slot_from_bundle(bundle_ev_paths: list[str], i: int, reason: str) -> dict[str, str]:
    if bundle_ev_paths and i < len(bundle_ev_paths):
        return {"file": bundle_ev_paths[i], "reason": reason}
    return {"file": "unknown", "reason": reason}


class ImportantFilesAgent(BaseTypedAgent):
    def __init__(
        self,
        provider_service: ProviderConfigService,
        retrieval_service: RetrievalService,
    ) -> None:
        super().__init__(provider_service)
        self._retrieval = retrieval_service

    def _fallback(self, reason: str, paths: list[str]) -> dict[str, Any]:
        r = f"fallback — could not be determined ({reason})"
        return {
            "entrypoint": _slot_from_bundle(paths, 0, r),
            "backbone": _slot_from_bundle(paths, 1, r),
            "critical_config": _slot_from_bundle(paths, 2, r),
            "highest_centrality": _slot_from_bundle(paths, 3, r),
            "most_dangerous_to_touch": _slot_from_bundle(paths, 4, r),
            "read_first": _slot_from_bundle(paths, 5, r),
            "other_important": [],
            "confidence": "low",
            "evidence_files": paths[:8],
            "blind_spots": [f"Agent failed: {reason}"],
        }

    async def run(
        self,
        provider_id: str,
        model_id: str,
        snapshot_id: str,
        graph_summary: StructuralGraphSummary | None = None,
        profile: AnalysisProfile | None = None,
    ) -> dict[str, Any]:
        t0 = time.monotonic()
        n_chunks = 0
        paths: list[str] = []
        _profile = profile or NORMAL_PROFILE
        self._session_chunk_ids = []
        try:
            graph_lines: list[str] = []
            if graph_summary and graph_summary.top_central_files:
                graph_lines.append("Graph centrality (top files by import score):")
                for n in graph_summary.top_central_files[:10]:
                    graph_lines.append(f"  {n.rel_path} (score={n.score}, indegree={n.indegree})")
            graph_block = "\n".join(graph_lines)
            bundle = await self._retrieval.retrieve(
                RetrieveRequest(
                    snapshot_id=snapshot_id,
                    query="entrypoint main bootstrap central high-import",
                    section=RetrievalSection.IMPORTANT_FILES,
                    mode=RetrievalMode.HYBRID,
                    max_results=_profile.retrieval_max_results,
                )
            )
            self._record_bundle(bundle)
            n_chunks = len(bundle.evidences)
            paths = [e.rel_path for e in bundle.evidences]
            prefix = f"{graph_block}\n\n" if graph_block else ""
            user_prompt = f"{prefix}snapshot_id={snapshot_id}\n\nEvidence:\n{render_bundle(bundle)}"
            data = await self._chat_json_with_augment(
                retrieval=self._retrieval,
                snapshot_id=snapshot_id,
                provider_id=provider_id,
                model_id=model_id,
                system_prompt=AGENT_G_SYSTEM,
                user_prompt=user_prompt,
                schema_hint=AGENT_G_SCHEMA_STR,
                max_completion_tokens=_profile.tokens_important_files,
                retrieval_section=RetrievalSection.IMPORTANT_FILES,
                profile=_profile,
                agent_tag="ImportantFilesAgent",
            )
            slot_keys = (
                "entrypoint",
                "backbone",
                "critical_config",
                "highest_centrality",
                "most_dangerous_to_touch",
                "read_first",
            )
            for i, key in enumerate(slot_keys):
                slot = data.get(key)
                if not isinstance(slot, dict) or "file" not in slot:
                    data[key] = _slot_from_bundle(paths, i, "filled from retrieval fallback")
                else:
                    data[key] = {
                        "file": str(slot.get("file", "") or "unknown"),
                        "reason": str(slot.get("reason", "") or ""),
                    }
            oi = data.get("other_important")
            if not isinstance(oi, list):
                data["other_important"] = []
            else:
                norm: list[dict[str, str]] = []
                for item in oi:
                    if isinstance(item, dict) and "file" in item:
                        norm.append(
                            {
                                "file": str(item.get("file", "")),
                                "reason": str(item.get("reason", "")),
                            }
                        )
                data["other_important"] = norm
            for key in ("evidence_files", "blind_spots"):
                raw = data.get(key)
                if not isinstance(raw, list):
                    data[key] = []
                else:
                    data[key] = [str(x) for x in raw if x is not None]
            data["confidence"] = _normalize_conf(str(data.get("confidence", "medium")))
            validate_section("G", data)
            data["retrieved_chunk_ids"] = self._pop_chunk_ids()
            ms = int((time.monotonic() - t0) * 1000)
            logger.info(
                "[ImportantFilesAgent] %d chunks retrieved, completed in %dms", n_chunks, ms
            )
            return data
        except Exception as e:
            ms = int((time.monotonic() - t0) * 1000)
            logger.warning("[ImportantFilesAgent] failed in %dms: %s", ms, e)
            return self._fallback(str(e), paths)
