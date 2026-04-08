"""Repo structure narrative agent (section C)."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from domain.model_connector.service import ProviderConfigService
from domain.retrieval.service import RetrievalService
from domain.retrieval.types import RetrievalBundle, RetrievalMode, RetrievalSection, RetrieveRequest
from shared.logger import logger

from ..agent_pipeline import _normalize_conf
from ..prompts import AGENT_C_SCHEMA_STR, AGENT_C_SYSTEM, render_bundle
from ..schemas import validate_section
from ._context_builders import extract_a_identity_context, fetch_folder_tree
from .base import BaseTypedAgent

_ALLOWED_FOLDER_ROLES = frozenset(
    {
        "domain",
        "infrastructure",
        "delivery",
        "shared",
        "test",
        "generated",
        "unknown",
    }
)


class StructureAgent(BaseTypedAgent):
    def __init__(
        self,
        provider_service: ProviderConfigService,
        retrieval_service: RetrievalService,
    ) -> None:
        super().__init__(provider_service)
        self._retrieval = retrieval_service

    def _fallback(self, reason: str) -> dict[str, Any]:
        return {
            "folders": [],
            "summary": "",
            "confidence": "low",
            "evidence_files": [],
            "blind_spots": [f"Agent failed: {reason}"],
        }

    async def run(
        self,
        provider_id: str,
        model_id: str,
        snapshot_id: str,
        arch_bundle: RetrievalBundle | None = None,
        folder_tree: str = "",
        identity_output: dict | None = None,
    ) -> dict[str, Any]:
        t0 = time.monotonic()
        n_chunks = 0
        try:
            if arch_bundle is not None:
                bundle = arch_bundle
                logger.info(
                    "[StructureAgent] using arch_bundle from mem_ctx: %d chunks",
                    len(arch_bundle.evidences),
                )
                if not folder_tree:
                    folder_tree = await fetch_folder_tree(snapshot_id)
            else:
                logger.info("[StructureAgent] arch_bundle not in mem_ctx, using own retrieval")
                bundle, folder_tree = await asyncio.gather(
                    self._retrieval.retrieve(
                        RetrieveRequest(
                            snapshot_id=snapshot_id,
                            query=("folder module package structure layer boundary domain"),
                            section=RetrievalSection.ARCHITECTURE,
                            mode=RetrievalMode.HYBRID,
                            max_results=30,
                        )
                    ),
                    fetch_folder_tree(snapshot_id),
                )
            n_chunks = len(bundle.evidences)
            identity_block = extract_a_identity_context(identity_output)
            user_prompt_parts = [f"snapshot_id={snapshot_id}"]
            if identity_block:
                user_prompt_parts.insert(0, identity_block)
            if folder_tree:
                n_files = folder_tree.count("\n") + 1
                user_prompt_parts.append(
                    f"\n--- Repo file listing ({n_files} files) ---\n{folder_tree}"
                )
            user_prompt_parts.append(f"\n--- Retrieval evidence ---\n{render_bundle(bundle)}")
            user_prompt = "\n".join(user_prompt_parts)
            data = await self._chat_json_typed(
                provider_id,
                model_id,
                AGENT_C_SYSTEM,
                user_prompt,
                AGENT_C_SCHEMA_STR,
                max_completion_tokens=2000,
            )
            raw_folders = data.get("folders")
            folders: list[dict[str, str]] = []
            if isinstance(raw_folders, list):
                for item in raw_folders:
                    if not isinstance(item, dict) or "path" not in item:
                        continue
                    role = str(item.get("role", "") or "unknown").strip().lower()
                    if role not in _ALLOWED_FOLDER_ROLES:
                        role = "unknown"
                    folders.append(
                        {
                            "path": str(item.get("path", "") or ""),
                            "role": role,
                            "description": str(item.get("description", "") or ""),
                        }
                    )
            data["folders"] = folders
            data["summary"] = str(data.get("summary", "") or "")
            for key in ("evidence_files", "blind_spots"):
                raw = data.get(key)
                if isinstance(raw, list):
                    data[key] = [str(x) for x in raw if x is not None]
                else:
                    data[key] = []
            data["confidence"] = _normalize_conf(str(data.get("confidence", "medium")))
            validate_section("C", data)
            ms = int((time.monotonic() - t0) * 1000)
            logger.info("[StructureAgent] %d chunks retrieved, completed in %dms", n_chunks, ms)
            return data
        except Exception as e:
            ms = int((time.monotonic() - t0) * 1000)
            logger.warning("[StructureAgent] failed in %dms: %s", ms, e)
            return self._fallback(str(e))
