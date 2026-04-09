"""Project identity agent (section A)."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from domain.model_connector.service import ProviderConfigService
from domain.retrieval.service import RetrievalService
from domain.retrieval.types import RetrievalMode, RetrievalSection, RetrieveRequest
from shared.logger import logger

from ..agent_pipeline import _normalize_conf
from ..profiles import NORMAL_PROFILE, AnalysisProfile
from ..prompts import AGENT_A_SCHEMA_STR, AGENT_A_SYSTEM, render_bundle
from ..schemas import validate_section
from ._context_builders import (
    PipelineMemoryContext,
    _DOC_PATTERNS,
    _MANIFEST_PATTERNS,
    _fetch_files_by_pattern,
    fetch_folder_tree,
)
from .base import BaseTypedAgent


async def _gather_context(
    retrieval: RetrievalService,
    snapshot_id: str,
    profile: AnalysisProfile,
) -> tuple[Any, str, str, str]:
    """Run retrieval + doc fetch + manifest fetch + folder tree in parallel."""
    bundle_task = retrieval.retrieve(
        RetrieveRequest(
            snapshot_id=snapshot_id,
            query="README entrypoint package manifest setup pyproject cargo",
            section=RetrievalSection.IMPORTANT_FILES,
            mode=RetrievalMode.HYBRID,
            max_results=profile.retrieval_max_results,
        )
    )
    doc_task = _fetch_files_by_pattern(
        snapshot_id, _DOC_PATTERNS, char_limit=profile.retrieval_doc_char_limit, max_rows=4
    )
    manifest_task = _fetch_files_by_pattern(
        snapshot_id, _MANIFEST_PATTERNS, char_limit=profile.retrieval_manifest_char_limit
    )
    tree_task = fetch_folder_tree(snapshot_id)
    return await asyncio.gather(bundle_task, doc_task, manifest_task, tree_task)


class ProjectIdentityAgent(BaseTypedAgent):
    def __init__(
        self,
        provider_service: ProviderConfigService,
        retrieval_service: RetrievalService,
    ) -> None:
        super().__init__(provider_service)
        self._retrieval = retrieval_service

    def _fallback(self, snapshot_id: str, reason: str, repo_name: str = "") -> dict[str, Any]:
        return {
            "repo_name": repo_name or snapshot_id,
            "domain": "unknown",
            "purpose": "",
            "runtime_type": "unknown",
            "tech_stack": [],
            "business_context": "",
            "confidence": "low",
            "evidence_files": [],
            "blind_spots": [f"Agent failed: {reason}"],
        }

    async def run(
        self,
        provider_id: str,
        model_id: str,
        snapshot_id: str,
        repo_name: str = "",
        mem_ctx: PipelineMemoryContext | None = None,
        profile: AnalysisProfile | None = None,
    ) -> dict[str, Any]:
        t0 = time.monotonic()
        n_chunks = 0
        _profile = profile or NORMAL_PROFILE
        try:
            if mem_ctx is not None:
                bundle = await self._retrieval.retrieve(
                    RetrieveRequest(
                        snapshot_id=snapshot_id,
                        query="README entrypoint package manifest setup pyproject cargo",
                        section=RetrievalSection.IMPORTANT_FILES,
                        mode=RetrievalMode.HYBRID,
                        max_results=_profile.retrieval_max_results,
                    )
                )
                doc_ctx = mem_ctx.doc_files
                manifest_ctx = mem_ctx.manifest_files
                folder_tree = mem_ctx.folder_tree
                logger.info(
                    "[ProjectIdentityAgent] using mem_ctx: doc_files=%d chars, "
                    "manifest_files=%d chars, folder_tree=%d chars",
                    len(mem_ctx.doc_files),
                    len(mem_ctx.manifest_files),
                    len(mem_ctx.folder_tree),
                )
            else:
                bundle, doc_ctx, manifest_ctx, folder_tree = await _gather_context(
                    self._retrieval, snapshot_id, _profile
                )
            n_chunks = len(bundle.evidences)
            hint = f"repo_name={repo_name}\n" if repo_name else ""
            user_prompt_parts = [f"{hint}snapshot_id={snapshot_id}"]
            if folder_tree:
                n_files = folder_tree.count("\n") + 1
                user_prompt_parts.append(
                    f"\n--- Repo file listing ({n_files} files) ---\n{folder_tree}"
                )
            if doc_ctx:
                user_prompt_parts.append(f"\n--- Documentation files (full content) ---\n{doc_ctx}")
            if manifest_ctx:
                user_prompt_parts.append(f"\n--- Manifest / dependency files ---\n{manifest_ctx}")
            user_prompt_parts.append(f"\n--- Retrieval evidence ---\n{render_bundle(bundle)}")
            user_prompt = "\n".join(user_prompt_parts)
            data = await self._chat_json_typed(
                provider_id,
                model_id,
                AGENT_A_SYSTEM,
                user_prompt,
                AGENT_A_SCHEMA_STR,
                max_completion_tokens=_profile.tokens_project_identity,
            )
            for key in ("repo_name", "domain", "purpose", "runtime_type", "business_context"):
                data[key] = str(data.get(key, "") or "")
            if not data["repo_name"] or data["repo_name"].lower() in ("unknown", ""):
                data["repo_name"] = repo_name
            for key in ("tech_stack", "evidence_files", "blind_spots"):
                raw = data.get(key)
                if not isinstance(raw, list):
                    data[key] = []
                else:
                    data[key] = [str(x) for x in raw if x is not None]
            data["confidence"] = _normalize_conf(str(data.get("confidence", "medium")))
            validate_section("A", data)
            ms = int((time.monotonic() - t0) * 1000)
            logger.info(
                "[ProjectIdentityAgent] %d chunks retrieved, completed in %dms", n_chunks, ms
            )
            return data
        except Exception as e:
            ms = int((time.monotonic() - t0) * 1000)
            logger.warning("[ProjectIdentityAgent] failed in %dms: %s", ms, e)
            return self._fallback(snapshot_id, str(e), repo_name)
