"""Project identity agent (section A)."""
from __future__ import annotations

import time
from typing import Any

from domain.model_connector.service import ProviderConfigService
from domain.retrieval.service import RetrievalService
from domain.retrieval.types import RetrievalMode, RetrievalSection, RetrieveRequest
from infrastructure.db.database import get_db  # still used by _fetch_files_by_pattern
from shared.logger import logger

from ..agent_pipeline import _normalize_conf
from ..prompts import AGENT_A_SCHEMA_STR, AGENT_A_SYSTEM, render_bundle
from ..schemas import validate_section
from ._context_builders import fetch_folder_tree
from .base import BaseTypedAgent

# Manifest files: build/dependency descriptors — fetch up to 3000 chars each
_MANIFEST_PATTERNS = (
    "pyproject.toml", "package.json", "Cargo.toml", "go.mod",
    "requirements.txt", "setup.py", "setup.cfg", "pom.xml",
    "build.gradle", "composer.json", "Gemfile", "Package.swift",
)

# Doc files: human-readable docs — fetch in FULL (these are the important context files)
_DOC_PATTERNS = (
    "README.md", "README.rst", "README.txt", "README",
    "CHANGELOG.md", "CONTRIBUTING.md", "ARCHITECTURE.md",
)


async def _fetch_files_by_pattern(
    snapshot_id: str,
    patterns: tuple[str, ...],
    char_limit: int,
    max_rows: int = 6,
) -> str:
    """Fetch file content from retrieval_chunks matching any of the given filename patterns."""
    db = get_db()
    conditions = " OR ".join("rc.rel_path LIKE ?" for _ in patterns)
    params: list[Any] = [snapshot_id] + [f"%{p}" for p in patterns]
    rows = []
    try:
        async with db.execute(
            f"""SELECT rc.rel_path, rc.content
                FROM retrieval_chunks rc
                WHERE rc.snapshot_id=? AND ({conditions})
                ORDER BY LENGTH(rc.rel_path) ASC
                LIMIT {max_rows}""",
            params,
        ) as cur:
            rows = await cur.fetchall()
    except Exception:
        pass
    if not rows:
        return ""
    # De-duplicate by rel_path (multiple chunks per file — merge up to char_limit)
    merged: dict[str, list[str]] = {}
    for row in rows:
        path = row["rel_path"]
        merged.setdefault(path, []).append(row["content"] or "")
    parts = []
    for path, chunks in merged.items():
        full = "\n".join(chunks)
        if char_limit and len(full) > char_limit:
            full = full[:char_limit] + "\n...(truncated)"
        parts.append(f"=== {path} ===\n{full}")
    return "\n\n".join(parts)


async def _gather_context(
    retrieval: RetrievalService, snapshot_id: str
) -> tuple[Any, str, str, str]:
    """Run retrieval + doc fetch + manifest fetch + folder tree in parallel."""
    import asyncio
    bundle_task = retrieval.retrieve(
        RetrieveRequest(
            snapshot_id=snapshot_id,
            query="README entrypoint package manifest setup pyproject cargo",
            section=RetrievalSection.IMPORTANT_FILES,
            mode=RetrievalMode.HYBRID,
            max_results=20,
        )
    )
    # Doc files: no char limit (fetch full content — README is the most important)
    doc_task = _fetch_files_by_pattern(snapshot_id, _DOC_PATTERNS, char_limit=0, max_rows=4)
    # Manifest files: 3000 chars each
    manifest_task = _fetch_files_by_pattern(snapshot_id, _MANIFEST_PATTERNS, char_limit=3000)
    tree_task = fetch_folder_tree(snapshot_id)
    return await asyncio.gather(bundle_task, doc_task, manifest_task, tree_task)


class AgentA(BaseTypedAgent):
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
        self, provider_id: str, model_id: str, snapshot_id: str, repo_name: str = ""
    ) -> dict[str, Any]:
        t0 = time.monotonic()
        n_chunks = 0
        try:
            bundle, doc_ctx, manifest_ctx, folder_tree = await _gather_context(
                self._retrieval, snapshot_id
            )
            n_chunks = len(bundle.evidences)
            hint = f"repo_name={repo_name}\n" if repo_name else ""
            user_prompt_parts = [f"{hint}snapshot_id={snapshot_id}"]
            if folder_tree:
                n_files = folder_tree.count("\n") + 1
                user_prompt_parts.append(f"\n--- Repo file listing ({n_files} files) ---\n{folder_tree}")
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
                max_completion_tokens=16000,
            )
            for key in ("repo_name", "domain", "purpose", "runtime_type", "business_context"):
                data[key] = str(data.get(key, "") or "")
            # If LLM failed to identify repo_name, use the known name from DB
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
            logger.info("[AgentA] %d chunks retrieved, completed in %dms", n_chunks, ms)
            return data
        except Exception as e:
            ms = int((time.monotonic() - t0) * 1000)
            logger.warning("[AgentA] failed in %dms: %s", ms, e)
            return self._fallback(snapshot_id, str(e), repo_name)
