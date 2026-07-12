"""Factory functions that build mocked agent collaborators from synthetic case data — generalizes backend/tests/conftest.py's pytest fixtures into plain callables usable outside pytest."""
from __future__ import annotations

import contextlib
import dataclasses
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from agent_eval_harness.injection.backend_bridge import ensure_backend_importable

ensure_backend_importable()

from domain.analysis.agents._context_builders import PipelineMemoryContext  # noqa: E402
from domain.analysis.profiles import NORMAL_PROFILE  # noqa: E402
from domain.model_connector.service import ProviderConfigService  # noqa: E402
from domain.model_connector.types import ChatResponse  # noqa: E402
from domain.retrieval.service import RetrievalService  # noqa: E402
from domain.retrieval.types import (  # noqa: E402
    RetrievalBundle,
    RetrievalEvidence,
    RetrievalMode,
    RetrievalSection,
)


def build_retrieval_bundle(data: dict[str, Any]) -> RetrievalBundle:
    evidences = [
        RetrievalEvidence(
            chunk_id=e.get("chunk_id", f"synthetic-{i}"),
            rel_path=e["rel_path"],
            chunk_index=e.get("chunk_index", i),
            reason_codes=e.get("reason_codes", ["synthetic"]),
            score=e.get("score", 1.0),
            token_estimate=e.get("token_estimate", len(str(e.get("excerpt", "")))//4 or 1),
            excerpt=e.get("excerpt", ""),
        )
        for i, e in enumerate(data.get("evidences", []))
    ]
    return RetrievalBundle(
        snapshot_id=data.get("snapshot_id", "synthetic"),
        mode=RetrievalMode(data["mode"]) if data.get("mode") else RetrievalMode.HYBRID,
        section=RetrievalSection(data["section"]) if data.get("section") else RetrievalSection.IMPORTANT_FILES,
        query=data.get("query", ""),
        budget_tokens=data.get("budget_tokens", 4000),
        used_tokens=data.get("used_tokens", 0),
        evidences=evidences,
    )


def make_mock_retrieval(bundles: RetrievalBundle | list[RetrievalBundle]) -> MagicMock:
    """A list is served one-per-call so multi-sub-query agents (D/F/J) get distinct evidence per query instead of a deduped repeat."""
    svc = MagicMock(spec=RetrievalService)
    if isinstance(bundles, list):
        svc.retrieve = AsyncMock(side_effect=list(bundles))
    else:
        svc.retrieve = AsyncMock(return_value=bundles)
    return svc


def make_chat_sequence(contents: list[str]) -> MagicMock:
    """Serves canned chat responses in order; extra calls beyond the list return '{}'."""
    svc = MagicMock(spec=ProviderConfigService)
    queue = list(contents)

    async def _chat(_req):
        c = queue.pop(0) if queue else "{}"
        return ChatResponse(provider_id="synthetic", model_id="synthetic", content=c)

    svc.chat = AsyncMock(side_effect=_chat)
    return svc


def make_chat_single(content: dict[str, Any] | str) -> MagicMock:
    body = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    svc = MagicMock(spec=ProviderConfigService)
    svc.chat = AsyncMock(
        return_value=ChatResponse(provider_id="synthetic", model_id="synthetic", content=body)
    )
    return svc


def make_pipeline_memory_context(
    *, arch_bundle: RetrievalBundle | None = None,
    folder_tree: str = "", doc_files: str = "", manifest_files: str = "",
) -> PipelineMemoryContext:
    return PipelineMemoryContext(
        arch_bundle=arch_bundle,  # type: ignore[arg-type]  # None is a safe, handled value
        folder_tree=folder_tree,
        doc_files=doc_files,
        manifest_files=manifest_files,
    )


def neutral_profile():
    """NORMAL_PROFILE with the retrieval-augment loop disabled so a fixed injected context is never re-queried against a (nonexistent) DB mid-run."""
    return dataclasses.replace(NORMAL_PROFILE, retrieval_augment_rounds=0)


@contextlib.asynccontextmanager
async def patched_folder_summary(folder_tree: str):
    """Patches build_folder_summary in every module that imported it locally, since `from X import build_folder_summary` binds a module-local name that patching only _context_builders would miss."""
    import domain.analysis.agents._context_builders as _ctx_builders
    import domain.analysis.agents.agent_project_identity as _agent_a_module
    import domain.analysis.agents.agent_structure as _agent_c_module

    async def _stub(snapshot_id: str, max_depth: int = 3) -> str:
        return folder_tree

    patched_modules = (_ctx_builders, _agent_a_module, _agent_c_module)
    originals = {mod: mod.build_folder_summary for mod in patched_modules}
    for mod in patched_modules:
        mod.build_folder_summary = _stub
    try:
        yield
    finally:
        for mod, original in originals.items():
            mod.build_folder_summary = original
