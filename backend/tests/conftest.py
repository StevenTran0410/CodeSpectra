from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from domain.analysis.agents._context_builders import PipelineMemoryContext
from domain.model_connector.service import ProviderConfigService
from domain.model_connector.types import ChatResponse
from domain.retrieval.service import RetrievalService
from domain.retrieval.types import (
    RetrievalBundle,
    RetrievalEvidence,
    RetrievalMode,
    RetrievalSection,
)

_ASPECT = {"description": "", "evidence_files": []}
_FILE = {"file": "x.py", "reason": "r"}

MINIMAL_SECTION_JSON: dict[str, object] = {
    "A": {
        "repo_name": "r",
        "domain": "d",
        "purpose": "p",
        "runtime_type": "python",
        "tech_stack": [],
        "business_context": "",
        "confidence": "high",
        "evidence_files": [],
        "blind_spots": [],
        "content": "",
    },
    "B": {
        "main_layers": [],
        "frameworks": [],
        "entrypoints": [],
        "main_services": [],
        "external_integrations": [],
        "config_sources": [],
        "database_hints": [],
        "confidence": "high",
        "evidence_files": [],
        "blind_spots": [],
        "content": "",
    },
    "C": {
        "folders": [],
        "summary": "",
        "confidence": "high",
        "evidence_files": [],
        "blind_spots": [],
        "content": "",
    },
    "D": {
        "naming_style": dict(_ASPECT),
        "error_handling": dict(_ASPECT),
        "async_style": dict(_ASPECT),
        "di_style": dict(_ASPECT),
        "class_vs_functional": dict(_ASPECT),
        "test_style": dict(_ASPECT),
        "signals": [],
        "confidence": "high",
        "evidence_files": [],
        "blind_spots": [],
        "content": "",
    },
    "E": {
        "rules": [],
        "violations_found": [],
        "confidence": "high",
        "evidence_files": [],
        "blind_spots": [],
        "content": "",
    },
    "F": {
        "features": [],
        "confidence": "high",
        "evidence_files": [],
        "blind_spots": [],
        "content": "",
    },
    "G": {
        "entrypoint": dict(_FILE),
        "backbone": dict(_FILE),
        "critical_config": dict(_FILE),
        "highest_centrality": dict(_FILE),
        "most_dangerous_to_touch": dict(_FILE),
        "read_first": dict(_FILE),
        "other_important": [],
        "confidence": "high",
        "evidence_files": [],
        "blind_spots": [],
        "content": "",
    },
    "H": {
        "steps": [],
        "total_estimated_minutes": 0,
        "confidence": "high",
        "evidence_files": [],
        "blind_spots": [],
        "content": "",
    },
    "I": {
        "terms": [],
        "confidence": "high",
        "blind_spots": [],
        "content": "",
    },
    "J": {
        "findings": [],
        "summary": "",
        "confidence": "high",
        "evidence_files": [],
        "blind_spots": [],
        "content": "",
    },
    "K": {
        "overall_confidence": "high",
        "section_scores": {},
        "weakest_sections": [],
        "coverage_percentage": 0.0,
        "notes": "",
        "blind_spots": [],
        "content": "",
    },
}


def _evidence(i: int) -> RetrievalEvidence:
    return RetrievalEvidence(
        chunk_id=f"c{i}",
        rel_path=f"src/f{i}.py",
        chunk_index=0,
        reason_codes=["test"],
        score=0.5,
        token_estimate=10,
        excerpt=f"ex{i}",
    )


@pytest.fixture
def canned_retrieval_bundle() -> RetrievalBundle:
    return RetrievalBundle(
        snapshot_id="snap-test",
        mode=RetrievalMode.HYBRID,
        section=RetrievalSection.ARCHITECTURE,
        query="q",
        budget_tokens=100,
        used_tokens=10,
        evidences=[_evidence(0), _evidence(1), _evidence(2)],
    )


@pytest.fixture
def mock_retrieval(canned_retrieval_bundle: RetrievalBundle) -> MagicMock:
    svc = MagicMock(spec=RetrievalService)
    svc.retrieve = AsyncMock(return_value=canned_retrieval_bundle)
    return svc


@pytest.fixture
def mock_provider_chat_json(request: pytest.FixtureRequest) -> MagicMock:
    letter = getattr(request, "param", "A")
    payload = MINIMAL_SECTION_JSON[letter]
    content = json.dumps(payload)

    svc = MagicMock(spec=ProviderConfigService)
    svc.chat = AsyncMock(
        return_value=ChatResponse(provider_id="test-prov", model_id="test-model", content=content)
    )
    return svc


@pytest.fixture
def sample_pipeline_memory_context(
    canned_retrieval_bundle: RetrievalBundle,
) -> PipelineMemoryContext:
    return PipelineMemoryContext(
        arch_bundle=canned_retrieval_bundle,
        folder_tree="a.py\nb.py",
        doc_files="README\ndoc",
        manifest_files="pyproject.toml",
    )


def chat_response_sequence(contents: list[str]) -> MagicMock:
    svc = MagicMock(spec=ProviderConfigService)
    queue = list(contents)

    async def _chat(_req):
        if not queue:
            return ChatResponse(provider_id="p", model_id="m", content="{}")
        c = queue.pop(0)
        return ChatResponse(provider_id="p", model_id="m", content=c)

    svc.chat = AsyncMock(side_effect=_chat)
    return svc
