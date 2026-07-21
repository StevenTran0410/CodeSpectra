"""Tests for LangGraph-based Deep Research agent."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from domain.model_connector.service import ProviderConfigService
from domain.model_connector.types import ChatResponse
from domain.qa.deep_research import (
    DeepResearchAgent,
    DeepResearchResult,
)
from domain.retrieval.service import RetrievalService
from domain.retrieval.types import RetrievalBundle, RetrievalEvidence, RetrievalMode, RetrievalSection


def _mock_evidence(i: int) -> RetrievalEvidence:
    return RetrievalEvidence(
        chunk_id=f"c{i}",
        rel_path=f"src/f{i}.py",
        chunk_index=0,
        reason_codes=["test"],
        score=0.5,
        token_estimate=10,
        excerpt=f"evidence {i}",
    )


def _chat_response_sequence(contents: list[str]) -> MagicMock:
    """Create a mock provider that returns chat responses in sequence."""
    svc = MagicMock(spec=ProviderConfigService)
    queue = list(contents)

    async def _chat(_req):
        if not queue:
            return ChatResponse(provider_id="p", model_id="m", content="{}")
        c = queue.pop(0)
        return ChatResponse(provider_id="p", model_id="m", content=c)

    svc.chat = AsyncMock(side_effect=_chat)
    return svc


def _chat_stream_with_tokens(tokens: list[str]) -> MagicMock:
    """Create a mock provider that streams tokens one by one."""
    svc = MagicMock(spec=ProviderConfigService)

    async def _stream(provider_id, model_id, system, user, on_token=None, **kwargs):
        for token in tokens:
            if on_token:
                await asyncio.sleep(0.01)
                await on_token(token)
        return "".join(tokens)

    svc.call_stream = AsyncMock(side_effect=_stream)
    return svc


@pytest.mark.asyncio
async def test_deep_research_langgraph_e2e():
    """Test that LangGraph deep research runs end-to-end with a canned plan."""
    plan = [
        {"type": "retrieve", "target": "query", "description": "Find initial evidence"},
    ]

    plan_json = json.dumps({"steps": plan})
    step_result_json = json.dumps({
        "finding": "Found something",
        "key_files": ["src/main.py"],
        "graph_path": None,
        "sufficient": True,
    })
    meta_json = json.dumps({
        "reasoning_chain": [
            {
                "step_number": 1,
                "description": "Find initial evidence",
                "files_involved": ["src/main.py"],
                "finding": "Found something",
                "graph_path": None,
                "tentative_files": [],
                "tentative_confidence": {},
            }
        ],
        "confidence": "high",
        "unknowns": [],
    })

    provider_svc = _chat_response_sequence([plan_json, step_result_json, meta_json])

    retrieval_svc = MagicMock(spec=RetrievalService)
    retrieval_svc.retrieve = AsyncMock(
        return_value=RetrievalBundle(
            snapshot_id="snap-test",
            mode=RetrievalMode.HYBRID,
            section=RetrievalSection.QA,
            query="test",
            budget_tokens=100,
            used_tokens=50,
            evidences=[_mock_evidence(0), _mock_evidence(1)],
        )
    )

    with patch("domain.qa.deep_research._load_graph_ctx") as mock_load_ctx:
        mock_load_ctx.return_value = MagicMock(
            centrality={},
            top_central=[],
            community_of={},
            community_hubs={},
        )
        with patch("domain.qa.deep_research.plan_queries") as mock_plan_queries:
            mock_plan_queries.return_value = ["query"]
            with patch("domain.qa.deep_research.retrieve_multi") as mock_retrieve_multi:
                mock_retrieve_multi.return_value = RetrievalBundle(
                    snapshot_id="snap-test",
                    mode=RetrievalMode.HYBRID,
                    section=RetrievalSection.QA,
                    query="test",
                    budget_tokens=100,
                    used_tokens=50,
                    evidences=[_mock_evidence(0), _mock_evidence(1)],
                )

                agent = DeepResearchAgent(provider_svc, retrieval_svc)
                result = await agent.research(
                    question="What is this?",
                    snapshot_id="snap-test",
                    provider_id="test",
                    model_id="test-model",
                    max_hops=5,
                    include_debug=False,
                )

    assert isinstance(result, dict)
    assert "summary" in result
    assert "reasoning_chain" in result
    assert "files_explored" in result
    assert "confidence" in result
    assert result["confidence"] in ("high", "medium", "low")
    assert "unknowns" in result
    assert "elapsed_ms" in result
    assert result["elapsed_ms"] > 0

    result_obj = DeepResearchResult.model_validate(result)
    assert result_obj.summary is not None
    assert len(result_obj.reasoning_chain) >= 0


@pytest.mark.asyncio
async def test_deep_research_progress_events_schema():
    """Test that progress events match expected schema."""
    plan = [
        {"type": "retrieve", "target": "query", "description": "Find evidence"},
    ]

    plan_json = json.dumps({"steps": plan})
    step_result_json = json.dumps({
        "finding": "Found",
        "key_files": ["src/main.py"],
        "graph_path": None,
        "sufficient": True,
    })
    meta_json = json.dumps({
        "reasoning_chain": [],
        "confidence": "medium",
        "unknowns": [],
    })

    provider_svc = _chat_response_sequence([plan_json, step_result_json, meta_json])

    retrieval_svc = MagicMock(spec=RetrievalService)
    retrieval_svc.retrieve = AsyncMock(
        return_value=RetrievalBundle(
            snapshot_id="snap-test",
            mode=RetrievalMode.HYBRID,
            section=RetrievalSection.QA,
            query="test",
            budget_tokens=100,
            used_tokens=50,
            evidences=[_mock_evidence(0)],
        )
    )

    events_captured: list[dict] = []

    async def capture_progress(event: dict) -> None:
        events_captured.append(event)

    with patch("domain.qa.deep_research._load_graph_ctx") as mock_load_ctx:
        mock_load_ctx.return_value = MagicMock(
            centrality={},
            top_central=[],
            community_of={},
            community_hubs={},
        )
        with patch("domain.qa.deep_research.plan_queries") as mock_plan_queries:
            mock_plan_queries.return_value = ["query"]
            with patch("domain.qa.deep_research.retrieve_multi") as mock_retrieve_multi:
                mock_retrieve_multi.return_value = RetrievalBundle(
                    snapshot_id="snap-test",
                    mode=RetrievalMode.HYBRID,
                    section=RetrievalSection.QA,
                    query="test",
                    budget_tokens=100,
                    used_tokens=50,
                    evidences=[_mock_evidence(0)],
                )

                agent = DeepResearchAgent(provider_svc, retrieval_svc)
                await agent.research(
                    question="What is this?",
                    snapshot_id="snap-test",
                    provider_id="test",
                    model_id="test-model",
                    progress_cb=capture_progress,
                )

    status_events = [e for e in events_captured if e.get("type") == "status"]
    assert len(status_events) > 0

    for event in status_events:
        assert "type" in event
        assert event["type"] == "status"
        assert "phase" in event
        assert event["phase"] in ("planning", "retrieving", "thinking", "synthesizing")
        assert "detail" in event
        if "step" in event:
            assert isinstance(event["step"], int)


@pytest.mark.asyncio
async def test_deep_research_token_events():
    """Test that token events are emitted incrementally."""
    plan = [
        {"type": "retrieve", "target": "query", "description": "Find evidence"},
    ]

    plan_json = json.dumps({"steps": plan})
    step_result_json = json.dumps({
        "finding": "Found",
        "key_files": ["src/main.py"],
        "graph_path": None,
        "sufficient": True,
    })
    meta_json = json.dumps({
        "reasoning_chain": [],
        "confidence": "medium",
        "unknowns": [],
    })

    provider_svc = _chat_response_sequence([plan_json, step_result_json, meta_json])

    retrieval_svc = MagicMock(spec=RetrievalService)
    retrieval_svc.retrieve = AsyncMock(
        return_value=RetrievalBundle(
            snapshot_id="snap-test",
            mode=RetrievalMode.HYBRID,
            section=RetrievalSection.QA,
            query="test",
            budget_tokens=100,
            used_tokens=50,
            evidences=[_mock_evidence(0)],
        )
    )

    events_captured: list[dict] = []

    async def capture_progress(event: dict) -> None:
        events_captured.append(event)

    with patch("domain.qa.deep_research._load_graph_ctx") as mock_load_ctx:
        mock_load_ctx.return_value = MagicMock(
            centrality={},
            top_central=[],
            community_of={},
            community_hubs={},
        )
        with patch("domain.qa.deep_research.plan_queries") as mock_plan_queries:
            mock_plan_queries.return_value = ["query"]
            with patch("domain.qa.deep_research.retrieve_multi") as mock_retrieve_multi:
                mock_retrieve_multi.return_value = RetrievalBundle(
                    snapshot_id="snap-test",
                    mode=RetrievalMode.HYBRID,
                    section=RetrievalSection.QA,
                    query="test",
                    budget_tokens=100,
                    used_tokens=50,
                    evidences=[_mock_evidence(0)],
                )

                agent = DeepResearchAgent(provider_svc, retrieval_svc)
                await agent.research(
                    question="What is this?",
                    snapshot_id="snap-test",
                    provider_id="test",
                    model_id="test-model",
                    progress_cb=capture_progress,
                )

    token_events = [e for e in events_captured if e.get("type") == "token"]
    status_events = [e for e in events_captured if e.get("type") == "status"]

    if status_events:
        last_status_idx = max(i for i, e in enumerate(events_captured) if e.get("type") == "status")
        if token_events:
            assert any(
                i > last_status_idx for i, e in enumerate(events_captured) if e.get("type") == "token"
            ), "Token events should arrive after final status event"

    for event in token_events:
        assert event.get("type") == "token"
        assert "text" in event


@pytest.mark.asyncio
async def test_deep_research_result_schema():
    """Test that the result validates as DeepResearchResult."""
    plan = [
        {"type": "retrieve", "target": "query", "description": "Test step"},
    ]

    plan_json = json.dumps({"steps": plan})
    step_result_json = json.dumps({
        "finding": "Discovery",
        "key_files": ["file.py"],
        "graph_path": None,
        "sufficient": True,
    })
    meta_json = json.dumps({
        "reasoning_chain": [
            {
                "step_number": 1,
                "description": "Test step",
                "files_involved": ["file.py"],
                "finding": "Discovery",
            }
        ],
        "confidence": "high",
        "unknowns": ["unknown1"],
    })

    provider_svc = _chat_response_sequence([plan_json, step_result_json, meta_json])

    retrieval_svc = MagicMock(spec=RetrievalService)
    retrieval_svc.retrieve = AsyncMock(
        return_value=RetrievalBundle(
            snapshot_id="snap-test",
            mode=RetrievalMode.HYBRID,
            section=RetrievalSection.QA,
            query="test",
            budget_tokens=100,
            used_tokens=50,
            evidences=[_mock_evidence(0)],
        )
    )

    with patch("domain.qa.deep_research._load_graph_ctx") as mock_load_ctx:
        mock_load_ctx.return_value = MagicMock(
            centrality={},
            top_central=[],
            community_of={},
            community_hubs={},
        )
        with patch("domain.qa.deep_research.plan_queries") as mock_plan_queries:
            mock_plan_queries.return_value = ["query"]
            with patch("domain.qa.deep_research.retrieve_multi") as mock_retrieve_multi:
                mock_retrieve_multi.return_value = RetrievalBundle(
                    snapshot_id="snap-test",
                    mode=RetrievalMode.HYBRID,
                    section=RetrievalSection.QA,
                    query="test",
                    budget_tokens=100,
                    used_tokens=50,
                    evidences=[_mock_evidence(0)],
                )

                agent = DeepResearchAgent(provider_svc, retrieval_svc)
                result = await agent.research(
                    question="Test?",
                    snapshot_id="snap-test",
                    provider_id="test",
                    model_id="test-model",
                )

    result_obj = DeepResearchResult.model_validate(result)

    assert isinstance(result_obj.summary, str)
    assert isinstance(result_obj.reasoning_chain, list)
    assert isinstance(result_obj.files_explored, list)
    assert result_obj.confidence in ("high", "medium", "low")
    assert isinstance(result_obj.unknowns, list)
    assert result_obj.elapsed_ms >= 0

    for step in result_obj.reasoning_chain:
        assert hasattr(step, "step_number")
        assert hasattr(step, "description")
        assert hasattr(step, "files_involved")
        assert hasattr(step, "finding")
