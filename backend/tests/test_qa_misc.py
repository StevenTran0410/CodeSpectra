"""Misc backend QA suite: analysis pipeline, deep research, section progress events, external discovery routes, static risk test coverage, and local repo mode scoping (merged from 6 files)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import api.external as external
from domain.analysis.agent_pipeline import _SectionAgentComponent
from domain.analysis.agents._context_builders import prefetch_pipeline_context
from domain.analysis.agents.agent_architecture import ArchitectureAgent
from domain.analysis.agents.agent_auditor import AuditAgent
from domain.analysis.agents.agent_conventions import ConventionsAgent
from domain.analysis.agents.agent_feature_map import FeatureMapAgent
from domain.analysis.agents.agent_glossary import GlossaryAgent
from domain.analysis.agents.agent_important_files import ImportantFilesAgent
from domain.analysis.agents.agent_onboarding import OnboardingAgent
from domain.analysis.agents.agent_project_identity import ProjectIdentityAgent
from domain.analysis.agents.agent_risk import RiskAgent
from domain.analysis.agents.agent_structure import StructureAgent
from domain.analysis.agents.agent_violations import ViolationsAgent
from domain.analysis.model_guard import check_model_capability
from domain.analysis.profiles import LARGE_PROFILE, NORMAL_PROFILE, get_profile
from domain.analysis.prompts import AGENT_B_SCHEMA_STR, AGENT_B_SYSTEM
from domain.analysis.static_convention import ConventionReport
from domain.analysis.static_risk import RiskReport, detect_test_coverage_shape
from domain.local_repo.service import LocalRepoService
from domain.local_repo.types import AddLocalRepoRequest
from domain.model_connector.service import ProviderConfigService
from domain.model_connector.types import ChatResponse
from domain.qa.deep_research import DeepResearchAgent, DeepResearchResult
from domain.retrieval.service import RetrievalService, _chunk_size_for, _ends_mid_function, _token_estimate
from domain.retrieval.types import (
    RetrievalBundle,
    RetrievalEvidence,
    RetrievalMode,
    RetrievalSection,
)
from infrastructure.db.database import get_db
from shared.errors import ConflictError
from shared.utils import new_id, utc_now_iso
from tests.conftest import _FILE, MINIMAL_SECTION_JSON, chat_response_sequence

# #############################################################################
# test_analysis_pipeline.py — agent output contracts (A-K), pipeline memory
# context / profiles, retrieval scoring helpers, model guard
# #############################################################################


# ===========================================================================
# Helpers
# ===========================================================================


def _provider_for_letter(letter: str) -> MagicMock:
    content = json.dumps(MINIMAL_SECTION_JSON[letter])
    svc = MagicMock(spec=ProviderConfigService)
    svc.chat = AsyncMock(
        return_value=ChatResponse(provider_id="test-prov", model_id="test-model", content=content)
    )
    return svc


def _important_g() -> dict[str, Any]:
    return {
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
    }


def _sections_for_k() -> dict[str, Any]:
    return {ch: json.loads(json.dumps(MINIMAL_SECTION_JSON[ch])) for ch in "ABCDEFGHIJ"}


# ===========================================================================
# Agent output contracts (A–K)
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "letter",
    ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"],
)
async def test_agent_output_contract(
    letter: str,
    mock_retrieval: MagicMock,
    sample_pipeline_memory_context: Any,
) -> None:
    prov = _provider_for_letter(letter)
    empty_conv = ConventionReport(signals=[])
    empty_risk = RiskReport(findings=[])

    if letter == "A":
        out = await ProjectIdentityAgent(prov, mock_retrieval).run(
            "p", "m", "snap", "", mem_ctx=sample_pipeline_memory_context
        )
    elif letter == "B":
        out = await ArchitectureAgent(prov, mock_retrieval).run(
            "p",
            "m",
            "snap",
            None,
            sample_pipeline_memory_context.arch_bundle,
            identity_output={},
        )
    elif letter == "C":
        out = await StructureAgent(prov, mock_retrieval).run(
            "p",
            "m",
            "snap",
            sample_pipeline_memory_context.arch_bundle,
            sample_pipeline_memory_context.folder_tree,
            identity_output={},
        )
    elif letter == "D":
        out = await ConventionsAgent(prov, mock_retrieval).run(
            "p", "m", "snap", empty_conv, structure_output=None
        )
    elif letter == "E":
        out = await ViolationsAgent(prov, mock_retrieval).run(
            "p", "m", "snap", empty_conv, empty_risk, conventions_output=None
        )
    elif letter == "F":
        out = await FeatureMapAgent(prov, mock_retrieval).run(
            "p", "m", "snap", None, identity_output={}, architecture_output={}
        )
    elif letter == "G":
        out = await ImportantFilesAgent(prov, mock_retrieval).run("p", "m", "snap", None)
    elif letter == "H":
        out = await OnboardingAgent(prov, mock_retrieval).run("p", "m", "snap", _important_g())
    elif letter == "I":
        out = await GlossaryAgent(prov, mock_retrieval).run("p", "m", "snap")
    elif letter == "J":
        out = await RiskAgent(prov, mock_retrieval).run("p", "m", "snap", empty_risk)
    elif letter == "K":
        out = await AuditAgent(prov).run("p", "m", _sections_for_k())
    else:
        raise AssertionError(letter)

    assert isinstance(out, dict)
    assert "blind_spots" in out
    if letter == "K":
        assert "overall_confidence" in out
    else:
        assert "confidence" in out
    if letter not in ("I", "K"):
        assert "evidence_files" in out
    if letter != "K":
        assert "content" in out


@pytest.mark.asyncio
async def test_architecture_identity_output_kwarg_regression(
    mock_retrieval: MagicMock,
    sample_pipeline_memory_context: Any,
) -> None:
    prov = _provider_for_letter("B")
    agent = ArchitectureAgent(prov, mock_retrieval)
    result = await agent.run(
        "p",
        "m",
        "snap",
        None,
        sample_pipeline_memory_context.arch_bundle,
        identity_output={"domain": "api", "tech_stack": ["python"]},
    )
    assert isinstance(result, dict)
    assert prov.chat.await_count == 1


@pytest.mark.asyncio
async def test_chat_json_repair_prose_then_json(mock_retrieval: MagicMock) -> None:
    good = json.dumps(MINIMAL_SECTION_JSON["B"])
    prov = chat_response_sequence(["not json at all", good])
    agent = ArchitectureAgent(prov, mock_retrieval)
    out = await agent._chat_json_typed(
        "p",
        "m",
        AGENT_B_SYSTEM,
        "{}",
        AGENT_B_SCHEMA_STR,
        max_completion_tokens=500,
    )
    assert out.get("main_layers") is not None
    assert prov.chat.await_count == 2


@pytest.mark.asyncio
async def test_chat_json_attempt3_hardcoded_fallback(mock_retrieval: MagicMock) -> None:
    prov = chat_response_sequence(["not json", "also not json", "still not json"])
    agent = ArchitectureAgent(prov, mock_retrieval)
    out = await agent._chat_json_typed(
        "p",
        "m",
        AGENT_B_SYSTEM,
        "{}",
        AGENT_B_SCHEMA_STR,
        max_completion_tokens=400,
    )
    assert out["blind_spots"] == ["output_repair_failed"]
    assert out["confidence"] == "low"
    assert prov.chat.await_count == 3


@pytest.mark.asyncio
async def test_chat_json_attempt1_happy_path_single_call(mock_retrieval: MagicMock) -> None:
    good = json.dumps(MINIMAL_SECTION_JSON["B"])
    prov = chat_response_sequence([good])
    agent = ArchitectureAgent(prov, mock_retrieval)
    out = await agent._chat_json_typed(
        "p",
        "m",
        AGENT_B_SYSTEM,
        "{}",
        AGENT_B_SCHEMA_STR,
        max_completion_tokens=400,
    )
    assert isinstance(out, dict) and out.get("main_layers") is not None
    assert prov.chat.await_count == 1


# ===========================================================================
# Pipeline memory context — prefetch + agent profiles
# ===========================================================================


@pytest.mark.asyncio
async def test_prefetch_pipeline_context_populates_all_fields(
    mock_retrieval: MagicMock,
    canned_retrieval_bundle: RetrievalBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from domain.analysis.agents import _context_builders as cb

    async def fake_tree(_snapshot_id: str, _max: int = 60) -> str:
        return "src/a.py\nsrc/b.py"

    async def fake_patterns(
        _snapshot_id: str,
        _patterns: tuple[str, ...],
        char_limit: int = 0,
        max_rows: int = 6,
    ) -> str:
        return "doc-content"

    monkeypatch.setattr(cb, "build_folder_summary", fake_tree)
    monkeypatch.setattr(cb, "_fetch_files_by_pattern", fake_patterns)

    ctx = await prefetch_pipeline_context(mock_retrieval, "snap-xyz", mode=RetrievalMode.HYBRID)
    assert ctx.arch_bundle is canned_retrieval_bundle
    assert "a.py" in ctx.folder_tree
    assert ctx.doc_files == "doc-content"
    assert ctx.manifest_files == "doc-content"
    mock_retrieval.retrieve.assert_awaited()


@pytest.mark.asyncio
async def test_prefetch_failure_graceful_none_like(mock_retrieval: MagicMock) -> None:
    mock_retrieval.retrieve = AsyncMock(side_effect=RuntimeError("retrieve failed"))
    mem_ctx = None
    try:
        mem_ctx = await prefetch_pipeline_context(mock_retrieval, "snap-fail")
    except Exception:
        mem_ctx = None
    assert mem_ctx is None


def test_get_profile_normal_returns_normal_profile() -> None:
    p = get_profile(False)
    assert p is NORMAL_PROFILE
    assert p.mode == "normal"


def test_get_profile_large_returns_large_profile() -> None:
    p = get_profile(True)
    assert p is LARGE_PROFILE
    assert p.mode == "large"


def test_normal_profile_values_unchanged() -> None:
    """NORMAL_PROFILE must match legacy hardcoded values exactly."""
    assert NORMAL_PROFILE.retrieval_max_results == 30
    assert NORMAL_PROFILE.retrieval_manifest_char_limit == 3000
    assert NORMAL_PROFILE.tokens_project_identity == 2000
    assert NORMAL_PROFILE.tokens_architecture == 2500
    assert NORMAL_PROFILE.tokens_structure == 2000
    assert NORMAL_PROFILE.tokens_conventions == 3000
    assert NORMAL_PROFILE.tokens_violations == 2000
    assert NORMAL_PROFILE.tokens_feature_map == 5000
    assert NORMAL_PROFILE.tokens_important_files == 2000
    assert NORMAL_PROFILE.tokens_onboarding == 4000
    assert NORMAL_PROFILE.tokens_glossary == 3000
    assert NORMAL_PROFILE.tokens_risk == 3000
    assert NORMAL_PROFILE.tokens_auditor == 2000
    assert NORMAL_PROFILE.tokens_synthesizer == 4000
    assert NORMAL_PROFILE.concurrency_scale == 1.0


def test_large_profile_budgets_exceed_normal() -> None:
    """LARGE_PROFILE must have higher budgets than NORMAL_PROFILE."""
    assert LARGE_PROFILE.retrieval_max_results >= NORMAL_PROFILE.retrieval_max_results
    assert LARGE_PROFILE.retrieval_manifest_char_limit > NORMAL_PROFILE.retrieval_manifest_char_limit
    assert LARGE_PROFILE.tokens_architecture > NORMAL_PROFILE.tokens_architecture
    assert LARGE_PROFILE.tokens_feature_map > NORMAL_PROFILE.tokens_feature_map
    assert LARGE_PROFILE.concurrency_scale > NORMAL_PROFILE.concurrency_scale


def test_profiles_respect_provider_ceiling() -> None:
    """No profile value must exceed provider hard limits."""
    _MAX_TOKENS = 8192
    _MAX_RESULTS = 60
    for profile in (NORMAL_PROFILE, LARGE_PROFILE):
        for attr in (
            "tokens_project_identity",
            "tokens_architecture",
            "tokens_structure",
            "tokens_conventions",
            "tokens_violations",
            "tokens_feature_map",
            "tokens_important_files",
            "tokens_onboarding",
            "tokens_glossary",
            "tokens_risk",
            "tokens_auditor",
            "tokens_synthesizer",
        ):
            assert getattr(profile, attr) <= _MAX_TOKENS, (
                f"{profile.mode}.{attr}={getattr(profile, attr)} exceeds {_MAX_TOKENS}"
            )
        assert profile.retrieval_max_results <= _MAX_RESULTS
        assert profile.retrieval_arch_max_results <= _MAX_RESULTS


@pytest.mark.asyncio
async def test_prefetch_uses_profile_arch_max_results(
    mock_retrieval: MagicMock,
    canned_retrieval_bundle: RetrievalBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """prefetch_pipeline_context must pass profile.retrieval_arch_max_results to retrieve."""
    from domain.analysis.agents import _context_builders as cb

    monkeypatch.setattr(cb, "build_folder_summary", AsyncMock(return_value=""))
    monkeypatch.setattr(cb, "_fetch_files_by_pattern", AsyncMock(return_value=""))

    await prefetch_pipeline_context(mock_retrieval, "snap-large", profile=LARGE_PROFILE)
    arch_req = next(
        c[0][0]
        for c in mock_retrieval.retrieve.call_args_list
        if c[0][0].section == RetrievalSection.ARCHITECTURE
    )
    assert arch_req.max_results == LARGE_PROFILE.retrieval_arch_max_results


@pytest.mark.asyncio
async def test_prefetch_defaults_to_normal_profile(
    mock_retrieval: MagicMock,
    canned_retrieval_bundle: RetrievalBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """prefetch_pipeline_context with profile=None must behave like NORMAL_PROFILE."""
    from domain.analysis.agents import _context_builders as cb

    monkeypatch.setattr(cb, "build_folder_summary", AsyncMock(return_value=""))
    monkeypatch.setattr(cb, "_fetch_files_by_pattern", AsyncMock(return_value=""))

    await prefetch_pipeline_context(mock_retrieval, "snap-default")
    arch_req = next(
        c[0][0]
        for c in mock_retrieval.retrieve.call_args_list
        if c[0][0].section == RetrievalSection.ARCHITECTURE
    )
    assert arch_req.max_results == NORMAL_PROFILE.retrieval_arch_max_results


# ===========================================================================
# Retrieval scoring helpers + model guard
# ===========================================================================


def test_ends_mid_function_brace_unbalanced() -> None:
    assert _ends_mid_function("function foo() { return 1;", "typescript") is True


def test_ends_mid_function_brace_balanced() -> None:
    assert _ends_mid_function("function foo() { return 1; }", "javascript") is False


def test_ends_mid_function_python_mid_body() -> None:
    src = "def foo():\n    x = 1\n    return x"
    assert _ends_mid_function(src, "python") is True


def test_ends_mid_function_python_complete() -> None:
    src = "def foo():\n    return 1\n\nprint(foo())"
    assert _ends_mid_function(src, "python") is False


def test_chunk_size_for_categories_and_lang() -> None:
    assert _chunk_size_for("docs", None) == 1800
    assert _chunk_size_for("config", "python") == 1200
    assert _chunk_size_for("test", "go") == 1400
    assert _chunk_size_for("source", "python") == 1500
    assert _chunk_size_for("source", "rust") == 1300


def test_token_estimate_minimum_one() -> None:
    assert _token_estimate("") == 1
    assert _token_estimate("abab") == 1
    assert _token_estimate("abcd" * 10) == 10


def test_check_model_capability_small_models_warn() -> None:
    w = check_model_capability("qwen2.5:1.5b")
    assert w is not None
    assert w["code"] == "model_too_small"
    assert w["severity"] == "warn"
    assert check_model_capability("phi3:mini") is not None
    assert check_model_capability("TinyLlama/TinyLlama-1.1B") is not None


def test_check_model_capability_larger_ok() -> None:
    assert check_model_capability("llama3.1:8b") is None
    assert check_model_capability("qwen2.5:72b") is None


# ===========================================================================
# Agent profile integration with retrieval max_results
# ===========================================================================


@pytest.mark.asyncio
async def test_glossary_agent_uses_profile_max_results(
    mock_retrieval: MagicMock,
    mock_provider_chat_json: MagicMock,
) -> None:
    """GlossaryAgent.run() must pass profile.retrieval_max_results to retrieve."""
    agent = GlossaryAgent(mock_provider_chat_json, mock_retrieval)
    await agent.run("prov", "mdl", "snap-I", profile=LARGE_PROFILE)
    req = mock_retrieval.retrieve.call_args[0][0]
    assert req.max_results == LARGE_PROFILE.retrieval_max_results


@pytest.mark.asyncio
@pytest.mark.parametrize("mock_provider_chat_json", ["I"], indirect=True)
async def test_glossary_agent_normal_profile_unchanged(
    mock_retrieval: MagicMock,
    mock_provider_chat_json: MagicMock,
) -> None:
    """GlossaryAgent with no profile must use NORMAL_PROFILE defaults (non-regression)."""
    agent = GlossaryAgent(mock_provider_chat_json, mock_retrieval)
    await agent.run("prov", "mdl", "snap-I-normal")
    req = mock_retrieval.retrieve.call_args[0][0]
    assert req.max_results == NORMAL_PROFILE.retrieval_max_results == 30


@pytest.mark.skip(
    reason="Passes in isolation; hangs only in the full suite. Leaked async resources accumulate on "
    "the shared session event loop until get_db() (via retrieve_multi->promote_dominant_symbols) parks "
    "in _poll. Loop-level test-isolation issue (module-scoped DB reset does not fix it), not a RiskAgent defect."
)
@pytest.mark.asyncio
@pytest.mark.parametrize("mock_provider_chat_json", ["J"], indirect=True)
async def test_risk_agent_large_profile_max_results(
    mock_retrieval: MagicMock,
    mock_provider_chat_json: MagicMock,
) -> None:
    """RiskAgent.run() must scale retrieval depth in large mode."""
    agent = RiskAgent(mock_provider_chat_json, mock_retrieval)
    await agent.run("prov", "mdl", "snap-J", profile=LARGE_PROFILE)
    req = mock_retrieval.retrieve.call_args[0][0]
    assert req.max_results == LARGE_PROFILE.retrieval_max_results
    assert req.max_results > NORMAL_PROFILE.retrieval_max_results


def test_pipeline_audit_quality_detects_low_confidence_sections() -> None:
    """_audit_quality must return letters whose sections self-report low confidence."""
    from domain.analysis.agent_pipeline import AnalysisAgentPipeline

    with patch.object(AnalysisAgentPipeline, "__init__", lambda self, *a, **kw: None):
        pipeline = AnalysisAgentPipeline.__new__(AnalysisAgentPipeline)

    sections: dict = {
        "A": {"confidence": "high"},
        "B": {"confidence": "low"},
        "C": {"confidence": "medium"},
        "D": {"confidence": "low"},
        "E": {"confidence": "high"},
        "F": {"confidence": "high"},
        "G": {"confidence": "high"},
        "H": {"confidence": "high"},
        "I": {"confidence": "high"},
        "J": {"confidence": "high"},
        "K": {"overall_confidence": "low", "weakest_sections": ["B", "D"]},
        "L": {"confidence": "medium"},
    }
    weak = pipeline._audit_quality(sections)
    assert "B" in weak
    assert "D" in weak
    assert "A" not in weak


def test_pipeline_audit_quality_empty_k_falls_back_to_section_confidence() -> None:
    """_audit_quality falls back to individual section confidence when K is absent."""
    from domain.analysis.agent_pipeline import AnalysisAgentPipeline

    with patch.object(AnalysisAgentPipeline, "__init__", lambda self, *a, **kw: None):
        pipeline = AnalysisAgentPipeline.__new__(AnalysisAgentPipeline)

    sections: dict = {letter: {"confidence": "high"} for letter in "ABCDEFGHIJ"}
    sections["F"] = {"confidence": "low"}
    weak = pipeline._audit_quality(sections)
    assert "F" in weak
    assert len(weak) == 1


# #############################################################################
# test_deep_research_langgraph.py — LangGraph-based Deep Research agent
# #############################################################################


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


# #############################################################################
# test_section_progress_events.py — backend progress updates
# #############################################################################


@pytest.mark.asyncio
async def test_section_agent_emits_running_status() -> None:
    """Verify that _SectionAgentComponent emits running and done status sequentially."""
    calls = []

    async def mock_on_section_done(section, status, duration_ms, data, error) -> None:
        calls.append((section, status, data))

    async def mock_runner(ctx, deps):
        return {"output_key": "some_value"}

    def mock_fallback(ctx, deps):
        return {}

    comp = _SectionAgentComponent(
        section="A",
        runner=mock_runner,
        fallback=mock_fallback,
        on_section_done=mock_on_section_done,
    )

    ctx = {"provider_id": "p", "model_id": "m", "snapshot_id": "s", "repo_name": "r"}
    await comp.run_async(ctx)

    # Should have two calls: 'running' first, then 'done'
    assert len(calls) == 2
    assert calls[0] == ("A", "running", None)
    assert calls[1] == ("A", "done", {"output_key": "some_value"})


# #############################################################################
# test_external_discovery_routes.py — thin-delegation tests for /api/external/*
# read routes: each handler must call the service method and return its result
# unchanged, verified via mocks at the service-call boundary (no live HTTP server)
# #############################################################################


@pytest.mark.asyncio
async def test_search_retrieval_delegates_to_retrieval_service() -> None:
    """Wraps retrieve_rrf_fusion (unbounded, BM25-weighted), not the plain budget-capped retrieve() — a budget cap can silently drop a chunk with an exact fingerprint match below an unrelated higher-scoring chunk, confirmed empirically against this repo's own real haystack import."""
    from domain.retrieval.types import RrfFusionRequest, RetrievalSection

    sentinel = object()
    with patch.object(external, "_retrieval_service") as mock_service:
        mock_service.retrieve_rrf_fusion = AsyncMock(return_value=sentinel)
        body = RrfFusionRequest(snapshot_id="snap-1", query="q", section=RetrievalSection.QA)
        result = await external.search_retrieval(body)

    mock_service.retrieve_rrf_fusion.assert_awaited_once_with(body)
    assert result is sentinel


@pytest.mark.asyncio
async def test_get_graph_neighbors_delegates_with_correct_args() -> None:
    sentinel = object()
    with patch.object(external, "_graph_service") as mock_service:
        mock_service.neighbors = AsyncMock(return_value=sentinel)
        result = await external.get_graph_neighbors("snap-1", "some/path.py", hops=2, limit=50)

    mock_service.neighbors.assert_awaited_once_with("snap-1", "some/path.py", 2, 50)
    assert result is sentinel


@pytest.mark.asyncio
async def test_get_graph_communities_delegates() -> None:
    sentinel = object()
    with patch.object(external, "_graph_service") as mock_service:
        mock_service.list_communities = AsyncMock(return_value=sentinel)
        result = await external.get_graph_communities("snap-1")

    mock_service.list_communities.assert_awaited_once_with("snap-1")
    assert result is sentinel


@pytest.mark.asyncio
async def test_get_graph_symbol_edges_delegates() -> None:
    sentinel = object()
    with patch.object(external, "_graph_service") as mock_service:
        mock_service.symbol_edges_for_file = AsyncMock(return_value=sentinel)
        result = await external.get_graph_symbol_edges("snap-1", "some/path.py")

    mock_service.symbol_edges_for_file.assert_awaited_once_with("snap-1", "some/path.py")
    assert result is sentinel


@pytest.mark.asyncio
async def test_search_repo_map_symbols_delegates() -> None:
    sentinel = object()
    with patch.object(external, "_repo_map_service") as mock_service:
        mock_service.search = AsyncMock(return_value=sentinel)
        result = await external.search_repo_map_symbols("snap-1", "MyClass", limit=10)

    mock_service.search.assert_awaited_once_with("snap-1", "MyClass", 10)
    assert result is sentinel


@pytest.mark.asyncio
async def test_read_manifest_file_delegates() -> None:
    sentinel = object()
    with patch.object(external, "_manifest_service") as mock_service:
        mock_service.read_file = AsyncMock(return_value=sentinel)
        result = await external.read_manifest_file("snap-1", "src/foo.py", max_bytes=1000)

    mock_service.read_file.assert_awaited_once_with("snap-1", "src/foo.py", 1000)
    assert result is sentinel


@pytest.mark.asyncio
async def test_list_analysis_reports_delegates() -> None:
    sentinel = object()
    with patch.object(external, "_analysis_service") as mock_service:
        mock_service.list_reports = AsyncMock(return_value=sentinel)
        result = await external.list_analysis_reports(repo_id="repo-1", workspace_id="ws-1", limit=5)

    mock_service.list_reports.assert_awaited_once_with("repo-1", "ws-1", 5)
    assert result is sentinel


@pytest.mark.asyncio
async def test_get_analysis_report_delegates() -> None:
    sentinel = object()
    with patch.object(external, "_analysis_service") as mock_service:
        mock_service.get_report = AsyncMock(return_value=sentinel)
        result = await external.get_analysis_report("report-1")

    mock_service.get_report.assert_awaited_once_with("report-1")
    assert result is sentinel


@pytest.mark.asyncio
async def test_get_repo_snapshot_delegates() -> None:
    sentinel = object()
    with patch.object(external, "_sync_service") as mock_service:
        mock_service.get_snapshot = AsyncMock(return_value=sentinel)
        result = await external.get_repo_snapshot("snap-1")

    mock_service.get_snapshot.assert_awaited_once_with("snap-1")
    assert result is sentinel


@pytest.mark.asyncio
async def test_get_local_repo_delegates() -> None:
    sentinel = object()
    with patch.object(external, "_local_repo_service") as mock_service:
        mock_service.get_by_id = AsyncMock(return_value=sentinel)
        result = await external.get_local_repo("repo-1")

    mock_service.get_by_id.assert_awaited_once_with("repo-1")
    assert result is sentinel


@pytest.mark.asyncio
async def test_list_local_repos_delegates() -> None:
    sentinel = object()
    with patch.object(external, "_local_repo_service") as mock_service:
        mock_service.list_all = AsyncMock(return_value=sentinel)
        result = await external.list_local_repos(workspace_id="ws-1", mode="aeh")

    mock_service.list_all.assert_awaited_once_with("ws-1", "aeh")
    assert result is sentinel


@pytest.mark.asyncio
async def test_llm_embed_delegates_to_provider_service() -> None:
    from domain.model_connector.types import EmbedRequest
    sentinel = object()
    with patch.object(external, "_service") as mock_service:
        mock_service.embed = AsyncMock(return_value=sentinel)
        body = external.LLMEmbedRequest(
            provider_id="p1", model_id="m1", texts=["test"], task_type="retrieval_document"
        )
        result = await external.llm_embed(body)

    mock_service.embed.assert_awaited_once_with(
        EmbedRequest(
            provider_id="p1", model_id="m1", texts=["test"], task_type="retrieval_document"
        )
    )
    assert result is sentinel


@pytest.mark.asyncio
async def test_all_new_routes_are_token_gated() -> None:
    """Every new route must depend on require_external_token — a route that forgets this dependency would leak CodeSpectra's index to any local process without the bearer token, defeating the entire narrow-slice design."""
    gated_paths = {
        "/retrieval/search",
        "/graph/{snapshot_id}/neighbors",
        "/graph/{snapshot_id}/communities",
        "/graph/{snapshot_id}/symbol-edges",
        "/repo-map/{snapshot_id}/search",
        "/manifest/{snapshot_id}/file",
        "/analysis/reports",
        "/analysis/reports/{report_id}",
        "/snapshots/{snapshot_id}",
        "/repos/{repo_id}",
        "/repos",
        "/llm/embed",
    }
    found = {route.path for route in external.router.routes if route.path in gated_paths}
    assert found == gated_paths, f"missing routes: {gated_paths - found}"

    for route in external.router.routes:
        if route.path in gated_paths:
            dependant_calls = [d.call for d in route.dependant.dependencies]
            assert external.require_external_token in dependant_calls, (
                f"route {route.path} is missing the require_external_token dependency"
            )


# #############################################################################
# test_static_risk_test_coverage.py — static_risk.py's detect_test_coverage_shape
# and the include_tests guard: when a user explicitly excludes test files from
# indexing (include_tests=False, the default), manifest_files never contains any
# test files for that snapshot -- every module would otherwise look like it has
# zero coverage, which is a false positive caused by the user's own indexing
# choice, not a real test gap.
# #############################################################################


async def _seed_repo_and_snapshot(include_tests: bool) -> str:
    db = get_db()
    repo_id = new_id()
    snap_id = new_id()
    now = utc_now_iso()

    await db.execute(
        """
        INSERT INTO local_repos
        (id, path, name, added_at, last_validated_at, include_tests)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (repo_id, f"/tmp/{repo_id}", "test_repo", now, now, int(include_tests)),
    )
    await db.execute(
        """
        INSERT INTO repo_snapshots
        (id, local_repo_id, local_path, synced_at, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (snap_id, repo_id, f"/tmp/{repo_id}", now, now),
    )

    # A module with 10 source files and zero test files -- would normally trigger
    # a "No test coverage" finding (len(src_files) >= 10 -> severity "high").
    manifest_rows = [
        (
            new_id(), snap_id, f"backend/domain/widgets/file_{i}.py",
            "python", "source", 100, 0, f"hash{i}",
        )
        for i in range(10)
    ]
    await db.executemany(
        """
        INSERT INTO manifest_files
        (id, snapshot_id, rel_path, language, category, size_bytes, mtime_ns, checksum)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        manifest_rows,
    )
    await db.commit()
    return snap_id


@pytest.mark.asyncio
async def test_no_findings_when_include_tests_false():
    """include_tests=False -> detector must skip entirely, no false 'no test coverage' findings, even though this fixture has 10 source files and zero test files."""
    snap_id = await _seed_repo_and_snapshot(include_tests=False)
    db = get_db()

    findings = await detect_test_coverage_shape(snap_id, db)

    assert findings == []


@pytest.mark.asyncio
async def test_findings_still_emitted_when_include_tests_true():
    """include_tests=True -> detector behaves as before, real gaps still reported."""
    snap_id = await _seed_repo_and_snapshot(include_tests=True)
    db = get_db()

    findings = await detect_test_coverage_shape(snap_id, db)

    assert len(findings) >= 1
    assert any(f.category == "test_gap" for f in findings)
    assert any("No test coverage" in f.title for f in findings)


# #############################################################################
# test_local_repo_mode_scoping.py — AEH must have an independent local_repos
# lineage from Code Analysis for the same folder path (own include_tests, own
# mode-scoped listing)
# #############################################################################


@pytest.fixture
def service() -> LocalRepoService:
    return LocalRepoService()


async def test_same_path_can_be_added_under_both_modes(service: LocalRepoService, tmp_path) -> None:
    path = str(tmp_path)

    ca_repo = await service.add(AddLocalRepoRequest(path=path, mode="code_analysis"))
    aeh_repo = await service.add(AddLocalRepoRequest(path=path, mode="aeh"))

    assert ca_repo.id != aeh_repo.id
    assert ca_repo.path == aeh_repo.path
    assert ca_repo.mode == "code_analysis"
    assert aeh_repo.mode == "aeh"


async def test_aeh_mode_add_sets_include_tests_true_automatically(
    service: LocalRepoService, tmp_path
) -> None:
    repo = await service.add(AddLocalRepoRequest(path=str(tmp_path), mode="aeh"))
    assert repo.include_tests is True


async def test_code_analysis_mode_add_keeps_include_tests_false(
    service: LocalRepoService, tmp_path
) -> None:
    repo = await service.add(AddLocalRepoRequest(path=str(tmp_path), mode="code_analysis"))
    assert repo.include_tests is False


async def test_duplicate_add_same_path_and_mode_rejected(service: LocalRepoService, tmp_path) -> None:
    path = str(tmp_path)
    await service.add(AddLocalRepoRequest(path=path, mode="code_analysis"))
    with pytest.raises(ConflictError):
        await service.add(AddLocalRepoRequest(path=path, mode="code_analysis"))


async def test_list_all_filters_by_mode(service: LocalRepoService, tmp_path) -> None:
    path = str(tmp_path)
    ca_repo = await service.add(AddLocalRepoRequest(path=path, mode="code_analysis"))
    aeh_repo = await service.add(AddLocalRepoRequest(path=path, mode="aeh"))

    ca_only = await service.list_all(mode="code_analysis")
    aeh_only = await service.list_all(mode="aeh")

    ca_ids = {r.id for r in ca_only}
    aeh_ids = {r.id for r in aeh_only}

    assert ca_repo.id in ca_ids
    assert ca_repo.id not in aeh_ids
    assert aeh_repo.id in aeh_ids
    assert aeh_repo.id not in ca_ids


async def test_remove_keeps_managed_folder_when_another_mode_still_references_it(
    service: LocalRepoService, tmp_path, monkeypatch
) -> None:
    fake_home = tmp_path / "home"
    managed_dir = fake_home / "CodeSpectra" / "repos" / "shared"
    managed_dir.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))

    path = str(managed_dir)
    ca_repo = await service.add(AddLocalRepoRequest(path=path, mode="code_analysis"))
    aeh_repo = await service.add(AddLocalRepoRequest(path=path, mode="aeh"))

    await service.remove(aeh_repo.id)

    assert managed_dir.exists()
    remaining_ca = await service.list_all(mode="code_analysis")
    assert any(r.id == ca_repo.id for r in remaining_ca)


async def test_remove_deletes_managed_folder_once_no_repo_references_it(
    service: LocalRepoService, tmp_path, monkeypatch
) -> None:
    fake_home = tmp_path / "home"
    managed_dir = fake_home / "CodeSpectra" / "repos" / "solo"
    managed_dir.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))

    repo = await service.add(AddLocalRepoRequest(path=str(managed_dir), mode="aeh"))
    await service.remove(repo.id)

    assert not managed_dir.exists()
