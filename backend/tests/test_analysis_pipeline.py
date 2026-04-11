"""Analysis pipeline tests — agent contracts, memory context, retrieval scoring, profiles.

Merged from:
  - test_agents_contract.py     (agent A–K output contracts, JSON repair logic)
  - test_pipeline_memory_context.py (prefetch pipeline context, CS-012 profiles)
  - test_retrieval_scoring.py   (retrieval helpers, model guard, profile integration)
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
from domain.analysis.static_risk import RiskReport
from domain.model_connector.service import ProviderConfigService
from domain.model_connector.types import ChatResponse
from domain.retrieval.service import _chunk_size_for, _ends_mid_function, _token_estimate
from domain.retrieval.types import RetrievalBundle, RetrievalMode, RetrievalSection
from tests.conftest import _FILE, MINIMAL_SECTION_JSON, chat_response_sequence


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
# Pipeline memory context — prefetch + CS-012 profiles
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

    monkeypatch.setattr(cb, "fetch_folder_tree", fake_tree)
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
    """NORMAL_PROFILE must match pre-CS-012 hardcoded values exactly."""
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

    monkeypatch.setattr(cb, "fetch_folder_tree", AsyncMock(return_value=""))
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

    monkeypatch.setattr(cb, "fetch_folder_tree", AsyncMock(return_value=""))
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
# CS-012: agent profile integration with retrieval max_results
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
