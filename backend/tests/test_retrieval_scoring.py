from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from domain.analysis.model_guard import check_model_capability
from domain.analysis.profiles import LARGE_PROFILE, NORMAL_PROFILE
from domain.retrieval.service import _chunk_size_for, _ends_mid_function, _token_estimate
from domain.retrieval.types import RetrievalSection


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


# ---------------------------------------------------------------------------
# CS-012: agent profile integration with retrieval max_results
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_glossary_agent_uses_profile_max_results(
    mock_retrieval: MagicMock,
    mock_provider_chat_json: MagicMock,
) -> None:
    """GlossaryAgent.run() must pass profile.retrieval_max_results to retrieve."""
    from domain.analysis.agents.agent_glossary import GlossaryAgent

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
    from domain.analysis.agents.agent_glossary import GlossaryAgent

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
    from domain.analysis.agents.agent_risk import RiskAgent

    agent = RiskAgent(mock_provider_chat_json, mock_retrieval)
    await agent.run("prov", "mdl", "snap-J", profile=LARGE_PROFILE)
    req = mock_retrieval.retrieve.call_args[0][0]
    assert req.max_results == LARGE_PROFILE.retrieval_max_results
    assert req.max_results > NORMAL_PROFILE.retrieval_max_results


def test_pipeline_audit_quality_detects_low_confidence_sections() -> None:
    """_audit_quality must return letters whose sections self-report low confidence."""
    from unittest.mock import MagicMock, patch

    from domain.analysis.agent_pipeline import AnalysisAgentPipeline

    # Build a minimal pipeline stub (no actual agents needed)
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
    # B and D should be flagged as weak
    assert "B" in weak
    assert "D" in weak
    # A, C, E .. J (non-low) should not be flagged
    assert "A" not in weak


def test_pipeline_audit_quality_empty_k_falls_back_to_section_confidence() -> None:
    """_audit_quality falls back to individual section confidence when K is absent."""
    from unittest.mock import patch

    from domain.analysis.agent_pipeline import AnalysisAgentPipeline

    with patch.object(AnalysisAgentPipeline, "__init__", lambda self, *a, **kw: None):
        pipeline = AnalysisAgentPipeline.__new__(AnalysisAgentPipeline)

    sections: dict = {letter: {"confidence": "high"} for letter in "ABCDEFGHIJ"}
    sections["F"] = {"confidence": "low"}
    # No K section — fallback to individual confidence
    weak = pipeline._audit_quality(sections)
    assert "F" in weak
    assert len(weak) == 1
