from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

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
from domain.analysis.prompts import AGENT_B_SCHEMA_STR, AGENT_B_SYSTEM
from domain.analysis.static_convention import ConventionReport
from domain.analysis.static_risk import RiskReport
from domain.model_connector.service import ProviderConfigService
from domain.model_connector.types import ChatResponse
from tests.conftest import _FILE, MINIMAL_SECTION_JSON, chat_response_sequence


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
