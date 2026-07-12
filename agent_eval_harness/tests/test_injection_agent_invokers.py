"""Injection runner -- per-agent invokers (CS-289 Phases 0-5).

Requires the backend source importable via the sys.path bridge (agent_eval_harness venv already
has every transitive dep the agents need). If domain.* can't resolve, these skip rather than error.
"""
from __future__ import annotations

import json

import pytest

from agent_eval_harness.injection.backend_bridge import ensure_backend_importable

try:
    ensure_backend_importable()
    from domain.model_connector.service import ProviderConfigService
    from domain.model_connector.types import ChatResponse
    _BACKEND = True
except Exception:
    _BACKEND = False

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not _BACKEND, reason="backend source not importable (set backend_source_path)"),
]


def _mock_provider(canned: dict) -> "ProviderConfigService":
    from unittest.mock import AsyncMock, MagicMock

    svc = MagicMock(spec=ProviderConfigService)
    svc.chat = AsyncMock(
        return_value=ChatResponse(provider_id="p", model_id="m", content=json.dumps(canned))
    )
    return svc


def _mock_provider_sequence(contents: list[str]) -> "ProviderConfigService":
    """Serves each chat() call the next raw string in order -- for query-planning agents
    (D/J/F), whose plan-step call isn't json_mode and expects a plain JSON-array-shaped
    string, not a schema-validated object."""
    from unittest.mock import AsyncMock, MagicMock

    svc = MagicMock(spec=ProviderConfigService)
    queue = list(contents)

    async def _chat(_req):
        content = queue.pop(0) if queue else "{}"
        return ChatResponse(provider_id="p", model_id="m", content=content)

    svc.chat = AsyncMock(side_effect=_chat)
    return svc


async def test_invoke_auditor_runs_real_agent_and_calls_llm_once() -> None:
    from agent_eval_harness.injection.agent_invokers import invoke_auditor

    prov = _mock_provider(
        {"overall_confidence": "high", "section_scores": {"A": "high"}, "weakest_sections": ["B"]}
    )
    case_input = {
        "shape": "all_sections",
        "all_sections": {L: {"confidence": "high", "blind_spots": [], "summary": f"s{L}"} for L in "ABCDEFGHIJ"},
    }
    out = await invoke_auditor(case_input, prov, "p", "m")

    assert prov.chat.await_count == 1  # the agent's real LLM boundary, mocked
    assert out["overall_confidence"] == "high"
    assert "section_scores" in out  # real run() post-processing ran


async def test_invoke_synthesizer_runs_real_agent() -> None:
    from agent_eval_harness.injection.agent_invokers import invoke_synthesizer

    prov = _mock_provider({"executive_summary": "ok", "architecture_narrative": "n"})
    case_input = {
        "shape": "all_sections",
        "all_sections": {L: {"confidence": "high", "blind_spots": []} for L in "ABCDEFGHIJK"},
    }
    out = await invoke_synthesizer(case_input, prov, "p", "m")

    assert prov.chat.await_count == 1
    assert isinstance(out, dict) and out  # produced a section-L dict


def _bundle(n: int = 2) -> dict:
    return {
        "evidences": [
            {"rel_path": f"src/file{i}.py", "excerpt": f"def f{i}(): pass", "score": 0.9 - i * 0.1}
            for i in range(n)
        ]
    }


async def test_invoke_glossary_runs_real_agent_once() -> None:
    from agent_eval_harness.injection.agent_invokers import invoke_glossary

    prov = _mock_provider({"terms": [{"term": "Foo", "definition": "d", "evidence_files": []}], "confidence": "high"})
    out = await invoke_glossary({"bundle": _bundle()}, prov, "p", "m")

    assert prov.chat.await_count == 1
    assert out["terms"][0]["term"] == "Foo"


async def test_invoke_important_files_runs_real_agent_once() -> None:
    from agent_eval_harness.injection.agent_invokers import invoke_important_files

    prov = _mock_provider({"entrypoint": {"file": "main.py", "reason": "r"}, "confidence": "high"})
    out = await invoke_important_files({"bundle": _bundle()}, prov, "p", "m")

    assert prov.chat.await_count == 1
    assert out["entrypoint"]["file"] == "main.py"


async def test_invoke_project_identity_uses_mem_ctx_and_runs_once() -> None:
    from agent_eval_harness.injection.agent_invokers import invoke_project_identity

    prov = _mock_provider({"repo_name": "r", "domain": "web", "confidence": "high"})
    case_input = {
        "bundle": _bundle(), "folder_tree": "src/\n  foo.py\n", "doc_ctx": "readme text",
        "manifest_ctx": "deps text", "repo_name": "myrepo",
    }
    out = await invoke_project_identity(case_input, prov, "p", "m")

    assert prov.chat.await_count == 1
    assert out["domain"] == "web"


async def test_invoke_violations_passes_upstream_conventions_output() -> None:
    from agent_eval_harness.injection.agent_invokers import invoke_violations

    prov = _mock_provider({"rules": [], "violations_found": [], "confidence": "high"})
    case_input = {
        "bundle": _bundle(),
        "conventions_output": {"signals": [{"category": "naming", "pattern": "snake_case"}]},
    }
    out = await invoke_violations(case_input, prov, "p", "m")

    assert prov.chat.await_count == 1
    assert out["rules"] == []


async def test_invoke_onboarding_passes_upstream_important_files_output() -> None:
    from agent_eval_harness.injection.agent_invokers import invoke_onboarding

    prov = _mock_provider({"steps": [], "total_estimated_minutes": 20, "confidence": "high"})
    case_input = {
        "bundle": _bundle(),
        "important_files_output": {"entrypoint": {"file": "main.py", "reason": "r"}},
    }
    out = await invoke_onboarding(case_input, prov, "p", "m")

    assert prov.chat.await_count == 1
    assert out["total_estimated_minutes"] == 20


async def test_invoke_architecture_inherits_bundle_and_runs_once() -> None:
    from agent_eval_harness.injection.agent_invokers import invoke_architecture

    prov = _mock_provider({"main_layers": ["domain"], "confidence": "high"})
    case_input = {"bundle": _bundle(), "identity_output": {"domain": "web"}, "folder_tree": "src/\n"}
    out = await invoke_architecture(case_input, prov, "p", "m")

    assert prov.chat.await_count == 1
    assert out["main_layers"] == ["domain"]


async def test_invoke_structure_own_retrieval_path_when_not_inherited() -> None:
    from agent_eval_harness.injection.agent_invokers import invoke_structure

    prov = _mock_provider({"folders": [], "summary": "s", "confidence": "high"})
    case_input = {"bundle": _bundle(), "inherit_bundle": False, "folder_tree": "src/\n"}
    out = await invoke_structure(case_input, prov, "p", "m")

    assert prov.chat.await_count == 1
    assert out["summary"] == "s"


async def test_invoke_conventions_serves_two_call_sequence() -> None:
    from agent_eval_harness.injection.agent_invokers import invoke_conventions

    prov = _mock_provider_sequence(
        ['["naming convention", "error handling"]',
         json.dumps({"naming_style": {"description": "x", "evidence_files": []}, "confidence": "high"})]
    )
    case_input = {"bundle": _bundle(), "structure_output": {"folders": []}}
    out = await invoke_conventions(case_input, prov, "p", "m")

    assert prov.chat.await_count == 2  # plan-step + main call
    assert out["naming_style"]["description"] == "x"


async def test_invoke_risk_serves_two_call_sequence() -> None:
    from agent_eval_harness.injection.agent_invokers import invoke_risk

    prov = _mock_provider_sequence(
        ['["risk hotspots"]', json.dumps({"findings": [], "summary": "s", "confidence": "high"})]
    )
    out = await invoke_risk({"bundle": _bundle()}, prov, "p", "m")

    assert prov.chat.await_count == 2
    assert out["summary"] == "s"


async def test_invoke_feature_map_serves_two_call_sequence() -> None:
    from agent_eval_harness.injection.agent_invokers import invoke_feature_map

    prov = _mock_provider_sequence(
        ['["routes"]', json.dumps({"features": [], "confidence": "high"})]
    )
    case_input = {
        "bundle": _bundle(), "identity_output": {"domain": "web"},
        "architecture_output": {"main_layers": []}, "folder_tree": "src/\n",
    }
    out = await invoke_feature_map(case_input, prov, "p", "m")

    assert prov.chat.await_count == 2
    assert out["features"] == []


async def test_invoke_agent_dispatch_unknown_raises() -> None:
    from agent_eval_harness.injection.agent_invokers import invoke_agent

    with pytest.raises(NotImplementedError):
        await invoke_agent("not_a_real_agent", {}, None, "p", "m")
