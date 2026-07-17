"""Per-agent invokers: build the real agent with mocked collaborators and call its real run(); _SNAPSHOT_ID is an opaque placeholder since every collaborator that would resolve it is mocked."""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from agent_eval_harness.injection.backend_bridge import ensure_backend_importable
from agent_eval_harness.injection.collaborators import (
    build_retrieval_bundle,
    make_mock_retrieval,
    make_pipeline_memory_context,
    neutral_profile,
    patched_folder_summary,
)

ensure_backend_importable()

from domain.analysis.agents.agent_architecture import ArchitectureAgent  # noqa: E402
from domain.analysis.agents.agent_auditor import AuditAgent  # noqa: E402
from domain.analysis.agents.agent_conventions import ConventionsAgent  # noqa: E402
from domain.analysis.agents.agent_feature_map import FeatureMapAgent  # noqa: E402
from domain.analysis.agents.agent_glossary import GlossaryAgent  # noqa: E402
from domain.analysis.agents.agent_important_files import ImportantFilesAgent  # noqa: E402
from domain.analysis.agents.agent_onboarding import OnboardingAgent  # noqa: E402
from domain.analysis.agents.agent_project_identity import ProjectIdentityAgent  # noqa: E402
from domain.analysis.agents.agent_risk import RiskAgent  # noqa: E402
from domain.analysis.agents.agent_structure import StructureAgent  # noqa: E402
from domain.analysis.agents.agent_synthesis import SynthesisAgent  # noqa: E402
from domain.analysis.agents.agent_violations import ViolationsAgent  # noqa: E402

_SNAPSHOT_ID = "synthetic-snapshot"


async def invoke_auditor(
    case_input: dict[str, Any], provider_service: Any, provider_id: str, model_id: str
) -> dict[str, Any]:
    """K — no retrieval, no mem_ctx."""
    all_sections = case_input["all_sections"]
    agent = AuditAgent(provider_service)
    return await agent.run(provider_id, model_id, all_sections, profile=neutral_profile())


async def invoke_synthesizer(
    case_input: dict[str, Any], provider_service: Any, provider_id: str, model_id: str
) -> dict[str, Any]:
    """L — no retrieval, no mem_ctx."""
    all_sections = case_input["all_sections"]
    agent = SynthesisAgent(provider_service)
    return await agent.run(provider_id, model_id, all_sections, profile=neutral_profile())


async def invoke_glossary(
    case_input: dict[str, Any], provider_service: Any, provider_id: str, model_id: str
) -> dict[str, Any]:
    """I — one fixed-query retrieval call, no upstream, no mem_ctx."""
    bundle = build_retrieval_bundle(case_input["bundle"])
    agent = GlossaryAgent(provider_service, make_mock_retrieval(bundle))
    return await agent.run(provider_id, model_id, _SNAPSHOT_ID, profile=neutral_profile())


async def invoke_important_files(
    case_input: dict[str, Any], provider_service: Any, provider_id: str, model_id: str
) -> dict[str, Any]:
    """G — one fixed-query retrieval call; graph_summary is left unset (not yet synthesized)."""
    bundle = build_retrieval_bundle(case_input["bundle"])
    agent = ImportantFilesAgent(provider_service, make_mock_retrieval(bundle))
    return await agent.run(
        provider_id, model_id, _SNAPSHOT_ID, graph_summary=None, profile=neutral_profile()
    )


async def invoke_project_identity(
    case_input: dict[str, Any], provider_service: Any, provider_id: str, model_id: str
) -> dict[str, Any]:
    """A — always runs its own retrieval regardless of mem_ctx (mem_ctx only supplies doc/manifest/folder_tree strings, not the bundle)."""
    bundle = build_retrieval_bundle(case_input["bundle"])
    mem_ctx = make_pipeline_memory_context(
        folder_tree=case_input.get("folder_tree", ""),
        doc_files=case_input.get("doc_ctx", ""),
        manifest_files=case_input.get("manifest_ctx", ""),
    )
    agent = ProjectIdentityAgent(provider_service, make_mock_retrieval(bundle))
    async with patched_folder_summary(case_input.get("folder_tree", "")):
        return await agent.run(
            provider_id, model_id, _SNAPSHOT_ID,
            repo_name=case_input.get("repo_name", ""), mem_ctx=mem_ctx, profile=neutral_profile(),
        )


async def invoke_violations(
    case_input: dict[str, Any], provider_service: Any, provider_id: str, model_id: str
) -> dict[str, Any]:
    """E — one retrieval call plus upstream D's output; static_convention/static_risk are left unset since the agent handles None for both."""
    bundle = build_retrieval_bundle(case_input["bundle"])
    agent = ViolationsAgent(provider_service, make_mock_retrieval(bundle))
    return await agent.run(
        provider_id, model_id, _SNAPSHOT_ID,
        conventions_output=case_input.get("conventions_output"), profile=neutral_profile(),
    )


async def invoke_onboarding(
    case_input: dict[str, Any], provider_service: Any, provider_id: str, model_id: str
) -> dict[str, Any]:
    """H — one fixed-query retrieval call plus upstream G's output."""
    bundle = build_retrieval_bundle(case_input["bundle"])
    agent = OnboardingAgent(provider_service, make_mock_retrieval(bundle))
    return await agent.run(
        provider_id, model_id, _SNAPSHOT_ID,
        important_files_output=case_input.get("important_files_output"), profile=neutral_profile(),
    )


async def invoke_architecture(
    case_input: dict[str, Any], provider_service: Any, provider_id: str, model_id: str
) -> dict[str, Any]:
    """B — fires 3 parallel sub-queries regardless of arch_bundle; a single-bundle mock keeps content deterministic without depending on asyncio.gather's call order."""
    bundle = build_retrieval_bundle(case_input["bundle"])
    agent = ArchitectureAgent(provider_service, make_mock_retrieval(bundle))
    return await agent.run(
        provider_id, model_id, _SNAPSHOT_ID,
        arch_bundle=(bundle if case_input.get("inherit_bundle", True) else None),
        identity_output=case_input.get("identity_output"),
        folder_tree=case_input.get("folder_tree", ""),
        profile=neutral_profile(),
    )


async def invoke_structure(
    case_input: dict[str, Any], provider_service: Any, provider_id: str, model_id: str
) -> dict[str, Any]:
    """C — an inherited arch_bundle skips retrieval entirely; otherwise exactly one own retrieval call."""
    bundle = build_retrieval_bundle(case_input["bundle"])
    agent = StructureAgent(provider_service, make_mock_retrieval(bundle))
    async with patched_folder_summary(case_input.get("folder_tree", "")):
        return await agent.run(
            provider_id, model_id, _SNAPSHOT_ID,
            arch_bundle=(bundle if case_input.get("inherit_bundle", True) else None),
            folder_tree=case_input.get("folder_tree", ""),
            identity_output=case_input.get("identity_output"),
            profile=neutral_profile(),
        )


async def invoke_conventions(
    case_input: dict[str, Any], provider_service: Any, provider_id: str, model_id: str
) -> dict[str, Any]:
    """D — query-planning sub-call then per-sub-query retrieves; provider_service must serve at least 2 chat responses in order (plan-step, main call)."""
    bundle = build_retrieval_bundle(case_input["bundle"])
    agent = ConventionsAgent(provider_service, make_mock_retrieval(bundle))
    return await agent.run(
        provider_id, model_id, _SNAPSHOT_ID,
        structure_output=case_input.get("structure_output"), profile=neutral_profile(),
    )


async def invoke_risk(
    case_input: dict[str, Any], provider_service: Any, provider_id: str, model_id: str
) -> dict[str, Any]:
    """J — same shape as D; static_risk is left unset."""
    bundle = build_retrieval_bundle(case_input["bundle"])
    agent = RiskAgent(provider_service, make_mock_retrieval(bundle))
    return await agent.run(provider_id, model_id, _SNAPSHOT_ID, profile=neutral_profile())


async def invoke_feature_map(
    case_input: dict[str, Any], provider_service: Any, provider_id: str, model_id: str
) -> dict[str, Any]:
    """F — query-planning + retrieve_multi run concurrently with one more retrieve() via asyncio.gather; a single-bundle mock sidesteps their call-interleaving order."""
    bundle = build_retrieval_bundle(case_input["bundle"])
    agent = FeatureMapAgent(provider_service, make_mock_retrieval(bundle))
    return await agent.run(
        provider_id, model_id, _SNAPSHOT_ID,
        identity_output=case_input.get("identity_output"),
        architecture_output=case_input.get("architecture_output"),
        folder_tree=case_input.get("folder_tree", ""),
        profile=neutral_profile(),
    )


_INVOKERS: dict[str, Callable[..., Awaitable[dict[str, Any]]]] = {
    "auditor": invoke_auditor,
    "synthesizer": invoke_synthesizer,
    "glossary": invoke_glossary,
    "important_files": invoke_important_files,
    "project_identity": invoke_project_identity,
    "violations": invoke_violations,
    "onboarding": invoke_onboarding,
    "architecture": invoke_architecture,
    "structure": invoke_structure,
    "conventions": invoke_conventions,
    "risk": invoke_risk,
    "feature_map": invoke_feature_map,
}


async def invoke_agent(
    agent_id: str, case_input: dict[str, Any], provider_service: Any, provider_id: str, model_id: str
) -> dict[str, Any]:
    invoker = _INVOKERS.get(agent_id)
    if invoker is None:
        raise NotImplementedError(f"no injection invoker for agent '{agent_id}' yet")
    return await invoker(case_input, provider_service, provider_id, model_id)
