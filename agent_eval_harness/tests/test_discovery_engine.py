"""CS-270 — Discovery engine golden + negative-control tests."""
from __future__ import annotations

import pytest

from agent_eval_harness.discovery.engine import discover_agentic_systems
from agent_eval_harness.llm.client import LLMResponse
from agent_eval_harness.llm.fake_client import FakeLLMClient

HAYSTACK_EXCERPT = (
    "from haystack import Pipeline\n"
    "from haystack.components.builders import PromptBuilder\n\n"
    "pipeline = Pipeline()\n"
    "pipeline.add_component('prompt_builder', PromptBuilder())\n"
    "pipeline.connect('prompt_builder.prompt', 'llm.prompt')"
)
CREWAI_EXCERPT = (
    "from crewai import Agent, Task, Crew\n\n"
    "def build_crew():\n"
    "    return Crew(agents=[])"
)
NON_AGENTIC_EXCERPT = (
    "def add(a, b):\n"
    "    return a + b\n\n"
    "def multiply(a, b):\n"
    "    return a * b"
)


class _StubClient:
    """Duck-typed stand-in for CodeSpectraClient — same method names, no HTTP."""

    def __init__(self, evidences: list[dict], node_index: dict[str, int], communities: list[dict]):
        self._evidences = evidences
        self._node_index = node_index
        self._communities = communities

    async def search_repo_map(self, snapshot_id: str, query: str, limit: int = 120) -> dict:
        return {"symbols": []}

    async def search_retrieval(self, snapshot_id: str, query: str, section: str = "qa") -> dict:
        # "fused" matches retrieve_rrf_fusion's response shape (not the plain-retrieve one)
        return {"fused": self._evidences}

    async def get_communities(self, snapshot_id: str) -> dict:
        return {"node_index": self._node_index, "communities": self._communities}

    async def get_snapshot(self, snapshot_id: str) -> dict:
        # No CA-mode sibling in these tests — engine must degrade gracefully
        raise RuntimeError("no sibling snapshot in test fixture")

    async def read_file(self, snapshot_id: str, path: str) -> dict:
        for ev in self._evidences:
            if ev["rel_path"] == path:
                return {"content": ev["excerpt"]}
        return {"content": ""}

    async def aclose(self) -> None:
        pass


def _golden_fixture_client() -> _StubClient:
    evidences = [
        {"rel_path": "agents/haystack_pipeline.py", "excerpt": HAYSTACK_EXCERPT},
        {"rel_path": "agents/crew_setup.py", "excerpt": CREWAI_EXCERPT},
        {"rel_path": "utils/math_helpers.py", "excerpt": NON_AGENTIC_EXCERPT},
    ]
    node_index = {
        "agents/haystack_pipeline.py": 1,
        "agents/crew_setup.py": 2,
        "utils/math_helpers.py": 3,
    }
    communities = [
        {"community_id": 1, "hub_paths": ["agents/haystack_pipeline.py"]},
        {"community_id": 2, "hub_paths": ["agents/crew_setup.py"]},
        {"community_id": 3, "hub_paths": ["utils/math_helpers.py"]},
    ]
    return _StubClient(evidences, node_index, communities)


@pytest.mark.asyncio
async def test_golden_two_real_agentic_clusters_plus_non_agentic_package() -> None:
    client = _golden_fixture_client()
    llm_client = FakeLLMClient([
        LLMResponse(
            content='{"is_agentic_system": true, "name": "Haystack Pipeline", '
            '"frameworks": ["haystack"], "entry_points": ["agents/haystack_pipeline.py"], '
            '"confidence": "high"}',
            model="fake",
        ),
        LLMResponse(
            content='{"is_agentic_system": true, "name": "Crew Setup", '
            '"frameworks": ["crewai"], "entry_points": ["agents/crew_setup.py"], '
            '"confidence": "high"}',
            model="fake",
        ),
    ])

    candidates = await discover_agentic_systems("snap-golden", "repo-golden", client, llm_client)

    assert len(candidates) == 2, (
        f"Expected exactly 2 candidates (Haystack + CrewAI clusters), got {len(candidates)}: "
        f"{[c['name'] for c in candidates]}"
    )
    names = {c["name"] for c in candidates}
    assert names == {"Haystack Pipeline", "Crew Setup"}

    # Correctly localized — the non-agentic file must not appear as evidence for either candidate
    all_evidence_files = {
        ev["file"] for c in candidates for ev in c["evidence"]
    }
    assert "utils/math_helpers.py" not in all_evidence_files
    assert "agents/haystack_pipeline.py" in all_evidence_files
    assert "agents/crew_setup.py" in all_evidence_files

    # Assert community_id, cluster_files, and hub_paths are present and correct
    by_name = {c["name"]: c for c in candidates}

    haystack_cand = by_name["Haystack Pipeline"]
    assert haystack_cand["community_id"] == "1"
    assert haystack_cand["cluster_files"] == ["agents/haystack_pipeline.py"]
    assert haystack_cand["hub_paths"] == ["agents/haystack_pipeline.py"]
    for ev in haystack_cand["evidence"]:
        assert "token_estimate" in ev
        assert ev["token_estimate"] > 0
    assert haystack_cand["wiring_block"] is not None
    assert haystack_cand["wiring_block"]["framework"] == "haystack"
    assert len(haystack_cand["wiring_block"]["nodes"]) == 1
    assert haystack_cand["wiring_block"]["nodes"][0]["alias"] == "prompt_builder"

    crew_cand = by_name["Crew Setup"]
    assert crew_cand["wiring_block"] is None
    assert crew_cand["community_id"] == "2"
    assert crew_cand["cluster_files"] == ["agents/crew_setup.py"]
    assert crew_cand["hub_paths"] == ["agents/crew_setup.py"]
    for ev in crew_cand["evidence"]:
        assert "token_estimate" in ev
        assert ev["token_estimate"] > 0


@pytest.mark.asyncio
async def test_negative_control_non_agentic_only_yields_zero_candidates() -> None:
    """A repo with no framework/import fingerprint hits at all must never hallucinate a candidate."""
    client = _StubClient(
        evidences=[{"rel_path": "utils/math_helpers.py", "excerpt": NON_AGENTIC_EXCERPT}],
        node_index={"utils/math_helpers.py": 1},
        communities=[{"community_id": 1, "hub_paths": ["utils/math_helpers.py"]}],
    )
    # No LLM calls should even happen (Pass A finds zero hits, discovery ends before Pass C)
    llm_client = FakeLLMClient(LLMResponse(content="not json — should never be called", model="fake"))

    candidates = await discover_agentic_systems("snap-negative", "repo-negative", client, llm_client)

    assert candidates == [], f"Expected zero candidates for non-agentic-only repo, got: {candidates}"
    assert llm_client.calls == [], "Pass C must not run an LLM call when Pass A found zero hits"


@pytest.mark.asyncio
async def test_llm_synthesis_budget_cap_marks_overflow_needs_human() -> None:
    """More clusters than MAX_LLM_SYNTHESIZED_CLUSTERS must still all be surfaced, never dropped."""
    from agent_eval_harness.discovery import engine as engine_module

    n_clusters = engine_module.MAX_LLM_SYNTHESIZED_CLUSTERS + 3
    evidences = [
        {"rel_path": f"agents/haystack_{i}.py", "excerpt": HAYSTACK_EXCERPT}
        for i in range(n_clusters)
    ]
    node_index = {f"agents/haystack_{i}.py": i for i in range(n_clusters)}
    communities = [{"community_id": i, "hub_paths": [f"agents/haystack_{i}.py"]} for i in range(n_clusters)]
    client = _StubClient(evidences, node_index, communities)

    llm_response = LLMResponse(
        content='{"is_agentic_system": true, "name": "Haystack Pipeline", '
        '"frameworks": ["haystack"], "entry_points": [], "confidence": "high"}',
        model="fake",
    )
    llm_client = FakeLLMClient(llm_response)

    candidates = await discover_agentic_systems("snap-budget", "repo-budget", client, llm_client)

    assert len(candidates) == n_clusters, "every cluster must still be surfaced, never dropped"
    needs_human_count = sum(1 for c in candidates if c.get("needs_human"))
    assert needs_human_count == 3, (
        f"expected exactly 3 clusters beyond the LLM budget to be marked needs_human, "
        f"got {needs_human_count}"
    )
    assert len(llm_client.calls) == engine_module.MAX_LLM_SYNTHESIZED_CLUSTERS, (
        "LLM must be called at most MAX_LLM_SYNTHESIZED_CLUSTERS times, never once per cluster"
    )
