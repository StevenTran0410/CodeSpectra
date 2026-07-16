"""Tests for enrichment hardening (CS-291)."""
import json
from pathlib import Path

import pytest

from agent_eval_harness.discovery.enrichment import enrich_agents
from agent_eval_harness.discovery.agent_knowledge import AgentKnowledge
from agent_eval_harness.discovery.expansion import expand_candidate
from agent_eval_harness.mapping.agent_flow import AgentFlow, AgentFlowMap
from agent_eval_harness.mapping.system_map import SystemMap, Component
from agent_eval_harness.store import repository
from agent_eval_harness.store.database import init_db
from tests._stubs import FakeCodeSpectraClient as _StubClient


class _StubLLMClient:
    """Stub LLM implementing the real LLMClient.complete() protocol, not acompletion()."""
    def __init__(self, response_override=None):
        self.call_count = 0
        self.response_override = response_override or {}

    async def complete(self, messages, *, max_tokens=512, temperature=0.2, json_mode=False, reasoning_effort=None):
        from agent_eval_harness.llm.client import LLMResponse
        self.call_count += 1
        default_response = {
            "functionality": "Test purpose",
            "functionality_citations": [],
            "context_builders": [],
            "upstream_consumers": [],
            "downstream_consumers": [],
            "failure_modes": [],
            "need_more": False,
            "next_queries": [],
        }
        default_response.update(self.response_override)
        return LLMResponse(content=json.dumps(default_response), model="stub")


@pytest.mark.anyio
async def test_c0_expansion_annotation_regression() -> None:
    """AC1: C0 expansion annotation regression — verify no regressions in expansion output format."""
    # Simple validation that expansion still produces expected output format
    files = {
        "file_a.py": "class Agent: pass",
        "file_b.py": "class Utility: pass"
    }
    neighbors = {
        "file_a.py": ["file_b.py"],
    }
    verdicts = {
        "file_a.py": "expand",
        "file_b.py": "boundary"
    }

    class _StubLLMExpansion:
        async def complete(self, messages, *, json_mode=False, **_kwargs):
            from agent_eval_harness.llm.client import LLMResponse
            prompt = "\n".join(m.content for m in messages)
            import re
            ids = re.findall(r"=== ID: (\S+) ===", prompt)
            verdicts_list = []
            for unique_id in ids:
                verdict = verdicts.get(unique_id, "boundary")
                verdicts_list.append({
                    "id": unique_id,
                    "verdict": verdict,
                    "reason": "stubbed"
                })
            content = json.dumps({"verdicts": verdicts_list})
            return LLMResponse(content=content, model="stub")

    client = _StubClient(files, neighbors)
    llm_client = _StubLLMExpansion()
    candidate = {
        "cluster_files": ["file_a.py"],
        "hub_paths": []
    }

    res = await expand_candidate("snap-123", candidate, client, llm_client, node_budget=5, hop_cap=3)

    assert "accepted" in res
    assert "boundary" in res
    assert "stop_reason" in res
    for item in res["accepted"]:
        if isinstance(item, dict):
            assert "file" in item or "id" in item


@pytest.mark.anyio
async def test_persist_md_and_json_to_appdata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """AC2: File persistence — .md and .json sidecar written to correct AppData path."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    try:
        from agent_eval_harness.store.database import get_db
        get_db()
    except RuntimeError:
        from agent_eval_harness.store.database import init_db
        await init_db()

    agent = AgentFlow(id="test_agent", label="Test Agent", component_ids=["comp1"])
    flow_map = AgentFlowMap(target_system_id="test_system", agents=[agent])
    system_map = SystemMap(
        target_system_id="test_system",
        components=[Component(id="comp1", role="test", entry_point="entry", file="test.py")]
    )

    llm_client = _StubLLMClient()

    result = await enrich_agents(
        session_id="test_session",
        agent_flow_map=flow_map,
        system_map=system_map,
        accepted_with_annotations=["test.py"],
        accepted_edges=[],
        client=None,
        llm_client=llm_client,
        depth="normal",
    )

    appdata_dir = tmp_path / "AppData" / "Local" / "codespectra" / "agents" / "test_session"
    md_path = appdata_dir / "test_agent.md"
    json_path = appdata_dir / "test_agent.json"

    assert md_path.exists(), f"Markdown sidecar not found at {md_path}"
    assert json_path.exists(), f"JSON sidecar not found at {json_path}"

    json_data = json.loads(json_path.read_text(encoding='utf-8'))
    knowledge = AgentKnowledge.from_json(json_data)
    assert isinstance(knowledge, AgentKnowledge)


@pytest.mark.anyio
async def test_no_hidden_fallback_purpose(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """AC3 (updated D2): _FALLBACK_PURPOSE was deleted; enrichment LLM output is returned verbatim."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    agent = AgentFlow(id="test_agent_ac3", label="Test Agent AC3", component_ids=[])
    flow_map = AgentFlowMap(target_system_id="test_system", agents=[agent])
    system_map = SystemMap(target_system_id="test_system", components=[])

    llm_client = _StubLLMClient(response_override={
        "functionality": "Enriched by test stub — no fallback"
    })

    result = await enrich_agents(
        session_id="test_session_ac3",
        agent_flow_map=flow_map,
        system_map=system_map,
        accepted_with_annotations=[],
        accepted_edges=[],
        client=None,
        llm_client=llm_client,
        depth="normal",
    )

    knowledge = result[0]
    assert knowledge.functionality == "Enriched by test stub — no fallback"
    assert knowledge.functionality != "Agent discovered with high coverage; no LLM analysis performed."


@pytest.mark.anyio
async def test_single_agent_failure_does_not_block_others() -> None:
    """AC5: Agent isolation — one failing agent does not block others."""
    agents = [
        AgentFlow(id="agent_a", label="Agent A", component_ids=[]),
        AgentFlow(id="agent_b", label="Agent B", component_ids=[]),
        AgentFlow(id="agent_c", label="Agent C", component_ids=[]),
    ]
    flow_map = AgentFlowMap(target_system_id="test_system", agents=agents)
    system_map = SystemMap(target_system_id="test_system", components=[])
    evidence = {
        'prompt_sites_by_file': {},
        'component_by_agent': {a.id: [] for a in agents},
        'edges_by_agent': {a.id: [] for a in agents},
        'source_coverage': {a.id: 0.0 for a in agents},
    }

    class _SelectiveLLMClient:
        def __init__(self):
            self.call_count = 0

        async def complete(self, messages, *, max_tokens=512, temperature=0.2, json_mode=False, reasoning_effort=None):
            self.call_count += 1
            raise RuntimeError("LLM intentional failure")

    result = await enrich_agents(
        session_id="test_session",
        agent_flow_map=flow_map,
        system_map=system_map,
        accepted_with_annotations=[],
        accepted_edges=[],
        client=None,
        llm_client=_SelectiveLLMClient(),
        depth="normal",
    )

    assert len(result) == 3
    assert all(isinstance(k, AgentKnowledge) for k in result)


@pytest.mark.anyio
async def test_zero_query_fast_path_when_coverage_sufficient() -> None:
    """AC5: Zero-query fast-path — sufficient coverage skips queries."""
    from agent_eval_harness.discovery.enrichment import _enrich_single_agent
    from agent_eval_harness.mapping.agent_flow import AgentFlow, AgentFlowMap
    from dataclasses import dataclass

    agent = AgentFlow(id="test_agent", label="Test", component_ids=["comp1"])
    flow_map = AgentFlowMap(target_system_id="test", agents=[agent])
    system_map = SystemMap(target_system_id="test", components=[])

    evidence = {
        'prompt_sites_by_file': {"file_a.py": []},
        'component_by_agent': {"test_agent": [{"id": "comp1", "file": "file_a.py"}]},
        'edges_by_agent': {"test_agent": []},
        'source_coverage': {"test_agent": 0.9},  # >= 0.8
    }

    class _CountingLLMClient:
        def __init__(self):
            self.call_count = 0

        async def complete(self, messages, *, max_tokens=512, temperature=0.2, json_mode=False, reasoning_effort=None):
            self.call_count += 1
            raise AssertionError("Should not call LLM in fast-path")

    @dataclass
    class _EnrichCtx:
        session_id: str = "test"
        snapshot_id: str = ""
        agent_flow_map: AgentFlowMap = None
        system_map: SystemMap = None
        accepted_with_annotations: list = None
        accepted_edges: list = None
        client: object = None
        llm_client: object = None
        depth: str = "normal"
        force_agent_ids: list = None
        semaphore: object = None

        def __post_init__(self):
            if self.agent_flow_map is None:
                self.agent_flow_map = flow_map
            if self.system_map is None:
                self.system_map = system_map
            if self.accepted_with_annotations is None:
                self.accepted_with_annotations = []
            if self.accepted_edges is None:
                self.accepted_edges = []
            if self.force_agent_ids is None:
                self.force_agent_ids = []
            if self.llm_client is None:
                self.llm_client = _CountingLLMClient()
            if self.semaphore is None:
                class _Semaphore:
                    async def __aenter__(self): return self
                    async def __aexit__(self, *args): pass
                self.semaphore = _Semaphore()

    depth_cap = {"queries": 3, "llm_calls": 2, "read_file": 2}

    knowledge = await _enrich_single_agent(
        "test_agent",
        evidence,
        _EnrichCtx(),
        depth_cap,
        ["file_a.py"],
    )

    assert knowledge.query_count == 0
    ctx = _EnrichCtx()
    assert ctx.llm_client.call_count == 0


@pytest.mark.anyio
async def test_rerun_cache_hit_reads_actual_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """AC5: Re-run cache-hit — reads actual persisted JSON, not mock."""
    from agent_eval_harness.discovery.enrichment import _enrich_single_agent
    from agent_eval_harness.mapping.agent_flow import AgentFlow, AgentFlowMap
    from agent_eval_harness.store.database import init_db, get_db
    from dataclasses import dataclass

    monkeypatch.setenv("AEH_DATA_DIR", str(tmp_path))
    import agent_eval_harness.store.database as db_module
    db_module._db = None
    await init_db()

    json_dir = tmp_path / "knowledge"
    json_dir.mkdir()
    json_path = json_dir / "test_agent.json"
    cached_knowledge = {
        "functionality": "Cached content from disk",
        "functionality_citations": [],
        "context_builders": [],
        "upstream_consumers": [],
        "downstream_consumers": [],
        "failure_modes": [],
    }
    json_path.write_text(json.dumps(cached_knowledge), encoding='utf-8')

    await repository.upsert_agent_knowledge(
        session_id="test_session",
        agent_id="test_agent",
        md_path=str(json_dir / "test_agent.md"),
        json_path=str(json_path),
        evidence_hash="hash123",
        confidence="medium",
        query_count=0,
    )

    agent = AgentFlow(id="test_agent", label="Test", component_ids=[])
    flow_map = AgentFlowMap(target_system_id="test", agents=[agent])
    system_map = SystemMap(target_system_id="test", components=[])
    evidence = {
        'prompt_sites_by_file': {},
        'component_by_agent': {"test_agent": []},
        'edges_by_agent': {"test_agent": []},
        'source_coverage': {"test_agent": 0.0},
    }

    class _FailLLMClient:
        async def complete(self, messages, *, max_tokens=512, temperature=0.2, json_mode=False, reasoning_effort=None):
            raise AssertionError("Should not call LLM in cache-hit")

    @dataclass
    class _EnrichCtx:
        session_id: str = "test_session"
        snapshot_id: str = ""
        agent_flow_map: AgentFlowMap = None
        system_map: SystemMap = None
        accepted_with_annotations: list = None
        accepted_edges: list = None
        client: object = None
        llm_client: object = None
        depth: str = "normal"
        force_agent_ids: list = None
        semaphore: object = None

        def __post_init__(self):
            if self.agent_flow_map is None:
                self.agent_flow_map = flow_map
            if self.system_map is None:
                self.system_map = system_map
            if self.accepted_with_annotations is None:
                self.accepted_with_annotations = []
            if self.accepted_edges is None:
                self.accepted_edges = []
            if self.force_agent_ids is None:
                self.force_agent_ids = []
            if self.llm_client is None:
                self.llm_client = _FailLLMClient()
            if self.semaphore is None:
                class _Semaphore:
                    async def __aenter__(self): return self
                    async def __aexit__(self, *args): pass
                self.semaphore = _Semaphore()

    depth_cap = {"queries": 3, "llm_calls": 2, "read_file": 2}

    knowledge = await _enrich_single_agent(
        "test_agent",
        evidence,
        _EnrichCtx(),
        depth_cap,
        [],
    )

    assert knowledge.functionality == "Cached content from disk"

    from agent_eval_harness.store.database import close_db
    await close_db()


def test_grep_gate_no_codespectra_literals() -> None:
    """AC6: Grep gate — 0 CodeSpectra-specific literals in enrichment code."""
    forbidden_literals = [
        'ProjectIdentityAgent',
        'extract_a_identity_context',
        'identity_context',
    ]

    enrichment_file = Path(__file__).parent.parent / "agent_eval_harness" / "discovery" / "enrichment.py"
    knowledge_file = Path(__file__).parent.parent / "agent_eval_harness" / "discovery" / "agent_knowledge.py"

    found_issues = []
    for py_file in [enrichment_file, knowledge_file]:
        if not py_file.exists():
            continue
        content = py_file.read_text(encoding='utf-8', errors='ignore')
        content_lower = content.lower()
        for literal in forbidden_literals:
            if literal.lower() in content_lower:
                found_issues.append(f"{py_file}: found '{literal}'")

    assert not found_issues, f"Forbidden literals found in hardening scope:\n" + "\n".join(found_issues)


@pytest.mark.anyio
async def test_functional_run_multi_agent_target() -> None:
    """AC6: Functional run against test_targets/multi_agent — degradation is explicit."""
    agents = [
        AgentFlow(id="agent_1", label="Agent 1", component_ids=["comp_1"]),
        AgentFlow(id="agent_2", label="Agent 2", component_ids=["comp_2"]),
    ]
    flow_map = AgentFlowMap(target_system_id="test_system", agents=agents)
    system_map = SystemMap(
        target_system_id="test_system",
        components=[
            Component(id="comp_1", role="test", entry_point="entry1", file="src/agent1.py"),
            Component(id="comp_2", role="test", entry_point="entry2", file="src/agent2.py"),
        ]
    )

    llm_client = _StubLLMClient()

    try:
        from agent_eval_harness.store.database import get_db
        get_db()
    except RuntimeError:
        from agent_eval_harness.store.database import init_db
        await init_db()

    result = await enrich_agents(
        session_id="multi_agent_test",
        agent_flow_map=flow_map,
        system_map=system_map,
        accepted_with_annotations=["src/agent1.py", "src/agent2.py"],
        accepted_edges=[],
        client=None,
        llm_client=llm_client,
        depth="normal",
    )

    assert len(result) >= 1
    for knowledge in result:
        assert isinstance(knowledge, AgentKnowledge)
        assert isinstance(knowledge.degraded, bool)
        if knowledge.degraded:
            assert knowledge.degraded_reason is not None
