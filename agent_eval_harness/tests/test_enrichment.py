"""Tests for enrichment hardening."""
import json
from pathlib import Path

import pytest

from agent_eval_harness.discovery.enrichment import enrich_agents
from agent_eval_harness.discovery.agent_knowledge import AgentKnowledge
from agent_eval_harness.discovery.expansion import expand_candidate
from agent_eval_harness.mapping.agent_flow import AgentFlow, AgentFlowMap
from agent_eval_harness.mapping.system_map import SystemMap, Component
from agent_eval_harness.store import repository
from agent_eval_harness.store.database import get_db, init_db
from tests._stubs import FakeCodeSpectraClient as _StubClient

_DEPTH_CAP = {"queries": 3, "llm_calls": 2, "read_file": 2}


@pytest.fixture(autouse=True)
async def _ensure_db() -> None:
    """Defensive re-init: an earlier test file may have closed the DB without reopening it."""
    try:
        get_db()
    except RuntimeError:
        await init_db()


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
    """File persistence: .md/.json sidecars are written under AEH_DATA_DIR (Roaming), not AppData/Local."""
    monkeypatch.setenv("AEH_DATA_DIR", str(tmp_path))

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

    appdata_dir = tmp_path / "agents" / "test_session"
    md_path = appdata_dir / "test_agent.md"
    json_path = appdata_dir / "test_agent.json"

    assert md_path.exists(), f"Markdown sidecar not found at {md_path}"
    assert json_path.exists(), f"JSON sidecar not found at {json_path}"

    json_data = json.loads(json_path.read_text(encoding='utf-8'))
    knowledge = AgentKnowledge.from_json(json_data)
    assert isinstance(knowledge, AgentKnowledge)


@pytest.mark.anyio
async def test_no_hidden_fallback_purpose(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_FALLBACK_PURPOSE was deleted; enrichment LLM output is returned verbatim."""
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
    """Agent isolation: one failing agent does not block others."""
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
    """Zero-query fast-path: sufficient coverage skips queries."""
    from agent_eval_harness.discovery.enrichment import _enrich_single_agent
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
        repo_root: object = None
        system_type: str | None = None

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

    depth_cap = _DEPTH_CAP

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
    """Re-run cache-hit: reads actual persisted JSON, not mock."""
    from agent_eval_harness.discovery.enrichment import _enrich_single_agent
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

    import hashlib
    from agent_eval_harness.discovery.enrichment import _STRUCTURAL_PRODUCER_VERSION
    components_list = evidence['component_by_agent'].get("test_agent", [])
    component_ids = sorted([c['id'] for c in components_list])
    edges = sorted([(e['src'], e['dst']) for e in evidence['edges_by_agent'].get("test_agent", [])])
    accepted_files = []
    component_attrs = sorted([
        f"{c['id']}:{c.get('motif', 'N')}:{c.get('is_tool', False)}:{c.get('constructor_fanout', 0)}:{len(c.get('conditional_downstream', []))}"
        for c in components_list
    ])
    hash_input = '|'.join([
        str(_STRUCTURAL_PRODUCER_VERSION),
        ':'.join(component_ids),
        ':'.join(str(len(accepted_files))),
        ':'.join(f"{s}→{d}" for s, d in edges),
        ':'.join(component_attrs),
    ])
    correct_hash = hashlib.sha256(hash_input.encode('utf-8')).hexdigest()

    await repository.upsert_agent_knowledge(
        session_id="test_session",
        agent_id="test_agent",
        md_path=str(json_dir / "test_agent.md"),
        json_path=str(json_path),
        evidence_hash=correct_hash,
        confidence="medium",
        query_count=0,
    )

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
        repo_root: object = None
        system_type: str | None = None

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

    depth_cap = _DEPTH_CAP

    knowledge = await _enrich_single_agent(
        "test_agent",
        evidence,
        _EnrichCtx(),
        depth_cap,
        accepted_files,
    )

    assert knowledge.functionality == "Cached content from disk"

    from agent_eval_harness.store.database import close_db
    await close_db()


def test_grep_gate_no_codespectra_literals() -> None:
    """Grep gate: zero CodeSpectra-specific literals in enrichment code."""
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
    """Functional run against test_targets/multi_agent: degradation is explicit."""
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


class _CapturingLLMClient:
    """Records every user prompt it is sent, answers with an empty-but-valid profile."""

    def __init__(self):
        self.captured_prompts: list[str] = []

    async def complete(self, messages, *, max_tokens=512, temperature=0.2, json_mode=False, reasoning_effort=None):
        from agent_eval_harness.llm.client import LLMResponse
        user_msg = next((m.content for m in messages if m.role == "user"), "")
        self.captured_prompts.append(user_msg)
        return LLMResponse(content=json.dumps({
            "component_roles": [],
            "functionality": "captured",
            "functionality_citations": [], "context_builders": [],
            "upstream_consumers": [], "downstream_consumers": [], "failure_modes": [],
            "need_more": False, "next_queries": [],
        }), model="fake")


@pytest.mark.anyio
async def test_prompt_text_reaches_the_llm_on_a_foreign_repo_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drives the real enrich_agents -> _gather_evidence -> scan_for_prompt_sites path against a
    foreign, non-AEH repo root with a module deliberately not named prompts.py, so repo-root
    resolution plumbing is exercised, not just a resolver unit test that would pass while the
    feature stayed dead on every target but this repo."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "_appdata")
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "agent_texts.py").write_text(
        'PLANNER_SYSTEM = "You are a planning component. Decompose the query into intents."\n'
        '_RUBRIC = "Score 1-5."\n'
        "JUDGE_SYSTEM = f\"You are a critical judge reviewing another component's output. {_RUBRIC}\"\n"
        'WRITER_PROMPT = "Write an answer."\n'
        'WRITER_PROMPT += " Cite every source."\n',
        encoding="utf-8",
    )
    (pkg / "agent.py").write_text(
        "from pkg.agent_texts import PLANNER_SYSTEM, JUDGE_SYSTEM\n", encoding="utf-8"
    )
    (pkg / "agent_writer.py").write_text(
        "from .agent_texts import WRITER_PROMPT\n", encoding="utf-8"
    )

    system_map = SystemMap(
        target_system_id="pkgsys",
        components=[
            Component(id="planner_comp", role="unknown", entry_point="pkg.agent:X", file="pkg/agent.py"),
        ],
    )
    agent_flow_map = AgentFlowMap(
        target_system_id="pkgsys",
        agents=[AgentFlow(id="planner_agent", label="Planner", component_ids=["planner_comp"])],
    )

    llm_client = _CapturingLLMClient()

    await enrich_agents(
        session_id="ac6bis",
        agent_flow_map=agent_flow_map,
        system_map=system_map,
        accepted_with_annotations=["pkg/agent.py", "pkg/agent_writer.py"],
        accepted_edges=[],
        client=None,
        llm_client=llm_client,
        snapshot_id="",  # gates off DB/RRF paths — only the direct AST scan is exercised
        repo_root=tmp_path,
    )

    assert llm_client.captured_prompts
    full_prompt = "\n".join(llm_client.captured_prompts)
    assert "critical judge reviewing another component's output" in full_prompt
    assert "Decompose the query into intents" in full_prompt


def test_resolve_repo_root_never_falls_back_to_aeh_own_root_when_supplied(tmp_path: Path) -> None:
    """Regression guard: with repo_root supplied, _resolve_repo_root() never falls back to the AEH-own-repo default."""
    from agent_eval_harness.discovery.enrichment import _resolve_repo_root

    assert _resolve_repo_root(tmp_path) == tmp_path
    assert _resolve_repo_root(tmp_path) != _resolve_repo_root(None)


class _FixedRoleClient:
    """Every named component gets the SAME role, at a confidence that survives the gate."""

    def __init__(self, role: str = "worker"):
        self._role = role

    async def complete(self, messages, *, max_tokens=512, temperature=0.2, json_mode=False, reasoning_effort=None):
        import re
        from agent_eval_harness.llm.client import LLMResponse
        prompt = messages[-1].content
        ids = re.findall(r"^  - (\S+) @", prompt, re.MULTILINE)
        component_roles = [
            {"id": cid, "role": self._role, "confidence": 0.95, "reasoning": "fixed"} for cid in ids
        ]
        return LLMResponse(content=json.dumps({
            "component_roles": component_roles,
            "functionality": "fixed", "functionality_citations": [], "context_builders": [],
            "upstream_consumers": [], "downstream_consumers": [], "failure_modes": [],
            "need_more": False, "next_queries": [],
        }), model="fake")


@pytest.mark.anyio
async def test_write_back_map_path_none_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Write-back: map_path=None must degrade-don't-break — no system_map YAML written to
    disk (the sidecar .md/.json still are; that write-path is independent and unconditional)."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    system_map = SystemMap(
        target_system_id="t",
        components=[Component(id="c1", role="unknown", entry_point="m:C1", file="c1.py")],
    )
    agent_flow_map = AgentFlowMap(
        target_system_id="t", agents=[AgentFlow(id="a1", label="A1", component_ids=["c1"])]
    )

    await enrich_agents(
        session_id="writeback-none",
        agent_flow_map=agent_flow_map,
        system_map=system_map,
        accepted_with_annotations=[],
        accepted_edges=[],
        client=None,
        llm_client=_FixedRoleClient("worker"),
        snapshot_id="",
    )

    yaml_files = list(tmp_path.rglob("*.yaml"))
    assert yaml_files == [], f"expected no system_map YAML written, found: {yaml_files}"


@pytest.mark.anyio
async def test_write_back_map_path_set_lands_role_on_right_component_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_eval_harness.mapping.system_map import load_system_map

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    system_map = SystemMap(
        target_system_id="t",
        components=[
            Component(id="c1", role="unknown", entry_point="m:C1", file="c1.py"),
            Component(id="c2", role="unknown", entry_point="m:C2", file="c2.py"),
        ],
    )
    agent_flow_map = AgentFlowMap(
        target_system_id="t", agents=[AgentFlow(id="a1", label="A1", component_ids=["c1", "c2"])]
    )
    map_path = tmp_path / "map.yaml"

    await enrich_agents(
        session_id="writeback-set",
        agent_flow_map=agent_flow_map,
        system_map=system_map,
        accepted_with_annotations=[],
        accepted_edges=[],
        client=None,
        llm_client=_FixedRoleClient("worker"),
        snapshot_id="",
        map_path=map_path,
    )

    saved = load_system_map(map_path)
    assert saved.component_by_id("c1").role == "worker"
    assert saved.component_by_id("c2").role == "worker"


@pytest.mark.anyio
async def test_partial_run_agent_ids_subset_merges_not_rewrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PARTIAL-RUN HAZARD: a subset run via agent_ids must MERGE into the freshly loaded map,
    never rewrite the components list from the subset — untouched agents must keep their prior role."""
    from agent_eval_harness.mapping.system_map import load_system_map

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    system_map = SystemMap(
        target_system_id="t",
        components=[
            Component(id="c1", role="unknown", entry_point="m:C1", file="c1.py"),
            Component(id="c2", role="validator", role_confidence=0.9, role_source="llm_constrained", entry_point="m:C2", file="c2.py"),
        ],
    )
    agent_flow_map = AgentFlowMap(
        target_system_id="t",
        agents=[
            AgentFlow(id="agent_a", label="A", component_ids=["c1"]),
            AgentFlow(id="agent_b", label="B", component_ids=["c2"]),
        ],
    )
    map_path = tmp_path / "map.yaml"

    await enrich_agents(
        session_id="partial-run",
        agent_flow_map=agent_flow_map,
        system_map=system_map,
        accepted_with_annotations=[],
        accepted_edges=[],
        client=None,
        llm_client=_FixedRoleClient("worker"),
        snapshot_id="",
        agent_ids=["agent_a"],
        map_path=map_path,
    )

    saved = load_system_map(map_path)
    assert saved.component_by_id("c1").role == "worker"
    # agent_b was not part of this run — its previously-good role must survive untouched.
    assert saved.component_by_id("c2").role == "validator"
    assert saved.component_by_id("c2").role_confidence == 0.9


@pytest.mark.anyio
async def test_degraded_agent_never_blanks_siblings_nor_aborts_map_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A degraded agent's verdicts are skipped entirely — one agent failing must never blank
    the other agents' roles, nor prevent the map write for the agents that succeeded."""
    from agent_eval_harness.mapping.system_map import load_system_map

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    system_map = SystemMap(
        target_system_id="t",
        components=[
            Component(id="good", role="unknown", entry_point="m:Good", file="good.py"),
            Component(id="bad", role="unknown", entry_point="m:Bad", file="bad.py"),
        ],
    )
    agent_flow_map = AgentFlowMap(
        target_system_id="t",
        agents=[
            AgentFlow(id="agent_good", label="Good", component_ids=["good"]),
            AgentFlow(id="agent_bad", label="Bad", component_ids=["bad"]),
        ],
    )

    class _SelectivelyFailingClient:
        async def complete(self, messages, *, max_tokens=512, temperature=0.2, json_mode=False, reasoning_effort=None):
            prompt = messages[-1].content
            if "bad" in prompt:
                raise RuntimeError("intentional failure for agent_bad")
            from agent_eval_harness.llm.client import LLMResponse
            return LLMResponse(content=json.dumps({
                "component_roles": [{"id": "good", "role": "worker", "confidence": 0.9, "reasoning": "ok"}],
                "functionality": "ok", "functionality_citations": [], "context_builders": [],
                "upstream_consumers": [], "downstream_consumers": [], "failure_modes": [],
                "need_more": False, "next_queries": [],
            }), model="fake")

    map_path = tmp_path / "map.yaml"

    result = await enrich_agents(
        session_id="degraded-isolation",
        agent_flow_map=agent_flow_map,
        system_map=system_map,
        accepted_with_annotations=[],
        accepted_edges=[],
        client=None,
        llm_client=_SelectivelyFailingClient(),
        snapshot_id="",
        map_path=map_path,
    )

    assert any(k.degraded for k in result)
    assert any(not k.degraded for k in result)

    saved = load_system_map(map_path)
    assert saved.component_by_id("good").role == "worker"
    assert saved.component_by_id("bad").role == "unknown"  # degraded agent's verdict never applied


@pytest.mark.anyio
async def test_ac1_unit_proxy_validator_survives_for_high_fan_in_auditor_shaped_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proxies an empirical live-LLM run: a scripted fake LLM returns 'validator' for a
    high-fan-in, auditor-shaped component; assert the verdict survives the hard gate and lands
    on the component in the saved map."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "auditor_texts.py").write_text(
        "AGENT_K_SYSTEM = 'You are a critical auditor reviewing the outputs of 10 code "
        "analysis agents (sections A-J). Do not simply parrot self-reported confidence — "
        "evaluate it.'\n",
        encoding="utf-8",
    )
    (pkg / "auditor.py").write_text(
        "from pkg.auditor_texts import AGENT_K_SYSTEM\n", encoding="utf-8"
    )

    system_map = SystemMap(
        target_system_id="t",
        components=[
            Component(
                id="auditor", role="unknown", entry_point="pkg.auditor:Auditor", file="pkg/auditor.py",
                is_tool=False, constructor_fanout=0,
                upstream=[f"section_{i}" for i in range(10)],
            ),
        ],
    )
    agent_flow_map = AgentFlowMap(
        target_system_id="t", agents=[AgentFlow(id="auditor_agent", label="Auditor", component_ids=["auditor"])]
    )

    class _ValidatorClient:
        async def complete(self, messages, *, max_tokens=512, temperature=0.2, json_mode=False, reasoning_effort=None):
            from agent_eval_harness.llm.client import LLMResponse
            return LLMResponse(content=json.dumps({
                "component_roles": [
                    {"id": "auditor", "role": "validator", "confidence": 0.93, "reasoning": "judges 10 upstream agents"}
                ],
                "functionality": "Audits section agent outputs", "functionality_citations": [],
                "context_builders": [], "upstream_consumers": [], "downstream_consumers": [],
                "failure_modes": [], "need_more": False, "next_queries": [],
            }), model="fake")

    map_path = tmp_path / "map.yaml"

    await enrich_agents(
        session_id="ac1-proxy",
        agent_flow_map=agent_flow_map,
        system_map=system_map,
        accepted_with_annotations=["pkg/auditor.py"],
        accepted_edges=[],
        client=None,
        llm_client=_ValidatorClient(),
        snapshot_id="",
        repo_root=tmp_path,
        map_path=map_path,
    )

    from agent_eval_harness.mapping.system_map import load_system_map
    saved = load_system_map(map_path)
    assert saved.component_by_id("auditor").role == "validator"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "returned_id",
    [
        "worker_c @ pkg/w.py",             # the exact shape observed in a real run
        'id="worker_c"',
        "worker_c (file: pkg/w.py)",
    ],
)
async def test_decorated_component_id_still_lands_its_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, returned_id: str
) -> None:
    """The prompt renders components as `id="x" (file: ...)`; a model that echoes the decoration
    instead of the bare id used to have its verdict silently dropped, leaving a stray 'unknown'."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "w.py").write_text("class W:\n    pass\n", encoding="utf-8")

    system_map = SystemMap(
        target_system_id="t",
        components=[Component(id="worker_c", role="unknown", entry_point="pkg.w:W",
                              file="pkg/w.py", is_tool=False, constructor_fanout=0)],
    )
    agent_flow_map = AgentFlowMap(
        target_system_id="t",
        agents=[AgentFlow(id="a", label="A", component_ids=["worker_c"])],
    )

    class _DecoratedIdClient:
        async def complete(self, messages, *, max_tokens=512, temperature=0.2, json_mode=False, reasoning_effort=None):
            from agent_eval_harness.llm.client import LLMResponse
            return LLMResponse(content=json.dumps({
                "component_roles": [
                    {"id": returned_id, "role": "worker", "confidence": 0.9, "reasoning": "transforms input"}
                ],
                "functionality": "does work", "functionality_citations": [],
                "context_builders": [], "upstream_consumers": [], "downstream_consumers": [],
                "failure_modes": [], "need_more": False, "next_queries": [],
            }), model="fake")

    map_path = tmp_path / "map.yaml"
    await enrich_agents(
        session_id=f"decorated-{abs(hash(returned_id))}",
        agent_flow_map=agent_flow_map,
        system_map=system_map,
        accepted_with_annotations=["pkg/w.py"],
        accepted_edges=[],
        client=None,
        llm_client=_DecoratedIdClient(),
        snapshot_id="",
        repo_root=tmp_path,
        map_path=map_path,
    )

    from agent_eval_harness.mapping.system_map import load_system_map
    assert load_system_map(map_path).component_by_id("worker_c").role == "worker"


@pytest.mark.asyncio
async def test_genuinely_unknown_component_id_is_still_discarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tolerant match must not become 'accept anything' — an id for a different component
    still gets dropped rather than inventing a role."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "w.py").write_text("class W:\n    pass\n", encoding="utf-8")

    system_map = SystemMap(
        target_system_id="t",
        components=[Component(id="worker_c", role="unknown", entry_point="pkg.w:W",
                              file="pkg/w.py", is_tool=False, constructor_fanout=0)],
    )
    agent_flow_map = AgentFlowMap(
        target_system_id="t", agents=[AgentFlow(id="a", label="A", component_ids=["worker_c"])]
    )

    class _WrongIdClient:
        async def complete(self, messages, *, max_tokens=512, temperature=0.2, json_mode=False, reasoning_effort=None):
            from agent_eval_harness.llm.client import LLMResponse
            return LLMResponse(content=json.dumps({
                "component_roles": [
                    {"id": "some_other_agent @ pkg/other.py", "role": "writer", "confidence": 0.9, "reasoning": "x"}
                ],
                "functionality": "does work", "functionality_citations": [],
                "context_builders": [], "upstream_consumers": [], "downstream_consumers": [],
                "failure_modes": [], "need_more": False, "next_queries": [],
            }), model="fake")

    map_path = tmp_path / "map.yaml"
    await enrich_agents(
        session_id="wrong-id", agent_flow_map=agent_flow_map, system_map=system_map,
        accepted_with_annotations=["pkg/w.py"], accepted_edges=[], client=None,
        llm_client=_WrongIdClient(), snapshot_id="", repo_root=tmp_path, map_path=map_path,
    )

    # No verdict survived, so nothing is written back and the component keeps its prior role.
    assert system_map.component_by_id("worker_c").role == "unknown"
    assert not map_path.exists()


def test_role_vocabulary_present_and_framework_neutral() -> None:
    """The enrichment prompt must DEFINE each role (so a model doesn't invent its own meaning and
    drift run-to-run) while staying framework/target-neutral — no repo, framework, or concrete
    symbol may leak into the role definitions (nguyên tắc số 0)."""
    from agent_eval_harness.discovery.enrichment import _ENRICH_SYSTEM

    for role in ("retrieval_agent", "worker", "validator", "writer", "orchestrator", "tool"):
        assert f"- {role}:" in _ENRICH_SYSTEM, f"{role} has no definition in the enrichment prompt"

    lowered = _ENRICH_SYSTEM.lower()
    for banned in (
        "haystack", "langgraph", "langchain", "crewai", "autogen", "deepresearch",
        "codespectra", "plan_queries", "retrieve_multi", "_node_", "stategraph",
    ):
        assert banned not in lowered, f"role vocabulary leaked a target/framework literal: {banned!r}"


def test_agent_flow_role_derivation_is_deterministic_never_llm() -> None:
    """AgentFlow.role is derived in code, never asked of the LLM."""
    from agent_eval_harness.discovery.enrichment import _derive_agent_role

    all_unknown_map = SystemMap(
        target_system_id="t",
        components=[Component(id="c1", role="unknown", entry_point="m:C1")],
    )
    all_unknown_agent = AgentFlow(id="a", label="A", component_ids=["c1"])
    assert _derive_agent_role(all_unknown_agent, all_unknown_map) == "unknown"

    single_special_map = SystemMap(
        target_system_id="t",
        components=[
            Component(id="c1", role="validator", entry_point="m:C1"),
            Component(id="c2", role="worker", entry_point="m:C2"),
        ],
    )
    single_special_agent = AgentFlow(id="a", label="A", component_ids=["c1", "c2"])
    assert _derive_agent_role(single_special_agent, single_special_map) == "validator"

    diverse_map = SystemMap(
        target_system_id="t",
        components=[
            Component(id="c1", role="orchestrator", entry_point="m:C1"),
            Component(id="c2", role="retrieval_agent", entry_point="m:C2"),
            Component(id="c3", role="validator", entry_point="m:C3"),
            Component(id="c4", role="writer", entry_point="m:C4"),
        ],
    )
    diverse_agent = AgentFlow(id="a", label="A", component_ids=["c1", "c2", "c3", "c4"])
    assert _derive_agent_role(diverse_agent, diverse_map) == "worker"


@pytest.mark.anyio
async def test_cache_hit_coerces_pre_cs300_empty_role_to_unknown_never_crashes(tmp_path: Path) -> None:
    """Cache path: an old-format sidecar (component_roles entries with role='') must coerce to 'unknown', not crash."""
    from agent_eval_harness.discovery.enrichment import _enrich_single_agent, _STRUCTURAL_PRODUCER_VERSION
    from dataclasses import dataclass
    import hashlib

    json_dir = tmp_path / "knowledge"
    json_dir.mkdir()
    json_path = json_dir / "test_agent.json"
    cached_knowledge = {
        "functionality": "Cached content",
        "component_roles": [{"id": "c1", "role": "", "confidence": 0.5, "reasoning": ""}],
    }
    json_path.write_text(json.dumps(cached_knowledge), encoding="utf-8")

    agent = AgentFlow(id="test_agent", label="Test", component_ids=[])
    flow_map = AgentFlowMap(target_system_id="test", agents=[agent])
    system_map = SystemMap(target_system_id="test", components=[])
    evidence = {
        "prompt_sites_by_file": {}, "component_by_agent": {"test_agent": []},
        "edges_by_agent": {"test_agent": []}, "source_coverage": {"test_agent": 0.0},
    }

    components_list = evidence['component_by_agent'].get("test_agent", [])
    component_ids = sorted([c['id'] for c in components_list])
    edges = sorted([(e['src'], e['dst']) for e in evidence['edges_by_agent'].get("test_agent", [])])
    accepted_files = []
    component_attrs = sorted([
        f"{c['id']}:{c.get('motif', 'N')}:{c.get('is_tool', False)}:{c.get('constructor_fanout', 0)}:{len(c.get('conditional_downstream', []))}"
        for c in components_list
    ])
    hash_input = '|'.join([
        str(_STRUCTURAL_PRODUCER_VERSION),
        ':'.join(component_ids),
        ':'.join(str(len(accepted_files))),
        ':'.join(f"{s}→{d}" for s, d in edges),
        ':'.join(component_attrs),
    ])
    correct_hash = hashlib.sha256(hash_input.encode('utf-8')).hexdigest()

    await repository.upsert_agent_knowledge(
        session_id="cache-coerce", agent_id="test_agent",
        md_path=str(json_dir / "test_agent.md"), json_path=str(json_path),
        evidence_hash=correct_hash, confidence="medium", query_count=0,
    )

    @dataclass
    class _EnrichCtx:
        session_id: str = "cache-coerce"
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
        repo_root: object = None
        system_type: str | None = None

        def __post_init__(self):
            self.agent_flow_map = self.agent_flow_map or flow_map
            self.system_map = self.system_map or system_map
            self.accepted_with_annotations = self.accepted_with_annotations or []
            self.accepted_edges = self.accepted_edges or []
            self.force_agent_ids = self.force_agent_ids or []

    depth_cap = _DEPTH_CAP
    knowledge = await _enrich_single_agent("test_agent", evidence, _EnrichCtx(), depth_cap, accepted_files)

    assert knowledge.functionality == "Cached content"
    assert knowledge.component_roles[0].role == "unknown"


@pytest.mark.anyio
async def test_cs301_slice1_cache_hash_comparison(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cache re-enriches when evidence_hash differs from stored; uses AEH_DATA_DIR (Roaming), not AppData/Local."""
    monkeypatch.setenv("AEH_DATA_DIR", str(tmp_path))

    json_dir = tmp_path / "agents" / "test-session"
    json_dir.mkdir(parents=True, exist_ok=True)
    json_path = json_dir / "test_agent.json"
    old_cached = {"functionality": "Old cached", "component_roles": []}
    json_path.write_text(json.dumps(old_cached), encoding="utf-8")

    old_hash = "old_hash_123"
    await repository.upsert_agent_knowledge(
        session_id="test-session", agent_id="test_agent",
        md_path=str(json_dir / "test_agent.md"), json_path=str(json_path),
        evidence_hash=old_hash, confidence="medium", query_count=0,
    )

    from dataclasses import dataclass
    from agent_eval_harness.discovery.enrichment import _enrich_single_agent

    agent = AgentFlow(id="test_agent", label="Test", component_ids=[])
    flow_map = AgentFlowMap(target_system_id="test", agents=[agent])
    system_map = SystemMap(target_system_id="test", components=[])
    evidence = {
        "prompt_sites_by_file": {}, "component_by_agent": {"test_agent": []},
        "edges_by_agent": {"test_agent": []}, "source_coverage": {"test_agent": 0.0},
    }

    llm_client = _StubLLMClient({"functionality": "Fresh LLM response"})

    @dataclass
    class _EnrichCtx:
        session_id: str = "test-session"
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
        repo_root: object = None
        system_type: str | None = None

        def __post_init__(self):
            self.agent_flow_map = self.agent_flow_map or flow_map
            self.system_map = self.system_map or system_map
            self.accepted_with_annotations = self.accepted_with_annotations or []
            self.accepted_edges = self.accepted_edges or []
            self.force_agent_ids = self.force_agent_ids or []

    depth_cap = _DEPTH_CAP
    knowledge = await _enrich_single_agent("test_agent", evidence, _EnrichCtx(client=_StubClient({}, {}), llm_client=llm_client), depth_cap, [])

    # Hash differs from stored, so the LLM was called and we get a fresh response
    assert knowledge.functionality == "Fresh LLM response"
    assert llm_client.call_count == 1


@pytest.mark.anyio
async def test_cache_version_bump_busts_pre_fix_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CS-326 finding #1: _STRUCTURAL_PRODUCER_VERSION now also depends on a contract-producer-version
    token (the last element of _HASH_INPUT_FIELDS) so the harvest-logic fixes in this ticket bust
    every pre-fix sidecar exactly once. A sidecar hashed under the field shape from BEFORE that token
    was added must MISS on the next enrichment (re-harvest), never silently serve stale pre-fix
    output_contract/input_schemas/virtual_inputs; a subsequent run then warm-cache-hits normally."""
    monkeypatch.setenv("AEH_DATA_DIR", str(tmp_path))
    import hashlib
    from dataclasses import dataclass

    from agent_eval_harness.discovery.enrichment import (
        _enrich_single_agent,
        _HASH_INPUT_FIELDS,
        _STRUCTURAL_PRODUCER_VERSION,
    )

    json_dir = tmp_path / "agents" / "test-session"
    json_dir.mkdir(parents=True, exist_ok=True)
    json_path = json_dir / "test_agent.json"
    json_path.write_text(json.dumps({"functionality": "STALE pre-fix content"}), encoding="utf-8")

    # Reproduces exactly what evidence_hash would have been under the field shape from BEFORE this
    # ticket's cache-bump (one fewer element in _HASH_INPUT_FIELDS), for these same empty inputs.
    pre_bump_version = hashlib.sha256("|".join(_HASH_INPUT_FIELDS[:-1]).encode("utf-8")).hexdigest()[:12]
    pre_bump_hash_input = '|'.join([str(pre_bump_version), '', '0', '', ''])
    pre_bump_hash = hashlib.sha256(pre_bump_hash_input.encode('utf-8')).hexdigest()

    await repository.upsert_agent_knowledge(
        session_id="test-session", agent_id="test_agent",
        md_path=str(json_dir / "test_agent.md"), json_path=str(json_path),
        evidence_hash=pre_bump_hash, confidence="medium", query_count=0,
    )

    agent = AgentFlow(id="test_agent", label="Test", component_ids=[])
    flow_map = AgentFlowMap(target_system_id="test", agents=[agent])
    system_map = SystemMap(target_system_id="test", components=[])
    evidence = {
        "prompt_sites_by_file": {}, "component_by_agent": {"test_agent": []},
        "edges_by_agent": {"test_agent": []}, "source_coverage": {"test_agent": 0.0},
    }
    llm_client = _StubLLMClient({"functionality": "FRESH post-fix content"})

    @dataclass
    class _EnrichCtx:
        session_id: str = "test-session"
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
        repo_root: object = None
        system_type: str | None = None

        def __post_init__(self):
            self.agent_flow_map = self.agent_flow_map or flow_map
            self.system_map = self.system_map or system_map
            self.accepted_with_annotations = self.accepted_with_annotations or []
            self.accepted_edges = self.accepted_edges or []
            self.force_agent_ids = self.force_agent_ids or []

    depth_cap = _DEPTH_CAP
    knowledge = await _enrich_single_agent("test_agent", evidence, _EnrichCtx(llm_client=llm_client), depth_cap, [])

    assert knowledge.functionality == "FRESH post-fix content", (
        "a sidecar cached under the pre-cache-bump field shape must MISS and re-harvest, "
        "never silently serve stale pre-fix data"
    )
    assert llm_client.call_count == 1

    # Warm-cache path: _enrich_single_agent alone never persists (that is enrich_agents' _persist
    # node), so simulate what the first run above would have written -- the CURRENT-version hash and
    # the freshly re-harvested content -- then a second lookup with matching evidence_hash cache-hits
    # (mirrors test_cs301_slice1_cache_hash_comparison's precedent).
    json_path.write_text(json.dumps({"functionality": "FRESH post-fix content"}), encoding="utf-8")
    current_hash_input = '|'.join([str(_STRUCTURAL_PRODUCER_VERSION), '', '0', '', ''])
    current_hash = hashlib.sha256(current_hash_input.encode('utf-8')).hexdigest()
    await repository.upsert_agent_knowledge(
        session_id="test-session", agent_id="test_agent",
        md_path=str(json_dir / "test_agent.md"), json_path=str(json_path),
        evidence_hash=current_hash, confidence="medium", query_count=0,
    )

    llm_client_2 = _StubLLMClient({"functionality": "SHOULD NOT BE CALLED"})
    knowledge_2 = await _enrich_single_agent(
        "test_agent", evidence, _EnrichCtx(llm_client=llm_client_2), depth_cap, []
    )
    assert knowledge_2.functionality == "FRESH post-fix content"
    assert llm_client_2.call_count == 0, "a warm cache (current version) must hit, not re-call the LLM"


@pytest.mark.anyio
async def test_cs301_slice4_confidence_needs_human_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """needs_human non-empty => confidence != 'high'."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from dataclasses import dataclass
    from agent_eval_harness.discovery.enrichment import _enrich_single_agent

    agent = AgentFlow(id="test_agent", label="Test", component_ids=[])
    flow_map = AgentFlowMap(target_system_id="test", agents=[agent])
    system_map = SystemMap(target_system_id="test", components=[])
    evidence = {
        "prompt_sites_by_file": {}, "component_by_agent": {"test_agent": []},
        "edges_by_agent": {"test_agent": []}, "source_coverage": {"test_agent": 0.0},
    }

    # LLM response with all fields filled but will have needs_human flagged
    llm_client = _StubLLMClient({
        "functionality": "Full function",
        "functionality_citations": [{"file": "test.py", "line": 10, "snippet": "test"}],
        "context_builders": [{"name": "builder1"}],
        "failure_modes": [{"description": "fail mode"}],
    })

    @dataclass
    class _EnrichCtx:
        session_id: str = "cs301-test"
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
        repo_root: object = None
        system_type: str | None = None

        def __post_init__(self):
            self.agent_flow_map = self.agent_flow_map or flow_map
            self.system_map = self.system_map or system_map
            self.accepted_with_annotations = self.accepted_with_annotations or []
            self.accepted_edges = self.accepted_edges or []
            self.force_agent_ids = self.force_agent_ids or []

    depth_cap = _DEPTH_CAP
    knowledge = await _enrich_single_agent(
        "test_agent", evidence,
        _EnrichCtx(client=_StubClient({}, {}), llm_client=llm_client),
        depth_cap, []
    )

    # Manually add needs_human to simulate citation verification finding issues
    knowledge.needs_human.append("Unverified citation")

    assert knowledge.needs_human
    assert knowledge.confidence != 'high'


@pytest.mark.anyio
async def test_cs301_slice4_confidence_degraded_is_low() -> None:
    """degraded => confidence 'low'."""
    from dataclasses import dataclass
    from agent_eval_harness.discovery.enrichment import _enrich_single_agent

    agent = AgentFlow(id="test_agent", label="Test", component_ids=[])
    flow_map = AgentFlowMap(target_system_id="test", agents=[agent])
    system_map = SystemMap(target_system_id="test", components=[])
    evidence = {
        "prompt_sites_by_file": {}, "component_by_agent": {"test_agent": []},
        "edges_by_agent": {"test_agent": []}, "source_coverage": {"test_agent": 0.0},
    }

    # LLM that returns an error (will cause degraded)
    class _ErrorLLM:
        async def complete(self, *args, **kwargs):
            raise Exception("LLM error")

    @dataclass
    class _EnrichCtx:
        session_id: str = "cs301-test"
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
        repo_root: object = None
        system_type: str | None = None

        def __post_init__(self):
            self.agent_flow_map = self.agent_flow_map or flow_map
            self.system_map = self.system_map or system_map
            self.accepted_with_annotations = self.accepted_with_annotations or []
            self.accepted_edges = self.accepted_edges or []
            self.force_agent_ids = self.force_agent_ids or []

    depth_cap = _DEPTH_CAP
    knowledge = await _enrich_single_agent(
        "test_agent", evidence,
        _EnrichCtx(client=_StubClient({}, {}), llm_client=_ErrorLLM()),
        depth_cap, []
    )

    assert knowledge.degraded
    assert knowledge.confidence == 'low'


def test_cross_validate_output_schema_on_real_deep_research_nodes() -> None:
    """CS-326 §2.2 real-path check: exercises the entry-scoped prompt cross-check directly against
    the REAL, parsed backend/domain/qa/deep_research.py (not a fabricated snippet). node_plan and
    node_synthesize are LLM-calling siblings on the SAME DeepResearchAgent class -- the exact shape
    of the historical class-leak bug. node_plan's own real schema (`plan_schema`) is a function-local
    lowercase variable no AST-name resolver reaches, so the AST harvest alone is silent; its own
    prompt (_PLAN_SYSTEM) embeds a parseable {"steps": [...]} block, which the cross-check adopts.
    node_synthesize's own AST constant (_SYNTHESIZE_META_SCHEMA) and its own prompt
    (_SYNTHESIZE_META_SYSTEM) genuinely agree, so no discrepancy is recorded. Neither ever leaks
    the other's schema."""
    import ast

    from agent_eval_harness.discovery.enrichment import _cross_validate_output_schema, _entry_scoped_prompt_sites
    from agent_eval_harness.discovery.prompt_site_scan import scan_for_prompt_sites
    from agent_eval_harness.mapping.builder.contract_harvest import harvest_component_contract
    from agent_eval_harness.mapping.system_map import Component

    package_root = Path(__file__).parent.parent.parent / "backend"
    rel_path = "domain/qa/deep_research.py"
    disk_path = package_root / rel_path
    if not disk_path.exists():
        pytest.skip("deep_research.py not on this checkout")

    tree = ast.parse(disk_path.read_text(encoding="utf-8"))
    asts = {disk_path: tree}
    sites_by_file = scan_for_prompt_sites(package_root, [rel_path])
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "DeepResearchAgent")

    def _knowledge_for(method_name: str) -> AgentKnowledge:
        component = Component(
            id=method_name, role="worker",
            entry_point=f"domain.qa.deep_research:DeepResearchAgent.{method_name}",
            file=rel_path, entry_kind="bound_method",
        )
        _invocation, output, _constants, _notes, _kind = harvest_component_contract(component, asts, package_root)
        entry = next(
            n for n in cls.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == method_name
        )
        scoped_sites = _entry_scoped_prompt_sites(cls, entry, tree, sites_by_file.get(rel_path, []))
        validated_output, note = _cross_validate_output_schema(output, scoped_sites)
        knowledge = AgentKnowledge(output_contract=validated_output)
        if note:
            knowledge.needs_human.append(note)
        return knowledge

    plan = _knowledge_for("_node_plan")
    synthesize = _knowledge_for("_node_synthesize")

    assert plan.output_contract is not None and plan.output_contract.json_schema is not None
    assert set(plan.output_contract.json_schema.get("properties", {})) == {"debug", "hop_cap", "plan", "plan_cursor"}
    assert plan.output_contract.schema_source and "state mutation delta" in plan.output_contract.schema_source
    assert plan.output_contract.cardinality == "object"

    assert synthesize.output_contract is not None and synthesize.output_contract.json_schema is not None
    assert set(synthesize.output_contract.json_schema.get("properties", {})) == {"confidence", "reasoning_chain", "summary", "unknowns"}
    assert synthesize.output_contract.cardinality == "object"
    # node_synthesize streams the markdown answer (self._call_stream) AND separately returns this
    # meta-JSON -- the streamed text is never the scorable gold, only this schema is (§2.4).
    assert synthesize.output_contract.has_streamed_output is True
    assert plan.output_contract.has_streamed_output is False  # node_plan never streams

    assert plan.output_contract.json_schema != synthesize.output_contract.json_schema


@pytest.mark.anyio
async def test_bare_dict_kwarg_flags_needs_human_not_silently_dropped() -> None:
    """CS-326 §2.3a real-path check: JudgeComponent.worker_output: dict (a bare, unresolvable
    container type hint) against the REAL test_targets/multi_agent/components.py. With no upstream
    agent contract available (the generic, no-wiring-context case), it is flagged needs_human --
    never silently dropped from input_schemas as a bare "dict" with no trace it was ever seen."""
    import asyncio

    from agent_eval_harness.discovery.agent_knowledge import ContractArg
    from agent_eval_harness.discovery.enrichment import _EnrichmentContext, _finalize_input_contract
    from agent_eval_harness.mapping.builder.contract_harvest import _parse_files, harvest_component_contract
    from agent_eval_harness.mapping.system_map import Component

    repo_root = Path(__file__).parent.parent / "test_targets"
    comp_file = repo_root / "multi_agent" / "components.py"
    asts = _parse_files([comp_file])
    component = Component(
        id="judge", role="validator",
        entry_point="test_targets.multi_agent.components:JudgeComponent",
        file="multi_agent/components.py",
    )
    invocation, _output, _constants, _notes, _kind = harvest_component_contract(component, asts, repo_root)
    assert invocation is not None
    input_contract = [
        ContractArg(kwarg=k.name, type_hint=k.annotation or "", example="")
        for k in invocation.kwargs
    ]
    assert {a.kwarg for a in input_contract} == {"query", "worker_output"}

    knowledge = AgentKnowledge(input_contract=input_contract)
    ctx = _EnrichmentContext(
        session_id="t", snapshot_id="", agent_flow_map=AgentFlowMap(target_system_id="t", agents=[]),
        system_map=SystemMap(target_system_id="t", components=[]),
        accepted_with_annotations=[], accepted_edges=[], client=None, llm_client=None,
        depth="normal", force_agent_ids=[], semaphore=asyncio.Semaphore(1),
    )
    await _finalize_input_contract(ctx, knowledge, None)

    assert "worker_output" not in knowledge.input_schemas
    assert any("worker_output" in n for n in knowledge.needs_human)
    worker_output_arg = next(a for a in knowledge.input_contract if a.kwarg == "worker_output")
    assert worker_output_arg.source_kind == "runtime-state"


@pytest.mark.anyio
async def test_schema_enum_values_populated_on_persisted_evaluation_contract_for_live_agents(
    tmp_path, monkeypatch
) -> None:
    """CS-326 §2.4 real-path check: enum-whitelist post-processing guards on 3 real live-system
    agents (StructureAgent's folder-role guard, ViolationsAgent's rule-strength/violation-severity
    guards, AuditAgent's _normalize_conf-based confidence normalizer) populate
    OutputContract.schema_enum_values -- previously always {} (declared, consumed by
    injection/scoring.py, never written). Runs the full enrich_agents DAG, persists to a temp session
    dir, and asserts on the RELOADED sidecar's evaluation_contract, not the in-memory
    harvest_contracts() result and not suite-count."""
    monkeypatch.setenv("AEH_DATA_DIR", str(tmp_path))
    from agent_eval_harness.discovery.enrichment import agent_knowledge_dir
    from agent_eval_harness.llm.client import LLMResponse
    from agent_eval_harness.llm.fake_client import FakeLLMClient

    package_root = Path(__file__).parent.parent.parent / "backend"
    targets = {
        "structure": (
            "domain/analysis/agents/agent_structure.py",
            "domain.analysis.agents.agent_structure:StructureAgent",
        ),
        "violations": (
            "domain/analysis/agents/agent_violations.py",
            "domain.analysis.agents.agent_violations:ViolationsAgent",
        ),
        "auditor": (
            "domain/analysis/agents/agent_auditor.py",
            "domain.analysis.agents.agent_auditor:AuditAgent",
        ),
    }
    for rel_file, _entry_point in targets.values():
        if not (package_root / rel_file).exists():
            pytest.skip(f"{rel_file} not on this checkout")

    components = [
        Component(id=agent_id, role="worker", entry_point=entry_point, file=rel_file)
        for agent_id, (rel_file, entry_point) in targets.items()
    ]
    system_map = SystemMap(target_system_id="codespectra-analysis", components=components)
    agent_flow_map = AgentFlowMap(
        target_system_id="codespectra-analysis",
        agents=[AgentFlow(id=c.id, component_ids=[c.id]) for c in components],
    )
    accepted_files = [rel_file for rel_file, _entry_point in targets.values()]

    session_id = "cs326-schema-enum-values"
    llm_client = FakeLLMClient(LLMResponse(content="{}", model="fake"))
    await enrich_agents(
        session_id=session_id, agent_flow_map=agent_flow_map, system_map=system_map,
        accepted_with_annotations=accepted_files, accepted_edges=[], client=None,
        llm_client=llm_client, snapshot_id="", repo_root=package_root,
    )

    sidecar_dir = agent_knowledge_dir(session_id)
    reloaded: dict[str, AgentKnowledge] = {}
    for agent_id in targets:
        json_path = sidecar_dir / f"{agent_id}.json"
        assert json_path.exists(), f"no sidecar written for {agent_id}"
        reloaded[agent_id] = AgentKnowledge.from_json(json.loads(json_path.read_text(encoding="utf-8")))
        assert reloaded[agent_id].evaluation_contract is not None
        assert reloaded[agent_id].evaluation_contract.output is not None

    structure_enum = reloaded["structure"].evaluation_contract.output.schema_enum_values
    assert set(structure_enum.get("role", [])) == {
        "domain", "infrastructure", "delivery", "shared", "test", "generated", "unknown",
    }

    violations_enum = reloaded["violations"].evaluation_contract.output.schema_enum_values
    assert "severity" in violations_enum and violations_enum["severity"]

    auditor_enum = reloaded["auditor"].evaluation_contract.output.schema_enum_values
    assert set(auditor_enum.get("overall_confidence", [])) == {"high", "medium", "low"}
    assert set(auditor_enum.get("section_scores", [])) == {"high", "medium", "low"}
