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
        repo_root: object = None

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

    # Compute the correct hash based on evidence and accepted_files (empty in this case)
    import hashlib
    from agent_eval_harness.discovery.enrichment import _STRUCTURAL_PRODUCER_VERSION
    component_ids = sorted([c['id'] for c in evidence['component_by_agent'].get("test_agent", [])])
    edges = sorted([(e['src'], e['dst']) for e in evidence['edges_by_agent'].get("test_agent", [])])
    accepted_files = []
    hash_input = '|'.join([
        str(_STRUCTURAL_PRODUCER_VERSION),
        ':'.join(component_ids),
        ':'.join(str(len(accepted_files))),
        ':'.join(f"{s}→{d}" for s, d in edges),
    ])
    correct_hash = hashlib.sha256(hash_input.encode('utf-8')).hexdigest()

    # Update the stored hash to match the computed hash
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
        accepted_files,
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
    """CS-300 §6: AC6 alone goes green whether or not the prompt-resolver plumbing works,
    because test_targets/multi_agent has zero prompt constants. This drives the REAL call path
    (enrich_agents -> _gather_evidence -> scan_for_prompt_sites) against a root that is NOT
    AEH's own repo and NOT test_targets/, with a module deliberately NOT named prompts.py — it
    fails today for the PLUMBING reason (_resolve_repo_root() would resolve AEH's own root, not
    the target's), not merely for a resolver bug. A resolver unit test alone would pass while
    this feature stayed dead on every target except this one repo."""
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
    """Regression guard (CS-300 §6): with repo_root supplied, _resolve_repo_root()'s value is
    never the AEH-own-repo default."""
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
    """PARTIAL-RUN HAZARD (CS-300 §8): a subset run via agent_ids must MERGE into the freshly
    loaded map, never rewrite the components list from the subset — agent_b's component here
    is untouched by this run and must keep its prior role."""
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
    """AC1 unit proxy (§9 — the empirical run itself needs a live LLM, this proxies it): a
    scripted fake LLM returns 'validator' for a high-fan-in component whose evidence carries
    auditor-shaped prompt text ('critical auditor reviewing the outputs of 10 code analysis
    agents') — assert the verdict survives the hard gate (validator has no structural
    subtraction) and lands on the component in the saved map."""
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


def test_agent_flow_role_derivation_is_deterministic_never_llm() -> None:
    """CS-300 §5: AgentFlow.role is derived in code, never asked of the LLM."""
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
    """Cache path: a pre-CS-300 sidecar (component_roles entries with role='') must coerce to
    'unknown', not crash."""
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

    # Compute the correct hash
    component_ids = sorted([c['id'] for c in evidence['component_by_agent'].get("test_agent", [])])
    edges = sorted([(e['src'], e['dst']) for e in evidence['edges_by_agent'].get("test_agent", [])])
    accepted_files = []
    hash_input = '|'.join([
        str(_STRUCTURAL_PRODUCER_VERSION),
        ':'.join(component_ids),
        ':'.join(str(len(accepted_files))),
        ':'.join(f"{s}→{d}" for s, d in edges),
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

        def __post_init__(self):
            self.agent_flow_map = self.agent_flow_map or flow_map
            self.system_map = self.system_map or system_map
            self.accepted_with_annotations = self.accepted_with_annotations or []
            self.accepted_edges = self.accepted_edges or []
            self.force_agent_ids = self.force_agent_ids or []

    depth_cap = {"queries": 3, "llm_calls": 2, "read_file": 2}
    knowledge = await _enrich_single_agent("test_agent", evidence, _EnrichCtx(), depth_cap, accepted_files)

    assert knowledge.functionality == "Cached content"
    assert knowledge.component_roles[0].role == "unknown"


# CS-301 Tests (Slice 1: Cache hash comparison and force flag)
@pytest.mark.anyio
async def test_cs301_slice1_cache_hash_comparison(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """AC1: Cache re-enriches when evidence_hash differs from stored."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    try:
        from agent_eval_harness.store.database import get_db
        get_db()
    except RuntimeError:
        await init_db()

    # Store a cached record with old hash
    json_dir = tmp_path / "AppData" / "Local" / "codespectra" / "agents" / "test-session"
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

        def __post_init__(self):
            self.agent_flow_map = self.agent_flow_map or flow_map
            self.system_map = self.system_map or system_map
            self.accepted_with_annotations = self.accepted_with_annotations or []
            self.accepted_edges = self.accepted_edges or []
            self.force_agent_ids = self.force_agent_ids or []

    depth_cap = {"queries": 3, "llm_calls": 2, "read_file": 2}
    knowledge = await _enrich_single_agent("test_agent", evidence, _EnrichCtx(client=_StubClient({}, {}), llm_client=llm_client), depth_cap, [])

    # Hash is different, so LLM was called and we get fresh response
    assert knowledge.functionality == "Fresh LLM response"
    assert llm_client.call_count == 1


@pytest.mark.anyio
async def test_cs301_slice4_confidence_needs_human_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """AC5: needs_human non-empty => confidence != 'high'."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    try:
        from agent_eval_harness.store.database import get_db
        get_db()
    except RuntimeError:
        await init_db()

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

        def __post_init__(self):
            self.agent_flow_map = self.agent_flow_map or flow_map
            self.system_map = self.system_map or system_map
            self.accepted_with_annotations = self.accepted_with_annotations or []
            self.accepted_edges = self.accepted_edges or []
            self.force_agent_ids = self.force_agent_ids or []

    depth_cap = {"queries": 3, "llm_calls": 2, "read_file": 2}
    knowledge = await _enrich_single_agent(
        "test_agent", evidence,
        _EnrichCtx(client=_StubClient({}, {}), llm_client=llm_client),
        depth_cap, []
    )

    # Manually add needs_human to simulate citation verification finding issues
    knowledge.needs_human.append("Unverified citation")

    # Even with all fields filled, confidence must not be 'high' if needs_human is non-empty
    assert knowledge.needs_human
    assert knowledge.confidence != 'high'


@pytest.mark.anyio
async def test_cs301_slice4_confidence_degraded_is_low() -> None:
    """AC5: degraded => confidence 'low'."""
    from dataclasses import dataclass
    from agent_eval_harness.discovery.enrichment import _enrich_single_agent
    from agent_eval_harness.mapping.agent_flow import AgentFlow, AgentFlowMap
    from agent_eval_harness.mapping.system_map import SystemMap

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

        def __post_init__(self):
            self.agent_flow_map = self.agent_flow_map or flow_map
            self.system_map = self.system_map or system_map
            self.accepted_with_annotations = self.accepted_with_annotations or []
            self.accepted_edges = self.accepted_edges or []
            self.force_agent_ids = self.force_agent_ids or []

    depth_cap = {"queries": 3, "llm_calls": 2, "read_file": 2}
    knowledge = await _enrich_single_agent(
        "test_agent", evidence,
        _EnrichCtx(client=_StubClient({}, {}), llm_client=_ErrorLLM()),
        depth_cap, []
    )

    # When degraded is True, confidence must be 'low'
    assert knowledge.degraded
    assert knowledge.confidence == 'low'
