"""Tests for agent_knowledge schema hardening (CS-291)."""
import json
from pathlib import Path

import pytest

from agent_eval_harness.discovery.agent_knowledge import (
    AgentKnowledge,
    Citation,
    ConsumerRef,
    ContextBuilderRef,
    FailureModeRef,
    verify_citations,
)
from agent_eval_harness.code_injection.plan_renderer import render_eval_plan_md
from agent_eval_harness.mapping.system_map import SystemMap, Component


@pytest.mark.anyio
async def test_verify_citations_all_five_sources(tmp_path: Path) -> None:
    """AC4: Phantom detection across all 5 semantic sources."""
    knowledge = AgentKnowledge(
        functionality="Test functionality",
        functionality_citations=[
            Citation(file="phantom.py", line=10, symbol="phantom_func")
        ],
        context_builders=[
            ContextBuilderRef(name="ctx_builder", file="phantom.py", line=20, builds_kwarg="ctx")
        ],
        upstream_consumers=[
            ConsumerRef(name="upstream_svc", file="phantom.py", line=30)
        ],
        downstream_consumers=[
            ConsumerRef(name="downstream_svc", file="phantom.py", line=40)
        ],
        failure_modes=[
            FailureModeRef(description="Fails when timeout", file="phantom.py", line=50)
        ],
    )

    report = verify_citations(knowledge, tmp_path)

    assert len(report.claims) == 5
    assert all(c.status == "phantom" for c in report.claims), \
        f"Expected all phantom, got: {[c.status for c in report.claims]}"

    assert len(knowledge.needs_human) == 5
    assert all("Phantom citation" in item for item in knowledge.needs_human)


@pytest.mark.anyio
async def test_verify_citations_union_resolves_offline_and_promptsite(tmp_path: Path) -> None:
    """A citation is valid if it resolves to readable context by ANY of three routes; only a
    citation that resolves to nothing is flagged. Guards the CS-301 follow-up union rule."""
    from agent_eval_harness.discovery.agent_knowledge import PromptSiteRef

    src = tmp_path / "agent.py"
    src.write_text(
        "import os\n"                       # 1
        "from .prompts import SYS_PROMPT\n"  # 2  (prompt-site line)
        "\n"                                  # 3
        "@component\n"                        # 4  (decorator, one above the class)
        "class MyAgent:\n"                    # 5  (class def)
        "    def run(self):\n"                # 6
        "        return SYS_PROMPT\n",        # 7  (mid-class, class name NOT on this line)
        encoding="utf-8",
    )
    symbols = {"agent.py": [{"name": "MyAgent", "kind": "class", "line_start": 5, "line_end": 7}]}

    knowledge = AgentKnowledge(
        functionality="x",
        prompt_sites=[PromptSiteRef(file="agent.py", line=2, kind="prompt_import", snippet="")],
        functionality_citations=[
            Citation(file="agent.py", line=4, symbol="MyAgent"),   # near-above span -> resolves
            Citation(file="agent.py", line=7, symbol="MyAgent"),   # mid-class span  -> resolves
            Citation(file="agent.py", line=2, symbol="prompt_import"),  # prompt-site -> resolves
            Citation(file="agent.py", line=1, symbol="ghost"),     # nothing there   -> flagged
        ],
    )

    report = verify_citations(knowledge, tmp_path, symbols_by_file=symbols)

    by_line = {c.citation.line: c.status for c in report.claims}
    assert by_line[4] == "verified"
    assert by_line[7] == "verified"
    assert by_line[2] == "verified"
    assert by_line[1] == "unverified"
    assert knowledge.needs_human == ["Unverified citation: agent.py:1:ghost"]


@pytest.mark.anyio
async def test_static_wins_llm_cannot_override() -> None:
    """AC4: Static-wins cross-check — LLM cannot override structural fields."""
    from agent_eval_harness.discovery.enrichment import _enrich_single_agent
    from agent_eval_harness.mapping.agent_flow import AgentFlow, AgentFlowMap
    from agent_eval_harness.store.database import init_db, get_db
    from dataclasses import dataclass

    try:
        get_db()
    except RuntimeError:
        await init_db()

    class _StubLLMClient:
        """Stub LLM that returns an attempt to override location."""
        async def complete(self, messages, *, max_tokens=512, temperature=0.2, json_mode=False, reasoning_effort=None):
            from agent_eval_harness.llm.client import LLMResponse
            return LLMResponse(content=json.dumps({
                "functionality": "LLM purpose",
                "location": {
                    "file": "llm_lie.py",
                    "line_start": 0,
                    "line_end": 0,
                    "entry_method": "fake",
                    "entry_line": 0,
                },
            }), model="stub")

    flow_map = AgentFlowMap(
        target_system_id="test_system",
        agents=[AgentFlow(id="test_agent", label="Test Agent", component_ids=[])]
    )
    system_map = SystemMap(target_system_id="test", components=[])
    evidence = {
        'prompt_sites_by_file': {},
        'component_by_agent': {"test_agent": []},
        'edges_by_agent': {"test_agent": []},
        'source_coverage': {"test_agent": 0.0},
    }

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
                self.llm_client = _StubLLMClient()
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

    assert knowledge.evidence_hash
    assert knowledge.generated_at
    assert knowledge.query_count == 0
    # LLM tried to set location to llm_lie.py — static-wins must reject it
    assert knowledge.location is None or knowledge.location.file != "llm_lie.py"


def test_plan_renderer_byte_identical_no_knowledge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """AC7: plan_renderer output byte-identical when no knowledge sidecar exists."""
    from unittest.mock import Mock
    import datetime

    fixed_time = datetime.datetime(2026, 7, 14, 12, 0, 0)
    mock_datetime_module = Mock()
    mock_datetime_module.datetime.utcnow.return_value = fixed_time
    monkeypatch.setattr("agent_eval_harness.code_injection.plan_renderer.datetime", mock_datetime_module)

    minimal_system_map = SystemMap(
        target_system_id="test_system",
        components=[
            Component(
                id="test_comp",
                role="test_role",
                entry_point="test.entry",
                file="test.py",
            ),
        ],
    )
    minimal_wiring = {"plan_id": "test", "entrypoint": "test.entry", "component_ids": ["test_comp"]}
    minimal_ds = [{"dataset_id": "ds1", "kind": "qa", "case_count": 0, "gate_ids": []}]

    output_1 = render_eval_plan_md(
        minimal_system_map,
        minimal_wiring,
        minimal_ds,
        "sess_test",
        "aeh/eval-sess_test",
        knowledge_dir=None,
    )

    output_2 = render_eval_plan_md(
        minimal_system_map,
        minimal_wiring,
        minimal_ds,
        "sess_test",
        "aeh/eval-sess_test",
        knowledge_dir=None,
    )

    assert output_1 == output_2

    empty_knowledge_dir = tmp_path / "empty_knowledge"
    empty_knowledge_dir.mkdir()
    output_3 = render_eval_plan_md(
        minimal_system_map,
        minimal_wiring,
        minimal_ds,
        "sess_test",
        "aeh/eval-sess_test",
        knowledge_dir=empty_knowledge_dir,
    )

    assert output_3 == output_1


@pytest.mark.anyio
async def test_migration_v21_agent_knowledge_table_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Migration v21: agent_knowledge table exists after init_db()."""
    monkeypatch.setenv("AEH_DATA_DIR", str(tmp_path))

    from agent_eval_harness.store.database import get_db, init_db as init_db_for_test
    from agent_eval_harness.store import database as db_module

    db_module._db = None

    await init_db_for_test()
    db = get_db()

    async with db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='agent_knowledge'"
    ) as cur:
        row = await cur.fetchone()

    assert row is not None, "agent_knowledge table does not exist"

    async with db.execute("PRAGMA table_info(agent_knowledge)") as cur:
        columns_rows = await cur.fetchall()

    column_names = [r[1] for r in columns_rows]
    expected_columns = [
        "session_id", "agent_id", "md_path", "json_path",
        "evidence_hash", "confidence", "query_count", "generated_at"
    ]
    for expected in expected_columns:
        assert expected in column_names, f"Column {expected} not in agent_knowledge table"

    from agent_eval_harness.store.database import close_db
    await close_db()


def test_from_json_backward_compat_str_consumers() -> None:
    """Backward-compat: from_json() degrades gracefully when legacy list[str] consumers loaded."""
    legacy_data = {
        "upstream_consumers": ["plain_string_consumer"],
        "downstream_consumers": ["another_string"],
        "failure_modes": ["string mode"],
    }

    result = AgentKnowledge.from_json(legacy_data)

    assert isinstance(result, AgentKnowledge)
    assert result.degraded is True
    assert "Validation error" in result.degraded_reason
