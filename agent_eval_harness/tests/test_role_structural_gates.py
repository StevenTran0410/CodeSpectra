"""CS-299 B2 + CS-300: structure SUBTRACTS impossible roles, LLM chooses among what remains.

CS-300 moved role classification from Stage 2 (SystemMapBuilder.build()) to Stage 2.5
(enrich_agents()) — TestGeneralizationGate below drives the REAL Stage 2.5 call path over
real multi_agent source, with an adversarial fake LLM. This is the proof of the STRUCTURAL
channel only: multi_agent has zero prompt constants, so it can never exercise the prompt-content
channel (see test_enrichment.py::test_prompt_text_reaches_the_llm_on_a_foreign_repo_root for
that proof)."""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

from agent_eval_harness.discovery.enrichment import enrich_agents
from agent_eval_harness.llm.client import LLMResponse
from agent_eval_harness.mapping.agent_flow import AgentFlow, AgentFlowMap
from agent_eval_harness.mapping.builder.pipeline import SystemMapBuilder
from agent_eval_harness.mapping.builder.roles import (
    VALID_ROLES,
    admissible_roles,
    structural_facts,
)
from agent_eval_harness.mapping.system_map import Component, SystemMap, load_system_map


@pytest.fixture(autouse=True)
async def _ensure_db() -> None:
    """TestGeneralizationGate drives enrich_agents(), which needs the DB — defensive against
    an earlier test file in the same session explicitly closing it (e.g. a rerun-cache-hit
    test) and nothing else in the run re-opening it before this file's turn."""
    from agent_eval_harness.store.database import get_db, init_db
    try:
        get_db()
    except RuntimeError:
        await init_db()

# ──────────────────────────────────────────────────────────────────────────────
# admissible_roles — pure, no LLM, primitives only (CS-300: no CandidateComponent/TopologyEdges)
# ──────────────────────────────────────────────────────────────────────────────


class TestAdmissibleRoles:
    def test_both_none_degrades_to_full_valid_roles(self):
        """A hand-written or pre-CS-300 map has neither field — no subtraction fires."""
        assert admissible_roles(None, None) == VALID_ROLES

    def test_is_tool_false_removes_tool(self):
        admissible = admissible_roles(False, 0)
        assert "tool" not in admissible

    def test_is_tool_none_keeps_tool_admissible(self):
        """Unknown is not evidence of absence — None must not act like False."""
        assert "tool" in admissible_roles(None, None)

    def test_is_tool_true_keeps_tool_admissible(self):
        admissible = admissible_roles(True, 2)
        assert "tool" in admissible

    def test_constructor_fanout_zero_removes_orchestrator(self):
        assert "orchestrator" not in admissible_roles(False, 0)

    def test_constructor_fanout_one_removes_orchestrator(self):
        assert "orchestrator" not in admissible_roles(False, 1)

    def test_constructor_fanout_none_keeps_orchestrator_admissible(self):
        """Unknown is not evidence of absence — None must not act like 0."""
        assert "orchestrator" in admissible_roles(None, None)

    def test_constructor_fanout_two_plus_keeps_orchestrator_admissible(self):
        assert "orchestrator" in admissible_roles(False, 2)

    def test_worker_always_admissible(self):
        """worker is never structurally excluded — it's the safe bucket every ordinary node falls into."""
        assert "worker" in admissible_roles(False, 0)
        assert "worker" in admissible_roles(True, 2)
        assert "worker" in admissible_roles(None, None)


class TestStructuralFacts:
    """Facts are EVIDENCE shown to the LLM, never a rule — degree alone can't separate the
    semantic axis. CS-300 dropped the terminal(fan-out==0) clause: it fired on 3 infra
    services with fan-out 0 and missed the real terminal writer (fan-out 1)."""

    def test_high_fan_in_is_reported(self):
        # auditor-shaped: 10 siblings feed it — the signal that it judges their output
        facts = structural_facts(fan_in=10, fan_out=1)
        assert "fan-in: 10" in facts
        assert "fan-out: 1" in facts

    def test_zero_fan_out_gets_no_special_marker(self):
        # CS-300 B2: terminal marker used to fire here and mislead 3 infra services — gone now.
        facts = structural_facts(fan_in=11, fan_out=0)
        assert "fan-in: 11" in facts
        assert "fan-out: 0" in facts
        assert "terminal" not in facts

    def test_ordinary_node_gets_no_special_marker(self):
        # project_identity-shaped: nothing distinctive, so the LLM must fall back to the code
        facts = structural_facts(fan_in=0, fan_out=5)
        assert "fan-in: 0" in facts
        assert "terminal" not in facts


class _AlwaysRoleAdversarialClient:
    """Every component gets the SAME claimed role, regardless of evidence — this is what
    proves the hard gate's subtraction fires, not a per-class-name fixture that already
    "knows" the right answer (a fixture like that tests nothing — see test_map_builder_golden.py
    for the kind of test that would prove nothing here)."""

    def __init__(self, role: str) -> None:
        self._role = role

    async def complete(self, messages, *, max_tokens=512, temperature=0.2, json_mode=False, reasoning_effort=None):
        prompt = messages[-1].content
        ids = re.findall(r"^  - (\S+) @", prompt, re.MULTILINE)
        component_roles = [
            {"id": cid, "role": self._role, "confidence": 0.95, "reasoning": "adversarial fake"}
            for cid in ids
        ]
        content = json.dumps({
            "component_roles": component_roles,
            "functionality": "adversarial fake — content irrelevant to this test",
            "functionality_citations": [], "context_builders": [],
            "upstream_consumers": [], "downstream_consumers": [], "failure_modes": [],
            "need_more": False, "next_queries": [],
        })
        return LLMResponse(content=content, model="fake-adversarial")


def _single_agent_owning_everything(system_map: SystemMap) -> AgentFlowMap:
    return AgentFlowMap(
        target_system_id=system_map.target_system_id,
        agents=[AgentFlow(id="all", label="All", component_ids=[c.id for c in system_map.components])],
    )


class TestGeneralizationGate:
    def test_orchestrator_only_survives_for_real_constructor_fanout(
        self, target_root: Path, tmp_path: Path, monkeypatch
    ):
        """Every candidate's LLM answer is 'orchestrator' (the real observed B2 failure mode).
        Only PlannerComponent (real constructor_fanout=3, no docstring) may keep it; everyone
        else (constructor_fanout<2) must be coerced away from the false claim."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        async def run_test():
            t2_dir = target_root / "multi_agent"
            builder = SystemMapBuilder(LLMResponseNeverCallClient())
            system_map, _ = await builder.build(t2_dir)
            agent_flow_map = _single_agent_owning_everything(system_map)

            map_path = tmp_path / "map_orch.yaml"
            flows_path = tmp_path / "flows_orch.yaml"

            await enrich_agents(
                session_id="gen-gate-orch",
                agent_flow_map=agent_flow_map,
                system_map=system_map,
                accepted_with_annotations=[],
                accepted_edges=[],
                client=None,
                llm_client=_AlwaysRoleAdversarialClient("orchestrator"),
                snapshot_id="",
                map_path=map_path,
                agent_flows_path=flows_path,
            )

            saved = load_system_map(map_path)
            planner = saved.component_by_id("planner")
            assert planner is not None
            assert planner.role == "orchestrator"

            others = [c for c in saved.components if c.id != "planner"]
            assert others, "expected other components besides planner"
            assert all(c.role != "orchestrator" for c in others)

        asyncio.run(run_test())

    def test_tool_only_survives_for_real_is_tool_candidates(
        self, target_root: Path, tmp_path: Path, monkeypatch
    ):
        """Every candidate's LLM answer is 'tool'. Only the dict-registered async functions
        (real is_tool=True) may keep it; every class-based component (is_tool=False) must not."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        async def run_test():
            t2_dir = target_root / "multi_agent"
            builder = SystemMapBuilder(LLMResponseNeverCallClient())
            system_map, _ = await builder.build(t2_dir)
            agent_flow_map = _single_agent_owning_everything(system_map)

            map_path = tmp_path / "map_tool.yaml"
            flows_path = tmp_path / "flows_tool.yaml"

            await enrich_agents(
                session_id="gen-gate-tool",
                agent_flow_map=agent_flow_map,
                system_map=system_map,
                accepted_with_annotations=[],
                accepted_edges=[],
                client=None,
                llm_client=_AlwaysRoleAdversarialClient("tool"),
                snapshot_id="",
                map_path=map_path,
                agent_flows_path=flows_path,
            )

            saved = load_system_map(map_path)
            non_tool_candidates = [
                c for c in saved.components if c.id not in ("case_law_search", "decoy_lookup")
            ]
            assert non_tool_candidates, "expected class-based components besides the tool candidates"
            assert all(c.role != "tool" for c in non_tool_candidates)

            tool_candidates = [
                c for c in saved.components if c.id in ("case_law_search", "decoy_lookup")
            ]
            assert tool_candidates, "expected case_law_search/decoy_lookup to be present"
            assert all(c.role == "tool" for c in tool_candidates)

        asyncio.run(run_test())

    def test_none_fields_degrade_to_no_subtraction(self, tmp_path: Path, monkeypatch):
        """A pre-CS-300/hand-written map has is_tool=None, constructor_fanout=None on every
        component — the hard gate must NOT reject a claimed role just because the structural
        data is absent (None is 'unknown', not 'false')."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        async def run_test():
            system_map = SystemMap(
                target_system_id="legacy",
                components=[
                    Component(id="x", role="unknown", entry_point="m:X", is_tool=None, constructor_fanout=None),
                ],
            )
            agent_flow_map = _single_agent_owning_everything(system_map)
            map_path = tmp_path / "map_legacy.yaml"

            await enrich_agents(
                session_id="gen-gate-legacy",
                agent_flow_map=agent_flow_map,
                system_map=system_map,
                accepted_with_annotations=[],
                accepted_edges=[],
                client=None,
                llm_client=_AlwaysRoleAdversarialClient("orchestrator"),
                snapshot_id="",
                map_path=map_path,
            )

            saved = load_system_map(map_path)
            assert saved.component_by_id("x").role == "orchestrator"

        asyncio.run(run_test())


class LLMResponseNeverCallClient:
    """SystemMapBuilder.build() makes 0 LLM calls post-CS-300 (role classification is gone,
    constraint Phase B fires 0 calls on multi_agent) — this stub proves it by raising if used."""

    async def complete(self, *args, **kwargs):
        raise AssertionError("SystemMapBuilder.build() should not call the LLM for multi_agent")
