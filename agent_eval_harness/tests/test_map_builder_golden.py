"""Golden tests for map builder against T1 and T2 targets."""
from pathlib import Path

import pytest

from agent_eval_harness.llm.client import LLMResponse
from agent_eval_harness.mapping.builder.pipeline import SystemMapBuilder
from tests._stubs import KeyedFakeLLMClient


@pytest.fixture
def t1_fixtures() -> dict[str, LLMResponse]:
    """Role classification responses for T1."""
    import json

    return {
        "RetrieverComponent": LLMResponse(
            content=json.dumps({
                "role": "retrieval_agent",
                "confidence": 0.9,
                "reasoning": "Pure Python keyword overlap ranker, reads corpus to produce ranked "
                "results.",
            }),
            model="fake-gpt",
        ),
        "WriterComponent": LLMResponse(
            content=json.dumps({
                "role": "writer",
                "confidence": 0.95,
                "reasoning": "Answers strictly from retrieved context using LLM."
            }),
            model="fake-gpt",
        ),
    }


@pytest.fixture
def t2_fixtures() -> dict[str, LLMResponse]:
    """Role classification responses for T2."""
    import json

    return {
        "GuardComponent (sub-role: rule)": LLMResponse(
            content=json.dumps({
                "role": "input_guard.rule",
                "confidence": 0.95,
                "reasoning": "Deterministic length-based input filtering."
            }),
            model="fake-gpt",
        ),
        "GuardComponent (sub-role: llm)": LLMResponse(
            content=json.dumps({
                "role": "input_guard.llm",
                "confidence": 0.9,
                "reasoning": "LLM-based policy violation detection."
            }),
            model="fake-gpt",
        ),
        "PlannerComponent": LLMResponse(
            content=json.dumps({
                "role": "orchestrator",
                "confidence": 0.95,
                "reasoning": "Decomposes intents, routes to worker, manages retry logic with judge."
            }),
            model="fake-gpt",
        ),
        "WorkerComponent": LLMResponse(
            content=json.dumps({
                "role": "retrieval_agent",
                "confidence": 0.9,
                "reasoning": "Manages tools to gather context based on decomposed intents."
            }),
            model="fake-gpt",
        ),
        "JudgeComponent": LLMResponse(
            content=json.dumps({
                "role": "validator",
                "confidence": 0.9,
                "reasoning": "Judges sufficiency of retrieved context, can trigger retry."
            }),
            model="fake-gpt",
        ),
        "WriterComponent": LLMResponse(
            content=json.dumps({
                "role": "writer",
                "confidence": 0.95,
                "reasoning": "Produces final grounded output from judged context."
            }),
            model="fake-gpt",
        ),
        "case_law_search": LLMResponse(
            content=json.dumps({
                "role": "tool",
                "confidence": 0.95,
                "reasoning": "Leaf capability invoked by worker to search case law."
            }),
            model="fake-gpt",
        ),
        "decoy_lookup": LLMResponse(
            content=json.dumps({
                "role": "tool",
                "confidence": 0.95,
                "reasoning": "Leaf capability invoked by worker (decoy tool)."
            }),
            model="fake-gpt",
        ),
    }


class TestMapBuilderGolden:
    def test_golden_t1_components_and_roles(self, target_root: Path, t1_fixtures: dict):
        """T1 golden test: find 2 components with correct roles."""
        import asyncio

        async def run_test():
            t1_dir = target_root / "linear_rag"
            builder = SystemMapBuilder(KeyedFakeLLMClient(t1_fixtures))
            system_map, summary = await builder.build(t1_dir)

            # Should have 2 components
            assert len(system_map.components) == 2

            # Check roles
            roles = {c.role for c in system_map.components}
            assert roles == {"retrieval_agent", "writer"}

            # Check IDs
            ids = {c.id for c in system_map.components}
            assert "retriever" in ids
            assert "writer" in ids

        asyncio.run(run_test())

    def test_golden_t1_topology(self, target_root: Path, t1_fixtures: dict):
        """T1 golden test: retriever -> writer edge present."""
        import asyncio

        async def run_test():
            t1_dir = target_root / "linear_rag"
            builder = SystemMapBuilder(KeyedFakeLLMClient(t1_fixtures))
            system_map, summary = await builder.build(t1_dir)

            retriever = system_map.component_by_id("retriever")
            writer = system_map.component_by_id("writer")

            assert retriever is not None
            assert writer is not None
            assert "writer" in retriever.downstream
            assert "retriever" in writer.upstream

        asyncio.run(run_test())

    def test_golden_t2_components_and_roles(self, target_root: Path, t1_fixtures: dict,
                                              t2_fixtures: dict):
        """T2 golden test: 8 logical components with correct roles."""
        import asyncio

        async def run_test():
            t2_dir = target_root / "multi_agent"
            # Merge fixtures (T2 imports WriterComponent from T1)
            merged_fixtures = {**t1_fixtures, **t2_fixtures}
            builder = SystemMapBuilder(KeyedFakeLLMClient(merged_fixtures))
            system_map, summary = await builder.build(t2_dir)

            # 8 components: guard_rule, guard_llm, planner, worker, judge, writer
            assert len(system_map.components) == 8

            # Check that core roles are present
            roles = [c.role for c in system_map.components]
            role_set = set(roles)
            expected_core = {
                "input_guard.rule",
                "input_guard.llm",
                "orchestrator",
                "retrieval_agent",
                "validator",
                "tool",
            }
            assert expected_core.issubset(role_set)

        asyncio.run(run_test())

    def test_golden_t2_constraints_cited(self, target_root: Path, t1_fixtures: dict,
                                         t2_fixtures: dict):
        """T2 golden test: both planted constraints have correct value + citation."""
        import asyncio

        async def run_test():
            t2_dir = target_root / "multi_agent"
            # Merge fixtures (T2 imports WriterComponent from T1)
            merged_fixtures = {**t1_fixtures, **t2_fixtures}
            builder = SystemMapBuilder(KeyedFakeLLMClient(merged_fixtures))
            system_map, summary = await builder.build(t2_dir)

            # Find guard and planner
            guard_candidates = [c for c in system_map.components if "guard" in c.id]
            planner = system_map.component_by_id("planner")

            # Check MIN_QUERY_LENGTH in at least one guard variant
            guard_has_constraint = False
            for guard in guard_candidates:
                for constraint in guard.constraints:
                    if (
                        constraint.value == 5
                        and "GuardComponent.MIN_QUERY_LENGTH" in constraint.source
                    ):
                        guard_has_constraint = True
                        break

            assert guard_has_constraint, "MIN_QUERY_LENGTH constraint not found in guard"

            # Check MAX_ITEMS_PER_CALL in planner
            planner_has_constraint = False
            if planner:
                for constraint in planner.constraints:
                    if (
                        constraint.value == 2
                        and "PlannerComponent.MAX_ITEMS_PER_CALL" in constraint.source
                    ):
                        planner_has_constraint = True
                        break

            assert planner_has_constraint, "MAX_ITEMS_PER_CALL constraint not found in planner"

        asyncio.run(run_test())

    def test_golden_t2_topology_planner_fanout(self, target_root: Path, t2_fixtures: dict):
        """T2 golden test: planner has worker/judge/writer downstream."""
        import asyncio

        async def run_test():
            t2_dir = target_root / "multi_agent"
            builder = SystemMapBuilder(KeyedFakeLLMClient(t2_fixtures))
            system_map, summary = await builder.build(t2_dir)

            planner = system_map.component_by_id("planner")
            assert planner is not None

            downstream = set(planner.downstream)
            assert "worker" in downstream
            assert "judge" in downstream
            assert "writer" in downstream

        asyncio.run(run_test())

    def test_golden_schema_validation_hard_gate(self):
        """SystemMap schema validation is a hard gate."""
        from pydantic import ValidationError

        from agent_eval_harness.mapping.system_map import SystemMap

        # Missing entry_point should fail
        with pytest.raises(ValidationError):
            SystemMap.model_validate({
                "target_system_id": "test",
                "components": [{
                    "id": "test",
                    "role": "tool",
                    # "entry_point" is required
                    "span_match": [],
                }],
            })

    def test_golden_summary_format_has_fixed_headers(self, target_root: Path, t1_fixtures: dict,
                                                     t2_fixtures: dict):
        """Summary should have fixed header format."""
        import asyncio

        async def run_test():
            t2_dir = target_root / "multi_agent"
            # Merge fixtures (T2 imports WriterComponent from T1)
            merged_fixtures = {**t1_fixtures, **t2_fixtures}
            builder = SystemMapBuilder(KeyedFakeLLMClient(merged_fixtures))
            system_map, summary = await builder.build(t2_dir)

            # Check for fixed headers
            assert "=== AEH System Map Summary ===" in summary
            assert "target:" in summary
            assert "components_found:" in summary
            assert "unknown:" in summary
            assert "discrepancies:" in summary

        asyncio.run(run_test())
