"""Golden tests for map builder against T1 and T2 targets."""
import asyncio
from pathlib import Path

import pytest

from agent_eval_harness.mapping.builder.pipeline import SystemMapBuilder


class _NeverCallClient:
    """CS-300: Stage 2 no longer classifies role, and constraint Phase B fires 0 calls on
    T1/T2 (verified in test_map_builder_unit.py) — the builder must never call the LLM here."""

    async def complete(self, *args, **kwargs):
        raise AssertionError("SystemMapBuilder.build() should not call the LLM for T1/T2")


class TestMapBuilderGolden:
    def test_golden_t1_components_and_roles(self, target_root: Path):
        """T1 golden test: find 2 components; role is 'unknown' — pending Stage 2.5 (CS-300)."""

        async def run_test():
            t1_dir = target_root / "linear_rag"
            builder = SystemMapBuilder(_NeverCallClient())
            system_map, summary = await builder.build(t1_dir)

            assert len(system_map.components) == 2
            assert all(c.role == "unknown" for c in system_map.components)

            ids = {c.id for c in system_map.components}
            assert "retriever" in ids
            assert "writer" in ids

        asyncio.run(run_test())

    def test_golden_t1_topology(self, target_root: Path):
        """T1 golden test: retriever -> writer edge present."""

        async def run_test():
            t1_dir = target_root / "linear_rag"
            builder = SystemMapBuilder(_NeverCallClient())
            system_map, summary = await builder.build(t1_dir)

            retriever = system_map.component_by_id("retriever")
            writer = system_map.component_by_id("writer")

            assert retriever is not None
            assert writer is not None
            assert "writer" in retriever.downstream
            assert "retriever" in writer.upstream

        asyncio.run(run_test())

    def test_golden_t2_components_and_roles(self, target_root: Path):
        """T2 golden test: 8 logical components; role is 'unknown' — pending Stage 2.5 (CS-300)."""

        async def run_test():
            t2_dir = target_root / "multi_agent"
            builder = SystemMapBuilder(_NeverCallClient())
            system_map, summary = await builder.build(t2_dir)

            # 8 components: guard_rule, guard_llm, planner, worker, judge, writer, 2 tools
            assert len(system_map.components) == 8
            assert all(c.role == "unknown" for c in system_map.components)

            ids = {c.id for c in system_map.components}
            expected_ids = {
                "guard_rule", "guard_llm", "planner", "worker",
                "judge", "writer", "case_law_search", "decoy_lookup",
            }
            assert expected_ids.issubset(ids)

        asyncio.run(run_test())

    def test_golden_t2_constraints_cited(self, target_root: Path):
        """T2 golden test: both planted constraints have correct value + citation."""

        async def run_test():
            t2_dir = target_root / "multi_agent"
            builder = SystemMapBuilder(_NeverCallClient())
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

    def test_golden_t2_topology_planner_fanout(self, target_root: Path):
        """T2 golden test: planner has worker/judge/writer downstream."""

        async def run_test():
            t2_dir = target_root / "multi_agent"
            builder = SystemMapBuilder(_NeverCallClient())
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

    def test_golden_summary_format_has_fixed_headers(self, target_root: Path):
        """Summary should have fixed header format."""

        async def run_test():
            t2_dir = target_root / "multi_agent"
            builder = SystemMapBuilder(_NeverCallClient())
            system_map, summary = await builder.build(t2_dir)

            # Check for fixed headers
            assert "=== AEH System Map Summary ===" in summary
            assert "target:" in summary
            assert "components_found:" in summary
            assert "unknown:" in summary
            assert "discrepancies:" in summary

        asyncio.run(run_test())
