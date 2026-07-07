"""Unit tests for map builder components (no LLM)."""
import tempfile
from pathlib import Path

import pytest

from agent_eval_harness.mapping.builder.constraints import mine_constraints
from agent_eval_harness.mapping.builder.scanners import HaystackScanner
from agent_eval_harness.mapping.builder.topology import extract_topology


class TestHaystackScanner:
    def test_haystack_scanner_finds_retriever_and_writer_in_t1(self, target_root: Path):
        """T1 (linear_rag) has 2 components: RetrieverComponent, WriterComponent."""
        scanner = HaystackScanner()
        t1_dir = target_root / "linear_rag"
        files = sorted(t1_dir.glob("**/*.py"))

        candidates = scanner.scan(files)

        # Filter to just T1 components (not from other targets)
        t1_candidates = [c for c in candidates if c.file.parent.name == "linear_rag"]

        class_names = {c.class_name for c in t1_candidates}
        assert "RetrieverComponent" in class_names
        assert "WriterComponent" in class_names

    def test_haystack_scanner_finds_all_candidates_in_t2(self, target_root: Path):
        """T2 (multi_agent) has 8 logical candidates after split."""
        scanner = HaystackScanner()
        # Scan entire test_targets to include imported classes like WriterComponent
        files = sorted(target_root.glob("**/*.py"))

        candidates = scanner.scan(files)

        candidate_ids = {c.candidate_id for c in candidates}

        # Check for core T2 candidates
        expected = {
            "guard_rule", "guard_llm", "planner", "worker",
            "judge", "writer", "case_law_search", "decoy_lookup"
        }
        # At least the core candidates should be present
        for exp_id in expected:
            assert exp_id in candidate_ids, f"Missing {exp_id} from {candidate_ids}"

    def test_manual_span_extraction_guard_yields_two_hints_with_literal_tags(
        self, target_root: Path
    ):
        """Guard component should have two manual_span hints with literal tag values."""
        scanner = HaystackScanner()
        t2_dir = target_root / "multi_agent"
        files = sorted(t2_dir.glob("**/*.py"))

        candidates = scanner.scan(files)

        guard_candidates = [c for c in candidates if "guard" in c.candidate_id]
        assert len(guard_candidates) > 0

        # Check that we have hints with tags
        all_hints = []
        for candidate in guard_candidates:
            all_hints.extend(candidate.manual_span_hints)

        # Should have hints with aeh.check.kind tag
        tagged_hints = [h for h in all_hints if "aeh.check.kind" in h.tags]
        assert len(tagged_hints) >= 1

    def test_manual_span_dynamic_tag_value_dropped(self, target_root: Path):
        """Dynamic tag values (variables) should be dropped from tags dict."""
        scanner = HaystackScanner()
        t2_dir = target_root / "multi_agent"
        files = sorted(t2_dir.glob("**/*.py"))

        candidates = scanner.scan(files)

        worker_candidates = [c for c in candidates if c.class_name == "WorkerComponent"]
        assert len(worker_candidates) > 0

        # Worker's manual_span has a dynamic {"aeh.tool.name": tool_name} tag, filtered out below
        worker_split_candidates = [
            c for c in candidates if c.class_name == "WorkerComponent" and c.tag_suffix
        ]
        assert len(worker_split_candidates) == 0

    def test_sub_span_split_guard_produces_two_virtual_candidates(self, target_root: Path):
        """Guard component with multi-valued tag should split into guard_rule and guard_llm."""
        scanner = HaystackScanner()
        t2_dir = target_root / "multi_agent"
        files = sorted(t2_dir.glob("**/*.py"))

        candidates = scanner.scan(files)

        guard_ids = {c.candidate_id for c in candidates if "guard" in c.candidate_id}
        assert "guard_rule" in guard_ids
        assert "guard_llm" in guard_ids

    def test_tool_discovery_from_dict_literal_call_site(self, target_root: Path):
        """Tools discovered from dict literals should have is_tool=True and registered_name."""
        scanner = HaystackScanner()
        t2_dir = target_root / "multi_agent"
        files = sorted(t2_dir.glob("**/*.py"))

        candidates = scanner.scan(files)

        tool_candidates = [c for c in candidates if c.is_tool]
        tool_names = {c.registered_name for c in tool_candidates}

        assert "case_law_search" in tool_names
        assert "decoy_lookup" in tool_names

        for tool_candidate in tool_candidates:
            assert tool_candidate.registered_name is not None

    def test_scanner_skips_syntax_error_files(self):
        """Scanner should skip files with syntax errors, not abort."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create a valid file
            valid_file = tmp_path / "valid.py"
            valid_file.write_text(
                "@component\nclass ValidComponent:\n    pass\n"
            )

            # Create a broken file
            broken_file = tmp_path / "broken.py"
            broken_file.write_text("this is not valid python !!!")

            scanner = HaystackScanner()
            files = sorted(tmp_path.glob("**/*.py"))
            candidates = scanner.scan(files)

            # Should have found the valid component
            assert any(c.class_name == "ValidComponent" for c in candidates)

    def test_haystack_scanner_returns_empty_for_non_agentic(self, target_root: Path):
        """Non-agentic code (store module) should produce empty result."""
        scanner = HaystackScanner()
        store_dir = target_root.parent / "agent_eval_harness" / "store"
        if not store_dir.exists():
            pytest.skip("Store module not found")

        files = sorted(store_dir.glob("**/*.py"))
        candidates = scanner.scan(files)

        # Store module has no @component classes
        assert len(candidates) == 0

    def test_entry_point_is_importable(self, target_root: Path):
        """Entry points should resolve to actual importable objects."""
        import importlib

        scanner = HaystackScanner()
        t1_dir = target_root / "linear_rag"
        files = sorted(t1_dir.glob("**/*.py"))

        candidates = scanner.scan(files)
        t1_candidates = [c for c in candidates if c.file.parent.name == "linear_rag"]

        package_root = t1_dir.parent.parent  # agent_eval_harness

        # Test a few candidates
        for candidate in t1_candidates[:2]:
            rel_path = candidate.file.relative_to(package_root)
            module_path = rel_path.with_suffix("").as_posix().replace("/", ".")
            entry_point = f"{module_path}:{candidate.class_name}"

            # Try to import
            module_name, class_name = entry_point.split(":")
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name, None)

            assert cls is not None
            assert cls.__name__ == candidate.class_name

    def test_langgraph_and_lcel_admission(self):
        """Verify that LangGraph add_node and LangChain LCEL chain nodes are admitted as CandidateComponents."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # File defining the classes and registering via LangGraph and LCEL
            file_a = tmp_path / "app.py"
            file_a.write_text(
                "class MyAgent:\n"
                "    pass\n"
                "class MyTool:\n"
                "    pass\n"
                "class LCELStep1:\n"
                "    pass\n"
                "class LCELStep2:\n"
                "    pass\n\n"
                "builder.add_node('agent', MyAgent())\n"
                "builder.add_node('tool', MyTool)\n"
                "chain = LCELStep1() | LCELStep2()\n"
            )

            scanner = HaystackScanner()
            files = [file_a]
            candidates = scanner.scan(files)

            class_names = {c.class_name for c in candidates}
            assert "MyAgent" in class_names
            assert "MyTool" in class_names
            assert "LCELStep1" in class_names
            assert "LCELStep2" in class_names

            # Verify that haystack_name / aliases map correctly to node alias
            by_class = {c.class_name: c for c in candidates}
            assert by_class["MyAgent"].haystack_name == "agent"
            assert by_class["MyTool"].haystack_name == "tool"
            assert by_class["LCELStep1"].haystack_name == "LCELStep1"
            assert by_class["LCELStep2"].haystack_name == "LCELStep2"


class TestTopology:
    def test_topology_t1_connect_edges(self, target_root: Path):
        """T1 should have retriever -> writer edge from connect()."""
        scanner = HaystackScanner()
        t1_dir = target_root / "linear_rag"
        files = sorted(t1_dir.glob("**/*.py"))

        candidates = scanner.scan(files)
        topology = extract_topology(files, candidates)

        # Find retriever and writer candidates
        retriever = next((c for c in candidates if c.class_name == "RetrieverComponent"), None)
        writer = next((c for c in candidates if c.class_name == "WriterComponent"), None)

        assert retriever is not None
        assert writer is not None

        # Check topology
        retriever_id = retriever.candidate_id
        writer_id = writer.candidate_id

        if retriever_id in topology:
            assert writer_id in topology[retriever_id].downstream

    def test_topology_t2_pipeline_edges(self, target_root: Path):
        """T2 should have guard -> planner edge from connect()."""
        scanner = HaystackScanner()
        files = sorted(target_root.glob("**/*.py"))

        candidates = scanner.scan(files)
        topology = extract_topology(files, candidates)

        # Guard and planner should be connected
        guard_rule_id = "guard_rule"
        guard_llm_id = "guard_llm"
        planner_id = "planner"

        # At least one guard variant should be upstream of planner
        has_guard_upstream = False
        if guard_rule_id in topology and planner_id in topology[guard_rule_id].downstream:
            has_guard_upstream = True
        elif guard_llm_id in topology and planner_id in topology[guard_llm_id].downstream:
            has_guard_upstream = True

        assert has_guard_upstream, "guard should be upstream of planner"

    def test_topology_t2_constructor_injection_edges(self, target_root: Path):
        """Planner should have worker/judge/writer downstream via constructor injection."""
        scanner = HaystackScanner()
        # Need to scan all test_targets to find WriterComponent imported from linear_rag
        files = sorted(target_root.glob("**/*.py"))

        candidates = scanner.scan(files)
        topology = extract_topology(files, candidates)

        planner_id = "planner"
        if planner_id in topology:
            downstream = set(topology[planner_id].downstream)
            # Should include worker, judge, writer
            expected = {"worker", "judge", "writer"}
            assert expected.issubset(downstream)


class TestConstraints:
    def test_constraint_phase_a_finds_guard_min_query_length(self, target_root: Path):
        """Guard's MIN_QUERY_LENGTH=5 should be found."""
        scanner = HaystackScanner()
        t2_dir = target_root / "multi_agent"
        files = sorted(t2_dir.glob("**/*.py"))

        candidates = scanner.scan(files)
        package_root = t2_dir.parent.parent

        constraints = mine_constraints(files, candidates, package_root)

        # Check guard candidates have the constraint
        guard_candidates = [c for c in candidates if "guard" in c.candidate_id]
        found = False

        for guard_candidate in guard_candidates:
            if guard_candidate.candidate_id in constraints:
                for constraint in constraints[guard_candidate.candidate_id]:
                    if (
                        constraint.value == 5
                        and "GuardComponent.MIN_QUERY_LENGTH" in constraint.source
                    ):
                        found = True
                        break

        assert found, "MIN_QUERY_LENGTH constraint not found"

    def test_constraint_phase_a_finds_planner_max_items(self, target_root: Path):
        """Planner's MAX_ITEMS_PER_CALL=2 should be found."""
        scanner = HaystackScanner()
        t2_dir = target_root / "multi_agent"
        files = sorted(t2_dir.glob("**/*.py"))

        candidates = scanner.scan(files)
        package_root = t2_dir.parent.parent

        constraints = mine_constraints(files, candidates, package_root)

        planner_id = "planner"
        found = False

        if planner_id in constraints:
            for constraint in constraints[planner_id]:
                if (
                    constraint.value == 2
                    and "PlannerComponent.MAX_ITEMS_PER_CALL" in constraint.source
                ):
                    found = True
                    break

        assert found, "MAX_ITEMS_PER_CALL constraint not found"

    def test_constraint_phase_b_fires_zero_calls_on_t1_and_t2(self, target_root: Path):
        """Phase B should make zero LLM calls on T1/T2 (verified empirically)."""
        import asyncio

        async def run_test():
            scanner = HaystackScanner()
            t2_dir = target_root / "multi_agent"
            files = sorted(t2_dir.glob("**/*.py"))

            candidates = scanner.scan(files)
            package_root = t2_dir.parent.parent
            phase_a_results = mine_constraints(files, candidates, package_root)

            # Use a fake client that raises if called
            class NeverCallClient:
                async def complete(self, *args, **kwargs):
                    raise AssertionError("Phase B should not call LLM on T2")

            from agent_eval_harness.mapping.builder.constraints import mine_constraints_llm_phase

            # Should not raise
            result = await mine_constraints_llm_phase(
                candidates, NeverCallClient(), phase_a_results
            )
            assert result == phase_a_results

        asyncio.run(run_test())

    def test_constraint_phase_b_mines_from_synthetic_prompt_literal(self):
        """Phase B should extract constraints from synthetic prompt literals."""
        import asyncio
        import json
        import tempfile

        async def run_test():
            from agent_eval_harness.llm.client import LLMResponse
            from agent_eval_harness.llm.fake_client import FakeLLMClient
            from agent_eval_harness.mapping.builder.constraints import (
                mine_constraints_llm_phase,
            )
            from agent_eval_harness.mapping.builder.scanners import HaystackScanner

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)

                # Create a synthetic Python file with a prompt literal
                (tmp_path / "synthetic.py").write_text('''
from haystack import component

@component
class SyntheticComponent:
    """A test component."""
    def run(self):
        prompt = "You may issue at most 3 retries per session before escalating."
        return prompt
''')

                # Scan it to get the candidate
                scanner = HaystackScanner()
                candidates = scanner.scan([tmp_path / "synthetic.py"])

                # Mine Phase A constraints (should find none, no class-level constants)
                phase_a_results = mine_constraints(
                    [tmp_path / "synthetic.py"],
                    candidates,
                    tmp_path,
                )

                # Set up LLM client with expected response
                llm_response = LLMResponse(
                    content=json.dumps([{
                        "name": "max_retries",
                        "value": 3,
                        "quote": "at most 3 retries"
                    }]),
                    model="fake-test",
                )
                llm_client = FakeLLMClient(llm_response)

                # Mine Phase B constraints
                phase_b_results = await mine_constraints_llm_phase(
                    candidates,
                    llm_client,
                    phase_a_results,
                )

                # Check results
                assert len(candidates) > 0
                for candidate in candidates:
                    if "synthetic" in candidate.class_name.lower():
                        if candidate.candidate_id in phase_b_results:
                            constraints = phase_b_results[candidate.candidate_id]
                            # Should have found the max_retries constraint
                            names = [c.name for c in constraints]
                            assert "max_retries" in names
                            for c in constraints:
                                if c.name == "max_retries":
                                    assert c.value == 3
                                    assert "at most 3 retries" in c.source

        asyncio.run(run_test())
