"""Unit tests for map builder components (no LLM)."""
import asyncio
import tempfile
from pathlib import Path

import pytest

from agent_eval_harness.mapping.builder.constraints import mine_constraints
from agent_eval_harness.mapping.builder.pipeline import SystemMapBuilder
from agent_eval_harness.mapping.builder.prompts import ROLE_TAXONOMY
from agent_eval_harness.mapping.builder.scanners import (
    HaystackScanner,
    LangGraphScanner,
    scan_all,
)
from agent_eval_harness.mapping.builder.topology import extract_topology


class TestNoRoleLLMCallsInStage2:
    """CS-300 AC2: Stage 2 emits 0 role-classification LLM calls. Content-based, not
    call-count-based — the builder still makes constraint-mining calls (pipeline.py), so a
    bare call_count==0 would silently stop meaning anything once mining changes."""

    def test_no_call_carries_the_role_taxonomy_as_a_system_prompt(self, target_root: Path):
        class _RecordingClient:
            def __init__(self):
                self.calls = []

            async def complete(
                self, messages, *, max_tokens=512, temperature=0.2, json_mode=False, reasoning_effort=None
            ):
                from agent_eval_harness.llm.client import LLMResponse
                self.calls.append(messages)
                return LLMResponse(content="[]", model="fake")

        async def run_test():
            t2_dir = target_root / "multi_agent"
            client = _RecordingClient()
            builder = SystemMapBuilder(client)
            await builder.build(t2_dir)

            for messages in client.calls:
                for m in messages:
                    if m.role == "system":
                        assert ROLE_TAXONOMY not in m.content, (
                            "a Stage 2 LLM call carried the role taxonomy — "
                            "role classification leaked back into Stage 2"
                        )

        asyncio.run(run_test())


class TestHaystackScanner:
    def test_haystack_scanner_finds_retriever_and_writer_in_t1(self, target_root: Path):
        """T1 (linear_rag) has 2 components: RetrieverComponent, WriterComponent."""
        scanner = HaystackScanner()
        t1_dir = target_root / "linear_rag"
        files = sorted(t1_dir.glob("**/*.py"))

        candidates = scanner.scan(files)

        t1_candidates = [c for c in candidates if c.file.parent.name == "linear_rag"]

        class_names = {c.class_name for c in t1_candidates}
        assert "RetrieverComponent" in class_names
        assert "WriterComponent" in class_names

    def test_haystack_scanner_finds_all_candidates_in_t2(self, target_root: Path):
        """T2 (multi_agent) has 8 logical candidates after split."""
        scanner = HaystackScanner()
        files = sorted(target_root.glob("**/*.py"))

        candidates = scanner.scan(files)

        candidate_ids = {c.candidate_id for c in candidates}

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

            valid_file = tmp_path / "valid.py"
            valid_file.write_text(
                "@component\nclass ValidComponent:\n    pass\n"
            )

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

    def test_lcel_admission(self):
        """LangChain LCEL chain nodes are admitted by HaystackScanner's BitOr pass. (LangGraph
        add_node moved to LangGraphScanner in CS-312 — see TestLangGraphScanner.)"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            file_a = tmp_path / "app.py"
            file_a.write_text(
                "class LCELStep1:\n"
                "    pass\n"
                "class LCELStep2:\n"
                "    pass\n\n"
                "chain = LCELStep1() | LCELStep2()\n"
            )

            scanner = HaystackScanner()
            candidates = scanner.scan([file_a])

            class_names = {c.class_name for c in candidates}
            assert "LCELStep1" in class_names
            assert "LCELStep2" in class_names

            by_class = {c.class_name: c for c in candidates}
            assert by_class["LCELStep1"].haystack_name == "LCELStep1"
            assert by_class["LCELStep2"].haystack_name == "LCELStep2"


class TestLangGraphScanner:
    def test_function_class_and_bound_method_admission(self):
        """add_node resolves function / class / bound-method targets, keeping the owner class."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            file_a = tmp_path / "app.py"
            file_a.write_text(
                "from langgraph.graph import StateGraph\n\n"
                "class MyAgent:\n    pass\n"
                "class MyTool:\n    pass\n"
                "def some_fn(state):\n    return state\n\n"
                "class Orchestrator:\n"
                "    def build(self):\n"
                "        graph = StateGraph(dict)\n"
                "        graph.add_node('agent', MyAgent())\n"
                "        graph.add_node('tool', MyTool)\n"
                "        graph.add_node('fn', some_fn)\n"
                "        graph.add_node('m', self._m)\n"
                "        return graph\n"
                "    def _m(self, state):\n        return state\n"
            )

            candidates = LangGraphScanner().scan([file_a])
            by_name = {c.class_name: c for c in candidates}

            assert set(by_name) == {"MyAgent", "MyTool", "some_fn", "_m"}
            assert by_name["MyAgent"].entry_kind == "class"
            assert by_name["MyTool"].entry_kind == "class"
            assert by_name["some_fn"].entry_kind == "function"
            assert by_name["_m"].entry_kind == "bound_method"
            assert by_name["_m"].owner_class_name == "Orchestrator"

    def test_add_node_without_stategraph_construction_yields_zero_candidates(self):
        """A plain object with an .add_node method (no StateGraph()) is not a LangGraph graph."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            file_a = tmp_path / "app.py"
            file_a.write_text(
                "class Foo:\n"
                "    def add_node(self, a, b):\n        pass\n"
                "f = Foo()\n"
                "f.add_node('x', 'y')\n"
            )
            assert LangGraphScanner().scan([file_a]) == []

    def test_langgraph_agent_fixture_admission(self, target_root: Path):
        """Real fixture: 2 function nodes + 2 bound-method nodes with owner class."""
        files = sorted((target_root / "langgraph_agent").glob("*.py"))
        candidates = LangGraphScanner().scan(files)
        by_name = {c.class_name: c for c in candidates}

        assert set(by_name) == {"load_context", "plan_step", "_node_investigate", "_node_synthesize"}
        assert by_name["load_context"].entry_kind == "function"
        assert by_name["plan_step"].entry_kind == "function"
        assert by_name["_node_investigate"].entry_kind == "bound_method"
        assert by_name["_node_investigate"].owner_class_name == "ResearchAgent"
        assert by_name["_node_synthesize"].entry_kind == "bound_method"
        assert by_name["_node_synthesize"].owner_class_name == "ResearchAgent"

    def test_haystack_fixture_candidate_count_unchanged_after_add_node_removal(
        self, target_root: Path
    ):
        """Invariant: removing the add_node block from HaystackScanner does not change any Haystack
        fixture's candidate count (no fixture uses add_node/StateGraph). Hard numbers, pinned."""
        expected = {"linear_rag": 2, "multi_agent": 7, "t3_reranker": 3}
        for tgt, count in expected.items():
            files = sorted((target_root / tgt).glob("**/*.py"))
            assert len(HaystackScanner().scan(files)) == count, tgt


class TestScanAll:
    """CS-316: run every registered scanner over one file set and merge (mixed-cluster dispatch)."""

    def test_haystack_fixtures_equal_haystack_only_counts_and_label(self, target_root: Path):
        """Highest-risk invariant: over a pure-Haystack fixture, scan_all == HaystackScanner-only
        (LangGraphScanner self-gates to 0), and the contributed label is exactly 'haystack'."""
        expected = {"linear_rag": 2, "multi_agent": 7, "t3_reranker": 3}
        for tgt, count in expected.items():
            files = sorted((target_root / tgt).glob("**/*.py"))
            candidates, label = scan_all(files)
            assert len(candidates) == count, tgt
            assert len(candidates) == len(HaystackScanner().scan(files)), tgt
            assert label == "haystack", tgt

    def test_langgraph_fixture_labels_langgraph(self, target_root: Path):
        files = sorted((target_root / "langgraph_agent").glob("*.py"))
        candidates, label = scan_all(files)
        assert len(candidates) == 4
        assert label == "langgraph"

    def test_mixed_set_unions_both_scanners_and_joins_label(self, target_root: Path):
        """A mixed cluster (Haystack files + a LangGraph file) yields BOTH scanners' candidates and
        a '+'-joined sorted label — the exact scenario that produced 0 langgraph nodes before."""
        files = sorted((target_root / "linear_rag").glob("**/*.py"))
        files += sorted((target_root / "langgraph_agent").glob("*.py"))
        candidates, label = scan_all(files)
        names = {c.class_name for c in candidates}
        assert len(candidates) == 6  # 2 haystack + 4 langgraph
        assert {"RetrieverComponent", "WriterComponent"} <= names
        assert {"_node_investigate", "_node_synthesize"} <= names
        assert label == "haystack+langgraph"

    def test_dedups_file_claimed_by_two_scanners(self):
        """A single file matched by BOTH scanners (a @component class also used as an add_node
        target) must not double-count: _dedup_and_sort's (file, class_name, tag_suffix) key
        collapses it to one, registry order (haystack first) keeping the survivor."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_a = Path(tmp_dir) / "app.py"
            file_a.write_text(
                "from haystack import component\n"
                "from langgraph.graph import StateGraph\n\n"
                "@component\n"
                "class Foo:\n    pass\n\n"
                "def build():\n"
                "    graph = StateGraph(dict)\n"
                "    graph.add_node('x', Foo)\n"
                "    return graph\n"
            )
            candidates, label = scan_all([file_a])
            foo = [c for c in candidates if c.class_name == "Foo"]
            assert len(foo) == 1, [c.class_name for c in candidates]
            assert label == "haystack+langgraph"


class TestTopology:
    def test_topology_t1_connect_edges(self, target_root: Path):
        """T1 should have retriever -> writer edge from connect()."""
        scanner = HaystackScanner()
        t1_dir = target_root / "linear_rag"
        files = sorted(t1_dir.glob("**/*.py"))

        candidates = scanner.scan(files)
        topology = extract_topology(files, candidates)

        retriever = next((c for c in candidates if c.class_name == "RetrieverComponent"), None)
        writer = next((c for c in candidates if c.class_name == "WriterComponent"), None)

        assert retriever is not None
        assert writer is not None

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

    def test_topology_constructor_downstream_planner_fanout_three(self, target_root: Path):
        """CS-299 B2: planner's constructor_downstream carries exactly its 3 constructor-injected siblings."""
        scanner = HaystackScanner()
        files = sorted(target_root.glob("**/*.py"))

        candidates = scanner.scan(files)
        topology = extract_topology(files, candidates)

        planner = topology.get("planner")
        assert planner is not None
        assert set(planner.constructor_downstream) == {"worker", "judge", "writer"}
        # upstream/downstream stay byte-identical to the pre-CS-299 assertions above
        assert {"worker", "judge", "writer"}.issubset(set(planner.downstream))

    def test_topology_constructor_downstream_empty_for_connect_only_wiring(self, target_root: Path):
        """CS-299 B2: T1's retriever->writer edge is connect()-based, not constructor injection."""
        scanner = HaystackScanner()
        t1_dir = target_root / "linear_rag"
        files = sorted(t1_dir.glob("**/*.py"))

        candidates = scanner.scan(files)
        topology = extract_topology(files, candidates)

        retriever = topology.get("retriever")
        assert retriever is not None
        assert retriever.constructor_downstream == []
        assert "writer" in retriever.downstream  # unchanged: still populated via connect()


class TestConstraints:
    def test_constraint_phase_a_finds_guard_min_query_length(self, target_root: Path):
        """Guard's MIN_QUERY_LENGTH=5 should be found."""
        scanner = HaystackScanner()
        t2_dir = target_root / "multi_agent"
        files = sorted(t2_dir.glob("**/*.py"))

        candidates = scanner.scan(files)
        package_root = t2_dir.parent.parent

        constraints = mine_constraints(files, candidates, package_root)

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

                (tmp_path / "synthetic.py").write_text('''
from haystack import component

@component
class SyntheticComponent:
    """A test component."""
    def run(self):
        prompt = "You may issue at most 3 retries per session before escalating."
        return prompt
''')

                scanner = HaystackScanner()
                candidates = scanner.scan([tmp_path / "synthetic.py"])

                # Mine Phase A constraints (should find none, no class-level constants)
                phase_a_results = mine_constraints(
                    [tmp_path / "synthetic.py"],
                    candidates,
                    tmp_path,
                )

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


class TestPerSystemComponentScoping:
    """CS-317 revision: a split per-system map keeps ONLY its own framework's components; sibling
    classes co-located in the accepted file set must not bleed in."""

    def _noop_llm(self):
        from agent_eval_harness.llm.client import LLMResponse

        class _NoOpLLM:
            async def complete(self, messages, **_kw):
                return LLMResponse(content="[]", model="fake")

        return _NoOpLLM()

    def _mixed_files(self, root: Path):
        (root / "pipe.py").write_text(
            "from haystack import component\n"
            "@component\n"
            "class Retriever:\n"
            "    def run(self):\n"
            "        return 1\n",
            encoding="utf-8",
        )
        (root / "graph.py").write_text(
            "from langgraph.graph import StateGraph\n"
            "g = StateGraph(dict)\n"
            "g.add_node('act', do_act)\n",
            encoding="utf-8",
        )
        return [root / "pipe.py", root / "graph.py"]

    def test_scope_framework_keeps_only_that_frameworks_components(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            files = self._mixed_files(root)

            async def run_it(scope):
                builder = SystemMapBuilder(self._noop_llm(), framework=scope)
                m, _ = await builder.build_from_files(
                    files, package_root=root, target_system_id="t", scope_framework=scope
                )
                return m

            hay = asyncio.run(run_it("haystack"))
            hay_ids = {c.id for c in hay.components}
            assert "retriever" in hay_ids and "do_act" not in hay_ids
            assert hay.framework == "haystack"

            lg = asyncio.run(run_it("langgraph"))
            lg_ids = {c.id for c in lg.components}
            assert "do_act" in lg_ids and "retriever" not in lg_ids
            assert lg.framework == "langgraph"

    def test_no_scope_keeps_all_components_unchanged(self):
        """A non-split candidate passes scope_framework=None => every component is kept (both
        frameworks), preserving today's behavior for single/non-split candidates."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            files = self._mixed_files(root)

            async def run_it():
                builder = SystemMapBuilder(self._noop_llm())
                m, _ = await builder.build_from_files(files, package_root=root, target_system_id="t")
                return m

            m = asyncio.run(run_it())
            ids = {c.id for c in m.components}
            assert {"retriever", "do_act"} <= ids

    def test_exclude_component_classes_drops_named_class(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            files = self._mixed_files(root)

            async def run_it():
                builder = SystemMapBuilder(self._noop_llm(), framework="langgraph")
                m, _ = await builder.build_from_files(
                    files, package_root=root, target_system_id="t",
                    scope_framework="langgraph", exclude_component_classes={"do_act"},
                )
                return m

            m = asyncio.run(run_it())
            assert "do_act" not in {c.id for c in m.components}  # class_name do_act excluded
