"""Mapping pipeline tests: map builder (golden/robustness), span-mapping engine, LCEL/plain-python
scanners, and role-gate contracts."""
from __future__ import annotations

import asyncio
import json
import re
import tempfile
from pathlib import Path

import pytest

from agent_eval_harness.discovery.enrichment import enrich_agents
from agent_eval_harness.discovery.wiring import (
    WiringBlock,
    WiringNode,
    detect_wiring_block_static,
)
from agent_eval_harness.instrumentation.base import CapturedSpan
from agent_eval_harness.llm.client import LLMResponse
from agent_eval_harness.llm.fake_client import FakeLLMClient
from agent_eval_harness.mapping.agent_flow import AgentFlow, AgentFlowMap
from agent_eval_harness.mapping.builder.lcel_scanner import LCELScanner
from agent_eval_harness.mapping.builder.pipeline import SystemMapBuilder
from agent_eval_harness.mapping.builder.plain_python_scanner import PlainPythonScanner
from agent_eval_harness.mapping.builder.roles import (
    VALID_ROLES,
    admissible_roles,
    structural_facts,
)
from agent_eval_harness.mapping.builder.scanners import scan_all
from agent_eval_harness.mapping.engine import map_spans_to_components
from agent_eval_harness.mapping.system_map import (
    Component,
    SpanMatchBlock,
    SystemMap,
    load_system_map,
)
from tests._stubs import KeyedFakeLLMClient


# ---------------------------------------------------------------------------
# test_map_builder_golden.py: golden tests for map builder against T1/T2 targets
# ---------------------------------------------------------------------------

class _NeverCallClient:
    """SystemMapBuilder.build() must never call the LLM for T1/T2 (CS-300 moved role classification to Stage 2.5)."""

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
        """T2 golden test: 9 logical components — GuardComponent unsplit widens Haystack's guard_rule/guard_llm to a 9th; role is 'unknown' pending Stage 2.5 (CS-300)."""

        async def run_test():
            t2_dir = target_root / "multi_agent"
            builder = SystemMapBuilder(_NeverCallClient())
            system_map, summary = await builder.build(t2_dir)

            # 9 components: guard_rule, guard_llm, guard, planner, worker, judge, writer, 2 tools
            assert len(system_map.components) == 9
            assert all(c.role == "unknown" for c in system_map.components)

            ids = {c.id for c in system_map.components}
            expected_ids = {
                "guard_rule", "guard_llm", "guard", "planner", "worker",
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

            guard_candidates = [c for c in system_map.components if "guard" in c.id]
            planner = system_map.component_by_id("planner")

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

            assert "=== AEH System Map Summary ===" in summary
            assert "target:" in summary
            assert "components_found:" in summary
            assert "unknown:" in summary
            assert "discrepancies:" in summary

        asyncio.run(run_test())


# ---------------------------------------------------------------------------
# test_map_builder_robustness.py: robustness and negative-control tests for map builder
# ---------------------------------------------------------------------------

class TestRobustness:
    def test_stripped_t2_degrades_toward_unknown(self, target_root: Path):
        """Stripped docstrings should degrade confidence, resulting in 'unknown' roles."""
        import ast
        import asyncio

        async def run_test():
            t2_dir = target_root / "multi_agent"

            # Copy T2 to a temp dir and strip docstrings
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_t2 = Path(tmp_dir) / "multi_agent"
                tmp_t2.mkdir()

                # Copy all Python files and strip docstrings
                for py_file in t2_dir.glob("**/*.py"):
                    rel_path = py_file.relative_to(t2_dir)
                    dest_file = tmp_t2 / rel_path
                    dest_file.parent.mkdir(parents=True, exist_ok=True)

                    source = py_file.read_text(encoding="utf-8")
                    tree = ast.parse(source)

                    # Strip docstrings by setting them to empty strings
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            if (
                                node.body
                                and isinstance(node.body[0], ast.Expr)
                                and isinstance(node.body[0].value, ast.Constant)
                            ):
                                node.body[0].value.value = ""

                    dest_file.write_text(ast.unparse(tree), encoding="utf-8")

                default_response = LLMResponse(
                    content=json.dumps({
                        "role": "unknown",
                        "confidence": 0.5,
                        "reasoning": "Low confidence due to stripped docs."
                    }),
                    model="fake-gpt",
                )

                builder = SystemMapBuilder(KeyedFakeLLMClient({}, default=default_response))
                system_map, summary = await builder.build(tmp_t2)

                # All components should be classified as unknown (confidence < 0.7 threshold)
                for component in system_map.components:
                    assert component.role == "unknown", (
                        f"Component {component.id} should be 'unknown'"
                    )

        asyncio.run(run_test())

    def test_stripped_t2_no_confident_misclassification(self, target_root: Path):
        """Stripped T2 should not have confident wrong classifications."""
        import asyncio

        async def run_test():
            t2_dir = target_root / "multi_agent"

            # Use a default response with confidence below threshold
            default_response = LLMResponse(
                content=json.dumps({
                    "role": "tool",  # Intentionally wrong
                    "confidence": 0.5,
                    "reasoning": "Misguessed, but low confidence"
                }),
                model="fake-gpt",
            )

            builder = SystemMapBuilder(KeyedFakeLLMClient({}, default=default_response))
            system_map, summary = await builder.build(t2_dir)

            # Below-threshold confidence must degrade to "unknown", never keep the wrong guess
            for component in system_map.components:
                assert component.role == "unknown", (
                    f"Component {component.id} should be 'unknown', not a confident wrong guess"
                )

        asyncio.run(run_test())

    def test_negative_control_store_module_empty_map(self, target_root: Path):
        """Non-agentic code (store module) should produce empty map with special message."""
        import asyncio

        async def run_test():
            store_dir = target_root.parent / "agent_eval_harness" / "store"
            if not store_dir.exists():
                pytest.skip("Store module not found")

            # Use a dummy LLM client (should never be called)
            default_response = LLMResponse(
                content=json.dumps({"role": "unknown", "confidence": 0.0, "reasoning": ""}),
                model="fake-gpt",
            )

            builder = SystemMapBuilder(KeyedFakeLLMClient({}, default=default_response))
            system_map, summary = await builder.build(store_dir)

            # Should have no components
            assert len(system_map.components) == 0

            # Summary should contain the special message
            assert "no agentic components identified" in summary

        asyncio.run(run_test())


# ---------------------------------------------------------------------------
# test_mapping_engine.py: unit tests for the span -> component mapping engine (pure, no I/O)
# ---------------------------------------------------------------------------

_T1_MAP_PATH = Path(__file__).parent.parent / "test_targets" / "linear_rag" / "system_map.yaml"
_RENAMED_MAP_PATH = Path(__file__).parent / "fixtures" / "linear_rag_system_map_renamed.yaml"


def _span(span_id: str, component_name: str, tags: dict | None = None) -> CapturedSpan:
    return CapturedSpan(
        span_id=span_id,
        parent_span_id=None,
        operation_name="haystack.component.run",
        span_type="agent",
        component_name=component_name,
        tags=tags or {},
    )


def test_exact_match_by_component_name() -> None:
    system_map = load_system_map(_T1_MAP_PATH)
    spans = [_span("s1", "retriever"), _span("s2", "writer")]

    result = map_spans_to_components(spans, system_map)

    assert result.unmatched == []
    assert result.ambiguous == []
    assigned = {span.span_id: cid for span, cid in result.matched}
    assert assigned == {"s1": "retriever", "s2": "writer"}


def test_unmatched_span_kept_and_counted_not_dropped() -> None:
    system_map = load_system_map(_T1_MAP_PATH)
    span = _span("s1", "mystery_component")

    result = map_spans_to_components([span], system_map)

    assert len(result.unmatched) == 1
    assert result.unmatched[0] is span
    assert span.component_id is None


def test_ambiguous_match_assigns_first_and_records_all_candidates() -> None:
    system_map = SystemMap(
        target_system_id="synthetic",
        components=[
            Component(
                id="a", role="agent", entry_point="x:y",
                span_match=[SpanMatchBlock(component_name="dup")],
            ),
            Component(
                id="b", role="agent", entry_point="x:y",
                span_match=[SpanMatchBlock(component_name="dup")],
            ),
        ],
    )
    span = _span("s1", "dup")

    result = map_spans_to_components([span], system_map)

    assert span.component_id == "a"
    assert len(result.ambiguous) == 1
    assert result.ambiguous[0][1] == ["a", "b"]


def test_rename_component_id_changes_mapping_with_zero_code_edits() -> None:
    """Renaming a component id in the system map changes the mapping outcome with zero code edits on the span-emitting side."""
    original_map = load_system_map(_T1_MAP_PATH)
    renamed_map = load_system_map(_RENAMED_MAP_PATH)

    span_against_original = _span("s1", "writer")
    span_against_renamed = _span("s1", "writer")

    map_spans_to_components([span_against_original], original_map)
    map_spans_to_components([span_against_renamed], renamed_map)

    assert span_against_original.component_id == "writer"
    assert span_against_renamed.component_id == "responder"


def test_tags_based_match() -> None:
    system_map = SystemMap(
        target_system_id="synthetic",
        components=[
            Component(
                id="tool_x", role="tool", entry_point="x:y",
                span_match=[SpanMatchBlock(tags={"aeh.tool.name": "search"})],
            ),
        ],
    )
    span = _span("s1", "worker", tags={"aeh.tool.name": "search"})

    result = map_spans_to_components([span], system_map)

    assert span.component_id == "tool_x"
    assert result.unmatched == []


# ---------------------------------------------------------------------------
# test_lcel_scanner.py: LCELScanner — pipe idiom (fixture) + factory idiom (real LangChain dogfood)
# ---------------------------------------------------------------------------

_LCEL_FIXTURE = Path(__file__).parent.parent / "test_targets" / "lcel_chain" / "chain.py"
# Real LangChain conversational-RAG app cloned for the factory-idiom dogfood (external target).
_CLONE = Path(
    r"D:/Program Files (x86)/Python VS Code/test_repo/conversational-rag-chatbot"
)


def _by_name_lcel(cands):
    return {c.class_name: c for c in cands}


def test_pipe_idiom_library_objects_degrade_explicitly():
    """Every pure-library link degrades explicitly (is_library_object=True) instead of vanishing."""
    cands = _by_name_lcel(LCELScanner().scan([_LCEL_FIXTURE]))
    for lib in ("ChatOpenAI", "StrOutputParser"):
        assert lib in cands, f"library link {lib} not surfaced"
        assert cands[lib].is_library_object is True


def test_pipe_idiom_runnable_lambda_unwraps_to_user_function():
    """RunnableLambda(postprocess) must surface the wrapped USER function as harvestable, not the wrapper."""
    cands = _by_name_lcel(LCELScanner().scan([_LCEL_FIXTURE]))
    assert "postprocess" in cands
    assert cands["postprocess"].is_library_object is False
    assert cands["postprocess"].entry_kind == "function"
    assert "RunnableLambda" not in cands


def test_pipe_idiom_guard_rejects_type_and_flag_unions():
    """The PEP-604 / flag-enum shapes in the same file must not become candidates."""
    cands = _by_name_lcel(LCELScanner().scan([_LCEL_FIXTURE]))
    assert "RetryPolicy" not in cands
    assert "backoff_strategy" not in cands
    assert "_verbose_levels" not in cands


@pytest.mark.skipif(not _CLONE.exists(), reason="cloned LangChain app not present on this machine")
def test_factory_idiom_dogfood_maps_real_chain_graph():
    """Dogfood on a REAL LangChain app built entirely with the factory-function idiom (zero `|`) — proves the dominant production idiom is mapped, not a strawman."""
    files = list(_CLONE.glob("**/*.py"))
    cands = _by_name_lcel(LCELScanner().scan(files))

    expected = {"retriever", "history_aware_retriever", "question_answer_chain", "rag_chain"}
    found = expected & set(cands)
    # Report measured coverage (the acceptance is the real graph, not a fixture).
    assert found == expected, f"factory-idiom coverage {len(found)}/{len(expected)}: missing {expected - found}"
    for name in expected:
        assert cands[name].is_library_object is False
        assert cands[name].entry_kind == "function"


class TestLCELGenericCallClosure:
    """Generic call-closure coverage (call_downstream + closure-completeness) extended from haystack/plain_python to LCEL candidates, so CS-324's False verdict can fire here too."""

    async def test_lcel_component_gets_call_downstream_and_non_hardwired_closure_bit(self):
        target_root = _LCEL_FIXTURE.parent.parent
        package_root = target_root.parent
        files = sorted(_LCEL_FIXTURE.parent.glob("*.py"))
        file_contents = {str(f): f.read_text(encoding="utf-8") for f in files}
        wb = detect_wiring_block_static(file_contents)
        builder = SystemMapBuilder(
            FakeLLMClient(LLMResponse(content="[]", model="fake")),
            framework=(wb.framework if wb else None),
        )
        system_map, _ = await builder.build_from_files(
            files, package_root=package_root, target_system_id="lcel_chain",
            wiring_block=wb, retrieval_client=None, snapshot_id=None,
        )
        by_id = {c.id: c for c in system_map.components}

        finalize = by_id["finalize"]
        assert finalize.call_downstream, "expected an LCEL component with non-empty call_downstream"
        assert finalize.closure_complete is True

        # text: str is a typed builtin receiver, so .strip() is a validated out-of-scope call, not an unresolved seam.
        assert by_id["postprocess"].closure_complete is True


# ---------------------------------------------------------------------------
# test_plain_python_scanner.py: signal-based detection — fixture, real ask-mode dogfood, Haystack
# precision floor, wiring_block llm_fallback path
# ---------------------------------------------------------------------------

_PLAIN_AGENT_FIXTURE = Path(__file__).parent.parent / "test_targets" / "plain_agent" / "plain_agent.py"
_TARGETS = Path(__file__).parent.parent / "test_targets"
# Real ask-mode agent (provider-abstraction shape) — the one real plain-python target we have.
_QA_AGENT = Path(
    r"D:/Program Files (x86)/Python VS Code/CodeSpectra/backend/domain/qa/agent.py"
)


def _by_name_plain_python(cands):
    return {c.class_name: c for c in cands}


def test_provider_abstraction_agent_detected():
    """A `*Agent` class corroborated by an entry method is emitted, even with NO raw SDK call."""
    cands = _by_name_plain_python(PlainPythonScanner().scan([_PLAIN_AGENT_FIXTURE]))
    assert "ResearchAgent" in cands
    assert cands["ResearchAgent"].entry_kind == "class"


def test_bare_agent_class_without_corroborator_rejected():
    """Precision guard: a `*Agent` class with no entry method / LLM call / prompt ref is NOT emitted."""
    cands = _by_name_plain_python(PlainPythonScanner().scan([_PLAIN_AGENT_FIXTURE]))
    assert "DisabledAgent" not in cands


def test_class_with_no_english_agent_suffix_is_emitted():
    """Emission is structural, not name-derived — a class with no 'Agent' suffix but the same shape must still be emitted."""
    cands = _by_name_plain_python(PlainPythonScanner().scan([_PLAIN_AGENT_FIXTURE]))
    assert "TroLyNghienCuu" in cands
    assert cands["TroLyNghienCuu"].entry_kind == "class"


def test_class_snippet_does_not_bleed_into_the_next_class():
    """Regression (CS-323, ported here for plain-python): a class's snippet must end at its own end_lineno, never bleed into the next class."""
    cands = _by_name_plain_python(PlainPythonScanner().scan([_PLAIN_AGENT_FIXTURE]))
    tiny = cands["TinyAgent"]
    assert "LEAK_MARKER_must_not_appear_in_tiny_agent_snippet" not in tiny.source_snippet
    assert "class LeakGuardAgent" not in tiny.source_snippet


def test_raw_sdk_tool_list_secondary_heuristic():
    """SECONDARY: a tools=[...] list of module-level functions surfaces tool components."""
    cands = _by_name_plain_python(PlainPythonScanner().scan([_PLAIN_AGENT_FIXTURE]))
    for tool in ("tool_search", "tool_summarize"):
        assert tool in cands
        assert cands[tool].is_tool is True
        assert cands[tool].owner_class_name == "ToolAgent"


def test_wiring_block_llm_fallback_path():
    """The wiring_block(llm_fallback) escape hatch resolves AST-missed names to real defs and dedups against AST hits (no doubles)."""
    wb = WiringBlock(
        nodes=[
            WiringNode(alias="a", callee_name="ResearchAgent", source_hint_file="plain_agent.py"),
            WiringNode(alias="s", callee_name="tool_search", source_hint_file="plain_agent.py"),
        ],
        edges=[],
        framework="llm_inferred",
        source="llm_fallback",
    )
    cands = PlainPythonScanner().scan([_PLAIN_AGENT_FIXTURE], wiring_block=wb)
    names = [c.class_name for c in cands]
    assert names.count("ResearchAgent") == 1  # deduped against the AST-heuristic hit
    assert "tool_search" in names


def test_precision_floor_no_new_candidates_on_haystack_fixtures():
    """Structural widening traded for precision on purpose: multi_agent's unsplit GuardComponent lands as an 8th candidate alongside Haystack's own split guard_rule/guard_llm — narrowing happens at a later verdict layer."""
    expected = {"multi_agent": 8, "linear_rag": 2, "t3_reranker": 3}
    for tgt, count in expected.items():
        files = list((_TARGETS / tgt).glob("**/*.py"))
        cands, _label = scan_all(files)
        assert len(cands) == count, f"{tgt}: expected {count}, got {len(cands)}"


@pytest.mark.skipif(not _QA_AGENT.exists(), reason="backend QA agent not present")
def test_dogfood_real_ask_mode_agent():
    """Dogfood on the REAL ask-mode agent (provider-abstraction shape, no raw SDK/tools list) — the scanner must still surface its class."""
    cands = _by_name_plain_python(PlainPythonScanner().scan([_QA_AGENT]))
    agent_classes = [name for name, c in cands.items() if c.entry_kind == "class" and name.endswith("Agent")]
    assert agent_classes, f"no *Agent class surfaced from the real ask-mode agent; got {list(cands)}"


# ---------------------------------------------------------------------------
# test_role_colors_gate.py: CS-300 AC4 cross-language gate — ROLE_COLORS (TS) must stay
# set-equal to VALID_ROLES (Python); tsc can't see Python, so this is the mechanical catch
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ROLE_COLORS_TSX = _REPO_ROOT / "src" / "renderer" / "src" / "screens" / "aeh" / "AgentSubGraphPanel.tsx"


def _parse_role_colors_keys(path: Path) -> set[str]:
    """Extract the object-literal keys of `export const ROLE_COLORS: Record<AEHRole, string> = {...}`."""
    content = path.read_text(encoding="utf-8")
    match = re.search(r"ROLE_COLORS:\s*Record<AEHRole,\s*string>\s*=\s*\{(.*?)\}", content, re.DOTALL)
    assert match, f"Could not find the ROLE_COLORS object literal in {path}"
    body = match.group(1)

    keys: set[str] = set()
    for line in body.splitlines():
        line = line.strip().rstrip(",")
        if not line or line.startswith("//"):
            continue
        key_part = line.split(":", 1)[0].strip()
        key = key_part[1:-1] if key_part[:1] in ("'", '"') else key_part
        if key:
            keys.add(key)
    return keys


def test_role_colors_keys_are_set_equal_to_valid_roles():
    keys = _parse_role_colors_keys(_ROLE_COLORS_TSX)

    # A silently-empty parse would pass a subset/equality check for the wrong reason — a gate
    # that can silently stop gating is not a gate.
    assert keys, f"Parsed 0 keys from ROLE_COLORS in {_ROLE_COLORS_TSX} — the gate found nothing to check"

    assert keys == VALID_ROLES, (
        f"ROLE_COLORS keys and VALID_ROLES have drifted.\n"
        f"  in ROLE_COLORS but not VALID_ROLES: {sorted(keys - VALID_ROLES)}\n"
        f"  in VALID_ROLES but not ROLE_COLORS: {sorted(VALID_ROLES - keys)}"
    )


def test_aeh_role_union_in_electron_d_ts_is_set_equal_to_valid_roles():
    """Second independent channel: the AEHRole TS union itself must also stay set-equal to VALID_ROLES."""
    electron_d_ts = _REPO_ROOT / "src" / "renderer" / "src" / "types" / "electron.d.ts"
    content = electron_d_ts.read_text(encoding="utf-8")
    match = re.search(r"export type AEHRole =\s*((?:\s*\|\s*'[^']+')+)", content)
    assert match, f"Could not find `export type AEHRole = ...` in {electron_d_ts}"

    roles = set(re.findall(r"'([^']+)'", match.group(1)))
    assert roles, f"Parsed 0 members from the AEHRole union in {electron_d_ts}"
    assert roles == VALID_ROLES, (
        f"AEHRole union and VALID_ROLES have drifted.\n"
        f"  in AEHRole but not VALID_ROLES: {sorted(roles - VALID_ROLES)}\n"
        f"  in VALID_ROLES but not AEHRole: {sorted(VALID_ROLES - roles)}"
    )


# ---------------------------------------------------------------------------
# test_role_structural_gates.py: CS-299 B2 + CS-300 — structure SUBTRACTS impossible roles,
# the LLM only chooses among what remains
# ---------------------------------------------------------------------------

class _EnsureDbMixin:
    """Scopes the DB-readiness autouse fixture to the role-gate classes below only — no file-wide leakage."""

    @pytest.fixture(autouse=True)
    async def _ensure_db(self) -> None:
        """Re-init the DB if an earlier test in this session explicitly closed it."""
        from agent_eval_harness.store.database import get_db, init_db
        try:
            get_db()
        except RuntimeError:
            await init_db()


class TestAdmissibleRoles(_EnsureDbMixin):
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


class TestStructuralFacts(_EnsureDbMixin):
    """Facts are evidence shown to the LLM, never a rule — CS-300 dropped the terminal(fan-out==0) clause after it misfired on 3 infra services."""

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
    """Every component gets the SAME claimed role regardless of evidence — proves the hard gate's subtraction fires, not a fixture that already "knows" the answer."""

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


class TestGeneralizationGate(_EnsureDbMixin):
    def test_orchestrator_only_survives_for_real_constructor_fanout(
        self, target_root: Path, tmp_path: Path, monkeypatch
    ):
        """Every candidate's LLM answer is 'orchestrator' (the real observed B2 failure) — only PlannerComponent (real constructor_fanout=3, no docstring) may keep it."""
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
        """Every candidate's LLM answer is 'tool' — only the dict-registered async functions (real is_tool=True) may keep it."""
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
        """A pre-CS-300/hand-written map has is_tool=None, constructor_fanout=None — the hard gate must not reject a claimed role just because structural data is absent."""
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
    """SystemMapBuilder.build() makes 0 LLM calls post-CS-300 — this stub proves it by raising if used."""

    async def complete(self, *args, **kwargs):
        raise AssertionError("SystemMapBuilder.build() should not call the LLM for multi_agent")
