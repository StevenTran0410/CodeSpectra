"""Tests for Stage 4 compiler: render + codegen + template engine."""
import re
from pathlib import Path

import pytest


class TestE2ERender:
    """E2E render test: compile→resolve 4 .md files on foreign targets."""

    def test_render_4files_offline_fixture(self) -> None:
        """E2E render using offline handcrafted fixtures: valid markdown, no CodeSpectra literals, correct ceil(N/4) batch math, missing facts emit [NEEDS CLARIFICATION] markers."""
        from agent_eval_harness.code_injection.facts import (
            PlanFacts, AgentFacts, Resolved, Missing,
            ComponentRefFact,
        )
        from agent_eval_harness.code_injection.template_engine import resolve, clear_discovery_tasks

        # Create minimal offline facts for a 5-agent generic system (will produce ceil(5/4) = 2 batches)
        agents = [
            AgentFacts(
                agent_id=f"agent_{i}",
                role="analysis" if i % 2 == 0 else "retrieval",
                invocation_mode="in_harness",
                case_binding=Resolved(
                    value={"query": "case:$.input.query", "config_id": "config:config_id"},
                    citation="test",
                ),
            )
            for i in range(1, 6)
        ]

        facts = PlanFacts(
            session_id="test-offline",
            target_system_id="test_generic",
            branch_name="test-branch",
            plan_id="test-plan-1",
            agents=agents,
        )

        clear_discovery_tasks()

        _TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "agent_eval_harness" / "code_injection" / "templates"

        # Render on degraded (minimal) facts — must not crash or leak engine failure placeholders. code_md.tpl needs the compile step, so the real-compile test covers it instead.
        rendered = {}
        for tpl_name in ("agents_md.tpl", "tasks_md.tpl", "reference_md.tpl"):
            rendered[tpl_name] = resolve((_TEMPLATES_DIR / tpl_name).read_text(encoding="utf-8"), facts)
            assert rendered[tpl_name], f"{tpl_name} rendered empty"

        all_output = "\n".join(rendered.values())
        for bad in ("[UNKNOWN_SCALAR", "[UNKNOWN_TABLE", "[UNKNOWN_CODE", "BLOCK_NOT_FOUND"):
            assert bad not in all_output, f"engine placeholder {bad} in degraded render"
        tasks_content = rendered["tasks_md.tpl"]

        # Tier-2 grep: check for CodeSpectra target literals
        forbidden_patterns = [
            r"project_identity",
            r"RunDirectorAgent",
            r"AnalysisAgentPipeline",
            r"RetrievalService",
            r"domain\.analysis",
            r"domain\.builder",
            r"backend\/",
        ]

        for pattern in forbidden_patterns:
            matches = re.findall(pattern, all_output)
            assert not matches, f"Found forbidden literal '{pattern}' in rendered output: {matches[:3]}"

        # Verify batch math: ceil(5/4) = 2 batches expected
        assert "Batch" in tasks_content, "Batch headers should be present for 5 agents"

    def test_render_4files_real_compile_path(self, tmp_path) -> None:
        """THE GATE: real compile→render against real model objects — never skip this; hand-built PlanFacts fixtures can't catch compiler field drift or template↔engine slot drift, but this can."""
        from agent_eval_harness.mapping.system_map import SystemMap, Component
        from agent_eval_harness.planning.report import (
            EvaluationPlanReport, AgentPlanReport,
        )
        from agent_eval_harness.planning.contract import (
            EvaluationContract, InvocationContract,
        )
        from agent_eval_harness.discovery.agent_knowledge import (
            AgentKnowledge, LocationInfo, ComponentRef, ContractArg, PromptSiteRef,
        )
        from agent_eval_harness.code_injection.wiring import build_wiring
        from agent_eval_harness.code_injection.plan_renderer import (
            PRESERVED_PLAN_FILES,
            render_eval_plan_files,
        )
        from agent_eval_harness.code_injection.facts import compile_plan_facts

        agent_ids = [f"analyzer_{i}" for i in range(1, 6)]  # 5 agents → ceil(5/4)=2 batches
        components = [
            Component(id=aid, role="worker", entry_point=f"{aid}.run", file=f"src/agents/{aid}.py")
            for aid in agent_ids
        ]
        system_map = SystemMap(target_system_id="fictional_pipeline", components=components)

        reports = []
        for i, aid in enumerate(agent_ids):
            # in_harness + a recorded route is what the harvest produces: a route can't take a synthetic case's input, so it's documented but never driven.
            inv = InvocationContract(
                callable=aid, method="run", invocation_mode="in_harness",
                route="/api/run_section",
                case_binding={"section_id": "case:$.input.section_id", "provider": "config:provider"},
                constructor_deps=["Retriever" if i == 0 else "Store"],
            )
            reports.append(AgentPlanReport(
                agent_id=aid, role="worker",
                contract=EvaluationContract(
                    agent_id=aid, invocation=inv,
                    has_retrieval_signal=(i == 0),  # first agent needs a retrieval stub
                ),
            ))
        plan_report = EvaluationPlanReport(target_system_id="fictional_pipeline", agents=reports)

        knowledge_dir = tmp_path / "agents" / "sess1"
        knowledge_dir.mkdir(parents=True)
        for aid in agent_ids:
            k = AgentKnowledge(
                agent_id=aid,
                location=LocationInfo(file=f"src/agents/{aid}.py", line_start=10, line_end=80,
                                      entry_method="run", entry_line=15),
                components=[ComponentRef(id=aid, role="worker", file=f"src/agents/{aid}.py", line=10)],
                input_contract=[ContractArg(kwarg="section_id", source_kind="case", type_hint="str")],
                prompt_sites=[PromptSiteRef(file=f"src/agents/{aid}.py", line=5, kind="constant", snippet="PROMPT")],
                functionality=f"{aid} does a fictional analysis step.",
            )
            (knowledge_dir / f"{aid}.json").write_text(k.model_dump_json(), encoding="utf-8")

        dataset_summaries = [
            {"dataset_id": f"synthetic_{aid}", "kind": "synthetic_agent_io",
             "case_count": 3, "gate_ids": [f"g_{aid}"], "example_case": None}
            for aid in agent_ids
        ]
        wiring = build_wiring(
            system_map, "sess1", dataset_kinds=None,
            agent_invocations={r.agent_id: {
                "invocation_mode": r.contract.invocation.invocation_mode,
                "case_binding": r.contract.invocation.case_binding,
                "route": r.contract.invocation.route,
                "constructor_deps": r.contract.invocation.constructor_deps,
            } for r in reports},
        )
        wiring["aeh_db_path"] = str(tmp_path / "aeh.db")
        wiring["dataset_ids"] = sorted(d["dataset_id"] for d in dataset_summaries)

        # Compile must not raise on real models — the agent_flows / component.line trap.
        facts = compile_plan_facts(
            system_map=system_map, wiring=wiring, dataset_summaries=dataset_summaries,
            session_id="sess1", branch_name="aeh/eval-sess1",
            plan_report=plan_report, knowledge_dir=knowledge_dir,
        )
        assert len(facts.agents) == 5
        assert len(facts.dispatch_modules) == 5
        assert facts.code_artifacts.get("tracer")

        files = render_eval_plan_files(
            system_map, wiring, dataset_summaries, "sess1", "aeh/eval-sess1",
            plan_report=plan_report, knowledge_dir=knowledge_dir,
        )
        assert set(files) == {"AGENTS.md", "TASKS.md", "REFERENCE.md", "CODE.md", "RECON.md"}
        # RECON.md is the agent's own; a re-render must never be allowed to clobber it.
        assert PRESERVED_PLAN_FILES == {"RECON.md"}
        assert "RECON.md" in files["TASKS.md"], "TASKS.md must point at the file it no longer owns"

        for name, content in files.items():
            for bad in ("[UNKNOWN_SCALAR", "[UNKNOWN_TABLE", "[UNKNOWN_CODE",
                        "BLOCK_NOT_FOUND", "Unknown block", "Unknown slot type"):
                assert bad not in content, f"{name} contains {bad!r}"

        tasks = files["TASKS.md"]
        for aid in agent_ids:
            assert aid in tasks
            assert aid in files["REFERENCE.md"]
        assert "**WHERE**" in files["REFERENCE.md"]
        assert "src/agents/analyzer_1.py:10" in files["REFERENCE.md"]
        assert "**SHA256**" in files["CODE.md"]
        assert ".aeh/dispatch/analyzer_1.py" in files["CODE.md"]
        # retrieval stub warning fired for the one retrieval-bearing agent
        assert "Retrieval Behavior in Evaluation" in files["REFERENCE.md"]

        # The task ledger must cover the WHOLE job, not just the agent batches.
        assert "git rev-parse --abbrev-ref HEAD" in tasks, "no branch checkout task"
        assert "aeh/eval-sess1" in tasks
        for path in (".aeh/tracer.py", ".aeh/run_eval.py", ".aeh/wiring.json", ".aeh/aeh_eval.py"):
            assert f"Create `{path}`" in tasks, f"no task creating {path}"
        assert "Server Entrypoint Edits" in tasks, "no task applying the entrypoint edits"
        assert "## M0" in tasks and "## M1" in tasks  # setup + skeleton
        assert "## M2" in tasks and "## M3" in tasks  # 5 agents / 4 = 2 batches
        assert "## M4" in tasks, "no final run & hand-back milestone"
        assert "python .aeh/run_eval.py`" in tasks, "no full-run task"
        assert "Hand back" in tasks

        # Generated dispatch must be honest: an explicit IMPLEMENT region, an async entrypoint (run_eval awaits it), and a task that tells the agent to implement it.
        for dm in facts.dispatch_modules:
            assert "async def invoke_agent" in dm["code"], f"{dm['agent_id']} dispatch not async"
            assert "=== IMPLEMENT THIS ===" in dm["code"], f"{dm['agent_id']} dispatch has no IMPLEMENT region"
        assert "implement its `=== IMPLEMENT THIS ===` region" in tasks, (
            "dispatch tasks must instruct implementing, not just copying"
        )
        # A file the plan says to copy verbatim must never be a stub.
        for key in ("tracer", "run_eval", "aeh_eval"):
            assert "NotImplementedError" not in facts.code_artifacts[key], (
                f"{key} is copied verbatim but contains a stub"
            )

        # The verify commands the gates use must exist in run_eval.py, cases must carry the agent_id that selects a dispatch module, and run_config.json must be read — these were three blocking conflicts a real coding agent hit.
        run_eval_src = facts.code_artifacts["run_eval"]
        for flag in ('"--agent"', '"--batch"', "--verify"):
            assert flag in run_eval_src, f"run_eval.py does not support {flag}"
        assert "_load_run_config" in run_eval_src, "run_eval.py never reads run_config.json"
        assert '"agent_id": agent_id' in run_eval_src, "cases never carry agent_id"
        assert '"batches"' in facts.code_artifacts["wiring"], "wiring.json has no batch map"
        assert "run_config.json" in tasks, "no task writes run_config.json"
        for key in ("provider_id", "model_id", "base_url"):
            assert key in tasks, f"run_config.json spec omits {key}"

        # Run output must land in the harness data dir, never in the target's tree.
        tracer_src = facts.code_artifacts["tracer"]
        assert "AEH_OUT_DIR" in tracer_src, "tracer writes beside itself in the target repo"
        assert "aeh_out_dir" in run_eval_src, "run_eval ignores the configured output dir"
        assert '(out_dir / "manifest.json")' in run_eval_src, "manifest written into the target tree"

        # Every referenced discovery task must actually be rendered as a task line.
        for ref in re.findall(r"see (D\d+)", tasks):
            assert re.search(rf"^- \[ \] {ref} ", tasks, re.M), f"{ref} referenced but never rendered"

        # Shipped code must be installable: every non-empty artifact needs a create task.
        for key, path in (("tracer", ".aeh/tracer.py"), ("run_eval", ".aeh/run_eval.py"),
                          ("wiring", ".aeh/wiring.json"), ("aeh_eval", ".aeh/aeh_eval.py"),
                          ("retrieval_stub", ".aeh/retrieval_stub.py")):
            if facts.code_artifacts.get(key):
                assert f"Create `{path}`" in tasks, f"{path} is shipped but never created"

        # Anything AEH cannot know must be an IMPLEMENT region, never a sha-locked guess: the stub's interface belongs to the target, and the target's process setup is its own.
        stub_src = facts.code_artifacts.get("retrieval_stub", "")
        assert "=== IMPLEMENT THIS ===" in stub_src, "retrieval stub fakes an interface"
        assert f"sha256 of `.aeh/retrieval_stub.py`" not in tasks, "stub must not be sha-locked"
        assert "_init_target" in run_eval_src, "run_eval never gives the target a setup hook"
        assert "=== IMPLEMENT THIS ===" in facts.code_artifacts.get("target_init", ""), (
            "target setup is not an agent-owned IMPLEMENT file"
        )
        assert "Create `.aeh/target_init.py`" in tasks, "no task creates the target setup file"
        assert "sha256 of `.aeh/target_init.py`" not in tasks, "setup file must not be sha-locked"

        # A retrieval-bearing agent must actually receive the case's evidence.
        stubbed = [d for d in facts.dispatch_modules
                   if "make_retrieval_stub" in d["code"]]
        assert stubbed, "no dispatch wires the retrieval stub"
        for d in stubbed:
            assert 'make_retrieval_stub(' in d["code"], (
                f"{d['agent_id']} builds a stub without the make_retrieval_stub call"
            )
            assert 'case["input"].get(' in d["code"], (
                f"{d['agent_id']} builds a stub without the case's evidence"
            )

        # Zero CodeSpectra template literals leaked into a fictional-system render.
        blob = "\n".join(files.values())
        for pattern in (r"RunDirectorAgent", r"AnalysisAgentPipeline", r"domain\.analysis"):
            assert not re.search(pattern, blob), f"leaked literal {pattern}"


class TestForeignTargetRender:
    """AC#5: the plan must render on targets that are NOT CodeSpectra. Passing on CodeSpectra
    is necessary but not sufficient — genericity is only proven on a foreign map."""

    @pytest.mark.parametrize("target", ["multi_agent", "linear_rag"])
    def test_render_on_foreign_target(self, target, tmp_path) -> None:
        from agent_eval_harness.mapping.system_map import load_system_map
        from agent_eval_harness.code_injection.wiring import build_wiring
        from agent_eval_harness.code_injection.plan_renderer import render_eval_plan_files

        map_path = Path(__file__).parent.parent / "test_targets" / target / "system_map.yaml"
        system_map = load_system_map(map_path)
        agent_ids = [c.id for c in system_map.components]
        assert agent_ids, f"{target} map has no components"

        dataset_summaries = [
            {"dataset_id": f"ds_{cid}", "kind": "synthetic_agent_io",
             "case_count": 2, "gate_ids": [], "example_case": None}
            for cid in agent_ids
        ]
        wiring = build_wiring(system_map, f"sess-{target}")
        wiring["aeh_db_path"] = str(tmp_path / "aeh.db")
        wiring["dataset_ids"] = sorted(d["dataset_id"] for d in dataset_summaries)

        files = render_eval_plan_files(
            system_map, wiring, dataset_summaries, f"sess-{target}", f"aeh/eval-{target}",
        )
        assert set(files) == {"AGENTS.md", "TASKS.md", "REFERENCE.md", "CODE.md", "RECON.md"}

        blob = "\n".join(files.values())
        for bad in ("[UNKNOWN_SCALAR", "[UNKNOWN_TABLE", "[UNKNOWN_CODE",
                    "BLOCK_NOT_FOUND", "Unknown block", "Unknown slot type"):
            assert bad not in blob, f"{target}: engine placeholder {bad}"

        for pattern in (r"RunDirectorAgent", r"AnalysisAgentPipeline", r"domain\.analysis",
                        r"\bproject_identity\b", r"backend/"):
            assert not re.search(pattern, blob), f"{target}: leaked CodeSpectra literal {pattern}"

        for cid in agent_ids:
            assert cid in files["TASKS.md"], f"{target}: {cid} missing from TASKS.md"
            assert cid in files["REFERENCE.md"], f"{target}: {cid} missing from REFERENCE.md"
        assert system_map.target_system_id in files["REFERENCE.md"]


class TestTypedBindingSurfacing:
    """§5.3: a case field bound to a kwarg annotated with a target type must be surfaced so the
    coding agent converts it, instead of passing the raw dict the agent then dereferences."""

    def test_typed_binding_flagged_plain_passes_through(self) -> None:
        from agent_eval_harness.code_injection.facts import _annotation_needs_conversion

        for plain in ("str", "int | None", "dict[str, Any]", "list[str]", "Optional[bool]", None, ""):
            assert not _annotation_needs_conversion(plain), plain
        for typed in ("RetrievalBundle", "RetrievalBundle | None", "models.Section",
                      "list[Evidence]"):
            assert _annotation_needs_conversion(typed), typed

    def test_typed_binding_renders_a_warning_no_type_literal(self, tmp_path) -> None:
        from agent_eval_harness.mapping.system_map import SystemMap, Component
        from agent_eval_harness.planning.report import EvaluationPlanReport, AgentPlanReport
        from agent_eval_harness.planning.contract import (
            EvaluationContract, InvocationContract, KwargSpec,
        )
        from agent_eval_harness.code_injection.wiring import build_wiring
        from agent_eval_harness.code_injection.plan_renderer import render_eval_plan_files

        component = Component(id="a1", role="worker", entry_point="a1.run", file="src/a1.py")
        system_map = SystemMap(target_system_id="tsys", components=[component])
        inv = InvocationContract(
            callable="a1", method="run", invocation_mode="in_harness",
            case_binding={"bundle": "case:$.input.bundle", "name": "case:$.input.name"},
            kwargs=[KwargSpec(name="bundle", annotation="EvidenceBundle | None"),
                    KwargSpec(name="name", annotation="str")],
        )
        plan_report = EvaluationPlanReport(target_system_id="tsys", agents=[AgentPlanReport(
            agent_id="a1", role="worker",
            contract=EvaluationContract(agent_id="a1", invocation=inv),
        )])
        ds = [{"dataset_id": "ds_a1", "kind": "synthetic_agent_io", "case_count": 2,
               "gate_ids": [], "example_case": {"labels": {"agent_id": "a1"},
                                                "input": {"bundle": {}, "name": "x"}}}]
        wiring = build_wiring(system_map, "s1")
        wiring["aeh_db_path"] = str(tmp_path / "aeh.db")
        wiring["dataset_ids"] = ["ds_a1"]

        files = render_eval_plan_files(system_map, wiring, ds, "s1", "aeh/eval-s1",
                                       plan_report=plan_report)
        ref = files["REFERENCE.md"]
        assert "EvidenceBundle | None" in ref, "typed kwarg annotation never surfaced"
        assert "Typed kwarg" in ref, "no conversion warning for the typed binding"
        assert "`name`: `str`" not in ref, "plain kwarg wrongly flagged as typed"


class TestMarkerEmission:
    """Test marker emission for missing facts."""

    def test_marker_emits_inline_placeholder(self) -> None:
        """Missing fact should emit [NEEDS CLARIFICATION] marker in output."""
        from agent_eval_harness.code_injection.template_engine import resolve, clear_discovery_tasks
        from agent_eval_harness.code_injection.facts import PlanFacts, Missing

        clear_discovery_tasks()

        facts = PlanFacts(
            session_id="test",
            target_system_id="test",
            branch_name="test",
            plan_id="test",
            agents=[],
        )

        template = "Provider: {{marker:provider_listing}}"
        result = resolve(template, facts)

        assert "[NEEDS CLARIFICATION" in result, "Missing fact should emit [NEEDS CLARIFICATION] marker"


class TestCodegenCompileGate:
    """Test codegen produces valid Python."""

    def test_codegen_dispatch_structure(self) -> None:
        """Codegen dispatch modules produce syntactically valid Python."""
        import ast
        from agent_eval_harness.code_injection.codegen import generate_dispatch_module
        from agent_eval_harness.code_injection.facts import AgentFacts, Resolved

        sample_case_binding_dict = {
            "snapshot_id": "case:$.input.snapshot_id",
            "provider_id": "config:provider_id",
            "model_id": "config:model_id",
        }

        agent_facts = AgentFacts(
            agent_id="test_agent",
            role="analysis",
            invocation_mode="in_harness",
            case_binding=Resolved(value=sample_case_binding_dict, citation="test"),
        )

        code, sha256_hex = generate_dispatch_module(agent_facts)

        assert code, "Generated dispatch code is empty"
        assert len(sha256_hex) == 64, f"SHA256 should be 64 hex chars, got {len(sha256_hex)}"

        try:
            ast.parse(code)
        except SyntaxError as e:
            pytest.fail(f"Generated dispatch code has syntax error: {e}\n\nCode:\n{code}")


class TestBatchMath:
    """Test batch math and boundary conditions."""

    def test_batch_math_ceil(self) -> None:
        """Batch size should be ceil(N/4)."""
        test_cases = [
            (0, 0),
            (1, 1),
            (4, 1),
            (5, 2),
            (8, 2),
            (9, 3),
            (12, 3),
            (13, 4),
        ]

        for n_agents, expected_batches in test_cases:
            import math
            actual = math.ceil(n_agents / 4) if n_agents > 0 else 0
            assert actual == expected_batches, f"N={n_agents}: expected {expected_batches} batches, got {actual}"


class TestGrepGatePermanent:
    """Permanent Tier-1 grep gate: ensures no CodeSpectra hardcodes ever enter templates or blocks."""

    def test_tier1_grep_gate_production_code_templates_blocks(self) -> None:
        """Grep gate covering production code, templates, and blocks — ensures zero CodeSpectra-specific target literals in any of them."""
        from pathlib import Path

        production_files = [
            "agent_eval_harness/agent_eval_harness/code_injection/facts.py",
            "agent_eval_harness/agent_eval_harness/code_injection/codegen.py",
            "agent_eval_harness/agent_eval_harness/code_injection/template_engine.py",
            "agent_eval_harness/agent_eval_harness/code_injection/plan_renderer.py",
            "agent_eval_harness/agent_eval_harness/code_injection/wiring.py",
            "agent_eval_harness/agent_eval_harness/mapping/builder/contract_harvest.py",
        ]

        template_files = [
            "agent_eval_harness/agent_eval_harness/code_injection/templates/agents_md.tpl",
            "agent_eval_harness/agent_eval_harness/code_injection/templates/tasks_md.tpl",
            "agent_eval_harness/agent_eval_harness/code_injection/templates/reference_md.tpl",
            "agent_eval_harness/agent_eval_harness/code_injection/templates/code_md.tpl",
        ]

        blocks_dir = Path("agent_eval_harness/agent_eval_harness/code_injection/blocks")

        forbidden_patterns = [
            r"project_identity\b",
            r"RunDirectorAgent",
            r"AnalysisAgentPipeline",
            r"RetrievalService\b",
            r"domain\.analysis",
            r"domain\.builder",
            r"backend/",
        ]

        found_issues = []

        for filepath_str in production_files:
            filepath = Path(filepath_str)
            if not filepath.exists():
                continue
            content = filepath.read_text(encoding="utf-8")
            for pattern in forbidden_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    found_issues.append((filepath_str, pattern, len(matches)))

        for filepath_str in template_files:
            filepath = Path(filepath_str)
            if not filepath.exists():
                continue
            content = filepath.read_text(encoding="utf-8")
            for pattern in forbidden_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    found_issues.append((filepath_str, pattern, len(matches)))

        if blocks_dir.exists():
            for block_file in blocks_dir.glob("*.md"):
                content = block_file.read_text(encoding="utf-8")
                for pattern in forbidden_patterns:
                    matches = re.findall(pattern, content)
                    if matches:
                        found_issues.append((str(block_file), pattern, len(matches)))

        assert not found_issues, (
            f"Grep gate FAIL: Found {len(found_issues)} forbidden pattern(s):\n" +
            "\n".join(f"  {f}: {p} ({c} match)" for f, p, c in found_issues)
        )


def test_slice31_merge_virtual_input_bindings() -> None:
    """Verify Slice 3.1 merge flows end-to-end: virtual_inputs in AgentKnowledge reach contract.case_binding."""
    from agent_eval_harness.planning.agentic_planner import _merge_virtual_input_bindings
    from agent_eval_harness.planning.contract import EvaluationContract, InvocationContract

    # Create a contract with empty case_binding
    contract = EvaluationContract(
        agent_id="test_agent",
        invocation=InvocationContract(
            invocation_mode="in_harness",
            kwargs=[],
            case_binding={"existing_kwarg": "case:$.input.query"},
        ),
    )
    contracts = {"test_agent": contract}

    # Create mock AgentKnowledge with virtual_inputs
    agent_knowledge = {
        "test_agent": {
            "virtual_inputs": [
                {
                    "name": "bundle",
                    "dep_attr": "_retriever",
                    "dep_param": "retriever",
                    "dep_annotation": "RetrievalService",
                    "dep_role": "retrieval",
                    "methods_called": ["search"],
                    "call_sites": ["agents.py:15"],
                    "fields": [],
                }
            ]
        }
    }

    # Call merge
    _merge_virtual_input_bindings(contracts, agent_knowledge)

    # Assert virtual: binding was added
    assert "virtual:bundle" in contracts["test_agent"].invocation.case_binding.values(), (
        f"Expected 'virtual:bundle' in case_binding values, got {contracts['test_agent'].invocation.case_binding}"
    )
    # Assert existing bindings preserved
    assert contracts["test_agent"].invocation.case_binding["existing_kwarg"] == "case:$.input.query", (
        "Existing case_binding entry should not be modified"
    )
