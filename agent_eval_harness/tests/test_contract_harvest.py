"""Tests for static evaluation-contract harvest."""
from __future__ import annotations

from pathlib import Path

import pytest

from agent_eval_harness.config import ContractConventions
from agent_eval_harness.mapping.agent_flow import AgentFlow, AgentFlowMap
from agent_eval_harness.mapping.builder.contract_harvest import (
    _SchemaResolveCtx,
    _parse_files,
    _resolve_class_schema,
    harvest_component_contract,
    harvest_contracts,
)
from agent_eval_harness.mapping.system_map import Component, SystemMap
from agent_eval_harness.planning.agentic_planner import _merge_observability
from agent_eval_harness.planning.contract import EvaluationContract
from agent_eval_harness.planning.report import (
    AgentDataProfile,
    AgentPlanReport,
    EvaluationPlanReport,
    load_plan_report,
    save_plan_report,
)

AGENT_SRC = '''
from myapp.schemas import SectionA, validate_section
from myapp.prompts import AGENT_A_SCHEMA_STR

MAX_RETRIES = 3

class FooAgent:
    """Analyzes a snapshot."""

    N_ROUNDS = 2

    def __init__(self, provider_service: ProviderConfigService, retrieval_service: RetrievalService):
        self._retrieval = retrieval_service

    def _fallback(self, snapshot_id: str, reason: str, repo_name: str = "") -> dict:
        return {
            "repo_name": repo_name,
            "domain": "unknown",
            "tech_stack": [],
            "confidence": "low",
            "blind_spots": [reason],
        }

    async def run(
        self,
        provider_id: str,
        model_id: str,
        snapshot_id: str,
        repo_name: str = "",
        profile=None,
    ) -> dict:
        data = await self._chat(AGENT_A_SCHEMA_STR)
        validate_section("A", data)
        return data
'''

SCHEMAS_SRC = '''
from typing import TypedDict
from typing_extensions import NotRequired

class SectionA(TypedDict):
    repo_name: str
    domain: str
    tech_stack: list[str]
    confidence: str
    mermaid: NotRequired[str]
'''

PROMPTS_SRC = '''
AGENT_A_SCHEMA_STR = "Return JSON with keys repo_name, domain (this is a template, not JSON)"
'''


def _write_fixture(tmp_path: Path) -> list[Path]:
    (tmp_path / "myapp" / "agents").mkdir(parents=True)
    files = {
        "myapp/agents/foo_agent.py": AGENT_SRC,
        "myapp/schemas.py": SCHEMAS_SRC,
        "myapp/prompts.py": PROMPTS_SRC,
    }
    paths = []
    for rel, src in files.items():
        p = tmp_path / rel
        p.write_text(src, encoding="utf-8")
        paths.append(p)
    return paths


def _foo_component() -> Component:
    return Component(
        id="foo",
        role="orchestrator",
        entry_point="myapp.agents.foo_agent:FooAgent",
        file="myapp/agents/foo_agent.py",
    )


def test_run_signature_kwargs_defaults_and_required(tmp_path: Path) -> None:
    asts = _parse_files(_write_fixture(tmp_path))
    invocation, _, _, _, _ = harvest_component_contract(_foo_component(), asts, tmp_path)

    assert invocation is not None
    assert invocation.method == "run"
    by_name = {k.name: k for k in invocation.kwargs}
    assert set(by_name) == {"provider_id", "model_id", "snapshot_id", "repo_name", "profile"}
    assert by_name["snapshot_id"].required is True
    assert by_name["snapshot_id"].annotation == "str"
    assert by_name["repo_name"].required is False
    assert by_name["repo_name"].default_repr == "''"
    assert invocation.constructor_deps == ["ProviderConfigService", "RetrievalService"]
    # Constructor deps do not force a route: only in-process invocation can substitute a retrieval stub.
    assert invocation.invocation_mode == "in_harness"
    assert invocation.route == "/api/analysis/rerun_section"  # recorded, not used to drive cases
    assert invocation.citations and "foo_agent.py" in invocation.citations[0]


def test_case_binding_uses_run_kwargs_even_when_a_route_exists(tmp_path: Path) -> None:
    """A per-section route resolves its own state from a stored report, so it can never take a synthetic case's input; the binding must be the real run() kwargs shape instead."""
    asts = _parse_files(_write_fixture(tmp_path))
    invocation, _, _, notes, input_kind = harvest_component_contract(_foo_component(), asts, tmp_path)

    assert invocation is not None
    assert "report_id" not in invocation.case_binding
    assert invocation.case_binding["snapshot_id"] == "case:$.input.snapshot_id"
    assert invocation.case_binding["provider_id"] == "config:provider_id"
    assert input_kind == "structured"
    # The route is surfaced as a note so a reader knows it exists and why it is unused.
    assert any("not used for evaluation" in n for n in notes)


def test_case_binding_uses_kwargs_shape_when_no_route_available(tmp_path: Path) -> None:
    """Constructor deps but no route: still in-process, binding is the raw kwargs shape."""
    (tmp_path / "noletter.py").write_text(
        "class NoLetterAgent:\n"
        "    def __init__(self, provider_service: ProviderConfigService):\n"
        "        self._provider = provider_service\n\n"
        "    async def run(self, provider_id: str, snapshot_id: str) -> dict:\n"
        "        return {}\n",
        encoding="utf-8",
    )
    asts = _parse_files([tmp_path / "noletter.py"])
    comp = Component(id="noletter", role="unknown", entry_point="noletter:NoLetterAgent", file="noletter.py")

    invocation, _, _, _, _ = harvest_component_contract(comp, asts, tmp_path)

    assert invocation is not None
    assert invocation.constructor_deps == ["ProviderConfigService"]
    assert invocation.invocation_mode == "in_harness"
    assert invocation.route is None
    assert invocation.case_binding == {
        "provider_id": "config:provider_id",
        "snapshot_id": "case:$.input.snapshot_id",
    }


def test_typeddict_schema_via_validate_call(tmp_path: Path) -> None:
    asts = _parse_files(_write_fixture(tmp_path))
    _, output, _, _, _ = harvest_component_contract(_foo_component(), asts, tmp_path)

    assert output is not None
    assert output.validated_in_target is True
    assert output.json_schema is not None
    props = output.json_schema["properties"]
    assert props["repo_name"] == {"type": "string"}
    assert props["tech_stack"] == {"type": "array", "items": {"type": "string"}}
    assert "mermaid" in props
    assert set(output.json_schema["required"]) == {"repo_name", "domain", "tech_stack", "confidence"}
    assert output.schema_source is not None and "SectionA" in output.schema_source


def test_fallback_literal_with_dynamic_marker(tmp_path: Path) -> None:
    asts = _parse_files(_write_fixture(tmp_path))
    _, output, _, _, _ = harvest_component_contract(_foo_component(), asts, tmp_path)

    assert output is not None and output.fallback_literal is not None
    assert output.fallback_literal["domain"] == "unknown"
    assert output.fallback_literal["tech_stack"] == []
    # repo_name / blind_spots reference variables -> dynamic marker, never guessed
    assert output.fallback_literal["repo_name"] == "<dynamic>"
    assert output.fallback_literal["blind_spots"] == "<dynamic>"
    assert output.fallback_source is not None


# An agent that returns a dict LITERAL (no validator letter, no SCHEMA constant) must still yield
# a harvested output schema, inferred from the return dict on the main path.
_DICT_LITERAL_AGENT_SRC = '''
class JudgeAgent:
    def __init__(self, llm_client):
        self._llm = llm_client

    async def run_async(self, query: str, worker_output: dict) -> dict:
        sufficient = bool(worker_output)
        return {"sufficient": sufficient, "context": worker_output.get("context", [])}
'''


def test_dict_literal_return_infers_output_schema_on_main_path(tmp_path: Path) -> None:
    src = tmp_path / "judge_agent.py"
    src.write_text(_DICT_LITERAL_AGENT_SRC, encoding="utf-8")
    asts = _parse_files([src])
    comp = Component(id="judge", role="validator", entry_point="judge_agent:JudgeAgent", file="judge_agent.py")

    _, output, _, _, _ = harvest_component_contract(comp, asts, tmp_path)

    assert output is not None
    assert output.json_schema is not None, "dict-literal return must yield an inferred schema, not None"
    assert set(output.json_schema["required"]) == {"sufficient", "context"}
    assert output.json_schema["properties"].keys() == {"sufficient", "context"}
    assert output.schema_source is not None and "inferred from return dict" in output.schema_source


def test_constants_harvested(tmp_path: Path) -> None:
    asts = _parse_files(_write_fixture(tmp_path))
    _, _, constants, _, _ = harvest_component_contract(_foo_component(), asts, tmp_path)
    assert constants == {"MAX_RETRIES": 3, "N_ROUNDS": 2}


def test_query_shaped_agent(tmp_path: Path) -> None:
    (tmp_path / "chat.py").write_text(
        "class ChatAgent:\n    async def run(self, query: str):\n        return query\n",
        encoding="utf-8",
    )
    asts = _parse_files([tmp_path / "chat.py"])
    comp = Component(id="chat", role="writer", entry_point="chat:ChatAgent", file="chat.py")
    invocation, _, _, _, input_kind = harvest_component_contract(comp, asts, tmp_path)
    assert input_kind == "query"
    assert invocation is not None and invocation.invocation_mode == "in_harness"


def test_varkwargs_and_decorator_shapes(tmp_path: Path) -> None:
    (tmp_path / "odd.py").write_text(
        "def deco(f):\n    return f\n\n"
        "class OddAgent:\n"
        "    @deco\n"
        "    async def run(self, snapshot_id: str, **extras):\n"
        "        return {}\n",
        encoding="utf-8",
    )
    asts = _parse_files([tmp_path / "odd.py"])
    comp = Component(id="odd", role="unknown", entry_point="odd:OddAgent", file="odd.py")
    invocation, _, _, notes, _ = harvest_component_contract(comp, asts, tmp_path)
    assert invocation is not None  # decorator does not hide the method
    assert [k.name for k in invocation.kwargs] == ["snapshot_id"]
    assert any("**extras" in n for n in notes)


def test_dynamic_schema_str_unresolvable(tmp_path: Path) -> None:
    (tmp_path / "dyn.py").write_text(
        'BASE = "x"\nDYN_SCHEMA_STR = f"schema {BASE}"\n\n'
        "class DynAgent:\n"
        "    async def run(self, snapshot_id: str):\n"
        "        return DYN_SCHEMA_STR\n",
        encoding="utf-8",
    )
    asts = _parse_files([tmp_path / "dyn.py"])
    comp = Component(id="dyn", role="unknown", entry_point="dyn:DynAgent", file="dyn.py")
    _, output, _, notes, _ = harvest_component_contract(comp, asts, tmp_path)
    assert output is not None and output.json_schema is None
    assert any("dynamically assembled" in n for n in notes)


def test_json_schema_str_fallback(tmp_path: Path) -> None:
    (tmp_path / "js.py").write_text(
        'JS_SCHEMA_STR = \'{"type": "object", "properties": {"a": {"type": "string"}}}\'\n\n'
        "class JsAgent:\n"
        "    async def run(self, snapshot_id: str):\n"
        "        return JS_SCHEMA_STR\n",
        encoding="utf-8",
    )
    asts = _parse_files([tmp_path / "js.py"])
    comp = Component(id="js", role="unknown", entry_point="js:JsAgent", file="js.py")
    _, output, _, _, _ = harvest_component_contract(comp, asts, tmp_path)
    assert output is not None and output.json_schema == {
        "type": "object",
        "properties": {"a": {"type": "string"}},
    }


def test_dynamic_schema_str_not_misattributed_across_files(tmp_path: Path) -> None:
    """A dynamic constant in the imported-from file must never fall through to an unrelated file's same-named literal constant — that would silently misattribute one component's schema to another."""
    (tmp_path / "agent_mod.py").write_text(
        "from shared_prompts import AGENT_SCHEMA_STR\n\n"
        "class SharedAgent:\n"
        "    async def run(self, snapshot_id: str):\n"
        "        return AGENT_SCHEMA_STR\n",
        encoding="utf-8",
    )
    (tmp_path / "shared_prompts.py").write_text(
        'BASE = "x"\nAGENT_SCHEMA_STR = f"schema {BASE}"\n',
        encoding="utf-8",
    )
    # Unrelated file that happens to define a same-named constant with a real literal — must NOT be picked up as this agent's schema.
    (tmp_path / "other_module.py").write_text(
        'AGENT_SCHEMA_STR = \'{"type": "object", "properties": {"unrelated": {"type": "string"}}}\'\n',
        encoding="utf-8",
    )
    asts = _parse_files(
        [tmp_path / "agent_mod.py", tmp_path / "shared_prompts.py", tmp_path / "other_module.py"]
    )
    comp = Component(id="shared", role="unknown", entry_point="agent_mod:SharedAgent", file="agent_mod.py")
    _, output, _, notes, _ = harvest_component_contract(comp, asts, tmp_path)
    assert output is not None and output.json_schema is None
    assert any("dynamically assembled" in n for n in notes)


def test_fallback_disambiguates_by_entry_method_call(tmp_path: Path) -> None:
    """Two methods contain 'fallback' in their name; the decoy sorts first in the class body but the real one is the one the entry method actually calls."""
    (tmp_path / "picky.py").write_text(
        "class PickyAgent:\n"
        "    def _fallback_provider_config(self) -> dict:\n"
        "        return {'provider': 'decoy', 'priority': 1}\n\n"
        "    def _output_fallback(self, reason: str) -> dict:\n"
        "        return {'domain': 'unknown', 'reason_note': 'real'}\n\n"
        "    async def run(self, snapshot_id: str):\n"
        "        try:\n"
        "            return {}\n"
        "        except Exception:\n"
        "            return self._output_fallback('err')\n",
        encoding="utf-8",
    )
    asts = _parse_files([tmp_path / "picky.py"])
    comp = Component(id="picky", role="unknown", entry_point="picky:PickyAgent", file="picky.py")
    _, output, _, notes, _ = harvest_component_contract(comp, asts, tmp_path)
    assert output is not None and output.fallback_literal is not None
    assert output.fallback_literal["reason_note"] == "real"
    assert any("multiple candidate fallback methods" in n for n in notes)


def test_harvest_fallback_falls_through_to_pipeline_module_for_k(tmp_path: Path) -> None:
    """K/L-shaped agents have no own try/except — validate_section() alone means the entry method's fallback lives in a module-level `_section_<letter>_pipeline_fallback()` in a different file (the pipeline orchestrator), never a class method."""
    (tmp_path / "agent_k.py").write_text(
        "from schemas import validate_section\n\n"
        "class KAgent:\n"
        "    async def run(self, all_sections: dict):\n"
        "        result = {}\n"
        "        validate_section('K', result)\n"
        "        return result\n",
        encoding="utf-8",
    )
    (tmp_path / "agent_pipeline.py").write_text(
        "def _section_k_pipeline_fallback() -> dict:\n"
        "    return {'overall_confidence': 'low', 'notes': 'Audit could not be completed.'}\n",
        encoding="utf-8",
    )
    asts = _parse_files([tmp_path / "agent_k.py", tmp_path / "agent_pipeline.py"])
    comp = Component(id="k", role="auditor", entry_point="agent_k:KAgent", file="agent_k.py")
    _, output, _, notes, _ = harvest_component_contract(comp, asts, tmp_path)
    assert output is not None and output.fallback_literal is not None
    assert output.fallback_literal == {
        "overall_confidence": "low", "notes": "Audit could not be completed.",
    }
    assert output.fallback_source is not None and "agent_pipeline.py" in output.fallback_source
    assert not any("multiple candidate fallback methods" in n for n in notes)


def test_harvest_fallback_no_fallthrough_without_a_validator_letter(tmp_path: Path) -> None:
    """The pipeline-fallback fallthrough is letter-keyed — an agent with no own fallback method AND no validator letter has nothing to fall through to (never guessed)."""
    (tmp_path / "agent_x.py").write_text(
        "class XAgent:\n"
        "    async def run(self, snapshot_id: str):\n"
        "        return {}\n",
        encoding="utf-8",
    )
    (tmp_path / "agent_pipeline.py").write_text(
        "def _section_k_pipeline_fallback() -> dict:\n"
        "    return {'overall_confidence': 'low'}\n",
        encoding="utf-8",
    )
    asts = _parse_files([tmp_path / "agent_x.py", tmp_path / "agent_pipeline.py"])
    comp = Component(id="x", role="unknown", entry_point="agent_x:XAgent", file="agent_x.py")
    _, output, _, _, _ = harvest_component_contract(comp, asts, tmp_path)
    assert output is not None and output.fallback_literal is None


def test_contract_conventions_rerun_section_route_overridable(tmp_path: Path) -> None:
    """rerun_section_route has no harvestable equivalent at all (statics can't discover a target's REST route) — config is the only source, so it must actually reach the output."""
    asts = _parse_files(_write_fixture(tmp_path))
    default_invocation, _, _, _, _ = harvest_component_contract(_foo_component(), asts, tmp_path)
    assert default_invocation is not None
    assert default_invocation.route == "/api/analysis/rerun_section"

    custom = ContractConventions(rerun_section_route="/api/custom/rerun")
    custom_invocation, _, _, _, _ = harvest_component_contract(
        _foo_component(), asts, tmp_path, conventions=custom
    )
    assert custom_invocation is not None
    assert custom_invocation.route == "/api/custom/rerun"


def test_contract_conventions_pipeline_fallback_pattern_fills_what_default_could_not_supply(
    tmp_path: Path,
) -> None:
    """Config fills a convention harvest could NOT obtain: KAgent has no own fallback method, so the module-level pipeline-fallback lookup is the only source — under the DEFAULT pattern the real function's custom name isn't found; a config matching its actual name resolves it."""
    (tmp_path / "agent_k.py").write_text(
        "from schemas import validate_section\n\n"
        "class KAgent:\n"
        "    async def run(self, all_sections: dict):\n"
        "        result = {}\n"
        "        validate_section('K', result)\n"
        "        return result\n",
        encoding="utf-8",
    )
    (tmp_path / "agent_pipeline.py").write_text(
        "def _k_fallback_custom() -> dict:\n"
        "    return {'overall_confidence': 'low', 'notes': 'custom-pattern fallback'}\n",
        encoding="utf-8",
    )
    asts = _parse_files([tmp_path / "agent_k.py", tmp_path / "agent_pipeline.py"])
    comp = Component(id="k", role="auditor", entry_point="agent_k:KAgent", file="agent_k.py")

    _, default_output, _, _, _ = harvest_component_contract(comp, asts, tmp_path)
    assert default_output is not None and default_output.fallback_literal is None

    custom = ContractConventions(pipeline_fallback_name_pattern="_{letter}_fallback_custom")
    _, custom_output, _, _, _ = harvest_component_contract(comp, asts, tmp_path, conventions=custom)
    assert custom_output is not None and custom_output.fallback_literal == {
        "overall_confidence": "low", "notes": "custom-pattern fallback",
    }


def test_contract_conventions_harvest_wins_over_config_when_class_fallback_present(
    tmp_path: Path,
) -> None:
    """Harvest wins when it already supplies a value: an agent with its OWN class-level fallback method must keep that literal regardless of what pipeline_fallback_name_pattern is configured — the module-level pattern search is never even reached."""
    (tmp_path / "agent_b.py").write_text(
        "from schemas import validate_section\n\n"
        "class BAgent:\n"
        "    def fallback(self) -> dict:\n"
        "        return {'result': 'from_class_method'}\n\n"
        "    async def run(self, snapshot_id: str) -> dict:\n"
        "        result = {}\n"
        "        validate_section('B', result)\n"
        "        return result\n",
        encoding="utf-8",
    )
    (tmp_path / "agent_pipeline.py").write_text(
        "def _b_totally_different_name() -> dict:\n"
        "    return {'result': 'should_never_be_used'}\n",
        encoding="utf-8",
    )
    asts = _parse_files([tmp_path / "agent_b.py", tmp_path / "agent_pipeline.py"])
    comp = Component(id="b", role="auditor", entry_point="agent_b:BAgent", file="agent_b.py")

    custom = ContractConventions(pipeline_fallback_name_pattern="_b_totally_different_name")
    _, output, _, _, _ = harvest_component_contract(comp, asts, tmp_path, conventions=custom)
    assert output is not None and output.fallback_literal == {"result": "from_class_method"}


def test_typeddict_required_field_in_total_false(tmp_path: Path) -> None:
    """PEP 655 Required[] fields must be included in `required` even under total=False."""
    (tmp_path / "td.py").write_text(
        "from typing import TypedDict\n"
        "from typing_extensions import Required\n\n"
        "class PartialSection(TypedDict, total=False):\n"
        "    optional_field: str\n"
        "    must_have: Required[str]\n",
        encoding="utf-8",
    )
    asts = _parse_files([tmp_path / "td.py"])
    ctx = _SchemaResolveCtx(asts=asts, files_root=tmp_path, visited=set(), depth=0, conventions=None)
    found = _resolve_class_schema("PartialSection", ctx)
    assert found is not None
    schema, _ = found
    assert schema["required"] == ["must_have"]
    assert "optional_field" not in schema["required"]


def test_constants_harvested_from_annotated_assignment(tmp_path: Path) -> None:
    """AnnAssign constants (`NAME: Final[int] = N`) must be harvested, same as plain Assign."""
    (tmp_path / "ann.py").write_text(
        "from typing import Final\n\n"
        "MAX_STEPS: Final[int] = 5\n\n"
        "class AnnAgent:\n"
        "    ROUND_LIMIT: int = 4\n\n"
        "    async def run(self, snapshot_id: str):\n"
        "        return {}\n",
        encoding="utf-8",
    )
    asts = _parse_files([tmp_path / "ann.py"])
    comp = Component(id="ann", role="unknown", entry_point="ann:AnnAgent", file="ann.py")
    _, _, constants, _, _ = harvest_component_contract(comp, asts, tmp_path)
    assert constants == {"MAX_STEPS": 5, "ROUND_LIMIT": 4}


def test_missing_file_yields_needs_human(tmp_path: Path) -> None:
    comp = Component(id="ghost", role="unknown", entry_point="nowhere:Ghost", file="gone.py")
    invocation, output, _, notes, _ = harvest_component_contract(comp, {}, tmp_path)
    assert invocation is None and output is None
    assert any("source file not found" in n for n in notes)


def test_harvest_contracts_has_tools_and_edges(tmp_path: Path) -> None:
    files = _write_fixture(tmp_path)
    system_map = SystemMap(
        target_system_id="t",
        components=[
            _foo_component(),
            Component(id="hammer", role="tool", entry_point="myapp.tools:hammer", file="", upstream=["foo"]),
        ],
    )
    system_map.components[0].downstream = ["hammer"]
    flow = AgentFlowMap(
        target_system_id="t",
        agents=[AgentFlow(id="foo", component_ids=["foo", "hammer"])],
    )
    contracts = harvest_contracts(system_map, flow, files, tmp_path)

    contract = contracts["foo"]
    assert contract.observability.has_tools is True
    assert {"src": "foo", "dst": "hammer"} in contract.connect_edges
    assert contract.invocation is not None and contract.output is not None


def test_merge_observability_llm_fills_and_conflict_demotes() -> None:
    contract = EvaluationContract(agent_id="a")
    contract.observability.input_kind = "structured"  # static fact
    profile = AgentDataProfile(
        agent_id="a", input_kind="query", has_separable_context=False,
        context_location="inlined:user_prompt",
    )
    notes = _merge_observability(contract, profile)

    assert contract.observability.input_kind == "structured"  # static kept
    assert any("input_kind conflict" in n for n in notes)
    # has_separable_context was None (not set by static) so LLM value is adopted
    assert contract.observability.has_separable_context is False
    assert contract.observability.context_location == "inlined:user_prompt"
    assert set(contract.observability.llm_fields) == {"has_separable_context", "context_location"}

    # has_separable_context conflict: static set True, LLM says False → static kept + note
    contract_b = EvaluationContract(agent_id="b")
    contract_b.observability.has_separable_context = True  # explicit static fact
    notes_b = _merge_observability(
        contract_b, AgentDataProfile(agent_id="b", has_separable_context=False)
    )
    assert contract_b.observability.has_separable_context is True  # static kept
    assert any("has_separable_context conflict" in n for n in notes_b)
    assert "has_separable_context" not in contract_b.observability.llm_fields

    # unknown static -> LLM value adopted and attributed
    contract2 = EvaluationContract(agent_id="c")
    notes2 = _merge_observability(contract2, AgentDataProfile(agent_id="c", input_kind="query"))
    assert notes2 == []
    assert contract2.observability.input_kind == "query"
    assert "input_kind" in contract2.observability.llm_fields


def test_contract_roundtrips_through_plan_report_yaml(tmp_path: Path) -> None:
    files = _write_fixture(tmp_path)
    system_map = SystemMap(target_system_id="t", components=[_foo_component()])
    flow = AgentFlowMap(target_system_id="t", agents=[AgentFlow(id="foo", component_ids=["foo"])])
    contracts = harvest_contracts(system_map, flow, files, tmp_path)

    report = EvaluationPlanReport(
        target_system_id="t",
        agents=[AgentPlanReport(agent_id="foo", contract=contracts["foo"])],
    )
    path = tmp_path / "report.yaml"
    save_plan_report(report, path)
    loaded = load_plan_report(path)

    contract = loaded.agents[0].contract
    assert contract is not None
    assert contract.invocation is not None and contract.invocation.method == "run"
    assert contract.output is not None and contract.output.json_schema is not None


@pytest.mark.asyncio
async def test_generate_plan_agentic_attaches_contracts(tmp_path: Path) -> None:
    from agent_eval_harness.llm.client import LLMResponse
    from agent_eval_harness.llm.fake_client import FakeLLMClient
    from agent_eval_harness.planning.agentic_planner import generate_plan_agentic

    files = _write_fixture(tmp_path)
    system_map = SystemMap(target_system_id="t", components=[_foo_component()])
    flow = AgentFlowMap(target_system_id="t", agents=[AgentFlow(id="foo", role="orchestrator", component_ids=["foo"])])
    llm_client = FakeLLMClient(LLMResponse(content="{}", model="fake"))

    _, report = await generate_plan_agentic(
        system_map, flow, {"foo": AGENT_SRC}, [], llm_client,
        run_critic_pass=False, files=files, files_root=tmp_path,
    )

    contract = report.agents[0].contract
    assert contract is not None
    assert contract.invocation is not None
    assert contract.observability.input_kind == "structured"  # static wins; LLM degraded to {}


async def test_generate_plan_agentic_emits_synthetic_agent_io_gate_for_non_fan_in_agent(tmp_path: Path) -> None:
    """The gate-emission rule in reconcile() must fire for ANY archetype-classifiable agent (here FooAgent classifies as rag_single_shot via _archetype_for) — not just fan-in agents like K/L."""
    from agent_eval_harness.llm.client import LLMResponse
    from agent_eval_harness.llm.fake_client import FakeLLMClient
    from agent_eval_harness.planning.agentic_planner import generate_plan_agentic

    files = _write_fixture(tmp_path)
    system_map = SystemMap(target_system_id="t", components=[_foo_component()])
    flow = AgentFlowMap(target_system_id="t", agents=[AgentFlow(id="foo", role="orchestrator", component_ids=["foo"])])
    llm_client = FakeLLMClient(LLMResponse(content="{}", model="fake"))

    suite, report = await generate_plan_agentic(
        system_map, flow, {"foo": AGENT_SRC}, [], llm_client,
        run_critic_pass=False, files=files, files_root=tmp_path,
    )

    contract = report.agents[0].contract
    assert contract is not None
    assert not contract.field_downstream_consumers  # not a fan-in agent

    synth_entries = [
        e for e in suite.entries
        if e.dataset and e.dataset.required and e.dataset.required.get("kind") == "synthetic_agent_io"
    ]
    assert len(synth_entries) == 1
    assert synth_entries[0].agent_id == "foo"


def test_derive_case_binding_v2_arch_bundle_vs_bundle() -> None:
    """_derive_case_binding v2 fixes arch_bundle → bundle mapping (was mapped to itself; archetype vocabulary specifies `bundle`)."""
    from agent_eval_harness.mapping.builder.contract_harvest import _derive_case_binding
    from agent_eval_harness.planning.contract import KwargSpec

    kwargs = [
        KwargSpec(name="provider_id", annotation="str", default_repr=None, required=True),
        KwargSpec(name="snapshot_id", annotation="str", default_repr=None, required=True),
        KwargSpec(name="arch_bundle", annotation="dict", default_repr=None, required=False),
        KwargSpec(name="graph_summary", annotation="str", default_repr=None, required=False),
    ]

    binding = _derive_case_binding(kwargs)

    assert binding.get("arch_bundle") == "case:$.input.bundle", \
        f"arch_bundle should map to bundle case key, got {binding.get('arch_bundle')}"
    assert binding.get("graph_summary") == "case:$.input.graph_summary", \
        f"graph_summary should fallback to same-name, got {binding.get('graph_summary')}"
    assert binding.get("provider_id") == "config:provider_id", \
        f"provider_id should use config: prefix, got {binding.get('provider_id')}"
    assert binding.get("snapshot_id") == "case:$.input.snapshot_id", \
        f"snapshot_id should map to case key, got {binding.get('snapshot_id')}"


def test_derive_case_binding_v2_fallback_same_name() -> None:
    """_derive_case_binding v2 falls back to same-name mapping for kwargs not in the archetype vocabulary — this allows generic/foreign architectures to work without requiring full vocabulary."""
    from agent_eval_harness.mapping.builder.contract_harvest import _derive_case_binding
    from agent_eval_harness.planning.contract import KwargSpec

    kwargs = [
        KwargSpec(name="custom_input", annotation="str", default_repr=None, required=True),
        KwargSpec(name="query", annotation="str", default_repr=None, required=True),
        KwargSpec(name="context", annotation="dict", default_repr=None, required=False),
    ]

    binding = _derive_case_binding(kwargs)

    assert binding.get("custom_input") == "case:$.input.custom_input", \
        "Unknown kwarg should fallback to same-name"
    assert binding.get("query") == "case:$.input.query", \
        "Unknown kwarg should fallback to same-name"
    assert "context" not in binding, \
        "Optional kwarg not in vocabulary should be skipped"


def test_derive_case_binding_v2_known_mapping() -> None:
    """_derive_case_binding v2 correctly maps known archetype kwargs per the documented field-vocabulary."""
    from agent_eval_harness.mapping.builder.contract_harvest import _derive_case_binding
    from agent_eval_harness.planning.contract import KwargSpec

    kwargs = [
        KwargSpec(name="identity_output", annotation="dict", default_repr=None, required=False),
        KwargSpec(name="folder_tree", annotation="str", default_repr=None, required=False),
        KwargSpec(name="repo_name", annotation="str", default_repr=None, required=False),
        KwargSpec(name="mem_ctx", annotation="dict", default_repr=None, required=False),
    ]

    binding = _derive_case_binding(kwargs)

    assert binding.get("identity_output") == "case:$.input.project_identity_output", \
        "identity_output should map to project_identity_output"
    assert binding.get("folder_tree") == "case:$.input.folder_tree", \
        "folder_tree should map to itself"
    assert binding.get("repo_name") == "case:$.input.repo_name", \
        "repo_name should map to itself"
    assert binding.get("mem_ctx") == "case:$.input.mem_ctx", \
        "mem_ctx should map to itself"


def test_harvest_preserves_complex_type_annotations_from_ast(tmp_path: Path) -> None:
    """CS-310 Slice 4 regression: harvested kwargs must preserve real type annotations from AST (not empty from truncated signature)."""
    # Create a fixture with complex type kwargs (like StructuralGraphSummary, RetrievalBundle, etc.)
    (tmp_path / "complex_types.py").write_text(
        """
from typing import TypedDict, Optional

class GraphSummary(TypedDict):
    nodes: int
    edges: int

class ComplexAgent:
    async def run(self, provider_id: str, graph_summary: GraphSummary, threshold: Optional[float] = None) -> dict:
        return {"result": "ok"}
""",
        encoding="utf-8",
    )
    asts = _parse_files([tmp_path / "complex_types.py"])
    comp = Component(
        id="complex",
        role="unknown",
        entry_point="complex_types:ComplexAgent",
        file="complex_types.py",
    )
    invocation, _, _, _, _ = harvest_component_contract(comp, asts, tmp_path)

    assert invocation is not None
    assert invocation.method == "run"
    by_name = {k.name: k for k in invocation.kwargs}

    # Verify annotations are NOT empty (this is the regression test for CS-310)
    assert by_name["provider_id"].annotation == "str", "Primitive type should have annotation"
    assert by_name["graph_summary"].annotation == "GraphSummary", (
        "Complex type annotation must be preserved (not empty), "
        "so _resolve_input_kwarg_schemas can resolve nested schema"
    )
    assert "Optional" in by_name["threshold"].annotation, (
        "Optional wrapper must be preserved in annotation"
    )
