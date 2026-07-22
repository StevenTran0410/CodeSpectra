"""Facts compiler: gathers resolved facts and missing-fact markers for plan rendering; every render-critical field is Resolved (w/ citation) or Missing (w/ reason), never bare falsy."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent_eval_harness.discovery.agent_knowledge import AgentKnowledge
from agent_eval_harness.mapping.system_map import SystemMap
from agent_eval_harness.planning.report import EvaluationPlanReport

logger = logging.getLogger("agent_eval_harness.code_injection.facts")


class Resolved(BaseModel):
    """A fact successfully determined."""
    status: Literal["resolved"] = "resolved"
    value: Any
    citation: str = ""


class Missing(BaseModel):
    """A fact that could not be determined."""
    status: Literal["missing"] = "missing"
    fact_id: str
    reason: str


class LocationFact(BaseModel):
    """File location with line span for an agent's entry point."""
    file: str
    line_start: int
    line_end: int
    entry_method: str
    entry_line: int


class PromptSiteFact(BaseModel):
    """Location of a prompt constant or LLM call."""
    file: str
    line: int
    kind: str
    snippet: str


class ComponentRefFact(BaseModel):
    """Component reference from system map."""
    id: str
    role: str
    file: str
    line: int


class ContractArgFact(BaseModel):
    """One input parameter to an agent."""
    kwarg: str
    source_kind: str = ""
    type_hint: str = ""
    example: str = ""


class OutputContractFact(BaseModel):
    """Agent's output schema."""
    json_schema: dict[str, Any] | None = None
    schema_source: str | None = None
    fallback_literal: dict[str, Any] | None = None


class RunbookFacts(BaseModel):
    """Executable commands and endpoints for running the target."""
    start_command: str = ""
    sha256_command_posix: str = ""
    sha256_command_windows: str = ""
    endpoints: list[str] = Field(default_factory=list)


class AgentFacts(BaseModel):
    """All facts needed to render one agent's cards in REFERENCE.md and CODE.md."""
    agent_id: str
    role: str = ""

    location: Resolved | Missing | None = None
    entry_method: Resolved | Missing | None = None
    components: Resolved | Missing | None = None  # value: list[ComponentRefFact]
    input_contract: list[Resolved | Missing] = Field(default_factory=list)  # each: ContractArgFact
    output_contract: Resolved | Missing | None = None  # value: OutputContractFact
    prompt_sites: Resolved | Missing | None = None  # value: list[PromptSiteFact]
    constructor_dep_roles: Resolved | Missing | None = None  # value: list[DepRoleVerdict]

    invocation_mode: str = "unsupported"
    case_binding: Resolved | Missing | None = None  # value: dict[str, str]
    route: Resolved | Missing | None = None  # value: str

    has_retrieval_signal: bool = False
    constructor_deps: list[str] = Field(default_factory=list)
    # Case fields the entry method cannot accept — surfaced, never silently dropped.
    unconsumed_case_fields: list[str] = Field(default_factory=list)
    # Required kwargs bound to a case field no case carries — they arrive as None.
    unsatisfiable_bindings: list[str] = Field(default_factory=list)
    # Same, but optional: the call still runs, just with less context than intended.
    degraded_bindings: list[str] = Field(default_factory=list)
    required_kwargs: list[str] = Field(default_factory=list)
    # Bound kwarg -> harvested annotation, kept only when the annotation names a type a raw case value wouldn't satisfy without conversion.
    typed_bindings: dict[str, str] = Field(default_factory=dict)
    transclude: dict[str, str] = Field(default_factory=dict)
    quirk_flags: dict[str, bool] = Field(default_factory=dict)


class PlanFacts(BaseModel):
    """All facts needed to compile and render the 4-file plan."""
    session_id: str
    target_system_id: str
    branch_name: str
    plan_id: str

    runbook: RunbookFacts = Field(default_factory=RunbookFacts)

    main_entry_path: Resolved | Missing | None = None

    db_path: str = ""
    dataset_ids: list[str] = Field(default_factory=list)
    dataset_summaries: list[dict] = Field(default_factory=list)

    provider_listing: Resolved | Missing | None = None

    agents: list[AgentFacts] = Field(default_factory=list)
    batch_size: int = 4

    code_artifacts: dict[str, str] = Field(default_factory=dict)
    code_shas: dict[str, str] = Field(default_factory=dict)
    dispatch_modules: list[dict] = Field(default_factory=list)  # {agent_id, filename, code, sha}


def compile_plan_facts(
    system_map: SystemMap,
    wiring: dict,
    dataset_summaries: list[dict],
    session_id: str,
    branch_name: str,
    plan_report: EvaluationPlanReport | None = None,
    repo_root: Path | None = None,
    knowledge_dir: Path | None = None,
) -> PlanFacts:
    """Compile every fact the render needs, each Resolved (with citation) or Missing."""
    facts = PlanFacts(
        session_id=session_id,
        target_system_id=system_map.target_system_id,
        branch_name=branch_name,
        plan_id=wiring.get("plan_id", "unknown"),
        dataset_summaries=dataset_summaries,
        db_path=wiring.get("aeh_db_path", ""),
        dataset_ids=list(wiring.get("dataset_ids", [])),
    )

    facts.runbook = _harvest_runbook(repo_root or Path.cwd())

    facts.provider_listing = _harvest_provider_listing(plan_report)

    facts.main_entry_path = _harvest_main_entry_path(system_map, plan_report)

    # plan_report.agents is the authoritative agent list; SystemMap is the fallback.
    if plan_report and plan_report.agents:
        agent_specs = [(a.agent_id, a.role or "") for a in plan_report.agents]
    else:
        agent_specs = [(c.id, c.role) for c in system_map.components]

    for agent_id, agent_role in agent_specs:
        agent_facts = AgentFacts(
            agent_id=agent_id,
            role=agent_role,
        )

        knowledge: AgentKnowledge | None = None
        if knowledge_dir:
            knowledge = _load_agent_knowledge(knowledge_dir, agent_id)

        if knowledge:
            if knowledge.location:
                loc = knowledge.location
                agent_facts.location = Resolved(
                    value=LocationFact(
                        file=loc.file,
                        line_start=loc.line_start,
                        line_end=loc.line_end,
                        entry_method=loc.entry_method,
                        entry_line=loc.entry_line,
                    ),
                    citation="AgentKnowledge.location",
                )
                agent_facts.entry_method = Resolved(
                    value=loc.entry_method,
                    citation="AgentKnowledge.location.entry_method",
                )
            else:
                agent_facts.location = Missing(
                    fact_id=f"{agent_id}_location",
                    reason="AgentKnowledge.location not set",
                )

            if knowledge.input_contract:
                agent_facts.input_contract = [
                    Resolved(
                        value=ContractArgFact(
                            kwarg=arg.kwarg,
                            source_kind=arg.source_kind,
                            type_hint=arg.type_hint,
                            example=arg.example,
                        ),
                        citation="AgentKnowledge.input_contract",
                    )
                    for arg in knowledge.input_contract
                ]

            if knowledge.output_contract:
                agent_facts.output_contract = Resolved(
                    value=OutputContractFact(
                        json_schema=knowledge.output_contract.json_schema,
                        schema_source=knowledge.output_contract.schema_source,
                        fallback_literal=knowledge.output_contract.fallback_literal,
                    ),
                    citation="AgentKnowledge.output_contract",
                )

            if knowledge.prompt_sites:
                agent_facts.prompt_sites = Resolved(
                    value=[
                        PromptSiteFact(
                            file=ps.file,
                            line=ps.line,
                            kind=ps.kind,
                            snippet=ps.snippet,
                        )
                        for ps in knowledge.prompt_sites
                    ],
                    citation="AgentKnowledge.prompt_sites",
                )

            if knowledge.constructor_dep_roles:
                agent_facts.constructor_dep_roles = Resolved(
                    value=knowledge.constructor_dep_roles,
                    citation="AgentKnowledge.constructor_dep_roles",
                )

            agent_facts.transclude = {
                "functionality": knowledge.functionality,
                "constraints": "\n".join(
                    c if isinstance(c, str) else str(c) for c in knowledge.constraints
                ),
                "method_steps": "\n".join(
                    s if isinstance(s, str) else str(s) for s in knowledge.method_steps
                ),
            }
        else:
            agent_facts.location = Missing(
                fact_id=f"{agent_id}_location",
                reason="AgentKnowledge not loaded",
            )

        # AgentKnowledge.components carries file:line; SystemMap components do not.
        if knowledge and knowledge.components:
            agent_facts.components = Resolved(
                value=[
                    ComponentRefFact(id=c.id, role=c.role, file=c.file, line=c.line)
                    for c in knowledge.components
                ],
                citation="AgentKnowledge.components",
            )
        else:
            comp = system_map.component_by_id(agent_id)
            if comp:
                agent_facts.components = Resolved(
                    value=[ComponentRefFact(id=comp.id, role=comp.role, file=comp.file, line=0)],
                    citation="SystemMap.components",
                )

        if plan_report:
            # Note: this reads the frozen plan_report; virtual: bindings depend on _merge_virtual_input_bindings having run upstream
            plan_agent = next(
                (a for a in plan_report.agents if a.agent_id == agent_id), None
            )
            if plan_agent and plan_agent.contract:
                invocation = plan_agent.contract.invocation
                if invocation:
                    agent_facts.invocation_mode = invocation.invocation_mode
                    if invocation.case_binding:
                        agent_facts.case_binding = Resolved(
                            value=invocation.case_binding,
                            citation="EvaluationContract.invocation.case_binding",
                        )
                    if invocation.route:
                        agent_facts.route = Resolved(
                            value=invocation.route,
                            citation="EvaluationContract.invocation.route",
                        )
                    agent_facts.constructor_deps = list(invocation.constructor_deps)
                    agent_facts.required_kwargs = [
                        k.name for k in invocation.kwargs if k.required
                    ]
                    bound = invocation.case_binding or {}
                    agent_facts.typed_bindings = {
                        k.name: k.annotation
                        for k in invocation.kwargs
                        if k.name in bound and _annotation_needs_conversion(k.annotation)
                    }
                agent_facts.has_retrieval_signal = plan_agent.contract.has_retrieval_signal

        facts.agents.append(agent_facts)

    _mark_unconsumed_case_fields(facts)
    _compile_code_artifacts(facts, wiring)

    return facts


_PLAIN_ANNOTATIONS = {
    "str", "int", "float", "bool", "bytes", "none", "any", "object",
    "dict", "list", "tuple", "set", "frozenset", "mapping", "sequence", "iterable",
    "optional", "union", "literal",
}


def _annotation_needs_conversion(annotation: str | None) -> bool:
    """True when the annotation names a project type (not a builtin/typing/container name) that a raw JSON case value wouldn't satisfy, e.g. `RetrievalBundle | None` or `list[Evidence]`, but not `Optional[bool]`."""
    if not annotation:
        return False
    names = re.findall(r"[A-Za-z_][\w.]*", annotation)
    return any(name.rsplit(".", 1)[-1].lower() not in _PLAIN_ANNOTATIONS for name in names)


def _mark_unconsumed_case_fields(facts: PlanFacts) -> None:
    """Flag case fields no kwarg can carry, so the plan surfaces them instead of silently dropping them (a generator can emit context the entry method never accepts)."""
    from agent_eval_harness.datasets.archetype_vocabulary import EVIDENCE_CASE_KEY, VIRTUAL_BINDING_PREFIX

    by_agent: dict[str, set[str]] = {}
    for ds in facts.dataset_summaries:
        example = ds.get("example_case") or {}
        agent_id = ((example.get("labels") or {}).get("agent_id")) or ""
        if agent_id:
            by_agent.setdefault(agent_id, set()).update(
                k for k in (example.get("input") or {}) if k not in ("kind", "shape")
            )

    for agent in facts.agents:
        present = by_agent.get(agent.agent_id)
        if not present:
            continue
        bound: set[str] = set()
        missing_required: set[str] = set()
        missing_optional: set[str] = set()
        if isinstance(agent.case_binding, Resolved) and agent.case_binding.value:
            for kwarg, expr in agent.case_binding.value.items():
                if not isinstance(expr, str):
                    continue
                if expr.startswith("case:"):
                    field = expr.rsplit(".", 1)[-1]
                    bound.add(field)
                    if field in present:
                        continue
                    # A required kwarg bound to a missing field arrives as None and can fail silently if swallowed; optional ones just degrade quietly.
                    if kwarg in agent.required_kwargs:
                        missing_required.add(field)
                    else:
                        missing_optional.add(field)
                elif expr.startswith(VIRTUAL_BINDING_PREFIX):
                    # Virtual input field: extract the case key name from "virtual:bundle"
                    field = expr[len(VIRTUAL_BINDING_PREFIX):]
                    bound.add(field)
        if agent.has_retrieval_signal:
            bound.add(EVIDENCE_CASE_KEY)  # reaches the agent through the retrieval stub
        agent.unconsumed_case_fields = sorted(present - bound)
        agent.unsatisfiable_bindings = sorted(missing_required)
        agent.degraded_bindings = sorted(missing_optional)


def _compile_code_artifacts(facts: PlanFacts, wiring: dict) -> None:
    """Load the static injected files, generate per-agent dispatch, and hash everything."""
    import hashlib

    from agent_eval_harness.code_injection import codegen

    templates_dir = Path(__file__).resolve().parent / "templates"

    def _store(name: str, code: str) -> None:
        facts.code_artifacts[name] = code
        facts.code_shas[name] = hashlib.sha256(code.encode()).hexdigest()

    for name, fname in [("tracer", "tracer.py"), ("run_eval", "run_eval.py"),
                        ("aeh_eval", "aeh_eval.py"), ("target_init", "target_init.py")]:
        p = templates_dir / fname
        _store(name, p.read_text(encoding="utf-8") if p.exists() else f"# MISSING TEMPLATE: {fname}")

    # Same batching TASKS.md renders, so `--batch N` selects exactly what gate N checks.
    bs = max(1, facts.batch_size)
    agent_ids = [a.agent_id for a in facts.agents]
    wiring["batches"] = [agent_ids[i:i + bs] for i in range(0, len(agent_ids), bs)]
    _store("wiring", json.dumps(wiring, indent=2, sort_keys=True, default=str))

    stub = ""
    for a in facts.agents:
        if a.has_retrieval_signal:
            stub = codegen.generate_retrieval_stub(a)
            if stub:
                break
    _store("retrieval_stub", stub)

    for a in facts.agents:
        code, sha = codegen.generate_dispatch_module(a)
        facts.dispatch_modules.append({
            "agent_id": a.agent_id,
            "filename": f".aeh/dispatch/{a.agent_id}.py",
            "code": code,
            "sha": sha,
        })

    # Entrypoint edits are rendered semantically — a wrong literal anchor would send the agent to a line that doesn't exist.


def _harvest_runbook(repo_root: Path) -> RunbookFacts:
    """Harvest start command and sha256 commands from scripts/Makefile/pyproject."""
    facts = RunbookFacts()

    facts.sha256_command_posix = "sha256sum <file>"
    facts.sha256_command_windows = "certUtil -hashfile <file> SHA256"


    return facts


def _harvest_provider_listing(plan_report: EvaluationPlanReport | None) -> Resolved | Missing | None:
    """Harvest provider/model listing from plan report or mark missing."""
    if not plan_report:
        return Missing(fact_id="provider_listing", reason="No plan_report provided")

    return Missing(fact_id="provider_listing", reason="Provider/model list not yet harvested")


def _harvest_main_entry_path(system_map: SystemMap, plan_report: EvaluationPlanReport | None) -> Resolved | Missing | None:
    """Always Missing — the SystemMap maps agents only, and guessing produced a wrong file."""
    return Missing(
        fact_id="main_entry_path",
        reason="server entrypoint is not part of the agent map — locate it in the target repo",
    )


def _load_agent_knowledge(knowledge_dir: Path, agent_id: str) -> AgentKnowledge | None:
    """Load an AgentKnowledge JSON sidecar for one agent, or None if absent/invalid."""
    path = knowledge_dir / f"{agent_id}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return AgentKnowledge.model_validate(data)
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return None
