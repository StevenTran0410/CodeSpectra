"""Stage 3 — DAG LLM orchestrator (CS-265 redesign): per-agent nodes fan out, handoff_gates fans in, reconcile joins against the baseline, critic reviews. See aeh_stage3_agentic_eval_planner_plan.md §4."""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import difflib

from agent_eval_harness.llm.client import LLMClient, LLMMessage
from agent_eval_harness.mapping.agent_flow import AgentFlow, AgentFlowMap
from agent_eval_harness.mapping.system_map import Component, SystemMap
from agent_eval_harness.metrics.assertions.registry import get_assertion, ASSERTIONS, _import_all as _import_all_assertions
from agent_eval_harness.metrics.registry import DATASET_KINDS, get_spec, validate_metric
from agent_eval_harness.metrics.suite import DatasetRef, Suite, SuiteEntry
from agent_eval_harness.planning.planner import baseline_gates_for_agent
from agent_eval_harness.planning.contract import EvaluationContract
from agent_eval_harness.planning.report import (
    AgentDataProfile,
    AgentPlanReport,
    EvaluationGate,
    EvaluationPlanReport,
)
from agent_eval_harness.store import repository

logger = logging.getLogger("agent_eval_harness.planning.agentic_planner")

# ragas.context_precision removed — no dispatch handler (CS-281 §2).
_KNOWN_RAGAS_METRICS = {"ragas.faithfulness", "ragas.answer_relevancy"}
_BASELINE_HANDOFF_METRICS = {
    "allowed_downstream", "max_items_per_call", "max_retries", "retry_on_reject_required",
}
MAX_LLM_JUDGE_GATES_PER_AGENT = 3  # §5 cap; baseline judges exempt

ANALYST_SYSTEM = (
    "You are an AI evaluation-planning analyst. Given one agent's real source code, its "
    "role hint, and its declared upstream/downstream agent context, describe its ACTUAL "
    "data flow — read the code, do not guess from the role label alone. Also cross-check: "
    "does the code's real behavior match the declared upstream/downstream/constraints? "
    "Surface any mismatch, do not silently resolve it.\n"
    'Return JSON only: {"input_data": "<what this agent receives, and from where>", '
    '"output_data": "<what this agent produces, and to where>", '
    '"internal_tools": ["<tool/helper: what it returns>", ...], '
    '"failure_modes": ["<a concrete, specific way this agent could produce bad output>", ...], '
    '"consistency_notes": ["<a specific code-vs-declared-flow mismatch>", ...] (empty list if none), '
    '"input_kind": "query|structured|unknown" (query = takes a free-text user question; '
    "structured = takes ids/config like snapshot_id), "
    '"has_separable_context": true|false (true ONLY if retrieved context lives in a distinct '
    "field/variable that instrumentation could capture separately; false if evidence is "
    "inlined into a prompt string), "
    '"context_location": "<where the context physically lives, e.g. inlined:user_prompt>" or null}'
)

GATE_DESIGNER_SYSTEM = (
    "You are an AI evaluation-planning gate designer. Given one agent's data-flow profile, "
    "its owned components, and the evaluation gates ALREADY covered by deterministic rules "
    "(do not duplicate these — only propose genuinely new gates for real failure modes not "
    "already covered), propose evaluation gates at this agent's `input`, `output`, and "
    "`internal_tool` locations.\n"
    "Toolkit selection — pick the CHEAPEST that can decide the property (proposing an "
    "llm_judge where a trace assertion would do is a review-blocking error):\n"
    "- assertion: deterministic, decidable from a trace. metric must be one of the known "
    "registered assertion names given to you.\n"
    "- classifier: labeled-data confusion matrix. metric = \"classifier.<component_id>_<property>\". "
    "Requires params.entry_point and a labeled dataset.\n"
    "- ragas: grounding/retrieval quality only. ONLY valid metrics: ragas.faithfulness "
    "(requires separable context — has_separable_context=true), "
    "ragas.answer_relevancy (requires query-shaped input — input_kind=query). "
    "Do NOT propose ragas gates when these preconditions are false.\n"
    "- deepeval: a flexible judge for any other describable quality. metric = "
    "\"geval.<short_name>\"; you MUST include \"rubric_text\" tailored to this agent's actual "
    "prompt/goal when toolkit=deepeval.\n"
    "`component` MUST be one of this agent's own component ids given to you.\n"
    f"dataset_kind MUST be one of: {sorted(DATASET_KINDS)} — any other value will be downgraded to needs_human.\n"
    'Return JSON only: {"gates": [{"component": "<id>", "location": '
    '"input|output|internal_tool", "property": "<quality checked>", "metric": "<name>", '
    '"metric_class": "assertion|classifier|llm_judge", "toolkit": '
    '"assertion|classifier|ragas|deepeval", "rationale": "<tied to a real failure mode>", '
    '"rubric_text": "<only if toolkit=deepeval>", "dataset_kind": "<optional>"}]}'
)

HANDOFF_GATES_SYSTEM = (
    "You are an AI evaluation-planning gate designer for CROSS-AGENT handoffs. Given every "
    "agent's data-flow profile plus the declared agent-to-agent flow, propose `handoff`-"
    "location gates: fan-out/routing limits, allowed-downstream correctness, and "
    "retry-on-reject behavior between two agents. Prefer assertions wherever the trace can "
    "decide it. `component` MUST be one of the SENDING agent's own component ids.\n"
    'Return JSON only: {"gates": [{"agent_id": "<sending agent id>", "component": "<its '
    'component id>", "property": "<quality checked>", "metric": "<name>", "metric_class": '
    '"assertion|classifier|llm_judge", "toolkit": "assertion|classifier|ragas|deepeval", '
    '"rationale": "<why>"}]}'
)

CRITIC_SYSTEM = (
    "You are reviewing a completed evaluation plan for an agentic system. Given a per-agent "
    "summary of proposed gates, flag: (a) any agent with zero gates, (b) any llm_judge "
    "proposed where a cheaper assertion/classifier would suffice, (c) any obvious coverage "
    "gap in the plan as a whole. Be specific and brief.\n"
    'Return JSON only: {"notes": ["<short actionable note>", ...]} (empty list if the plan '
    "looks solid)"
)


# ──────────────────────────────────────────────────────────────────────────────
# Tiny internal DAG runner
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class DagNode:
    name: str
    deps: list[str]
    run: Callable[[dict[str, Any]], Awaitable[Any]]


async def run_dag(nodes: list[DagNode]) -> dict[str, Any]:
    """Runs nodes whose deps are satisfied concurrently, round by round."""
    by_name = {n.name: n for n in nodes}
    results: dict[str, Any] = {}
    remaining = dict(by_name)

    while remaining:
        ready = [n for n in remaining.values() if all(d in results for d in n.deps)]
        if not ready:
            raise RuntimeError(
                f"agentic_planner DAG stuck — unresolved deps or a cycle among: "
                f"{sorted(remaining)}"
            )
        outputs = await asyncio.gather(*(n.run(results) for n in ready))
        for node, output in zip(ready, outputs):
            results[node.name] = output
            del remaining[node.name]

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Evidence gathering (root node)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class AgentEvidence:
    agent: AgentFlow
    owned: list[dict] = field(default_factory=list)  # per-component evidence dicts
    supporting_files: list[str] = field(default_factory=list)


def _owned_component_dict(component: Component, snippet: str) -> dict:
    return {
        "id": component.id,
        "role": component.role,
        "model": component.model,
        "entry_point": component.entry_point,
        "file": component.file,
        "upstream": component.upstream,
        "downstream": component.downstream,
        "constraints": [f"{c.name}={c.value}" for c in component.constraints],
        "source": snippet or "(no source available)",
    }


def _supporting_files_by_agent(
    agent_flow_map: AgentFlowMap,
    system_map: SystemMap,
    accepted_edges: list[dict],
) -> dict[str, list[str]]:
    """Reuses LLM-1's expansion BFS edges to find each agent's supporting files that never earned a component identity — same join Stage 2's AgentSubGraphPanel does."""
    components_by_id = {c.id: c for c in system_map.components}
    owned_file_to_agent: dict[str, str] = {}
    for agent in agent_flow_map.agents:
        for cid in agent.component_ids:
            component = components_by_id.get(cid)
            if component and component.file:
                owned_file_to_agent[component.file] = agent.id

    result: dict[str, list[str]] = {a.id: [] for a in agent_flow_map.agents}
    seen: dict[str, set[str]] = {a.id: set() for a in agent_flow_map.agents}

    for edge in accepted_edges:
        src, dst = edge.get("src"), edge.get("dst")
        if not src or not dst:
            continue
        for owned_file, other_file in ((src, dst), (dst, src)):
            owning_agent = owned_file_to_agent.get(owned_file)
            if not owning_agent or other_file in owned_file_to_agent:
                continue
            if other_file not in seen[owning_agent]:
                seen[owning_agent].add(other_file)
                result[owning_agent].append(other_file)

    return result


async def gather_evidence(
    system_map: SystemMap,
    agent_flow_map: AgentFlowMap,
    source_by_component: dict[str, str],
    accepted_edges: list[dict],
    llm_client: LLMClient,
) -> tuple[dict[str, AgentEvidence], dict[str, list[SuiteEntry]]]:
    """Root DAG node: deterministic evidence assembly + the rule-based baseline."""
    try:
        db_datasets = await repository.list_dataset_ids()
    except Exception:
        db_datasets = []

    supporting = _supporting_files_by_agent(agent_flow_map, system_map, accepted_edges)
    components_by_id = {c.id: c for c in system_map.components}

    evidence_by_agent: dict[str, AgentEvidence] = {}
    baseline_by_agent: dict[str, list[SuiteEntry]] = {}

    for agent in agent_flow_map.agents:
        owned = []
        for cid in agent.component_ids:
            component = components_by_id.get(cid)
            if component is None:
                continue
            owned.append(_owned_component_dict(component, source_by_component.get(cid, "")))

        evidence_by_agent[agent.id] = AgentEvidence(
            agent=agent, owned=owned, supporting_files=supporting.get(agent.id, [])
        )
        baseline_by_agent[agent.id] = await baseline_gates_for_agent(
            agent.component_ids, system_map, db_datasets, llm_client, agent_id=agent.id
        )

    return evidence_by_agent, baseline_by_agent


def _evidence_user_prompt(evidence: AgentEvidence) -> str:
    agent = evidence.agent
    lines = [
        f"Agent: {agent.id} (role_hint={agent.role})",
        f"Label: {agent.label}",
        f"Summary: {agent.summary}",
        f"Upstream agents: {agent.upstream_agents}",
        f"Downstream agents: {agent.downstream_agents}",
        f"Supporting/tool files owned (no component identity of their own): {evidence.supporting_files}",
        "",
        "Owned components:",
    ]
    for comp in evidence.owned:
        lines.append(f"### component: {comp['id']} (role={comp['role']}, model={comp['model']})")
        lines.append(f"file: {comp['file']}")
        lines.append(f"upstream: {comp['upstream']}  downstream: {comp['downstream']}")
        lines.append(f"constraints: {', '.join(comp['constraints']) or 'none'}")
        lines.append("source:")
        lines.append(comp["source"])
        lines.append("")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# analyst[i] — per-agent data-flow profile (parallel fan-out)
# ──────────────────────────────────────────────────────────────────────────────


async def _run_analyst(agent_id: str, evidence: AgentEvidence, llm_client: LLMClient) -> AgentDataProfile:
    try:
        response = await llm_client.complete(
            [
                LLMMessage(role="system", content=ANALYST_SYSTEM),
                LLMMessage(role="user", content=_evidence_user_prompt(evidence)),
            ],
            max_tokens=1536,
            json_mode=True,
        )
        parsed = json.loads(response.content)
        if not isinstance(parsed, dict):
            raise ValueError("analyst response was not a JSON object")
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning(f"agentic_planner: analyst[{agent_id}] unparseable response: {e}")
        return AgentDataProfile(agent_id=agent_id)

    def _str_list(key: str) -> list[str]:
        val = parsed.get(key, [])
        return [v for v in val if isinstance(v, str)] if isinstance(val, list) else []

    input_kind = parsed.get("input_kind")
    has_ctx = parsed.get("has_separable_context")
    ctx_loc = parsed.get("context_location")
    return AgentDataProfile(
        agent_id=agent_id,
        input_data=parsed.get("input_data") if isinstance(parsed.get("input_data"), str) else "",
        output_data=parsed.get("output_data") if isinstance(parsed.get("output_data"), str) else "",
        internal_tools=_str_list("internal_tools"),
        failure_modes=_str_list("failure_modes"),
        consistency_notes=_str_list("consistency_notes"),
        input_kind=input_kind if input_kind in ("query", "structured", "unknown") else None,
        has_separable_context=has_ctx if isinstance(has_ctx, bool) else None,
        context_location=ctx_loc if isinstance(ctx_loc, str) and ctx_loc else None,
    )


# ──────────────────────────────────────────────────────────────────────────────
# agent_gates[i] — per-agent gate design (parallel fan-out; depends on analyst[i])
# ──────────────────────────────────────────────────────────────────────────────


def _known_assertion_names() -> list[str]:
    _import_all_assertions()
    return sorted(ASSERTIONS)


def _validate_metric(metric: str, metric_class: str) -> bool:
    """Delegate to the unified METRIC_REGISTRY (CS-281 §2)."""
    return validate_metric(metric, metric_class)


def _parse_gates(raw: Any, *, valid_toolkits: frozenset[str] = frozenset({"assertion", "classifier", "ragas", "deepeval"})) -> list[dict]:
    if not isinstance(raw, dict):
        return []
    gates = raw.get("gates", [])
    if not isinstance(gates, list):
        return []
    out = []
    for g in gates:
        if not isinstance(g, dict):
            continue
        if not isinstance(g.get("component"), str) or not isinstance(g.get("metric"), str):
            continue
        if g.get("metric_class") not in ("assertion", "classifier", "llm_judge"):
            continue
        if g.get("toolkit") not in valid_toolkits:
            continue
        # CS-281 §2: closed dataset_kind vocabulary — unknown kind → needs_human
        dk = g.get("dataset_kind")
        if dk and dk not in DATASET_KINDS:
            g = dict(g, status="needs_human", dataset_kind=dk)
        out.append(g)
    return out


async def _run_gate_designer(
    agent_id: str,
    evidence: AgentEvidence,
    profile: AgentDataProfile,
    baseline: list[SuiteEntry],
    llm_client: LLMClient,
) -> list[EvaluationGate]:
    owned_ids = {c["id"] for c in evidence.owned}
    already_covered = [f"{e.component}.{e.metric}" for e in baseline]

    user_prompt = (
        _evidence_user_prompt(evidence)
        + "\n\nData-flow profile:\n"
        + f"input_data: {profile.input_data}\n"
        + f"output_data: {profile.output_data}\n"
        + f"internal_tools: {profile.internal_tools}\n"
        + f"failure_modes: {profile.failure_modes}\n"
        + f"\nAlready covered by deterministic rules (do not duplicate): {already_covered}\n"
        + f"Known registered assertion names: {_known_assertion_names()}"
    )

    try:
        response = await llm_client.complete(
            [
                LLMMessage(role="system", content=GATE_DESIGNER_SYSTEM),
                LLMMessage(role="user", content=user_prompt),
            ],
            max_tokens=2048,
            json_mode=True,
        )
        parsed_gates = _parse_gates(json.loads(response.content))
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning(f"agentic_planner: agent_gates[{agent_id}] unparseable response: {e}")
        return []

    result: list[EvaluationGate] = []
    for i, g in enumerate(parsed_gates):
        if g["component"] not in owned_ids:
            continue
        if g.get("location") not in ("input", "output", "internal_tool"):
            continue
        params = g.get("params") if isinstance(g.get("params"), dict) else {}
        if g["toolkit"] == "deepeval" and isinstance(g.get("rubric_text"), str):
            params = {**params, "rubric_text": g["rubric_text"]}
        valid = _validate_metric(g["metric"], g["metric_class"])
        result.append(
            EvaluationGate(
                id=f"{g['component']}.{g['metric']}.llm{i}",
                agent_id=agent_id,
                component=g["component"],
                location=g["location"],
                property=g.get("property", "") if isinstance(g.get("property"), str) else "",
                metric=g["metric"],
                metric_class=g["metric_class"],
                toolkit=g["toolkit"],
                params=params,
                dataset_kind=g.get("dataset_kind") if isinstance(g.get("dataset_kind"), str) else None,
                rationale=g.get("rationale", "") if isinstance(g.get("rationale"), str) else "",
                provenance="llm_suggested",
                status=None if valid else "needs_human",
            )
        )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# handoff_gates — cross-agent join (fan-in of ALL analyst[*])
# ──────────────────────────────────────────────────────────────────────────────


async def _run_handoff_gates(
    evidence_by_agent: dict[str, AgentEvidence],
    profiles_by_agent: dict[str, AgentDataProfile],
    llm_client: LLMClient,
) -> list[EvaluationGate]:
    lines = []
    for agent_id, evidence in evidence_by_agent.items():
        profile = profiles_by_agent.get(agent_id, AgentDataProfile(agent_id=agent_id))
        lines.append(
            f"Agent {agent_id}: downstream_agents={evidence.agent.downstream_agents} "
            f"component_ids={[c['id'] for c in evidence.owned]} "
            f"output_data={profile.output_data}"
        )
    user_prompt = "\n".join(lines)

    try:
        response = await llm_client.complete(
            [
                LLMMessage(role="system", content=HANDOFF_GATES_SYSTEM),
                LLMMessage(role="user", content=user_prompt),
            ],
            max_tokens=2048,
            json_mode=True,
        )
        raw = json.loads(response.content)
        gates = raw.get("gates", []) if isinstance(raw, dict) else []
        if not isinstance(gates, list):
            gates = []
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning(f"agentic_planner: handoff_gates unparseable response: {e}")
        return []

    result: list[EvaluationGate] = []
    for i, g in enumerate(gates):
        if not isinstance(g, dict):
            continue
        agent_id = g.get("agent_id")
        component = g.get("component")
        if not isinstance(agent_id, str) or agent_id not in evidence_by_agent:
            continue
        owned_ids = {c["id"] for c in evidence_by_agent[agent_id].owned}
        if not isinstance(component, str) or component not in owned_ids:
            continue
        metric, metric_class, toolkit = g.get("metric"), g.get("metric_class"), g.get("toolkit")
        if not isinstance(metric, str) or metric_class not in ("assertion", "classifier", "llm_judge"):
            continue
        if toolkit not in ("assertion", "classifier", "ragas", "deepeval"):
            continue
        valid = _validate_metric(metric, metric_class)
        result.append(
            EvaluationGate(
                id=f"{component}.{metric}.handoff{i}",
                agent_id=agent_id,
                component=component,
                location="handoff",
                property=g.get("property", "") if isinstance(g.get("property"), str) else "",
                metric=metric,
                metric_class=metric_class,
                toolkit=toolkit,
                rationale=g.get("rationale", "") if isinstance(g.get("rationale"), str) else "",
                provenance="llm_suggested",
                status=None if valid else "needs_human",
            )
        )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# reconcile — deterministic fan-in join
# ──────────────────────────────────────────────────────────────────────────────


def _gate_to_suite_entry(gate: EvaluationGate) -> SuiteEntry:
    # Prefer the explicit gate.dataset; fall back to dataset_kind-based required block.
    if gate.dataset is not None:
        dataset_ref = gate.dataset
    elif gate.dataset_kind:
        dataset_ref = DatasetRef(required={"kind": gate.dataset_kind, "min_cases": 20})
    else:
        dataset_ref = None
    # CS-281 §6: classifier gates run in entrypoint mode (CS-284 codegen key).
    execution = "entrypoint" if gate.metric_class == "classifier" else "harness"
    return SuiteEntry(
        id=gate.id,
        component=gate.component,
        metric=gate.metric,
        metric_class=gate.metric_class,
        execution=execution,
        dataset=dataset_ref,
        params=gate.params,
        rationale=gate.rationale,
        provenance=gate.provenance,
        status=gate.status,
        agent_id=gate.agent_id,
    )


def _merge_observability(
    contract: EvaluationContract, profile: AgentDataProfile | None
) -> list[str]:
    """Analyst fills what statics can't; a static-vs-LLM conflict demotes to needs_human, never overrides."""
    notes: list[str] = []
    if profile is None:
        return notes
    obs = contract.observability
    if profile.input_kind is not None:
        if obs.input_kind == "unknown":
            obs.input_kind = profile.input_kind  # type: ignore[assignment]
            obs.llm_fields.append("input_kind")
        elif profile.input_kind != obs.input_kind:
            notes.append(
                f"input_kind conflict: static={obs.input_kind} llm={profile.input_kind} — static kept"
            )
    if profile.has_separable_context is not None:
        if obs.has_separable_context is None:
            # Static harvest did not determine this — LLM fills.
            obs.has_separable_context = profile.has_separable_context
            obs.llm_fields.append("has_separable_context")
        elif profile.has_separable_context != obs.has_separable_context:
            # Static fact exists and LLM disagrees — static wins, surface conflict.
            notes.append(
                f"has_separable_context conflict: static={obs.has_separable_context} "
                f"llm={profile.has_separable_context} — static kept"
            )
    if profile.context_location is not None:
        obs.context_location = profile.context_location
        obs.llm_fields.append("context_location")
    return notes


# ──────────────────────────────────────────────────────────────────────────────
# CS-281 §4 — params completion pass
# ──────────────────────────────────────────────────────────────────────────────

def _derive_expected_tools(component: Component, system_map: SystemMap) -> list[str]:
    """Downstream tool component names for tool_correctness / no_unnecessary_calls."""
    from agent_eval_harness.planning.planner import get_tool_name_from_component
    components_by_id = {c.id: c for c in system_map.components}
    tool_comps = []
    for d in component.downstream:
        if d in components_by_id and components_by_id[d].role == "tool":
            tool_comps.append(components_by_id[d])
    for c in system_map.components:
        if c.role == "tool" and component.id in c.upstream and c not in tool_comps:
            tool_comps.append(c)
    return [get_tool_name_from_component(tc) for tc in tool_comps]


def _complete_params(
    gate: EvaluationGate,
    system_map: SystemMap | None,
    contract: EvaluationContract | None,
    report_notes: list[str],
) -> EvaluationGate:
    """Fill registry-required params from the CS-287 contract / system map.
    Returns a (possibly new) gate. Sets status='needs_human' when underivable."""
    if system_map is None and contract is None:
        return gate

    params = dict(gate.params)
    spec = get_spec(gate.metric)
    needs_human = gate.status == "needs_human"

    component_obj = (
        system_map.component_by_id(gate.component) if system_map else None
    )

    # allowed_downstream.allowed ← contract connect_edges (citation; sweep fallback demoted)
    if gate.metric == "allowed_downstream" and "allowed" not in params:
        if contract is not None and contract.connect_edges:
            downstream = [
                e["dst"] for e in contract.connect_edges if e.get("src") == gate.component
            ]
            if downstream:
                params["allowed"] = downstream
        elif component_obj and system_map:
            params["allowed"] = component_obj.downstream  # system_map fallback

    # tool_correctness.expected_tools ← downstream tool components
    if gate.metric == "tool_correctness" and "expected_tools" not in params:
        if component_obj and system_map:
            tools = _derive_expected_tools(component_obj, system_map)
            params["expected_tools"] = tools

    # arg_schema → replace with schema_valid seeded from contract output schema
    if gate.metric == "arg_schema" and contract is not None:
        if contract.output and contract.output.json_schema:
            gate = gate.model_copy(update={
                "metric": "schema_valid",
                "params": {**params, "schema": contract.output.json_schema,
                           "schema_citation": contract.output.schema_source or ""},
            })
            report_notes.append(
                f"{gate.id}: arg_schema replaced by schema_valid "
                f"(contract schema: {contract.output.schema_source})"
            )
            return gate

    # Check required_params still missing — mark needs_human
    if spec:
        for rp in spec.required_params:
            val = params.get(rp)
            if not val or (isinstance(val, str) and not val.strip()):
                needs_human = True
                report_notes.append(
                    f"{gate.id}: required param '{rp}' for metric '{gate.metric}' "
                    "could not be derived — needs_human"
                )
                break

    return gate.model_copy(update={"params": params, "status": "needs_human" if needs_human else gate.status})


# ──────────────────────────────────────────────────────────────────────────────
# CS-281 §4 — feasibility pass
# ──────────────────────────────────────────────────────────────────────────────

def _apply_feasibility(
    gates: list[EvaluationGate],
    contract: EvaluationContract | None,
    agent_id: str,
    report_notes: list[str],
) -> list[EvaluationGate]:
    """Evaluate meaningless_when against the CS-287 contract; replace/drop/needs_human.
    Baseline gates (provenance='rule') are NEVER dropped.
    LLM-only observability flags (in llm_fields) demote to needs_human; static flags execute."""
    if contract is None:
        return gates

    obs = contract.observability
    llm_fields = set(obs.llm_fields)
    result: list[EvaluationGate] = []
    seen_metrics: set[tuple[str, str]] = {(g.component, g.metric) for g in gates}

    for gate in gates:
        spec = get_spec(gate.metric)
        if spec is None or gate.provenance == "rule":
            result.append(gate)
            continue

        # ragas.faithfulness / ragas.*context* on !has_separable_context
        if gate.metric in ("ragas.faithfulness",) or "context" in gate.metric:
            static_false = obs.has_separable_context is False and "has_separable_context" not in llm_fields
            llm_false = obs.has_separable_context is False and "has_separable_context" in llm_fields
            if static_false:
                report_notes.append(
                    f"{gate.id}: {gate.metric} replaced by referential_integrity + "
                    "grounding_judge_span_prompt (has_separable_context=False, static)"
                )
                # Replace
                ri_id = f"{gate.component}.referential_integrity.feasibility"
                gj_id = f"{gate.component}.grounding_judge_span_prompt.feasibility"
                if (gate.component, "referential_integrity") not in seen_metrics:
                    seen_metrics.add((gate.component, "referential_integrity"))
                    result.append(gate.model_copy(update={
                        "id": ri_id, "metric": "referential_integrity",
                        "metric_class": "assertion", "toolkit": "assertion",
                        "rationale": f"Replaced {gate.metric}: context inlined in prompt",
                        "status": None,
                    }))
                if (gate.component, "grounding_judge_span_prompt") not in seen_metrics:
                    seen_metrics.add((gate.component, "grounding_judge_span_prompt"))
                    result.append(gate.model_copy(update={
                        "id": gj_id, "metric": "grounding_judge_span_prompt",
                        "metric_class": "llm_judge", "toolkit": "deepeval",
                        "params": {"rubric_text": "Evaluate whether the response is grounded in the input context."},
                        "rationale": f"Replaced {gate.metric}: context inlined in prompt",
                        "status": None,
                    }))
                continue
            elif llm_false:
                report_notes.append(
                    f"{gate.id}: {gate.metric} has_separable_context=False (LLM-only) — demoted to needs_human"
                )
                result.append(gate.model_copy(update={"status": "needs_human"}))
                continue

        # ragas.answer_relevancy on input_kind != query
        if gate.metric == "ragas.answer_relevancy":
            static_non_query = obs.input_kind != "query" and "input_kind" not in llm_fields
            llm_non_query = obs.input_kind != "query" and "input_kind" in llm_fields
            if obs.input_kind == "unknown":
                pass  # no evidence to act on
            elif static_non_query:
                report_notes.append(
                    f"{gate.id}: ragas.answer_relevancy dropped (input_kind={obs.input_kind}, static)"
                )
                continue  # drop
            elif llm_non_query:
                report_notes.append(
                    f"{gate.id}: ragas.answer_relevancy demoted to needs_human (input_kind={obs.input_kind}, LLM-only)"
                )
                result.append(gate.model_copy(update={"status": "needs_human"}))
                continue

        # tool_correctness / no_unnecessary_calls on has_tools=False
        if gate.metric in ("tool_correctness", "no_unnecessary_calls"):
            static_no_tools = obs.has_tools is False and "has_tools" not in llm_fields
            llm_no_tools = obs.has_tools is False and "has_tools" in llm_fields
            # has_tools is always static (role:tool descendants — never LLM-inferred)
            if static_no_tools:
                report_notes.append(
                    f"{gate.id}: {gate.metric} replaced by llm_call_budget (has_tools=False, static)"
                )
                bud_id = f"{gate.component}.llm_call_budget.feasibility"
                if (gate.component, "llm_call_budget") not in seen_metrics:
                    seen_metrics.add((gate.component, "llm_call_budget"))
                    result.append(gate.model_copy(update={
                        "id": bud_id, "metric": "llm_call_budget",
                        "metric_class": "assertion", "toolkit": "assertion",
                        "rationale": f"Replaced {gate.metric}: no tool descendants",
                        "status": None, "params": {},
                    }))
                continue

        result.append(gate)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# CS-281 §5 — judge/assertion rebalance
# ──────────────────────────────────────────────────────────────────────────────

def _rebalance_gates(
    gates: list[EvaluationGate],
    agent_id: str,
    report_notes: list[str],
) -> list[EvaluationGate]:
    """1) Prefer assertions over llm_judge for trace-decidable properties.
    2) Merge near-duplicate geval rubrics (SequenceMatcher ≥ 0.85).
    3) Cap llm_suggested judges at MAX_LLM_JUDGE_GATES_PER_AGENT (baseline exempt)."""
    _import_all_assertions()

    # 1. Prefer assertions
    assertion_props: set[tuple[str, str]] = set()  # (component, property)
    for g in gates:
        if g.metric_class == "assertion":
            assertion_props.add((g.component, g.property))

    result: list[EvaluationGate] = []
    for g in gates:
        if (
            g.metric_class == "llm_judge"
            and g.provenance != "rule"
            and g.property
            and (g.component, g.property) in assertion_props
        ):
            report_notes.append(
                f"{g.id}: dropped (assertion already covers ({g.component}, {g.property!r}))"
            )
            continue
        result.append(g)
    gates = result

    # 2. Merge near-duplicate geval rubrics on same (agent_id, component)
    geval_gates: list[EvaluationGate] = [g for g in gates if g.metric.startswith("geval.") and g.provenance != "rule"]
    non_geval: list[EvaluationGate] = [g for g in gates if not (g.metric.startswith("geval.") and g.provenance != "rule")]
    merged_ids: set[str] = set()
    merged_geval: list[EvaluationGate] = []
    for i, gi in enumerate(geval_gates):
        if gi.id in merged_ids:
            continue
        ri = gi.params.get("rubric_text", "")
        for j, gj in enumerate(geval_gates):
            if j <= i or gj.id in merged_ids or gi.component != gj.component:
                continue
            rj = gj.params.get("rubric_text", "")
            ratio = difflib.SequenceMatcher(None, ri.lower(), rj.lower()).ratio()
            if ratio >= 0.85:
                merged_ids.add(gj.id)
                merged_rationale = gi.rationale + " | " + gj.rationale
                gi = gi.model_copy(update={"rationale": merged_rationale})
                report_notes.append(f"{gj.id}: merged into {gi.id} (rubric similarity {ratio:.2f})")
        merged_geval.append(gi)
    gates = non_geval + merged_geval

    # 3. Cap llm_suggested judges (baseline exempt)
    baseline_judges = [g for g in gates if g.metric_class == "llm_judge" and g.provenance == "rule"]
    suggested_judges = [g for g in gates if g.metric_class == "llm_judge" and g.provenance != "rule"]
    non_judges = [g for g in gates if g.metric_class != "llm_judge"]

    remaining_slots = max(0, MAX_LLM_JUDGE_GATES_PER_AGENT - len(baseline_judges))
    kept = suggested_judges[:remaining_slots]
    dropped = suggested_judges[remaining_slots:]
    for g in dropped:
        report_notes.append(
            f"{g.id}: removed (llm_judge cap {MAX_LLM_JUDGE_GATES_PER_AGENT}/agent reached; re-add via plan editor)"
        )

    return non_judges + baseline_judges + kept


def reconcile(
    agent_flow_map: AgentFlowMap,
    evidence_by_agent: dict[str, AgentEvidence],
    profiles_by_agent: dict[str, AgentDataProfile],
    baseline_by_agent: dict[str, list[SuiteEntry]],
    llm_gates_by_agent: dict[str, list[EvaluationGate]],
    handoff_gates: list[EvaluationGate],
    contracts: dict[str, EvaluationContract] | None = None,
    system_map: SystemMap | None = None,
) -> tuple[Suite, EvaluationPlanReport]:
    """Baseline always wins on (component, metric) conflict and is never dropped.
    CS-281: runs params completion, feasibility, and judge rebalance after merge."""
    handoff_by_agent: dict[str, list[EvaluationGate]] = {}
    for gate in handoff_gates:
        handoff_by_agent.setdefault(gate.agent_id, []).append(gate)

    all_suite_entries: list[SuiteEntry] = []
    agent_reports: list[AgentPlanReport] = []

    for agent in agent_flow_map.agents:
        baseline = baseline_by_agent.get(agent.id, [])
        seen: set[tuple[str, str]] = {(e.component, e.metric) for e in baseline}

        agent_gates: list[EvaluationGate] = []
        for gate in baseline:
            agent_gates.append(
                EvaluationGate(
                    id=gate.id,
                    agent_id=agent.id,
                    component=gate.component,
                    location=(
                        "handoff" if gate.metric in _BASELINE_HANDOFF_METRICS
                        else "output" if gate.metric_class == "llm_judge"
                        else "input"
                    ),
                    metric=gate.metric,
                    metric_class=gate.metric_class,
                    toolkit=(
                        "deepeval" if gate.metric.startswith("geval.") or gate.metric == "tool_correctness"
                        else "ragas" if gate.metric.startswith("ragas.")
                        else "classifier" if gate.metric_class == "classifier"
                        else "assertion"
                    ),
                    params=gate.params,
                    dataset=gate.dataset,  # preserve baseline dataset ref
                    rationale=gate.rationale,
                    provenance="rule",
                )
            )

        for gate in [*llm_gates_by_agent.get(agent.id, []), *handoff_by_agent.get(agent.id, [])]:
            key = (gate.component, gate.metric)
            if key in seen:
                continue
            seen.add(key)
            agent_gates.append(gate)

        contract = (contracts or {}).get(agent.id)
        post_notes: list[str] = []

        # CS-281 §4 — merge observability
        if contract is not None:
            conflict_notes = _merge_observability(contract, profiles_by_agent.get(agent.id))
            contract.needs_human.extend(conflict_notes)
            post_notes.extend(conflict_notes)

        # CS-281 §4 — params completion
        agent_gates = [
            _complete_params(g, system_map, contract, post_notes)
            for g in agent_gates
        ]

        # CS-281 §4 — feasibility pass
        agent_gates = _apply_feasibility(agent_gates, contract, agent.id, post_notes)

        # CS-281 §5 — rebalance
        agent_gates = _rebalance_gates(agent_gates, agent.id, post_notes)

        all_suite_entries.extend(_gate_to_suite_entry(g) for g in agent_gates)

        agent_reports.append(
            AgentPlanReport(
                agent_id=agent.id,
                role=agent.role,
                label=agent.label,
                data_profile=profiles_by_agent.get(agent.id),
                contract=contract,
                gates=agent_gates,
                needs_human=[g.id for g in agent_gates if g.status == "needs_human"] + post_notes,
            )
        )

    suite = Suite(entries=all_suite_entries)
    report = EvaluationPlanReport(
        target_system_id=agent_flow_map.target_system_id,
        agents=agent_reports,
    )
    return suite, report


# ──────────────────────────────────────────────────────────────────────────────
# critic — holistic self-review (single node; depends on reconcile)
# ──────────────────────────────────────────────────────────────────────────────


async def run_critic(report: EvaluationPlanReport, llm_client: LLMClient) -> list[str]:
    lines = []
    for agent_report in report.agents:
        gate_summary = [f"{g.metric}({g.metric_class})" for g in agent_report.gates]
        lines.append(f"Agent {agent_report.agent_id} (role={agent_report.role}): {gate_summary}")
    user_prompt = "\n".join(lines)

    try:
        response = await llm_client.complete(
            [
                LLMMessage(role="system", content=CRITIC_SYSTEM),
                LLMMessage(role="user", content=user_prompt),
            ],
            max_tokens=1024,
            json_mode=True,
        )
        parsed = json.loads(response.content)
        notes = parsed.get("notes", []) if isinstance(parsed, dict) else []
        return [n for n in notes if isinstance(n, str)] if isinstance(notes, list) else []
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning(f"agentic_planner: critic unparseable response: {e}")
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Top-level entry point — assembles and runs the DAG
# ──────────────────────────────────────────────────────────────────────────────


async def generate_plan_agentic(
    system_map: SystemMap,
    agent_flow_map: AgentFlowMap,
    source_by_component: dict[str, str],
    accepted_edges: list[dict],
    llm_client: LLMClient,
    *,
    run_critic_pass: bool = True,
    files: list[Path] | None = None,
    files_root: Path | None = None,
) -> tuple[Suite, EvaluationPlanReport]:
    """Builds and executes the Stage-3 DAG; every agent gets its own gate set."""
    agents = agent_flow_map.agents

    async def _gather(_: dict[str, Any]) -> tuple[dict[str, AgentEvidence], dict[str, list[SuiteEntry]]]:
        return await gather_evidence(system_map, agent_flow_map, source_by_component, accepted_edges, llm_client)

    # Contract harvest is pure AST over already-fetched files — no LLM, runs inline (CS-287).
    contracts: dict[str, EvaluationContract] = {}
    if files:
        from agent_eval_harness.mapping.builder.contract_harvest import harvest_contracts

        contracts = harvest_contracts(system_map, agent_flow_map, files, files_root)

    nodes: list[DagNode] = [DagNode("gather", [], _gather)]

    for agent in agents:
        analyst_name = f"analyst:{agent.id}"

        async def _analyst(results: dict[str, Any], agent_id: str = agent.id) -> AgentDataProfile:
            evidence_by_agent, _ = results["gather"]
            return await _run_analyst(agent_id, evidence_by_agent[agent_id], llm_client)

        nodes.append(DagNode(analyst_name, ["gather"], _analyst))

        gates_name = f"agent_gates:{agent.id}"

        async def _gates(results: dict[str, Any], agent_id: str = agent.id, analyst_name: str = analyst_name) -> list[EvaluationGate]:
            evidence_by_agent, baseline_by_agent = results["gather"]
            profile = results[analyst_name]
            return await _run_gate_designer(
                agent_id, evidence_by_agent[agent_id], profile, baseline_by_agent[agent_id], llm_client
            )

        nodes.append(DagNode(gates_name, ["gather", analyst_name], _gates))

    analyst_names = [f"analyst:{a.id}" for a in agents]

    async def _handoff(results: dict[str, Any]) -> list[EvaluationGate]:
        evidence_by_agent, _ = results["gather"]
        profiles_by_agent = {a.id: results[f"analyst:{a.id}"] for a in agents}
        return await _run_handoff_gates(evidence_by_agent, profiles_by_agent, llm_client)

    nodes.append(DagNode("handoff_gates", analyst_names, _handoff))

    gate_names = [f"agent_gates:{a.id}" for a in agents]

    async def _reconcile(results: dict[str, Any]) -> tuple[Suite, EvaluationPlanReport]:
        evidence_by_agent, baseline_by_agent = results["gather"]
        profiles_by_agent = {a.id: results[f"analyst:{a.id}"] for a in agents}
        llm_gates_by_agent = {a.id: results[f"agent_gates:{a.id}"] for a in agents}
        return reconcile(
            agent_flow_map, evidence_by_agent, profiles_by_agent, baseline_by_agent,
            llm_gates_by_agent, results["handoff_gates"], contracts=contracts, system_map=system_map,
        )

    nodes.append(DagNode("reconcile", [*gate_names, "handoff_gates"], _reconcile))

    if run_critic_pass:

        async def _critic(results: dict[str, Any]) -> list[str]:
            _, report = results["reconcile"]
            return await run_critic(report, llm_client)

        nodes.append(DagNode("critic", ["reconcile"], _critic))

    results = await run_dag(nodes)
    suite, report = results["reconcile"]
    if run_critic_pass:
        report = report.model_copy(update={"advisory_notes": results["critic"]})
    return suite, report
