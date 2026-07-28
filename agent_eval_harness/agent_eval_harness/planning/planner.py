"""planner.py — Evaluation Plan Generator."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Literal

from agent_eval_harness.config import DEFAULT_CASES_PER_AGENT
from agent_eval_harness.llm.client import LLMClient, LLMMessage
from agent_eval_harness.mapping.system_map import Component, SystemMap, load_system_map
from agent_eval_harness.metrics.suite import DatasetRef, Suite, SuiteEntry
from agent_eval_harness.store import repository

logger = logging.getLogger("agent_eval_harness.planning.planner")

DEFAULT_MIN_CASES = DEFAULT_CASES_PER_AGENT  # single global knob (config.py / AEH_CASES_PER_AGENT env)
GUARD_CLASSIFICATION_MIN_CASES = 40
_SAMPLE_QUERY_LIMIT = 2  # how many dataset-case queries to surface as example params
_SOURCE_SNIPPET_MAX_LINES = 50

# System prompts for LLM assistant
TAILOR_RUBRIC_SYSTEM = (
    "You are an AI evaluation planner. Given a component's docstring and source "
    "code snippet, write a clear, concise G-Eval evaluation rubric (rubric_text) "
    "for evaluating its output quality. The rubric should specifically target the "
    "component's goal (e.g. intent decomposition coverage, query rewriting "
    "quality, formatting, etc.) as described in the source. "
    'Return JSON only: {"rubric_text": "<detailed G-Eval rubric instructions>"}'
)

SUITE_SUGGESTION_SYSTEM = (
    "You are an AI evaluation planner. Given a component's entry point, role "
    "(which is unknown), docstring, and source code, suggest one or more "
    "evaluation suite entries for it.\n"
    "Available metrics:\n"
    "- assertions: allowed_downstream, arg_schema, max_items_per_call, "
    "max_retries, no_unnecessary_calls, retry_on_reject_required\n"
    "- llm_judges: ragas.faithfulness, ragas.answer_relevancy, "
    "ragas.context_precision, geval.<rubric_name>, tool_correctness\n"
    "- classifier: classifier.<component_id>_accuracy\n\n"
    'Return JSON only: {"entries": [{"metric": "<metric_name>", "metric_class": '
    '"assertion|llm_judge|classifier", "dataset_kind": "<optional_dataset_kind>", '
    '"rationale": "<explanation>", "params": {}}]}'
)


def get_component_info(entry_point: str, search_roots: list[Path] | None = None) -> dict[str, str]:
    """Static (AST) docstring/source extraction — never imports target code."""
    info = {"docstring": "", "source_snippet": ""}
    if not entry_point or entry_point.startswith(("http://", "https://")):
        return info

    module_path, _, class_name = entry_point.partition(":")
    class_name = class_name.split(".")[0]
    if not module_path or not class_name:
        return info

    # Default root = the agent_eval_harness project dir (covers the in-repo test targets).
    roots = search_roots or [Path(__file__).resolve().parents[2]]
    rel = Path(*module_path.split(".")).with_suffix(".py")
    for root in roots:
        candidate = root / rel
        if not candidate.is_file():
            continue
        try:
            import ast

            content = candidate.read_text(encoding="utf-8")
            tree = ast.parse(content)
        except (OSError, SyntaxError, UnicodeDecodeError) as e:
            logger.debug(f"Could not parse {candidate}: {e}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                info["docstring"] = ast.get_docstring(node) or ""
                from agent_eval_harness.discovery.expansion import extract_symbol_snippet

                snippet = extract_symbol_snippet(content, class_name)
                info["source_snippet"] = "\n".join(snippet.splitlines()[:_SOURCE_SNIPPET_MAX_LINES])
                return info
    return info


def get_tool_name_from_component(tc: Component) -> str:
    """Extract tool name from tool component's span_match tags, falling back to id."""
    for sm in tc.span_match:
        if sm.tags and "aeh.tool.name" in sm.tags:
            return sm.tags["aeh.tool.name"]
    return tc.id.removesuffix("_tool")


def derive_expected_tool_names(
    component: Component, components_by_id: dict[str, Component], system_map: SystemMap
) -> list[str]:
    """Downstream tool component names for tool_correctness / no_unnecessary_calls."""
    tool_comps: list[Component] = []
    for d in component.downstream:
        if d in components_by_id and components_by_id[d].role == "tool":
            tool_comps.append(components_by_id[d])
    for tc in system_map.components:
        if tc.role == "tool" and component.id in tc.upstream and tc not in tool_comps:
            tool_comps.append(tc)
    return [get_tool_name_from_component(tc) for tc in tool_comps]


def _resolve_dataset_ref(dataset_kind: str) -> DatasetRef | None:
    """Always starts fresh (never links to an existing DB dataset) so regenerating a plan can't silently resurface stale data — Fulfill Datasets re-runs the real generators."""
    if not dataset_kind:
        return None
    min_cases = GUARD_CLASSIFICATION_MIN_CASES if dataset_kind == "guard_classification" else DEFAULT_MIN_CASES
    return DatasetRef(required={"kind": dataset_kind, "min_cases": min_cases})


def _component_role_rules(
    component: Component,
    components_by_id: dict[str, Component],
    validator_comp: Component | None,
    system_map: SystemMap,
) -> list[dict]:
    """Pure, deterministic: which role-based rule dicts fire for this component."""
    role = component.role
    role_rules: list[dict] = []

    if role in ("input_guard.rule", "input_guard.llm"):
        role_rules.append({
            "metric": f"classifier.{component.id}_accuracy",
            "metric_class": "classifier",
            "dataset_kind": "guard_classification",
            "rationale": (
                f"role={role} ⇒ classification accuracy over labeled "
                "guard dataset. scored via sklearn."
            ),
        })
    elif role == "validator":
        role_rules.append({
            "metric": f"classifier.{component.id}_accuracy",
            "metric_class": "classifier",
            "dataset_kind": "sufficiency_labeled",
            "rationale": (
                f"role={role} ⇒ sufficiency classification accuracy. "
                "scored via sklearn."
            ),
        })
    elif role == "orchestrator":
        # Assigned unconditionally; agentic_planner._apply_feasibility drops/demotes it downstream when input_kind != "query".
        role_rules.append({
            "metric": "geval.decomposition_coverage",
            "metric_class": "llm_judge",
            "dataset_kind": "decomposition_gold",
            "rationale": (
                f"role={role} ⇒ decomposition coverage. evaluates "
                "if decomposed intents cover query."
            ),
        })
        role_rules.append({
            "metric": "allowed_downstream",
            "metric_class": "assertion",
            "rationale": (
                f"role={role} ⇒ orchestrator must only fan out to "
                "its declared downstream components."
            ),
        })
    elif role == "retrieval_agent":
        # Add no_unnecessary_calls if it invokes any tools downstream
        has_downstream_tools = any(
            components_by_id[d].role == "tool"
            for d in component.downstream
            if d in components_by_id
        ) or any(
            tc.role == "tool" and component.id in tc.upstream
            for tc in system_map.components
        )
        if has_downstream_tools:
            role_rules.append({
                "metric": "no_unnecessary_calls",
                "metric_class": "assertion",
                "rationale": (
                    f"role={role} ⇒ checks that every tool result "
                    "is used by some downstream component."
                ),
            })
            # Add tool_correctness (llm_judge)
            role_rules.append({
                "metric": "tool_correctness",
                "metric_class": "llm_judge",
                "rationale": (
                    f"role={role} ⇒ evaluates if the correct tools "
                    "were selected."
                ),
            })

        # Check if this retrieval agent is part of a retry loop
        is_in_retry_loop = False
        retry_rationale = None
        if validator_comp:
            # check if retrieval_agent goes to validator, and validator goes to planner
            has_downstream_validator = any(d == validator_comp.id for d in component.downstream)
            has_validator_to_orchestrator = any(
                components_by_id[d].role == "orchestrator"
                for d in validator_comp.downstream
                if d in components_by_id
            )
            if has_downstream_validator and has_validator_to_orchestrator:
                is_in_retry_loop = True
                retry_rationale = (
                    f"role={role} ⇒ verifies orchestrator dispatches "
                    "a retry when validator rejects."
                )

        if is_in_retry_loop:
            role_rules.append({
                "metric": "retry_on_reject_required",
                "metric_class": "assertion",
                "rationale": retry_rationale or (
                    f"role={role} ⇒ verifies orchestrator dispatches "
                    "a retry when validator rejects."
                ),
            })
    elif role == "writer":
        role_rules.append({
            "metric": "ragas.faithfulness",
            "metric_class": "llm_judge",
            "rationale": (
                f"role={role} ⇒ faithfulness is the central correctness "
                "property. writer must not fabricate facts."
            ),
        })
        # Same as the orchestrator branch above — feasibility pass corrects it downstream.
        role_rules.append({
            "metric": "ragas.answer_relevancy",
            "metric_class": "llm_judge",
            "rationale": (
                f"role={role} ⇒ answer should be relevant to the "
                "user's original query."
            ),
        })
    elif role == "worker":
        role_rules.append({
            "metric": "schema_valid",
            "metric_class": "assertion",
            "rationale": (
                f"role={role} ⇒ worker output must match contract schema."
            ),
        })
        role_rules.append({
            "metric": "fallback_sentinel",
            "metric_class": "assertion",
            "rationale": (
                f"role={role} ⇒ worker output should not be the contract's fallback literal."
            ),
        })
    elif role == "unknown":
        role_rules.append({
            "metric": "schema_valid",
            "metric_class": "assertion",
            "rationale": (
                f"role={role} ⇒ component output must match contract schema."
            ),
        })
        role_rules.append({
            "metric": "fallback_sentinel",
            "metric_class": "assertion",
            "rationale": (
                f"role={role} ⇒ component output should not be the contract's fallback literal."
            ),
        })

    # Trigger retry_on_reject_required for any component with loop motif
    if component.motif == "loop":
        role_rules.append({
            "metric": "retry_on_reject_required",
            "metric_class": "assertion",
            "rationale": "motif=loop ⇒ verifies loop component handles rejection/retry.",
        })

    return role_rules


async def _hydrate_rule_to_entry(
    rule: dict,
    component: Component,
    components_by_id: dict[str, Component],
    system_map: SystemMap,
    llm_client: LLMClient,
    *,
    agent_id: str | None = None,
    level: Literal["component", "trace", "session"] = "component",
    agent_knowledge_by_id: dict[str, dict] | None = None,
) -> SuiteEntry:
    """Resolve dataset refs and params, producing the SuiteEntry for one fired role rule."""
    dataset_ref = None
    if "dataset_kind" in rule:
        dataset_ref = _resolve_dataset_ref(rule["dataset_kind"])

    params: dict[str, Any] = {}
    # Fetch queries from dataset cases if we have resolved a reference
    if dataset_ref and dataset_ref.ref:
        try:
            cases = await repository.get_dataset_cases(dataset_ref.ref)
            if cases:
                sample_queries = []
                for case in cases[:_SAMPLE_QUERY_LIMIT]:
                    inp = json.loads(case.get("input_json") or "{}")
                    q = inp.get("query", inp.get("text"))
                    if q:
                        sample_queries.append(q)
                if sample_queries:
                    params["queries"] = sample_queries
        except Exception as e:
            logger.debug(f"Could not fetch sample queries for dataset {dataset_ref.ref}: {e}")

    if "queries" not in params:
        knowledge = agent_knowledge_by_id.get(component.id) if agent_knowledge_by_id else None
        synthesized = _synthesize_queries_from_knowledge(component, knowledge)
        params["queries"] = synthesized if synthesized else [
            "<TODO: add a representative query for this target>"
        ]

    if rule["metric"] == "allowed_downstream":
        params["allowed"] = component.downstream
    elif rule["metric"] == "tool_correctness":
        params["expected_tools"] = derive_expected_tool_names(component, components_by_id, system_map)

    if rule["metric"] == "geval.decomposition_coverage":
        info = get_component_info(component.entry_point)
        params["rubric_text"] = await _tailor_geval_rubric(
            component, info, llm_client
        )

    if rule["metric_class"] == "classifier":
        params["entry_point"] = component.entry_point

    metric_suffix = (
        rule["metric"]
        .replace("classifier.", "")
        .replace("geval.", "")
        .replace("ragas.", "")
    )
    if rule["metric_class"] == "classifier":
        metric_suffix = "classifier"

    return SuiteEntry(
        id=f"{component.id}.{metric_suffix}",
        component=component.id,
        metric=rule["metric"],
        metric_class=rule["metric_class"],
        level=level,
        dataset=dataset_ref,
        params=params,
        rationale=rule["rationale"],
        provenance="rule",
        agent_id=agent_id,
    )


def _constraint_entries(
    component: Component,
    components_by_id: dict[str, Component],
    *,
    agent_id: str | None = None,
    agent_knowledge_by_id: dict[str, dict] | None = None,
) -> list[SuiteEntry]:
    """Constraints → assertion SuiteEntry list. Pure/deterministic — no I/O."""
    entries: list[SuiteEntry] = []
    for constraint in component.constraints:
        try:
            from agent_eval_harness.metrics.assertions.registry import get_assertion
            get_assertion(constraint.name)
        except KeyError:
            logger.debug(f"{component.id}: constraint '{constraint.name}' has no registered assertion — skipped")
            continue

        # Handle max_items_per_call gotcha: observable in downstream retrieval_agent (worker)
        target_comp_id = component.id
        if constraint.name == "max_items_per_call":
            # find a downstream retrieval_agent
            downstream_retrievers = [
                d for d in component.downstream
                if d in components_by_id and components_by_id[d].role == "retrieval_agent"
            ]
            if downstream_retrievers:
                target_comp_id = downstream_retrievers[0]

        knowledge = agent_knowledge_by_id.get(component.id) if agent_knowledge_by_id else None
        synthesized = _synthesize_queries_from_knowledge(component, knowledge)
        params = {
            "limit": constraint.value,
            "source": f"system_map constraint ({constraint.source})",
            "queries": synthesized if synthesized else ["<TODO: add a representative query for this target>"],
        }

        entries.append(
            SuiteEntry(
                id=f"{component.id}.{constraint.name}",
                component=target_comp_id,
                metric=constraint.name,
                metric_class="assertion",
                params=params,
                rationale=f"constraint mined from code: {constraint.name}={constraint.value}",
                provenance="rule",
                agent_id=agent_id,
            )
        )

    # Emit up to 2 geval gates from AgentKnowledge.constraints
    if agent_id and agent_knowledge_by_id:
        agent_knowledge = agent_knowledge_by_id.get(agent_id)
        if agent_knowledge and isinstance(agent_knowledge.get("constraints"), list):
            constraints = agent_knowledge["constraints"]
            for i, constraint_text in enumerate(constraints[:2]):  # Cap at 2
                if not isinstance(constraint_text, str) or not constraint_text.strip():
                    continue
                # Create deterministic rubric name from constraint text (slug format)
                slug = (
                    constraint_text.lower()
                    .strip()[:50]  # Truncate to first 50 chars
                    .replace(" ", "_")
                    .replace("-", "_")
                )
                slug = re.sub(r"[^a-z0-9_]", "", slug)
                if not slug:
                    continue
                entries.append(
                    SuiteEntry(
                        id=f"{component.id}.constraint_rubric_{i}",
                        component=component.id,
                        metric=f"geval.{slug}",
                        metric_class="llm_judge",
                        params={"rubric_text": constraint_text},
                        rationale=f"hard rule from prompt: {constraint_text[:60]}",
                        provenance="rule",
                        agent_id=agent_id,
                    )
                )

    return entries


def _find_validator(system_map: SystemMap) -> Component | None:
    for c in system_map.components:
        if c.role == "validator":
            return c
    return None


def _richness_from_agent_knowledge(knowledge: dict | None) -> int:
    """Extract richness count from AgentKnowledge dict (sidecar format)."""
    if not knowledge:
        return 0
    data = knowledge if isinstance(knowledge, dict) else {}
    return (
        len(data.get("functionality_citations") or []) +
        len(data.get("context_builders") or []) +
        len(data.get("upstream_consumers") or []) +
        len(data.get("downstream_consumers") or []) +
        len(data.get("failure_modes") or []) +
        len(data.get("constraints") or []) +
        len(data.get("method_steps") or [])
    )


def _synthesize_queries_from_knowledge(
    component: Component,
    knowledge: dict | None,
) -> list[str]:
    """Synthesize 2-3 deterministic query examples from AgentKnowledge."""
    if not knowledge:
        return []

    queries: list[str] = []

    # Try to synthesize from functionality
    functionality = knowledge.get("functionality")
    if isinstance(functionality, str) and functionality.strip():
        queries.append(f"Test {component.id} with: {functionality[:60].strip()}")

    # Try to synthesize from failure modes
    failure_modes = knowledge.get("failure_modes") or []
    if failure_modes and isinstance(failure_modes, list) and len(failure_modes) > 0:
        first_mode = failure_modes[0]
        if isinstance(first_mode, str) and first_mode.strip():
            queries.append(f"Verify {component.id} handles: {first_mode[:60].strip()}")

    # Try to synthesize from input examples
    input_examples = knowledge.get("input_examples") or []
    if input_examples and isinstance(input_examples, list) and len(input_examples) > 0:
        first_example = input_examples[0]
        if isinstance(first_example, str) and first_example.strip():
            queries.append(f"Input: {first_example[:60].strip()}")

    return queries[:3]  # Return up to 3 queries


def _is_thin_component(
    component: Component,
    components_by_id: dict[str, Component],
    agent_knowledge_by_id: dict[str, dict] | None = None,
) -> bool:
    """True if component is a near-deterministic transform (thin) vs independent judgment (rich).

    Thin = parse/format/validate with no independent decisions.
    Rich = makes independent judgments, queries, or complex logic.
    """
    role = component.role

    if role in ("tool", "unknown"):
        return False

    if role in ("retrieval_agent", "orchestrator", "validator"):
        return False

    if role == "writer":
        return False

    if role == "input_guard.llm":
        return False

    if role == "worker":
        richness = 0
        if agent_knowledge_by_id and component.id in agent_knowledge_by_id:
            richness = _richness_from_agent_knowledge(agent_knowledge_by_id[component.id])

        has_downstream_tools = any(
            d in components_by_id and components_by_id[d].role == "tool"
            for d in component.downstream
        )

        if richness > 2 or has_downstream_tools:
            return False

        return True

    return False


def _is_system_thin_chain(
    system_map: SystemMap,
    agent_knowledge_by_id: dict[str, dict] | None = None,
) -> bool:
    """True if system has only one substantive decision node, rest plumbing."""
    components_by_id = {c.id: c for c in system_map.components}

    substantial_count = 0
    for component in system_map.components:
        if not _is_thin_component(component, components_by_id, agent_knowledge_by_id):
            substantial_count += 1

    return substantial_count == 1


def _select_granularity_level(
    component: Component,
    system_map: SystemMap,
    components_by_id: dict[str, Component],
    agent_knowledge_by_id: dict[str, dict] | None = None,
    agent_id: str | None = None,
) -> tuple[Literal["component", "trace", "session"], bool]:
    """Select granularity level for a component's entries.

    Returns (level, needs_trajectory) where:
    - level="component": per-node entry (default)
    - level="session": end-to-end entry (thin-chain collapse)
    - needs_trajectory: True if this component should also get a trajectory entry (dual-channel: motif=loop OR control_motif)
    """
    # Sidecars are keyed by AGENT id; for a single-component agent that equals the component id,
    # but a grouped agent needs the agent-id lookup too or its control_motif is missed.
    knowledge = None
    if agent_knowledge_by_id:
        knowledge = agent_knowledge_by_id.get(component.id) or (
            agent_knowledge_by_id.get(agent_id) if agent_id else None
        )
    has_control_loop = bool(
        component.motif == "loop"
        or (knowledge or {}).get("control_motif") is not None
    )

    is_thin_chain = _is_system_thin_chain(system_map, agent_knowledge_by_id)

    if is_thin_chain:
        return ("session", has_control_loop)

    return ("component", has_control_loop)


def role_skip_note(component: Component) -> str | None:
    """Legibility note: tool/unknown roles have no role-derived rule, only mined constraints."""
    if component.role in ("tool", "unknown"):
        return f"{component.id}: role={component.role} has no role-derived gate (constraints, if any, still apply)"
    return None


async def baseline_gates_for_component(
    component: Component,
    components_by_id: dict[str, Component],
    validator_comp: Component | None,
    system_map: SystemMap,
    llm_client: LLMClient,
    *,
    agent_id: str | None = None,
    agent_knowledge_by_id: dict[str, dict] | None = None,
) -> list[SuiteEntry]:
    """Deterministic role+constraint baseline for ONE component; role rules no-op for tool/unknown/worker, constraints always apply."""
    entries: list[SuiteEntry] = []

    level, needs_trajectory = _select_granularity_level(
        component, system_map, components_by_id, agent_knowledge_by_id, agent_id=agent_id
    )

    for rule in _component_role_rules(component, components_by_id, validator_comp, system_map):
        entries.append(
            await _hydrate_rule_to_entry(
                rule, component, components_by_id, system_map, llm_client,
                agent_id=agent_id,
                level=level,
                agent_knowledge_by_id=agent_knowledge_by_id,
            )
        )

    entries.extend(_constraint_entries(component, components_by_id, agent_id=agent_id, agent_knowledge_by_id=agent_knowledge_by_id))

    if needs_trajectory:
        trajectory_entry = SuiteEntry(
            id=f"{component.id}.trajectory",
            component=component.id,
            metric="trajectory_termination",
            metric_class="assertion",
            level="trace",
            rationale=f"Component {component.id} has control-loop/retry behavior (motif={'loop' if component.motif == 'loop' else 'control'}); verify termination within bounds and feedback incorporation.",
            provenance="rule",
            agent_id=agent_id,
        )
        entries.append(trajectory_entry)

    return entries


async def baseline_gates_for_agent(
    agent_component_ids: list[str],
    system_map: SystemMap,
    llm_client: LLMClient,
    *,
    agent_id: str,
    agent_knowledge_by_id: dict[str, dict] | None = None,
) -> list[SuiteEntry]:
    """Deterministic-rule baseline for every component one AgentFlowMap agent owns.

    On a thin chain, collapses non-anchor components to session level, skipping
    per-component role/constraint entries and keeping only trajectory entries.
    """
    components_by_id = {c.id: c for c in system_map.components}
    validator_comp = _find_validator(system_map)
    entries: list[SuiteEntry] = []

    is_thin_chain = _is_system_thin_chain(system_map, agent_knowledge_by_id)
    thin_chain_component: Component | None = None

    if is_thin_chain:
        for component in system_map.components:
            if not _is_thin_component(component, components_by_id, agent_knowledge_by_id):
                thin_chain_component = component
                break

    for cid in agent_component_ids:
        component = components_by_id.get(cid)
        if component is None:
            continue

        skip_component_entries = is_thin_chain and component.id != (thin_chain_component.id if thin_chain_component else "")

        if not skip_component_entries:
            entries.extend(
                await baseline_gates_for_component(
                    component, components_by_id, validator_comp, system_map,
                    llm_client, agent_id=agent_id,
                    agent_knowledge_by_id=agent_knowledge_by_id,
                )
            )
        elif is_thin_chain and (
            component.motif == "loop" or
            (agent_knowledge_by_id and component.id in agent_knowledge_by_id and
             agent_knowledge_by_id[component.id].get("control_motif") is not None)
        ):
            trajectory_entry = SuiteEntry(
                id=f"{component.id}.trajectory",
                component=component.id,
                metric="trajectory_termination",
                metric_class="assertion",
                level="trace",
                rationale=f"Component {component.id} has control-loop/retry behavior (motif={'loop' if component.motif == 'loop' else 'control'}); verify termination within bounds and feedback incorporation.",
                provenance="rule",
                agent_id=agent_id,
            )
            entries.append(trajectory_entry)

    return entries


async def generate_plan(
    system_map_path: str | Path,
    llm_client: LLMClient,
    agent_knowledge_by_id: dict[str, dict] | None = None,
) -> Suite:
    """Generate an evaluation plan from a system map (flat, role+constraint rules only).

    Applies CS-322 granularity selector: per-node (component), trajectory (trace), or
    end-to-end (session) based on motif + richness.
    """
    system_map = load_system_map(system_map_path)
    entries: list[SuiteEntry] = []

    components_by_id = {c.id: c for c in system_map.components}
    validator_comp = _find_validator(system_map)

    is_thin_chain = _is_system_thin_chain(system_map, agent_knowledge_by_id)
    thin_chain_component: Component | None = None

    if is_thin_chain:
        for component in system_map.components:
            if not _is_thin_component(component, components_by_id, agent_knowledge_by_id):
                thin_chain_component = component
                break

    for component in system_map.components:
        role = component.role

        if role == "unknown":
            info = get_component_info(component.entry_point)
            suggested_entries = await _suggest_unknown_component_suite(
                component, info, llm_client
            )
            entries.extend(suggested_entries)
            continue

        if role == "tool":
            continue

        skip_component_entries = is_thin_chain and component.id != (thin_chain_component.id if thin_chain_component else "")

        if not skip_component_entries:
            entries.extend(
                await baseline_gates_for_component(
                    component, components_by_id, validator_comp, system_map,
                    llm_client,
                    agent_knowledge_by_id=agent_knowledge_by_id,
                )
            )
        elif is_thin_chain and (
            component.motif == "loop" or
            (agent_knowledge_by_id and component.id in agent_knowledge_by_id and
             agent_knowledge_by_id[component.id].get("control_motif") is not None)
        ):
            trajectory_entry = SuiteEntry(
                id=f"{component.id}.trajectory",
                component=component.id,
                metric="trajectory_termination",
                metric_class="assertion",
                level="trace",
                rationale=f"Component {component.id} has control-loop/retry behavior (motif={'loop' if component.motif == 'loop' else 'control'}); verify termination within bounds and feedback incorporation.",
                provenance="rule",
            )
            entries.append(trajectory_entry)

    return Suite(entries=entries)


async def _tailor_geval_rubric(
    component: Component,
    info: dict[str, str],
    llm_client: LLMClient,
) -> str:
    """Tailor the G-Eval rubric text to the component's actual prompt/purpose."""
    doc = info.get("docstring") or ""
    src = info.get("source_snippet") or ""

    user_prompt = f"Component ID: {component.id}\nDocstring: {doc}\nSource:\n{src}"
    response = await llm_client.complete(
        [
            LLMMessage(role="system", content=TAILOR_RUBRIC_SYSTEM),
            LLMMessage(role="user", content=user_prompt),
        ],
        json_mode=True,
        # reasoning_effort left to the client's config default — never hardcoded.
    )

    fallback_rubric = (
        "Evaluate whether the planner's decomposed intents collectively cover all "
        "aspects of the user's query. A score of 1.0 means all query aspects are "
        "addressed; 0.0 means none are."
    )

    if response.content == "This is a fallback offline demo answer.":
        return fallback_rubric

    try:
        parsed = json.loads(response.content)
        return parsed.get("rubric_text", fallback_rubric)
    except Exception as e:
        logger.warning(f"Could not parse tailored rubric for {component.id}, using fallback: {e}")
        return fallback_rubric


async def _suggest_unknown_component_suite(
    component: Component,
    info: dict[str, str],
    llm_client: LLMClient,
) -> list[SuiteEntry]:
    """Ask LLM to suggest one or more suite entries for an unknown-role component."""
    doc = info.get("docstring") or ""
    src = info.get("source_snippet") or ""

    user_prompt = f"Component ID: {component.id}\nDocstring: {doc}\nSource:\n{src}"
    response = await llm_client.complete(
        [
            LLMMessage(role="system", content=SUITE_SUGGESTION_SYSTEM),
            LLMMessage(role="user", content=user_prompt),
        ],
        json_mode=True,
        reasoning_effort="disable", 
    )

    fallback_entries = [
        SuiteEntry(
            id=f"{component.id}.unknown",
            component=component.id,
            metric="unknown",
            metric_class="assertion",
            rationale="Component role is unknown. Human must define metrics.",
            provenance="llm_suggested",
            status="needs_human",
        )
    ]

    if response.content == "This is a fallback offline demo answer.":
        return fallback_entries

    try:
        parsed = json.loads(response.content)
        suggested = parsed.get("entries", [])
        if not suggested:
            return fallback_entries

        results = []
        for i, item in enumerate(suggested):
            dataset_ref = None
            if "dataset_kind" in item:
                dataset_ref = _resolve_dataset_ref(item["dataset_kind"])

            metric_suffix = (
                item["metric"]
                .replace("classifier.", "")
                .replace("geval.", "")
                .replace("ragas.", "")
            )
            results.append(
                SuiteEntry(
                    id=f"{component.id}.{metric_suffix}",
                    component=component.id,
                    metric=item["metric"],
                    metric_class=item["metric_class"],
                    dataset=dataset_ref,
                    params=item.get("params", {}),
                    rationale=item.get("rationale", "LLM suggested check."),
                    provenance="llm_suggested",
                    status="needs_human",
                )
            )
        return results
    except Exception as e:
        logger.warning(f"Could not parse suggested suite entries for {component.id}, using fallback: {e}")
        return fallback_entries
