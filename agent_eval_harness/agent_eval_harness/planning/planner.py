"""planner.py — Evaluation Plan Generator."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from agent_eval_harness.llm.client import LLMClient, LLMMessage
from agent_eval_harness.mapping.system_map import Component, SystemMap, load_system_map
from agent_eval_harness.metrics.suite import DatasetRef, Suite, SuiteEntry
from agent_eval_harness.store import repository

logger = logging.getLogger("agent_eval_harness.planning.planner")

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
                info["source_snippet"] = "\n".join(snippet.splitlines()[:50])
                return info
    return info


def get_tool_name_from_component(tc: Component) -> str:
    """Extract tool name from tool component's span_match tags, falling back to id."""
    for sm in tc.span_match:
        if sm.tags and "aeh.tool.name" in sm.tags:
            return sm.tags["aeh.tool.name"]
    # Fallback to class name or stripped ID
    base = tc.id
    if base.endswith("_tool"):
        base = base[:-5]
    return base


def _resolve_dataset_ref(dataset_kind: str) -> DatasetRef | None:
    """Every dataset requirement always starts fresh (never links to an already-existing
    dataset in the DB) — so regenerating a plan can't silently resurface a stale dataset;
    Fulfill Datasets always re-runs the real generators and produces a brand-new one."""
    if not dataset_kind:
        return None
    min_cases = 40 if dataset_kind == "guard_classification" else 20
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
        # Assigned unconditionally; agentic_planner.py::_apply_feasibility drops/demotes
        # this downstream when input_kind != "query" (CS-288) — don't fix it here.
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

        if is_in_retry_loop:
            role_rules.append({
                "metric": "retry_on_reject_required",
                "metric_class": "assertion",
                "rationale": (
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
        # Same as the orchestrator branch above — feasibility pass corrects it (CS-288).
        role_rules.append({
            "metric": "ragas.answer_relevancy",
            "metric_class": "llm_judge",
            "rationale": (
                f"role={role} ⇒ answer should be relevant to the "
                "user's original query."
            ),
        })
    elif role == "worker":
        pass  # ordinary transform node — no role-derived gate

    return role_rules


async def _hydrate_rule_to_entry(
    rule: dict,
    component: Component,
    components_by_id: dict[str, Component],
    system_map: SystemMap,
    llm_client: LLMClient,
    *,
    agent_id: str | None = None,
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
                for case in cases[:2]:
                    inp = json.loads(case.get("input_json") or "{}")
                    q = inp.get("query", inp.get("text"))
                    if q:
                        sample_queries.append(q)
                if sample_queries:
                    params["queries"] = sample_queries
        except Exception:
            pass

    if "queries" not in params:
        params["queries"] = [
            "<TODO: add a representative query for this target>"
        ]

    if rule["metric"] == "allowed_downstream":
        params["allowed"] = component.downstream
    elif rule["metric"] == "tool_correctness":
        # Find all downstream tools or tools having this component
        expected_tool_comps = []
        for d in component.downstream:
            if d in components_by_id and components_by_id[d].role == "tool":
                expected_tool_comps.append(components_by_id[d])
        for tc in system_map.components:
            if tc.role == "tool" and component.id in tc.upstream:
                if tc not in expected_tool_comps:
                    expected_tool_comps.append(tc)
        params["expected_tools"] = [
            get_tool_name_from_component(tc) for tc in expected_tool_comps
        ]

    if rule["metric"] == "geval.decomposition_coverage":
        # Tailor rubric wording via LLM
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
) -> list[SuiteEntry]:
    """Constraints → assertion SuiteEntry list. Pure/deterministic — no I/O."""
    entries: list[SuiteEntry] = []
    for constraint in component.constraints:
        # Check if this constraint name is a registered assertion
        try:
            from agent_eval_harness.metrics.assertions.registry import get_assertion
            get_assertion(constraint.name)
        except KeyError:
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

        params = {
            "limit": constraint.value,
            "source": f"system_map constraint ({constraint.source})",
            "queries": ["<TODO: add a representative query for this target>"],
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
    return entries


def _find_validator(system_map: SystemMap) -> Component | None:
    for c in system_map.components:
        if c.role == "validator":
            return c
    return None


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
) -> list[SuiteEntry]:
    """Deterministic role+constraint baseline for ONE component; role rules no-op for tool/unknown/worker, constraints always apply."""
    entries: list[SuiteEntry] = []
    for rule in _component_role_rules(component, components_by_id, validator_comp, system_map):
        entries.append(
            await _hydrate_rule_to_entry(
                rule, component, components_by_id, system_map, llm_client,
                agent_id=agent_id,
            )
        )
    entries.extend(_constraint_entries(component, components_by_id, agent_id=agent_id))
    return entries


async def baseline_gates_for_agent(
    agent_component_ids: list[str],
    system_map: SystemMap,
    llm_client: LLMClient,
    *,
    agent_id: str,
) -> list[SuiteEntry]:
    """Deterministic-rule baseline for every component one AgentFlowMap agent owns."""
    components_by_id = {c.id: c for c in system_map.components}
    validator_comp = _find_validator(system_map)
    entries: list[SuiteEntry] = []
    for cid in agent_component_ids:
        component = components_by_id.get(cid)
        if component is None:
            continue
        entries.extend(
            await baseline_gates_for_component(
                component, components_by_id, validator_comp, system_map,
                llm_client, agent_id=agent_id,
            )
        )
    return entries


async def generate_plan(
    system_map_path: str | Path,
    llm_client: LLMClient,
) -> Suite:
    """Generate an evaluation plan from a system map (flat, role+constraint rules only)."""
    system_map = load_system_map(system_map_path)
    entries: list[SuiteEntry] = []

    # Map for quick component lookup
    components_by_id = {c.id: c for c in system_map.components}
    validator_comp = _find_validator(system_map)

    for component in system_map.components:
        role = component.role

        # Handle unknown components via LLM Suggestion
        if role == "unknown":
            info = get_component_info(component.entry_point)
            suggested_entries = await _suggest_unknown_component_suite(
                component, info, llm_client
            )
            entries.extend(suggested_entries)
            continue

        # Handle tools (tools have no explicit suite entries by default)
        if role == "tool":
            continue

        entries.extend(
            await baseline_gates_for_component(
                component, components_by_id, validator_comp, system_map,
                llm_client,
            )
        )

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
        reasoning_effort="low",
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
    except Exception:
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
        reasoning_effort="low",
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
    except Exception:
        return fallback_entries
