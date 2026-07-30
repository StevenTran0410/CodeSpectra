"""Auto-fulfillment walk: turns a plan's unfulfilled `dataset.required` blocks into real datasets, one per (kind, agent-or-component) group, or an explicit failed/needs_human reason."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from agent_eval_harness.config import DEFAULT_CASES_PER_AGENT
from agent_eval_harness.datasets.generator_utils import (
    config_kwarg_names_from_case_binding,
    seed_cases_to_dataset_cases,
)
from agent_eval_harness.datasets.registry import get_generator
from agent_eval_harness.datasets.types import TRUSTED_PROVENANCE
from agent_eval_harness.datasets.versioning import next_version
from agent_eval_harness.discovery.client import CodeSpectraClient
from agent_eval_harness.llm.client import LLMClient
from agent_eval_harness.llm.embedding_client import EmbeddingClient
from agent_eval_harness.metrics.suite import DatasetRef, Suite, SuiteEntry, load_suite
from agent_eval_harness.planning.contract import EvaluationContract
from agent_eval_harness.planning.report import load_plan_report
from agent_eval_harness.store import repository

logger = logging.getLogger("agent_eval_harness.datasets.fulfillment")

# Some kinds are 1:1 with an execution, not a text-sample pool — min_cases scales differently.
_MIN_CASES_OVERRIDE: dict[str, int] = {
    "snapshot_fixture": 1,
    "field_match_gold": 1,
}
_DEFAULT_MIN_CASES: dict[str, int] = {"guard_classification": 40}

# Producers must run before consumers (sufficiency_labeled needs a same-run qa_testset id).
_TOPO_ORDER = [
    "qa_testset", "decomposition_gold", "guard_classification",
    "snapshot_fixture", "field_match_gold",
    "sufficiency_labeled",
    "recorded_report_replay",
    "synthetic_agent_io",  # no dependents downstream; order among the rest doesn't matter
]


def _group_key(kind: str, entry: SuiteEntry) -> str:
    return f"{kind}/{entry.agent_id or entry.component}"


def _collect_unfulfilled_groups(suite: Suite) -> dict[str, list[SuiteEntry]]:
    groups: dict[str, list[SuiteEntry]] = {}
    for entry in suite.entries:
        if not entry.dataset or not entry.dataset.required or entry.dataset.ref:
            continue
        kind = entry.dataset.required.get("kind")
        if not kind:
            continue
        groups.setdefault(_group_key(kind, entry), []).append(entry)
    return groups


def _effective_min_cases(kind: str, group_entries: list[SuiteEntry]) -> int:
    if kind in _MIN_CASES_OVERRIDE:
        return _MIN_CASES_OVERRIDE[kind]
    declared = [
        e.dataset.required.get("min_cases")
        for e in group_entries
        if e.dataset and e.dataset.required and isinstance(e.dataset.required.get("min_cases"), int)
    ]
    return max(declared) if declared else _DEFAULT_MIN_CASES.get(kind, DEFAULT_CASES_PER_AGENT)


def _qa_corpus_paths(local_path: str) -> list[str]:
    """**/*.md + **/*.txt under the snapshot, excluding .aeh/** — never retrieve the answer key itself."""
    root = Path(local_path)
    paths: list[str] = []
    for pattern in ("**/*.md", "**/*.txt"):
        for p in root.glob(pattern):
            rel = p.relative_to(root).as_posix()
            if rel.startswith(".aeh/"):
                continue
            paths.append(str(p))
    return paths


def _derive_guard_categories(group_entries: list[SuiteEntry]) -> list[dict] | None:
    """Categories must already be present on the gate's own params — never guessed."""
    for entry in group_entries:
        cats = entry.params.get("categories")
        if isinstance(cats, list) and cats:
            return cats
    return None


def _qa_testset_backend(group_entries: list[SuiteEntry]) -> str:
    """Match the qa_testset backend to the consuming gate's scoring toolkit (metric namespace prefix); ragas wins ties since its metrics are pickier about shape."""
    if any(e.metric.startswith("ragas.") for e in group_entries):
        return "ragas"
    return "deepeval"


def _validate_case_input_against_contract(case_input: Any, contract: EvaluationContract | None) -> list[str]:
    """Kind-agnostic missing-required check against the harvested contract. Foreign-field detection needs the per-kind case envelope, so it lives in each generator (e.g. synthetic_agent_io._validate_input)."""
    if contract is None or not isinstance(case_input, dict):
        return []
    invocation = contract.invocation
    if not invocation:
        return []
    config_names = config_kwarg_names_from_case_binding(invocation.case_binding)

    errors: list[str] = []
    for kw in (invocation.kwargs or []):
        if kw.name and kw.name not in config_names and kw.required:
            if not kw.default_repr and kw.name not in case_input:
                errors.append(f"missing required input field '{kw.name}'")

    return errors



async def _derive_config(
    kind: str,
    dataset_id: str,
    group_key: str,
    group_entries: list[SuiteEntry],
    *,
    map_path: str,
    snapshot_id: str,
    snapshot_local_path: str,
    provider_id: str,
    model_id: str,
    dataset_id_by_group: dict[str, str],
    painpoint: str | None,
    min_cases: int,
    codespectra_client: CodeSpectraClient | None = None,
    group_instructions: dict[str, Any] | None = None,
    eval_enabled_by_agent: dict[str, bool] | None = None,
    contract_by_agent: dict[str, EvaluationContract] | None = None,
    profile_by_agent: dict[str, dict[str, Any]] | None = None,
    knowledge_by_agent: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Returns a generator config dict, or None if underivable (caller marks needs_human)."""
    component = group_entries[0].component
    instructions = group_instructions or {}
    extra_snapshot_ids = instructions.get("extra_snapshot_ids") or []

    if kind == "synthetic_agent_io":
        agent_id = group_entries[0].agent_id or component
        if not (eval_enabled_by_agent or {}).get(agent_id, False):
            return None
        contract = (contract_by_agent or {}).get(agent_id)
        if contract is None:
            return None
        # 3-tier purpose fallback: blend profile purpose and knowledge functionality without blindly overwriting
        knowledge = (knowledge_by_agent or {}).get(agent_id) or {}
        base_profile = dict((profile_by_agent or {}).get(agent_id) or {})
        prof_purpose = base_profile.get("purpose") or ""
        know_func = knowledge.get("functionality") or ""
        if prof_purpose and know_func and know_func not in prof_purpose:
            base_profile["purpose"] = f"{prof_purpose} Functionality: {know_func}"
        elif know_func and not prof_purpose:
            base_profile["purpose"] = know_func

        section_output_schemas: dict[str, dict[str, Any]] = {}
        if contract.field_downstream_consumers and contract_by_agent:
            for _aid, up_contract in contract_by_agent.items():
                sec_letter = up_contract.output.section_letter if up_contract.output else None
                if sec_letter and up_contract.output.json_schema:
                    section_output_schemas[sec_letter] = up_contract.output.json_schema

        return {
            "dataset_name": dataset_id, "agent_id": agent_id,
            "archetype": contract.archetype,
            "contract": contract.model_dump(),
            "profile": base_profile,
            "failure_modes": knowledge.get("failure_modes") or [],
            "input_contract": knowledge.get("input_contract") or [],
            "input_schemas": knowledge.get("input_schemas") or {},
            "virtual_inputs": knowledge.get("virtual_inputs") or [],
            "context_builders": knowledge.get("context_builders") or [],
            "section_output_schemas": section_output_schemas,
            "count": min_cases, "painpoint": painpoint,
        }

    if kind == "qa_testset":
        corpus_paths = _qa_corpus_paths(snapshot_local_path)
        if not corpus_paths:
            return None
        return {
            "dataset_name": dataset_id, "corpus_paths": corpus_paths, "count": min_cases,
            "backend": _qa_testset_backend(group_entries),
        }

    if kind == "decomposition_gold":
        return {
            "dataset_name": dataset_id, "system_map_path": map_path,
            "component_id": component, "count": min_cases, "painpoint": painpoint,
        }

    if kind == "guard_classification":
        categories = _derive_guard_categories(group_entries)
        if categories is None:
            return None
        return {"dataset_name": dataset_id, "categories": categories}

    if kind == "sufficiency_labeled":
        suffix = group_key.split("/", 1)[1]
        source_id = dataset_id_by_group.get(f"qa_testset/{suffix}")
        if not source_id:
            return None
        return {"dataset_name": dataset_id, "source_dataset_id": source_id}

    if kind == "snapshot_fixture":
        if extra_snapshot_ids and not codespectra_client:
            return None
        snapshot_ids = [snapshot_id, *extra_snapshot_ids]
        if extra_snapshot_ids:
            try:
                for extra_id in extra_snapshot_ids:
                    await codespectra_client.get_snapshot(extra_id)
            except Exception as e:
                logger.warning(f"Failed to validate extra snapshots for snapshot_fixture: {e}")
                return None
        return {
            "dataset_name": dataset_id, "snapshot_ids": snapshot_ids,
            "provider_id": provider_id, "model_id": model_id,
        }

    if kind == "field_match_gold":
        agent_id = group_entries[0].agent_id or component
        contract = (contract_by_agent or {}).get(agent_id)
        if contract and contract.output and contract.output.json_schema:
            props = contract.output.json_schema.get("properties") or {}
            # field_match_gold emits repo_name and tech_stack fields; gate targeting requires output schema match
            if not ("repo_name" in props or "tech_stack" in props):
                return None
        if extra_snapshot_ids and not codespectra_client:
            return None
        snapshot_ids = [snapshot_id, *extra_snapshot_ids]
        local_paths = {snapshot_id: snapshot_local_path}
        if extra_snapshot_ids:
            try:
                for extra_id in extra_snapshot_ids:
                    snap = await codespectra_client.get_snapshot(extra_id)
                    lp = snap.get("local_path")
                    if not lp:
                        return None
                    local_paths[extra_id] = lp
            except Exception as e:
                logger.warning(f"Failed to resolve extra snapshot paths for field_match_gold: {e}")
                return None
        return {
            "dataset_name": dataset_id,
            "snapshot_ids": snapshot_ids,
            "local_paths": local_paths,
        }

    if kind == "recorded_report_replay":
        if not codespectra_client:
            return None
        try:
            snapshot = await codespectra_client.get_snapshot(snapshot_id)
            repo_id = snapshot.get("local_repo_id")
            if not repo_id:
                return None
            reports = await codespectra_client.list_reports(repo_id=repo_id, limit=min_cases)
            reports_config = []
            agent_id = group_entries[0].agent_id or group_entries[0].component
            # Sections come from the fan-in agent's own harvested field_downstream_consumers, never a fixed literal; no contract on record -> use every section the report has.
            contract = (contract_by_agent or {}).get(agent_id)
            harvested_letters = sorted((contract.field_downstream_consumers or {}).keys()) if contract else []

            for r in reports:
                r_id = r["id"]
                report_data = await codespectra_client.get_report(r_id)
                full_report_dict = report_data.get("report") or {}
                sections = full_report_dict.get("sections") or {}
                letters = harvested_letters or sorted(sections.keys())
                sliced_sections = {let: sections[let] for let in letters if let in sections}
                reports_config.append({
                    "report_id": r_id,
                    "snapshot_id": r.get("snapshot_id") or snapshot_id,
                    "sections": sliced_sections,
                })
            return {
                "dataset_name": dataset_id,
                "reports": reports_config,
            }
        except Exception as e:
            logger.warning(f"Failed to derive config for recorded_report_replay: {e}")
            return None

    return None


def _save_suite(suite: Suite, plan_path: str | Path) -> None:
    Path(plan_path).write_text(yaml.dump(suite.model_dump(), allow_unicode=True), encoding="utf-8")


async def _prune_prior_versions(base_name: str, keep: str) -> int:
    """Delete older versions of the same (kind, agent) dataset so re-fulfilling doesn't pile up v1/v2/v3 — a regen keeps exactly one dataset per agent (the just-written `keep`)."""
    pattern = re.compile(rf"^{re.escape(base_name)}_v\d+$")
    deleted = 0
    for d in await repository.list_dataset_ids():
        did = d["dataset_id"]
        if did != keep and pattern.match(did):
            await repository.delete_dataset(did)
            deleted += 1
    return deleted


async def fulfill_plan(
    plan_path: str | Path,
    map_path: str,
    snapshot_id: str,
    snapshot_local_path: str,
    provider_id: str,
    model_id: str,
    llm_client: LLMClient,
    instructions: dict[str, dict] | None = None,
    embedding_client: EmbeddingClient | None = None,
    codespectra_client: CodeSpectraClient | None = None,
    only_agent_ids: list[str] | None = None,
    only_kind: str | None = None,
    force_agent_ids: list[str] | None = None,
    session_id: str = "",
) -> dict[str, dict]:
    """Auto-fulfillment walk. Returns {group_key: {status, dataset_id?, reason?}}; every group in scope gets fulfilled/failed/needs_human/skipped, none silently dropped. `only_agent_ids`, when given, scopes the walk to just those agents' groups; `only_kind` additionally scopes to a single dataset kind — others are left untouched and absent from the report. `force_agent_ids` deletes and re-links each listed agent's already-fulfilled synthetic_agent_io dataset (if any) so it re-enters this walk as unfulfilled instead of being skipped."""
    suite = load_suite(plan_path)

    if force_agent_ids:
        forced = set(force_agent_ids)
        forced_any = False
        for entry in suite.entries:
            if (entry.agent_id or entry.component) not in forced:
                continue
            if not entry.id.endswith(".synthetic_agent_io"):
                continue
            if not entry.dataset or not entry.dataset.ref:
                continue
            await repository.delete_dataset(entry.dataset.ref)
            entry.dataset = DatasetRef(required={"kind": "synthetic_agent_io", "min_cases": DEFAULT_CASES_PER_AGENT})
            forced_any = True
        if forced_any:
            _save_suite(suite, plan_path)

    groups = _collect_unfulfilled_groups(suite)
    instructions = instructions or {}

    if only_agent_ids is not None:
        allowed_agent_ids = set(only_agent_ids)
        groups = {
            key: entries for key, entries in groups.items()
            if (entries[0].agent_id or entries[0].component) in allowed_agent_ids
        }
    if only_kind is not None:
        groups = {key: entries for key, entries in groups.items() if key.split("/", 1)[0] == only_kind}

    eval_enabled_by_agent: dict[str, bool] = {}
    contract_by_agent: dict[str, EvaluationContract] = {}
    profile_by_agent: dict[str, dict[str, Any]] = {}
    plan_report_path = Path(map_path).with_name(Path(map_path).stem + "_plan_report.yaml")
    plan_report = None
    if plan_report_path.exists():
        try:
            plan_report = load_plan_report(plan_report_path)
            for agent_report in plan_report.agents:
                eval_enabled_by_agent[agent_report.agent_id] = agent_report.eval_enabled
                if agent_report.contract is not None:
                    contract_by_agent[agent_report.agent_id] = agent_report.contract
                if agent_report.data_profile is not None:
                    profile_by_agent[agent_report.agent_id] = agent_report.data_profile.model_dump()
        except Exception as e:
            logger.warning(f"fulfillment: could not load plan report for synthetic_agent_io wiring: {e}")

    knowledge_by_agent: dict[str, dict[str, Any]] = {}
    if session_id and plan_report is not None:
        for agent_report in plan_report.agents:
            try:
                row = await repository.get_agent_knowledge(session_id, agent_report.agent_id)
                if not row:
                    continue
                json_path = row.get("json_path")
                if not json_path or not Path(json_path).exists():
                    continue
                raw = json.loads(Path(json_path).read_text(encoding="utf-8"))
                if raw.get("evaluation_contract") and agent_report.agent_id not in contract_by_agent:
                    try:
                        contract_by_agent[agent_report.agent_id] = EvaluationContract.model_validate(raw["evaluation_contract"])
                    except Exception as exc:
                        logger.warning(f"fulfillment: could not parse evaluation_contract for {agent_report.agent_id}: {exc}")
                functionality = raw.get("functionality") or ""
                failure_modes = [
                    fm["description"]
                    for fm in (raw.get("failure_modes") or [])
                    if isinstance(fm, dict) and fm.get("description")
                ]
                knowledge_by_agent[agent_report.agent_id] = {
                    "functionality": functionality,
                    "failure_modes": failure_modes,
                    "input_contract": raw.get("input_contract") or [],
                    "input_schemas": raw.get("input_schemas") or {},
                    "virtual_inputs": raw.get("virtual_inputs") or [],
                    "context_builders": raw.get("context_builders") or [],
                }
            except Exception as e:
                logger.warning(f"fulfillment: could not load knowledge for {agent_report.agent_id}: {e}")

    ordered_keys = sorted(
        groups,
        key=lambda k: _TOPO_ORDER.index(k.split("/", 1)[0]) if k.split("/", 1)[0] in _TOPO_ORDER else 99,
    )

    report: dict[str, dict] = {}
    dataset_id_by_group: dict[str, str] = {}

    for group_key in ordered_keys:
        kind = group_key.split("/", 1)[0]
        group_entries = groups[group_key]
        group_instructions = instructions.get(group_key, {})
        painpoint = group_instructions.get("painpoint")

        if kind == "synthetic_agent_io":
            group_agent_id = group_entries[0].agent_id or group_entries[0].component
            if not eval_enabled_by_agent.get(group_agent_id, False):
                report[group_key] = {
                    "status": "skipped",
                    "reason": "eval_enabled is False for this agent — toggle it on in Stage3Screen to generate this dataset",
                }
                continue

        if kind == "qa_testset" and not embedding_client:
            report[group_key] = {
                "status": "needs_human",
                "reason": "No embedding provider configured or available — please configure one first",
            }
            continue

        min_cases = _effective_min_cases(kind, group_entries)
        base_name = f"{kind}_{group_key.split('/', 1)[1]}"
        dataset_id = await next_version(base_name)

        seed_cases_raw = group_instructions.get("seed_cases") or []
        seed_cases = (
            seed_cases_to_dataset_cases(dataset_id, kind, seed_cases_raw) if seed_cases_raw else []
        )

        config = await _derive_config(
            kind, dataset_id, group_key, group_entries,
            map_path=map_path, snapshot_id=snapshot_id, snapshot_local_path=snapshot_local_path,
            provider_id=provider_id, model_id=model_id,
            dataset_id_by_group=dataset_id_by_group, painpoint=painpoint, min_cases=min_cases,
            codespectra_client=codespectra_client,
            group_instructions=group_instructions,
            eval_enabled_by_agent=eval_enabled_by_agent,
            contract_by_agent=contract_by_agent,
            profile_by_agent=profile_by_agent,
            knowledge_by_agent=knowledge_by_agent,
        )
        if config is None:
            report[group_key] = {
                "status": "needs_human",
                "reason": f"could not auto-derive a generator config for kind '{kind}' — needs manual input",
            }
            continue

        generator_fn = get_generator(kind)
        _uses_embedding = kind in ("qa_testset", "synthetic_agent_io")
        try:
            if _uses_embedding:
                generated_cases = await generator_fn(
                    config, llm_client, embedding_client=embedding_client
                )
            else:
                generated_cases = await generator_fn(config, llm_client)
        except Exception as e:
            logger.warning(f"fulfillment: generator for {group_key} failed: {e}")
            generated_cases = []

        all_cases = seed_cases + generated_cases

        if len(all_cases) < min_cases:
            # One top-up generation attempt before giving up.
            try:
                if _uses_embedding:
                    topup_cases = await generator_fn(
                        config, llm_client, embedding_client=embedding_client
                    )
                else:
                    topup_cases = await generator_fn(config, llm_client)
                all_cases = all_cases + topup_cases
            except Exception as e:
                logger.warning(f"fulfillment: top-up for {group_key} failed: {e}")

        # Kind-agnostic fulfillment input acceptance gate cross-check against harvested contract
        agent_id_for_group = group_entries[0].agent_id or group_entries[0].component
        contract_for_group = (contract_by_agent or {}).get(agent_id_for_group)
        if contract_for_group:
            # Fill harness-supplied run handles (snapshot_id etc.) into cases — injected at execution, not synthetic content the generator can invent; only required, case-bound (non-config), currently-missing kwargs.
            _run_env = {
                "snapshot_id": snapshot_id, "snapshot_local_path": snapshot_local_path,
                "provider_id": provider_id, "model_id": model_id, "map_path": map_path,
            }
            _inv = contract_for_group.invocation
            _config_names = config_kwarg_names_from_case_binding(_inv.case_binding if _inv else {})
            _fill = {
                kw.name: _run_env[kw.name]
                for kw in (_inv.kwargs if _inv else [])
                if kw.required and kw.name in _run_env and kw.name not in _config_names and _run_env[kw.name]
            }
            if _fill:
                for c in all_cases:
                    if isinstance(c.input, dict):
                        for _n, _v in _fill.items():
                            c.input.setdefault(_n, _v)
            valid_cases = []
            for c in all_cases:
                # Pipeline-driver cases (shape=kwargs, e.g. snapshot_fixture) drive a full run; their input is not the agent's direct contract input, so the per-agent input gate doesn't apply.
                if isinstance(c.input, dict) and c.input.get("shape") == "kwargs":
                    valid_cases.append(c)
                    continue
                input_errs = _validate_case_input_against_contract(c.input, contract_for_group)
                if input_errs:
                    logger.warning(f"fulfillment: case {c.id} input failed contract validation: {input_errs}")
                else:
                    valid_cases.append(c)
            all_cases = valid_cases

        if len(all_cases) == 0:
            report[group_key] = {
                "status": "failed",
                "reason": f"generated {len(all_cases)} cases, need >= {min_cases} — zero cases produced",
            }
            continue

        # Persist partial datasets; only fail on zero cases
        max_sims = [
            c.labels["max_sim"] for c in all_cases
            if isinstance(c.labels, dict) and "max_sim" in c.labels
        ]
        dataset_metrics = {"diversity": round(1.0 - sum(max_sims) / len(max_sims), 4)} if max_sims else None

        pruned = await _prune_prior_versions(base_name, keep=dataset_id)
        if pruned:
            logger.info(f"fulfillment: pruned {pruned} old version(s) of {base_name}")
        await repository.insert_dataset_cases_bulk(dataset_id, all_cases)
        await repository.insert_dataset_metadata(
            dataset_id, kind,
            instructions=group_instructions or None,
            source_gate_ids=[e.id for e in group_entries],
            min_cases=min_cases,
            metrics=dataset_metrics,
            expansion_session_id=session_id or None,
        )
        dataset_id_by_group[group_key] = dataset_id

        if len(all_cases) < min_cases:
            report[group_key] = {
                "status": "needs_human",
                "reason": f"generated {len(all_cases)} of {min_cases} cases — dataset persisted but undersized",
                "dataset_id": dataset_id,
            }
        else:
            report[group_key] = {"status": "fulfilled", "dataset_id": dataset_id}

    if dataset_id_by_group:
        for entry in suite.entries:
            if not entry.dataset or not entry.dataset.required or entry.dataset.ref:
                continue
            kind = entry.dataset.required.get("kind")
            if not kind:
                continue
            key = _group_key(kind, entry)
            if key in dataset_id_by_group:
                entry.dataset.ref = dataset_id_by_group[key]
                entry.dataset.required = None
        _save_suite(suite, plan_path)

    return report


async def export_dataset(dataset_id: str) -> list:
    """Only `generated+reviewed`/`handwritten` cases ever leave AEH; `synthetic` never does."""
    from agent_eval_harness.datasets.types import DatasetCase

    metadata = await repository.get_dataset_metadata(dataset_id)
    kind = metadata["kind"] if metadata else ""

    db_cases = await repository.get_dataset_cases(dataset_id)
    out: list[DatasetCase] = []
    for db_case in db_cases:
        if db_case["provenance"] not in TRUSTED_PROVENANCE:
            continue
        input_data = json.loads(db_case["input_json"])
        input_data.pop("kind", None)  # kind-sniffing artifact from an older storage shape
        out.append(
            DatasetCase(
                id=db_case["id"],
                dataset=dataset_id,
                kind=kind,
                input=input_data,
                expected=json.loads(db_case["expected_json"]) if db_case["expected_json"] else None,
                labels=json.loads(db_case["labels_json"]) if db_case["labels_json"] else None,
                provenance=db_case["provenance"],
            )
        )
    return out
