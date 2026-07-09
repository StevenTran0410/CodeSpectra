"""Auto-fulfillment walk: turns a plan's unfulfilled `dataset.required` blocks into real
datasets, one per (kind, agent-or-component) group, or an explicit failed/needs_human reason."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from agent_eval_harness.datasets.generator_utils import seed_cases_to_dataset_cases
from agent_eval_harness.datasets.registry import get_generator
from agent_eval_harness.datasets.versioning import next_version
from agent_eval_harness.llm.client import LLMClient
from agent_eval_harness.llm.embedding_client import EmbeddingClient
from agent_eval_harness.metrics.suite import Suite, SuiteEntry, load_suite
from agent_eval_harness.store import repository

logger = logging.getLogger("agent_eval_harness.datasets.fulfillment")

# Some kinds are 1:1 with an execution, not a text-sample pool — min_cases scales differently.
_MIN_CASES_OVERRIDE: dict[str, int] = {
    "snapshot_fixture": 1,
    "snapshot_regression_baseline": 1,
    "field_match_gold": 1,
}
_DEFAULT_MIN_CASES: dict[str, int] = {"guard_classification": 40}

# Producers must run before consumers (sufficiency_labeled needs a same-run qa_testset id).
_TOPO_ORDER = [
    "qa_testset", "decomposition_gold", "guard_classification",
    "snapshot_fixture", "field_match_gold",
    "sufficiency_labeled", "snapshot_regression_baseline",
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
    return max(declared) if declared else _DEFAULT_MIN_CASES.get(kind, 20)


def _qa_corpus_paths(local_path: str) -> list[str]:
    """corpus = **/*.md + **/*.txt under the snapshot, excluding .aeh/** — a RAG target
    must never be able to retrieve its own generated answer key."""
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
    """Match the qa_testset synthesis backend to the toolkit the consuming gate(s)
    actually score with at eval time — llm_judge entries carry it as the metric's
    namespace prefix (e.g. "ragas.faithfulness", "geval.decomposition_coverage").
    A group can only be produced once, so a mix of consumers still gets one real
    backend rather than silently guessing; ragas wins the tie since its metrics are
    pickier about context/answer shape than deepeval's G-Eval rubric scoring."""
    if any(e.metric.startswith("ragas.") for e in group_entries):
        return "ragas"
    return "deepeval"


def _derive_config(
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
) -> dict[str, Any] | None:
    """Returns a generator config dict, or None if underivable (caller marks needs_human)."""
    component = group_entries[0].component

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
        return {
            "dataset_name": dataset_id, "snapshot_ids": [snapshot_id],
            "provider_id": provider_id, "model_id": model_id,
        }

    if kind == "field_match_gold":
        return {"dataset_name": dataset_id, "snapshot_id": snapshot_id, "local_path": snapshot_local_path}

    return None


def _save_suite(suite: Suite, plan_path: str | Path) -> None:
    Path(plan_path).write_text(yaml.dump(suite.model_dump(), allow_unicode=True), encoding="utf-8")


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
) -> dict[str, dict]:
    """Auto-fulfillment walk. Returns {group_key: {status, dataset_id?, reason?}} — every
    group gets exactly one of fulfilled/failed/needs_human, none silently skipped."""
    suite = load_suite(plan_path)
    groups = _collect_unfulfilled_groups(suite)
    instructions = instructions or {}

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

        if kind == "snapshot_regression_baseline":
            report[group_key] = {
                "status": "needs_human",
                "reason": (
                    "snapshot_regression_baseline requires a live target-execution seam "
                    "that doesn't exist yet — cannot auto-generate"
                ),
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

        config = _derive_config(
            kind, dataset_id, group_key, group_entries,
            map_path=map_path, snapshot_id=snapshot_id, snapshot_local_path=snapshot_local_path,
            provider_id=provider_id, model_id=model_id,
            dataset_id_by_group=dataset_id_by_group, painpoint=painpoint, min_cases=min_cases,
        )
        if config is None:
            report[group_key] = {
                "status": "needs_human",
                "reason": f"could not auto-derive a generator config for kind '{kind}' — needs manual input",
            }
            continue

        generator_fn = get_generator(kind)
        try:
            if kind == "qa_testset":
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
                if kind == "qa_testset":
                    topup_cases = await generator_fn(
                        config, llm_client, embedding_client=embedding_client
                    )
                else:
                    topup_cases = await generator_fn(config, llm_client)
                all_cases = all_cases + topup_cases
            except Exception as e:
                logger.warning(f"fulfillment: top-up for {group_key} failed: {e}")

        if len(all_cases) < min_cases:
            report[group_key] = {
                "status": "failed",
                "reason": f"generated {len(all_cases)} cases, need >= {min_cases} — undersized dataset not written",
            }
            continue

        await repository.insert_dataset_cases_bulk(dataset_id, all_cases)
        await repository.insert_dataset_metadata(
            dataset_id, kind,
            instructions=group_instructions or None,
            source_gate_ids=[e.id for e in group_entries],
            min_cases=min_cases,
        )
        dataset_id_by_group[group_key] = dataset_id
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
    """Only `generated+reviewed`/`handwritten` cases ever leave AEH; `synthetic` never does.
    Called at inject time — injection refuses when the dataset isn't review-complete."""
    from agent_eval_harness.datasets.types import DatasetCase

    metadata = await repository.get_dataset_metadata(dataset_id)
    kind = metadata["kind"] if metadata else ""

    db_cases = await repository.get_dataset_cases(dataset_id)
    out: list[DatasetCase] = []
    for db_case in db_cases:
        if db_case["provenance"] not in ("generated+reviewed", "handwritten"):
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
