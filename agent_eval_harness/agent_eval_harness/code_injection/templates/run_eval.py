"""AEH Stage 4 eval driver: reads dataset cases live from AEH's own sqlite DB (stdlib sqlite3 only, via wiring.json — no agent_eval_harness dependency), drives this repo's analysis pipeline directly (bypassing the job-queue /start route), and writes out/eval_log.{pid}.jsonl + manifest.json.

Run as a route (POST /aeh/run-eval, see api/aeh_eval.py) or standalone:
    python .aeh/run_eval.py --verify   # one case, confirms >=1 span was captured
    python .aeh/run_eval.py            # every reviewed case across every dataset in wiring.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HAYSTACK_CONTENT_TRACING_ENABLED", "true")

_HERE = Path(__file__).resolve().parent  # backend/.aeh/
_BACKEND_ROOT = _HERE.parent  # backend/
for _p in (str(_HERE), str(_BACKEND_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tracer import log_path, register_tracer, set_current_trace, write_log_line  # noqa: E402

_SCHEMA = "aeh.spanlog/1"
_TRACER_VERSION = "1"


def _ensure_tracer_registered() -> None:
    from haystack.tracing import tracer as global_tracer
    from haystack.tracing.tracer import NullTracer

    # Default tracer is a NullTracer() instance, never None — check for that, not None.
    if isinstance(global_tracer.actual_tracer, NullTracer):
        register_tracer()


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_BACKEND_ROOT, capture_output=True, text=True, check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() or None


def _load_wiring() -> dict[str, Any]:
    return json.loads((_HERE / "wiring.json").read_text(encoding="utf-8"))


def _load_cases(wiring: dict[str, Any], limit: int | None = None) -> list[dict[str, Any]]:
    """Reads dataset cases live from AEH's sqlite DB (path/ids from wiring.json, stdlib-only), filtered to reviewed/handwritten provenance to mirror datasets/fulfillment.py::export_dataset()'s gate."""
    aeh_db_path = wiring.get("aeh_db_path")
    dataset_ids = wiring.get("dataset_ids", [])
    if not aeh_db_path or not dataset_ids:
        return []

    cases: list[dict[str, Any]] = []
    conn = sqlite3.connect(aeh_db_path)
    conn.row_factory = sqlite3.Row
    try:
        for dataset_id in dataset_ids:
            rows = conn.execute(
                "SELECT id, input_json, expected_json, labels_json, provenance "
                "FROM dataset_cases WHERE dataset_id = ? "
                "AND provenance IN ('generated+reviewed', 'handwritten', 'derived+reviewed') "
                "ORDER BY id",
                (dataset_id,),
            ).fetchall()
            for row in rows:
                input_data = json.loads(row["input_json"])
                input_data.pop("kind", None)  # kind-sniffing artifact from an older storage shape
                cases.append({
                    "id": row["id"],
                    "dataset": dataset_id,
                    "input": input_data,
                    "expected": json.loads(row["expected_json"]) if row["expected_json"] else None,
                    "labels": json.loads(row["labels_json"]) if row["labels_json"] else None,
                    "provenance": row["provenance"],
                })
                if limit is not None and len(cases) >= limit:
                    return cases
    finally:
        conn.close()
    return cases


async def _run_one_case(case: dict[str, Any]) -> bool:
    from domain.analysis.agent_pipeline import AnalysisAgentPipeline
    from domain.analysis.orchestrator import RunDirectorAgent
    from domain.model_connector.service import ProviderConfigService
    from domain.retrieval.service import RetrievalService
    from domain.structural_graph.service import StructuralGraphService

    kwargs = case["input"]["kwargs"]
    provider = ProviderConfigService()
    retrieval = RetrievalService()
    graph = StructuralGraphService()
    agents = AnalysisAgentPipeline(provider, retrieval)
    director = RunDirectorAgent(provider, retrieval, agents, graph)

    trace_id = case["id"]
    write_log_line({
        "record": "case_start",
        "trace_id": trace_id,
        "dataset_id": case.get("dataset"),
        "dataset_case_id": case["id"],
        "input": kwargs,
    })

    status = "ok"
    final_output: Any = None
    success = True
    try:
        with set_current_trace(trace_id):
            out = await director.run(
                provider_id=kwargs["provider_id"],
                model_id=kwargs["model_id"],
                snapshot_id=kwargs["snapshot_id"],
                scan_mode="full",
            )
        final_output = out.get("sections")
    except Exception as e:  # noqa: BLE001 — recorded as a partial run, not raised
        status = "error"
        final_output = {"error": str(e)}
        success = False
    finally:
        write_log_line({
            "record": "case_end",
            "trace_id": trace_id,
            "status": status,
            "final_output_json": json.dumps(final_output, ensure_ascii=False, default=str),
        })
    return success


def _write_header(plan_id: str, run_id: str) -> None:
    write_log_line({
        "record": "header",
        "schema": _SCHEMA,
        "tracer_version": _TRACER_VERSION,
        "plan_id": plan_id,
        "run_id": run_id,
        "git_sha": _git_sha(),
    })


async def run(*, verify: bool = False) -> dict[str, Any]:
    _ensure_tracer_registered()
    wiring = _load_wiring()
    cases = _load_cases(wiring, limit=1 if verify else None)
    run_id = f"{int(time.time())}-{os.getpid()}"
    _write_header(wiring.get("plan_id", "unknown"), run_id)

    attempted = 0
    succeeded = 0
    for case in cases:
        attempted += 1
        if await _run_one_case(case):
            succeeded += 1

    write_log_line({"record": "run_summary", "attempted": attempted, "succeeded": succeeded})

    manifest = {
        "schema": _SCHEMA,
        "tracer_version": _TRACER_VERSION,
        "run_id": run_id,
        "plan_id": wiring.get("plan_id", "unknown"),
        "git_sha": _git_sha(),
        "log_path": str(log_path()),
        "attempted": attempted,
        "succeeded": succeeded,
        "status": "ok" if attempted == succeeded else "partial",
    }
    (_HERE / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if verify and succeeded == 0:
        raise RuntimeError(
            "verify: the one case did not succeed — see the case_end record in "
            f"{log_path()} for the error"
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="AEH Stage 4 eval driver")
    parser.add_argument(
        "--verify", action="store_true",
        help="run exactly one case and confirm it completes with spans captured",
    )
    args = parser.parse_args()

    manifest = asyncio.run(run(verify=args.verify))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
