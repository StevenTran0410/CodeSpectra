"""AEH Stage 4 eval driver: reads dataset cases live from AEH's own sqlite DB (stdlib sqlite3 only, via wiring.json — no agent_eval_harness dependency), drives this repo's analysis pipeline directly (bypassing the job-queue /start route), and writes out/eval_log.{pid}.jsonl + manifest.json.

Run as a route (POST /aeh/run-eval, see api/aeh_eval.py) or standalone:
    python .aeh/run_eval.py --verify --agent <id>   # one case for that agent
    python .aeh/run_eval.py --verify --batch <n>    # one case per agent in batch n
    python .aeh/run_eval.py --verify                # one case overall (smoke)
    python .aeh/run_eval.py                         # every reviewed case in wiring.json
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HAYSTACK_CONTENT_TRACING_ENABLED", "true")

_HERE = Path(__file__).resolve().parent  # <import_root>/.aeh/
_IMPORT_ROOT = _HERE.parent  # the target's Python import root
for _p in (str(_HERE), str(_IMPORT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tracer import (  # noqa: E402
    log_path,
    register_tracer,
    set_current_trace,
    span_facts_since,
    spans_written,
    write_log_line,
)

_SCHEMA = "aeh.spanlog/1"
_TRACER_VERSION = "1"

# Cases whose labels carry no owning agent have no dispatch module; counted, never silent.
_SKIPPED_NO_AGENT = 0


def _ensure_tracer_registered() -> None:
    from haystack.tracing import tracer as global_tracer
    from haystack.tracing.tracer import NullTracer

    # Default tracer is a NullTracer() instance, never None — check for that, not None.
    if isinstance(global_tracer.actual_tracer, NullTracer):
        register_tracer()


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_IMPORT_ROOT, capture_output=True, text=True, check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() or None


def _load_wiring() -> dict[str, Any]:
    return json.loads((_HERE / "wiring.json").read_text(encoding="utf-8"))


def _load_run_config() -> dict[str, Any]:
    """Runtime config written by task T004; missing file = empty, dispatch fails loudly."""
    path = _HERE / "run_config.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as e:
        print(f"warning: could not read {path}: {e}", file=sys.stderr)
        return {}


def _agents_in_batch(wiring: dict[str, Any], batch: int) -> list[str]:
    """Agent ids for a 1-based batch number, using the batching frozen into wiring.json."""
    batches = wiring.get("batches") or []
    if batch < 1 or batch > len(batches):
        raise SystemExit(f"batch {batch} out of range (wiring.json defines {len(batches)})")
    return list(batches[batch - 1])


def _load_cases(
    wiring: dict[str, Any],
    agents: set[str] | None = None,
    limit: int | None = None,
    per_agent_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Reads dataset cases live from AEH's sqlite DB (path/ids from wiring.json, stdlib-only), filtered to reviewed/handwritten provenance to mirror datasets/fulfillment.py::export_dataset()'s gate. A case's `agent_id` label selects its dispatch module, so a case without one is skipped."""
    aeh_db_path = wiring.get("aeh_db_path")
    dataset_ids = wiring.get("dataset_ids", [])
    if not aeh_db_path or not dataset_ids:
        return []

    global _SKIPPED_NO_AGENT
    _SKIPPED_NO_AGENT = 0
    cases: list[dict[str, Any]] = []
    per_agent_seen: dict[str, int] = {}
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
                labels = json.loads(row["labels_json"]) if row["labels_json"] else None
                agent_id = (labels or {}).get("agent_id") or ""
                if not agent_id:
                    _SKIPPED_NO_AGENT += 1  # no owning agent -> no dispatch module can run it
                    continue
                if agents is not None and agent_id not in agents:
                    continue
                if per_agent_limit is not None and per_agent_seen.get(agent_id, 0) >= per_agent_limit:
                    continue
                per_agent_seen[agent_id] = per_agent_seen.get(agent_id, 0) + 1
                cases.append({
                    "id": row["id"],
                    "dataset": dataset_id,
                    "agent_id": agent_id,
                    "input": input_data,
                    "expected": json.loads(row["expected_json"]) if row["expected_json"] else None,
                    "labels": labels,
                    "provenance": row["provenance"],
                })
                if limit is not None and len(cases) >= limit:
                    return cases
    finally:
        conn.close()
    if _SKIPPED_NO_AGENT:
        print(
            f"warning: skipped {_SKIPPED_NO_AGENT} reviewed case(s) with no agent_id label — "
            "no dispatch module owns them",
            file=sys.stderr,
        )
    return cases


async def _run_one_case(case: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Generic per-case dispatcher. Loads dispatch/<agent_id>.py and invokes it.

    This is the generic harness — agent-specific dispatch logic is in dispatch/<agent>.py
    (generated by codegen.py). This keeps run_eval.py target-agnostic.
    """
    import importlib.util
    import sys

    agent_id = case.get("agent_id", "unknown")
    trace_id = case["id"]
    spans_before = spans_written()

    write_log_line({
        "record": "case_start",
        "trace_id": trace_id,
        "dataset_id": case.get("dataset"),
        "dataset_case_id": case["id"],
        "agent_id": agent_id,
        "input": case.get("input", {}),
    })

    status = "ok"
    final_output: Any = None
    success = True
    skipped = False
    try:
        # Dispatch modules live beside this file, so a different cwd cannot break the run.
        dispatch_module_path = _HERE / "dispatch" / f"{agent_id}.py"
        if dispatch_module_path.exists():
            spec = importlib.util.spec_from_file_location(f"dispatch_{agent_id}", dispatch_module_path)
            dispatch_module = importlib.util.module_from_spec(spec)
            sys.modules[f"dispatch_{agent_id}"] = dispatch_module
            spec.loader.exec_module(dispatch_module)

            # Honour the dispatch module's skip_case() if present — a genuinely unsupported
            # agent should be tallied as skipped, not crashed as an error.
            if hasattr(dispatch_module, "skip_case") and dispatch_module.skip_case(case):
                status = "skipped"
                skipped = True
                final_output = {"skipped": True}
            else:
                # Call the dispatch module's invoke_agent function
                with set_current_trace(trace_id):
                    final_output = await dispatch_module.invoke_agent(case, config)
                success = True
        else:
            status = "error"
            final_output = {"error": f"dispatch module not found: {dispatch_module_path}"}
            success = False
    except Exception as e:  # noqa: BLE001 — recorded as a partial run, not raised
        status = "error"
        final_output = {"error": str(e)}
        success = False
    facts = span_facts_since(spans_before)
    spans = len(facts)
    output_json = json.dumps(final_output, ensure_ascii=False, default=str)
    write_log_line({
        "record": "case_end",
        "trace_id": trace_id,
        "status": status,
        "spans": spans,
        "final_output_json": output_json,
    })
    latencies = [f["latency_ms"] for f in facts if isinstance(f.get("latency_ms"), int)]
    return {
        "agent_id": agent_id,
        "trace_id": trace_id,
        "success": success,
        "skipped": skipped,
        "spans": spans,
        "latency_ms": sum(latencies) if latencies else 0,
        "span_error": next((f["error"] for f in facts if f.get("error")), None),
        "output_len": len(output_json),
        "output_digest": hashlib.sha256(output_json.encode("utf-8")).hexdigest(),
    }


def _degenerate_findings(results: list[dict[str, Any]], wiring: dict[str, Any]) -> list[str]:
    """Cases that returned without doing real work.

    An agent that catches its own exception and returns a schema-valid empty result counts as
    a success everywhere else, so "did it return?" cannot be the gate. Every signal below is
    target-neutral — none inspects the shape or wording of the target's own output.
    """
    gate = wiring.get("gate") or {}
    findings: list[str] = []

    for r in results:
        if r["spans"] == 0:
            findings.append(f"{r['agent_id']}/{r['trace_id']}: no spans — nothing observable ran")
        if r["span_error"]:
            findings.append(f"{r['agent_id']}/{r['trace_id']}: span reported error: {r['span_error']}")

    by_agent: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        by_agent.setdefault(r["agent_id"], []).append(r)

    # Latency floor is derived from the agent's own cases, never a hardcoded millisecond count:
    # what counts as "impossibly fast" depends entirely on the target being measured.
    ratio = gate.get("latency_floor_ratio", 0.1)
    min_peers = gate.get("latency_min_peers", 3)
    for agent_id, rs in by_agent.items():
        lat = sorted(r["latency_ms"] for r in rs)
        if len(lat) < min_peers:
            continue
        median = lat[len(lat) // 2]
        floor = median * ratio
        for r in rs:
            if r["latency_ms"] < floor:
                findings.append(
                    f"{agent_id}/{r['trace_id']}: {r['latency_ms']}ms is far below this agent's "
                    f"median {median}ms — it likely failed instantly and was swallowed"
                )

    # Different inputs producing a byte-identical output means a constant is being returned;
    # this catches a fallback value without needing to know what the fallback looks like.
    for agent_id, rs in by_agent.items():
        if len(rs) < 2:
            continue
        seen: dict[str, str] = {}
        for r in rs:
            first = seen.setdefault(r["output_digest"], r["trace_id"])
            if first != r["trace_id"]:
                findings.append(
                    f"{agent_id}/{r['trace_id']}: output byte-identical to {first} despite a "
                    "different case input — likely a constant fallback, not a real answer"
                )
    return findings


def _print_run_table(results: list[dict[str, Any]]) -> None:
    """Per-agent latency/output spread, so a degenerate run is visible without a bespoke script."""
    by_agent: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        by_agent.setdefault(r["agent_id"], []).append(r)
    print(f"\n{'agent':<28}{'cases':>6}{'spans':>7}{'lat_min':>9}{'lat_med':>9}{'lat_max':>9}{'out_min':>9}", file=sys.stderr)
    for agent_id in sorted(by_agent):
        rs = by_agent[agent_id]
        lat = sorted(r["latency_ms"] for r in rs)
        out = sorted(r["output_len"] for r in rs)
        print(
            f"{agent_id:<28}{len(rs):>6}{sum(r['spans'] for r in rs):>7}"
            f"{lat[0]:>9}{lat[len(lat) // 2]:>9}{lat[-1]:>9}{out[0]:>9}",
            file=sys.stderr,
        )


def _write_header(plan_id: str, run_id: str) -> None:
    write_log_line({
        "record": "header",
        "schema": _SCHEMA,
        "tracer_version": _TRACER_VERSION,
        "plan_id": plan_id,
        "run_id": run_id,
        "git_sha": _git_sha(),
    })


async def _init_target(config: dict[str, Any]) -> None:
    """Run the target's own process setup, which lives in an agent-owned file.

    Kept out of this file so re-copying a newer run_eval.py never erases that work.
    """
    try:
        from target_init import init_target
    except ImportError:
        raise SystemExit(
            ".aeh/target_init.py is missing — create it from CODE.md and implement it"
        )
    result = init_target(config)
    if inspect.isawaitable(result):
        await result


async def _teardown_target(config: dict[str, Any]) -> None:
    """Release what _init_target opened. Never fatal: the run's results already exist."""
    try:
        from target_init import teardown_target
    except ImportError:
        return  # an older target_init.py predates this hook
    try:
        result = teardown_target(config)
        if inspect.isawaitable(result):
            await result
    except Exception as exc:  # noqa: BLE001 — a teardown failure must not void a finished run
        print(f"warning: teardown_target failed: {exc}", file=sys.stderr)


def _out_dir(wiring: dict[str, Any]) -> Path:
    """Run output goes to the harness data dir from wiring.json, never the target's tree."""
    configured = wiring.get("aeh_out_dir")
    out = Path(configured) if configured else _HERE / "out"
    out.mkdir(parents=True, exist_ok=True)
    os.environ["AEH_OUT_DIR"] = str(out)  # the tracer reads this when it opens the log
    return out


async def run(
    *, verify: bool = False, agent: str | None = None, batch: int | None = None
) -> dict[str, Any]:
    wiring = _load_wiring()
    out_dir = _out_dir(wiring)
    _ensure_tracer_registered()
    config = _load_run_config()
    await _init_target(config)

    selected: set[str] | None = None
    if agent:
        selected = {agent}
    elif batch is not None:
        selected = set(_agents_in_batch(wiring, batch))

    # --verify runs one case per selected agent so a batch gate proves every agent in it.
    if verify:
        cases = _load_cases(
            wiring, agents=selected,
            per_agent_limit=1,
            limit=1 if selected is None else None,
        )
    else:
        cases = _load_cases(wiring, agents=selected)

    run_id = f"{int(time.time())}-{os.getpid()}"
    _write_header(wiring.get("plan_id", "unknown"), run_id)

    attempted = 0
    succeeded = 0
    skipped = 0
    per_agent: dict[str, dict[str, int]] = {}
    results: list[dict[str, Any]] = []
    try:
        for case in cases:
            aid = case.get("agent_id", "unknown")
            stats = per_agent.setdefault(aid, {"attempted": 0, "succeeded": 0, "skipped": 0, "spans": 0})
            outcome = await _run_one_case(case, config)
            results.append(outcome)
            stats["spans"] += outcome["spans"]
            if outcome.get("skipped"):
                skipped += 1
                stats["skipped"] += 1
            else:
                attempted += 1
                stats["attempted"] += 1
                if outcome["success"]:
                    succeeded += 1
                    stats["succeeded"] += 1
    finally:
        # Without this the process can hang after the last case: a non-daemon worker thread
        # left by the target keeps it alive, and atexit runs too late to help.
        await _teardown_target(config)

    findings = _degenerate_findings(results, wiring)
    if results:
        _print_run_table(results)
    write_log_line({
        "record": "run_summary",
        "attempted": attempted,
        "succeeded": succeeded,
        "degenerate": findings,
    })

    manifest = {
        "schema": _SCHEMA,
        "tracer_version": _TRACER_VERSION,
        "run_id": run_id,
        "plan_id": wiring.get("plan_id", "unknown"),
        "git_sha": _git_sha(),
        "log_path": str(log_path()),
        "scope": {"agent": agent, "batch": batch, "verify": verify},
        "attempted": attempted,
        "succeeded": succeeded,
        "skipped": skipped,
        "skipped_no_agent": _SKIPPED_NO_AGENT,
        "per_agent": per_agent,
        "degenerate": findings,
        # A run that returned without doing the work is NOT ok, whatever the success count says.
        "status": (
            "degraded" if findings
            else "ok" if attempted and attempted == succeeded
            else "partial"
        ),
    }
    # Per-run copy so an earlier gate's evidence survives the next run, plus a stable pointer.
    (out_dir / f"manifest.{run_id}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if verify:
        if attempted == 0:
            raise SystemExit(
                "verify: no reviewed cases matched this scope — check wiring.json "
                "dataset_ids and that the cases carry an agent_id label"
            )
        failed = sorted(a for a, s in per_agent.items() if s["succeeded"] == 0)
        if failed:
            raise SystemExit(
                "verify failed for: " + ", ".join(failed)
                + f" — see the case_end records in {log_path()}"
            )

    # Applies to the full run too, not just --verify: a swallowed failure is exactly what an
    # unguarded full run reports as a clean success.
    if findings:
        raise SystemExit(
            f"{len(findings)} case(s) reported success without doing the work:\n  "
            + "\n  ".join(findings)
            + f"\nManifest status is 'degraded'. Evidence: {log_path()}"
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="AEH Stage 4 eval driver")
    parser.add_argument(
        "--verify", action="store_true",
        help="run one case per selected agent and require each to succeed",
    )
    parser.add_argument("--agent", default=None, help="restrict the run to a single agent id")
    parser.add_argument("--batch", type=int, default=None, help="restrict the run to a 1-based batch from wiring.json")
    args = parser.parse_args()
    if args.agent and args.batch is not None:
        raise SystemExit("--agent and --batch are mutually exclusive")

    manifest = asyncio.run(run(verify=args.verify, agent=args.agent, batch=args.batch))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
