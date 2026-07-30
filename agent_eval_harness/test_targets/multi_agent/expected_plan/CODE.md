# CODE — Generated Artifacts

Each code block below is ready to copy verbatim to the specified file.

---

## Tracer

→ Create file at `.aeh/tracer.py`

**SHA256**: c0e1eb07a982bce10d3e3119c47004d9ecfb430be5a530e762c112d3881bb426

```python
"""AEH Stage 4 span logger: implements Haystack's own tracing.Tracer protocol (matching AEH's in-process HarnessTracer) so spans carry real component identity rather than tag-matching, and appends JSON lines to out/eval_log.{pid}.jsonl without a lock since asyncio only interleaves at await points."""
from __future__ import annotations

import contextlib
import contextvars
import json
import os
import re
import time
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from haystack.tracing import Span as HaystackSpan
from haystack.tracing import Tracer as HaystackTracer

_PIPELINE_ROOT_OPS = {"haystack.pipeline.run", "haystack.async_pipeline.run"}


def _out_dir() -> Path:
    """Run output belongs in the harness data dir, never in the target's source tree."""
    env = os.getenv("AEH_OUT_DIR")
    return Path(env) if env else Path(__file__).resolve().parent / "out"

_current_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "aeh_current_trace_id", default=None
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


_REDACT_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{8,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE),
    re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),  # JWT
]

_SPANS_WRITTEN = 0
# Lightweight per-span facts the driver's gate reads back. Kept to scalars so a long run
# cannot grow this without bound the way holding the span payloads would.
_SPAN_FACTS: list[dict[str, Any]] = []


def _redact(text: str) -> str:
    for pattern in _REDACT_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def spans_written() -> int:
    """Span records emitted so far; the driver samples this to prove a case did work."""
    return _SPANS_WRITTEN


def span_facts_since(marker: int) -> list[dict[str, Any]]:
    """Facts for spans emitted after `marker` (a prior `spans_written()`). The driver's gate
    reads these to tell a case that did real work from one that returned without doing any."""
    return _SPAN_FACTS[marker:]


def write_log_line(record: dict[str, Any]) -> None:
    global _SPANS_WRITTEN
    if record.get("record") == "span":
        _SPANS_WRITTEN += 1
        _SPAN_FACTS.append({
            "latency_ms": record.get("latency_ms"),
            "error": record.get("error"),
            "output_len": len(record.get("output_json") or ""),
        })
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Redact on the way to disk: an upstream error body can echo a credential, and a log
    # file that already holds one cannot be un-written by redacting at ingest time.
    line = _redact(json.dumps(record, ensure_ascii=False, default=str))
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.write("\n")


def log_path() -> Path:
    return _out_dir() / f"eval_log.{os.getpid()}.jsonl"


@contextlib.contextmanager
def set_current_trace(trace_id: str) -> Iterator[None]:
    """Wraps a dataset case's pipeline invocation so spans emitted while active are stamped with this trace_id, giving each case its own root trace."""
    token = _current_trace_id.set(trace_id)
    try:
        yield
    finally:
        _current_trace_id.reset(token)


class AehSpan(HaystackSpan):
    def __init__(self, span_id: str, parent_span_id: str | None, operation_name: str) -> None:
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self.operation_name = operation_name
        self.tags: dict[str, Any] = {}
        self.started_at = _utc_now_iso()
        self.latency_ms: int | None = None
        self._start_monotonic = time.monotonic()

    def set_tag(self, key: str, value: Any) -> None:
        self.tags[key] = value


def _classify_span_type(operation_name: str, tags: dict[str, Any]) -> str:
    if tags.get("haystack.component.type"):
        return "agent"
    return "other"


def _extract_tokens(output: Any) -> tuple[str | None, int | None, int | None, str | None]:
    """Best-effort: looks for usage under common key names in the agent's dict output; anything not found is left None (never guessed), and token_source is "measured" only when a real usage block was found."""
    if not isinstance(output, dict):
        return None, None, None, None
    usage = output.get("usage") or output.get("token_usage")
    if isinstance(usage, dict):
        tokens_in = usage.get("prompt_tokens") or usage.get("input_tokens")
        tokens_out = usage.get("completion_tokens") or usage.get("output_tokens")
        model = usage.get("model") or output.get("model")
        if tokens_in is not None or tokens_out is not None:
            return model, tokens_in, tokens_out, "measured"
    return output.get("model"), None, None, None


def _safe_json(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return json.dumps(str(value))


class AehTracer(HaystackTracer):
    def __init__(self) -> None:
        self._current: contextvars.ContextVar[AehSpan | None] = contextvars.ContextVar(
            "aeh_current_span", default=None
        )

    @contextlib.contextmanager
    def trace(
        self,
        operation_name: str,
        tags: dict[str, Any] | None = None,
        parent_span: HaystackSpan | None = None,
    ) -> Iterator[AehSpan]:
        parent = parent_span if isinstance(parent_span, AehSpan) else self._current.get()
        # Pipeline-root spans are never written, so their direct children are top-level (never a dangling parent_span_id).
        has_written_parent = parent is not None and parent.operation_name not in _PIPELINE_ROOT_OPS
        parent_id = parent.span_id if has_written_parent else None
        span = AehSpan(str(uuid.uuid4()), parent_id, operation_name)
        if tags:
            span.set_tags(tags)
        token = self._current.set(span)
        error: BaseException | None = None
        try:
            yield span
        except BaseException as exc:  # noqa: BLE001 — re-raised below, only recorded here
            error = exc
            raise
        finally:
            self._current.reset(token)
            span.latency_ms = int((time.monotonic() - span._start_monotonic) * 1000)
            if operation_name not in _PIPELINE_ROOT_OPS:
                self._write_span(span, error)

    def current_span(self) -> HaystackSpan | None:
        return self._current.get()

    def _write_span(self, span: AehSpan, error: BaseException | None) -> None:
        output = span.tags.get("haystack.component.output")
        input_value = span.tags.get("haystack.component.input")
        model, tokens_in, tokens_out, token_source = _extract_tokens(output)
        write_log_line({
            "record": "span",
            "trace_id": _current_trace_id.get(),
            "span_id": span.span_id,
            "parent_span_id": span.parent_span_id,
            "component_id": span.tags.get("haystack.component.name"),
            "span_type": _classify_span_type(span.operation_name, span.tags),
            "operation": span.operation_name,
            "started_at": span.started_at,
            "latency_ms": span.latency_ms,
            "input_json": _safe_json(input_value),
            "output_json": _safe_json(output),
            "model": model,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "token_source": token_source,
            "error": str(error) if error is not None else None,
        })


def register_tracer() -> None:
    """Call once, before any other `haystack` import in the process (haystack.tracing checks HAYSTACK_CONTENT_TRACING_ENABLED only at its own first import)."""
    from haystack.tracing import tracer as haystack_global_tracer
    from haystack.tracing import enable_tracing

    if not haystack_global_tracer.is_content_tracing_enabled:
        raise RuntimeError(
            "HAYSTACK_CONTENT_TRACING_ENABLED must be 'true' before any `haystack` import in "
            "this process. Set it as the literal first statement of the entry point."
        )
    enable_tracing(AehTracer())

```

---

## Run Eval Script

→ Create file at `.aeh/run_eval.py`

**SHA256**: 7d224279306cb9098e17f1fd2e7534b3695f0156c3f9724df1b7abd732d16bb2

```python
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

```

---

## Wiring Configuration

→ Create file at `.aeh/wiring.json`

**SHA256**: 5e21a42e1ebc0dcfc552b440dfc991e317210f415868d9eb1ad6d2ef042b2b3e

```json
{
  "aeh_db_path": "tmp/aeh.db",
  "agent_owned_files": [
    "target_init.py",
    "retrieval_stub.py",
    "run_config.json"
  ],
  "batches": [
    [
      "guard_rule",
      "guard_llm",
      "planner",
      "worker"
    ],
    [
      "case_law_search_tool",
      "decoy_tool",
      "judge",
      "writer"
    ]
  ],
  "component_ids": [
    "case_law_search_tool",
    "decoy_tool",
    "guard_llm",
    "guard_rule",
    "judge",
    "planner",
    "worker",
    "writer"
  ],
  "dataset_ids": [
    "ds_case_law_search_tool",
    "ds_decoy_tool",
    "ds_guard_llm",
    "ds_guard_rule",
    "ds_judge",
    "ds_planner",
    "ds_worker",
    "ds_writer"
  ],
  "plan_id": "sess-multi_agent"
}
```

---

## Eval API Route

→ Create file at `.aeh/aeh_eval.py`

**SHA256**: 5983cebe1173b01e4e25d796371457efe4df445d64178960ff37e47e94e52b8b

```python
"""AEH Stage 4 eval route: loads .aeh/run_eval.py by path (".aeh" isn't a valid dotted package segment) and calls its driver in-process; makes no LLM/network calls of its own."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter()

_RUN_EVAL_PATH = Path(__file__).resolve().parent.parent / ".aeh" / "run_eval.py"


def _load_run_eval():
    spec = importlib.util.spec_from_file_location("aeh_run_eval", _RUN_EVAL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {_RUN_EVAL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@router.post("/run-eval")
async def run_eval(verify: bool = False) -> dict:
    try:
        run_eval_module = _load_run_eval()
        return await run_eval_module.run(verify=verify)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

```

---

## Per-Agent Dispatch Modules

### `.aeh/dispatch/guard_rule.py`  (agent: guard_rule)

**SHA256**: `b05c3d6d9552eba85a7afbf7ab1c185a9a7a8d398f5cfd77a0d8bfc8414eb40f`

```python
"""Dispatch module for agent: guard_rule

Invocation mode: UNSUPPORTED — no invocation contract could be harvested.
Every case is skipped and recorded as skipped_unsupported. Do NOT invent an
invocation here; report it back instead so the plan can be regenerated.
"""
from typing import Any


async def invoke_agent(case: dict[str, Any], config: dict[str, Any]) -> Any:
    """Not invocable — skip_case() returns True so this is never called."""
    raise NotImplementedError("guard_rule has no supported invocation contract")


def skip_case(case: dict[str, Any]) -> bool:
    """This agent cannot be invoked — skip all cases."""
    return True
```

### `.aeh/dispatch/guard_llm.py`  (agent: guard_llm)

**SHA256**: `1e57caffe8a90bba8994f5a5ae2092575fd4f05ed018c8ac8d1c990ee374f2af`

```python
"""Dispatch module for agent: guard_llm

Invocation mode: UNSUPPORTED — no invocation contract could be harvested.
Every case is skipped and recorded as skipped_unsupported. Do NOT invent an
invocation here; report it back instead so the plan can be regenerated.
"""
from typing import Any


async def invoke_agent(case: dict[str, Any], config: dict[str, Any]) -> Any:
    """Not invocable — skip_case() returns True so this is never called."""
    raise NotImplementedError("guard_llm has no supported invocation contract")


def skip_case(case: dict[str, Any]) -> bool:
    """This agent cannot be invoked — skip all cases."""
    return True
```

### `.aeh/dispatch/planner.py`  (agent: planner)

**SHA256**: `b97f6d5ae1b4b97fee243cb2c8c74d49ecb2e98ff0d77fedb8335a5eb8a0c352`

```python
"""Dispatch module for agent: planner

Invocation mode: UNSUPPORTED — no invocation contract could be harvested.
Every case is skipped and recorded as skipped_unsupported. Do NOT invent an
invocation here; report it back instead so the plan can be regenerated.
"""
from typing import Any


async def invoke_agent(case: dict[str, Any], config: dict[str, Any]) -> Any:
    """Not invocable — skip_case() returns True so this is never called."""
    raise NotImplementedError("planner has no supported invocation contract")


def skip_case(case: dict[str, Any]) -> bool:
    """This agent cannot be invoked — skip all cases."""
    return True
```

### `.aeh/dispatch/worker.py`  (agent: worker)

**SHA256**: `4b0aa96eb2edf189f6c29d669098583cb092b7893550dc3ca1ad9500aef0b68d`

```python
"""Dispatch module for agent: worker

Invocation mode: UNSUPPORTED — no invocation contract could be harvested.
Every case is skipped and recorded as skipped_unsupported. Do NOT invent an
invocation here; report it back instead so the plan can be regenerated.
"""
from typing import Any


async def invoke_agent(case: dict[str, Any], config: dict[str, Any]) -> Any:
    """Not invocable — skip_case() returns True so this is never called."""
    raise NotImplementedError("worker has no supported invocation contract")


def skip_case(case: dict[str, Any]) -> bool:
    """This agent cannot be invoked — skip all cases."""
    return True
```

### `.aeh/dispatch/case_law_search_tool.py`  (agent: case_law_search_tool)

**SHA256**: `4057747350ea30e263ee0394bc5357fde7f51dc3c86ab682cc79f32fa0274c8e`

```python
"""Dispatch module for agent: case_law_search_tool

Invocation mode: UNSUPPORTED — no invocation contract could be harvested.
Every case is skipped and recorded as skipped_unsupported. Do NOT invent an
invocation here; report it back instead so the plan can be regenerated.
"""
from typing import Any


async def invoke_agent(case: dict[str, Any], config: dict[str, Any]) -> Any:
    """Not invocable — skip_case() returns True so this is never called."""
    raise NotImplementedError("case_law_search_tool has no supported invocation contract")


def skip_case(case: dict[str, Any]) -> bool:
    """This agent cannot be invoked — skip all cases."""
    return True
```

### `.aeh/dispatch/decoy_tool.py`  (agent: decoy_tool)

**SHA256**: `ff8994193397ac75c6df998493374241f429ee5042145bd968596ef51addbda1`

```python
"""Dispatch module for agent: decoy_tool

Invocation mode: UNSUPPORTED — no invocation contract could be harvested.
Every case is skipped and recorded as skipped_unsupported. Do NOT invent an
invocation here; report it back instead so the plan can be regenerated.
"""
from typing import Any


async def invoke_agent(case: dict[str, Any], config: dict[str, Any]) -> Any:
    """Not invocable — skip_case() returns True so this is never called."""
    raise NotImplementedError("decoy_tool has no supported invocation contract")


def skip_case(case: dict[str, Any]) -> bool:
    """This agent cannot be invoked — skip all cases."""
    return True
```

### `.aeh/dispatch/judge.py`  (agent: judge)

**SHA256**: `8cc9ba3a859ee66d0a843421f49c6a16f0d2326a12b61e75f044992a2047fb27`

```python
"""Dispatch module for agent: judge

Invocation mode: UNSUPPORTED — no invocation contract could be harvested.
Every case is skipped and recorded as skipped_unsupported. Do NOT invent an
invocation here; report it back instead so the plan can be regenerated.
"""
from typing import Any


async def invoke_agent(case: dict[str, Any], config: dict[str, Any]) -> Any:
    """Not invocable — skip_case() returns True so this is never called."""
    raise NotImplementedError("judge has no supported invocation contract")


def skip_case(case: dict[str, Any]) -> bool:
    """This agent cannot be invoked — skip all cases."""
    return True
```

### `.aeh/dispatch/writer.py`  (agent: writer)

**SHA256**: `86a2097f117ec4ff3ab8d1ba7fcf8f5632516fd5b7bd3c27e1597b2409a790a5`

```python
"""Dispatch module for agent: writer

Invocation mode: UNSUPPORTED — no invocation contract could be harvested.
Every case is skipped and recorded as skipped_unsupported. Do NOT invent an
invocation here; report it back instead so the plan can be regenerated.
"""
from typing import Any


async def invoke_agent(case: dict[str, Any], config: dict[str, Any]) -> Any:
    """Not invocable — skip_case() returns True so this is never called."""
    raise NotImplementedError("writer has no supported invocation contract")


def skip_case(case: dict[str, Any]) -> bool:
    """This agent cannot be invoked — skip all cases."""
    return True
```

---

## Server Entrypoint Edits

→ Target file: [NEEDS CLARIFICATION] — locate it in the target repo (see TASKS.md M1)

Insert three marker-wrapped blocks into the server entrypoint file. Find each
position yourself, then record the file and line you used in TASKS.md § RECON.
Markers make the edits idempotent and removable — keep them exactly as written.

**Block 1 — enable tracing** · position: the very top of the file, **before any
framework or application import** (the tracer must be enabled before the framework
is first imported, or no spans are captured).

```python
# aeh:begin aeh_tracer
import os
os.environ.setdefault("HAYSTACK_CONTENT_TRACING_ENABLED", "true")
# aeh:end aeh_tracer
```

**Block 2 — import the eval router** · position: beside the file's other router /
blueprint imports. `.aeh` is not an importable package name, so either move
`aeh_eval.py` next to the app's other route modules, or keep it in `.aeh/` and put
that directory on `sys.path` first. Record which you chose in § RECON.

```python
# aeh:begin aeh_eval_import
from aeh_eval import router as aeh_eval_router
# aeh:end aeh_eval_import
```

**Block 3 — register the eval router** · position: beside the other route
registrations, using the same registration call and indentation the file already uses.

```python
# aeh:begin aeh_eval_route
app.include_router(aeh_eval_router, prefix="/aeh")
# aeh:end aeh_eval_route
```

> If the entrypoint does not use this framework's API, adapt the *registration call*
> to whatever it does use — but keep the markers and the route path `/aeh`.
> If you cannot find a registration site at all, stop and report it as a [Conflict].

---

## Target Setup

→ Create file at `.aeh/target_init.py`, **then implement its marked region**

`run_eval.py` calls into this once before the first case. It is deliberately a separate,
**not** sha-locked file so a newer plan can ship a newer `run_eval.py` without erasing your
implementation.

```python
"""Process-level setup the target needs before any of its agents can run.

This file is yours: run_eval.py calls into it and never overwrites it, so a newer plan can
ship a newer run_eval.py without erasing what you write here.
"""
from typing import Any


def init_target(config: dict[str, Any]) -> Any:
    """Prepare this process so the target's agents can run. May be sync or async."""
    # === IMPLEMENT THIS ===
    #   Call whatever the target requires once per process before its agents work —
    #   typically a database/connection initialiser, plus any config or registry bootstrap.
    #   Copy what the target's own server entrypoint does at startup; services that look
    #   stateless still resolve a shared connection on first use and raise if it was never
    #   opened.
    #   Point it at the same data directory the harness uses: derive paths from
    #   `.aeh/wiring.json` rather than assuming a location, since a target often ships a
    #   second, empty database inside its source tree.
    #   Leave this returning None only if you have confirmed the target needs no setup.
    # === END IMPLEMENT THIS ===
    return None


def teardown_target(config: dict[str, Any]) -> Any:
    """Release what init_target opened. May be sync or async; run_eval.py awaits either."""
    # === IMPLEMENT THIS ===
    #   Mirror init_target: whatever the target's entrypoint closes on shutdown, close here.
    #   This is not optional tidiness. A driver that opened a connection pool and never closed
    #   it can hang after the last case finishes: a non-daemon worker thread keeps the process
    #   alive, and CPython joins those threads BEFORE running atexit handlers, so an atexit
    #   teardown cannot rescue it and `Thread.daemon` cannot be set after start(). The symptom
    #   reads as "the eval is very slow" rather than as a leak.
    #   Leave this as None only if init_target opened nothing.
    # === END IMPLEMENT THIS ===
    return None

```

---

## Retrieval Stub

→ Create file at `.aeh/retrieval_stub.py`, **then implement its marked region**

Every agent that retrieves internally imports `make_retrieval_stub` from here and feeds it the
case's own evidence, so the agent reads the case instead of the real repository. Only the
plumbing is generated: the method shape belongs to the target's retrieval dependency, so you
give it that shape. This file is deliberately **not** sha-locked.

(not applicable for this target)
