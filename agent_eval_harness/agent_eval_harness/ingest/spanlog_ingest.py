"""Ingests an injected target's spanlog JSONL into AEH's own store: redacts credential-shaped strings, refuses an unrecognized schema, and marks a run 'partial' rather than 'completed' whenever any case failed or produced zero spans, so coverage is never reported fake-green."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_eval_harness.instrumentation.base import CapturedSpan
from agent_eval_harness.store import repository

SUPPORTED_SCHEMA = "aeh.spanlog/1"

_REDACT_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{8,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE),
    re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),  # JWT
]


class IngestError(Exception):
    pass


def _redact(text: str | None) -> str | None:
    if text is None:
        return None
    redacted = text
    for pattern in _REDACT_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _blank_case(dataset_case_id: str | None = None) -> dict[str, Any]:
    return {
        "dataset_case_id": dataset_case_id,
        "root_input": "",
        "final_output": None,
        "status": "unknown",
        "spans": [],
    }


@dataclass
class ParsedSpanlog:
    header: dict[str, Any]
    cases: dict[str, dict[str, Any]] = field(default_factory=dict)
    run_summary: dict[str, Any] | None = None


def parse_spanlog(log_path: Path) -> ParsedSpanlog:
    header: dict[str, Any] | None = None
    cases: dict[str, dict[str, Any]] = {}
    run_summary: dict[str, Any] | None = None

    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        kind = record.get("record")
        trace_id = record.get("trace_id")

        if kind == "header":
            header = record
        elif kind == "case_start" and trace_id is not None:
            cases[trace_id] = _blank_case(record.get("dataset_case_id"))
            cases[trace_id]["root_input"] = json.dumps(record.get("input"), ensure_ascii=False, default=str)
        elif kind == "case_end" and trace_id is not None:
            case = cases.setdefault(trace_id, _blank_case())
            case["final_output"] = record.get("final_output_json")
            case["status"] = record.get("status", "unknown")
        elif kind == "span" and trace_id is not None:
            cases.setdefault(trace_id, _blank_case())["spans"].append(record)
        elif kind == "run_summary":
            run_summary = record

    if header is None:
        raise IngestError("no header record found — not a valid AEH spanlog")
    if header.get("schema") != SUPPORTED_SCHEMA:
        raise IngestError(
            f"unsupported spanlog schema {header.get('schema')!r} — expected {SUPPORTED_SCHEMA!r}"
        )

    return ParsedSpanlog(header=header, cases=cases, run_summary=run_summary)


def referential_dry_run(parsed: ParsedSpanlog, known_component_ids: set[str]) -> tuple[int, int]:
    """(total_spans, unmatched_count) across every case — the caller decides whether an
    unmatched% this high should block scoring; ingest itself always persists what came in."""
    total = 0
    unmatched = 0
    for case in parsed.cases.values():
        for span in case["spans"]:
            total += 1
            if span.get("component_id") not in known_component_ids:
                unmatched += 1
    return total, unmatched


def _span_record_to_captured(record: dict[str, Any]) -> CapturedSpan:
    return CapturedSpan(
        span_id=record["span_id"],
        parent_span_id=record.get("parent_span_id"),
        operation_name=record.get("operation", ""),
        span_type=record.get("span_type") or "agent",
        component_id=record.get("component_id"),
        input_json=_redact(record.get("input_json")),
        output_json=_redact(record.get("output_json")),
        model=record.get("model"),
        tokens_in=record.get("tokens_in"),
        tokens_out=record.get("tokens_out"),
        token_source=record.get("token_source"),
        started_at=record.get("started_at", ""),
        latency_ms=record.get("latency_ms"),
        tier="tier1",
    )


async def persist_spanlog(
    parsed: ParsedSpanlog,
    *,
    target_system_id: str,
    eval_plan_id: str | None = None,
    map_path: str | None = None,
) -> str:
    run_id = await repository.insert_run(
        target_system_id=target_system_id,
        eval_plan_id=eval_plan_id,
        map_path=map_path,
        source="ingested",
    )

    any_partial = not parsed.cases
    for case in parsed.cases.values():
        if case["status"] != "ok" or not case["spans"]:
            any_partial = True

        trace_id = await repository.insert_trace(
            run_id,
            root_input=_redact(case["root_input"]) or "",
            dataset_case_id=case.get("dataset_case_id"),
        )
        captured = [_span_record_to_captured(s) for s in case["spans"]]
        if captured:
            await repository.insert_spans_bulk(trace_id, captured)

        total_tokens = sum((s.tokens_in or 0) + (s.tokens_out or 0) for s in captured)
        total_latency_ms = sum(s.latency_ms or 0 for s in captured)
        await repository.finalize_trace(
            trace_id,
            final_output=_redact(case.get("final_output")) or "",
            total_tokens=total_tokens,
            total_latency_ms=total_latency_ms,
        )

    await repository.finish_run(run_id, status="partial" if any_partial else "completed")
    return run_id
