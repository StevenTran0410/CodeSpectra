"""Span-tree stdout printer + unmatched-rate warnings."""
from __future__ import annotations

from agent_eval_harness.instrumentation.base import CapturedSpan, TraceResult
from agent_eval_harness.mapping.system_map import SystemMap

DEFAULT_UNMATCHED_WARN_THRESHOLD = 0.2


def _build_children_map(spans: list[CapturedSpan]) -> dict[str | None, list[CapturedSpan]]:
    children: dict[str | None, list[CapturedSpan]] = {}
    for span in spans:
        children.setdefault(span.parent_span_id, []).append(span)
    return children


def _format_span_line(span: CapturedSpan, system_map: SystemMap, depth: int) -> str:
    component = system_map.component_by_id(span.component_id) if span.component_id else None
    label = span.component_id or "UNMATCHED"
    role = component.role if component else "?"
    parts = [
        f"{'  ' * depth}[{label}]",
        f"role={role}",
        f"type={span.span_type}",
        f"{span.latency_ms}ms",
    ]
    if span.model:
        parts.append(f"model={span.model}")
    if span.tokens_in is not None or span.tokens_out is not None:
        parts.append(f"tok_in={span.tokens_in} tok_out={span.tokens_out}")
    return " ".join(parts)


def _walk(
    span: CapturedSpan,
    system_map: SystemMap,
    children: dict[str | None, list[CapturedSpan]],
    depth: int,
    lines: list[str],
) -> None:
    lines.append(_format_span_line(span, system_map, depth))
    for child in children.get(span.span_id, []):
        _walk(child, system_map, children, depth + 1, lines)


def format_trace(
    result: TraceResult,
    system_map: SystemMap,
    unmatched_warn_threshold: float = DEFAULT_UNMATCHED_WARN_THRESHOLD,
) -> str:
    lines: list[str] = []
    children = _build_children_map(result.spans)
    for root in children.get(None, []):
        _walk(root, system_map, children, 0, lines)

    total = len(result.spans)
    unmatched_count = len(result.unmatched)
    matched = total - unmatched_count
    pct = (matched / total * 100) if total else 100.0
    lines.append(f"mapped {matched}/{total} spans ({pct:.1f}%)")

    if unmatched_count:
        lines.append(
            f"WARNING: {unmatched_count} span(s) unmatched (component_id=NULL)"
            " — not silently dropped:"
        )
        for span in result.unmatched:
            lines.append(
                f"    span {span.span_id[:8]} op={span.operation_name}"
                f" component_name={span.component_name}"
            )
        if total and (unmatched_count / total) > unmatched_warn_threshold:
            rate_pct = unmatched_count / total * 100
            lines.append(
                f"HIGH UNMATCHED RATE: {rate_pct:.1f}% of spans have no component_id"
                " — check span_match rules in system_map.yaml"
            )

    for span, candidates in result.ambiguous:
        joined = ", ".join(candidates)
        lines.append(
            f"NOTE: span {span.span_id[:8]} matched {len(candidates)} components ({joined})"
            f" — used first match '{candidates[0]}'; tighten span_match to disambiguate."
        )

    lines.append(
        f"trace complete  total_tokens={result.total_tokens}"
        f"  total_latency_ms={result.total_latency_ms}"
    )
    return "\n".join(lines)
