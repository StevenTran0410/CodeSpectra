"""Span -> component mapping engine. Entirely data-driven from the map's span_match"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from agent_eval_harness.instrumentation.base import CapturedSpan
from agent_eval_harness.mapping.system_map import Component, SpanMatchBlock, SystemMap


@dataclass
class MappingResult:
    matched: list[tuple[CapturedSpan, str]] = field(default_factory=list)
    unmatched: list[CapturedSpan] = field(default_factory=list)
    ambiguous: list[tuple[CapturedSpan, list[str]]] = field(default_factory=list)


def _block_matches(span: CapturedSpan, block: SpanMatchBlock) -> bool:
    """All given fields in one block must match (AND semantics)."""
    if block.component_name is not None and span.component_name != block.component_name:
        return False
    if block.component_type is not None and span.component_type != block.component_type:
        return False
    if block.span_name_pattern is not None:
        if not re.fullmatch(block.span_name_pattern, span.operation_name):
            return False
    for key, value in block.tags.items():
        if span.tags.get(key) != value:
            return False
    return True


def _component_matches(span: CapturedSpan, component: Component) -> bool:
    """OR across the component's span_match list."""
    return any(_block_matches(span, block) for block in component.span_match)


def map_spans_to_components(spans: list[CapturedSpan], system_map: SystemMap) -> MappingResult:
    """Components are evaluated in system_map.components (YAML list) order."""
    result = MappingResult()
    for span in spans:
        if span.component_id is not None:
            result.matched.append((span, span.component_id))
            continue

        candidates = [c.id for c in system_map.components if _component_matches(span, c)]
        if not candidates:
            result.unmatched.append(span)
            continue

        span.component_id = candidates[0]
        result.matched.append((span, candidates[0]))
        if len(candidates) > 1:
            result.ambiguous.append((span, candidates))
    return result
