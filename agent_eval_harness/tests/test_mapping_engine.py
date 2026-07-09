"""Unit tests for the span -> component mapping engine (pure, no I/O)."""
from __future__ import annotations

from pathlib import Path

from agent_eval_harness.instrumentation.base import CapturedSpan
from agent_eval_harness.mapping.engine import map_spans_to_components
from agent_eval_harness.mapping.system_map import (
    Component,
    SpanMatchBlock,
    SystemMap,
    load_system_map,
)

_T1_MAP_PATH = Path(__file__).parent.parent / "test_targets" / "linear_rag" / "system_map.yaml"
_RENAMED_MAP_PATH = Path(__file__).parent / "fixtures" / "linear_rag_system_map_renamed.yaml"


def _span(span_id: str, component_name: str, tags: dict | None = None) -> CapturedSpan:
    return CapturedSpan(
        span_id=span_id,
        parent_span_id=None,
        operation_name="haystack.component.run",
        span_type="agent",
        component_name=component_name,
        tags=tags or {},
    )


def test_exact_match_by_component_name() -> None:
    system_map = load_system_map(_T1_MAP_PATH)
    spans = [_span("s1", "retriever"), _span("s2", "writer")]

    result = map_spans_to_components(spans, system_map)

    assert result.unmatched == []
    assert result.ambiguous == []
    assigned = {span.span_id: cid for span, cid in result.matched}
    assert assigned == {"s1": "retriever", "s2": "writer"}


def test_unmatched_span_kept_and_counted_not_dropped() -> None:
    system_map = load_system_map(_T1_MAP_PATH)
    span = _span("s1", "mystery_component")

    result = map_spans_to_components([span], system_map)

    assert len(result.unmatched) == 1
    assert result.unmatched[0] is span
    assert span.component_id is None


def test_ambiguous_match_assigns_first_and_records_all_candidates() -> None:
    system_map = SystemMap(
        target_system_id="synthetic",
        components=[
            Component(
                id="a", role="agent", entry_point="x:y",
                span_match=[SpanMatchBlock(component_name="dup")],
            ),
            Component(
                id="b", role="agent", entry_point="x:y",
                span_match=[SpanMatchBlock(component_name="dup")],
            ),
        ],
    )
    span = _span("s1", "dup")

    result = map_spans_to_components([span], system_map)

    assert span.component_id == "a"
    assert len(result.ambiguous) == 1
    assert result.ambiguous[0][1] == ["a", "b"]


def test_rename_component_id_changes_mapping_with_zero_code_edits() -> None:
    """Renaming a component id in the system map changes the mapping outcome with
    zero code edits on the span-emitting side."""
    original_map = load_system_map(_T1_MAP_PATH)
    renamed_map = load_system_map(_RENAMED_MAP_PATH)

    span_against_original = _span("s1", "writer")
    span_against_renamed = _span("s1", "writer")

    map_spans_to_components([span_against_original], original_map)
    map_spans_to_components([span_against_renamed], renamed_map)

    assert span_against_original.component_id == "writer"
    assert span_against_renamed.component_id == "responder"


def test_tags_based_match() -> None:
    system_map = SystemMap(
        target_system_id="synthetic",
        components=[
            Component(
                id="tool_x", role="tool", entry_point="x:y",
                span_match=[SpanMatchBlock(tags={"aeh.tool.name": "search"})],
            ),
        ],
    )
    span = _span("s1", "worker", tags={"aeh.tool.name": "search"})

    result = map_spans_to_components([span], system_map)

    assert span.component_id == "tool_x"
    assert result.unmatched == []
