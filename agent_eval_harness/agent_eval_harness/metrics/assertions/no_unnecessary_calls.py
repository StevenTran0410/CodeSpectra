"""Assertion: no_unnecessary_calls — flags tool_call spans whose result was never used.

For every `tool_call` span in the trace, extracts the result string from
`output_json` and checks whether that string appears as a substring in any LATER
span's `input_json` within the same trace.

**Checks ALL tool_call spans, not just ones tagged with `component_id`**: the
mapping engine assigns each tool_call span's `component_id` to the SPECIFIC tool
component it matched (e.g. "case_law_search_tool", "decoy_tool" — see
test_targets/multi_agent/system_map.yaml's per-tool span_match rules), never to
the calling agent ("worker"). Filtering by `component_id == "worker"` would
therefore skip every tool_call span and always report zero flags, regardless of
target behavior. `component_id` here only labels which component this check's
*result* is reported under (the caller/orchestrating agent) — it does not
restrict which spans are inspected.

**Known limitation (documented per ticket instruction)**: this heuristic will
false-negative if the target paraphrases or reformats tool output before passing
it to downstream spans. A more robust approach (embedding similarity or semantic
diffing) is out of scope for Phase 0 — use this as a signal, not a hard gate in
production.

`DEFECT_WRONG_TOOL` causes the worker to call `decoy_lookup` before
`case_law_search`. In T2's implementation both tools are always called
(unconditionally) and only case_law_search's result is ever reused downstream —
so decoy_lookup's result is flagged as unused in EVERY run, defect or not. This
assertion is therefore not itself the defect/no-defect differentiator for
DEFECT_WRONG_TOOL (span tool-call ORDER is — see test_e2e_multi_agent.py's
existing coverage); it still correctly and consistently reports the one
genuinely-unused call in both cases, which is exercised in the gauntlet test.
"""
from __future__ import annotations

import json

from agent_eval_harness.metrics.assertions.registry import register
from agent_eval_harness.metrics.types import MetricResult


def _extract_result_string(output_json: str | None) -> str | None:
    """Best-effort extraction of a plain-text result from a tool output JSON."""
    if not output_json:
        return None
    try:
        data = json.loads(output_json)
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            for key in ("result", "content", "text", "output", "answer"):
                val = data.get(key)
                if isinstance(val, str) and val.strip():
                    return val
        return json.dumps(data)
    except json.JSONDecodeError:
        return output_json


@register("no_unnecessary_calls")
def no_unnecessary_calls(spans: list[dict], component_id: str, params: dict) -> MetricResult:
    # spans is assumed to already be in true execution order (see
    # retry_on_reject_required's docstring for why we don't re-sort by
    # started_at — fast synchronous runs can tie on timestamp resolution).
    sorted_spans = spans

    flagged: list[dict] = []

    for i, span in enumerate(sorted_spans):
        if span.get("span_type") != "tool_call":
            continue

        result = _extract_result_string(span.get("output_json"))
        if not result or not result.strip():
            # Can't check empty/missing result
            continue

        # Check if result string appears in any LATER span's input_json
        later_spans = sorted_spans[i + 1:]
        result_used = any(
            result in (s.get("input_json") or "") for s in later_spans
        )
        if not result_used:
            tool_name = None
            try:
                details = json.loads(span.get("details_json") or "{}")
                tool_name = details.get("raw_tags", {}).get("aeh.tool.name")
            except Exception:  # noqa: BLE001
                pass
            flagged.append(
                {
                    "span_id": span["id"],
                    "tool_name": tool_name,
                    "result_snippet": result[:120],
                }
            )

    passed = len(flagged) == 0
    return MetricResult(
        metric_name="assertion.no_unnecessary_calls",
        metric_class="assertion",
        score=None,
        passed=passed,
        details={
            "flagged_tool_calls": flagged,
            "note": (
                "Heuristic: result string must appear verbatim in a later span's input_json. "
                "False-negatives possible if the target paraphrases tool output before reuse."
            ),
        },
        component_id=component_id,
    )
