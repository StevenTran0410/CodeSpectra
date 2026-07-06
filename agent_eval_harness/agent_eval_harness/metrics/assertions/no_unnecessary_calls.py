"""Assertion: no_unnecessary_calls — flags tool_call spans whose result was never used."""
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

    flagged: list[dict] = []

    for i, span in enumerate(spans):
        if span.get("span_type") != "tool_call":
            continue

        result = _extract_result_string(span.get("output_json"))
        if not result or not result.strip():
            # Can't check empty/missing result
            continue

        # Check if result string appears in any LATER span's input_json
        later_spans = spans[i + 1:]
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
