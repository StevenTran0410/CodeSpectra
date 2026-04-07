"""Typed context assembly for convention / risk / cross-agent hints (RPA-055)."""
from __future__ import annotations

from typing import Any

from ..static_convention import ConventionReport
from ..static_risk import RiskReport


def build_convention_block(report: ConventionReport | None) -> str:
    if report is None or not report.signals:
        return ""
    return report.as_context_text()


def build_risk_block(report: RiskReport | None, categories: list[str] | None = None) -> str:
    cats = tuple(categories) if categories is not None else ("blast_radius", "anti_pattern")
    if report is None:
        return ""
    rows = [f for f in report.findings if f.category in cats]
    if not rows:
        return ""
    lines: list[str] = ["Blast radius and boundary violations (FACTS — do not contradict):"]
    for f in rows:
        lines.append(f"RISK [{f.category}] {f.title}: {f.rationale}")
    return "\n".join(lines)


def extract_d_hint_context(agent_d_output: dict[str, Any] | None) -> str:
    """D→E contract: Section D exposes `signals`, never `rules`."""
    if not agent_d_output:
        return ""
    raw = agent_d_output.get("signals", [])
    if not isinstance(raw, list):
        return ""
    out_lines: list[str] = []
    for item in raw[:5]:
        if not isinstance(item, dict):
            continue
        cat = str(item.get("category", "") or "")
        hint = str(
            item.get("pattern", "")
            or item.get("description", "")
            or item.get("evidence", "")
            or ""
        )[:90]
        if not hint and not cat:
            continue
        if cat:
            out_lines.append(f"{cat}: {hint}".strip())
        else:
            out_lines.append(hint)
    if not out_lines:
        return ""
    prefix = (
        "Team patterns already confirmed (use for negative-space inference — "
        "look for violations):\n"
    )
    return prefix + "\n".join(out_lines)
