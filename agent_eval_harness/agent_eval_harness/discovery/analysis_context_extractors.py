"""Pure extractors for analysis report sections (stdlib-only, no async, no AEH infrastructure)."""
from __future__ import annotations

from typing import Any


def _extract_important_files(section: dict[str, Any] | None) -> list[str]:
    """Extract important file paths from section dict (e.g., section G).

    Flattens all list values from the section dict, returns [] if None.
    """
    if section is None:
        return []

    important_files = []
    for key, value in section.items():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    important_files.append(item)

    return important_files


def _filter_risk_findings(section: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Extract risk findings from section dict (e.g., section J).

    Keeps only entries where severity in ('high', 'medium'), preserves evidence key.
    Returns [] if None.
    """
    if section is None:
        return []

    findings = section.get("findings", [])
    if not isinstance(findings, list):
        return []

    filtered = []
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        severity = finding.get("severity", "").lower()
        if severity in ("high", "medium"):
            filtered.append(finding)

    return filtered


def _extract_identity(section: dict[str, Any] | None) -> dict[str, Any]:
    """Extract project identity from section dict (e.g., section A).

    Defensive .get() pass-through, returns {} if None.
    """
    if section is None:
        return {}

    result: dict[str, Any] = {}
    for key in ("purpose", "domain", "runtime_type", "tech_stack", "key_frameworks"):
        if key in section:
            result[key] = section[key]

    return result


def _extract_synthesis(section: dict[str, Any] | None) -> dict[str, Any]:
    """Extract synthesis from section dict (e.g., section L).

    Defensive .get() pass-through, returns {} if None.
    """
    if section is None:
        return {}

    result: dict[str, Any] = {}
    for key in ("executive_summary", "architecture_narrative", "reading_path"):
        if key in section:
            result[key] = section[key]

    return result
