"""Shared section compression logic for AuditAgent (K) and SynthesisAgent (L).

When adding a new SectionX to schemas.py, add the corresponding entry to
_SECTION_PREVIEW_KEYS here.
"""

from __future__ import annotations

import json
from typing import Any

_SECTION_PREVIEW_KEYS: dict[str, list[str]] = {
    "A": ["purpose", "domain", "tech_stack", "runtime_type"],
    "B": ["main_layers", "frameworks", "entrypoints"],
    "C": ["summary", "folders"],
    "D": ["naming_style", "async_style", "error_handling"],
    "E": ["rules", "violations_found"],
    "F": ["features"],
    "G": ["entrypoint", "backbone", "read_first"],
    "H": ["steps"],
    "I": ["terms"],
    "J": ["summary", "findings"],
}


def compress_section(letter: str, section: dict, char_cap: int = 500) -> str:
    """Serialize only preview keys for a section, truncated to char_cap."""
    keys = _SECTION_PREVIEW_KEYS.get(letter, [])
    preview: dict[str, Any] = {}
    for key in keys:
        val = section.get(key)
        if val is not None:
            preview[key] = val
    serialized = json.dumps(preview, ensure_ascii=False)
    return serialized[:char_cap]


def compress_audit(section_k: dict, char_cap: int = 800) -> str:
    """Extract overall_confidence, section_scores, weakest_sections, notes from K."""
    subset = {
        "overall_confidence": section_k.get("overall_confidence"),
        "section_scores": section_k.get("section_scores"),
        "weakest_sections": section_k.get("weakest_sections"),
        "notes": section_k.get("notes"),
    }
    serialized = json.dumps(subset, ensure_ascii=False)
    return serialized[:char_cap]
