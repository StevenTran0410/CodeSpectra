"""Shared deterministic scoring helpers used by both the trace-based assertion registry and the dict-based injection scorer."""
from __future__ import annotations

DYNAMIC_MARKER = "<dynamic>"


def matches_fallback(data: dict, fallback: dict) -> bool:
    """A field matches if byte-equal, or the fallback marked it '<dynamic>' and the field is merely present."""
    for key, expected in fallback.items():
        if key not in data:
            return False
        if expected == DYNAMIC_MARKER:
            continue
        if data[key] != expected:
            return False
    return True
