"""Heuristic model capability checks (size / tier) for analysis runs."""

from __future__ import annotations

import re

_SMALL_MODEL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(^|[^0-9])1\.5b([^0-9]|$)", re.IGNORECASE),
    re.compile(r"(^|[^0-9])1b([^0-9]|$)", re.IGNORECASE),
    re.compile(r"(^|[^0-9])2b([^0-9]|$)", re.IGNORECASE),
    re.compile(r"(^|[^0-9])3b([^0-9]|$)", re.IGNORECASE),
    re.compile(r":mini\b", re.IGNORECASE),
    re.compile(r":tiny\b", re.IGNORECASE),
    re.compile(r"tinyllama", re.IGNORECASE),
    re.compile(r"gemma[-_]?2b", re.IGNORECASE),
    re.compile(r"qwen[^:]*:1\.5b", re.IGNORECASE),
    re.compile(r"phi[-_]?3[:_-]?mini", re.IGNORECASE),
    re.compile(r"smollm", re.IGNORECASE),
]

_MSG = (
    "This model is very small (≈1–3B or “mini/tiny” tier). Analysis quality may be poor; "
    "prefer at least 7–8B for reliable JSON sections."
)


def check_model_capability(model_id: str) -> dict | None:
    mid = (model_id or "").strip().lower()
    if not mid:
        return None
    for pat in _SMALL_MODEL_PATTERNS:
        if pat.search(mid):
            return {
                "code": "model_too_small",
                "message": _MSG,
                "severity": "warn",
            }
    return None
