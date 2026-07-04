"""Shared, tier-agnostic helpers: JSON-safe serialization + duck-typed
model/token extraction. Used by both tier1_haystack.py and tier2_boundary.py so
tier-2 spans get token data "only if the boundary happens to expose it, never
invented" (same extraction code, not tier-specific).
"""
from __future__ import annotations

import json
from typing import Any


def safe_json(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return json.dumps(value, default=str)
    except TypeError:
        return json.dumps(str(value))


def extract_model_and_tokens(
    payload: Any,
) -> tuple[str | None, int | None, int | None, str | None]:
    """Two duck-typed shapes recognized, opportunistically:
    1. Haystack-style: {"replies": [ChatMessage(meta={"model", "usage": {...}})]}
    2. Plain dict: {"model": ..., "tokens_in": ..., "tokens_out": ..., "token_source": ...}
    Anything else -> all None (never invented).
    """
    if not isinstance(payload, dict):
        return None, None, None, None

    replies = payload.get("replies")
    if isinstance(replies, list) and replies:
        meta = getattr(replies[0], "meta", None)
        if isinstance(meta, dict):
            model = meta.get("model")
            usage = meta.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            token_source = meta.get("aeh_token_source")
            if token_source is None and prompt_tokens is not None:
                token_source = "measured"
            if model is not None:
                return model, prompt_tokens, completion_tokens, token_source

    if "model" in payload or "tokens_in" in payload:
        return (
            payload.get("model"),
            payload.get("tokens_in"),
            payload.get("tokens_out"),
            payload.get("token_source"),
        )

    return None, None, None, None
