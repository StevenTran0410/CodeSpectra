"""Perturbation module: deterministic, shared degradation transforms on report sections.

Every function is field-name-parameterized (target_field/target_fields come from the caller,
e.g. a harvested contract or a metamorphic_relations.yaml entry) — no target-specific literal
field names are special-cased here."""
from __future__ import annotations

import copy


def drop_section(all_sections: dict, target_field: str) -> dict:
    """Returns a copy of all_sections with the specified field removed."""
    # Pure function: avoid mutating caller dict
    copied = copy.deepcopy(all_sections)
    if target_field in copied:
        del copied[target_field]
    return copied


def degrade_section(all_sections: dict, target_field: str) -> dict:
    """Returns a copy of all_sections with target_field replaced by its own shape-valid,
    typed-empty stub. This isinstance fallback is the entire degrade mechanism — no field-name
    special-casing. A non-dict field degrades to its own typed empty (list->[], str->"",
    dict->{}, else None), never to a stub {} regardless of its real shape."""
    copied = copy.deepcopy(all_sections)
    if target_field not in copied:
        return copied

    val = copied[target_field]
    if not isinstance(val, dict):
        if isinstance(val, list):
            copied[target_field] = []
        elif isinstance(val, str):
            copied[target_field] = ""
        else:
            copied[target_field] = None
        return copied

    degraded = {}
    for k, v in val.items():
        if isinstance(v, list):
            degraded[k] = []
        elif isinstance(v, dict):
            degraded[k] = {}
        elif isinstance(v, str):
            degraded[k] = ""
        else:
            degraded[k] = None

    copied[target_field] = degraded
    return copied


def inject_conflict(all_sections: dict, target_fields: list[str], token: str) -> dict:
    """Returns a copy of all_sections with `token` appended into each of target_fields
    (fields are caller-supplied data, never guessed by name)."""
    copied = copy.deepcopy(all_sections)
    for field in target_fields:
        current = copied.get(field)
        if isinstance(current, str):
            copied[field] = (current + " " + token).strip()
        else:
            copied[field] = token
    return copied


def oversize_section(all_sections: dict, target_field: str, target_len: int = 5000) -> dict:
    """Returns a copy of all_sections with a text field in target_field padded to exceed limits."""
    copied = copy.deepcopy(all_sections)
    if target_field not in copied or not isinstance(copied[target_field], dict):
        return copied

    sec = copied[target_field]
    # Generic English field-name heuristics for "the best text field to pad" — not target literals.
    preferred_keys = ["purpose", "description", "summary", "business_context", "rationale", "content"]

    padded = False
    for k in preferred_keys:
        if k in sec and isinstance(sec[k], str):
            curr_len = len(sec[k])
            if curr_len < target_len:
                sec[k] = sec[k] + " " + ("X" * (target_len - curr_len - 1))
            padded = True
            break

    if not padded:
        for k, v in sec.items():
            if isinstance(v, str) and k not in ("confidence", "repo_name"):
                curr_len = len(v)
                if curr_len < target_len:
                    sec[k] = v + " " + ("X" * (target_len - curr_len - 1))
                padded = True
                break

    if not padded:
        sec["oversized_content"] = "X" * target_len

    return copied
