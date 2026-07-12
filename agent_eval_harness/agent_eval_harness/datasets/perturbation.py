"""Perturbation module: deterministic, shared degradation transforms on report sections."""
from __future__ import annotations

import copy


def drop_section(all_sections: dict, letter: str) -> dict:
    """Returns a copy of all_sections with the specified section key removed."""
    # Pure function: avoid mutating caller dict
    copied = copy.deepcopy(all_sections)
    if letter in copied:
        del copied[letter]
    return copied


def degrade_section(all_sections: dict, letter: str, *, empty_confidence: bool = True) -> dict:
    """Returns a copy of all_sections with the specified section replaced by a shape-valid empty stub."""
    copied = copy.deepcopy(all_sections)
    if letter not in copied:
        return copied

    sec = copied[letter]
    if not isinstance(sec, dict):
        copied[letter] = {}
        return copied

    degraded = {}
    for k, v in sec.items():
        # Keep repo_name context for tracing
        if k == "repo_name":
            degraded[k] = v
        elif k == "confidence":
            degraded[k] = "low" if empty_confidence else v
        elif k == "blind_spots":
            degraded[k] = ["Agent failed: Degraded"]
        elif k in ("domain", "runtime_type"):
            degraded[k] = "unknown"
        elif isinstance(v, list):
            degraded[k] = []
        elif isinstance(v, dict):
            degraded[k] = {}
        elif isinstance(v, str):
            degraded[k] = ""
        else:
            degraded[k] = None

    if empty_confidence and "confidence" not in degraded:
        degraded["confidence"] = "low"
    if "blind_spots" not in degraded:
        degraded["blind_spots"] = ["Agent failed: Degraded"]

    copied[letter] = degraded
    return copied


def inject_conflict(
    all_sections: dict, letter_a: str, letter_b: str, claim_a: str, claim_b: str
) -> dict:
    """Returns a copy of all_sections with conflicting claims injected into the two letters."""
    copied = copy.deepcopy(all_sections)

    # Heuristic to find the best text field to inject the claim
    def _inject(sec: dict, claim: str) -> dict:
        target_keys = ["purpose", "description", "summary", "business_context", "rationale", "content"]
        for k in target_keys:
            if k in sec and isinstance(sec[k], str):
                sec[k] = (sec[k] + " " + claim).strip()
                return sec
        for k, v in sec.items():
            if isinstance(v, str) and k != "confidence" and k != "repo_name":
                sec[k] = (v + " " + claim).strip()
                return sec
        sec["injected_conflict"] = claim
        return sec

    if letter_a in copied and isinstance(copied[letter_a], dict):
        copied[letter_a] = _inject(copied[letter_a], claim_a)
    if letter_b in copied and isinstance(copied[letter_b], dict):
        copied[letter_b] = _inject(copied[letter_b], claim_b)

    return copied


def oversize_section(all_sections: dict, letter: str, target_len: int = 5000) -> dict:
    """Returns a copy of all_sections with a text field in the section padded to exceed limits."""
    copied = copy.deepcopy(all_sections)
    if letter not in copied or not isinstance(copied[letter], dict):
        return copied

    sec = copied[letter]
    target_keys = ["purpose", "description", "summary", "business_context", "rationale", "content"]

    # Heuristic to find the first large text field to pad
    padded = False
    for k in target_keys:
        if k in sec and isinstance(sec[k], str):
            curr_len = len(sec[k])
            if curr_len < target_len:
                sec[k] = sec[k] + " " + ("X" * (target_len - curr_len - 1))
            padded = True
            break

    if not padded:
        for k, v in sec.items():
            if isinstance(v, str) and k != "confidence" and k != "repo_name":
                curr_len = len(v)
                if curr_len < target_len:
                    sec[k] = v + " " + ("X" * (target_len - curr_len - 1))
                padded = True
                break

    if not padded:
        sec["oversized_content"] = "X" * target_len

    return copied
