"""T2 worker tools — one relevant, one decoy."""
from __future__ import annotations


async def case_law_search(query: str) -> str:
    """The relevant tool."""
    return f"Relevant precedent found regarding: {query}"


async def decoy_lookup(query: str) -> str:
    """The decoy tool — DEFECT_WRONG_TOOL tries this one first."""
    return f"Unrelated internal FAQ entry about: {query}"
