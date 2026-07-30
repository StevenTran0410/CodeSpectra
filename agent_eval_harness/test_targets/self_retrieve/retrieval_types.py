"""Retrieval data types for self_retrieve fixture."""
from __future__ import annotations

from pydantic import BaseModel


class Evidence(BaseModel):
    """A single piece of evidence."""
    content: str
    source: str
    relevance_score: float = 1.0


class EvidenceBundle(BaseModel):
    """Collection of retrieved evidence."""
    evidences: list[Evidence]
    total_count: int = 0
