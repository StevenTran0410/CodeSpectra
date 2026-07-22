"""Retrieval service for self_retrieve fixture."""
from __future__ import annotations

from test_targets.self_retrieve.retrieval_types import Evidence, EvidenceBundle


class SelfRetriever:
    """Simple retriever service that returns mock evidence."""

    def __init__(self) -> None:
        self._corpus = [
            Evidence(content="First document", source="doc1.txt", relevance_score=0.9),
            Evidence(content="Second document", source="doc2.txt", relevance_score=0.8),
            Evidence(content="Third document", source="doc3.txt", relevance_score=0.7),
        ]

    async def search(self, query: str) -> EvidenceBundle:
        """Search and return evidence bundle."""
        return EvidenceBundle(evidences=self._corpus, total_count=len(self._corpus))
