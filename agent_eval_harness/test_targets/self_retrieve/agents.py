"""Self-retrieve test agents for harvesting virtual inputs."""
from __future__ import annotations

from test_targets.self_retrieve.retrieval_types import EvidenceBundle
from test_targets.self_retrieve.retrieval_service import SelfRetriever
from test_targets.self_retrieve.retrieval_helpers import retrieve_multi


class AnnotationTierAgent:
    """Agent with typed retrieval dependency (annotation-tier)."""

    def __init__(self, retriever: SelfRetriever) -> None:
        self._retriever = retriever

    async def run(self, query: str) -> dict:
        """Run with typed retrieval."""
        evidence: EvidenceBundle = await self._retriever.search(query)
        return {"answer": f"Found {len(evidence.evidences)} pieces of evidence"}


class UsageTierAgent:
    """Agent with loosely-typed retrieval (usage-tier)."""

    def __init__(self, retriever: SelfRetriever) -> None:
        self._retriever = retriever

    async def run(self, query: str) -> dict:
        """Run with usage-tier evidence access."""
        result = await self._retriever.search(query)
        # Usage-tier: accessing fields via .get() pattern
        evidences = result.get("evidences", []) if hasattr(result, "get") else getattr(result, "evidences", [])
        first_content = evidences[0].content if evidences else "no evidence"
        return {"answer": first_content}


class FreeFunctionAgent:
    """Agent using free-function retrieval with typed return (free-function 1-hop pattern)."""

    def __init__(self, retrieval_service: SelfRetriever) -> None:
        self._retrieval = retrieval_service

    async def run(self, queries: list[str]) -> dict:
        """Run with free-function retrieval."""
        bundle: EvidenceBundle = await retrieve_multi(retrieval_service=self._retrieval, queries=queries)
        return {"answer": f"Found {len(bundle.evidences)} pieces of evidence"}
