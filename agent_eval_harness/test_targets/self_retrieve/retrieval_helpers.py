"""Free-function retrieval helper for multi-retrieve agent pattern."""
from __future__ import annotations

from test_targets.self_retrieve.retrieval_service import SelfRetriever
from test_targets.self_retrieve.retrieval_types import EvidenceBundle


async def retrieve_multi(
    retrieval_service: SelfRetriever,
    queries: list[str],
) -> EvidenceBundle:
    """Multi-query retrieval free function that takes service as argument."""
    all_evidences = []
    for query in queries:
        bundle = await retrieval_service.search(query)
        all_evidences.extend(bundle.evidences)
    return EvidenceBundle(evidences=all_evidences, total_count=len(all_evidences))
