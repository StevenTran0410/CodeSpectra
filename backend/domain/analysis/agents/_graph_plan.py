"""Graph-aware retrieval planning helper (CS-103).

Adds a lightweight "plan step" before main retrieval in agents D, F, J:
one small LLM call decomposes a broad analysis goal into 1-3 focused
sub-queries so that `retrieve()` targets distinct aspects of the codebase
rather than a single monolithic keyword blob.
"""
from __future__ import annotations

import json
import re

from domain.model_connector.service import ProviderConfigService
from domain.model_connector.types import ChatMessage, ChatRequest
from domain.retrieval.types import RetrievalBundle, RetrievalMode, RetrievalSection, RetrieveRequest
from shared.logger import logger

_PLAN_SYSTEM = (
    "You decompose code analysis goals into focused retrieval queries. "
    "Return ONLY a JSON array of 1-3 short search strings (no explanation). "
    'Example: ["auth middleware", "JWT token validation", "session store"]'
)

_PLAN_USER_TMPL = (
    "Analysis goal: {goal}\n\n"
    "Produce 1-3 focused retrieval queries that together cover the necessary "
    "code context. Each query should target a distinct aspect. "
    "Output a JSON array of strings only."
)

_ARRAY_RE = re.compile(r"\[.*?\]", re.DOTALL)


async def plan_queries(
    goal: str,
    provider_service: ProviderConfigService,
    provider_id: str,
    model_id: str,
    fallback: list[str],
) -> list[str]:
    """Return 1-3 targeted sub-queries for *goal*.

    Falls back to *fallback* on any error so callers are never blocked.
    """
    try:
        res = await provider_service.chat(
            ChatRequest(
                provider_id=provider_id,
                model_id=model_id,
                messages=[
                    ChatMessage(role="system", content=_PLAN_SYSTEM),
                    ChatMessage(role="user", content=_PLAN_USER_TMPL.format(goal=goal[:500])),
                ],
                max_completion_tokens=120,
                temperature=0.0,
                json_mode=False,
                stream=False,
            )
        )
        text = (res.content or "").strip()
        try:
            arr = json.loads(text)
        except Exception:
            m = _ARRAY_RE.search(text)
            arr = json.loads(m.group(0)) if m else None
        if isinstance(arr, list) and arr:
            queries = [str(q).strip() for q in arr if str(q).strip()][:3]
            if queries:
                logger.debug("[graph_plan] goal=%r → %s", goal[:60], queries)
                return queries
    except Exception as exc:
        logger.debug("[graph_plan] plan step failed (%s), using fallback", exc)
    return fallback


def merge_bundles(bundles: list[RetrievalBundle], section: RetrievalSection) -> RetrievalBundle:
    """Merge multiple RetrievalBundles, deduplicating by chunk_id, sorted by score desc."""
    if not bundles:
        raise ValueError("no bundles to merge")
    seen: set[str] = set()
    merged_evidences = []
    total_used = 0
    total_budget = 0
    for b in bundles:
        total_budget += b.budget_tokens
        total_used += b.used_tokens
        for ev in b.evidences:
            if ev.chunk_id not in seen:
                seen.add(ev.chunk_id)
                merged_evidences.append(ev)
    merged_evidences.sort(key=lambda e: -e.score)
    return RetrievalBundle(
        snapshot_id=bundles[0].snapshot_id,
        mode=bundles[0].mode,
        section=section,
        query=" | ".join(b.query for b in bundles),
        budget_tokens=total_budget,
        used_tokens=total_used,
        evidences=merged_evidences,
    )


async def retrieve_multi(
    retrieval_service,
    snapshot_id: str,
    queries: list[str],
    section: RetrievalSection,
    mode: RetrievalMode,
    max_results_each: int,
) -> RetrievalBundle:
    """Run `retrieve()` once per query, merge and deduplicate the results."""
    bundles: list[RetrievalBundle] = []
    for q in queries:
        try:
            b = await retrieval_service.retrieve(
                RetrieveRequest(
                    snapshot_id=snapshot_id,
                    query=q,
                    section=section,
                    mode=mode,
                    max_results=max_results_each,
                )
            )
            bundles.append(b)
        except Exception as exc:
            logger.warning("[graph_plan] retrieve failed for query=%r: %s", q[:60], exc)
    if not bundles:
        # Return empty bundle rather than raise so agents can still try LLM
        return RetrievalBundle(
            snapshot_id=snapshot_id,
            mode=mode,
            section=section,
            query=" | ".join(queries),
            budget_tokens=0,
            used_tokens=0,
            evidences=[],
        )
    return merge_bundles(bundles, section)
