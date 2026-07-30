"""Graph-aware retrieval planning helper.

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
from domain.retrieval.types import RetrievalBundle, RetrievalEvidence, RetrievalMode, RetrievalSection, RetrieveRequest
from infrastructure.db.database import get_db
from shared.logger import logger

_DOMINANT_THRESHOLD = 0.5

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


async def promote_dominant_symbols(bundle: RetrievalBundle, snapshot_id: str) -> RetrievalBundle:
    """After merging multiple sub-queries' results, a split function/class can have its
    parts scattered across DIFFERENT sub-queries (each part matching a different angle of
    the decomposed question) -- individually below the per-query completion threshold, but
    jointly a strong signal the symbol is central to the question. Complete any symbol
    whose parts reach _DOMINANT_THRESHOLD across the merged set and tag every member
    'dominant-seed' so render_bundle can surface it as one fully-assembled block instead
    of scattering its parts through the regular per-chunk list.

    Promotion is an enhancement over an already-usable merged bundle, so a failure here
    degrades to the un-promoted bundle instead of taking the whole section down with it."""
    if not bundle.evidences:
        return bundle
    try:
        return await _promote_dominant_symbols(bundle, snapshot_id)
    except Exception as e:
        logger.warning(
            "[retrieve_multi] dominant-symbol promotion skipped (%s: %s) — returning the merged "
            "bundle un-promoted; split symbols may appear as scattered parts.",
            type(e).__name__, e,
        )
        return bundle


async def _promote_dominant_symbols(bundle: RetrievalBundle, snapshot_id: str) -> RetrievalBundle:
    db = get_db()
    chunk_ids = [e.chunk_id for e in bundle.evidences]
    placeholders = ",".join("?" for _ in chunk_ids)
    async with db.execute(
        f"SELECT id, chunk_index, split_part, split_of, rel_path FROM retrieval_chunks "
        f"WHERE snapshot_id=? AND id IN ({placeholders})",
        (snapshot_id, *chunk_ids),
    ) as cur:
        meta_by_id = {r["id"]: dict(r) for r in await cur.fetchall()}

    groups: dict[tuple[str, int], dict] = {}
    for e in bundle.evidences:
        m = meta_by_id.get(e.chunk_id)
        if not m or (m["split_of"] or 1) <= 1:
            continue
        group_key = (m["rel_path"], m["chunk_index"] - m["split_part"])
        g = groups.setdefault(group_key, {"split_of": m["split_of"], "present_ids": {}})
        g["present_ids"][m["split_part"]] = e.chunk_id

    dominant_ids: set[str] = set()
    extra_evidences: list[RetrievalEvidence] = []
    used_tokens = bundle.used_tokens
    for (rel_path, group_start), g in groups.items():
        split_of = g["split_of"]
        present = g["present_ids"]
        if len(present) / split_of < _DOMINANT_THRESHOLD:
            continue
        dominant_ids.update(present.values())
        missing = [p for p in range(split_of) if p not in present]
        if not missing:
            continue
        ph = ",".join("?" for _ in missing)
        from domain.retrieval.service import _CHUNK_FULL_COLS

        async with db.execute(
            f"SELECT {_CHUNK_FULL_COLS} FROM retrieval_chunks "
            f"WHERE snapshot_id=? AND rel_path=? AND split_part IN ({ph}) "
            f"AND chunk_index BETWEEN ? AND ?",
            (snapshot_id, rel_path, *missing, group_start, group_start + split_of - 1),
        ) as cur:
            missing_rows = await cur.fetchall()
        for r in missing_rows:
            tok = int(r["token_estimate"] or 1)
            if used_tokens + tok > bundle.budget_tokens:
                continue
            extra_evidences.append(RetrievalEvidence(
                chunk_id=r["id"], rel_path=r["rel_path"], chunk_index=r["chunk_index"],
                reason_codes=["split-complete"], score=0.0, token_estimate=tok,
                excerpt=r["content"] or "", start_line=r["start_line"] or 0, end_line=r["end_line"] or 0,
            ))
            dominant_ids.add(r["id"])
            used_tokens += tok

    if not dominant_ids:
        return bundle

    updated_evidences = []
    for e in bundle.evidences:
        if e.chunk_id in dominant_ids and "dominant-seed" not in e.reason_codes:
            e = e.model_copy(update={"reason_codes": [*e.reason_codes, "dominant-seed"]})
        updated_evidences.append(e)
    for e in extra_evidences:
        e.reason_codes.append("dominant-seed")
        updated_evidences.append(e)

    return bundle.model_copy(update={"evidences": updated_evidences, "used_tokens": used_tokens})


async def retrieve_multi(
    retrieval_service,
    snapshot_id: str,
    queries: list[str],
    section: RetrievalSection,
    mode: RetrievalMode,
    max_results_each: int,
) -> RetrievalBundle:
    """Run `retrieve()` once per query, merge and deduplicate the results, then promote
    any symbol whose parts converge across multiple sub-queries to a dominant/seed block."""
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
    merged = merge_bundles(bundles, section)
    return await promote_dominant_symbols(merged, snapshot_id)
