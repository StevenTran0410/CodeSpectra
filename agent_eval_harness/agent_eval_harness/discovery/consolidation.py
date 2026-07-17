"""Pass E — Core-Seed Consolidation. Corrects Pass B's Louvain-assigned file→candidate
membership using Pass D's structurally-verified wiring blocks as ground truth, then a
global cross-candidate match (symbol-graph weight + path proximity), then a bounded LLM
judge for whatever's still ambiguous. Never mutates cluster_files/hub_paths/community_id —
adds matched_files + file_provenance alongside."""
from __future__ import annotations

import json
import logging

from agent_eval_harness.discovery.client import CodeSpectraClient
from agent_eval_harness.discovery.wiring import strip_json_code_fence
from agent_eval_harness.llm.client import LLMClient, LLMMessage, RateLimitExceeded

logger = logging.getLogger("agent_eval_harness.discovery.consolidation")

MIN_WIRING_NODES_FOR_ANCHOR = 2       # a 1-node "wiring block" is likely noise, not a real anchor
IMPORT_MATCH_MIN_SCORE = 1            # >=1 real symbol edge into a block's core seed = a match
MAX_CONSOLIDATION_LLM_JUDGE_CALLS = 15  # bounded, but generous
JUDGE_FILE_CONTENT_CHAR_BUDGET = 4000


def _core_seed_for(candidate: dict) -> tuple[set[str], str]:
    """Returns (core_seed_files, anchor_kind). anchor_kind is 'wiring' (strong) or 'hub' (weak,
    today's fallback behavior — used when there's no trustworthy wiring block)."""
    wb = candidate.get("wiring_block")
    if wb and isinstance(wb, dict) and len(wb.get("nodes") or []) >= MIN_WIRING_NODES_FOR_ANCHOR:
        files = {n["source_hint_file"] for n in wb["nodes"] if n.get("source_hint_file")}
        if files:
            return files, "wiring"
    return set(candidate.get("hub_paths", [])), "hub"


async def consolidate_candidates(
    candidates: list[dict],
    client: CodeSpectraClient,
    snapshot_id: str,
    llm_client: LLMClient,
) -> list[dict]:
    # 1. Anchors, one per candidate, ranked most-confident first (wiring > hub; more nodes = higher).
    anchors: list[tuple[dict, set[str], str]] = []
    for c in candidates:
        seed, kind = _core_seed_for(c)
        anchors.append((c, seed, kind))
    anchors.sort(
        key=lambda t: (t[2] == "wiring", len((t[0].get("wiring_block") or {}).get("nodes", []))),
        reverse=True,
    )

    # 2. Global pool: every file across every cluster, minus files already claimed as a core seed.
    already_claimed: dict[str, str] = {}  # file -> candidate community_id
    for c, seed, _kind in anchors:
        for f in seed:
            already_claimed[f] = str(c["community_id"])

    pool = set()
    for c in candidates:
        pool |= set(c.get("cluster_files", []))
    pool -= set(already_claimed.keys())

    # 3. Score each pooled file against each candidate's core seed via real symbol edges.
    file_provenance: dict[str, dict[str, str]] = {str(c["community_id"]): {} for c in candidates}
    matched: dict[str, set[str]] = {str(c["community_id"]): set(seed) for c, seed, _ in anchors}

    llm_judge_calls_used = 0
    for f in sorted(pool):
        scores: dict[str, int] = {str(c["community_id"]): 0 for c in candidates}
        try:
            edges = await client.get_symbol_edges(snapshot_id, f)
        except Exception as e:
            logger.warning(f"Failed to get symbol edges for {f} during consolidation: {e}")
            edges = {}

        neighbor_files: set[str] = set()
        for edge in (edges.get("outgoing") or []) + (edges.get("incoming") or []):
            for sym in (edge.get("src_symbol"), edge.get("dst_symbol")):
                if sym and "::" in sym:
                    nf, _ = sym.split("::", 1)
                    if nf != f:
                        neighbor_files.add(nf)

        for c, seed, _kind in anchors:
            cid = str(c["community_id"])
            scores[cid] = len(neighbor_files & seed)

        # Path-proximity tiebreaker only when symbol-edge signal is absent or tied.
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_score = ranked[0][1] if ranked else 0
        tied = [cid for cid, s in ranked if s == top_score]
        if top_score == 0 or len(tied) > 1:
            def _path_overlap(cid: str) -> int:
                # Exact-same-directory only — a shared-top-level-segment tier was tried and
                # dropped: it over-matches with only one candidate in the run.
                seed = next(s for cand, s, kind in anchors if str(cand["community_id"]) == cid)
                best = 0
                for sf in seed:
                    sf_dir = sf.rsplit("/", 1)[0] if "/" in sf else ""
                    f_dir = f.rsplit("/", 1)[0] if "/" in f else ""
                    if sf_dir and sf_dir == f_dir:
                        best = max(best, 2)
                return best
            candidates_for_tiebreak = tied if top_score > 0 else list(scores.keys())
            path_scores = {cid: _path_overlap(cid) for cid in candidates_for_tiebreak}
            best_path = max(path_scores.values(), default=0)
            if best_path > 0:
                ranked = sorted(path_scores.items(), key=lambda kv: kv[1], reverse=True)
                top_score = best_path
                tied = [cid for cid, s in ranked if s == best_path]

        if top_score >= IMPORT_MATCH_MIN_SCORE and len(tied) == 1:
            winner = tied[0]
            matched[winner].add(f)
            file_provenance[winner][f] = "import_matched" if any(
                scores[cid] > 0 for cid in scores
            ) else "path_matched"
            continue

        # No confident single winner from heuristics — real ambiguity, not a hopeless orphan, but
        # only spend an LLM call when there's an actual plausible shortlist to judge between.
        shortlist = tied[:2] if top_score > 0 else []
        if not shortlist:
            # No signal at all connecting this file to any block — leave unassigned rather
            # than force a guess or waste an LLM call arguing between two zero-signal options.
            continue

        if llm_judge_calls_used >= MAX_CONSOLIDATION_LLM_JUDGE_CALLS:
            for cid in shortlist:
                file_provenance[cid][f] = "needs_human"
            continue

        try:
            winner_cid = await _llm_judge_file_membership(f, shortlist, candidates, client, snapshot_id, llm_client)
            llm_judge_calls_used += 1
        except RateLimitExceeded:
            raise  # let the caller turn this into DiscoveryPaused, same as Pass C/D
        if winner_cid:
            matched[winner_cid].add(f)
            file_provenance[winner_cid][f] = "llm_linked"
        else:
            for cid in shortlist:
                file_provenance[cid][f] = "needs_human"

    for c, seed, kind in anchors:
        cid = str(c["community_id"])
        c["matched_files"] = sorted(matched[cid])
        c["file_provenance"] = {
            **{f: "core_seed" for f in seed},
            **file_provenance[cid],
        }
        c["seed_anchor_kind"] = kind

    return candidates


async def _llm_judge_file_membership(
    file_path: str,
    shortlist_cids: list[str],
    candidates: list[dict],
    client: CodeSpectraClient,
    snapshot_id: str,
    llm_client: LLMClient,
) -> str | None:
    """Ask the LLM which of the (at most 2) shortlisted candidates this file actually belongs
    to. Returns None (not a forced guess) if the LLM isn't confident either."""
    by_cid = {str(c["community_id"]): c for c in candidates}
    try:
        content = (await client.read_file(snapshot_id, file_path)).get("content", "")
    except Exception as e:
        logger.warning(f"Could not read {file_path} for consolidation judge, judging blind: {e}")
        content = ""
    content = content[:JUDGE_FILE_CONTENT_CHAR_BUDGET]

    options = []
    for cid in shortlist_cids:
        c = by_cid[cid]
        options.append(f"- id={cid} name=\"{c.get('name')}\" frameworks={c.get('frameworks')}")

    system_prompt = (
        "You are an expert AI software architect. A file could not be confidently assigned to "
        "one of a few candidate agentic systems by structural analysis alone. Decide which "
        "candidate (if any) this file actually belongs to.\n"
        "Respond ONLY in raw JSON: {\"candidate_id\": \"string or null\", \"confidence\": "
        "\"high\"|\"medium\"|\"low\"}. Use null if you are not reasonably confident."
    )
    user_prompt = (
        f"File: {file_path}\n\nCandidates:\n" + "\n".join(options) +
        f"\n\nFile content:\n{content}\n\nWhich candidate_id does this file belong to?"
    )

    response = await llm_client.complete(
        [LLMMessage(role="system", content=system_prompt), LLMMessage(role="user", content=user_prompt)],
        json_mode=True,
    )
    try:
        parsed = json.loads(strip_json_code_fence(response.content))
    except Exception as e:
        logger.warning(f"Could not parse consolidation judge response for {file_path}: {e}")
        return None
    if parsed.get("confidence") == "low":
        return None
    cid = parsed.get("candidate_id")
    return cid if cid in shortlist_cids else None
