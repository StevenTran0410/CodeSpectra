"""AEH Discovery Engine implementing Three-Pass agentic pipeline auto-discovery."""
from __future__ import annotations

import json
import logging
import re
import traceback
from pathlib import Path
from typing import Any
import yaml

from agent_eval_harness.discovery.client import CodeSpectraClient
from agent_eval_harness.discovery.analysis_context import ProjectContext, load_project_context
from agent_eval_harness.llm.client import LLMClient, LLMMessage, RateLimitExceeded
from agent_eval_harness.store import repository

logger = logging.getLogger("agent_eval_harness.discovery.engine")

# LLM budget limit: cap how many clusters get a real LLM call to save tokens.
MAX_LLM_SYNTHESIZED_CLUSTERS = 12

# Separate, smaller budget for Pass D's wiring-detection LLM fallback (static
# detection is free and always tried first; this only bounds the escalation).
MAX_WIRING_LLM_FALLBACK_CALLS = 5


def load_fingerprints() -> list[dict[str, Any]]:
    p = Path(__file__).parent / "fingerprints.yaml"
    if not p.exists():
        logger.warning(f"fingerprints.yaml not found at {p}")
        return []
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("fingerprints", [])


def _encode_hits_toon(hits: list[dict[str, Any]]) -> str:
    lines = [f"hits[{len(hits)}]{{file,symbol,framework,weight,token_estimate,snippet}}:"]
    for h in hits:
        symbol = h["symbol"] or ""
        # escape embedded newlines/commas in snippet so each hit stays one row
        snippet = h["snippet"].replace("\n", "\\n").replace(",", "\\,")
        lines.append(
            f"  {h['file']},{symbol},{h['framework']},{h['weight']},"
            f"{h.get('token_estimate', 0)},{snippet}"
        )
    return "\n".join(lines)


class DiscoveryPaused(Exception):
    """Raised when a sustained rate limit interrupts Pass C/D. Carries every cluster that
    finished successfully before the pause, so the caller can persist progress and resume
    without re-synthesizing clusters that are already done."""
    def __init__(
        self,
        candidates_so_far: list[dict],
        provider_id: str,
        model_id: str | None,
        reasoning_effort: str | None = None,
        thinking_budget: int | None = None,
    ) -> None:
        super().__init__(f"Discovery paused: rate limit on provider={provider_id} model={model_id}")
        self.candidates_so_far = candidates_so_far
        self.provider_id = provider_id
        self.model_id = model_id
        self.reasoning_effort = reasoning_effort
        self.thinking_budget = thinking_budget


async def discover_agentic_systems(
    snapshot_id: str,
    repo_ref: str,
    client: CodeSpectraClient,
    llm_client: LLMClient,
    already_named: dict[str, dict[str, Any]] | None = None,
    project_context: ProjectContext | None = None,
) -> list[dict[str, Any]]:
    logger.info("Starting AEH Discovery Pass A: Fingerprinting scan...")
    fingerprints = load_fingerprints()
    hits = []

    for fp in fingerprints:
        fp_id = fp["id"]
        fp_type = fp["type"]
        query = fp["query"]
        pattern = fp["pattern"]
        framework = fp.get("framework", "unknown")
        weight = fp.get("weight", 1.0)

        try:
            if fp_type == "symbol":
                res = await client.search_repo_map(snapshot_id, query)
                for sym in res.get("symbols", []):
                    if re.search(pattern, sym["name"], re.IGNORECASE):
                        hits.append({
                            "file": sym["rel_path"],
                            "symbol": sym["name"],
                            "fingerprint_id": fp_id,
                            "snippet": sym.get("signature") or sym["name"],
                            "framework": framework,
                            "weight": weight,
                            "token_estimate": len(sym.get("signature") or sym["name"]) // 4,
                        })
            elif fp_type == "retrieval":
                # Retrieve RRF fusion is used here to avoid budget-caps missing matches.
                res = await client.search_retrieval(snapshot_id, query)
                for entry in res.get("fused", []):
                    if re.search(pattern, entry["excerpt"], re.IGNORECASE):
                        hits.append({
                            "file": entry["rel_path"],
                            "symbol": None,
                            "fingerprint_id": fp_id,
                            "snippet": entry["excerpt"],
                            "framework": framework,
                            "weight": weight,
                            "token_estimate": len(entry["excerpt"]) // 4,
                            "chunk_id": entry.get("chunk_id"),
                        })
        except Exception as exc:
            logger.warning(f"Query failed for fingerprint {fp_id}: {exc}")

    if not hits:
        logger.info("No fingerprints matched. Ending discovery.")
        return []

    logger.info(f"Found {len(hits)} fingerprint hits. Starting Pass B: Graph Clustering...")

    # Fetch Louvain communities
    try:
        communities_res = await client.get_communities(snapshot_id)
        node_index = communities_res.get("node_index", {})
        communities_list = communities_res.get("communities", [])
    except Exception as exc:
        logger.warning(f"Failed to fetch structural communities: {exc}")
        node_index = {}
        communities_list = []

    clusters: dict[int | str, list[dict[str, Any]]] = {}
    community_files: dict[int | str, list[str]] = {}

    for path, cid in node_index.items():
        community_files.setdefault(cid, []).append(path)

    for hit in hits:
        path = hit["file"]
        cid: int | str = node_index.get(path, f"fallback_{path}")
        clusters.setdefault(cid, []).append(hit)
        if cid not in node_index:
            community_files.setdefault(cid, []).append(path)

    candidate_clusters = []
    for cid, c_hits in clusters.items():
        hit_files = list({h["file"] for h in c_hits})
        all_files = list(set(community_files.get(cid, hit_files)))

        density = sum(h["weight"] for h in c_hits)
        cohesion = len(hit_files) / max(1, len(all_files))

        frameworks = list({h["framework"] for h in c_hits if h["framework"] != "unknown"})

        hub_paths = []
        if isinstance(cid, int):
            for comm in communities_list:
                if comm.get("community_id") == cid:
                    hub_paths = comm.get("hub_paths", [])
                    break
        if not hub_paths:
            hub_paths = hit_files[:3]

        candidate_clusters.append({
            "community_id": cid,
            "files": all_files,
            "hit_files": hit_files,
            "hits": c_hits,
            "density": density,
            "cohesion": cohesion,
            "frameworks": frameworks,
            "hub_paths": hub_paths,
        })

    candidate_clusters.sort(key=lambda x: x["density"], reverse=True)
    logger.info(f"Clustered into {len(candidate_clusters)} candidates. Starting Pass C: LLM Synthesis...")

    candidates = []
    wiring_llm_fallback_used = 0
    for rank, cluster in enumerate(candidate_clusters):
        cid = cluster["community_id"]
        cid_str = str(cid)
        if already_named and cid_str in already_named:
            candidates.append(already_named[cid_str])
            continue

        sorted_hits = sorted(cluster["hits"], key=lambda h: h["weight"], reverse=True)
        top_hits = sorted_hits[:15]

        if rank >= MAX_LLM_SYNTHESIZED_CLUSTERS:
            # LLM budget limit exceeded: mark cluster as unknown/needs_human without LLM call.
            candidates.append({
                "name": "unknown",
                "frameworks": cluster["frameworks"] or ["unknown"],
                "entry_points": cluster["hub_paths"][:2],
                "confidence": "low",
                "verdict": "proposed",
                "needs_human": True,
                "evidence": top_hits,
                "skip_reason": "llm_synthesis_budget_exceeded",
                "community_id": str(cluster["community_id"]),
                "cluster_files": cluster["files"],
                "hub_paths": cluster["hub_paths"],
            })
            continue

        evidence_bundle_str = _encode_hits_toon(top_hits)

        system_prompt = (
            "You are an expert AI software architect. Analyze the provided codebase evidence cluster "
            "and determine if it represents a candidate agentic system in the codebase.\n"
            "Evidence is given in a compact tabular format: a header line declares the column names once, then one row per hit.\n"
            "An agentic system typically includes elements like an agent loop, tools, prompt templates, "
            "or orchestrators using frameworks like Haystack, LangChain, LangGraph, CrewAI, AutoGen, "
            "Semantic Kernel, or custom LLM API integrations.\n"
            "Respond ONLY in raw JSON format with the following schema:\n"
            "{\n"
            '  "is_agentic_system": boolean,\n'
            '  "name": "string (descriptive name of the agentic system, e.g. \'QA Pipeline\', or \'unknown\')",\n'
            '  "frameworks": ["string (e.g. \'haystack\', \'openai\', etc.)"],\n'
            '  "entry_points": ["string (key files/interfaces that entry into the system)"],\n'
            '  "component_count_estimate": integer,\n'
            '  "confidence": "string (\'high\', \'medium\', \'low\')"\n'
            "}"
        )

        user_prompt = (
            f"Community/Cluster ID: {cid}\n"
            f"Hub Paths (potential entry points): {', '.join(cluster['hub_paths'])}\n"
            f"All community files: {', '.join(cluster['files'][:30])}\n\n"
            f"Evidence Hits:\n{evidence_bundle_str}\n\n"
        )
        if project_context:
            if project_context.identity:
                user_prompt += project_context.identity.as_context_block()
            if project_context.synthesis:
                user_prompt += project_context.synthesis.as_context_block()
            if project_context.important_files:
                hub_set = set(cluster["hub_paths"])
                tagged = [p for p in project_context.important_files if p in hub_set]
                if tagged:
                    user_prompt += f"\nKnown important files in this cluster: {', '.join(tagged)}\n"

        user_prompt += "Does this cluster represent a candidate agentic system? Return the JSON profile."

        candidate_profile = {
            "name": "unknown",
            "frameworks": cluster["frameworks"] or ["unknown"],
            "entry_points": cluster["hub_paths"][:2],
            "confidence": "low",
            "verdict": "proposed",
            "evidence": top_hits,
            "community_id": str(cluster["community_id"]),
            "cluster_files": cluster["files"],
            "hub_paths": cluster["hub_paths"],
        }

        try:
            messages = [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ]
            response = await llm_client.complete(messages, max_tokens=1024, json_mode=True)
            content = response.content.strip()

            if content.startswith("```"):
                lines = content.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                content = "\n".join(lines).strip()

            parsed = json.loads(content)
            if not parsed.get("is_agentic_system", True):
                candidate_profile["name"] = "unknown"
                candidate_profile["confidence"] = "low"
            else:
                candidate_profile["name"] = parsed.get("name") or "unknown"
                candidate_profile["frameworks"] = parsed.get("frameworks") or cluster["frameworks"] or ["unknown"]
                candidate_profile["entry_points"] = parsed.get("entry_points") or cluster["hub_paths"][:2]
                candidate_profile["confidence"] = parsed.get("confidence") or "low"
        except RateLimitExceeded as rle:
            raise DiscoveryPaused(
                candidates, rle.provider_id, rle.model_id, rle.reasoning_effort, rle.thinking_budget
            ) from rle
        except Exception as exc:
            logger.warning(f"LLM synthesis failed for cluster {cid}: {exc}. Using fallback.")

        # Low confidence or unknown name candidates are flagged for human review.
        candidate_profile["needs_human"] = (
            candidate_profile["name"] == "unknown" or candidate_profile["confidence"] == "low"
        )

        # B3: Risk flag propagation from analysis to candidate profile
        if project_context and project_context.risk_findings:
            hub_path_set = set(cluster["hub_paths"])
            for finding in project_context.risk_findings:
                if hub_path_set.intersection(finding.get("evidence", [])):
                    candidate_profile.setdefault("risk_flags", []).append(finding)
                    candidate_profile["needs_human"] = True

        # Pass D: Detect wiring block
        file_contents = {}
        for path in candidate_profile.get("cluster_files", []):
            try:
                file_resp = await client.read_file(snapshot_id, path)
                file_contents[path] = file_resp.get("content", "")
            except Exception as e:
                logger.warning(f"Failed to read {path} for wiring detection: {e}")

        from agent_eval_harness.discovery.wiring import detect_wiring_block
        # Static detection is always attempted (free); the LLM fallback is only
        # allowed for candidates worth naming, within its own separate budget.
        allow_wiring_llm_fallback = (
            not candidate_profile["needs_human"]
            and wiring_llm_fallback_used < MAX_WIRING_LLM_FALLBACK_CALLS
        )
        try:
            wiring_block = await detect_wiring_block(
                file_contents, llm_client if allow_wiring_llm_fallback else None
            )
        except RateLimitExceeded as rle:
            raise DiscoveryPaused(
                candidates, rle.provider_id, rle.model_id, rle.reasoning_effort, rle.thinking_budget
            ) from rle

        if wiring_block and wiring_block.source == "llm_fallback":
            wiring_llm_fallback_used += 1
        candidate_profile["wiring_block"] = wiring_block.to_dict() if wiring_block else None

        candidates.append(candidate_profile)

    try:
        from agent_eval_harness.discovery.consolidation import consolidate_candidates
        candidates = await consolidate_candidates(candidates, client, snapshot_id, llm_client)
    except RateLimitExceeded as rle:
        raise DiscoveryPaused(
            candidates, rle.provider_id, rle.model_id, rle.reasoning_effort, rle.thinking_budget
        ) from rle

    return candidates


async def run_discovery_background(
    session_id: str,
    snapshot_id: str,
    repo_ref: str,
    client: CodeSpectraClient,
    llm_client: LLMClient,
    already_named: dict[str, dict] | None = None,
) -> None:
    """Executes discovery process asynchronously and stores the output candidates in the DB."""
    try:
        # Load project context (code-analysis report) for this snapshot
        project_context = await load_project_context(client, snapshot_id)
        await repository.update_discovery_session_project_context(session_id, project_context)

        try:
            candidates = await discover_agentic_systems(
                snapshot_id, repo_ref, client, llm_client, already_named=already_named,
                project_context=project_context
            )
            await repository.replace_discovery_candidates(session_id, candidates)
            await repository.finish_discovery_session(session_id, "completed")
            logger.info(f"Discovery session {session_id} completed successfully.")
        except DiscoveryPaused as p:
            await repository.replace_discovery_candidates(session_id, p.candidates_so_far)
            await repository.pause_discovery_session(
                session_id, p.provider_id, p.model_id, p.reasoning_effort, p.thinking_budget
            )
            logger.info(f"Discovery session {session_id} paused on rate limit (provider={p.provider_id}).")
        except Exception as e:
            err_msg = "".join(traceback.format_exception(None, e, e.__traceback__))
            logger.error(f"Discovery session {session_id} failed: {e}\n{err_msg}")
            await repository.finish_discovery_session(session_id, "failed", err_msg[:2000])
    finally:
        # Close connection pools to prevent leaking client resources.
        await client.aclose()
        if hasattr(llm_client, "aclose"):
            await llm_client.aclose()
