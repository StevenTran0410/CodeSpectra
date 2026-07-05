"""AEH Discovery Engine.

Three-Pass algorithm over external read APIs:
- Pass A: scan symbols + text retrieval hits using fingerprints.yaml.
- Pass B: group files using Louvain communities node index, ranking by density + cohesion.
- Pass C: synthesize with LLM per cluster, utilizing sibling CA report sections if present.
"""
from __future__ import annotations

import json
import logging
import re
import traceback
from pathlib import Path
from typing import Any
import yaml

from agent_eval_harness.discovery.client import CodeSpectraClient
from agent_eval_harness.llm.client import LLMClient, LLMMessage
from agent_eval_harness.store import repository

logger = logging.getLogger("agent_eval_harness.discovery.engine")

# Pass C is the only LLM-metered stage — cap how many clusters get a real LLM
# call so a repo with many communities can't blow an unbounded token budget.
# Clusters beyond this rank (by density, already sorted) are still emitted as
# candidates (never silently dropped, per epic §4e) but marked unknown/
# needs_human without spending an LLM call on them.
MAX_LLM_SYNTHESIZED_CLUSTERS = 12


def load_fingerprints() -> list[dict[str, Any]]:
    p = Path(__file__).parent / "fingerprints.yaml"
    if not p.exists():
        logger.warning(f"fingerprints.yaml not found at {p}")
        return []
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("fingerprints", [])


async def discover_agentic_systems(
    snapshot_id: str,
    repo_ref: str,
    client: CodeSpectraClient,
    llm_client: LLMClient,
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
                        })
            elif fp_type == "retrieval":
                res = await client.search_retrieval(snapshot_id, query)
                for evidence in res.get("evidences", []):
                    if re.search(pattern, evidence["excerpt"], re.IGNORECASE):
                        hits.append({
                            "file": evidence["rel_path"],
                            "symbol": None,
                            "fingerprint_id": fp_id,
                            "snippet": evidence["excerpt"],
                            "framework": framework,
                            "weight": weight,
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

    # Sibling CA report resolution
    ca_report_content = None
    try:
        snap = await client.get_snapshot(snapshot_id)
        repo_id = snap.get("local_repo_id")
        repo = await client.get_repo(repo_id)
        repo_path = repo.get("path")
        workspace_id = repo.get("workspace_id")

        all_repos = await client.list_repos(workspace_id=workspace_id, mode="code_analysis")
        sibling_repo = None
        for r in all_repos:
            if r.get("path") == repo_path:
                sibling_repo = r
                break

        if sibling_repo:
            reports = await client.list_reports(repo_id=sibling_repo["id"])
            if reports:
                latest_report_id = reports[0]["id"]
                full_report = await client.get_report(latest_report_id)
                sections = []
                report_data = full_report.get("report", {})
                for sec in report_data.get("sections", []):
                    sec_name = sec.get("section", "").lower()
                    if any(x in sec_name for x in ["architecture", "feature_map", "important_files"]):
                        sections.append(f"### {sec.get('section')}\n{sec.get('content')}")
                if sections:
                    ca_report_content = "\n\n".join(sections)
                    logger.info("Found matching Code Analysis report sections to use as context.")
    except Exception as exc:
        logger.warning(f"Could not retrieve sibling CA report: {exc}")

    candidates = []
    for rank, cluster in enumerate(candidate_clusters):
        cid = cluster["community_id"]
        sorted_hits = sorted(cluster["hits"], key=lambda h: h["weight"], reverse=True)
        top_hits = sorted_hits[:15]

        if rank >= MAX_LLM_SYNTHESIZED_CLUSTERS:
            # Budget exhausted — never silently drop the cluster (epic §4e), just
            # skip the LLM call and surface it as needing human triage instead of
            # confidently guessing or pretending it doesn't exist.
            candidates.append({
                "name": "unknown",
                "frameworks": cluster["frameworks"] or ["unknown"],
                "entry_points": cluster["hub_paths"][:2],
                "confidence": "low",
                "verdict": "proposed",
                "needs_human": True,
                "evidence": top_hits,
                "skip_reason": "llm_synthesis_budget_exceeded",
            })
            continue

        evidence_summary = []
        for h in top_hits:
            evidence_summary.append(
                f"File: {h['file']}\n"
                f"Symbol: {h['symbol'] or 'None'}\n"
                f"Framework: {h['framework']}\n"
                f"Snippet:\n{h['snippet']}\n"
                f"----------------------------------------"
            )
        evidence_bundle_str = "\n".join(evidence_summary)

        system_prompt = (
            "You are an expert AI software architect. Analyze the provided codebase evidence cluster "
            "and determine if it represents a candidate agentic system in the codebase.\n"
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
        if ca_report_content:
            user_prompt += f"Sibling Code Analysis Report Context:\n{ca_report_content}\n\n"

        user_prompt += "Does this cluster represent a candidate agentic system? Return the JSON profile."

        candidate_profile = {
            "name": "unknown",
            "frameworks": cluster["frameworks"] or ["unknown"],
            "entry_points": cluster["hub_paths"][:2],
            "confidence": "low",
            "verdict": "proposed",
            "evidence": top_hits,
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
        except Exception as exc:
            logger.warning(f"LLM synthesis failed for cluster {cid}: {exc}. Using fallback.")

        # Below-threshold or unnamed clusters surface as needs_human rather than
        # being silently promoted with a low-confidence guess (epic §4e) — the UI
        # must be able to tell "reviewed, low confidence" apart from "not
        # reviewed yet" candidates.
        candidate_profile["needs_human"] = (
            candidate_profile["name"] == "unknown" or candidate_profile["confidence"] == "low"
        )
        candidates.append(candidate_profile)

    return candidates


async def run_discovery_background(
    session_id: str,
    snapshot_id: str,
    repo_ref: str,
    client: CodeSpectraClient,
    llm_client: LLMClient,
) -> None:
    """Executes discovery process asynchronously and stores the output candidates in the DB."""
    try:
        try:
            candidates = await discover_agentic_systems(snapshot_id, repo_ref, client, llm_client)
            await repository.insert_discovery_candidates_bulk(session_id, candidates)
            await repository.finish_discovery_session(session_id, "completed")
            logger.info(f"Discovery session {session_id} completed successfully.")
        except Exception as e:
            err_msg = "".join(traceback.format_exception(None, e, e.__traceback__))
            logger.error(f"Discovery session {session_id} failed: {e}\n{err_msg}")
            await repository.finish_discovery_session(session_id, "failed", err_msg[:2000])
    finally:
        # Every call site builds a fresh CodeSpectraClient/LLM client per session —
        # without closing them here, each discovery run leaks an httpx.AsyncClient
        # connection pool for the lifetime of the (long-running) AEH backend process.
        await client.aclose()
        if hasattr(llm_client, "aclose"):
            await llm_client.aclose()
