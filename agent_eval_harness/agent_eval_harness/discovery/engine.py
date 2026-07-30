"""AEH Discovery Engine implementing Three-Pass agentic pipeline auto-discovery."""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import posixpath
import re
import traceback
from pathlib import Path
from typing import Any

import yaml

from agent_eval_harness.discovery.analysis_context import ProjectContext, load_project_context
from agent_eval_harness.discovery.client import CodeSpectraClient
from agent_eval_harness.discovery.consolidation import consolidate_candidates
from agent_eval_harness.discovery.wiring import (
    classify_system_type,
    detect_wiring_block,
    detect_wiring_systems,
    strip_json_code_fence,
)
from agent_eval_harness.mapping.builder.system_types import (
    ALL_SYSTEM_TYPES, DECIDED_BY_STRUCTURAL, DECIDED_BY_LLM, DECIDED_BY_UNRESOLVED
)
from agent_eval_harness.llm.client import LLMClient, LLMMessage, RateLimitExceeded
from agent_eval_harness.store import repository

logger = logging.getLogger("agent_eval_harness.discovery.engine")

# LLM budget limit: cap how many clusters get a real LLM call to save tokens.
MAX_LLM_SYNTHESIZED_CLUSTERS = 12

# Separate, smaller budget for Pass D's wiring-detection LLM fallback (static
# detection is free and tried first).
MAX_WIRING_LLM_FALLBACK_CALLS = 5

MAX_HITS_PER_CLUSTER_PROMPT = 15  # top-N highest-weight hits shown to the synthesis LLM
MAX_LISTED_FILES = 30             # cap on community file paths listed verbatim in the prompt
MAX_LISTED_COMPONENTS = 20        # cap on component aliases listed per detected system
MAX_FLOW_EDGES_PROMPT = 25        # cap on call-flow edges listed per detected system
SYNTHESIS_MAX_TOKENS = 1024


def load_fingerprints() -> list[dict[str, Any]]:
    p = Path(__file__).parent / "fingerprints.yaml"
    if not p.exists():
        logger.warning(f"fingerprints.yaml not found at {p}")
        return []
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("fingerprints", [])


def load_provider_boundaries() -> list[dict[str, Any]]:
    p = Path(__file__).parent / "provider_boundaries.yaml"
    if not p.exists():
        logger.warning(f"provider_boundaries.yaml not found at {p}")
        return []
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("boundaries", [])


def provider_boundaries_content_hash() -> str:
    """Content hash of provider_boundaries.yaml -- part of the LLM residue pass's cache key, so a
    change to the boundary vocabulary invalidates a stored verdict instead of silently reusing it."""
    p = Path(__file__).parent / "provider_boundaries.yaml"
    if not p.exists():
        return ""
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _module_path_for_rel_file(rel_file: str) -> str:
    """Repo-relative posix path -> dotted module path (drops a trailing __init__)."""
    stem = rel_file[:-3] if rel_file.endswith(".py") else rel_file
    parts = [p for p in stem.split("/") if p]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def match_provider_boundary_entry(
    boundaries: list[dict[str, Any]], *, defining_file: str, owner_class: str | None, resolved_symbol: str,
) -> dict[str, Any] | None:
    """A resolved (defining_file, owner_class, resolved_symbol) against an `entry:` row -- an in-repo
    interface/Protocol method named as `module.Class.method`."""
    if not owner_class or not resolved_symbol:
        return None
    dotted = f"{_module_path_for_rel_file(defining_file)}.{owner_class}.{resolved_symbol}"
    for row in boundaries:
        if row.get("entry") == dotted:
            return row
    return None


def match_provider_boundary_import(
    boundaries: list[dict[str, Any]], module: str | None,
) -> dict[str, Any] | None:
    """An external (unresolvable-on-disk) import module against an `import:` row -- top-level package
    identity match (`openai.types.chat` matches an `import: openai` row)."""
    if not module:
        return None
    top = module.split(".")[0]
    for row in boundaries:
        imp = row.get("import")
        if imp and imp.split(".")[0] == top:
            return row
    return None


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


def _system_signature(framework: str, aliases: list[str]) -> str:
    """Order-independent, run-stable hash of a system's (framework + node aliases) so a split
    candidate's system_id is deterministic across re-runs and pause/resume — never an iteration
    counter."""
    raw = framework + "|" + ",".join(sorted(aliases))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]


def _entry_aliases(nodes: list[dict], edges: list[dict]) -> list[str]:
    """Roots of a wiring subgraph = nodes that are never an edge destination; fall back to all
    aliases when everything is a destination (a cycle)."""
    dsts = {e["dst"] for e in edges}
    roots = sorted(n["alias"] for n in nodes if n["alias"] not in dsts)
    return roots or sorted(n["alias"] for n in nodes)


def _system_descriptor(idx: int, block: Any) -> str:
    """Compact description of one detected system for the synthesis prompt — framework, entry
    symbols, components, folders and call flow: enough for the LLM to name it by what it DOES."""
    wb = block.to_dict()
    nodes, edges = wb["nodes"], wb["edges"]
    files = sorted({n["source_hint_file"] for n in nodes if n.get("source_hint_file")})
    dirs = sorted({posixpath.dirname(f) or "." for f in files})
    lines = [
        f"System #{idx}:",
        f"  framework: {block.framework}",
        f"  entry symbols: {', '.join(_entry_aliases(nodes, edges)[:4]) or 'n/a'}",
        f"  components ({len(nodes)}): {', '.join(n['alias'] for n in nodes[:MAX_LISTED_COMPONENTS])}",
        f"  folders: {', '.join(dirs)}",
        f"  files: {', '.join(files[:MAX_LISTED_FILES])}",
    ]
    if edges:
        flow = ", ".join(f"{e['src']} -> {e['dst']}" for e in edges[:MAX_FLOW_EDGES_PROMPT])
        lines.append(f"  call flow: {flow}")
    return "\n".join(lines)


def _dedupe_system_names(names: dict[int, str], systems: list[Any]) -> dict[int, str]:
    """Guard against the LLM handing two systems the same name — collisions get their framework
    appended so every block in a community still shows a distinct label."""
    out: dict[int, str] = {}
    seen: set[str] = set()
    for i, block in enumerate(systems):
        nm = names.get(i + 1)
        if not nm:
            continue
        if nm.lower() in seen:
            nm = f"{nm} ({block.framework})"
        seen.add(nm.lower())
        out[i + 1] = nm
    return out


def _build_system_candidate(
    base_profile: dict, cluster: dict, block: Any, name_override: str | None = None
) -> dict:
    """One candidate for one wired system: its OWN pure sub-block, pure framework, tight core seed
    (the block's source files), deterministic system_id + name. cluster_files stays the FULL
    community pool so consolidation's global match still assigns every file to the nearest seed."""
    wb = block.to_dict()
    fw = block.framework
    prof = copy.deepcopy(base_profile)
    system_files = sorted({n["source_hint_file"] for n in wb["nodes"] if n.get("source_hint_file")})
    prof["system_id"] = f"{cluster['community_id']}#{fw}-{_system_signature(fw, [n['alias'] for n in wb['nodes']])}"
    prof["frameworks"] = [fw]
    prof["wiring_block"] = wb
    # Scope this split candidate's built map to only its own framework's components.
    prof["map_scope_framework"] = fw
    prof["hub_paths"] = system_files or base_profile.get("hub_paths", [])
    prof["entry_points"] = _entry_aliases(wb["nodes"], wb["edges"])[:2] or base_profile.get("entry_points", [])
    base_name = base_profile.get("name") or "unknown"
    # The synthesis LLM names each detected system individually (it sees this block's entry symbols,
    # components, folders and call flow); the community-name + framework suffix is only the fallback.
    prof["name"] = name_override or (
        f"{base_name} — {fw}" if base_name != "unknown" else f"{fw} system"
    )
    sysfiles = set(system_files)
    filtered_ev = [h for h in base_profile.get("evidence", []) if h.get("file") in sysfiles]
    if filtered_ev:
        prof["evidence"] = filtered_ev
    # CS-320: Structural system-type classification
    struct = classify_system_type(block)
    prof["system_type"] = struct.get("type")
    prof["system_type_signals"] = struct
    return prof


def _build_residual_candidate(base_profile: dict, cluster: dict, systems: list[Any]) -> dict | None:
    """A plain-python system has no wiring edges and appears in no detector output. After carving the
    wired systems, emit ONE residual candidate for the community's remaining agentic signal (hub /
    fingerprint-hit files not claimed by any wired system's source seed). Tight hub/hit seed only —
    never the full community pool — so its expansion stays bounded. Returns None when nothing is left."""
    wired_seed_files = {
        n["source_hint_file"]
        for b in systems for n in b.to_dict()["nodes"] if n.get("source_hint_file")
    }
    residual_hubs = [p for p in cluster.get("hub_paths", []) if p not in wired_seed_files]
    residual_hits = sorted({h["file"] for h in cluster.get("hits", [])} - wired_seed_files)
    seed = sorted(set(residual_hubs) | set(residual_hits))
    if not seed:
        return None
    prof = copy.deepcopy(base_profile)
    prof["system_id"] = f"{cluster['community_id']}#plain_python-{_system_signature('plain_python', seed)}"
    prof["frameworks"] = ["plain_python"]
    prof["wiring_block"] = None
    prof["hub_paths"] = seed
    prof["entry_points"] = seed[:2]
    # Scope the residual map to plain-python components, minus any class a wired sibling already owns
    # (e.g. a StateGraph node's owner class), so a sibling's agent class never bleeds into this map.
    prof["map_scope_framework"] = "plain_python"
    sibling_owned = {
        cls
        for b in systems for n in b.to_dict()["nodes"]
        for cls in (n.get("owner_class"), n.get("callee_name")) if cls
    }
    prof["excluded_component_classes"] = sorted(sibling_owned)
    base_name = base_profile.get("name") or "unknown"
    prof["name"] = f"{base_name} — plain_python" if base_name != "unknown" else "plain_python system"
    # CS-320: Residual (no block) gets low-confidence None type
    prof["system_type"] = None
    prof["system_type_signals"] = {
        "kind": "agent", "type": None, "confidence": "low", "candidate_types": [],
        "capability_tags": [], "signals": {}
    }
    return prof


class DiscoveryPaused(Exception):
    """Raised when a sustained rate limit interrupts Pass C/D; carries every cluster finished
    before the pause so the caller can resume without re-synthesizing them."""
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

    # Project context is identical for every cluster in a run, so it belongs in the shared prompt
    # prefix (system message) rather than after each cluster's data — otherwise it sits behind
    # variable text and never lands in the provider's prompt cache.
    project_ctx_block = ""
    if project_context:
        if project_context.identity:
            project_ctx_block += "\n" + project_context.identity.as_context_block()
        if project_context.synthesis:
            project_ctx_block += project_context.synthesis.as_context_block()

    candidates = []
    wiring_llm_fallback_used = 0
    for rank, cluster in enumerate(candidate_clusters):
        cid = cluster["community_id"]
        cid_str = str(cid)
        if already_named and cid_str in already_named:
            candidates.append(already_named[cid_str])
            continue

        sorted_hits = sorted(cluster["hits"], key=lambda h: h["weight"], reverse=True)
        top_hits = sorted_hits[:MAX_HITS_PER_CLUSTER_PROMPT]

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

        file_contents = {}
        for path in cluster["files"]:
            try:
                file_resp = await client.read_file(snapshot_id, path)
                file_contents[path] = file_resp.get("content", "")
            except Exception as e:
                logger.warning(f"Failed to read {path} for wiring detection: {e}")

        # Partition the community into one candidate per distinct agentic system =
        # (framework) x (connected wiring component). Static + free, and run BEFORE synthesis so the
        # single LLM call can name every system from its own components/folders/call flow.
        systems = detect_wiring_systems(file_contents)
        systems_block = (
            "\nDetected agentic systems in this community — each is a SEPARATE system needing its "
            "own name:\n" + "\n".join(_system_descriptor(i + 1, b) for i, b in enumerate(systems))
            + "\n"
        ) if systems else ""

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
            '  "confidence": "string (\'high\', \'medium\', \'low\')",\n'
            '  "systems": [{"index": integer, "name": "string", "system_type": "string or null"}]\n'
            "}\n"
            "When the evidence lists detected agentic systems, return one entry per system in "
            '"systems", with "index" matching its "System #N" and a name that describes what THAT '
            "system does — its functionality and domain, inferred from its entry symbols, component "
            "names, call flow and folder path. Each name must be distinct from the other systems' "
            "names, must not be a framework name, and must not be the community name with a "
            'framework suffix. The top-level "name" still names the community as a whole. '
            "For each system, optionally populate system_type from: pipeline, routing, parallelization, "
            "orchestrator, peer-collaboration, evaluator-optimizer, debate, blackboard, single-flow, "
            "tool-loop, plan-execute, reflection, or null if ambiguous."
            + project_ctx_block
        )

        user_prompt = (
            f"Community/Cluster ID: {cid}\n"
            f"Hub Paths (potential entry points): {', '.join(cluster['hub_paths'])}\n"
            f"All community files: {', '.join(cluster['files'][:MAX_LISTED_FILES])}\n"
            f"{systems_block}\n"
            f"Evidence Hits:\n{evidence_bundle_str}\n\n"
        )
        if project_context:
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

        system_names: dict[int, str] = {}
        system_types: dict[int, str | None] = {}
        try:
            messages = [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ]
            response = await llm_client.complete(
                messages, max_tokens=SYNTHESIS_MAX_TOKENS, json_mode=True
            )
            content = strip_json_code_fence(response.content)

            parsed = json.loads(content)
            if not parsed.get("is_agentic_system", True):
                candidate_profile["name"] = "unknown"
                candidate_profile["confidence"] = "low"
            else:
                candidate_profile["name"] = parsed.get("name") or "unknown"
                candidate_profile["frameworks"] = parsed.get("frameworks") or cluster["frameworks"] or ["unknown"]
                candidate_profile["entry_points"] = parsed.get("entry_points") or cluster["hub_paths"][:2]
                candidate_profile["confidence"] = parsed.get("confidence") or "low"
                for item in parsed.get("systems") or []:
                    try:
                        idx = int(item.get("index"))
                    except (TypeError, ValueError):
                        continue
                    name = (item.get("name") or "").strip()
                    if name:
                        system_names[idx] = name
                    # CS-320: Collect LLM's system_type from the schema (optional, validated)
                    stype = item.get("system_type")
                    if stype and stype in ALL_SYSTEM_TYPES:
                        system_types[idx] = stype
                system_names = _dedupe_system_names(system_names, systems)
        except RateLimitExceeded as rle:
            raise DiscoveryPaused(
                candidates, rle.provider_id, rle.model_id, rle.reasoning_effort, rle.thinking_budget
            ) from rle
        except Exception as exc:
            logger.warning(f"LLM synthesis failed for cluster {cid}: {exc}. Using fallback.")

        candidate_profile["needs_human"] = (
            candidate_profile["name"] == "unknown" or candidate_profile["confidence"] == "low"
        )

        # Risk flag propagation from analysis to candidate profile
        if project_context and project_context.risk_findings:
            hub_path_set = set(cluster["hub_paths"])
            for finding in project_context.risk_findings:
                if hub_path_set.intersection(finding.get("evidence", [])):
                    candidate_profile.setdefault("risk_flags", []).append(finding)
                    candidate_profile["needs_human"] = True

        # The residual candidate is the fallback for a plain-python agent that produced NO wiring.
        # Now that _detect_plain_python gives such agents a real call-flow block, skip the residual
        # when a plain_python system already exists — otherwise the community emits two plain_python
        # candidates (one wired QAAgent-style, one wireless leftover bucket).
        has_plain_python = any(s.framework == "plain_python" for s in systems)
        residual = (
            _build_residual_candidate(candidate_profile, cluster, systems)
            if (systems and not has_plain_python)
            else None
        )
        n_systems = len(systems) + (1 if residual else 0)
        if n_systems > 1:
            # Genuine multi-system community: split into one candidate per system.
            for i, block in enumerate(systems):
                cand = _build_system_candidate(
                    candidate_profile, cluster, block, system_names.get(i + 1)
                )
                # CS-320: Apply LLM's optional system_type override if structural confidence is low
                struct = cand.get("system_type_signals", {})
                if struct.get("confidence") == "low" and (i + 1) in system_types:
                    cand["system_type"] = system_types[i + 1]
                    struct["decided_by"] = DECIDED_BY_LLM
                else:
                    struct["decided_by"] = DECIDED_BY_STRUCTURAL
                cand["system_type_signals"] = struct
                candidates.append(cand)
            if residual is not None:
                candidates.append(residual)
        elif systems:
            # Exactly one static system, no residual: behave IDENTICALLY to the pre-split single
            # candidate — base name/profile kept, no system_id, just its own pure wiring_block.
            candidate_profile["wiring_block"] = systems[0].to_dict()
            # CS-320: Classify the single system
            struct = classify_system_type(systems[0])
            candidate_profile["system_type"] = struct.get("type")
            candidate_profile["system_type_signals"] = struct
            # Apply LLM override if structural confidence is low
            if struct.get("confidence") == "low" and 1 in system_types:
                candidate_profile["system_type"] = system_types[1]
                struct["decided_by"] = DECIDED_BY_LLM
            else:
                struct["decided_by"] = DECIDED_BY_STRUCTURAL
            candidates.append(candidate_profile)
        else:
            # No static wiring in this community — preserve today's EXACT single-candidate path,
            # including the LLM wiring fallback for candidates worth naming.
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
            # CS-320: Classify if we have a wiring block
            if wiring_block:
                struct = classify_system_type(wiring_block)
                candidate_profile["system_type"] = struct.get("type")
                candidate_profile["system_type_signals"] = struct
                struct["decided_by"] = DECIDED_BY_STRUCTURAL
            else:
                # No wiring block: low-confidence None type
                candidate_profile["system_type"] = None
                candidate_profile["system_type_signals"] = {
                    "kind": "agent", "type": None, "confidence": "low", "candidate_types": [],
                    "capability_tags": [], "signals": {}, "decided_by": DECIDED_BY_UNRESOLVED
                }

            candidates.append(candidate_profile)

    try:
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
