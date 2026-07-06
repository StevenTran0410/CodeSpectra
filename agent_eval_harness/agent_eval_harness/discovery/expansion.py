import json
import logging
from agent_eval_harness.llm.client import LLMClient, LLMMessage
from agent_eval_harness.discovery.client import CodeSpectraClient

logger = logging.getLogger("agent_eval_harness.discovery.expansion")


async def _classify_node(path: str, content: str, llm_client: LLMClient) -> str:
    """Classify node as 'accept' | 'boundary' | 'expand' using the LLM."""
    system_prompt = (
        "You are an expert AI software architect. Classify the provided Python source file from a target codebase.\n"
        "Determine if this file is:\n"
        "1. \"accept\" - represents a core agentic component (agent loops, tool orchestrators, custom agents using LangChain/LangGraph/Haystack/CrewAI, prompt template definitions, main entry points).\n"
        "2. \"boundary\" - represents pure non-agentic infrastructure, library utility, database helper, logging, test files, third-party libraries, or configuration. We should not expand its neighbors.\n"
        "3. \"expand\" - represents an intermediary connector module that bridges or imports agentic systems (e.g. API routers, connector files, shared models). We should accept it and continue expanding to its neighbors.\n\n"
        "Respond ONLY in raw JSON format matching this schema:\n"
        "{\n"
        "  \"verdict\": \"accept\" | \"boundary\" | \"expand\",\n"
        "  \"reason\": \"brief explanation\"\n"
        "}"
    )
    user_prompt = f"File path: {path}\nFile content:\n{content}\n\nClassify this node."

    try:
        response = await llm_client.complete(
            [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ],
            json_mode=True,
        )
        data = json.loads(response.content)
        verdict = data.get("verdict", "boundary")
        if verdict not in ("accept", "boundary", "expand"):
            verdict = "boundary"
        return verdict
    except Exception as e:
        logger.warning(f"Failed to classify node {path}: {e}")
        return "boundary"


async def expand_candidate(
    snapshot_id: str,
    candidate: dict,  # a discovery_candidates row (dict), has cluster_files/hub_paths/entry_points
    client: CodeSpectraClient,
    llm_client: LLMClient,
    *,
    node_budget: int = 40,
    hop_cap: int = 3,
) -> dict:  # {"accepted": [...], "boundary": [...], "stop_reason": str}
    seeds = set(candidate.get("cluster_files", [])) | set(candidate.get("hub_paths", []))
    frontier = sorted(seeds)
    accepted: set[str] = set()
    boundary: set[str] = set()
    visited: set[str] = set()
    hops_from_seed = {p: 0 for p in frontier}

    while frontier:
        if len(accepted) >= node_budget:
            return {"accepted": sorted(accepted), "boundary": sorted(boundary), "stop_reason": "node_budget"}
        path = frontier.pop(0)  # deterministic FIFO order
        if path in visited:
            continue
        visited.add(path)

        try:
            file_resp = await client.read_file(snapshot_id, path)
            content = file_resp.get("content", "")
        except Exception as e:
            logger.warning(f"Failed to read file {path}: {e}")
            content = ""

        verdict = await _classify_node(path, content, llm_client)

        if verdict == "boundary":
            boundary.add(path)
            continue

        accepted.add(path)
        if verdict == "expand" and hops_from_seed.get(path, 0) < hop_cap:
            try:
                neighbors = await client.get_neighbors(snapshot_id, path, hops=1)
                # GraphNeighborsResponse: nodes is under "nodes" key
                for n_path in sorted(neighbors.get("nodes", [])):
                    if n_path and n_path not in visited:
                        frontier.append(n_path)
                        hops_from_seed[n_path] = hops_from_seed.get(path, 0) + 1
            except Exception as e:
                logger.warning(f"Failed to get neighbors for {path}: {e}")

    return {"accepted": sorted(accepted), "boundary": sorted(boundary), "stop_reason": "frontier_exhausted"}
