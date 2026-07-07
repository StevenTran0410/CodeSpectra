import ast
import json
import logging
from typing import Any
from agent_eval_harness.llm.client import LLMClient, LLMMessage
from agent_eval_harness.discovery.client import CodeSpectraClient

logger = logging.getLogger("agent_eval_harness.discovery.expansion")


def extract_symbol_snippet(content: str, symbol_identifier: str) -> str:
    """
    Parse content using AST, find the ClassDef/FunctionDef matching symbol_identifier
    (e.g., 'MyClass.run', 'MyClass', or 'my_function').
    Returns a snippet of the matched node, or empty string.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return ""

    lines = content.splitlines()

    parts = symbol_identifier.split(".")
    class_name = None
    func_name = None
    if len(parts) == 2:
        class_name, func_name = parts[0], parts[1]
    elif len(parts) == 1:
        class_name = parts[0]
    else:
        return ""

    matched_node = None

    if class_name and func_name:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for sub_node in node.body:
                    if isinstance(sub_node, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub_node.name == func_name:
                        matched_node = sub_node
                        break
                if matched_node:
                    break
    elif class_name:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                matched_node = node
                break
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == class_name:
                matched_node = node
                break

    if not matched_node:
        return ""

    start_line = matched_node.lineno - 1
    end_line = getattr(matched_node, "end_lineno", None)
    if end_line is None:
        end_line = start_line + 50
    
    snippet_lines = lines[start_line:end_line]
    if len(snippet_lines) > 100:
        snippet_lines = snippet_lines[:100]
    return "\n".join(snippet_lines)


async def _classify_node(path: str, content: str, llm_client: LLMClient, candidate: dict) -> str:
    """Classify node as 'accept' | 'boundary' | 'expand' using the LLM."""
    name = candidate.get("name", "unknown")
    frameworks = ", ".join(candidate.get("frameworks", []))

    system_prompt = (
        "You are an expert AI software architect. Classify the provided Python source code chunk from a target codebase.\n"
        f"The candidate agentic system we are expanding is named \"{name}\" and uses frameworks: [{frameworks}].\n"
        "Determine if this code chunk is:\n"
        "1. \"accept\" - represents a core agentic component (agent loops, tool orchestrators, custom agents, prompt templates, main entry points).\n"
        "2. \"boundary\" - represents pure non-agentic infrastructure, library utility, database helper, logging, test files, third-party libraries, or configuration. We should not expand its neighbors.\n"
        "3. \"expand\" - represents an intermediary connector module that bridges or imports agentic systems (e.g. API routers, connector files, shared models). We should accept it and continue expanding to its neighbors.\n\n"
        "Respond ONLY in raw JSON format matching this schema:\n"
        "{\n"
        "  \"verdict\": \"accept\" | \"boundary\" | \"expand\",\n"
        "  \"reason\": \"brief explanation\"\n"
        "}"
    )
    user_prompt = f"File path: {path}\nCode chunk:\n{content}\n\nClassify this chunk."

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
    candidate: dict,
    client: CodeSpectraClient,
    llm_client: LLMClient,
    *,
    node_budget: int = 40,
    hop_cap: int = 3,
) -> dict:
    # 1. Chunks Extraction for Seeds
    seeds = []
    chunked_files = set()
    evidence_list = candidate.get("evidence", [])
    excluded = set(candidate.get("excluded_files", []))
    for ev in evidence_list:
        file_path = ev.get("file")
        if file_path in excluded:
            continue
        chunk_id = ev.get("chunk_id")
        snippet = ev.get("snippet")
        if file_path and chunk_id:
            seeds.append((file_path, chunk_id, snippet))
            chunked_files.add(file_path)

    # For files in cluster_files or hub_paths that don't have chunk evidence, add them as whole-file fallbacks (chunk_id = None)
    if candidate.get("matched_files"):
        fallback_files = set(candidate["matched_files"]) - excluded
    else:
        fallback_files = (set(candidate.get("cluster_files", [])) | set(candidate.get("hub_paths", []))) - excluded
    for f in sorted(fallback_files):
        if f not in chunked_files:
            seeds.append((f, None, None))

    # Frontier elements: list of (file_path, chunk_id, snippet)
    frontier = seeds[:]
    accepted_chunks = set()
    accepted_files = set()
    boundary_files = set()
    visited_chunks = set()
    
    # hops_from_seed tracks distance at (file_path, chunk_id) granularity
    hops_from_seed = {}
    for f_path, c_id, _ in frontier:
        hops_from_seed[(f_path, c_id)] = 0

    while frontier:
        if len(accepted_chunks) >= node_budget:
            return {
                "accepted": sorted(list(accepted_files)),
                "boundary": sorted(list(boundary_files - accepted_files)),
                "stop_reason": "node_budget"
            }
            
        file_path, chunk_id, snippet = frontier.pop(0)
        chunk_key = (file_path, chunk_id)
        if chunk_key in visited_chunks:
            continue
        visited_chunks.add(chunk_key)

        # Get content snippet
        if snippet is not None:
            content = snippet
        else:
            try:
                file_resp = await client.read_file(snapshot_id, file_path)
                full_content = file_resp.get("content", "")
            except Exception as e:
                logger.warning(f"Failed to read file {file_path}: {e}")
                full_content = ""

            if chunk_id is not None:
                content = extract_symbol_snippet(full_content, chunk_id)
                if not content:
                    content = full_content
            else:
                content = full_content

        # Check wiring_block hint skip
        wb = candidate.get("wiring_block")
        skip_classify = False
        if wb and isinstance(wb, dict):
            nodes = wb.get("nodes") or []
            for n in nodes:
                if n.get("source_hint_file") == file_path:
                    skip_classify = True
                    break

        if skip_classify:
            verdict = "expand"
        else:
            verdict = await _classify_node(file_path, content, llm_client, candidate)

        if verdict == "boundary":
            boundary_files.add(file_path)
            continue

        accepted_chunks.add(chunk_key)
        accepted_files.add(file_path)

        if verdict == "expand" and hops_from_seed.get(chunk_key, 0) < hop_cap:
            try:
                edges_resp = await client.get_symbol_edges(snapshot_id, file_path)
                
                for edge in edges_resp.get("outgoing", []):
                    src_sym = edge.get("src_symbol")
                    dst_sym = edge.get("dst_symbol")
                    if src_sym and dst_sym and "::" in src_sym and "::" in dst_sym:
                        sf, src_id = src_sym.split("::", 1)
                        df, dst_id = dst_sym.split("::", 1)
                        if chunk_id is None or src_id == chunk_id:
                            neighbor = (df, dst_id)
                            if neighbor not in visited_chunks:
                                frontier.append((df, dst_id, None))
                                hops_from_seed[neighbor] = hops_from_seed.get(chunk_key, 0) + 1

                for edge in edges_resp.get("incoming", []):
                    src_sym = edge.get("src_symbol")
                    dst_sym = edge.get("dst_symbol")
                    if src_sym and dst_sym and "::" in src_sym and "::" in dst_sym:
                        sf, src_id = src_sym.split("::", 1)
                        df, dst_id = dst_sym.split("::", 1)
                        if chunk_id is None or dst_id == chunk_id:
                            neighbor = (sf, src_id)
                            if neighbor not in visited_chunks:
                                frontier.append((sf, src_id, None))
                                hops_from_seed[neighbor] = hops_from_seed.get(chunk_key, 0) + 1
            except Exception as e:
                logger.warning(f"Failed to get symbol edges for {file_path}: {e}")

    return {
        "accepted": sorted(list(accepted_files)),
        "boundary": sorted(list(boundary_files - accepted_files)),
        "stop_reason": "frontier_exhausted"
    }
