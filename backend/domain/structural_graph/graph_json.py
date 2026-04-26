"""Persistent graph.json serialization for structural graphs (RPA-058 Part B).

Schema version: 1.0
  nodes: list of {path, type, indegree, outdegree, score, community_id}
  edges: list of {src, tgt, type, provenance, confidence}
  communities: list of {id, label, member_paths}

Note: community_id on nodes may be null if community detection has not completed.
Community detection runs as a background task after graph build.
"""

import asyncio
import json
from pathlib import Path

from shared.logger import logger

GRAPH_JSON_SCHEMA_VERSION = "1.0"


def graph_json_path(snapshot_id: str, data_dir: str) -> Path:
    """Canonical path: <data_dir>/graphs/<snapshot_id>/graph.json. Pure."""
    return Path(data_dir) / "graphs" / snapshot_id / "graph.json"


def build_graph_json_payload(nodes: list, edges: list, communities: list | None = None) -> dict:
    """Build payload dict from service types. Pure — no I/O."""
    if communities is None:
        communities = []

    return {
        "schema_version": GRAPH_JSON_SCHEMA_VERSION,
        "nodes": nodes,
        "edges": edges,
        "communities": communities,
    }


async def write_graph_json(snapshot_id: str, data_dir: str, payload: dict) -> Path:
    """Write graph.json to disk via asyncio.to_thread (non-blocking).

    Creates parent directories. Returns path written.
    """

    async def _write_sync() -> Path:
        path = graph_json_path(snapshot_id, data_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return path

    try:
        path = await asyncio.to_thread(_write_sync)
        logger.info(f"Wrote graph.json for snapshot {snapshot_id} to {path}")
        return path
    except Exception as e:
        logger.warning(f"Failed to write graph.json for snapshot {snapshot_id}: {e}")
        raise


def read_graph_json(snapshot_id: str, data_dir: str) -> dict | None:
    """Read graph.json from disk. Returns None if file does not exist."""
    path = graph_json_path(snapshot_id, data_dir)
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read graph.json for snapshot {snapshot_id}: {e}")
        return None
