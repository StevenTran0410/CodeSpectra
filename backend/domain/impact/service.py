"""ImpactService — blast_radius and plan endpoints (RPA-060)."""
from __future__ import annotations

from domain.retrieval.impact_retrieval import retrieve_impact
from domain.retrieval.two_stage_retrieval import _load_graph_context

from .types import (
    BlastRadiusRequest,
    BlastRadiusResponse,
    PlanRequest,
    PlanResponse,
)

# Hop distance -> hex color for subgraph visualization
_HOP_COLORS: dict[int, str] = {
    0: "#ef4444",  # red — seed
    1: "#f97316",  # orange
    2: "#eab308",  # yellow
    3: "#6b7280",  # gray
    4: "#6b7280",  # gray
}
_DEFAULT_NODE_COLOR = "#52525b"


class ImpactService:
    async def blast_radius(self, body: BlastRadiusRequest) -> BlastRadiusResponse:
        bundle = await retrieve_impact(
            snapshot_id=body.snapshot_id,
            seed_files=body.changed_files,
            query=None,
            max_hops=body.max_hops,
            budget=8000,
        )

        cone = bundle.impact_cone
        cone_files = set(cone.keys()) | set(body.changed_files)

        # Build subgraph: filter edge_tuples to edges where both endpoints are in cone
        ctx = await _load_graph_context(body.snapshot_id)
        subgraph_edges = [
            {"src": src, "dst": dst, "edge_type": etype, "is_external": bool(is_ext)}
            for src, dst, etype, is_ext in ctx.edge_tuples
            if src in cone_files and dst in cone_files and not is_ext
        ]

        # Build hop_colors mapping for frontend
        hop_colors: dict[str, str] = {}
        for f in cone_files:
            hop = cone.get(f, 0)
            hop_colors[f] = _HOP_COLORS.get(hop, _DEFAULT_NODE_COLOR)

        # Subgraph node list (deduplicated)
        subgraph_nodes = sorted(cone_files)

        # High-risk files: cone ∩ central_files
        high_risk_files = sorted(f for f in cone_files if f in ctx.central_files)

        # Assemble blast_radius payload
        blast_radius_payload = {
            "total_affected": bundle.risk_summary.get("total_affected", len(cone_files)),
            "by_hop": bundle.risk_summary.get("by_hop", {}),
            "high_risk_files": high_risk_files,
            "affected_communities": [c.model_dump() for c in bundle.affected_communities],
            "call_chains": bundle.call_chains if body.include_call_chains else [],
            "seed_community_ids": bundle.risk_summary.get("seed_community_ids", []),
        }

        subgraph_payload = {
            "nodes": subgraph_nodes,
            "edges": subgraph_edges,
            "seed_files": body.changed_files,
            "hop_colors": hop_colors,
        }

        context_chunks = [c.model_dump() for c in bundle.ranked_chunks]

        return BlastRadiusResponse(
            changed_files=body.changed_files,
            blast_radius=blast_radius_payload,
            subgraph=subgraph_payload,
            context_chunks=context_chunks,
        )

    async def plan(self, body: PlanRequest) -> PlanResponse:
        # v1: structured data only — no LLM synthesis (deferred to RPA-061)
        bundle = await retrieve_impact(
            snapshot_id=body.snapshot_id,
            seed_files=[],
            query=body.task_description,
            budget=8000,
        )

        # Suggested files: top-ranked chunks by score, deduplicated by rel_path
        seen_paths: set[str] = set()
        suggested_files: list[dict] = []
        for chunk in bundle.ranked_chunks:
            if chunk.rel_path not in seen_paths:
                seen_paths.add(chunk.rel_path)
                suggested_files.append({
                    "rel_path": chunk.rel_path,
                    "score": round(chunk.score, 3),
                    "reason": _chunk_reason(chunk),
                })
                if len(suggested_files) >= 20:
                    break

        # Important central files that should be considered
        important_files_affected: list[dict] = [
            {"rel_path": c.rel_path, "score": round(c.score, 3)}
            for c in bundle.ranked_chunks
            if c.centrality_bonus > 0
        ][:10]

        # Communities touched by top-scoring chunks
        related_features: list[dict] = [
            {"community_id": c.community_id, "member_count": c.member_count, "hub_paths": c.hub_paths}
            for c in bundle.affected_communities[:5]
        ]

        return PlanResponse(
            task=body.task_description,
            suggested_files=suggested_files,
            patterns_to_follow=[],
            violations_to_avoid=[],
            related_features=related_features,
            important_files_affected=important_files_affected,
            tests_to_add=[],
            blast_radius_preview=bundle.risk_summary if bundle.impact_cone else None,
        )


def _chunk_reason(chunk) -> str:
    reasons: list[str] = []
    if chunk.hop_distance is not None:
        reasons.append(f"hop-{chunk.hop_distance}")
    if chunk.symbol_bonus > 0:
        reasons.append("call-chain")
    if chunk.centrality_bonus > 0:
        reasons.append("central")
    if chunk.community_bonus > 0:
        reasons.append("community")
    if chunk.bm25_boost > 0:
        reasons.append("bm25")
    return ",".join(reasons) if reasons else "graph"
