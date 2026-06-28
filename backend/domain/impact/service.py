"""ImpactService — blast_radius and plan endpoints (RPA-060)."""
from __future__ import annotations

import json

from infrastructure.db.database import get_db

from domain.retrieval.impact_retrieval import retrieve_impact
from domain.retrieval.two_stage_retrieval import _load_graph_context

from .git_signals import get_cochange_files
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

        # Git-history co-change hint (CS-246) — additive, does not replace the
        # graph-based cone. Best-effort: empty dict if repo path is unavailable
        # or git fails.
        cochange_hints = await _load_cochange_hints(body.snapshot_id, body.changed_files)

        # Convention-violation wiring (CS-246) — Section E findings from an
        # existing analysis report, filtered to files in the impact cone.
        convention_violations = await _load_convention_violations(body.report_id, cone_files)
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
            "cochange_hints": cochange_hints,
            "convention_violations": convention_violations,
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

        # Convention-violation wiring (CS-246): Section E findings from the
        # report, filtered to the files this plan suggests touching.
        violations_to_avoid = await _load_convention_violations(body.report_id, set(seen_paths))

        return PlanResponse(
            task=body.task_description,
            suggested_files=suggested_files,
            patterns_to_follow=[],
            violations_to_avoid=violations_to_avoid,
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


async def _load_cochange_hints(snapshot_id: str, seed_files: list[str]) -> dict[str, list[dict]]:
    """Git-history co-change signal (CS-246). Best-effort: {} if snapshot has
    no resolvable local_path or git fails for any reason."""
    db = get_db()
    async with db.execute(
        "SELECT local_path FROM repo_snapshots WHERE id=?", (snapshot_id,)
    ) as cur:
        row = await cur.fetchone()
    if row is None or not row["local_path"]:
        return {}
    return await get_cochange_files(row["local_path"], seed_files)


async def _load_convention_violations(report_id: str | None, files: set[str]) -> list[dict]:
    """Pulls Section E (`violations_found`) from an existing analysis report
    and filters to entries whose `file` matches one of `files` (CS-246).

    Best-effort: returns [] if report_id is absent, the report doesn't exist,
    or Section E has no findings — this is an additive hint on top of the
    already-shipped impact cone, not a hard dependency.
    """
    if not report_id or not files:
        return []
    db = get_db()
    async with db.execute(
        "SELECT report_json FROM analysis_reports WHERE id=?", (report_id,)
    ) as cur:
        row = await cur.fetchone()
    if row is None or not row["report_json"]:
        return []
    try:
        raw = json.loads(row["report_json"])
    except Exception:
        return []

    sections = raw.get("sections") or raw.get("sections_v2") or {}
    section_e = sections.get("E") or {}
    violations = section_e.get("violations_found")
    if not isinstance(violations, list):
        return []

    out: list[dict] = []
    for v in violations:
        if not isinstance(v, dict):
            continue
        file_field = str(v.get("file", "") or "")
        if not file_field:
            continue
        # `file` is freeform (LLM-authored); match by suffix/substring against
        # known rel_paths since exact equality is unreliable.
        if any(f == file_field or f.endswith(file_field) or file_field.endswith(f) for f in files):
            out.append(v)
    return out
