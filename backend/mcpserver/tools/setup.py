"""setup_project — index a local project using existing CodeSpectra services."""
from __future__ import annotations

import json
import time
from pathlib import Path

from mcp.server.fastmcp import Context, FastMCP

from shared.logger import logger
from ..project_index import save_meta, _meta_path


def register(mcp: FastMCP) -> None:

    @mcp.tool()
    async def setup_project(
        project_path: str,
        force_reindex: bool = False,
        ctx: Context | None = None,
    ) -> dict:
        """Index a local project so CodeSpectra tools can query it.
        Creates .codespectra/ in the project root.
        No native C++ module required — all steps have Python fallbacks.
        May take 30-120 seconds for large codebases."""

        resolved = str(Path(project_path).resolve())
        if not Path(resolved).is_dir():
            return {"error": "invalid_path", "message": f"Not a directory: {resolved}"}

        start_ms = int(time.time() * 1000)

        async def _progress(msg: str):
            logger.info("[setup_project] %s", msg)
            if ctx:
                try:
                    await ctx.info(msg)
                except Exception:
                    pass  # ctx.info may not be available in all transports

        # Already indexed?
        meta_file = _meta_path(resolved)
        if meta_file.exists() and not force_reindex:
            meta = json.loads(meta_file.read_text())
            return {
                "status": "already_indexed",
                "project_path": resolved,
                "snapshot_id": meta["snapshot_id"],
                "message": "Already indexed. Use force_reindex=True to rebuild.",
            }

        # --- Call existing services ---
        # Import here to avoid circular imports at module load
        from domain.local_repo.service import LocalRepoService
        from domain.local_repo.types import AddLocalRepoRequest
        from domain.sync_engine.service import SyncEngineService
        from domain.sync_engine.types import PrepareSnapshotRequest
        from domain.manifest.service import ManifestService
        from domain.manifest.types import BuildManifestRequest
        from domain.repo_map.service import RepoMapService
        from domain.repo_map.types import BuildRepoMapRequest
        from domain.structural_graph.service import StructuralGraphService
        from domain.structural_graph.types import BuildGraphRequest
        from domain.retrieval.service import RetrievalService
        from domain.retrieval.types import BuildRetrievalIndexRequest

        # Step 1: Add/find local repo
        await _progress("Registering project...")
        repo = await LocalRepoService().add(AddLocalRepoRequest(path=resolved))

        # Step 2: Create snapshot
        await _progress("Creating snapshot...")
        snapshot = await SyncEngineService().prepare_snapshot(
            PrepareSnapshotRequest(local_repo_id=repo.id)
        )
        sid = snapshot.id
        await _progress(f"Snapshot: {sid}")

        # Step 3: Build manifest
        await _progress("Scanning files...")
        manifest_r = await ManifestService().build(BuildManifestRequest(snapshot_id=sid))
        await _progress(f"Files: {manifest_r.total_files}")

        if manifest_r.total_files == 0:
            save_meta(resolved, sid, repo.id)
            return {
                "status": "ok",
                "project_path": resolved,
                "files_indexed": 0,
                "message": "No indexable files found.",
            }

        # Step 4: Extract symbols
        await _progress("Extracting symbols...")
        symbols_r = await RepoMapService().build(BuildRepoMapRequest(snapshot_id=sid))
        await _progress(f"Symbols: {symbols_r.summary.total_symbols}")

        # Step 5: Build structural graph
        await _progress("Building dependency graph...")
        graph_r = await StructuralGraphService().build(BuildGraphRequest(snapshot_id=sid))
        await _progress(
            f"Graph: {graph_r.summary.total_nodes} nodes, "
            f"{graph_r.summary.total_edges} edges"
        )

        # Step 6: Build retrieval index
        await _progress("Building search index...")
        retrieval_r = await RetrievalService().build_index(
            BuildRetrievalIndexRequest(snapshot_id=sid)
        )
        await _progress(f"Chunks: {retrieval_r.chunk_count}")

        # Save marker
        save_meta(resolved, sid, repo.id)

        elapsed_ms = int(time.time() * 1000) - start_ms
        return {
            "status": "ok",
            "project_path": resolved,
            "snapshot_id": sid,
            "files_indexed": manifest_r.total_files,
            "symbols_extracted": symbols_r.summary.total_symbols,
            "graph_nodes": graph_r.summary.total_nodes,
            "graph_edges": graph_r.summary.total_edges,
            "chunks_indexed": retrieval_r.chunk_count,
            "elapsed_ms": elapsed_ms,
        }
