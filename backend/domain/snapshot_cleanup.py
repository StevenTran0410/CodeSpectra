"""Shared cleanup helpers for snapshot/repo deletion."""

from infrastructure.db.database import get_db


async def delete_snapshot_artifacts(snapshot_id: str) -> None:
    """Delete all DB artifacts tied to one snapshot."""
    db = get_db()
    await db.execute("DELETE FROM manifest_files WHERE snapshot_id=?", (snapshot_id,))
    await db.execute("DELETE FROM code_symbols WHERE snapshot_id=?", (snapshot_id,))
    await db.execute("DELETE FROM repo_maps WHERE snapshot_id=?", (snapshot_id,))
    await db.execute("DELETE FROM structural_graph_edges WHERE snapshot_id=?", (snapshot_id,))
    await db.execute("DELETE FROM structural_graph_summaries WHERE snapshot_id=?", (snapshot_id,))
    await db.execute("DELETE FROM retrieval_chunks WHERE snapshot_id=?", (snapshot_id,))
    await db.execute("DELETE FROM retrieval_indexes WHERE snapshot_id=?", (snapshot_id,))
    await db.execute("DELETE FROM retrieval_bm25_stats WHERE snapshot_id=?", (snapshot_id,))
    await db.execute("DELETE FROM analysis_reports WHERE snapshot_id=?", (snapshot_id,))


async def delete_repo_artifacts(repo_id: str) -> None:
    """Delete all DB artifacts tied to one local repository."""
    db = get_db()
    async with db.execute(
        "SELECT id FROM repo_snapshots WHERE local_repo_id=?",
        (repo_id,),
    ) as cur:
        rows = await cur.fetchall()

    for r in rows:
        await delete_snapshot_artifacts(r["id"])

    await db.execute("DELETE FROM repo_snapshots WHERE local_repo_id=?", (repo_id,))
    await db.execute("DELETE FROM analysis_reports WHERE repo_id=?", (repo_id,))
    await db.execute("DELETE FROM jobs WHERE repo_id=?", (repo_id,))
