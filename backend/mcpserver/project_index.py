"""Manages .codespectra/meta.json: project <-> snapshot_id mapping."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from shared.logger import logger


class ProjectNotIndexedError(Exception):
    def __init__(self, project_path: str):
        self.project_path = project_path
        super().__init__(
            f"Project not indexed. Run setup_project('{project_path}') first."
        )


def _meta_path(project_path: str) -> Path:
    return Path(project_path).resolve() / ".codespectra" / "meta.json"


def assert_indexed(project_path: str) -> dict:
    """Raise ProjectNotIndexedError if not indexed. Returns meta dict."""
    mp = _meta_path(project_path)
    if not mp.exists():
        raise ProjectNotIndexedError(project_path)
    return json.loads(mp.read_text())


def get_snapshot_id(project_path: str) -> str:
    """Return snapshot_id for an indexed project."""
    return assert_indexed(project_path)["snapshot_id"]


def save_meta(project_path: str, snapshot_id: str, repo_id: str) -> None:
    """Write .codespectra/meta.json after successful indexing."""
    meta_dir = Path(project_path).resolve() / ".codespectra"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "snapshot_id": snapshot_id,
        "repo_id": repo_id,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }
    (meta_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("[MCP] Saved meta for %s -> snapshot %s", project_path, snapshot_id)
