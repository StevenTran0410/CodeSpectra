import json

from infrastructure.db.database import get_db
from shared.errors import ConflictError, NotFoundError
from shared.logger import logger
from shared.utils import new_id, utc_now_iso

from .types import Workspace


class WorkspaceService:
    async def list_all(self) -> list[Workspace]:
        db = get_db()
        async with db.execute(
            "SELECT id, name, description, created_at, updated_at, settings FROM workspaces ORDER BY created_at ASC"
        ) as cur:
            rows = await cur.fetchall()
        return [self._row_to_workspace(r) for r in rows]

    async def get_by_id(self, workspace_id: str) -> Workspace:
        db = get_db()
        async with db.execute(
            "SELECT id, name, description, created_at, updated_at, settings FROM workspaces WHERE id = ?",
            (workspace_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            raise NotFoundError("Workspace", workspace_id)
        return self._row_to_workspace(row)

    async def create(self, name: str, description: str | None = None) -> Workspace:
        db = get_db()

        # Check for duplicate name
        async with db.execute(
            "SELECT 1 FROM workspaces WHERE name = ?", (name,)
        ) as cur:
            existing = await cur.fetchone()
        if existing:
            raise ConflictError(f"A workspace named '{name}' already exists")

        ws_id = new_id()
        now = utc_now_iso()
        await db.execute(
            "INSERT INTO workspaces (id, name, description, created_at, updated_at, settings) VALUES (?, ?, ?, ?, ?, ?)",
            (ws_id, name, description, now, now, "{}"),
        )
        await db.commit()
        logger.info(f"Created workspace '{name}' ({ws_id})")
        return Workspace(id=ws_id, name=name, description=description, created_at=now, updated_at=now, settings={})

    async def rename(self, workspace_id: str, new_name: str) -> Workspace:
        db = get_db()

        async with db.execute(
            "SELECT 1 FROM workspaces WHERE id = ?", (workspace_id,)
        ) as cur:
            if not await cur.fetchone():
                raise NotFoundError("Workspace", workspace_id)

        async with db.execute(
            "SELECT 1 FROM workspaces WHERE name = ? AND id != ?", (new_name, workspace_id)
        ) as cur:
            if await cur.fetchone():
                raise ConflictError(f"A workspace named '{new_name}' already exists")

        now = utc_now_iso()
        await db.execute(
            "UPDATE workspaces SET name = ?, updated_at = ? WHERE id = ?",
            (new_name, now, workspace_id),
        )
        await db.commit()
        logger.info(f"Renamed workspace {workspace_id} → '{new_name}'")
        return await self.get_by_id(workspace_id)

    async def delete(self, workspace_id: str) -> None:
        db = get_db()

        # Verify workspace exists before cascade
        async with db.execute(
            "SELECT 1 FROM workspaces WHERE id = ?", (workspace_id,)
        ) as cur:
            if not await cur.fetchone():
                raise NotFoundError("Workspace", workspace_id)

        # Explicit cascade: delete analysis_reports for jobs linked to repos in this workspace.
        # TODO(CS-020-follow-up): local_repos has no workspace_id FK so repo rows and their
        # downstream children (snapshots, manifest_files, code_symbols, graph tables,
        # retrieval tables) are NOT deleted here. A follow-up migration must add
        # workspace_id to local_repos before full cascade can be implemented.
        await db.execute(
            """
            DELETE FROM analysis_reports
            WHERE job_id IN (
                SELECT id FROM jobs WHERE repo_id IN (
                    SELECT id FROM local_repos
                )
            )
            """
        )

        await db.execute("DELETE FROM workspaces WHERE id = ?", (workspace_id,))
        await db.commit()
        logger.info(f"Deleted workspace {workspace_id}")

    @staticmethod
    def _row_to_workspace(row) -> Workspace:
        settings_raw = row["settings"] or "{}"
        settings = json.loads(settings_raw) if isinstance(settings_raw, str) else settings_raw
        return Workspace(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            settings=settings,
        )
