import importlib.metadata

from fastapi import APIRouter
from pydantic import BaseModel

from infrastructure.db.database import get_db

router = APIRouter(tags=["app"])


class VersionResponse(BaseModel):
    status: str
    version: str


class GitConfig(BaseModel):
    ssh_key_path: str | None = None


@router.get("/health", response_model=VersionResponse)
async def health() -> VersionResponse:
    try:
        version = importlib.metadata.version("codespectra-backend")
    except importlib.metadata.PackageNotFoundError:
        version = "dev"
    return VersionResponse(status="ok", version=version)


@router.get("/git-config", response_model=GitConfig)
async def get_git_config() -> GitConfig:
    db = get_db()
    async with db.execute(
        "SELECT value FROM app_metadata WHERE key='git_ssh_key_path'"
    ) as cur:
        row = await cur.fetchone()
    return GitConfig(ssh_key_path=row["value"] if row else None)


@router.put("/git-config", response_model=GitConfig)
async def set_git_config(body: GitConfig) -> GitConfig:
    db = get_db()
    if body.ssh_key_path:
        await db.execute(
            "INSERT OR REPLACE INTO app_metadata (key, value) VALUES ('git_ssh_key_path', ?)",
            (body.ssh_key_path,),
        )
    else:
        await db.execute("DELETE FROM app_metadata WHERE key='git_ssh_key_path'")
    await db.commit()
    return body
