"""Cloud consent management — stored as a flag in app_metadata."""
from fastapi import APIRouter
from pydantic import BaseModel

from infrastructure.db.database import get_db

router = APIRouter(tags=["consent"])

_KEY = "cloud_consent_given"


class ConsentStatus(BaseModel):
    given: bool


class GiveConsentRequest(BaseModel):
    given: bool


@router.get("/cloud", response_model=ConsentStatus)
async def get_cloud_consent() -> ConsentStatus:
    db = get_db()
    async with db.execute(
        "SELECT value FROM app_metadata WHERE key = ?", (_KEY,)
    ) as cur:
        row = await cur.fetchone()
    return ConsentStatus(given=row is not None and row["value"] == "true")


@router.post("/cloud", response_model=ConsentStatus)
async def give_cloud_consent(body: GiveConsentRequest) -> ConsentStatus:
    db = get_db()
    value = "true" if body.given else "false"
    await db.execute(
        "INSERT OR REPLACE INTO app_metadata (key, value) VALUES (?, ?)",
        (_KEY, value),
    )
    await db.commit()
    return ConsentStatus(given=body.given)
