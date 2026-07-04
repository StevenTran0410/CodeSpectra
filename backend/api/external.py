"""External-agent-facing API (CS-235 narrow slice, for AEH's CodeSpectraProxyClient).

Not full CS-235: no external_call_log table, no generalized multi-endpoint auth
framework — just a bearer-token-gated LLM passthrough so an external harness
(starting with AEH) can reuse whatever provider the user already configured
here, without AEH ever holding a provider API key of its own.
"""
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

from domain.model_connector.service import ProviderConfigService
from domain.model_connector.types import ChatMessage, ChatRequest, ChatResponse
from infrastructure.db.database import get_db

router = APIRouter(tags=["external"])
_service = ProviderConfigService()


class LLMCompleteRequest(BaseModel):
    provider_id: str
    model_id: str | None = None
    messages: list[ChatMessage]
    max_completion_tokens: int = 2048
    temperature: float | None = 0.2
    json_mode: bool = False


class ProviderSummary(BaseModel):
    provider_id: str
    display_name: str
    model_id: str


async def _get_external_token() -> str | None:
    import os

    env_token = os.getenv("CODESPECTRA_EXTERNAL_TOKEN")
    if env_token:
        return env_token
    db = get_db()
    async with db.execute(
        "SELECT value FROM app_metadata WHERE key='external_api_token'"
    ) as cur:
        row = await cur.fetchone()
    return row["value"] if row else None


async def require_external_token(authorization: str = Header(default="")) -> None:
    expected = await _get_external_token()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="external API token not configured — set CODESPECTRA_EXTERNAL_TOKEN "
            "or an 'external_api_token' app_metadata row",
        )
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")


@router.post(
    "/llm/complete",
    response_model=ChatResponse,
    dependencies=[Depends(require_external_token)],
)
async def llm_complete(body: LLMCompleteRequest) -> ChatResponse:
    return await _service.chat(
        ChatRequest(
            provider_id=body.provider_id,
            model_id=body.model_id,
            messages=body.messages,
            max_completion_tokens=body.max_completion_tokens,
            temperature=body.temperature,
            json_mode=body.json_mode,
            stream=False,
        )
    )


@router.get(
    "/llm/providers",
    response_model=list[ProviderSummary],
    dependencies=[Depends(require_external_token)],
)
async def list_llm_providers() -> list[ProviderSummary]:
    configs = await _service.list_all()
    return [
        ProviderSummary(provider_id=c.id, display_name=c.display_name, model_id=c.model_id)
        for c in configs
    ]
