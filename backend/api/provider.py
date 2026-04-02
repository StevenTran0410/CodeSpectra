from fastapi import APIRouter
from pydantic import BaseModel, field_validator

from domain.model_connector.service import ProviderConfigService
from domain.model_connector.types import ProviderCapabilities, ProviderConfig, ProviderKind

router = APIRouter(tags=["provider"])
_service = ProviderConfigService()


# ──────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ──────────────────────────────────────────────────────────────────────────────
class CreateProviderRequest(BaseModel):
    kind: ProviderKind
    display_name: str
    base_url: str
    model_id: str
    capabilities: ProviderCapabilities = ProviderCapabilities()
    extra: dict = {}

    @field_validator("display_name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("display_name cannot be empty")
        return v

    @field_validator("base_url")
    @classmethod
    def url_not_empty(cls, v: str) -> str:
        v = v.strip().rstrip("/")
        if not v:
            raise ValueError("base_url cannot be empty")
        return v


class UpdateProviderRequest(BaseModel):
    display_name: str | None = None
    base_url: str | None = None
    model_id: str | None = None
    capabilities: ProviderCapabilities | None = None
    extra: dict | None = None

    @field_validator("base_url")
    @classmethod
    def url_strip(cls, v: str | None) -> str | None:
        return v.strip().rstrip("/") if v else v


class TestConnectionResponse(BaseModel):
    ok: bool
    message: str
    warning: str | None = None


class ListModelsResponse(BaseModel):
    models: list[str]


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────
@router.get("/", response_model=list[ProviderConfig])
async def list_providers() -> list[ProviderConfig]:
    return await _service.list_all()


@router.post("/", response_model=ProviderConfig, status_code=201)
async def create_provider(body: CreateProviderRequest) -> ProviderConfig:
    return await _service.create(
        kind=body.kind,
        display_name=body.display_name,
        base_url=body.base_url,
        model_id=body.model_id,
        capabilities=body.capabilities,
        extra=body.extra,
    )


@router.get("/{provider_id}", response_model=ProviderConfig)
async def get_provider(provider_id: str) -> ProviderConfig:
    return await _service.get_by_id(provider_id)


@router.put("/{provider_id}", response_model=ProviderConfig)
async def update_provider(provider_id: str, body: UpdateProviderRequest) -> ProviderConfig:
    return await _service.update(
        provider_id=provider_id,
        display_name=body.display_name,
        base_url=body.base_url,
        model_id=body.model_id,
        capabilities=body.capabilities,
        extra=body.extra,
    )


@router.delete("/{provider_id}", status_code=204)
async def delete_provider(provider_id: str) -> None:
    await _service.delete(provider_id)


@router.post("/{provider_id}/test", response_model=TestConnectionResponse)
async def test_provider_connection(provider_id: str) -> TestConnectionResponse:
    result = await _service.test_connection(provider_id)
    return TestConnectionResponse(ok=result.ok, message=result.message, warning=result.warning)


@router.get("/{provider_id}/models", response_model=ListModelsResponse)
async def list_provider_models(provider_id: str) -> ListModelsResponse:
    models = await _service.list_models(provider_id)
    return ListModelsResponse(models=models)
