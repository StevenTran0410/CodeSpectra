"""Impact analysis API endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from domain.impact.service import ImpactService
from domain.impact.types import BlastRadiusRequest, BlastRadiusResponse, PlanRequest, PlanResponse
from shared.http_utils import handle_value_error

router = APIRouter(tags=["impact"])
_service = ImpactService()


@router.post("/blast-radius", response_model=BlastRadiusResponse)
@handle_value_error
async def blast_radius(body: BlastRadiusRequest) -> BlastRadiusResponse:
    return await _service.blast_radius(body)


@router.post("/plan", response_model=PlanResponse)
@handle_value_error
async def implementation_plan(body: PlanRequest) -> PlanResponse:
    return await _service.plan(body)
