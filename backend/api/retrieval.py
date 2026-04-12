"""Retrieval endpoints (RPA-034)."""
from fastapi import APIRouter, HTTPException

from domain.retrieval.service import RetrievalService
from domain.retrieval.types import (
    BuildRetrievalIndexRequest,
    BuildRetrievalIndexResponse,
    RetrievalCompareResponse,
    RetrievalBundle,
    RetrieveRequest,
    TwoStageBundle,
    TwoStageRequest,
)

router = APIRouter(tags=["retrieval"])
_service = RetrievalService()


@router.post("/build-index", response_model=BuildRetrievalIndexResponse)
async def build_retrieval_index(body: BuildRetrievalIndexRequest) -> BuildRetrievalIndexResponse:
    try:
        return await _service.build_index(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/retrieve", response_model=RetrievalBundle)
async def retrieve_context(body: RetrieveRequest) -> RetrievalBundle:
    try:
        return await _service.retrieve(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/compare", response_model=RetrievalCompareResponse)
async def compare_retrieval_modes(body: RetrieveRequest) -> RetrievalCompareResponse:
    try:
        return await _service.compare(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/retrieve-two-stage", response_model=TwoStageBundle)
async def retrieve_two_stage(body: TwoStageRequest) -> TwoStageBundle:
    try:
        return await _service.retrieve_two_stage(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
