"""Retrieval endpoints (RPA-034)."""

from fastapi import APIRouter

from domain.retrieval.service import RetrievalService
from domain.retrieval.types import (
    BuildRetrievalIndexRequest,
    BuildRetrievalIndexResponse,
    RetrievalBundle,
    RetrievalCompareResponse,
    RetrievalSummary,
    RetrieveRequest,
    RrfFusionBundle,
    RrfFusionRequest,
    TwoStageBundle,
    TwoStageRequest,
)
from shared.http_utils import handle_value_error

router = APIRouter(tags=["retrieval"])
_service = RetrievalService()


@router.get("/summary/{snapshot_id}", response_model=RetrievalSummary)
async def get_retrieval_summary(snapshot_id: str) -> RetrievalSummary:
    return await _service.summary(snapshot_id)


@router.post("/build-index", response_model=BuildRetrievalIndexResponse)
@handle_value_error
async def build_retrieval_index(body: BuildRetrievalIndexRequest) -> BuildRetrievalIndexResponse:
    return await _service.build_index(body)


@router.post("/retrieve", response_model=RetrievalBundle)
@handle_value_error
async def retrieve_context(body: RetrieveRequest) -> RetrievalBundle:
    return await _service.retrieve(body)


@router.post("/compare", response_model=RetrievalCompareResponse)
@handle_value_error
async def compare_retrieval_modes(body: RetrieveRequest) -> RetrievalCompareResponse:
    return await _service.compare(body)


@router.post("/retrieve-two-stage", response_model=TwoStageBundle)
@handle_value_error
async def retrieve_two_stage(body: TwoStageRequest) -> TwoStageBundle:
    return await _service.retrieve_two_stage(body)


@router.post("/retrieve-rrf-fusion", response_model=RrfFusionBundle)
@handle_value_error
async def retrieve_rrf_fusion(body: RrfFusionRequest) -> RrfFusionBundle:
    return await _service.retrieve_rrf_fusion(body)
