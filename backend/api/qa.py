"""Q&A API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from domain.model_connector.service import ProviderConfigService
from domain.qa.service import QAService
from domain.qa.types import QARequest, QAResponse
from domain.retrieval.service import RetrievalService

router = APIRouter(tags=["qa"])
_service = QAService(ProviderConfigService(), RetrievalService())


@router.post("/ask", response_model=QAResponse, status_code=200)
async def qa_ask(body: QARequest) -> QAResponse:
    try:
        return await _service.ask(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
