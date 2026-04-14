"""Q&A API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from domain.model_connector.service import ProviderConfigService
from domain.qa.service import QAService
from domain.qa.classifier_service import ClassifierService
from domain.qa.types import QARequest, QAResponse, ClassifyIntentRequest, ClassifyIntentResponse
from domain.qa.deep_research import DeepResearchRequest, DeepResearchResult
from domain.retrieval.service import RetrievalService

router = APIRouter(tags=["qa"])
_service = QAService(ProviderConfigService(), RetrievalService())
_classifier_svc = ClassifierService()


# ── Core QA ──────────────────────────────────────────────────────────────────

@router.post("/ask", response_model=QAResponse, status_code=200)
async def qa_ask(body: QARequest) -> QAResponse:
    try:
        return await _service.ask(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/classify-intent", response_model=ClassifyIntentResponse, status_code=200)
def qa_classify_intent(body: ClassifyIntentRequest) -> ClassifyIntentResponse:
    try:
        return _service.classify_intent(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/deep-research", response_model=DeepResearchResult, status_code=200)
async def qa_deep_research(body: DeepResearchRequest) -> DeepResearchResult:
    try:
        return await _service.deep_research(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Classifier management ─────────────────────────────────────────────────────

class AddExampleRequest(BaseModel):
    text: str
    is_deep_research: bool


class ClassifierStatusResponse(BaseModel):
    trained: bool
    backend: str        # "sklearn" | "pure_python" | "not_trained"
    builtin_examples: int
    user_examples: int


@router.get("/classifier/status", response_model=ClassifierStatusResponse, status_code=200)
async def classifier_status() -> ClassifierStatusResponse:
    status = await _classifier_svc.get_status()
    return ClassifierStatusResponse(**status)


@router.get("/classifier/examples", status_code=200)
async def classifier_list_examples() -> list[dict]:
    return await _classifier_svc.list_examples()


@router.post("/classifier/examples", status_code=201)
async def classifier_add_example(body: AddExampleRequest) -> dict:
    try:
        return await _classifier_svc.add_example(body.text, body.is_deep_research)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/classifier/examples/{example_id}", status_code=204)
async def classifier_delete_example(example_id: str) -> None:
    await _classifier_svc.delete_example(example_id)


@router.post("/classifier/retrain", response_model=ClassifierStatusResponse, status_code=200)
async def classifier_retrain() -> ClassifierStatusResponse:
    status = await _classifier_svc.retrain()
    return ClassifierStatusResponse(**status)
