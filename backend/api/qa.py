"""Q&A API endpoints."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from domain.model_connector.service import ProviderConfigService
from domain.qa.service import QAService
from domain.qa.classifier_service import ClassifierService
from domain.qa.types import QARequest, QAResponse, ClassifyIntentRequest, ClassifyIntentResponse
from domain.qa.deep_research import DeepResearchRequest, DeepResearchResult
from domain.retrieval.service import RetrievalService
from shared.http_utils import handle_value_error

router = APIRouter(tags=["qa"])
_service = QAService(ProviderConfigService(), RetrievalService())
_classifier_svc = ClassifierService()

_ProgressCb = Callable[[dict], Awaitable[None]]


async def _sse_generate(
    run: Callable[[_ProgressCb], Awaitable[Any]],
) -> AsyncGenerator[str, None]:
    """Shared SSE generator for any async operation that accepts a progress_cb.

    Runs `run(on_progress)` as a background task, forwarding every emitted event
    as an SSE frame. Terminates on 'done' or 'error' and cancels the task if the
    client disconnects.
    """
    queue: asyncio.Queue[dict] = asyncio.Queue()

    async def on_progress(event: dict) -> None:
        await queue.put(event)

    async def run_and_signal() -> None:
        try:
            result = await run(on_progress)
            await queue.put({"type": "done", "result": result.model_dump()})
        except Exception as exc:
            await queue.put({"type": "error", "message": str(exc)})

    task = asyncio.create_task(run_and_signal())
    try:
        while True:
            event = await queue.get()
            yield f"data: {json.dumps(event)}\n\n"
            if event["type"] in ("done", "error"):
                break
    finally:
        if not task.done():
            task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


# ── Core QA ──────────────────────────────────────────────────────────────────

@router.post("/ask", response_model=QAResponse, status_code=200)
@handle_value_error
async def qa_ask(body: QARequest) -> QAResponse:
    return await _service.ask(body)


@router.post("/ask/stream")
async def qa_ask_stream(body: QARequest) -> StreamingResponse:
    return StreamingResponse(
        _sse_generate(lambda cb: _service.ask(body, progress_cb=cb)),
        media_type="text/event-stream",
    )


@router.post("/classify-intent", response_model=ClassifyIntentResponse, status_code=200)
@handle_value_error
def qa_classify_intent(body: ClassifyIntentRequest) -> ClassifyIntentResponse:
    return _service.classify_intent(body)


@router.post("/deep-research", response_model=DeepResearchResult, status_code=200)
@handle_value_error
async def qa_deep_research(body: DeepResearchRequest) -> DeepResearchResult:
    return await _service.deep_research(body)


@router.post("/deep-research/stream")
async def qa_deep_research_stream(body: DeepResearchRequest) -> StreamingResponse:
    return StreamingResponse(
        _sse_generate(lambda cb: _service.deep_research(body, progress_cb=cb)),
        media_type="text/event-stream",
    )


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
@handle_value_error
async def classifier_add_example(body: AddExampleRequest) -> dict:
    return await _classifier_svc.add_example(body.text, body.is_deep_research)


@router.delete("/classifier/examples/{example_id}", status_code=204)
async def classifier_delete_example(example_id: str) -> None:
    await _classifier_svc.delete_example(example_id)


@router.post("/classifier/retrain", response_model=ClassifierStatusResponse, status_code=200)
async def classifier_retrain() -> ClassifierStatusResponse:
    status = await _classifier_svc.retrain()
    return ClassifierStatusResponse(**status)
