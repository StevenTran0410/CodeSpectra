"""Q&A request/response types."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Citation(BaseModel):
    file: str
    line_start: int | None = None
    line_end: int | None = None
    snippet: str


class SupportingSpan(BaseModel):
    path: str
    line_range: tuple[int, int]
    exact_quote: str


class QARequest(BaseModel):
    snapshot_id: str
    report_id: str | None = None
    question: str
    provider_id: str
    model_id: str
    include_debug: bool = False


class QAResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: str  # high | medium | low
    unknowns: list[str]
    suggested_files: list[str]
    deep_research_recommended: bool = False
    retrieval_debug: dict | None = None
    claim_polarity: Literal['positive', 'negative', 'neutral'] = 'neutral'
    is_supported_by_chunks: bool = True
    needs_more_retrieval: bool = False
    retrieval_quality_label: str = 'strong'


class ClassifyIntentRequest(BaseModel):
    question: str


class ClassifyIntentResponse(BaseModel):
    deep_research: bool


class QASectionCache(BaseModel):
    id: str
    snapshot_id: str
    report_id: str
    section_id: str          # "A" .. "L"
    topic_summary: str
    answerable_questions: list[str]  # 3-5 example questions
    key_terms: list[str]             # 5-15 extracted terms
    generated_at: str
