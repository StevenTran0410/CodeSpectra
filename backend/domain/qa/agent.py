"""QA Agent — answers codebase questions with evidence-backed citations."""

from __future__ import annotations

import json
import time
from typing import Any

from domain.model_connector.service import ProviderConfigService
from domain.retrieval.service import RetrievalService
from domain.retrieval.types import RetrievalMode, RetrievalSection
from shared.logger import logger

from domain.analysis.agents._graph_plan import plan_queries, retrieve_multi
from domain.analysis.agents.base import BaseTypedAgent
from domain.analysis.prompts import render_bundle

from .types import Citation, QAResponse


_QA_SYSTEM_PROMPT = """\
You are a senior engineer answering questions about a codebase.

Rules:
- Answer ONLY from the provided code evidence. Do not guess or invent information.
- Cite every claim with [file:line] references from the evidence provided.
- If you cannot determine something from the evidence, list it under "unknowns".
- Suggest 2-3 files the user should read next to deepen understanding.

ANSWER FORMATTING — MANDATORY:
- Write the "answer" field in rich Markdown so it renders beautifully in a chat UI.
- Use ### headings to organize sections.
- Use **bold** for key terms, file names, and function names.
- Use bullet lists (- item) or numbered lists for steps and enumerations.
- Use `backticks` for inline code, symbols, and file paths.
- Use ``` fenced code blocks for multi-line code snippets.
- Use --- horizontal rules to separate major sections.
- Do NOT write a flat wall of text — structure every answer.

Output JSON: {"answer": "...", "citations": [...], "confidence": "high|medium|low", "unknowns": [...], "suggested_files": [...]}

RESPONSE FORMAT — MANDATORY:
- Your ENTIRE response must be ONE valid JSON object.
- Start with { and end with }.
- No markdown fences wrapping the JSON, no prose before or after.
- Double-quoted keys and string values only.
- No trailing commas.
"""

_QA_SCHEMA_STR = (
    '{"answer": "string", "citations": [{"file": "string", "line_start": 0, '
    '"line_end": 0, "snippet": "string"}], "confidence": "high|medium|low", '
    '"unknowns": ["string"], "suggested_files": ["string"]}'
)

_QA_RETRIEVAL_BUDGET = 12_000


class QAAgent(BaseTypedAgent):
    """Answers free-form codebase questions using retrieval + LLM."""

    def __init__(
        self,
        provider_service: ProviderConfigService,
        retrieval_service: RetrievalService,
    ) -> None:
        super().__init__(provider_service)
        self._retrieval = retrieval_service

    async def run(
        self,
        provider_id: str,
        model_id: str,
        snapshot_id: str,
        question: str,
        include_debug: bool = False,
    ) -> dict[str, Any]:
        t0 = time.monotonic()

        # 1. Decompose question into sub-queries
        sub_queries = await plan_queries(
            goal=question,
            provider_service=self._providers,
            provider_id=provider_id,
            model_id=model_id,
            fallback=[question],
        )
        logger.info("[QAAgent] question=%r sub_queries=%s", question[:80], sub_queries)

        # 2. Retrieve evidence via two-stage pipeline
        bundle = await retrieve_multi(
            self._retrieval,
            snapshot_id,
            sub_queries,
            RetrievalSection.QA,
            RetrievalMode.HYBRID,
            20,
        )

        # 3. Build prompt with evidence
        evidence_block = render_bundle(bundle, limit=25, excerpt_chars=1500)
        user_prompt = f"QUESTION: {question}\n\nEVIDENCE:\n{evidence_block}"

        # 4. LLM call (single-shot, no augmentation loop)
        result = await self._chat_json_typed(
            provider_id,
            model_id,
            _QA_SYSTEM_PROMPT,
            user_prompt,
            _QA_SCHEMA_STR,
            max_completion_tokens=2000,
        )

        # 5. Normalize output
        result["answer"] = str(result.get("answer") or "")
        result["citations"] = result.get("citations") or []
        result["confidence"] = str(result.get("confidence", "medium")).lower()
        if result["confidence"] not in ("high", "medium", "low"):
            result["confidence"] = "medium"
        result["unknowns"] = result.get("unknowns") or []
        result["suggested_files"] = result.get("suggested_files") or []

        if include_debug:
            result["retrieval_debug"] = {
                "sub_queries": sub_queries,
                "chunks_used": len(bundle.evidences) if hasattr(bundle, "evidences") else 0,
            }
        else:
            result["retrieval_debug"] = None

        ms = int((time.monotonic() - t0) * 1000)
        logger.info("[QAAgent] answered in %dms confidence=%s", ms, result["confidence"])
        return result
