"""LLM agent that generates a semantic cache descriptor for one analysis section."""

from __future__ import annotations

import json
from typing import Any

from domain.analysis.agents.base import BaseTypedAgent
from domain.model_connector.service import ProviderConfigService
from shared.logger import logger

from .types import QASectionCache


_CACHE_SYSTEM_PROMPT = """\
You are a documentation analyst. Given the content of one analysis section from a \
codebase analysis report, your job is to generate a compact semantic descriptor so \
that at query time we can decide whether this section is relevant to a user's question.

Output JSON with exactly these fields:
{
  "topic_summary": "1-2 sentence description of what this section covers and what questions it answers.",
  "answerable_questions": [
    "Example question 1 that this section can directly answer",
    "Example question 2",
    "Example question 3"
  ],
  "key_terms": ["term1", "term2", ...]
}

Rules:
- topic_summary: 1-2 sentences max. Written to be matched against user questions.
- answerable_questions: 3-5 concrete questions a developer might ask that this section answers \
directly. Use natural developer language (English or mixed). Do NOT include questions this section \
cannot answer.
- key_terms: 5-15 lowercase terms that would appear in questions about this section's content. \
Include: framework names, language names, file paths (without slashes), domain nouns, function names, \
architectural pattern names. Exclude generic words (the, is, a, to).

RESPONSE FORMAT — MANDATORY:
- Your ENTIRE response must be ONE valid JSON object.
- Start with { and end with }.
- No markdown fences, no prose before or after.
- Double-quoted keys and string values only.
- No trailing commas.
"""

_CACHE_SCHEMA_STR = (
    '{"topic_summary": "string", "answerable_questions": ["string"], "key_terms": ["string"]}'
)

# Human-readable section labels injected into the user prompt for context
_SECTION_LABELS: dict[str, str] = {
    "A": "Section A — Project Identity (domain, purpose, runtime type, tech stack)",
    "B": "Section B — Architecture (layers, frameworks, entrypoints, services, integrations)",
    "C": "Section C — Folder Structure (directory layout, roles, descriptions)",
    "D": "Section D — Conventions (naming, error handling, async style, DI, test style)",
    "E": "Section E — Rules & Violations (coding rules, enforcement, violations found)",
    "F": "Section F — Feature Map (product features, entrypoints, key files)",
    "G": "Section G — Important Files (entrypoint, backbone, critical config, centrality)",
    "H": "Section H — Onboarding Path (learning order, goals, outcomes)",
    "I": "Section I — Glossary (domain terms and their definitions)",
    "J": "Section J — Risk Findings (security, quality, technical debt)",
    "K": "Section K — Audit Trail (decisions, changes, LLM reasoning log)",
    "L": "Section L — Executive Synthesis (summary, architecture narrative, conventions digest)",
}


class SectionCacheAgent(BaseTypedAgent):
    """Generates a QASectionCache descriptor for a single analysis section via one LLM call."""

    def __init__(self, provider_service: ProviderConfigService) -> None:
        super().__init__(provider_service)

    async def generate_for_section(
        self,
        section_id: str,
        section_data: dict[str, Any],
        provider_id: str,
        model_id: str,
    ) -> dict[str, Any] | None:
        """Run one LLM call to produce cache metadata for a single section.

        Returns the raw dict {topic_summary, answerable_questions, key_terms}
        or None on failure. Caller is responsible for wrapping into QASectionCache.
        """
        label = _SECTION_LABELS.get(section_id, f"Section {section_id}")
        # Serialize section content; truncate to ~3000 chars to keep this a cheap call
        try:
            section_text = json.dumps(section_data, ensure_ascii=False, indent=2)
        except Exception:
            section_text = str(section_data)
        section_text = section_text[:3000]

        user_prompt = (
            f"Analyze the following codebase analysis section and produce a semantic cache descriptor.\n\n"
            f"SECTION: {label}\n\n"
            f"SECTION CONTENT:\n{section_text}"
        )

        try:
            result = await self._chat_json_typed(
                provider_id,
                model_id,
                _CACHE_SYSTEM_PROMPT,
                user_prompt,
                _CACHE_SCHEMA_STR,
                max_completion_tokens=600,
            )
        except Exception as exc:
            logger.warning("[SectionCacheAgent] LLM call failed for section=%s: %s", section_id, exc)
            return None

        # Validate minimal structure
        if not isinstance(result.get("topic_summary"), str):
            logger.warning("[SectionCacheAgent] missing topic_summary for section=%s", section_id)
            return None

        # Normalize list fields
        result["answerable_questions"] = [
            str(q) for q in (result.get("answerable_questions") or [])
        ][:5]
        result["key_terms"] = [
            str(t).lower() for t in (result.get("key_terms") or [])
        ][:15]
        result["topic_summary"] = str(result["topic_summary"])[:500]

        return result
