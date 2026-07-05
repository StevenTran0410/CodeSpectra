"""Pass 2: Classify component roles via LLM."""
from __future__ import annotations

import json

from agent_eval_harness.llm.client import LLMClient, LLMMessage
from agent_eval_harness.mapping.builder.prompts import ROLE_CLASSIFICATION_SYSTEM
from agent_eval_harness.mapping.builder.types import CandidateComponent, RoleClassification

ROLE_CONFIDENCE_THRESHOLD: float = 0.7
VALID_ROLES = frozenset({
    "input_guard.rule",
    "input_guard.llm",
    "orchestrator",
    "retrieval_agent",
    "tool",
    "validator",
    "writer",
    "unknown",
})


async def classify_roles(
    candidates: list[CandidateComponent],
    llm_client: LLMClient,
    threshold: float = ROLE_CONFIDENCE_THRESHOLD,
) -> list[RoleClassification]:
    """Classify each candidate into a role."""
    results = []

    for candidate in candidates:
        # Build user prompt: class_name (with sub-role), docstring, source
        user_prompt = candidate.class_name
        if candidate.tag_suffix:
            user_prompt += f" (sub-role: {candidate.tag_suffix})"
        user_prompt += (
            f"\n\nDocstring: {candidate.docstring}\n\nSource:\n{candidate.source_snippet}"
        )

        response = await llm_client.complete(
            [
                LLMMessage(role="system", content=ROLE_CLASSIFICATION_SYSTEM),
                LLMMessage(role="user", content=user_prompt),
            ],
            json_mode=True,
        )

        try:
            parsed = json.loads(response.content)
            role = parsed.get("role", "unknown")
            confidence = float(parsed.get("confidence", 0.0))
            reasoning = parsed.get("reasoning", "")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            role, confidence, reasoning = "unknown", 0.0, "unparseable LLM response"

        # Apply threshold
        if role not in VALID_ROLES or confidence < threshold:
            role = "unknown"

        results.append(RoleClassification(candidate.candidate_id, role, confidence, reasoning))

    return results
