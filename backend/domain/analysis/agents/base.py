"""Base agent with JSON chat path for typed section schemas (RPA-053)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from domain.model_connector.service import ProviderConfigService
from domain.retrieval.types import RetrievalMode, RetrievalSection, RetrieveRequest
from shared.logger import logger

from ..agent_pipeline import BaseLLMAgent

if TYPE_CHECKING:
    from domain.retrieval.service import RetrievalService
    from ..profiles import AnalysisProfile
    from ..prompts import render_bundle as _render_bundle_type

# Strip control characters that make JSON request bodies invalid (e.g. null bytes
# from binary file content leaking into retrieved chunks).
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Keywords in blind_spots that signal the agent needs more context
_TRUNCATION_KEYWORDS = (
    "truncated",
    "not provided",
    "not present",
    "not included",
    "could not be verified",
    "were not provided",
    "not available",
    "incomplete",
    "limited to",
    "missing",
)


def _sanitize(text: str) -> str:
    """Remove non-printable control characters, keep \\t \\n \\r."""
    return _CTRL_RE.sub("", text)


def _blind_spots_need_retrieval(blind_spots: list[str]) -> bool:
    """Return True if any blind_spot entry signals missing or truncated evidence."""
    for bs in blind_spots:
        lower = bs.lower()
        if any(kw in lower for kw in _TRUNCATION_KEYWORDS):
            return True
    return False


def _extract_retrieval_queries(blind_spots: list[str]) -> list[str]:
    """Extract file paths and keyword phrases from blind_spot strings for targeted retrieval."""
    queries: list[str] = []
    # file paths: anything matching word chars + slashes + extension
    _path_re = re.compile(r'[\w./-]+\.\w{1,10}')
    seen: set[str] = set()
    for bs in blind_spots:
        paths = _path_re.findall(bs)
        for p in paths:
            if p not in seen:
                seen.add(p)
                queries.append(p)
        # Use the full blind_spot text as a fallback query (truncated to 150 chars)
        trimmed = bs[:150].strip()
        if trimmed and trimmed not in seen:
            seen.add(trimmed)
            queries.append(trimmed)
    return queries


class BaseTypedAgent(BaseLLMAgent):
    def __init__(self, provider_service: ProviderConfigService) -> None:
        super().__init__(provider_service)

    async def _call(
        self,
        provider_id,
        model_id,
        system_prompt,
        user_prompt,
        max_completion_tokens,
        temperature=0.2,
        json_mode=True,
    ):
        """Override to sanitize prompts before sending to LLM."""
        return await super()._call(
            provider_id,
            model_id,
            _sanitize(system_prompt),
            _sanitize(user_prompt),
            max_completion_tokens,
            temperature,
            json_mode,
        )

    async def _chat_json_typed(
        self,
        provider_id: str,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
        schema_hint: str = "",
        max_completion_tokens: int = 1200,
    ) -> dict[str, Any]:
        return await super()._chat_json(
            provider_id,
            model_id,
            system_prompt,
            user_prompt,
            max_completion_tokens=max_completion_tokens,
            schema_hint=schema_hint,
        )

    async def _chat_json_with_augment(
        self,
        *,
        retrieval: "RetrievalService",
        snapshot_id: str,
        provider_id: str,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
        schema_hint: str,
        max_completion_tokens: int,
        retrieval_section: RetrievalSection,
        profile: "AnalysisProfile",
        agent_tag: str = "Agent",
    ) -> dict[str, Any]:
        """Run _chat_json_typed, then retry with additional retrieval if blind_spots
        indicate missing or truncated evidence.

        Normal profile: up to profile.retrieval_augment_rounds (2) extra rounds.
        Large profile:  up to profile.retrieval_augment_rounds (5) extra rounds.
        """
        from ..prompts import render_bundle  # lazy import to avoid circulars

        data = await self._chat_json_typed(
            provider_id, model_id, system_prompt, user_prompt,
            schema_hint, max_completion_tokens,
        )

        max_rounds = profile.retrieval_augment_rounds
        augmented_prompt = user_prompt

        for round_idx in range(max_rounds):
            blind_spots = data.get("blind_spots") or []
            if not isinstance(blind_spots, list):
                break
            if not _blind_spots_need_retrieval(blind_spots):
                break

            queries = _extract_retrieval_queries(blind_spots)
            if not queries:
                break

            logger.info(
                "[%s] blind_spots signal missing context — augment round %d/%d, queries=%s",
                agent_tag, round_idx + 1, max_rounds, queries[:3],
            )

            # Fetch additional chunks for up to 3 queries per round
            new_evidences: list[Any] = []
            for q in queries[:3]:
                try:
                    bundle = await retrieval.retrieve(
                        RetrieveRequest(
                            snapshot_id=snapshot_id,
                            query=q,
                            section=retrieval_section,
                            mode=RetrievalMode.HYBRID,
                            max_results=min(10, profile.retrieval_max_results),
                        )
                    )
                    new_evidences.extend(bundle.evidences)
                except Exception as exc:
                    logger.warning("[%s] retrieval augment query '%s' failed: %s", agent_tag, q, exc)

            if not new_evidences:
                logger.info("[%s] augment round %d produced no new evidence, stopping.", agent_tag, round_idx + 1)
                break

            # Deduplicate by excerpt content (avoid re-sending same chunk)
            seen_excerpts: set[str] = set()
            unique_evidences = []
            for ev in new_evidences:
                key = (getattr(ev, "rel_path", ""), (getattr(ev, "excerpt", "") or "")[:100])
                if key not in seen_excerpts:
                    seen_excerpts.add(key)
                    unique_evidences.append(ev)

            from domain.retrieval.types import RetrievalBundle  # lazy
            augment_bundle = RetrievalBundle(
                snapshot_id=snapshot_id,
                mode=RetrievalMode.HYBRID,
                section=retrieval_section,
                query="augment",
                budget_tokens=0,
                used_tokens=0,
                evidences=unique_evidences[:15],
            )
            augment_block = render_bundle(augment_bundle, limit=15, excerpt_chars=1500)

            augmented_prompt = (
                augmented_prompt
                + f"\n\n--- Supplemental evidence (retrieval augment round {round_idx + 1}) ---\n"
                + augment_block
            )

            logger.info(
                "[%s] re-running LLM with %d new chunks (round %d/%d)",
                agent_tag, len(unique_evidences), round_idx + 1, max_rounds,
            )
            data = await self._chat_json_typed(
                provider_id, model_id, system_prompt, augmented_prompt,
                schema_hint, max_completion_tokens,
            )

        return data
