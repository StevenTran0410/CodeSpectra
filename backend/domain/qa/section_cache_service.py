"""Service that builds and loads QA section caches (CS-225)."""

from __future__ import annotations

import json
from typing import Any

from domain.model_connector.service import ProviderConfigService
from infrastructure.db.database import get_db
from shared.logger import logger
from shared.utils import new_id, utc_now_iso

from .section_cache_agent import SectionCacheAgent
from .types import QASectionCache

# Sections to generate cache for (all standard analysis sections)
_ALL_SECTIONS = list("ABCDEFGHIJKL")


class QASectionCacheService:
    """Builds section cache after analysis and loads it at QA query time."""

    def __init__(self, provider_service: ProviderConfigService) -> None:
        self._agent = SectionCacheAgent(provider_service)

    async def build_caches(
        self,
        snapshot_id: str,
        report_id: str,
        report_json: dict[str, Any],
        provider_id: str,
        model_id: str,
    ) -> None:
        """Generate and persist cache records for all sections in a report.

        Runs 12 lightweight LLM calls (one per section). Per-section failures are
        logged and skipped — they do not abort the overall build.

        Called fire-and-forget after analysis save; must never raise.
        """
        sections: dict[str, Any] = report_json.get("sections") or {}
        if not sections:
            logger.warning(
                "[QASectionCacheService] no sections found in report=%s, skipping cache build",
                report_id,
            )
            return

        db = get_db()
        built = 0

        for section_id in _ALL_SECTIONS:
            section_data = sections.get(section_id)
            if not section_data or not isinstance(section_data, dict):
                continue
            # Skip error/empty sections
            if "error" in section_data and len(section_data) <= 2:
                continue

            try:
                raw = await self._agent.generate_for_section(
                    section_id, section_data, provider_id, model_id
                )
            except Exception as exc:
                logger.warning(
                    "[QASectionCacheService] agent failed for section=%s report=%s: %s",
                    section_id, report_id, exc,
                )
                continue

            if raw is None:
                continue

            cache_id = new_id()
            now = utc_now_iso()
            try:
                await db.execute(
                    """
                    INSERT OR REPLACE INTO qa_section_caches
                        (id, snapshot_id, report_id, section_id,
                         topic_summary, answerable_questions, key_terms, generated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cache_id,
                        snapshot_id,
                        report_id,
                        section_id,
                        raw["topic_summary"],
                        json.dumps(raw["answerable_questions"]),
                        json.dumps(raw["key_terms"]),
                        now,
                    ),
                )
                await db.commit()
                built += 1
                logger.debug(
                    "[QASectionCacheService] cached section=%s report=%s", section_id, report_id
                )
            except Exception as exc:
                logger.warning(
                    "[QASectionCacheService] DB write failed for section=%s report=%s: %s",
                    section_id, report_id, exc,
                )

        logger.info(
            "[QASectionCacheService] build complete: %d/%d sections cached for report=%s",
            built, len(_ALL_SECTIONS), report_id,
        )

    async def load_caches(self, snapshot_id: str) -> list[QASectionCache]:
        """Load the latest set of section caches for a snapshot.

        Strategy: find the most recently generated report_id for this snapshot,
        then return all cache records matching that report_id.
        Returns empty list if no caches exist (caller falls back to code-retrieval-only).
        """
        db = get_db()

        # Find the latest report_id among caches for this snapshot
        async with db.execute(
            """
            SELECT report_id
            FROM qa_section_caches
            WHERE snapshot_id=?
            ORDER BY generated_at DESC
            LIMIT 1
            """,
            (snapshot_id,),
        ) as cur:
            row = await cur.fetchone()

        if row is None:
            return []

        latest_report_id = str(row["report_id"])

        async with db.execute(
            """
            SELECT id, snapshot_id, report_id, section_id,
                   topic_summary, answerable_questions, key_terms, generated_at
            FROM qa_section_caches
            WHERE snapshot_id=? AND report_id=?
            """,
            (snapshot_id, latest_report_id),
        ) as cur:
            rows = await cur.fetchall()

        caches: list[QASectionCache] = []
        for r in rows:
            try:
                caches.append(QASectionCache(
                    id=str(r["id"]),
                    snapshot_id=str(r["snapshot_id"]),
                    report_id=str(r["report_id"]),
                    section_id=str(r["section_id"]),
                    topic_summary=str(r["topic_summary"]),
                    answerable_questions=json.loads(r["answerable_questions"] or "[]"),
                    key_terms=json.loads(r["key_terms"] or "[]"),
                    generated_at=str(r["generated_at"]),
                ))
            except Exception as exc:
                logger.warning("[QASectionCacheService] malformed cache row: %s", exc)

        return caches
